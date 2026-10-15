"""LightGBM quantitative machine learning model trainer with sample weights support and binary meta-labeling."""

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from config.settings import settings
from src.utils.logger import logger


class QuantModelTrainer:
    """Trainer wrapper for LightGBM Classifier tailored for binary meta-labeling."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        num_leaves: int | None = None,
        min_child_samples: int | None = None,
        learning_rate: float | None = None,
        subsample: float | None = None,
        colsample_bytree: float | None = None,
        random_state: int = 42,
    ) -> None:
        params = settings.MODEL_PARAMS.copy()
        if max_depth is not None:
            params["max_depth"] = max_depth
        if num_leaves is not None:
            params["num_leaves"] = num_leaves
        if min_child_samples is not None:
            params["min_child_samples"] = min_child_samples
        if learning_rate is not None:
            params["learning_rate"] = learning_rate
        if subsample is not None:
            params["subsample"] = subsample
        if colsample_bytree is not None:
            params["colsample_bytree"] = colsample_bytree

        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            objective="binary",
            metric="binary_logloss",
            verbose=-1,
            **params,
        )
        self.feature_cols: list[str] = []

    def train(
        self,
        df: pl.DataFrame,
        feature_cols: list[str],
        target_col: str = "target_label",
        weight_col: str = "sample_weight",
    ) -> dict:
        """Train LightGBM binary classifier with sample weights and evaluate performance metrics."""
        logger.info(f"Training LightGBM binary model on {len(df)} samples using {len(feature_cols)} features...")
        self.feature_cols = feature_cols

        pdf = df.select(feature_cols + [target_col, weight_col]).to_pandas()
        pdf = pdf.dropna(subset=feature_cols + [target_col, weight_col])

        X = pdf[feature_cols]
        y = pdf[target_col].astype(int)
        weights = pdf[weight_col].astype(float)

        if len(X) == 0:
            raise ValueError("Training dataset is empty after removing nulls.")

        self.model.fit(X, y, sample_weight=weights)

        preds_proba = self.model.predict_proba(X)
        preds_class = self.model.predict(X)

        acc = accuracy_score(y, preds_class)
        loss = log_loss(y, preds_proba)
        
        try:
            auc = roc_auc_score(y, preds_proba[:, 1])
        except Exception:
            auc = 0.0

        metrics = {
            "accuracy": float(acc),
            "log_loss": float(loss),
            "roc_auc": float(auc),
            "train_samples": int(len(X)),
        }

        logger.success(f"Model training completed successfully. Metrics: {metrics}")
        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance ranking based on split and gain."""
        if not self.feature_cols:
            return pd.DataFrame()

        importances_gain = self.model.booster_.feature_importance(importance_type="gain")
        importances_split = self.model.booster_.feature_importance(importance_type="split")

        fi_df = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "importance_gain": importances_gain,
                "importance_split": importances_split,
            }
        )
        fi_df = fi_df.sort_values(by="importance_gain", ascending=False).reset_index(drop=True)
        return fi_df

    def predict_proba(self, df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
        """Predict binary class probabilities for new data."""
        pdf = df.select(feature_cols).to_pandas()
        return self.model.predict_proba(pdf)

    @staticmethod
    def oos_roc_auc(df: pl.DataFrame, oos_probs: np.ndarray, target_col: str = "target_label") -> float:
        """ROC AUC over the rows that were actually scored out-of-sample (NaN when undefined)."""
        scored = ~np.isnan(oos_probs[:, 1])
        y_true = df[target_col].to_numpy()[scored]
        if scored.sum() == 0 or len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, oos_probs[scored, 1]))

    def cross_val_predict_walk_forward(
        self,
        df: pl.DataFrame,
        feature_cols: list[str],
        target_col: str = "target_label",
        weight_col: str = "sample_weight",
        n_splits: int = 5,
        pct_embargo: float = 0.01,
    ) -> np.ndarray:
        """Perform Purged Walk-Forward Cross-Validation and return binary OOS probabilities."""
        logger.info(f"Performing Walk-Forward Cross-Validation predictions ({n_splits} splits, embargo={pct_embargo})...")
        from src.validation.cross_validation import PurgedGroupTimeSeriesSplit

        cv = PurgedGroupTimeSeriesSplit(n_splits=n_splits, pct_embargo=pct_embargo)
        pdf = df.to_pandas()

        # Rows never scored out-of-sample (the first fold has no training history) stay NaN
        oos_probs = np.full((len(df), 2), np.nan)

        for fold, (train_idx, test_idx) in enumerate(cv.split(pdf)):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            fold_trainer = QuantModelTrainer()
            fold_trainer.model = clone(self.model)
            fold_trainer.train(pl.from_pandas(pdf.iloc[train_idx]), feature_cols=feature_cols, target_col=target_col, weight_col=weight_col)
            oos_probs[test_idx] = fold_trainer.predict_proba(pl.from_pandas(pdf.iloc[test_idx]), feature_cols=feature_cols)

        logger.success("Walk-Forward OOS binary probability predictions completed.")
        return oos_probs
