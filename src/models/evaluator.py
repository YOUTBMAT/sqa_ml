"""Model evaluator module for out-of-sample cross-validation using Purged Group Time Series Split."""

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

from src.models.trainer import QuantModelTrainer
from src.validation.cross_validation import PurgedGroupTimeSeriesSplit
from src.utils.logger import logger


class ModelEvaluator:
    """Evaluator for out-of-sample cross-validation performance."""

    @staticmethod
    def evaluate_cv(
        df: pl.DataFrame,
        feature_cols: list[str],
        n_splits: int = 5,
        pct_embargo: float = 0.01,
        target_col: str = "target_label",
        weight_col: str = "sample_weight",
    ) -> dict:
        """Execute out-of-sample cross-validation using purged and embargoed splits."""
        logger.info(f"Starting Purged CV evaluation ({n_splits} folds, embargo={pct_embargo})...")

        cv = PurgedGroupTimeSeriesSplit(n_splits=n_splits, pct_embargo=pct_embargo)
        
        pdf = df.to_pandas()
        
        accuracies = []
        log_losses = []
        roc_aucs = []

        # Store out-of-sample predictions aligned with index
        oos_preds_list = []
        oos_indices_list = []

        classes_mapping = {-1: 0, 0: 1, 1: 2}

        for fold, (train_idx, test_idx) in enumerate(cv.split(pdf)):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            train_df = df.slice(int(train_idx[0]), len(train_idx)) # Approximate slice or take by index
            # To be precise with arbitrary indices:
            train_df = pl.from_pandas(pdf.iloc[train_idx])
            test_df = pl.from_pandas(pdf.iloc[test_idx])

            trainer = QuantModelTrainer(n_estimators=100, max_depth=5, learning_rate=0.05)
            
            # Train on train fold
            trainer.train(train_df, feature_cols=feature_cols, target_col=target_col, weight_col=weight_col)

            # Predict on test fold
            X_test = test_df.select(feature_cols).to_pandas()
            y_test_raw = test_df[target_col].to_pandas()
            y_test = y_test_raw.map(classes_mapping).astype(int)

            if len(X_test) == 0:
                continue

            preds_proba = trainer.predict_proba(test_df, feature_cols=feature_cols)
            preds_class = np.argmax(preds_proba, axis=1)

            acc = accuracy_score(y_test, preds_class)
            loss = log_loss(y_test, preds_proba, labels=[0, 1, 2])
            
            try:
                auc = roc_auc_score(y_test, preds_proba, multi_class="ovr", average="macro", labels=[0, 1, 2])
            except Exception:
                auc = 0.0

            accuracies.append(acc)
            log_losses.append(loss)
            roc_aucs.append(auc)

            oos_preds_list.append(preds_proba)
            oos_indices_list.append(test_idx)

            logger.info(f"Fold {fold+1} Metrics -> Accuracy: {acc:.4f}, LogLoss: {loss:.4f}, ROC-AUC: {auc:.4f}")

        if not accuracies:
            raise ValueError("Cross-validation failed: no valid folds evaluated.")

        mean_metrics = {
            "mean_accuracy": float(np.mean(accuracies)),
            "mean_log_loss": float(np.mean(log_losses)),
            "mean_roc_auc_macro": float(np.mean(roc_aucs)),
            "folds_evaluated": len(accuracies),
        }

        logger.success(f"Purged CV Evaluation Complete. Consolidated Metrics: {mean_metrics}")
        return {
            "metrics": mean_metrics,
            "fold_accuracies": accuracies,
            "fold_log_losses": log_losses,
            "fold_roc_aucs": roc_aucs,
        }
