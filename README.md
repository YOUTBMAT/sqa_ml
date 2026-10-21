# SQA-ML v2.0

Framework de pesquisa e execução sistemática para criptoativos, com foco atual em ETH/USDT no timeframe de 1h. A versão 2.0 reúne validação temporal purged, engenharia de fatores, seleção MDA, controle de drift e sizing com risco de portfólio. **LIVE permanece desativado**; os resultados atuais não comprovam edge sustentável e não devem ser interpretados como recomendação de operação.

## Arquitetura

O fluxo de pesquisa é:

1. Coleta de candles OHLCV via CCXT e cálculo de features em `src/features/`.
2. Rotulagem Meta-Labeling / Triple Barrier em `src/labeling/`, com TP = 1,5 ATR, SL = 1 ATR e barreira temporal configurável.
3. Pesos por unicidade e eventos concorrentes.
4. Seleção de features por **Purged MDA** nos folds temporais OOS.
5. Treino LightGBM e avaliação por Purged Walk-Forward.
6. Calibração de thresholds em simulação contínua, separando calibração e holdout.

Na execução PAPER, o runner compara populações de features não condicionadas a eventos, consulta o modelo e delega sizing, stops e limites ao `PortfolioManager`. O `ExecutionEngine` e o `PaperTrader` aceitam somente PAPER e aplicam limites de portfólio; fechamentos parciais contabilizam apenas a quantidade encerrada.

## Componentes v2.0

### Ciclo de vida de ordens — `src/execution/fsm.py`

`OrderFSM` aplica transições válidas entre `PENDING_SUBMIT`, `OPEN`, `PARTIALLY_FILLED`, `FILLED`, `CANCELING`, `EXPIRED`, `REJECTED` e `FAILED`. Registra histórico e rejeita mudanças ilegais. O engine PAPER encaminha ordens ao `PaperTrader`, que usa a FSM em ordens de abertura, encerramento e rejeição. A integração de execução LIVE/CCXT foi removida.

### Data drift por PSI — `src/validation/drift.py`

`DataDriftDetector` compara distribuições de features de treino e live usando Population Stability Index. O runner observa `vol_parkinson`, `frac_diff_close`, `vwap_dist` e `rsi_adaptive`; PSI > 0,25 ou referência indisponível ativa o modo protetor, que impede novas entradas.

| PSI | Interpretação | Ação |
|---:|---|---|
| < 0,10 | Estável | Normal |
| 0,10–0,25 | Alerta | Monitorar |
| > 0,25 | Severo | Pausar entradas |

### Seleção de features — `src/features/selection.py`

`PurgedFeatureSelector` calcula Mean Decrease Accuracy como a queda da ROC AUC OOS após permutar uma feature, em folds do `PurgedGroupTimeSeriesSplit`. `scripts/select_features.py` mantém features com MDA médio positivo e grava a lista em `config/selected_features.json`. Por padrão, a seleção exclui o período final reservado para holdout, evitando usar rótulos de holdout para escolher features. Cada novo treino recalcula a seleção em staging antes de qualquer promoção.

### Gestão de portfólio — `src/risk/portfolio.py`

`PortfolioManager` transforma probabilidade e payoff em fração de Kelly, aplica um fator fracionado de segurança e normaliza por ATR/preço. Também aplica limite de exposição total, drawdown diário/global, stops ATR e circuit breaker de volatilidade. O simulador contínuo usa as probabilidades OOS para o mesmo sizing e os mesmos limites de exposição. Configuração padrão: Kelly 0,25, drawdown diário 3%, drawdown global 15%, exposição aberta máxima 20% e bloqueio quando ATR/preço atinge 10%.

Equações principais:

\[
f^* = \frac{p b - (1-p)}{b}, \qquad
f_{\mathrm{frac}} = \max(0, f^*) \cdot 0{,}25
\]

\[
s_{\mathrm{ATR}} = \min\left(1,\frac{2\%}{ATR/\text{preço}}\right), \qquad
\text{notional} = \min(\text{saldo}\cdot f_{\mathrm{frac}}\cdot s_{\mathrm{ATR}}, \text{exposição restante})
\]

O runner limita `p` ao win rate observado antes de calcular Kelly; probabilidades estimadas não calibradas não podem aumentar o risco acima dessa taxa.

### Automação segura — `scripts/autotrain.py`

Executa coleta OHLCV, engenharia de features, rebuild do dataset, Purged MDA, treino e calibração/holdout em sequência. O modelo, metadata e features selecionadas são produzidos em staging. O script mantém `models/saved/autotrain_safety_lock.json` ativo durante o processo e em caso de falha; a promoção e remoção do lock só ocorrem quando existe threshold calibrável e o holdout apresenta PF > 1 e retorno positivo. Um advisory lock do kernel evita execução concorrente e é liberado automaticamente se o processo terminar abruptamente.

## Status de validação

- Seleção MDA de ETH/USDT: ROC AUC walk-forward OOS 0,5334 no experimento reportado.
- A calibração de 24h selecionou threshold 0,50, mas o holdout ficou em PF 0,60 e retorno -12,07%; não aprovada para operação.
- Na análise 12h/24h/48h, 12h teve PF OOS 1,2897 e retorno holdout +0,34%, mas PF holdout 1,1173. Não atingiu o critério mais rigoroso de PF > 1,20 também no holdout.
- Não há evidência atual de edge robusto. Modo PAPER apenas; o modo LIVE segue explicitamente desabilitado.

Consulte [ROADMAP.md](ROADMAP.md) para o histórico das fases de auditoria e etapas v2.0.

## Instalação e comandos

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Com `PYTHONPATH=src:config:.`:

```bash
./venv/bin/python scripts/run_collector.py --symbols ETH/USDT --timeframe 1h --days 365
./venv/bin/python scripts/generate_features.py --symbols ETH/USDT --timeframe 1h
./venv/bin/python scripts/build_dataset.py --symbol ETH/USDT
./venv/bin/python scripts/select_features.py --symbol ETH/USDT
./venv/bin/python scripts/train_model.py --symbol ETH/USDT
./venv/bin/python scripts/calibrate_thresholds.py --symbol ETH/USDT
```

Automação completa (requer exchange acessível):

```bash
PYTHONPATH=src:config:. ./venv/bin/python scripts/autotrain.py --symbol ETH/USDT
```

Verificações:

```bash
./venv/bin/pytest
PYTHONPATH=src:config:. ./venv/bin/python scripts/verify_phase1.py
PYTHONPATH=src:config:. ./venv/bin/python scripts/verify_phase2_3.py
PYTHONPATH=src:config:. ./venv/bin/python scripts/verify_phase4.py
```

## Estrutura do repositório

```text
config/                 Configuração e features selecionadas
data/                   Candles, features, datasets e estado PAPER
models/saved/           Modelo, metadata e lock de segurança do autotrain
scripts/                Coleta, features, seleção, treino, calibração e pesquisa
src/data/               Coletores e construção de datasets
src/features/           Pipeline, fatores, armazenamento e Purged MDA
src/labeling/           Triple Barrier e pesos de amostras
src/models/              LightGBM, validação e persistência
src/backtest/            Simulador contínuo e métricas
src/execution/           Runner, engine, PaperTrader e FSM
src/risk/                PortfolioManager e limites de risco
src/validation/          Purged CV e detector PSI
tests/                   Testes unitários e de regressão
```
