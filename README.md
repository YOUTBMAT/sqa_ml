# SQA-ML: Systematic Quantitative Architecture & Machine Learning Trading Bot

> **Institutional Quantitative Framework for Automated Crypto Assets Trading (1h Timeframe)**

---

## 1. Visão Geral
**SQA-ML** é um framework quantitativo institucional projetado para negociação automatizada e sistemática em mercados de criptoativos de alta liquidez. O sistema fundamenta-se nos princípios rigorosos de engenharia financeira quantitativa definidos por **Marcos López de Prado**, incorporando **Meta-Labeling**, **Método de Tripla Barreira (Triple Barrier Method)**, e **Purged Walk-Forward Cross-Validation** para garantir zero vazamento de dados futuros (*zero lookahead bias*) e robustez estatística em regime *out-of-sample*.

---

## 2. Objetivos do Projeto
- **Expectativa Matemática Positiva:** Desenvolvimento e validação de estratégias quantitativas com *Profit Factor > 1.25* e *Sharpe Ratio > 1.20* em dados *Out-Of-Sample* (OOS).
- **Simulação Realista Event-Driven:** Backtesting e simulação rigorosos com contabilização de custos operacionais e taxas reais (*taker/maker fee* de 0.05% por perna), com métricas calculadas estritamente por *round-trip* (operações encerradas).
- **Automação 24/7:** Execução autônoma contínua em ambiente Linux (Ubuntu Systemd) integrada via CCXT, operando em modo *Paper Trading* para estresse contínuo na Binance Spot API.

---

## 3. Relatório de Remediação e Auditoria Quantitativa (Fases 1 a 4)
O sistema passou por uma auditoria estrutural e matemática profunda dividida em 4 fases evolutivas:
1. **Fase 1 (Ingestão e Coleta Robusta):** Expansão da janela histórica para 1.000 velas horárias via CCXT, incorporando tratamento de falhas, timeouts e retransmissões exponenciais.
2. **Fase 2 (Persistência Atômica e Sanitização):** Implementação de escrita JSON atômica em operações de estado de portfólio (`paper_portfolio.json`) e limpeza estrita de valores nulos e infinitos no pipeline de dados.
3. **Fase 3 (Engine de Backtest Contínuo Barra a Barra):** Substituição do simulador de sinais esparsos por um motor *Event-Driven* contínuo barra a barra, eliminando saltos de sinais e permitindo rastreamento exato de posições abertas, *Take-Profit*, *Stop-Loss* e *Max Hold Hours*.
4. **Fase 4 (Regras de Warmup, Limpeza e Calibração):** Aplicação de regras de *warmup* para indicadores técnicos, alinhamento de timestamps e varredura paramétrica de limiares de probabilidade $P(y=1)$ entre 0.50 e 0.65.

---

## 4. Arquitetura da Estratégia (Squeeze + Meta-Labeling)
A arquitetura adota um modelo de duas camadas (Sinal Primário + Filtro de Machine Learning):

- **Sinal Primário (Filtro de Regime / Squeeze):**
  - Detecção de compressão de volatilidade através do indicador *Vol Ratio* (`vol_ratio_6_24 < 0.75`) combinado com rompimento dinâmico de VWAP.
- **Modelo Secundário (Meta-Labeling com LightGBM):**
  - Classificador Binário **LightGBM** regularizado (`max_depth=3`, `num_leaves=7`, `learning_rate=0.03`), treinado estritamente nas 10 principais *features* estacionárias ranqueadas por ganho de informação (*Gain*).
- **Gerenciamento de Risco (Triplo Barreira):**
  - **Take-Profit (TP):** $1.5 \times \text{ATR}$
  - **Stop-Loss (SL):** $1.0 \times \text{ATR}$
  - **Barreira Temporal ($t_1$):** Encerramento compulsório após 24 horas.
- **Limiar Operacional de Confirmação (Paper):**
  - O sinal primário para o ativo **ETH/USDT** é executado apenas quando $P(y=1) \ge 0.62$ (`settings.PROB_THRESHOLD`). Ver seção 5 antes de considerar capital real.

### Proteção contra Data Drift
Antes de abrir uma nova posição, o runner compara as features recentes (`vol_parkinson`, `frac_diff_close`, `vwap_dist` e `rsi_adaptive`) com a distribuição do dataset de treino usando PSI. Se qualquer feature ultrapassar 0.25, novas entradas para o ativo são pausadas até a distribuição normalizar. A ausência de dataset de referência ou de features válidas também mantém o ativo em modo protetor.

| PSI | Interpretação | Ação |
| :---: | --- | --- |
| `< 0.10` | Estável | Operação normal |
| `0.10–0.25` | Alerta | Monitorar |
| `> 0.25` | Desvio severo | Pausar novas entradas |

---

## 5. Desempenho Out-Of-Sample (Engine Contínuo Barra a Barra)
Resultados com probabilidades **somente out-of-sample** (Purged Walk-Forward), simuladas sobre as barras horárias contínuas com as mesmas barreiras do rótulo (SL vence em barra ambígua, gaps preenchem na abertura), custos de 0,07% por lado. Reproduza com `scripts/run_backtest.py --symbol ETH/USDT` e `scripts/calibrate_thresholds.py`.

| Ativo / Par | OOS ROC AUC | Taxa base (TP primeiro) | Threshold | Trades | Win Rate | Profit Factor | Retorno Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ETH/USDT | 0.489 | 40.1% | 0.62 | 24 | 41.7% | 0.73 | -4.24% |
| BTC/USDT | 0.484 | 41.0% | 0.62 | 13 | 38.5% | 0.85 | -0.82% |

**Status: sem edge comprovado.** O AUC fora da amostra é indistinguível de 0,5 e nenhum threshold entre 0,50 e 0,65 atinge Profit Factor >= 1 no período de calibração. Opere apenas em PAPER até um modelo passar no holdout de `calibrate_thresholds.py`.

> O valor publicado anteriormente (+9,32%, PF 1,81) vinha de rodar o backtest sobre o dataset esparso (apenas linhas com sinal), onde "24 barras" correspondiam a dias de mercado e a máxima/mínima de horas não observadas era ignorada.

---

## 6. Estrutura do Repositório
```tree
sqa_ml/
├── config/              # Configurações globais e de parâmetros (settings.py, .env)
├── data/                # Bases de dados históricas, datasets processados e logs de erro
├── models/              # Artefatos salvos de modelos treinados (.joblib)
├── scripts/             # Scripts executáveis de pipeline (build, treino, backtest, live)
├── src/
│   ├── data/            # Coordenadores e coletores CCXT (crypto.py)
│   ├── features/        # Engenharia de features, retornos e estacionariedade
│   ├── labels/          # Rotulagem Tripla Barreira e Meta-Labeling
│   ├── models/          # Implementação e treino do LightGBM
│   ├── backtest/        # Motor de backtest event-driven contínuo
│   ├── execution/       # Runner live, gerenciador de ordens e risk management
│   ├── risk/            # Gestão de risco e dimensionamento de posição
│   └── validation/      # Purged Walk-Forward CV
├── tests/               # Testes unitários e de integração (pytest)
├── requirements.txt     # Dependências do projeto Python
├── sqa_ml_bot.service   # Arquivo de serviço systemd para Ubuntu
└── README.md            # Documentação técnica do projeto
```

---

## 7. Guia de Execução e Comandos Systemd

### 1. Configuração do Ambiente Virtual
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Construção de Datasets e Treinamento
```bash
PYTHONPATH=src:config:. python scripts/build_dataset.py --symbol ETH/USDT
PYTHONPATH=src:config:. python scripts/train_model.py --symbol ETH/USDT
PYTHONPATH=src:config:. python scripts/calibrate_thresholds.py --symbol ETH/USDT
```

### 3. Testes Unitários de Regressão
```bash
pytest
```

### 4. Gerenciamento do Serviço Systemd (Paper Mode 24/7)
Para controlar o robô de trading em modo *Paper* no Ubuntu:
```bash
# Verificar status do serviço
systemctl --user status sqa_ml_bot.service --no-pager

# Reiniciar o bot
systemctl --user restart sqa_ml_bot.service

# Parar o bot
systemctl --user stop sqa_ml_bot.service

# Acompanhar logs em tempo real
journalctl --user -u sqa_ml_bot.service -n 25 --no-pager
```
