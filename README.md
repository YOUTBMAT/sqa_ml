# SQA-ML: Systematic Quantitative Architecture & Machine Learning Trading Bot

> **Institutional Quantitative Framework for Automated Crypto Assets Trading (1h Timeframe)**

---

## 1. Visão Geral
**SQA-ML** é um framework quantitativo institucional projetado para negociação automatizada e sistemática em mercados de criptoativos de alta liquidez. O sistema fundamenta-se nos princípios rigorosos de engenharia financeira quantitativa definidos por **Marcos López de Prado**, incorporando **Meta-Labeling**, **Método de Tripla Barreira (Triple Barrier Method)**, e **Purged Walk-Forward Cross-Validation** para garantir zero vazamento de dados futuros (*zero lookahead bias*) e robustez estatística em regime *out-of-sample*.

---

## 2. Objetivos do Projeto
- **Expectativa Matemática Positiva:** Desenvolvimento e validação de estratégias quantitativas com *Profit Factor > 1.25* e *Sharpe Ratio > 1.20* em dados *Out-Of-Sample* (OOS).
- **Simulação Realista Event-Driven:** Backtesting e simulação rigorosos com contabilização de custos operacionais e taxas reais (*taker/maker fee* de 0.05% por perna), com métricas calculadas estritamente por *round-trip* (operações encerradas).
- **Automação 24/7:** Execução autônoma contínua em ambiente Linux (Ubuntu Systemd) integrada via CCXT, operando inicialmente em modo *Paper Trading* antes do deploy em conta real na Binance Spot API.

---

## 3. Resumo da Remediação da Auditoria Quantitativa
O sistema passou por uma auditoria estrutural e matemática profunda para eliminação de vieses e falhas de engenharia:
1. **Correção de Dados:** Substituição de arquivos locais duplicados ou artificiais por ingestão direta e limpa de séries históricas reais via CCXT Spot. Exclusão formal de `SOL/USDT` da cesta ativa devido à incompatibilidade com o regime do setup.
2. **Validação sem Lookahead Bias:** Remoção do K-Fold padrão e adoção de **Purged Walk-Forward Cross-Validation** com janela expansiva, aplicando purga temporal rigorosa baseada na coluna de eventos temporal `t1`.
3. **Unificação de Features:** Desenvolvimento de um pipeline de engenharia de atributos 100% estacionário e idêntico entre o treinamento, o backtest e a inferência em tempo real (*live runner*).
4. **Correção do Motor Paper:** Implementação de suporte completo a posições *SHORT*, verificação contínua e ativa de *Stop-Loss* / *Take-Profit* dinâmicos no *runner*, filtro de *Candle Lock* para barras fechadas e sistema robusto de retentativas exponenciais contra interrupções/timeouts da API CCXT.

---

## 4. Arquitetura da Estratégia (Squeeze + Meta-Labeling)
A arquitetura adota um modelo de duas camadas (Sinal Primário + Filtro de Machine Learning):

- **Sinal Primário (Filtro de Regime / Squeeze):**
  - Detecção de compressão de volatilidade através do indicador *Vol Ratio* (`vol_ratio_6_24 < 0.75`) combinado com rompimento dinâmico de VWAP.
- **Modelo Secundário (Meta-Labeling com LightGBM):**
  - Classificador Binário **LightGBM** regularizado (`max_depth=3`, `num_leaves=7`, `learning_rate=0.03`), treinado estritamente nas 10 principais *features* ranqueadas por ganho de informação (*Gain*).
- **Gerenciamento de Risco (Triplo Barreira):**
  - **Take-Profit (TP):** $1.5 \times \text{ATR}$
  - **Stop-Loss (SL):** $1.0 \times \text{ATR}$
  - **Barreira Temporal ($t_1$):** Encerramento compulsório após 24 horas.
- **Limiar Operacional de Confirmação:**
  - O sinal primário só é executado se a probabilidade estimada pelo modelo secundário atender ao limiar: $P(y=1) \ge 0.65$.

---

## 5. Resultados de Validação Multi-Ativos (Threshold $\ge 0.65$)
A tabela abaixo consolida o desempenho *Out-Of-Sample* (OOS) verificado após a auditoria e calibração de limiares para o timeframe de 1h:

| Ativo / Par | Profit Factor | Win Rate (%) | Sharpe Ratio | Retorno Total OOS | Amostra (N Trades) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BTC/USDT** | **1.62** | 53.12% | **1.91** | +5.83% | 32 | **ATIVO** |
| **ETH/USDT** | **1.94** | 57.89% | **3.20** | +10.08% | 19 | **ATIVO** |
| **SOL/USDT** | 0.77 | 41.67% | -1.16 | -2.38% | 12 | **INATIVO (Excluído)** |

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
│   ├── backtest/        # Motor de backtest event-driven
│   ├── execution/       # Runner live, gerenciador de ordens e risk management
│   ├── risk/            # Gestão de risco e dimensionamento de posição
│   └── validation/      # Purged Walk-Forward CV
├── tests/               # Testes unitários e de integração (pytest)
├── requirements.txt     # Dependências do projeto Python
├── sqa_ml_bot.service   # Arquivo de serviço systemd para Ubuntu
└── README.md            # Documentação técnica do projeto
```

---

## 7. Guia de Execução e Comandos

### 1. Configuração do Ambiente Virtual
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Construção de Datasets Históricos
Para baixar e processar os dados históricos via CCXT para a cesta ativa:
```bash
python scripts/build_dataset.py
```

### 3. Treinamento e Validação (Purged Walk-Forward CV)
Para treinar o modelo LightGBM com validação cruzada sem vazamento:
```bash
python scripts/train_model.py
```

### 4. Calibração de Thresholds
Para calibrar a probabilidade ótima de disparo do Meta-Labeling:
```bash
python scripts/calibrate_thresholds.py
```

### 5. Execução de Backtest Event-Driven
Para rodar a simulação histórica completa com custos e slippage:
```bash
python scripts/run_backtest.py
```

### 6. Gerenciamento do Serviço Systemd (Paper Mode)
Para controlar o robô de trading em modo *Paper* no Ubuntu:
```bash
# Verificar status do serviço
systemctl --user status sqa_ml_bot.service --no-pager

# Reiniciar o bot
systemctl --user restart sqa_ml_bot.service

# Parar o bot
systemctl --user stop sqa_ml_bot.service

# Acompanhar logs em tempo real
journalctl --user -u sqa_ml_bot.service -f
```
