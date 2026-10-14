# SQA-ML: Sistema Autônomo de Trading Quantitativo com Machine Learning

## 🏛️ Visão Geral
O **SQA-ML** é uma plataforma avançada de trading quantitativo e MLOps projetada para ingestão em tempo real de dados de mercado (criptomoedas e ações), armazenamento de alta performance via **TimescaleDB**, treinamento de modelos preditivos e execução autônoma de estratégias.

---

## 📁 Estrutura do Projeto
```
sqa_ml/
├── config/
│   ├── __init__.py
│   ├── settings.py          # Configurações via Pydantic Settings
│   └── .env.example         # Template de variáveis de ambiente
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── database.py      # Conexão SQLAlchemy / Psycopg & Sessões
│   │   └── models.py        # Modelos ORM e Hypertables TimescaleDB
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py        # Logging estruturado com Loguru
│   │   └── healthcheck.py   # Validação de DB e APIs CCXT
├── tests/
│   ├── __init__.py
│   └── test_healthcheck.py
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Pré-requisitos & Instalação

1. **Python 3.11+**
2. **Docker & Docker Compose**

### 1. Clonar e Configurar Ambiente Virtual
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configurar Variáveis de Ambiente
Copie o arquivo de exemplo e ajuste se necessário:
```bash
cp config/.env.example config/.env
```

---

## 🐳 Executando a Infraestrutura (TimescaleDB)

Suba o container do TimescaleDB via Docker Compose:
```bash
docker-compose up -d
```

---

## 🔍 Executando o Healthcheck

Para validar a conexão com o banco de dados, a ativação da extensão TimescaleDB e a conectividade com as APIs públicas via CCXT:
```bash
PYTHONPATH=src:config:. python src/utils/healthcheck.py
```

---

## 🧪 Executando os Testes Unitários

```bash
PYTHONPATH=src:config:. pytest
```
