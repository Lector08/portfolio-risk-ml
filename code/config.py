# =============================================================================
# config.py — Глобальна конфігурація проекту
# =============================================================================

from datetime import date

# Список тикерів акцій (S&P 500, різні сектори)
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "XOM", "JNJ"]

# Часовий діапазон
START_DATE = "2018-01-01"
END_DATE   = date.today().strftime("%Y-%m-%d")   # завжди актуальна дата

# Параметри ризик-менеджменту
CONFIDENCE_LEVEL = 0.95
HOLDING_PERIOD   = 1       # днів

# Параметри Монте-Карло
N_MONTE_CARLO    = 10_000
RISK_FREE_RATE   = 0.04    # річна безризикова ставка

# Параметри LSTM
LSTM_SEQUENCE_LEN = 60
LSTM_EPOCHS       = 50
LSTM_BATCH_SIZE   = 32
LSTM_UNITS        = [128, 64]

# Параметри XGBoost
XGB_N_ESTIMATORS  = 500
XGB_MAX_DEPTH     = 6
XGB_LEARNING_RATE = 0.05

# Шляхи
DATA_RAW_PATH       = "data/raw/"
DATA_PROCESSED_PATH = "data/processed/"
MODELS_SAVE_PATH    = "models/saved/"
FIGURES_PATH        = "../docs/figures/"
