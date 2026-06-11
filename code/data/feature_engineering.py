# =============================================================================
# data/feature_engineering.py — Технічні індикатори та формування LSTM-sequences
# MILESTONE 1.3
# =============================================================================

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Технічні індикатори
# ---------------------------------------------------------------------------

def add_moving_averages(
    prices: pd.DataFrame,
    windows: list[int] = [5, 20],
) -> pd.DataFrame:
    """
    Додає ковзні середні MA та нормалізований відхід ціни від MA.

    Ознаки для кожного тикера і вікна w:
        - price_ma{w}_ratio = Close / MA(w) - 1   (відносне відхилення)

    Args:
        prices:  DataFrame з цінами (рядки = дати, стовпці = тикери)
        windows: список розмірів вікон

    Returns:
        pd.DataFrame: нові ознаки (кількість стовпців = n_tickers × len(windows))
    """
    features = {}
    for ticker in prices.columns:
        for w in windows:
            ma = prices[ticker].rolling(w).mean()
            features[f"{ticker}_ma{w}_ratio"] = prices[ticker] / ma - 1
    return pd.DataFrame(features, index=prices.index)


def add_rsi(
    prices: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Розраховує RSI (Relative Strength Index) для кожного тикера.

    Формула:
        RS  = mean(gains over period) / mean(losses over period)
        RSI = 100 - 100 / (1 + RS)

    RSI > 70 → перекупленість, RSI < 30 → перепроданість.
    Нормалізуємо до [-1, 1] щоб масштаб відповідав іншим ознакам.

    Args:
        prices: DataFrame з цінами
        period: розмір вікна RSI (default 14)

    Returns:
        pd.DataFrame: RSI значення в діапазоні [-1, 1]
    """
    features = {}
    for ticker in prices.columns:
        delta = prices[ticker].diff()
        gains  = delta.clip(lower=0)
        losses = (-delta).clip(lower=0)

        avg_gain = gains.rolling(period).mean()
        avg_loss = losses.rolling(period).mean()

        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Нормалізуємо: [0, 100] → [-1, 1]
        features[f"{ticker}_rsi{period}"] = rsi / 50 - 1

    return pd.DataFrame(features, index=prices.index)


def add_bollinger_bands(
    prices: pd.DataFrame,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """
    Додає ознаки на основі смуг Боллінджера.

    Смуги Боллінджера = MA ± n_std * rolling_std
    Ознака: %B = (price - lower) / (upper - lower)
              %B = 0 → ціна на нижній смузі
              %B = 1 → ціна на верхній смузі

    Args:
        prices: DataFrame з цінами
        window: розмір вікна (default 20)
        n_std:  кількість стандартних відхилень (default 2)

    Returns:
        pd.DataFrame: %B та bandwidth для кожного тикера
    """
    features = {}
    for ticker in prices.columns:
        ma  = prices[ticker].rolling(window).mean()
        std = prices[ticker].rolling(window).std()

        upper = ma + n_std * std
        lower = ma - n_std * std

        band_width = (upper - lower) / ma.replace(0, np.nan)
        pct_b      = (prices[ticker] - lower) / (upper - lower).replace(0, np.nan)

        features[f"{ticker}_pct_b"]      = pct_b.clip(0, 1)   # обрізаємо [0, 1]
        features[f"{ticker}_bandwidth"]  = band_width

    return pd.DataFrame(features, index=prices.index)


def add_rolling_stats(
    log_returns: pd.DataFrame,
    windows: list[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Статистичні ознаки: rolling std, skewness, kurtosis доходностей.

    Ці ознаки описують "форму" розподілу доходностей за останні w днів —
    важлива інформація для прогнозу волатильності.

    Args:
        log_returns: DataFrame з лог-доходностями
        windows:     список розмірів вікон

    Returns:
        pd.DataFrame: статистичні ознаки
    """
    features = {}
    for ticker in log_returns.columns:
        r = log_returns[ticker]
        for w in windows:
            features[f"{ticker}_std{w}"]  = r.rolling(w).std()
            features[f"{ticker}_skew{w}"] = r.rolling(w).skew()
            features[f"{ticker}_kurt{w}"] = r.rolling(w).kurt()
    return pd.DataFrame(features, index=log_returns.index)


def add_lag_returns(
    log_returns: pd.DataFrame,
    lags: list[int] = [1, 2, 3, 5, 10],
) -> pd.DataFrame:
    """
    Лагові доходності — базові ознаки для ML-моделей часових рядів.

    Args:
        log_returns: DataFrame з лог-доходностями
        lags:        список лагів у торгових днях

    Returns:
        pd.DataFrame: лагові ознаки
    """
    features = {}
    for ticker in log_returns.columns:
        for lag in lags:
            features[f"{ticker}_lag{lag}"] = log_returns[ticker].shift(lag)
    return pd.DataFrame(features, index=log_returns.index)


# ---------------------------------------------------------------------------
# Збирання фінальної матриці ознак
# ---------------------------------------------------------------------------

def build_feature_matrix(
    prices: pd.DataFrame,
    log_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Збирає всі ознаки в єдину матрицю X.

    Склад ознак:
        - MA(5), MA(20) ratio           → трендові сигнали
        - RSI(14)                       → моментум
        - Bollinger %B, bandwidth       → волатильність відносно норми
        - Rolling std/skew/kurt (5,10,20) → статистика розподілу
        - Lag returns (1,2,3,5,10)      → авторегресія

    Args:
        prices:      DataFrame з цінами закриття
        log_returns: DataFrame з лог-доходностями

    Returns:
        pd.DataFrame: матриця ознак (рядки де є NaN — дропаються)
    """
    parts = [
        add_moving_averages(prices),
        add_rsi(prices),
        add_bollinger_bands(prices),
        add_rolling_stats(log_returns),
        add_lag_returns(log_returns),
    ]

    X = pd.concat(parts, axis=1)
    X = X.dropna()

    print(f"[feature_engineering] Матриця ознак: {X.shape} (рядки × ознаки)")
    print(f"[feature_engineering] Ознак всього: {X.shape[1]}")
    return X


# ---------------------------------------------------------------------------
# Формування 3D-тензора для LSTM
# ---------------------------------------------------------------------------

def create_lstm_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Перетворює 2D матрицю ознак на 3D тензор для LSTM.

    LSTM очікує input форми: (samples, timesteps, features)
    Ми формуємо "ковзне вікно" розміром seq_len.

    Приклад з seq_len=3:
        X = [a, b, c, d, e]
        →  sequences = [[a,b,c], [b,c,d], [c,d,e]]
        →  targets   = [d, e, f]  (наступний крок після вікна)

    Args:
        X:       2D array (n_samples, n_features) — вже нормалізовані ознаки
        y:       1D або 2D array цільових значень (наприклад, реалізована волатильність)
        seq_len: розмір вікна (default 60 = ~3 місяці торгових днів)

    Returns:
        tuple: (X_seq, y_seq)
               X_seq: (n_samples - seq_len, seq_len, n_features)
               y_seq: (n_samples - seq_len,) або (n_samples - seq_len, n_targets)
    """
    X_seq, y_seq = [], []

    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len : i])   # вікно [i-60 : i]
        y_seq.append(y[i])                  # таргет — наступний день

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    print(f"[feature_engineering] LSTM sequences: X={X_seq.shape}, y={y_seq.shape}")
    print(f"  Вікно: {seq_len} днів → {X_seq.shape[0]} зразків")

    return X_seq, y_seq


# ---------------------------------------------------------------------------
# Швидкий тест при запуску напряму: python data/feature_engineering.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from data.loader import get_prices
    from data.preprocessor import compute_log_returns

    prices  = get_prices()
    returns = compute_log_returns(prices)

    X = build_feature_matrix(prices, returns)
    print("\n--- Перші 3 ознаки ---")
    print(X.iloc[:3, :5].round(4))

    # Тест LSTM sequences
    X_arr = X.values.astype(np.float32)
    y_arr = returns["AAPL"].reindex(X.index).values.astype(np.float32)
    X_seq, y_seq = create_lstm_sequences(X_arr, y_arr, seq_len=60)
    print(f"\nLSTM input tensor: {X_seq.shape}")
