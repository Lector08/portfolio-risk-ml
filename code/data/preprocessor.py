# =============================================================================
# data/preprocessor.py — Передобробка: log-returns, train/val/test split, scaling
# MILESTONE 1.2
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Розраховує логарифмічні доходності з цін закриття.

    Формула: r_t = ln(P_t / P_{t-1})

    Переваги лог-доходностей:
    - Адитивні в часі: r(0→2) = r(0→1) + r(1→2)
    - Симетричні: +10% і -10% мають однаковий масштаб
    - Краще апроксимуються нормальним розподілом (важливо для VaR)

    Args:
        prices: DataFrame з цінами закриття (рядки = дати, стовпці = тикери)

    Returns:
        pd.DataFrame: лог-доходності (перший рядок NaN — дропається)
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    print(f"[preprocessor] Лог-доходності: {log_returns.shape} (рядки × тикери)")
    return log_returns


def train_val_test_split(
    data: pd.DataFrame,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Хронологічний розподіл даних на train / validation / test.

    ВАЖЛИВО: НЕ перемішуємо рядки! Фінансові ряди мають часову залежність.
    Перемішування призведе до data leakage — модель "побачить майбутнє".

    Розподіл:
        [========= train 80% =========][= val 10% =][= test 10% =]
        2018                           2022          2023          2024

    Args:
        data:       DataFrame (рядки = дати, будь-які стовпці)
        val_ratio:  частка для валідації (default 10%)
        test_ratio: частка для тесту (default 10%)

    Returns:
        tuple: (train, val, test) DataFrames
    """
    n = len(data)
    test_size = int(n * test_ratio)
    val_size  = int(n * val_ratio)
    train_size = n - val_size - test_size

    train = data.iloc[:train_size]
    val   = data.iloc[train_size : train_size + val_size]
    test  = data.iloc[train_size + val_size :]

    print(f"[preprocessor] Split: train={len(train)} | val={len(val)} | test={len(test)}")
    print(f"  Train: {train.index[0].date()} → {train.index[-1].date()}")
    print(f"  Val:   {val.index[0].date()} → {val.index[-1].date()}")
    print(f"  Test:  {test.index[0].date()} → {test.index[-1].date()}")

    return train, val, test


def scale_features(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Нормалізує ознаки за допомогою StandardScaler.

    Scaler навчається ТІЛЬКИ на train — щоб уникнути data leakage.
    val і test трансформуються з параметрами train (mean, std).

    Args:
        train, val, test: numpy arrays форми (n_samples, n_features)

    Returns:
        tuple: (train_scaled, val_scaled, test_scaled, fitted_scaler)
               scaler потрібен щоб inverse_transform прогнози потім
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled   = scaler.transform(val)
    test_scaled  = scaler.transform(test)

    print(f"[preprocessor] Scaling: mean≈0, std≈1 (fit на train)")
    return train_scaled, val_scaled, test_scaled, scaler


def compute_realized_volatility(
    log_returns: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Розраховує реалізовану волатильність — цільову змінну для ML-моделей.

    Формула: RV_t = sqrt( sum_{i=t-w+1}^{t} r_i^2 )
    Це стандартний проксі для "істинної" волатильності в літературі.

    Args:
        log_returns: DataFrame з лог-доходностями
        window:      розмір ковзного вікна (default 20 = ~1 місяць)

    Returns:
        pd.DataFrame: реалізована волатильність, анулізована на 252 торгових дні
    """
    # Сума квадратів за вікном → корінь → ануалізація
    rv = log_returns.pow(2).rolling(window).sum().apply(np.sqrt)
    rv = rv * np.sqrt(252 / window)   # перетворюємо на річну волатильність
    rv = rv.dropna()

    print(f"[preprocessor] Реалізована волатильність: вікно={window} днів, shape={rv.shape}")
    return rv


# ---------------------------------------------------------------------------
# Швидкий тест при запуску напряму: python data/preprocessor.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from data.loader import get_prices

    prices = get_prices()
    returns = compute_log_returns(prices)
    train, val, test = train_val_test_split(returns)
    rv = compute_realized_volatility(returns)

    print("\n--- Статистика доходностей (AAPL) ---")
    print(returns["AAPL"].describe().round(5))
    print("\n--- Реалізована волатильність (перші 3 рядки) ---")
    print(rv.head(3).round(4))
