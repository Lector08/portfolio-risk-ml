# =============================================================================
# data/loader.py — Завантаження ринкових даних через yfinance
# MILESTONE 1.1
# =============================================================================

import os
import yfinance as yf
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import TICKERS, START_DATE, END_DATE, DATA_RAW_PATH


def download_prices(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    save: bool = True,
) -> pd.DataFrame:
    """
    Завантажує скориговані ціни закриття для списку тикерів через yfinance.

    Args:
        tickers: список тикерів, наприклад ["AAPL", "MSFT"]
        start:   початкова дата у форматі "YYYY-MM-DD"
        end:     кінцева дата у форматі "YYYY-MM-DD"
        save:    зберегти у data/raw/prices.csv

    Returns:
        pd.DataFrame: рядки = торгові дати, стовпці = тикери (Adj Close)
    """
    print(f"[loader] Завантаження {len(tickers)} тикерів: {tickers}")
    print(f"[loader] Період: {start} → {end}")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,   # Adj Close автоматично замінює Close
        progress=False,
    )

    # Якщо тикер один — yfinance повертає плоский DataFrame
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Прибираємо дні без торгів (вихідні, свята)
    prices = prices.dropna(how="all")

    # Forward-fill пропуски всередині ряду (не більше 2 днів поспіль)
    prices = prices.ffill(limit=2)

    # Дропаємо рядки з пропусками що залишились
    prices = prices.dropna()

    print(f"[loader] Завантажено {len(prices)} торгових днів, {prices.shape[1]} тикерів")
    print(f"[loader] Перший день: {prices.index[0].date()}, останній: {prices.index[-1].date()}")

    if save:
        os.makedirs(DATA_RAW_PATH, exist_ok=True)
        path = os.path.join(DATA_RAW_PATH, "prices.csv")
        prices.to_csv(path)
        print(f"[loader] Збережено → {path}")

    return prices


def load_cached_prices(path: str = None) -> pd.DataFrame:
    """
    Завантажує кешовані ціни з CSV (якщо вже качали раніше).

    Args:
        path: шлях до файлу; якщо None — використовує DATA_RAW_PATH/prices.csv

    Returns:
        pd.DataFrame: ціни закриття (рядки = дати, стовпці = тикери)

    Raises:
        FileNotFoundError: якщо кешований файл не існує
    """
    if path is None:
        path = os.path.join(DATA_RAW_PATH, "prices.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Кешовані дані не знайдено: {path}\n"
            "Спочатку запусти download_prices()"
        )

    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"[loader] Завантажено з кешу: {len(prices)} рядків, {prices.shape[1]} тикерів")
    return prices


def get_prices(force_download: bool = False) -> pd.DataFrame:
    """
    Зручна точка входу: повертає ціни з кешу або качає заново.

    Args:
        force_download: якщо True — завжди завантажує заново

    Returns:
        pd.DataFrame: скориговані ціни закриття
    """
    cache_path = os.path.join(DATA_RAW_PATH, "prices.csv")

    if not force_download and os.path.exists(cache_path):
        print("[loader] Знайдено кеш, використовуємо локальні дані")
        return load_cached_prices(cache_path)

    return download_prices(save=True)


# ---------------------------------------------------------------------------
# Швидкий тест при запуску напряму: python data/loader.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = get_prices(force_download=True)
    print("\n--- Перші 5 рядків ---")
    print(df.head())
    print("\n--- Статистика ---")
    print(df.describe().round(2))
