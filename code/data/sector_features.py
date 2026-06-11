# =============================================================================
# data/sector_features.py — Динамічні секторні ознаки
#
# Ключова ідея: секторні ETF (XLK, XLF, XLV...) як proxy поведінки сектора.
# 6 ознак що змінюються кожен день і несуть реальну інформацію:
#
#   sector_ret_5d    — середня доходність сектора за 5 днів
#   sector_ret_20d   — середня доходність сектора за 20 днів
#   sector_vol_20d   — поточна волатильність сектора (rolling 20d)
#   rel_to_sector    — акція vs сектор: хто сильніший за 20 днів
#   sector_trend     — MA5 > MA20 для сектора (+1 / -1)
#   sector_breadth   — частка позитивних днів сектора за 10 днів
# =============================================================================

import numpy as np
import pandas as pd
from typing import Optional

# ETF proxy для кожного сектора GICS
SECTOR_ETF = {
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Financials":             "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":       "XLP",
    "Energy":                 "XLE",
    "Industrials":            "XLI",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
    "Communication Services": "XLC",
    "ETF":                    "SPY",  # для самих ETF — широкий ринок
}

SECTOR_FEATURE_NAMES = [
    "sector_ret_5d",
    "sector_ret_20d",
    "sector_vol_20d",
    "rel_to_sector",
    "sector_trend",
    "sector_breadth",
]


class SectorCache:
    """
    Завантажує та кешує доходності секторних ETF.
    Один раз на сесію — потім використовує кеш.
    """
    def __init__(self):
        self._cache: dict[str, pd.Series] = {}

    def get(self, sector: str, start: str = "2015-01-01") -> Optional[pd.Series]:
        """
        Повертає лог-доходності ETF-proxy для сектора.
        При першому запиті завантажує з yfinance, далі з кешу.
        """
        if sector in self._cache:
            return self._cache[sector]

        etf = SECTOR_ETF.get(sector, "SPY")
        try:
            import yfinance as yf
            from datetime import date
            raw = yf.download(etf, start=start,
                              end=date.today().strftime("%Y-%m-%d"),
                              auto_adjust=True, progress=False)
            if raw.empty:
                return None
            prices  = raw["Close"].squeeze()
            log_ret = np.log(prices / prices.shift(1)).dropna()
            self._cache[sector] = log_ret
            return log_ret
        except Exception:
            return None

    def preload(self, sectors: list[str], start: str = "2015-01-01"):
        """Завантажує всі потрібні ETF за один раз (batch)."""
        etfs = list({SECTOR_ETF.get(s, "SPY") for s in sectors})
        try:
            import yfinance as yf
            from datetime import date
            raw = yf.download(etfs, start=start,
                              end=date.today().strftime("%Y-%m-%d"),
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw[["Close"]].rename(columns={"Close": etfs[0]})

            for sector in sectors:
                etf = SECTOR_ETF.get(sector, "SPY")
                if etf in prices.columns:
                    s = prices[etf].dropna()
                    self._cache[sector] = np.log(s / s.shift(1)).dropna()
            print(f"  Секторні ETF завантажено: {len(self._cache)} секторів")
        except Exception as e:
            print(f"  ⚠️  Не вдалось завантажити секторні ETF: {e}")


# Глобальний кеш — один екземпляр на весь процес
_global_cache = SectorCache()


def get_sector_cache() -> SectorCache:
    return _global_cache


def compute_sector_features(
    ticker_log_returns: pd.Series,
    sector:             str,
    cache:              Optional[SectorCache] = None,
) -> pd.DataFrame:
    """
    Розраховує 6 динамічних секторних ознак для тікера.

    Args:
        ticker_log_returns: лог-доходності тікера (pd.Series, DatetimeIndex)
        sector:             назва сектора GICS
        cache:              SectorCache екземпляр (або None → глобальний)

    Returns:
        pd.DataFrame: 6 стовпців, індекс = дати тікера
    """
    if cache is None:
        cache = _global_cache

    sector_ret = cache.get(sector)

    # Fallback: якщо ETF недоступний
    if sector_ret is None or len(sector_ret) == 0:
        return pd.DataFrame(
            {col: 0.0 for col in SECTOR_FEATURE_NAMES},
            index=ticker_log_returns.index,
        )

    # Вирівнюємо індекси
    common = ticker_log_returns.index.intersection(sector_ret.index)
    if len(common) < 25:
        return pd.DataFrame(
            {col: 0.0 for col in SECTOR_FEATURE_NAMES},
            index=ticker_log_returns.index,
        )

    t = ticker_log_returns.loc[common]
    s = sector_ret.loc[common]

    feat = pd.DataFrame(index=common)

    # 1. Доходність сектора за 5 і 20 днів
    feat["sector_ret_5d"]  = s.rolling(5).mean()
    feat["sector_ret_20d"] = s.rolling(20).mean()

    # 2. Поточна волатильність сектора
    feat["sector_vol_20d"] = s.rolling(20).std()

    # 3. Відносна сила тікера vs сектор
    #    >0: тікер обганяє сектор, <0: відстає
    feat["rel_to_sector"] = t.rolling(20).mean() - s.rolling(20).mean()

    # 4. Тренд сектора: MA5 > MA20 → +1.0, інакше → -1.0
    s_ma5  = s.rolling(5).mean()
    s_ma20 = s.rolling(20).mean()
    feat["sector_trend"] = np.where(s_ma5 > s_ma20, 1.0, -1.0)

    # 5. Breadth: частка позитивних днів за 10
    feat["sector_breadth"] = (s > 0).astype(float).rolling(10).mean()

    # Реіндексуємо на всі дати тікера і заповнюємо нулями
    return feat.reindex(ticker_log_returns.index).fillna(0.0)
