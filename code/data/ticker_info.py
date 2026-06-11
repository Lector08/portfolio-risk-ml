# =============================================================================
# data/ticker_info.py — Автоматичне визначення сектора для будь-якого тікера
# =============================================================================

import os
import json
# numpy не використовується — прибрано невикористаний імпорт

CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "saved", "ticker_sectors.json"
)

KNOWN_SECTORS = {
    "AAPL":  "Technology",       "MSFT":  "Technology",
    "NVDA":  "Technology",       "AMZN":  "Consumer Discretionary",
    "GOOGL": "Communication Services", "META": "Communication Services",
    "TSLA":  "Consumer Discretionary", "BRK-B": "Financials",
    "AVGO":  "Technology",       "JPM":   "Financials",
    "XOM":   "Energy",           "JNJ":   "Healthcare",
    "GS":    "Financials",       "KO":    "Consumer Staples",
    "PLTR":  "Technology",       "SCHD":  "ETF",
    "SPY":   "ETF",              "QQQ":   "ETF",
    "GLD":   "ETF",              "TLT":   "ETF",
    "XLK":   "ETF",              "XLF":   "ETF",
    "XLV":   "ETF",              "XLE":   "ETF",
    "XLI":   "ETF",              "XLY":   "ETF",
    "XLP":   "ETF",              "XLU":   "ETF",
    "XLRE":  "ETF",              "XLB":   "ETF",
    "XLC":   "ETF",              "VTI":   "ETF",
    "VOO":   "ETF",              "IWM":   "ETF",
    "AGG":   "ETF",              "VNQ":   "ETF",
    "ARKK":  "ETF",              "SOXX":  "ETF",
    "MSTR":  "Technology",       "ARM":   "Technology",
    "AMD":   "Technology",       "INTC":  "Technology",
    "ORCL":  "Technology",       "CRM":   "Technology",
    "NFLX":  "Communication Services",
    "DIS":   "Communication Services",
    "V":     "Financials",       "MA":    "Financials",
    "BAC":   "Financials",       "WFC":   "Financials",
    "UNH":   "Healthcare",       "LLY":   "Healthcare",
    "ABBV":  "Healthcare",       "MRK":   "Healthcare",
    "CVX":   "Energy",           "COP":   "Energy",
    "WMT":   "Consumer Staples", "PG":    "Consumer Staples",
    "PEP":   "Consumer Staples", "COST":  "Consumer Staples",
    "HD":    "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE":   "Consumer Discretionary",
    "NEE":   "Utilities",        "DUK":   "Utilities",
    "PLD":   "Real Estate",      "AMT":   "Real Estate",
    "LIN":   "Materials",        "APD":   "Materials",
    "CAT":   "Industrials",      "HON":   "Industrials",
    "GE":    "Industrials",      "BA":    "Industrials",
    "UNP":   "Industrials",
}

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
    "ETF":                    "SPY",
}


def _load_cache() -> dict:
    try:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_cache(cache: dict):
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


_disk_cache = _load_cache()


def get_ticker_sector(ticker: str) -> tuple[str, bool]:
    """
    Повертає (sector, is_etf) для тікера.
    Порядок пошуку: KNOWN_SECTORS → disk cache → yfinance API → fallback "Technology"
    """
    if ticker in KNOWN_SECTORS:
        sector = KNOWN_SECTORS[ticker]
        is_etf = (sector == "ETF")
        return ("Technology" if is_etf else sector), is_etf

    if ticker in _disk_cache:
        entry  = _disk_cache[ticker]
        sector = entry.get("sector", "Technology")
        is_etf = entry.get("is_etf", False)
        return sector, is_etf

    try:
        import yfinance as yf
        info       = yf.Ticker(ticker).info
        quote_type = info.get("quoteType", "").upper()
        is_etf     = quote_type in ("ETF", "MUTUALFUND")
        sector     = "Technology" if is_etf else _normalize_sector(info.get("sector"))
        _disk_cache[ticker] = {"sector": sector, "is_etf": is_etf}
        _save_cache(_disk_cache)
        return sector, is_etf
    except Exception:
        return "Technology", False


def get_sector_etf(sector: str) -> str:
    return SECTOR_ETF.get(sector, "SPY")


def _normalize_sector(raw: str | None) -> str:
    if not raw:
        return "Technology"
    mapping = {
        "Technology": "Technology", "Information Technology": "Technology",
        "Healthcare": "Healthcare", "Health Care": "Healthcare",
        "Financials": "Financials", "Financial Services": "Financials",
        "Financial": "Financials",
        "Consumer Discretionary": "Consumer Discretionary",
        "Consumer Cyclical": "Consumer Discretionary",
        "Consumer Staples": "Consumer Staples",
        "Consumer Defensive": "Consumer Staples",
        "Energy": "Energy",
        "Industrials": "Industrials", "Industrial Conglomerates": "Industrials",
        "Materials": "Materials", "Basic Materials": "Materials",
        "Real Estate": "Real Estate",
        "Utilities": "Utilities",
        "Communication Services": "Communication Services",
        "Communication": "Communication Services",
        "Telecommunications": "Communication Services",
    }
    if raw in mapping:
        return mapping[raw]
    raw_lower = raw.lower()
    for key, val in mapping.items():
        if key.lower() in raw_lower or raw_lower in key.lower():
            return val
    return "Technology"


def batch_fetch_sectors(tickers: list[str]) -> dict[str, tuple[str, bool]]:
    result, unknown = {}, []
    for t in tickers:
        if t in KNOWN_SECTORS:
            s = KNOWN_SECTORS[t]
            result[t] = ("Technology" if s == "ETF" else s, s == "ETF")
        elif t in _disk_cache:
            entry = _disk_cache[t]
            result[t] = (entry["sector"], entry["is_etf"])
        else:
            unknown.append(t)
    for t in unknown:
        result[t] = get_ticker_sector(t)
    return result
