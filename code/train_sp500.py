# =============================================================================
# train_sp500.py — Навчання XGBoost: 19 тех. + 6 динам.сект. = 25 ознак
#
# Ключова відмінність від попередньої версії:
#   БУЛО: one-hot сектора (sector_0..sector_10) — статичні константи,
#         модель їх ігнорувала (важливість < 1%)
#   СТАЛО: 6 динамічних секторних ознак (sector_ret_5d, sector_vol_20d, etc.)
#         — числа що змінюються кожен день, несуть реальну інформацію
# =============================================================================

import sys, os, warnings, time, random
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models", "saved")
MODEL_PATH  = os.path.join(MODELS_DIR, "xgb_sp500.json")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_sp500.npy")
META_PATH   = os.path.join(MODELS_DIR, "sp500_meta.json")
os.makedirs(MODELS_DIR, exist_ok=True)

SECTORS = [
    "Technology", "Healthcare", "Financials", "Consumer Discretionary",
    "Consumer Staples", "Energy", "Industrials", "Materials",
    "Real Estate", "Utilities", "Communication Services",
]


# ---------------------------------------------------------------------------
# Всесвіт інструментів
# ---------------------------------------------------------------------------

def get_sp500_with_sectors() -> pd.DataFrame:
    try:
        print("  S&P 500 з Wikipedia...", end=" ")
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0][["Symbol", "GICS Sector", "Security"]].copy()
        df.columns = ["ticker", "sector", "name"]
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        df["is_etf"] = 0
        print(f"✅ {len(df)} компаній")
        return df
    except Exception as e:
        print(f"⚠️  {e}")
        return pd.DataFrame(columns=["ticker", "sector", "name", "is_etf"])


def get_top_etfs() -> pd.DataFrame:
    etfs = {
        "SPY":"Technology","QQQ":"Technology","IWM":"Financials",
        "VTI":"Technology","VOO":"Technology","DIA":"Industrials",
        "XLK":"Technology","XLF":"Financials","XLV":"Healthcare",
        "XLE":"Energy","XLI":"Industrials","XLY":"Consumer Discretionary",
        "XLP":"Consumer Staples","XLU":"Utilities","XLRE":"Real Estate",
        "XLB":"Materials","XLC":"Communication Services",
        "EFA":"Technology","EEM":"Technology","AGG":"Financials",
        "BND":"Financials","LQD":"Financials","HYG":"Financials",
        "TLT":"Financials","GLD":"Materials","SLV":"Materials",
        "VNQ":"Real Estate","VXX":"Financials","ARKK":"Technology",
        "SOXX":"Technology","SMH":"Technology","SCHD":"Financials",
        "TQQQ":"Technology","SQQQ":"Technology",
    }
    rows = [{"ticker": t, "sector": s, "name": f"ETF:{t}", "is_etf": 1}
            for t, s in etfs.items()]
    df = pd.DataFrame(rows)
    print(f"  ETF: ✅ {len(df)} фондів")
    return df


def build_universe() -> pd.DataFrame:
    print("\n📋 Формування всесвіту...")
    sp500 = get_sp500_with_sectors()
    etfs  = get_top_etfs()
    universe = pd.concat([sp500, etfs], ignore_index=True)
    universe = universe.drop_duplicates(subset="ticker")
    universe["sector"] = universe["sector"].fillna("Technology")
    print(f"  Всього: {len(universe)} (акцій: {(universe['is_etf']==0).sum()}, ETF: {(universe['is_etf']==1).sum()})")
    return universe


# ---------------------------------------------------------------------------
# Завантаження цін із rate limiting
# ---------------------------------------------------------------------------

def download_with_retry(tickers, start="2018-01-01",
                        batch_size=20, delay_base=3.0,
                        delay_jitter=1.5, max_retries=3):
    import yfinance as yf
    from datetime import date
    end = date.today().strftime("%Y-%m-%d")
    n_batches  = (len(tickers) + batch_size - 1) // batch_size
    all_prices, failed = [], []
    t_start = time.time()

    print(f"\n  {len(tickers)} тікерів → {n_batches} батчів по {batch_size}")
    print(f"  Затримка: ~{delay_base:.0f}s між батчами (~{n_batches*delay_base/60:.0f} хв мінімум)\n")

    for i in range(n_batches):
        batch   = tickers[i * batch_size : (i + 1) * batch_size]
        success = False
        for attempt in range(max_retries):
            try:
                raw = yf.download(batch, start=start, end=end,
                                  auto_adjust=True, progress=False, timeout=30)
                if isinstance(raw.columns, pd.MultiIndex):
                    prices = raw["Close"]
                else:
                    col    = "Close" if "Close" in raw.columns else raw.columns[0]
                    prices = raw[[col]].rename(columns={col: batch[0]})
                prices  = prices.ffill(limit=3).dropna(how="all")
                ok_cols = [c for c in prices.columns if prices[c].count() >= 200]
                failed.extend([c for c in prices.columns if c not in ok_cols])
                if ok_cols:
                    all_prices.append(prices[ok_cols])
                success = True
                break
            except Exception as e:
                wait = delay_base * (2 ** attempt) + random.uniform(0, delay_jitter)
                print(f"\n    ⚠️  Спроба {attempt+1}: {str(e)[:50]} → {wait:.1f}s")
                time.sleep(wait)

        if not success:
            failed.extend(batch)

        done     = i + 1
        elapsed  = time.time() - t_start
        remaining = (n_batches - done) * (elapsed / done)
        ok_count  = sum(len(p.columns) for p in all_prices)
        print(f"  [{done:3d}/{n_batches}] ✅{ok_count:4d}  ❌{len(failed):3d}  "
              f"⏱ {elapsed/60:.1f}хв  залишилось ~{remaining/60:.0f}хв", end="\r")

        if i < n_batches - 1:
            time.sleep(delay_base + random.uniform(0, delay_jitter))

    print()
    if not all_prices:
        raise RuntimeError("Не вдалось завантажити жодного тікера")
    combined = pd.concat(all_prices, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    print(f"\n  ✅ Завантажено: {combined.shape[1]} тікерів, {combined.shape[0]} днів")
    return combined, failed


# ---------------------------------------------------------------------------
# Ознаки: 19 технічних + 6 динамічних секторних = 25
# ---------------------------------------------------------------------------

def compute_features(prices_series: pd.Series,
                     sector: str, is_etf: int,
                     sector_cache) -> pd.DataFrame | None:
    """
    25 ознак = 19 технічних (з даних тікера) + 6 динамічних (з даних сектора)
    + 1 is_etf = 26 ознак total
    """
    from data.sector_features import compute_sector_features

    try:
        r = np.log(prices_series / prices_series.shift(1)).dropna()
        if len(r) < 100:
            return None
        p = prices_series.loc[r.index]

        f = {}
        # 19 технічних ознак
        for w in [5, 20]:
            ma = p.rolling(w).mean()
            f[f"ma{w}_ratio"] = (p / ma - 1)
        delta = p.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta).clip(lower=0).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        f["rsi14"]     = (100 - 100 / (1 + rs)) / 50 - 1
        ma20  = p.rolling(20).mean()
        std20 = p.rolling(20).std()
        upper = ma20 + 2 * std20
        lb    = ma20 - 2 * std20
        f["pct_b"]     = ((p - lb) / (upper - lb).replace(0, np.nan)).clip(0, 1)
        f["bandwidth"] = ((upper - lb) / ma20.replace(0, np.nan))
        for w in [5, 10, 20]:
            f[f"std{w}"]  = r.rolling(w).std()
            f[f"skew{w}"] = r.rolling(w).skew()
            f[f"kurt{w}"] = r.rolling(w).kurt()
        for lag in [1, 2, 3, 5, 10]:
            f[f"lag{lag}"] = r.shift(lag)

        base_df = pd.DataFrame(f, index=p.index).dropna()

        # 6 динамічних секторних ознак
        sec_f = compute_sector_features(r, sector, cache=sector_cache)
        sec_f = sec_f.reindex(base_df.index).fillna(0.0)

        result = pd.concat([base_df, sec_f], axis=1)

        # ETF прапорець
        result["is_etf"] = float(is_etf)

        return result.dropna() if len(result.dropna()) > 50 else None

    except Exception:
        return None


def compute_target(prices_series: pd.Series) -> pd.Series | None:
    """Реалізована волатильність з лог-доходностей (однакова з per-ticker)."""
    try:
        r  = np.log(prices_series / prices_series.shift(1)).dropna()
        rv = r.pow(2).rolling(20).sum().apply(np.sqrt) * np.sqrt(252)
        return np.log1p(rv.dropna())
    except Exception:
        return None


def build_dataset(prices: pd.DataFrame,
                  universe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from data.sector_features import SectorCache, SECTOR_FEATURE_NAMES

    ticker_meta = universe.set_index("ticker").to_dict("index")

    # Попереднє завантаження всіх секторних ETF
    unique_sectors = list(universe["sector"].unique())
    print(f"  Завантаження {len(unique_sectors)} секторних ETF...")
    sector_cache = SectorCache()
    sector_cache.preload(unique_sectors, start="2015-01-01")

    X_list, y_list = [], []
    n_ok = n_skip = 0

    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if len(series) < 200:
            n_skip += 1
            continue

        meta       = ticker_meta.get(ticker, {})
        sector     = meta.get("sector", "Technology")
        is_etf     = int(meta.get("is_etf", 0))

        X_df = compute_features(series, sector, is_etf, sector_cache)
        y_s  = compute_target(series)

        if X_df is None or y_s is None:
            n_skip += 1
            continue

        common = X_df.index.intersection(y_s.index)
        if len(common) < 50:
            n_skip += 1
            continue

        X_list.append(X_df.loc[common].values.astype(np.float32))
        y_list.append(y_s.loc[common].values.astype(np.float32))
        n_ok += 1

    if not X_list:
        raise RuntimeError("Датасет порожній")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    tech_names = [
        "ma5_ratio","ma20_ratio","rsi14","pct_b","bandwidth",
        "std5","skew5","kurt5","std10","skew10","kurt10",
        "std20","skew20","kurt20","lag1","lag2","lag3","lag5","lag10",
    ]
    feature_names = tech_names + SECTOR_FEATURE_NAMES + ["is_etf"]

    print(f"\n  📊 Датасет:")
    print(f"     Інструментів: {n_ok}  (пропущено: {n_skip})")
    print(f"     Рядків:       {len(X):,}")
    print(f"     Ознак:        {X.shape[1]}  (19 тех. + 6 дин.сект. + 1 ETF)")
    return X, y, feature_names


# ---------------------------------------------------------------------------
# Навчання
# ---------------------------------------------------------------------------

def train(X, y, feature_names, universe):
    from sklearn.preprocessing import StandardScaler
    from models.volatility.xgboost_model import XGBoostVolatilityModel
    import json

    n  = len(X)
    t1 = int(n * 0.85)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n  Train: {t1:,}  |  Val: {n-t1:,}")
    model = XGBoostVolatilityModel()
    model.fit(X_scaled[:t1], y[:t1], X_scaled[t1:], y[t1:],
              feature_names=feature_names)

    print("\n  Метрики:")
    metrics = model.evaluate(X_scaled[t1:], y[t1:])

    model.save(MODEL_PATH)
    np.save(SCALER_PATH, {"mean": scaler.mean_, "scale": scaler.scale_})

    meta = {
        "n_features":    int(X.shape[1]),
        "feature_names": feature_names,
        "n_instruments": int(universe.shape[0]),
        "n_equities":    int((universe["is_etf"] == 0).sum()),
        "n_etfs":        int((universe["is_etf"] == 1).sum()),
        "sectors":       SECTORS,
        "feature_type":  "19_technical + 6_dynamic_sector + 1_etf_flag",
        "metrics":       {k: round(float(v), 6) for k, v in metrics.items()},
        "trained_at":    pd.Timestamp.now().isoformat(),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  💾 {MODEL_PATH}")
    print(f"  💾 {SCALER_PATH}")
    print(f"  💾 {META_PATH}")

    print("\n  🏆 Топ-15 ознак:")
    imp     = model.get_feature_importance(top_n=15)
    max_val = imp.max()
    from data.sector_features import SECTOR_FEATURE_NAMES
    for feat, val in imp.items():
        bar  = "█" * int(val / max_val * 20)
        flag = "🆕" if feat in SECTOR_FEATURE_NAMES else "  "
        print(f"  {flag} {feat:<22} {bar} {val:.4f}")

    return model, scaler, metrics


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  XGBoost: 19 тех. + 6 динамічних секторних ознак")
    print("=" * 65)

    if os.path.exists(MODEL_PATH):
        import json
        from datetime import datetime
        mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
        meta  = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}
        print(f"\n  Існуюча модель: {mtime.strftime('%Y-%m-%d %H:%M')}")
        print(f"    Інструментів: {meta.get('n_instruments','?')}")
        print(f"    Ознак: {meta.get('n_features','?')} ({meta.get('feature_type','?')})")
        print(f"    RMSE: {meta.get('metrics',{}).get('rmse','?')}")
        ans = input("\n  Перенавчити? [y/N]: ").strip().lower()
        if ans != "y":
            return

    t0 = time.time()

    universe         = build_universe()
    print(f"\n📥 Завантаження цін...")
    prices, _        = download_with_retry(
        universe["ticker"].tolist(),
        start="2018-01-01", batch_size=20,
        delay_base=3.0, delay_jitter=1.5, max_retries=3,
    )
    print(f"\n🔧 Формування датасету (з секторними ETF)...")
    X, y, feat_names = build_dataset(prices, universe)
    print(f"\n🔄 Навчання...")
    model, scaler, metrics = train(X, y, feat_names, universe)

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  ✅ ГОТОВО! Час: {elapsed/60:.1f} хв")
    print(f"  RMSE: {metrics['rmse']:.4f}  (per-ticker еталон: ~0.015)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
