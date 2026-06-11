"""
app.py — Portfolio Risk Manager
Запуск: python3 -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, sys, warnings, json
from datetime import date, datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

# Імпорт KNOWN_SECTORS на рівні модуля (БАГ виправлено: замість __import__ хаку)
from data.ticker_info import KNOWN_SECTORS

st.set_page_config(
    page_title="Portfolio Risk Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

TOP10 = {
    "AAPL":"Apple", "MSFT":"Microsoft", "NVDA":"NVIDIA", "AMZN":"Amazon",
    "GOOGL":"Alphabet", "META":"Meta", "TSLA":"Tesla",
    "BRK-B":"Berkshire", "AVGO":"Broadcom", "JPM":"JPMorgan",
}

SECTOR_EMOJI = {
    "Technology":"💻","Healthcare":"🏥","Financials":"🏦",
    "Consumer Discretionary":"🛍","Consumer Staples":"🛒",
    "Energy":"⚡","Industrials":"🏭","Materials":"⛏",
    "Real Estate":"🏢","Utilities":"💡","Communication Services":"📡",
    "ETF":"📦",
}

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models", "saved")
SP500_MODEL = os.path.join(MODELS_DIR, "xgb_sp500.json")
SP500_SCALER= os.path.join(MODELS_DIR, "scaler_sp500.npy")
SP500_META  = os.path.join(MODELS_DIR, "sp500_meta.json")
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Сектор для тікера
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def get_ticker_info(ticker: str) -> tuple[str, bool]:
    from data.ticker_info import get_ticker_sector
    return get_ticker_sector(ticker)


def resolve_sectors(tickers: list[str]) -> dict[str, tuple[str, bool]]:
    result = {}
    for t in tickers:
        try:
            result[t] = get_ticker_info(t)
        except Exception:
            result[t] = ("Technology", False)
    return result


# ---------------------------------------------------------------------------
# Ознаки
# ---------------------------------------------------------------------------

def build_features_base(prices_series: pd.Series) -> pd.DataFrame:
    r = np.log(prices_series / prices_series.shift(1)).dropna()
    p = prices_series.loc[r.index]
    f = {}
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
    return pd.DataFrame(f, index=p.index).dropna()


@st.cache_resource(show_spinner=False)
def get_sector_cache_instance():
    from data.sector_features import SectorCache
    return SectorCache()


def build_features_sp500(prices_series: pd.Series,
                          sector: str, is_etf: int) -> pd.DataFrame:
    from data.sector_features import compute_sector_features
    base = build_features_base(prices_series)
    if base.empty:
        return base
    r = np.log(prices_series / prices_series.shift(1)).dropna()
    cache = get_sector_cache_instance()
    sec_f = compute_sector_features(r, sector, cache=cache)
    sec_f = sec_f.reindex(base.index).fillna(0.0)
    result = pd.concat([base, sec_f], axis=1)
    result["is_etf"] = float(is_etf)
    return result.dropna()


# ---------------------------------------------------------------------------
# Ціни
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Завантаження котирувань...")
def load_prices(tickers: tuple, start: str = "2020-01-01") -> pd.DataFrame:
    import yfinance as yf
    end = date.today().strftime("%Y-%m-%d")
    raw = yf.download(list(tickers), start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].dropna(how="all").ffill(limit=2).dropna()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        prices = prices.dropna().ffill(limit=2).dropna()
    return prices


# ---------------------------------------------------------------------------
# Per-ticker моделі
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_per_ticker_model(ticker: str, _hash: int):
    from models.volatility.xgboost_model import XGBoostVolatilityModel
    from sklearn.preprocessing import StandardScaler
    mp = os.path.join(MODELS_DIR, f"xgb_{ticker}_v2.json")
    sp = os.path.join(MODELS_DIR, f"scaler_{ticker}_v2.npy")
    model  = XGBoostVolatilityModel()
    scaler = StandardScaler()
    if os.path.exists(mp) and os.path.exists(sp):
        model.load(mp)
        p = np.load(sp, allow_pickle=True).item()
        scaler.mean_  = p["mean"]
        scaler.scale_ = p["scale"]
        scaler.n_features_in_ = len(p["mean"])
        return model, scaler, "loaded"
    return model, scaler, "needs_training"


def train_per_ticker(ticker: str, prices_series: pd.Series):
    from models.volatility.xgboost_model import XGBoostVolatilityModel
    from sklearn.preprocessing import StandardScaler
    mp = os.path.join(MODELS_DIR, f"xgb_{ticker}_v2.json")
    sp = os.path.join(MODELS_DIR, f"scaler_{ticker}_v2.npy")
    X_df = build_features_base(prices_series)
    r    = np.log(prices_series / prices_series.shift(1)).dropna()
    rv   = r.pow(2).rolling(20).sum().apply(np.sqrt) * np.sqrt(252)
    rv   = np.log1p(rv.dropna())
    common = X_df.index.intersection(rv.index)
    X = X_df.loc[common].values.astype(np.float32)
    y = rv.loc[common].values.astype(np.float32)
    n = len(X); t1 = int(n * 0.85)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBoostVolatilityModel()
    model.fit(X_scaled[:t1], y[:t1], X_scaled[t1:], y[t1:],
              feature_names=list(X_df.columns))
    model.save(mp)
    np.save(sp, {"mean": scaler.mean_, "scale": scaler.scale_})
    return model, scaler


# ---------------------------------------------------------------------------
# S&P 500 модель
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Завантаження S&P 500 моделі...")
def load_sp500_model():
    from models.volatility.xgboost_model import XGBoostVolatilityModel
    from sklearn.preprocessing import StandardScaler
    if not (os.path.exists(SP500_MODEL) and os.path.exists(SP500_SCALER)):
        return None, None, {}
    model  = XGBoostVolatilityModel()
    model.load(SP500_MODEL)
    scaler = StandardScaler()
    p = np.load(SP500_SCALER, allow_pickle=True).item()
    scaler.mean_  = p["mean"]
    scaler.scale_ = p["scale"]
    scaler.n_features_in_ = len(p["mean"])
    meta = json.load(open(SP500_META)) if os.path.exists(SP500_META) else {}
    return model, scaler, meta


# ---------------------------------------------------------------------------
# Прогноз
# ---------------------------------------------------------------------------

def predict_vol(ticker: str, prices_series: pd.Series,
                model, scaler,
                sector: str = "Technology", is_etf: int = 0,
                use_sp500: bool = False) -> float:
    X_df = (build_features_sp500(prices_series, sector, is_etf)
            if use_sp500 else build_features_base(prices_series))
    if X_df.empty:
        return float(prices_series.pct_change().std())
    x_last = scaler.transform(X_df.values[-1:].astype(np.float32))
    sigma  = float(model.predict(x_last)[0])
    return max(np.expm1(sigma) / np.sqrt(252), 1e-4)


def delete_per_ticker_models():
    import glob
    for f in glob.glob(os.path.join(MODELS_DIR, "xgb_*_v2.json")):
        os.remove(f)
    for f in glob.glob(os.path.join(MODELS_DIR, "scaler_*_v2.npy")):
        os.remove(f)
    st.cache_resource.clear()


# ---------------------------------------------------------------------------
# Оптимізація
# ---------------------------------------------------------------------------

def optimize_portfolio(tickers, weights_input, returns, lb, ub, r_min=None):
    from models.portfolio.markowitz import MarkowitzOptimizer, InfeasibleConstraintError
    from data.weighting             import ewma_cov_matrix

    train_r  = returns.dropna().iloc[-504:]
    mu       = train_r.mean().values * 252
    cov_ewma = ewma_cov_matrix(train_r, lambda_=0.94)

    opt = MarkowitzOptimizer(mu, cov_ewma, tickers=list(tickers), risk_free_rate=0.04)

    if r_min is not None and r_min > 0:
        check = opt.check_min_return_feasibility(r_min, lb, ub)
        if not check["feasible"]:
            raise InfeasibleConstraintError(
                f"max_achievable={check['max_achievable']:.4f}"
            )

    w_ms = opt.max_sharpe(lb=lb, ub=ub, r_min=r_min)
    w_mv = opt.min_variance(lb=lb, ub=ub, r_min=r_min)
    ew   = np.clip(np.ones(len(tickers)) / len(tickers), lb, ub)
    ew  /= ew.sum()

    return {
        "max_sharpe":   (w_ms, opt.portfolio_metrics(w_ms)),
        "min_variance": (w_mv, opt.portfolio_metrics(w_mv)),
        "equal_weight": (ew,   opt.portfolio_metrics(ew)),
        "user_input":   (weights_input, opt.portfolio_metrics(weights_input)),
        "optimizer":    opt,
    }


def compute_var(returns, weights):
    from models.risk.var_cvar import historical_var, historical_cvar, parametric_var, parametric_cvar
    from data.weighting       import ewma_historical_var
    port_ret = (returns.values @ weights).flatten()
    mu_p  = float(np.mean(port_ret))
    sig_p = float(np.std(port_ret, ddof=1))
    return {
        "Hist (рівні ваги)":   (historical_var(port_ret),    historical_cvar(port_ret)),
        "BRW EWMA (λ=0.94)":   ewma_historical_var(port_ret, lambda_=0.94),
        "Parametric (σ виб.)": (parametric_var(mu_p, sig_p), parametric_cvar(mu_p, sig_p)),
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.title("📊 Система управління ризиками портфеля")
    st.caption("XGBoost волатильність · VaR / CVaR · Markowitz + EWMA + обмеження")

    # ── Сайдбар ─────────────────────────────────────────────────────────────
    st.sidebar.header("🗂 Портфель")
    if st.sidebar.button("⚡ Топ-10 за капіталізацією"):
        st.session_state["selected"] = list(TOP10.keys())
    if "selected" not in st.session_state:
        st.session_state["selected"] = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]

    custom_raw = st.sidebar.text_input("➕ Тікер", placeholder="NVDA, TSLA, MSTR...")
    if custom_raw:
        new_tickers = [x.strip().upper() for x in custom_raw.split(",") if x.strip()]
        for t in new_tickers:
            if t not in st.session_state["selected"]:
                st.session_state["selected"].append(t)
                with st.sidebar.spinner(f"Визначаю сектор {t}..."):
                    sector, is_etf = get_ticker_info(t)
                st.sidebar.caption(
                    f"{t}: {SECTOR_EMOJI.get(sector,'')}{sector}"
                    f"{' (ETF)' if is_etf else ''}"
                )

    st.sidebar.markdown("**Активні тікери:**")
    keep = []
    for t in st.session_state["selected"]:
        label = f"{t} — {TOP10[t]}" if t in TOP10 else t
        if st.sidebar.checkbox(label, value=True, key=f"chk_{t}"):
            keep.append(t)
    st.session_state["selected"] = keep

    tickers = tuple(sorted(st.session_state["selected"]))
    if not tickers:
        st.warning("Додайте хоча б один тікер")
        return

    start_date = st.sidebar.selectbox("Дані з:", ["2020-01-01", "2018-01-01", "2015-01-01"])

    # ── Режим ML ─────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🤖 Режим ML:**")

    sp500_exists = os.path.exists(SP500_MODEL)
    sp500_meta   = {}
    if sp500_exists:
        sp500_meta = json.load(open(SP500_META)) if os.path.exists(SP500_META) else {}
        n_inst = sp500_meta.get("n_instruments", "?")
        n_feat = sp500_meta.get("n_features", "?")
        rmse   = sp500_meta.get("metrics", {}).get("rmse", "?")
        model_mode = st.sidebar.radio(
            "Модель:",
            [f"Per-ticker (19 ознак)",
             f"S&P 500 ({n_feat} ознак, {n_inst} інстр.)"],
        )
        use_sp500 = "S&P 500" in model_mode
        if use_sp500:
            st.sidebar.success(f"✅ {n_inst} інстр. · {n_feat} ознак · RMSE={rmse}")
    else:
        use_sp500 = False
        st.sidebar.info("💡 Запусти `python3 train_sp500.py`")

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Перенавчати per-ticker"):
        delete_per_ticker_models()
        st.rerun()

    # ── Дані ─────────────────────────────────────────────────────────────────
    with st.spinner("Завантаження..."):
        try:
            prices = load_prices(tickers, start=start_date)
        except Exception as e:
            st.error(f"{e}")
            return

    available = [t for t in tickers if t in prices.columns]
    if not available:
        st.error("Немає даних")
        return

    with st.spinner("Визначення секторів..."):
        sector_info = resolve_sectors(list(tickers))

    returns = np.log(prices[available] / prices[available].shift(1)).dropna()

    # ── Прогнози ─────────────────────────────────────────────────────────────
    vol_forecasts = {}
    if use_sp500:
        sp500_model, sp500_scaler, _ = load_sp500_model()
        if sp500_model is None:
            st.error("Не вдалось завантажити S&P 500 модель")
            use_sp500 = False
        else:
            for t in available:
                sector, is_etf = sector_info.get(t, ("Technology", False))
                try:
                    vol_forecasts[t] = predict_vol(
                        t, prices[t], sp500_model, sp500_scaler,
                        sector=sector, is_etf=int(is_etf), use_sp500=True,
                    )
                except Exception:
                    vol_forecasts[t] = float(prices[t].pct_change().std())

    if not use_sp500:
        status_ph = st.sidebar.empty()
        for t in available:
            model, scaler, status = get_per_ticker_model(
                t, hash(float(prices[t].iloc[-1]))
            )
            if status == "needs_training":
                status_ph.info(f"🔄 {t}...")
                model, scaler = train_per_ticker(t, prices[t])
                get_per_ticker_model.clear()
            try:
                vol_forecasts[t] = predict_vol(t, prices[t], model, scaler)
            except Exception:
                vol_forecasts[t] = float(prices[t].pct_change().std())
        status_ph.success(f"✅ {len(available)} моделей готово")

    # ── Ініціалізація змінних портфеля ДО tabs (виправлення scope bug) ───────
    user_w_raw = {t: 0.0 for t in available}
    total_uw   = 0.0
    user_w     = np.ones(len(available)) / len(available)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Огляд", "⚖️ Оптимізація", "🛡 VaR / CVaR", "🤖 Моделі"
    ])

    # ══ TAB 1 ════════════════════════════════════════════════════════════════
    with tab1:
        rows = []
        for t in available:
            price   = float(prices[t].iloc[-1])
            ret_1d  = float(np.log(prices[t].iloc[-1] / prices[t].iloc[-2]) * 100)
            vol_a   = vol_forecasts[t] * np.sqrt(252) * 100
            hist_v  = float(prices[t].pct_change().std() * np.sqrt(252) * 100)
            sector, is_etf = sector_info.get(t, ("Technology", False))
            emoji   = SECTOR_EMOJI.get(sector, "")
            rows.append({
                "Тікер": t, "Назва": TOP10.get(t, t),
                "Сектор": f"{emoji} {sector}",
                "Ціна ($)": round(price, 2),
                "Зміна 1д (%)": round(ret_1d, 3),
                "ML σ̂ річна (%)": round(vol_a, 2),
                "Hist σ (%)": round(hist_v, 2),
                "Сигнал": "⚠️ Висока" if vol_a > hist_v * 1.2 else "✅ Норма",
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style
              .background_gradient(subset=["ML σ̂ річна (%)"], cmap="RdYlGn_r")
              .format({"Зміна 1д (%)": "{:+.3f}",
                       "ML σ̂ річна (%)": "{:.2f}",
                       "Hist σ (%)": "{:.2f}"}),
            use_container_width=True, hide_index=True,
        )

        sec_groups = {}
        for t in available:
            s, _ = sector_info.get(t, ("Technology", False))
            sec_groups.setdefault(s, []).append(vol_forecasts[t] * np.sqrt(252) * 100)
        if len(sec_groups) > 1:
            st.markdown("---")
            st.subheader("Волатильність по секторах")
            sec_df = pd.DataFrame([
                {"Сектор": f"{SECTOR_EMOJI.get(s,'')} {s}",
                 "Серед. σ̂ (%)": round(np.mean(v), 2),
                 "N тікерів": len(v)}
                for s, v in sec_groups.items()
            ]).sort_values("Серед. σ̂ (%)", ascending=False)
            st.dataframe(sec_df, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("Нормовані ціни (база = 100)")
        norm = prices[available] / prices[available].iloc[0] * 100
        st.line_chart(norm, use_container_width=True)

    # ══ TAB 2 — ОПТИМІЗАЦІЯ ══════════════════════════════════════════════════
    with tab2:
        st.subheader("Оптимізація портфеля (Markowitz + EWMA)")

        with st.expander("📌 Ваш поточний портфель (ваги у %)", expanded=False):
            cols_w = st.columns(min(len(available), 5))
            for i, t in enumerate(available):
                with cols_w[i % len(cols_w)]:
                    user_w_raw[t] = st.number_input(t, 0.0, 100.0, 0.0, 1.0, key=f"uw_{t}")
            total_uw = sum(user_w_raw.values())
            if total_uw > 0:
                user_w = np.array([user_w_raw[t] / 100 for t in available])
            if total_uw > 0 and abs(total_uw - 100) > 0.1:
                st.warning(f"Сума = {total_uw:.1f}%")

        st.markdown("---")
        st.subheader("📊 Мінімальна цільова доходність")

        col_sl, col_info = st.columns([2, 1])
        with col_sl:
            use_r_min = st.checkbox(
                "Задати мінімальну доходність портфеля",
                value=False,
            )
            r_min_pct = st.slider(
                "Мінімальна річна доходність (%)",
                min_value=0.0, max_value=20.0, value=8.0, step=0.5,
                format="%.1f%%", disabled=not use_r_min,
            )

        r_min = r_min_pct / 100.0 if use_r_min else None

        with col_info:
            if use_r_min:
                if r_min_pct <= 4.0:
                    st.info(f"ℹ️ {r_min_pct:.1f}% ≤ rf (4%) — краще держоблігації")
                elif r_min_pct <= 8.0:
                    st.success(f"✅ {r_min_pct:.1f}% — помірна мета")
                elif r_min_pct <= 15.0:
                    st.warning(f"⚠️ {r_min_pct:.1f}% — висока мета")
                else:
                    st.error(f"🔴 {r_min_pct:.1f}% — агресивна мета")

        # Перевірка досяжності r_min
        if use_r_min and r_min is not None:
            try:
                from models.portfolio.markowitz import MarkowitzOptimizer
                from data.weighting             import ewma_cov_matrix
                train_r   = returns.dropna().iloc[-504:]
                mu_check  = train_r.mean().values * 252
                cov_check = ewma_cov_matrix(train_r, lambda_=0.94)
                lb_check  = np.zeros(len(available))
                ub_check  = np.ones(len(available))
                opt_check = MarkowitzOptimizer(mu_check, cov_check,
                                               tickers=available, risk_free_rate=0.04)
                feasibility = opt_check.check_min_return_feasibility(r_min, lb_check, ub_check)
                max_ret = feasibility["max_achievable"]

                if not feasibility["feasible"]:
                    st.markdown("---")
                    st.error("## 🏦 Ринок не відповідає вашим вимогам")
                    st.markdown(f"""
**Задана мінімальна доходність:** {r_min_pct:.1f}%

**Максимально досяжна з поточних активів:** {max_ret*100:.1f}%

**Різниця (shortfall):** {feasibility['shortfall']*100:.1f}%

---
### Рекомендація: Держоблігації США (US Treasury)

Раціональний інвестор, який вимагає доходність не нижче {r_min_pct:.1f}%, але не може
отримати її на фондовому ринку, залишається у безризиковому активі.

| Актив | Доходність | Ризик |
|-------|:---:|:---:|
| T-Bills (3M) | ~5.0% | ~0% |
| T-Notes (2Y) | ~4.8% | ~1% |
| T-Bonds (10Y)| ~4.5% | ~8% |
                    """)

                    st.markdown("**Очікувана доходність по активах (EWMA, річна):**")
                    ret_rows = sorted([
                        {"Тікер": t,
                         "Очік. доходність (%)": round(mu_check[i]*100, 1),
                         "Сектор": f"{SECTOR_EMOJI.get(sector_info.get(t,('Technology',False))[0],'')} "
                                   f"{sector_info.get(t,('Technology',False))[0]}"}
                        for i, t in enumerate(available)
                    ], key=lambda x: -x["Очік. доходність (%)"])
                    st.dataframe(
                        pd.DataFrame(ret_rows).style.background_gradient(
                            subset=["Очік. доходність (%)"], cmap="RdYlGn"
                        ),
                        hide_index=True, use_container_width=True
                    )
                    st.stop()
                else:
                    st.success(
                        f"✅ Доходність {r_min_pct:.1f}% досяжна. "
                        f"Максимально можливо: **{max_ret*100:.1f}%**"
                    )
            except Exception as e:
                st.warning(f"Не вдалось перевірити досяжність: {e}")

        st.markdown("---")
        st.subheader("📏 Обмеження на позиції")

        lb_arr = np.zeros(len(available))
        ub_arr = np.ones(len(available))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("Max 25%", use_container_width=True):
                st.session_state["preset"] = "max25"
        with c2:
            if st.button("Max 40%", use_container_width=True):
                st.session_state["preset"] = "max40"
        with c3:
            if st.button("Min 5%", use_container_width=True):
                st.session_state["preset"] = "min5"
        with c4:
            if st.button("Без обмежень", use_container_width=True):
                st.session_state["preset"] = "none"
                for t in available:
                    st.session_state[f"use_lb_{t}"] = False
                    st.session_state[f"use_ub_{t}"] = False

        preset = st.session_state.get("preset", "none")
        if preset == "max25":
            for t in available:
                st.session_state[f"use_ub_{t}"] = True
                st.session_state[f"ub_val_{t}"] = 25.0
        elif preset == "max40":
            for t in available:
                st.session_state[f"use_ub_{t}"] = True
                st.session_state[f"ub_val_{t}"] = 40.0
        elif preset == "min5":
            for t in available:
                st.session_state[f"use_lb_{t}"] = True
                st.session_state[f"lb_val_{t}"] = 5.0

        st.markdown("---")
        for i, t in enumerate(available):
            sec, is_etf = sector_info.get(t, ("Technology", False))
            emoji = SECTOR_EMOJI.get(sec, "")
            col_n, col_lc, col_lv, col_uc, col_uv, col_p = st.columns([1.3,0.6,1.0,0.6,1.0,1.2])
            with col_n:
                st.markdown(f"**{t}** {emoji}")
            with col_lc:
                use_lb = st.checkbox("min %", value=st.session_state.get(f"use_lb_{t}", False), key=f"use_lb_{t}")
            with col_lv:
                lb_val = st.number_input("lb", 0.0, 99.0,
                                         float(st.session_state.get(f"lb_val_{t}", 0.0)),
                                         1.0, key=f"lb_val_{t}",
                                         label_visibility="collapsed", disabled=not use_lb)
            with col_uc:
                use_ub = st.checkbox("max %", value=st.session_state.get(f"use_ub_{t}", False), key=f"use_ub_{t}")
            with col_uv:
                ub_val = st.number_input("ub", 1.0, 100.0,
                                         float(st.session_state.get(f"ub_val_{t}", 100.0)),
                                         1.0, key=f"ub_val_{t}",
                                         label_visibility="collapsed", disabled=not use_ub)
            with col_p:
                if use_lb and use_ub:
                    st.markdown(f"🔒 `{lb_val:.0f}–{ub_val:.0f}%`")
                elif use_lb:
                    st.markdown(f"🔽 `min {lb_val:.0f}%`")
                elif use_ub:
                    st.markdown(f"🔼 `max {ub_val:.0f}%`")
                else:
                    st.markdown("✅")
            lb_arr[i] = (lb_val / 100) if use_lb else 0.0
            ub_arr[i] = (ub_val / 100) if use_ub else 1.0

        lb_sum = lb_arr.sum()
        issues = [f"**{t}**: min > max" for i,t in enumerate(available) if lb_arr[i] > ub_arr[i]]
        if lb_sum > 1.0:
            issues.append(f"Сума мінімумів {lb_sum*100:.1f}% > 100%")

        if issues:
            for msg in issues:
                st.error(f"❌ {msg}")
        else:
            if lb_sum > 0.01:
                st.info(f"ℹ️ Мінімуми: {lb_sum*100:.1f}% · вільна частина: {(1-lb_sum)*100:.1f}%")
            st.markdown("---")
            if st.button("🔍 Оптимізувати", type="primary", use_container_width=True):
                with st.spinner("Оптимізація..."):
                    try:
                        res = optimize_portfolio(available, user_w, returns,
                                                 lb_arr, ub_arr, r_min=r_min)
                        st.success("✅ Готово!")
                        rows = []
                        for key, label in [("user_input","Ваш"),("equal_weight","Equal-W"),
                                           ("min_variance","Min-Var"),("max_sharpe","Max-Sharpe")]:
                            w, m = res[key]
                            mark = " ✅" if (r_min and m["return"] >= r_min) else ""
                            rows.append({"Стратегія": label + mark,
                                         "Доходність": f"{m['return']*100:.1f}%",
                                         "Ризик": f"{m['risk']*100:.1f}%",
                                         "Sharpe": f"{m['sharpe']:.3f}"})
                        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                        w_ms, _ = res["max_sharpe"]
                        wdf = pd.DataFrame({
                            "Тікер":    available,
                            "Сектор":   [f"{SECTOR_EMOJI.get(sector_info.get(t,('',''))[0],'')}" for t in available],
                            "Вага (%)": (w_ms * 100).round(1),
                            "Min":  [f"{lb_arr[i]*100:.0f}%" if lb_arr[i]>0 else "—" for i in range(len(available))],
                            "Max":  [f"{ub_arr[i]*100:.0f}%" if ub_arr[i]<1 else "—" for i in range(len(available))],
                            "Статус": ["📌" if abs(w_ms[i]-lb_arr[i])<0.005 and lb_arr[i]>0
                                       else "🔒" if abs(w_ms[i]-ub_arr[i])<0.005 and ub_arr[i]<1
                                       else "✅" for i in range(len(available))]
                        }).sort_values("Вага (%)", ascending=False)
                        st.markdown("**Ваги Max-Sharpe:**")
                        st.dataframe(wdf.style.background_gradient(subset=["Вага (%)"], cmap="Blues"),
                                     hide_index=True, use_container_width=True)
                        st.bar_chart(wdf.set_index("Тікер")["Вага (%)"], use_container_width=True)
                        if r_min:
                            ms_ret = res["max_sharpe"][1]["return"]
                            st.caption(f"Мінімальна вимога: {r_min*100:.1f}% | "
                                       f"Max-Sharpe досягає: {ms_ret*100:.1f}%")
                    except Exception as e:
                        st.error(f"Помилка оптимізації: {e}")

    # ══ TAB 3 ════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("VaR та CVaR портфеля (95%)")
        var_strat = st.selectbox("Ваги:", ["Equal-Weight", "Ваші ваги"])
        w_var = (user_w if var_strat != "Equal-Weight" and total_uw > 0
                 else np.ones(len(available)) / len(available))
        var_res = compute_var(returns, w_var)
        cols4   = st.columns(len(var_res))
        for i, (method, (var, cvar)) in enumerate(var_res.items()):
            with cols4[i]:
                st.metric(method, f"VaR: {var*100:.2f}%",
                          f"CVaR: {cvar*100:.2f}%", delta_color="inverse")
        st.markdown("""
        ---
        - **Hist** — рівноважні ваги всіх спостережень
        - **BRW EWMA** — більша вага свіжим даням (~32 дні "пам'яті")
        - **Parametric** — нормальний розподіл з вибірковою σ
        """)

    # ══ TAB 4 ════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("🤖 Стан ML-моделей")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Per-ticker (19 ознак)**\n"
                        "- Окрема модель для кожної акції\n"
                        "- RMSE ~0.015 ✅")
        with col_b:
            n_feat = sp500_meta.get("n_features", 26)
            n_inst = sp500_meta.get("n_instruments", "?")
            rmse   = sp500_meta.get("metrics", {}).get("rmse", "?")
            st.markdown(f"**S&P 500 ({n_feat} ознак)**\n"
                        f"- {n_inst} інстр., RMSE {rmse}\n"
                        f"- 19 тех. + 6 дин.сект. + 1 ETF\n"
                        f"- {'✅' if sp500_exists else '❌'}")

        if not sp500_exists:
            st.warning("```bash\npython3 train_sp500.py\n```")

        st.markdown("---")
        rows = []
        for t in available:
            path = os.path.join(MODELS_DIR, f"xgb_{t}_v2.json")
            sector, _ = sector_info.get(t, ("Technology", False))
            source = "📚 вбудований" if t in KNOWN_SECTORS else "💾 кеш/API"
            if os.path.exists(path):
                mtime = datetime.fromtimestamp(os.path.getmtime(path))
                size  = os.path.getsize(path) / 1024
                rows.append({"Тікер": t,
                             "Сектор": f"{SECTOR_EMOJI.get(sector,'')} {sector}",
                             "Джерело": source,
                             "Статус": "✅", "Навчена": mtime.strftime("%Y-%m-%d %H:%M"),
                             "Розмір": f"{size:.0f} KB",
                             "σ̂ (%)": f"{vol_forecasts.get(t,0)*np.sqrt(252)*100:.1f}"})
            else:
                rows.append({"Тікер": t,
                             "Сектор": f"{SECTOR_EMOJI.get(sector,'')} {sector}",
                             "Джерело": source,
                             "Статус": "⚠️", "Навчена": "—", "Розмір": "—", "σ̂ (%)": "—"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.markdown("---")
        rv_chart = (returns.pow(2).rolling(20).sum().apply(np.sqrt)
                    * np.sqrt(252) * 100).dropna()
        st.line_chart(rv_chart, use_container_width=True)

    st.markdown("---")
    st.caption(
        f"📅 {date.today()} · Тікерів: {len(available)} · "
        f"Режим: {'S&P 500' if use_sp500 else 'Per-ticker'}"
    )


if __name__ == "__main__":
    main()
