# =============================================================================
# visualization/plots.py — Всі графіки для звіту та аналізу
# MILESTONE 6.1
# =============================================================================
# Графіки зберігаються у docs/figures/ у форматі PNG (300 dpi)
#
# fig01_price_history.png        — динаміка нормованих цін
# fig02_returns_distribution.png — розподіл доходностей + Q-Q plot
# fig03_correlation_heatmap.png  — матриця кореляцій
# fig04_vol_forecast.png         — прогноз волатильності (LSTM vs XGB vs GARCH)
# fig05_efficient_frontier.png   — ефективна межа Марковіца
# fig06_mc_fan_chart.png         — Монте-Карло fan chart
# fig07_var_backtest.png         — реалізований VaR vs фактичні збитки
# fig08_nav_comparison.png       — NAV трьох стратегій
# fig09_feature_importance.png   — XGBoost feature importance
# fig10_drawdown.png             — drawdown chart
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # без GUI — для headless середовища
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

# Шлях для збереження графіків
FIGURES_PATH = os.path.join(
    os.path.dirname(__file__), "../../docs/figures"
)
os.makedirs(FIGURES_PATH, exist_ok=True)

# Єдиний стиль для всіх графіків
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "legend.fontsize":   10,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f9fa",
})

COLORS = {
    "ml":      "#2196F3",   # синій — ML-Markowitz
    "classic": "#FF9800",   # помаранчевий — Classic-Markowitz
    "ew":      "#4CAF50",   # зелений — Equal-Weight
    "var":     "#F44336",   # червоний — VaR / ризик
    "accent":  "#9C27B0",   # фіолетовий — акцент
}


def _save(fig, filename: str, tight: bool = True):
    """Зберігає фігуру у docs/figures/"""
    path = os.path.join(FIGURES_PATH, filename)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plots] Збережено → {path}")
    return path


# ---------------------------------------------------------------------------
# Fig 01 — Динаміка нормованих цін
# ---------------------------------------------------------------------------

def plot_price_history(prices: pd.DataFrame, filename="fig01_price_history.png"):
    """
    Нормовані ціни закриття (база 100 = перший торговий день).
    Дозволяє порівнювати відносну динаміку активів різного масштабу.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    normalized = prices / prices.iloc[0] * 100

    palette = plt.cm.tab10(np.linspace(0, 1, len(prices.columns)))
    for i, col in enumerate(prices.columns):
        ax.plot(normalized.index, normalized[col],
                label=col, linewidth=1.4, color=palette[i])

    ax.set_title("Динаміка нормованих цін акцій (база = 100)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Нормована ціна")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(ncol=4, loc="upper left")
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 02 — Розподіл доходностей + Q-Q plot
# ---------------------------------------------------------------------------

def plot_returns_distribution(
    returns: pd.DataFrame,
    ticker: str = "AAPL",
    filename="fig02_returns_distribution.png",
):
    """
    Гістограма лог-доходностей з накладеним нормальним розподілом
    та Q-Q plot для візуальної перевірки нормальності.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    r = returns[ticker].dropna()

    # --- Гістограма ---
    ax = axes[0]
    ax.hist(r, bins=60, density=True, color=COLORS["ml"],
            alpha=0.7, edgecolor="white", linewidth=0.3)

    x = np.linspace(r.min(), r.max(), 300)
    mu, sigma = r.mean(), r.std()
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color=COLORS["var"],
            linewidth=2, label=f"N({mu:.4f}, {sigma:.4f}²)")

    ax.set_title(f"Розподіл лог-доходностей ({ticker})")
    ax.set_xlabel("Доходність")
    ax.set_ylabel("Щільність")
    ax.legend()

    kurt = float(r.kurtosis())
    skew = float(r.skew())
    ax.text(0.98, 0.95, f"Skew = {skew:.3f}\nKurt = {kurt:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # --- Q-Q plot ---
    ax = axes[1]
    (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
    ax.scatter(osm, osr, color=COLORS["ml"], alpha=0.4, s=8)
    line_x = np.array([osm[0], osm[-1]])
    ax.plot(line_x, slope * line_x + intercept,
            color=COLORS["var"], linewidth=2, label="Нормальний розподіл")
    ax.set_title(f"Q-Q Plot ({ticker})")
    ax.set_xlabel("Теоретичні квантилі")
    ax.set_ylabel("Емпіричні квантилі")
    ax.legend()

    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 03 — Матриця кореляцій
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    returns: pd.DataFrame,
    filename="fig03_correlation_heatmap.png",
):
    """Теплова карта матриці кореляцій лог-доходностей."""
    fig, ax = plt.subplots(figsize=(8, 6))

    corr = returns.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Матриця кореляцій доходностей (2018–2024)")

    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 04 — Прогноз волатильності: моделі vs реалізована
# ---------------------------------------------------------------------------

def plot_volatility_forecast(
    dates,
    realized: np.ndarray,
    lstm_pred: np.ndarray = None,
    xgb_pred: np.ndarray = None,
    garch_pred: np.ndarray = None,
    ensemble_pred: np.ndarray = None,
    ticker: str = "AAPL",
    filename="fig04_vol_forecast.png",
):
    """
    Порівняння прогнозів волатильності різних моделей
    з реалізованою волатильністю на тестовій вибірці.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.fill_between(dates, realized, alpha=0.15, color="gray", label="_nolegend_")
    ax.plot(dates, realized, color="black", linewidth=1.2,
            label="Реалізована RV", zorder=5)

    if lstm_pred is not None:
        ax.plot(dates, lstm_pred, color=COLORS["ml"],
                linewidth=1.4, linestyle="--", label="LSTM", alpha=0.85)
    if xgb_pred is not None:
        ax.plot(dates, xgb_pred, color=COLORS["classic"],
                linewidth=1.4, linestyle="-.", label="XGBoost", alpha=0.85)
    if garch_pred is not None:
        ax.plot(dates, garch_pred, color="#795548",
                linewidth=1.2, linestyle=":", label="GARCH(1,1)", alpha=0.75)
    if ensemble_pred is not None:
        ax.plot(dates, ensemble_pred, color=COLORS["accent"],
                linewidth=2.0, label="Ансамбль", zorder=4)

    ax.set_title(f"Прогноз реалізованої волатильності ({ticker}, тестова вибірка)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Волатильність (річна)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper right")

    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 05 — Ефективна межа Марковіца
# ---------------------------------------------------------------------------

def plot_efficient_frontier(
    frontier_risks: np.ndarray,
    frontier_returns: np.ndarray,
    max_sharpe_pt: tuple,
    min_var_pt: tuple,
    asset_risks: np.ndarray,
    asset_returns: np.ndarray,
    asset_names: list,
    risk_free_rate: float = 0.04,
    filename="fig05_efficient_frontier.png",
):
    """
    Ефективна межа з позначеними ключовими портфелями та окремими активами.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Ефективна межа
    ax.plot(frontier_risks, frontier_returns,
            color=COLORS["ml"], linewidth=2.5, label="Ефективна межа", zorder=4)

    # Capital Market Line (від rf до max Sharpe)
    if max_sharpe_pt:
        ms_risk, ms_ret = max_sharpe_pt
        cml_x = np.array([0, ms_risk * 1.5])
        cml_y = risk_free_rate + (ms_ret - risk_free_rate) / ms_risk * cml_x
        ax.plot(cml_x, cml_y, color=COLORS["var"],
                linestyle="--", linewidth=1.5,
                label="Capital Market Line", zorder=3)

    # Окремі активи
    palette = plt.cm.tab10(np.linspace(0, 1, len(asset_names)))
    for i, (r, mu, name) in enumerate(
        zip(asset_risks, asset_returns, asset_names)
    ):
        ax.scatter(r, mu, color=palette[i], s=80, zorder=6)
        ax.annotate(name, (r, mu), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)

    # Max-Sharpe портфель
    if max_sharpe_pt:
        ax.scatter(*max_sharpe_pt, color=COLORS["var"], s=150,
                   marker="*", zorder=7, label="Max Sharpe")
        ax.annotate("Max Sharpe", max_sharpe_pt,
                    textcoords="offset points", xytext=(6, 5), fontsize=9)

    # Min-Variance портфель
    if min_var_pt:
        ax.scatter(*min_var_pt, color=COLORS["accent"], s=120,
                   marker="D", zorder=7, label="Min Variance")
        ax.annotate("Min Var", min_var_pt,
                    textcoords="offset points", xytext=(6, 5), fontsize=9)

    ax.set_title("Ефективна межа Марковіца (річні параметри, 2018–2024)")
    ax.set_xlabel("Ризик (σ, річна)")
    ax.set_ylabel("Очікувана доходність (річна)")
    ax.legend(loc="lower right")

    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 06 — Монте-Карло fan chart
# ---------------------------------------------------------------------------

def plot_mc_fan_chart(
    p05: np.ndarray,
    p25: np.ndarray,
    p50: np.ndarray,
    p75: np.ndarray,
    p95: np.ndarray,
    var_level: float = None,
    filename="fig06_mc_fan_chart.png",
):
    """
    Fan chart Монте-Карло симуляції портфеля.
    Смуги відповідають перцентилям 5/25/75/95%.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    T = len(p50)
    days = np.arange(T)

    ax.fill_between(days, p05, p95, alpha=0.12, color=COLORS["ml"], label="5–95 перцентиль")
    ax.fill_between(days, p25, p75, alpha=0.25, color=COLORS["ml"], label="25–75 перцентиль")
    ax.plot(days, p50, color=COLORS["ml"], linewidth=2, label="Медіана (50%)")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    if var_level is not None:
        ax.axhline(1 - var_level, color=COLORS["var"], linewidth=1.5,
                   linestyle=":", label=f"1-day VaR 95% ({var_level:.3f})")

    ax.set_title(f"Монте-Карло: 10 000 симуляцій GBM, горизонт {T-1} днів")
    ax.set_xlabel("Торгових днів")
    ax.set_ylabel("Нормована вартість портфеля")
    ax.legend(loc="upper left")

    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 07 — VaR backtest (exceedances)
# ---------------------------------------------------------------------------

def plot_var_backtest(
    returns: np.ndarray,
    var_estimates: np.ndarray,
    dates,
    alpha: float = 0.95,
    filename="fig07_var_backtest.png",
):
    """
    Графік верифікації VaR: реалізовані збитки vs VaR-оцінка.
    Перевищення VaR позначені червоними крапками.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    exceedances = returns < -np.abs(var_estimates)

    ax.fill_between(dates, returns, 0,
                    where=(returns < 0), color="gray",
                    alpha=0.25, label="Збитки")
    ax.plot(dates, -np.abs(var_estimates), color=COLORS["var"],
            linewidth=1.5, label=f"VaR ({int(alpha*100)}%)", zorder=3)
    ax.scatter(
        np.array(dates)[exceedances],
        returns[exceedances],
        color=COLORS["var"], s=25, zorder=5,
        label=f"Перевищення ({exceedances.sum()} з {len(returns)})"
    )
    ax.axhline(0, color="black", linewidth=0.6)

    ax.set_title(f"Бектестинг VaR (рівень довіри {int(alpha*100)}%)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Доходність портфеля")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower right")

    rate = exceedances.mean()
    ax.text(0.02, 0.05, f"Частота перевищень: {rate:.3f} (очік. {1-alpha:.3f})",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 08 — Порівняння NAV стратегій
# ---------------------------------------------------------------------------

def plot_nav_comparison(
    nav_ml: pd.Series,
    nav_classic: pd.Series,
    nav_ew: pd.Series,
    filename="fig08_nav_comparison.png",
):
    """
    NAV трьох інвестиційних стратегій на спільній осі.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                              gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(nav_ml.index,      nav_ml.values,
            color=COLORS["ml"],      linewidth=2,   label="ML-Markowitz")
    ax.plot(nav_classic.index, nav_classic.values,
            color=COLORS["classic"], linewidth=1.8, label="Classic-Markowitz",
            linestyle="--")
    ax.plot(nav_ew.index,      nav_ew.values,
            color=COLORS["ew"],      linewidth=1.6, label="Equal-Weight",
            linestyle="-.")
    ax.axhline(1.0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_title("Динаміка NAV трьох стратегій портфеля (Walk-Forward бектестинг)")
    ax.set_ylabel("NAV (початок = 1.0)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left")

    # Нижня панель: drawdown ML-стратегії
    ax2 = axes[1]
    peak = np.maximum.accumulate(nav_ml.values)
    dd   = (nav_ml.values - peak) / peak
    ax2.fill_between(nav_ml.index, dd, 0,
                     color=COLORS["ml"], alpha=0.4)
    ax2.plot(nav_ml.index, dd, color=COLORS["ml"], linewidth=1)
    ax2.set_ylabel("Просадка ML")
    ax2.set_xlabel("Дата")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 09 — Feature Importance (XGBoost)
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 20,
    filename="fig09_feature_importance.png",
):
    """
    Горизонтальний bar chart топ-N найважливіших ознак XGBoost.
    """
    top = importance.head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top.index, top.values,
                   color=COLORS["ml"], alpha=0.85, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
    ax.set_title(f"XGBoost Feature Importance (Топ-{top_n})")
    ax.set_xlabel("Importance (Gain)")
    ax.set_xlim(0, top.values.max() * 1.15)

    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 10 — Drawdown chart усіх стратегій
# ---------------------------------------------------------------------------

def plot_drawdown(
    nav_ml: pd.Series,
    nav_classic: pd.Series,
    nav_ew: pd.Series,
    filename="fig10_drawdown.png",
):
    """Просадка (drawdown) трьох стратегій."""
    fig, ax = plt.subplots(figsize=(13, 5))

    for nav, color, name, ls in [
        (nav_ml,      COLORS["ml"],      "ML-Markowitz",      "-"),
        (nav_classic, COLORS["classic"], "Classic-Markowitz", "--"),
        (nav_ew,      COLORS["ew"],      "Equal-Weight",      "-."),
    ]:
        peak = np.maximum.accumulate(nav.values)
        dd   = (nav.values - peak) / peak
        ax.fill_between(nav.index, dd, 0, alpha=0.08, color=color)
        ax.plot(nav.index, dd, color=color, linewidth=1.5,
                linestyle=ls, label=name)

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Просадка портфелів (Drawdown Analysis)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, _: f"{y*100:.0f}%")
    )

    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Зручна функція: генерує всі графіки одним викликом
# ---------------------------------------------------------------------------

def generate_all_plots(backtest, optimizer=None, returns=None, prices=None):
    """
    Генерує всі 10 графіків для звіту за один виклик.

    Args:
        backtest:  завершений WalkForwardBacktest об'єкт
        optimizer: завершений MarkowitzOptimizer (для ефективної межі)
        returns:   DataFrame лог-доходностей
        prices:    DataFrame цін закриття

    Returns:
        list: шляхи до збережених файлів
    """
    paths = []

    nav_df = backtest.get_nav_dataframe()
    nav_ml      = nav_df.get("ml_markowitz",      nav_df.iloc[:, 0])
    nav_classic = nav_df.get("classic_markowitz", nav_df.iloc[:, 0])
    nav_ew      = nav_df.get("equal_weight",      nav_df.iloc[:, 0])

    if prices is not None:
        paths.append(plot_price_history(prices))

    if returns is not None:
        paths.append(plot_returns_distribution(returns))
        paths.append(plot_correlation_heatmap(returns))

    paths.append(plot_nav_comparison(nav_ml, nav_classic, nav_ew))
    paths.append(plot_drawdown(nav_ml, nav_classic, nav_ew))

    if optimizer is not None and optimizer.frontier_risks is not None:
        mu    = optimizer.mu
        Sigma = optimizer.Sigma
        asset_risks   = np.sqrt(np.diag(Sigma))
        asset_returns = mu
        paths.append(plot_efficient_frontier(
            optimizer.frontier_risks,
            optimizer.frontier_returns,
            (np.sqrt(optimizer.max_sharpe_weights @ Sigma @ optimizer.max_sharpe_weights),
             optimizer.mu @ optimizer.max_sharpe_weights)
            if optimizer.max_sharpe_weights is not None else None,
            (np.sqrt(optimizer.min_var_weights @ Sigma @ optimizer.min_var_weights),
             optimizer.mu @ optimizer.min_var_weights)
            if optimizer.min_var_weights is not None else None,
            asset_risks, asset_returns, optimizer.tickers,
        ))

    print(f"\n[plots] Всього збережено графіків: {len(paths)}")
    return paths
