# =============================================================================
# backtesting/metrics.py — Метрики якості портфеля та ML-моделей
# MILESTONE 5.2
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Метрики портфеля
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.04) -> float:
    """
    Коефіцієнт Шарпа: відношення надлишкової доходності до волатильності.

    Формула: S = (μ_annual - r_f) / σ_annual
    Ануалізація: μ_annual = mean(r) * 252, σ_annual = std(r) * sqrt(252)

    Args:
        returns:        1D масив денних доходностей портфеля
        risk_free_rate: річна безризикова ставка

    Returns:
        float: коефіцієнт Шарпа (річний)
    """
    if len(returns) == 0:
        return 0.0
    mu_daily  = np.mean(returns)
    std_daily = np.std(returns, ddof=1)
    if std_daily < 1e-10:
        return 0.0
    return float((mu_daily * 252 - risk_free_rate) / (std_daily * np.sqrt(252)))


def max_drawdown(nav: np.ndarray) -> float:
    """
    Максимальна просадка (peak-to-trough drawdown).

    MDD = min_t { (NAV_t - max_{s<=t} NAV_s) / max_{s<=t} NAV_s }

    Args:
        nav: 1D масив значень Net Asset Value

    Returns:
        float: максимальна просадка (від'ємне число, наприклад -0.35 = -35%)
    """
    if len(nav) == 0:
        return 0.0
    peak    = np.maximum.accumulate(nav)
    drawdown = (nav - peak) / peak
    return float(np.min(drawdown))


def calmar_ratio(returns: np.ndarray, nav: np.ndarray) -> float:
    """
    Коефіцієнт Калмара: CAGR / |MaxDrawdown|.

    Показує, скільки одиниць річної доходності припадає на одиницю
    максимальної просадки. Чим вищий — тим краща стратегія.

    Args:
        returns: денні доходності
        nav:     NAV портфеля

    Returns:
        float: коефіцієнт Калмара
    """
    mdd = max_drawdown(nav)
    if abs(mdd) < 1e-10:
        return 0.0
    cagr = float(np.mean(returns) * 252)
    return cagr / abs(mdd)


def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.04) -> float:
    """
    Коефіцієнт Сортіно: враховує лише негативну волатильність (downside risk).

    На відміну від Шарпа, не штрафує за позитивні відхилення.

    Формула: Sortino = (μ_annual - r_f) / σ_downside_annual
    де σ_downside = std(r[r < 0]) * sqrt(252)

    Args:
        returns:        денні доходності
        risk_free_rate: річна безризикова ставка

    Returns:
        float: коефіцієнт Сортіно
    """
    if len(returns) == 0:
        return 0.0
    mu_annual  = float(np.mean(returns) * 252)
    downside   = returns[returns < 0]
    if len(downside) == 0:
        return np.inf
    sigma_down = float(np.std(downside, ddof=1) * np.sqrt(252))
    if sigma_down < 1e-10:
        return 0.0
    return (mu_annual - risk_free_rate) / sigma_down


def cagr(nav: np.ndarray, n_days: int = None) -> float:
    """
    Compound Annual Growth Rate (CAGR).

    CAGR = (NAV_T / NAV_0)^(252/T) - 1

    Args:
        nav:    NAV портфеля
        n_days: кількість торгових днів; якщо None — len(nav)

    Returns:
        float: CAGR (річна)
    """
    if len(nav) < 2:
        return 0.0
    T = n_days or len(nav)
    return float((nav[-1] / nav[0]) ** (252 / T) - 1)


# ---------------------------------------------------------------------------
# Метрики ML-моделей (волатильність)
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return float(np.mean(np.abs(y_true - y_pred)))


def qlike_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    QLIKE loss — стандартна метрика для прогнозу волатильності.

    QLIKE = mean( log(σ̂²) + σ²/σ̂² )

    Асиметрична: штрафує більше за недооцінку волатильності.
    """
    eps = 1e-8
    return float(np.mean(np.log(y_pred**2 + eps) + y_true**2 / (y_pred**2 + eps)))


# ---------------------------------------------------------------------------
# Статистичні тести VaR (Backtesting VaR)
# ---------------------------------------------------------------------------

def var_exceedance_rate(returns: np.ndarray, var_estimates: np.ndarray) -> float:
    """
    Частота перевищень VaR (має бути ≈ 1-alpha).

    Для VaR_95% очікується 5% перевищень.
    Якщо перевищень суттєво більше — модель недооцінює ризик.

    Args:
        returns:       реалізовані денні доходності (від'ємні = збитки)
        var_estimates: VaR-оцінки на кожен день (від'ємні числа)

    Returns:
        float: частка днів де збиток перевищив VaR
    """
    exceedances = returns < -np.abs(var_estimates)
    return float(np.mean(exceedances))


def kupiec_test(
    returns: np.ndarray,
    var_estimates: np.ndarray,
    alpha: float = 0.95,
) -> dict:
    """
    Тест Купєця (Proportion of Failures, POF test) для верифікації VaR-моделі.

    Нульова гіпотеза H₀: частка перевищень VaR дорівнює (1-α).
    Статистика: LR_POF = -2 ln(L(p₀)/L(p̂))
    де p₀ = 1-α (теоретична частка), p̂ — спостережена частка.

    При H₀ LR_POF ~ χ²(1), критичне значення при 5% = 3.84.

    Args:
        returns:       реалізовані денні доходності
        var_estimates: VaR-оцінки (від'ємні)
        alpha:         рівень довіри VaR (0.95)

    Returns:
        dict: {lr_stat, p_value, exceedances, expected, reject_h0}
    """
    T = len(returns)
    p0 = 1 - alpha   # очікувана частка перевищень

    exceedances = (returns < -np.abs(var_estimates)).sum()
    p_hat = exceedances / T   # спостережена частка

    # Захист від крайніх значень
    if p_hat <= 0:
        p_hat = 0.5 / T
    if p_hat >= 1:
        p_hat = 1 - 0.5 / T

    # LR-статистика
    lr_stat = -2 * (
        exceedances * np.log(p0 / p_hat)
        + (T - exceedances) * np.log((1 - p0) / (1 - p_hat))
    )

    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    result = {
        "lr_stat":      float(lr_stat),
        "p_value":      float(p_value),
        "exceedances":  int(exceedances),
        "expected":     float(T * p0),
        "exceedance_rate": float(p_hat),
        "reject_h0":    bool(lr_stat > stats.chi2.ppf(0.95, df=1)),
    }

    verdict = "ВІДХИЛЯЄМО H₀ (модель некоректна)" if result["reject_h0"] \
              else "Не відхиляємо H₀ (модель коректна)"
    print(f"[KupiecTest] LR={lr_stat:.4f}, p={p_value:.4f} | {verdict}")
    print(f"  Перевищень: {exceedances}/{T} спостереж. ({p_hat:.3f} vs {p0:.3f} очік.)")

    return result


def comprehensive_report(
    strategy_name: str,
    nav: np.ndarray,
    returns: np.ndarray,
    var_estimates: np.ndarray = None,
    alpha: float = 0.95,
    risk_free_rate: float = 0.04,
) -> dict:
    """
    Повний звіт метрик для однієї стратегії.

    Args:
        strategy_name:  назва стратегії
        nav:            NAV портфеля (починається з 1.0)
        returns:        денні доходності
        var_estimates:  VaR-оцінки (опційно, для тесту Купєця)
        alpha:          рівень довіри VaR
        risk_free_rate: безризикова ставка

    Returns:
        dict: всі метрики
    """
    report = {
        "strategy":    strategy_name,
        "cagr":        cagr(nav),
        "sharpe":      sharpe_ratio(returns, risk_free_rate),
        "sortino":     sortino_ratio(returns, risk_free_rate),
        "max_drawdown": max_drawdown(nav),
        "calmar":      calmar_ratio(returns, nav),
        "total_return": float(nav[-1] / nav[0] - 1),
        "volatility":  float(np.std(returns, ddof=1) * np.sqrt(252)),
    }

    if var_estimates is not None:
        kupiec = kupiec_test(returns, var_estimates, alpha)
        report["var_exceedance_rate"] = kupiec["exceedance_rate"]
        report["kupiec_lr"]           = kupiec["lr_stat"]
        report["kupiec_p"]            = kupiec["p_value"]
        report["kupiec_reject"]       = kupiec["reject_h0"]

    return report
