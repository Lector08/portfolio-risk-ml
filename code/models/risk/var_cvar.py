# =============================================================================
# models/risk/var_cvar.py — VaR та CVaR: класичні + EWMA методи
# MILESTONE 3.1 (оновлено: EWMA Historical Simulation)
# =============================================================================

import numpy as np
from scipy import stats
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config import CONFIDENCE_LEVEL


# ---------------------------------------------------------------------------
# 1. Historical Simulation (рівні ваги)
# ---------------------------------------------------------------------------

def historical_var(returns: np.ndarray, alpha: float = CONFIDENCE_LEVEL) -> float:
    """VaR = -quantile(returns, 1-alpha)"""
    return float(-np.quantile(returns, 1 - alpha))


def historical_cvar(returns: np.ndarray, alpha: float = CONFIDENCE_LEVEL) -> float:
    """CVaR = -mean(returns[returns ≤ -VaR])"""
    var  = historical_var(returns, alpha)
    tail = returns[returns <= -var]
    return float(-np.mean(tail)) if len(tail) > 0 else var


# ---------------------------------------------------------------------------
# 2. EWMA Historical Simulation (часові ваги, метод BRW 1998)
# ---------------------------------------------------------------------------

def ewma_historical_var(
    returns: np.ndarray,
    alpha:   float = CONFIDENCE_LEVEL,
    lambda_: float = 0.94,
) -> tuple[float, float]:
    """
    Зважений Historical Simulation VaR/CVaR.

    Метод BRW (Boudoukh, Richardson, Whitelaw, 1998):
    Більш свіжі спостереження отримують вищу вагу.
    λ=0.94 — стандарт RiskMetrics.

    Args:
        returns: 1D масив лог-доходностей
        alpha:   рівень довіри
        lambda_: коефіцієнт затухання

    Returns:
        tuple: (VaR, CVaR)
    """
    from data.weighting import ewma_historical_var as _ewma_var
    return _ewma_var(returns, lambda_=lambda_, alpha=alpha)


# ---------------------------------------------------------------------------
# 3. Параметричний метод (нормальний розподіл)
# ---------------------------------------------------------------------------

def parametric_var(mu: float, sigma: float, alpha: float = CONFIDENCE_LEVEL) -> float:
    """Parametric VaR: -(μ + σ · z_α)"""
    z_alpha = stats.norm.ppf(1 - alpha)
    return float(-(mu + sigma * z_alpha))


def parametric_cvar(mu: float, sigma: float, alpha: float = CONFIDENCE_LEVEL) -> float:
    """Parametric CVaR: -(μ - σ · φ(z_α) / (1-α))"""
    z_alpha = stats.norm.ppf(1 - alpha)
    phi_z   = stats.norm.pdf(z_alpha)
    return float(-(mu - sigma * phi_z / (1 - alpha)))


# ---------------------------------------------------------------------------
# 4. VaR портфеля
# ---------------------------------------------------------------------------

def portfolio_var_cvar(
    weights:    np.ndarray,
    cov_matrix: np.ndarray,
    mu_vector:  np.ndarray = None,
    alpha:      float = CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """Portfolio VaR/CVaR через ковар. матрицю."""
    if mu_vector is None:
        mu_vector = np.zeros(len(weights))
    mu_p    = float(weights @ mu_vector)
    sigma_p = float(np.sqrt(max(weights @ cov_matrix @ weights, 0)))
    return parametric_var(mu_p, sigma_p, alpha), parametric_cvar(mu_p, sigma_p, alpha)


# ---------------------------------------------------------------------------
# 5. Зведена таблиця: всі методи
# ---------------------------------------------------------------------------

def compute_all_var(
    returns:  np.ndarray,
    sigma_ml: float = None,
    alpha:    float = CONFIDENCE_LEVEL,
    lambda_:  float = 0.94,
) -> dict:
    """
    Розраховує VaR і CVaR чотирма методами:
        1. Historical Simulation (рівні ваги)
        2. BRW EWMA Historical Simulation (λ=0.94)
        3. Parametric (вибіркова σ)
        4. Parametric (ML σ̂)  — якщо sigma_ml задано
    """
    mu    = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))

    var_ewma, cvar_ewma = ewma_historical_var(returns, alpha, lambda_)

    results = {
        "historical": {
            "var":   historical_var(returns, alpha),
            "cvar":  historical_cvar(returns, alpha),
            "label": "Historical (рівні ваги)",
        },
        "ewma_historical": {
            "var":   var_ewma,
            "cvar":  cvar_ewma,
            "label": f"BRW EWMA (λ={lambda_})",
        },
        "parametric_sample": {
            "var":   parametric_var(mu, sigma, alpha),
            "cvar":  parametric_cvar(mu, sigma, alpha),
            "label": "Parametric (sample σ)",
        },
    }

    if sigma_ml is not None:
        results["parametric_ml"] = {
            "var":   parametric_var(mu, sigma_ml, alpha),
            "cvar":  parametric_cvar(mu, sigma_ml, alpha),
            "label": "Parametric (ML σ̂)",
        }

    print(f"\n{'Метод':<28} {'VaR':>8} {'CVaR':>8}")
    print("-" * 48)
    for m in results.values():
        print(f"{m['label']:<28} {m['var']:>8.4f} {m['cvar']:>8.4f}")

    return results
