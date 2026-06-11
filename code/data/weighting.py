# =============================================================================
# data/weighting.py — Часові ваги для оцінки ризику (EWMA)
# =============================================================================
# Концепція: не всі спостереження рівноцінні.
# Дані двомісячної давності важливіші за дані 2018 року.
#
# Метод EWMA (Exponentially Weighted Moving Average):
#   w_t = (1 - λ) · λ^(T-t)
#   λ = 0.94 — стандарт RiskMetrics (JP Morgan, 1994)
#
# При λ=0.94:
#   - Останній день має вагу ~6%
#   - Через 10 днів: ~3.3%
#   - Через 50 днів: ~1.8%
#   - Через 250 днів (1 рік): ~0.2%
#   → Ефективний "горизонт пам'яті": ~75 торгових днів
# =============================================================================

import numpy as np
import pandas as pd


def ewma_weights(T: int, lambda_: float = 0.94) -> np.ndarray:
    """
    Генерує вектор EWMA-ваг для T спостережень.

    w_t = (1 - λ) · λ^(T-1-t),  t = 0..T-1  (від старих до нових)
    Ваги нормуються щоб sum = 1.

    Args:
        T:       кількість спостережень
        lambda_: коефіцієнт затухання (0.94 — RiskMetrics денний)
                 0.97 — RiskMetrics місячний
                 0.99 — "повільне забування"

    Returns:
        np.ndarray: ваги (T,), w[0] = найстаріше, w[-1] = найновіше
    """
    t = np.arange(T)
    # Чим старіше спостереження, тим більше T-1-t, тим менша вага
    raw = (1 - lambda_) * lambda_ ** (T - 1 - t)
    return raw / raw.sum()


def effective_sample_size(lambda_: float) -> float:
    """
    Ефективний розмір вибірки (скільки рівноважних спостережень
    еквівалентно нескінченній EWMA вибірці).

    ESS = (1 + λ) / (1 - λ)
    При λ=0.94: ESS ≈ 32 дні
    """
    return (1 + lambda_) / (1 - lambda_)


def ewma_cov_matrix(
    returns: pd.DataFrame,
    lambda_: float = 0.94,
    annualize: bool = True,
) -> np.ndarray:
    """
    EWMA-коваріаційна матриця (метод RiskMetrics).

    Замість рівноважного середнього:  Σ = (X-μ)^T (X-μ) / (T-1)
    Використовуємо зважене:            Σ_EWMA = Σ_t w_t · (r_t - μ_w)(r_t - μ_w)^T

    Args:
        returns:   DataFrame лог-доходностей (рядки = дати, стовпці = тикери)
        lambda_:   коефіцієнт затухання
        annualize: множити на 252 для річної матриці

    Returns:
        np.ndarray: EWMA ковар. матриця (n × n)
    """
    X = returns.values.astype(np.float64)
    T, n = X.shape

    w = ewma_weights(T, lambda_)  # (T,)

    # Зважене середнє
    mu_w = (w[:, None] * X).sum(axis=0)  # (n,)

    # Зважена коваріаційна матриця
    X_c = X - mu_w  # центрування
    cov = (X_c * w[:, None]).T @ X_c  # (n × n)

    # Гарантуємо симетричність (числова точність)
    cov = (cov + cov.T) / 2

    if annualize:
        cov *= 252

    return cov


def ewma_historical_var(
    returns: np.ndarray,
    lambda_: float = 0.94,
    alpha:   float = 0.95,
) -> tuple[float, float]:
    """
    Зважений Historical Simulation VaR/CVaR (метод BRW — Boudoukh, Richardson, Whitelaw 1998).

    Замість рівноважних перцентилів, ваги розподіляються EWMA-способом.
    Ефект: кризи давнього минулого мають менший вплив на VaR.

    Алгоритм:
        1. Розрахувати EWMA-ваги для всіх T спостережень
        2. Відсортувати доходності від найгірших до найкращих
        3. Накопичити ваги — VaR = рівень де накопичена вага = (1-α)

    Args:
        returns: 1D масив лог-доходностей
        lambda_: коефіцієнт затухання
        alpha:   рівень довіри

    Returns:
        tuple: (VaR, CVaR) — обидва додатні (= втрати)
    """
    T = len(returns)
    w = ewma_weights(T, lambda_)

    # Сортуємо доходності від найменшої (найгірша) до найбільшої
    sort_idx = np.argsort(returns)
    sorted_ret = returns[sort_idx]
    sorted_w   = w[sort_idx]

    # Накопичені ваги
    cum_w = np.cumsum(sorted_w)

    # VaR: знаходимо де кумулятивна вага переходить поріг (1-alpha)
    threshold = 1 - alpha
    var_idx = np.searchsorted(cum_w, threshold)
    var_idx = min(var_idx, T - 1)
    var = float(-sorted_ret[var_idx])

    # CVaR: зважене середнє хвоста
    tail_mask = cum_w <= threshold
    if tail_mask.sum() == 0:
        cvar = var
    else:
        tail_ret = sorted_ret[tail_mask]
        tail_w   = sorted_w[tail_mask]
        cvar = float(-np.average(tail_ret, weights=tail_w))

    return var, cvar


def compare_ewma_vs_sample(
    returns: pd.DataFrame,
    lambda_: float = 0.94,
) -> dict:
    """
    Порівнює EWMA і вибіркову коваріаційні матриці.
    Виводить ключові відмінності.

    Args:
        returns: DataFrame лог-доходностей
        lambda_: коефіцієнт EWMA

    Returns:
        dict: {'sample_cov', 'ewma_cov', 'sample_vols', 'ewma_vols'}
    """
    sample_cov = returns.cov().values * 252
    ewma_cov   = ewma_cov_matrix(returns, lambda_=lambda_)

    sample_vols = np.sqrt(np.diag(sample_cov))
    ewma_vols   = np.sqrt(np.diag(ewma_cov))

    print(f"\n{'Тикер':<8} {'Sample σ':>10} {'EWMA σ':>10} {'Різниця':>10}")
    print("-" * 42)
    for i, col in enumerate(returns.columns):
        diff = ewma_vols[i] - sample_vols[i]
        sign = "↑" if diff > 0 else "↓"
        print(f"{col:<8} {sample_vols[i]:>10.4f} {ewma_vols[i]:>10.4f} "
              f"{sign}{abs(diff):>8.4f}")

    ess = effective_sample_size(lambda_)
    print(f"\nESS при λ={lambda_}: ~{ess:.0f} ефективних спостережень")
    print(f"(еквівалентно ~{ess/21:.1f} місяцям торгових даних)")

    return {
        "sample_cov": sample_cov,
        "ewma_cov":   ewma_cov,
        "sample_vols": sample_vols,
        "ewma_vols":   ewma_vols,
    }


# ---------------------------------------------------------------------------
# Тест: python data/weighting.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from data.loader import get_prices
    from data.preprocessor import compute_log_returns
    from models.risk.var_cvar import historical_var, historical_cvar

    prices  = get_prices()
    returns = compute_log_returns(prices)

    print("=" * 50)
    print("EWMA vs Sample — порівняння волатильностей")
    print("=" * 50)
    result = compare_ewma_vs_sample(returns, lambda_=0.94)

    print("\n" + "=" * 50)
    print("Зважений VaR (BRW) vs Класичний VaR — AAPL")
    print("=" * 50)
    r = returns["AAPL"].values

    var_classic = historical_var(r, alpha=0.95)
    cvar_classic = historical_cvar(r, alpha=0.95)
    var_ewma, cvar_ewma = ewma_historical_var(r, lambda_=0.94, alpha=0.95)

    print(f"\n{'Метод':<25} {'VaR':>8} {'CVaR':>8}")
    print("-" * 45)
    print(f"{'Historical (рівні ваги)':<25} {var_classic:>8.4f} {cvar_classic:>8.4f}")
    print(f"{'BRW EWMA (λ=0.94)':<25} {var_ewma:>8.4f} {cvar_ewma:>8.4f}")
    print(f"\nРізниця VaR:  {var_ewma - var_classic:+.4f}")
    print(f"Різниця CVaR: {cvar_ewma - cvar_classic:+.4f}")

    # Перевірка при різних λ
    print("\n" + "=" * 50)
    print("Чутливість VaR до вибору λ")
    print("=" * 50)
    print(f"\n{'λ':<8} {'VaR':>8} {'CVaR':>8} {'ESS':>8}")
    print("-" * 36)
    for lam in [0.90, 0.94, 0.97, 0.99]:
        v, cv = ewma_historical_var(r, lambda_=lam, alpha=0.95)
        ess = effective_sample_size(lam)
        print(f"{lam:<8.2f} {v:>8.4f} {cv:>8.4f} {ess:>8.0f}")
