# =============================================================================
# models/risk/monte_carlo.py — Монте-Карло симуляція портфеля (GBM + Cholesky)
# MILESTONE 3.2
# =============================================================================

import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config import N_MONTE_CARLO, CONFIDENCE_LEVEL


def simulate_gbm(
    S0: np.ndarray,
    mu: np.ndarray,
    sigma_forecast: np.ndarray,
    corr_matrix: np.ndarray,
    T: int = 252,
    n_sim: int = N_MONTE_CARLO,
    seed: int = 42,
) -> np.ndarray:
    """
    Симулює траєкторії цін активів за геометричним броунівським рухом (GBM).

    Модель:
        S_{t+1} = S_t · exp((μ - σ²/2)·dt + σ·√dt·ε_t)
        ε_t ~ N(0, Corr)  — корельовані шоки (Cholesky)

    Cholesky декомпозиція гарантує що симульовані шоки мають
    задану кореляційну структуру:
        L = chol(Corr),  ε_corr = L · z,  z ~ N(0, I)

    Args:
        S0:             початкові ціни (n_assets,)
        mu:             очікувані денні доходності (n_assets,)
        sigma_forecast: ML-прогнозована денна волатильність (n_assets,)
        corr_matrix:    матриця кореляцій (n_assets × n_assets)
        T:              горизонт симуляції у днях (default 252 = 1 рік)
        n_sim:          кількість симульованих траєкторій (default 10 000)
        seed:           seed для відтворюваності

    Returns:
        np.ndarray: траєкторії цін (n_sim, T+1, n_assets)
                    [:, 0, :] = S0 (початкові ціни)
    """
    np.random.seed(seed)
    n_assets = len(S0)
    dt = 1.0          # денний крок

    # Cholesky декомпозиція кореляційної матриці
    # Якщо матриця не є точно позитивно-визначеною — регуляризуємо
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # Додаємо невелику діагональну регуляризацію
        reg = corr_matrix + np.eye(n_assets) * 1e-6
        L   = np.linalg.cholesky(reg)

    # Ініціалізуємо масив траєкторій
    paths = np.zeros((n_sim, T + 1, n_assets))
    paths[:, 0, :] = S0

    # Drift член: (μ - σ²/2) · dt
    drift = (mu - 0.5 * sigma_forecast ** 2) * dt

    for t in range(1, T + 1):
        # Незалежні стандартні нормальні шоки (n_sim × n_assets)
        z = np.random.standard_normal((n_sim, n_assets))

        # Корельовані шоки: ε = z @ L^T
        eps_corr = z @ L.T

        # GBM крок
        paths[:, t, :] = paths[:, t-1, :] * np.exp(
            drift + sigma_forecast * np.sqrt(dt) * eps_corr
        )

    return paths


def mc_portfolio_paths(
    paths: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Агрегує траєкторії активів у траєкторії портфеля.

    Args:
        paths:   (n_sim, T+1, n_assets) — ціни активів
        weights: (n_assets,) — ваги портфеля

    Returns:
        np.ndarray: (n_sim, T+1) — нормована вартість портфеля (починається з 1.0)
    """
    # Зважена сума нормованих цін
    norm_paths = paths / paths[:, 0:1, :]           # нормуємо на початкову ціну
    portfolio  = norm_paths @ weights                # (n_sim, T+1)
    return portfolio


def mc_var_cvar(
    paths: np.ndarray,
    weights: np.ndarray,
    alpha: float = CONFIDENCE_LEVEL,
    horizon: int = 1,
) -> tuple[float, float]:
    """
    Розраховує VaR і CVaR портфеля з симульованих траєкторій.

    Args:
        paths:   (n_sim, T+1, n_assets) — симульовані ціни
        weights: (n_assets,) — ваги портфеля
        alpha:   рівень довіри
        horizon: горизонт (кількість кроків від початку)

    Returns:
        tuple: (VaR, CVaR)
    """
    portfolio = mc_portfolio_paths(paths, weights)

    # Доходності портфеля за горизонт horizon
    returns_h = portfolio[:, horizon] / portfolio[:, 0] - 1

    var  = float(-np.quantile(returns_h, 1 - alpha))
    tail = returns_h[returns_h <= -var]
    cvar = float(-np.mean(tail)) if len(tail) > 0 else var

    return var, cvar


def mc_summary(
    paths: np.ndarray,
    weights: np.ndarray,
    alpha: float = CONFIDENCE_LEVEL,
) -> dict:
    """
    Повна статистика Monte Carlo симуляції портфеля.

    Args:
        paths:   (n_sim, T+1, n_assets)
        weights: (n_assets,)
        alpha:   рівень довіри

    Returns:
        dict: VaR, CVaR, очікувана доходність, перцентилі для fan chart
    """
    portfolio = mc_portfolio_paths(paths, weights)
    final     = portfolio[:, -1]                     # фінальна вартість
    returns_1 = portfolio[:, 1] / portfolio[:, 0] - 1  # 1-денна доходність

    var_1d, cvar_1d = mc_var_cvar(paths, weights, alpha, horizon=1)

    result = {
        "var_1d":        var_1d,
        "cvar_1d":       cvar_1d,
        "mean_return":   float(np.mean(final) - 1),
        "std_return":    float(np.std(final)),
        "prob_loss":     float(np.mean(final < 1.0)),
        # Перцентилі траєкторій для fan chart (n_timesteps,)
        "p05":  np.percentile(portfolio, 5,  axis=0),
        "p25":  np.percentile(portfolio, 25, axis=0),
        "p50":  np.percentile(portfolio, 50, axis=0),
        "p75":  np.percentile(portfolio, 75, axis=0),
        "p95":  np.percentile(portfolio, 95, axis=0),
    }

    print(f"\n[MonteCarlo] n_sim={paths.shape[0]}, T={paths.shape[1]-1} днів")
    print(f"  1-day VaR (95%):  {result['var_1d']:.4f}")
    print(f"  1-day CVaR (95%): {result['cvar_1d']:.4f}")
    print(f"  Очік. дохідність: {result['mean_return']:+.4f}")
    print(f"  P(збиток):        {result['prob_loss']:.3f}")

    return result


# ---------------------------------------------------------------------------
# Тест: python models/risk/monte_carlo.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data.loader import get_prices
    from data.preprocessor import compute_log_returns

    prices  = get_prices()
    returns = compute_log_returns(prices)

    # Параметри симуляції
    n       = returns.shape[1]
    S0      = prices.iloc[-1].values
    mu      = returns.mean().values
    sigma   = returns.std().values
    corr    = returns.corr().values
    weights = np.ones(n) / n             # рівні ваги

    print("=== Монте-Карло: 10 000 симуляцій, горизонт 252 дні ===")
    paths = simulate_gbm(S0, mu, sigma, corr, T=252, n_sim=10_000)
    mc_summary(paths, weights)
