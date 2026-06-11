# =============================================================================
# models/portfolio/ml_optimizer.py — ML + EWMA оптимізація портфеля
# MILESTONE 4.2 (оновлено: EWMA коваріаційна матриця)
# =============================================================================

import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config import RISK_FREE_RATE
from models.portfolio.markowitz import MarkowitzOptimizer


class MLPortfolioOptimizer:
    """
    Оптимізатор портфеля з ML-прогнозованою та EWMA-зваженою
    коваріаційною матрицею.

    Три режими оцінки Σ:
        1. 'sample'  — класична вибіркова матриця (baseline)
        2. 'ewma'    — EWMA-зважена (RiskMetrics, λ=0.94)
        3. 'ml_ewma' — ML-прогноз волатильностей + EWMA кореляції (найкраще)

    Ідея ml_ewma:
        Σ = D_ML · Corr_EWMA · D_ML
        де D_ML = diag(σ̂₁·√252, ..., σ̂ₙ·√252) — ML-прогнози (майбутнє)
        Corr_EWMA — EWMA-зважені кореляції (структура залежностей)
    """

    def __init__(
        self,
        tickers:        list[str],
        risk_free_rate: float = RISK_FREE_RATE,
        lambda_:        float = 0.94,
        corr_window:    int   = 60,
    ):
        self.tickers     = tickers
        self.rf          = risk_free_rate
        self.lambda_     = lambda_
        self.corr_window = corr_window
        self.n           = len(tickers)

    # ------------------------------------------------------------------
    # Побудова коваріаційних матриць
    # ------------------------------------------------------------------

    def _sample_cov(self, returns) -> np.ndarray:
        """Класична вибіркова коваріаційна матриця (річна)."""
        if hasattr(returns, 'cov'):
            return returns.cov().values * 252
        return np.cov(returns.T) * 252

    def _ewma_cov(self, returns) -> np.ndarray:
        """EWMA-зважена коваріаційна матриця (річна)."""
        from data.weighting import ewma_cov_matrix
        import pandas as pd
        if not isinstance(returns, pd.DataFrame):
            import pandas as pd
            returns = pd.DataFrame(returns, columns=self.tickers)
        return ewma_cov_matrix(returns, lambda_=self.lambda_, annualize=True)

    def _ml_ewma_cov(self, sigma_forecasts: np.ndarray, returns) -> np.ndarray:
        """
        ML-прогноз σ + EWMA кореляції.

        Формула: Σ = D · Corr_EWMA · D
        де D = diag(σ̂_ML * √252)
        """
        from data.weighting import ewma_cov_matrix
        import pandas as pd

        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(returns, columns=self.tickers)

        recent = returns.tail(self.corr_window)

        # EWMA кореляційна матриця
        ewma_cov_r = ewma_cov_matrix(recent, lambda_=self.lambda_, annualize=False)
        ewma_std   = np.sqrt(np.diag(ewma_cov_r))
        ewma_std   = np.where(ewma_std < 1e-8, 1e-8, ewma_std)
        corr_ewma  = ewma_cov_r / np.outer(ewma_std, ewma_std)

        # Регуляризація
        eigvals, eigvecs = np.linalg.eigh(corr_ewma)
        eigvals = np.maximum(eigvals, 1e-6)
        corr_ewma = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # ML σ → річні
        sigma_annual = sigma_forecasts * np.sqrt(252)
        D = np.diag(sigma_annual)

        cov_ml_ewma = D @ corr_ewma @ D
        return cov_ml_ewma

    # ------------------------------------------------------------------
    # Оптимізація
    # ------------------------------------------------------------------

    def optimize(
        self,
        returns,
        sigma_forecasts:  np.ndarray = None,
        method:           str = "max_sharpe",
        cov_method:       str = "ml_ewma",
    ) -> dict:
        """
        Знаходить оптимальні ваги портфеля.

        Args:
            returns:          DataFrame лог-доходностей (тренувальне вікно)
            sigma_forecasts:  ML-прогнози денної σ (n,); потрібні для 'ml_ewma'
            method:           'max_sharpe' або 'min_variance'
            cov_method:       'sample' | 'ewma' | 'ml_ewma'

        Returns:
            dict: weights, metrics, cov_matrix, cov_method
        """
        if hasattr(returns, 'mean'):
            mu = returns.mean().values * 252
        else:
            mu = np.mean(returns, axis=0) * 252

        # Вибір методу ковар. матриці
        if cov_method == "ewma":
            cov = self._ewma_cov(returns)
            label = f"EWMA (λ={self.lambda_})"
        elif cov_method == "ml_ewma" and sigma_forecasts is not None:
            cov = self._ml_ewma_cov(sigma_forecasts, returns)
            label = f"ML + EWMA (λ={self.lambda_})"
        else:
            cov = self._sample_cov(returns)
            label = "Sample"

        opt = MarkowitzOptimizer(
            expected_returns = mu,
            cov_matrix       = cov,
            risk_free_rate   = self.rf,
            tickers          = self.tickers,
        )

        weights = opt.max_sharpe() if method == "max_sharpe" else opt.min_variance()
        metrics = opt.portfolio_metrics(weights)

        return {
            "weights":    weights,
            "metrics":    metrics,
            "cov_matrix": cov,
            "cov_label":  label,
            "optimizer":  opt,
        }

    # ------------------------------------------------------------------
    # Rolling rebalancing (з вибором методу Σ)
    # ------------------------------------------------------------------

    def rolling_optimize(
        self,
        returns_df,
        sigma_forecasts_df = None,
        rebal_freq:  int = 21,
        method:      str = "max_sharpe",
        cov_method:  str = "ml_ewma",
    ) -> dict:
        """
        Щомісячна переоптимізація з вибраним методом оцінки Σ.
        """
        dates = returns_df.index
        weights_dict = {}
        current_w = np.ones(self.n) / self.n

        for i, date in enumerate(dates):
            if i % rebal_freq == 0 and i >= self.corr_window:
                window = returns_df.iloc[i - self.corr_window : i]
                sigma = None
                if sigma_forecasts_df is not None and date in sigma_forecasts_df.index:
                    sigma = sigma_forecasts_df.loc[date].values

                try:
                    result  = self.optimize(window, sigma, method, cov_method)
                    current_w = result["weights"]
                except Exception:
                    pass

            weights_dict[date] = current_w.copy()

        return weights_dict

    # ------------------------------------------------------------------
    # Порівняння всіх трьох методів
    # ------------------------------------------------------------------

    def compare_all_methods(
        self,
        returns,
        sigma_forecasts: np.ndarray,
    ) -> dict:
        """
        Порівнює Sample, EWMA та ML+EWMA на одних даних.

        Returns:
            dict: результати для кожного методу
        """
        results = {}
        for cov_m in ["sample", "ewma", "ml_ewma"]:
            try:
                r = self.optimize(returns, sigma_forecasts,
                                  method="max_sharpe", cov_method=cov_m)
                results[cov_m] = r
            except Exception as e:
                results[cov_m] = {"error": str(e)}

        print(f"\n{'Метод Σ':<20} {'Дохідність':>12} {'Ризик':>8} {'Sharpe':>8}")
        print("-" * 52)
        for name, r in results.items():
            if "error" not in r:
                m = r["metrics"]
                print(f"{r['cov_label']:<20} {m['return']:>12.4f} "
                      f"{m['risk']:>8.4f} {m['sharpe']:>8.4f}")

        return results
