# =============================================================================
# backtesting/backtest.py — Walk-Forward бектестинг трьох стратегій
# MILESTONE 5.1
# =============================================================================

import numpy as np
import pandas as pd
import sys, os
from scipy.stats import norm  # імпорт на рівні модуля (не в циклі)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import CONFIDENCE_LEVEL, RISK_FREE_RATE
from backtesting.metrics import (
    comprehensive_report, kupiec_test, var_exceedance_rate
)


class WalkForwardBacktest:
    """
    Walk-Forward бектестинг трьох інвестиційних стратегій.

    Методологія Walk-Forward:
        - Тренувальне вікно: 504 торгових дні (2 роки)
        - Крок ребалансування: 21 торговий день (1 місяць)
        - На кожному кроці: перенавчання моделей → нові ваги → торгівля
        - Порівнювані стратегії:
            1. ML-Markowitz (ML-прогноз σ → Σ_ML → max Sharpe)
            2. Classic-Markowitz (вибіркова Σ → max Sharpe)
            3. Equal-Weight (рівні ваги 1/n — benchmark)

    Особливість: walk-forward виключає data leakage — на кожному кроці
    модель бачить лише дані, які були доступні на той момент часу.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        train_window: int = 504,
        rebal_freq:   int = 21,
        alpha:        float = CONFIDENCE_LEVEL,
        risk_free_rate: float = RISK_FREE_RATE,
    ):
        self.prices    = prices
        self.returns   = np.log(prices / prices.shift(1)).dropna()
        self.tickers   = list(prices.columns)
        self.n         = len(self.tickers)
        self.train_window  = train_window
        self.rebal_freq    = rebal_freq
        self.alpha         = alpha
        self.rf            = risk_free_rate

        self.results   = {}
        self.nav       = {}
        self.weights_history = {}

    def run(self, verbose: bool = True) -> dict:
        test_dates = self.returns.index[self.train_window:]

        if verbose:
            print(f"[Backtest] Walk-Forward: {len(test_dates)} торгових днів")
            print(f"  Тренувальне вікно: {self.train_window} днів")
            print(f"  Ребалансування: кожні {self.rebal_freq} днів")
            print(f"  Тикери: {self.tickers}\n")

        strategies = ["ml_markowitz", "classic_markowitz", "equal_weight"]
        nav          = {s: [1.0] for s in strategies}
        weights_hist = {s: [] for s in strategies}
        var_hist     = {s: [] for s in strategies}
        current_weights = {s: np.ones(self.n) / self.n for s in strategies}

        for step, date in enumerate(test_dates):
            idx = self.returns.index.get_loc(date)
            train_ret = self.returns.iloc[idx - self.train_window : idx]

            if step % self.rebal_freq == 0:
                current_weights = self._rebalance(train_ret, current_weights, step, verbose)

            day_ret = self.returns.loc[date].values

            for s in strategies:
                w        = current_weights[s]
                port_ret = float(w @ day_ret)
                new_nav  = nav[s][-1] * (1 + port_ret)
                nav[s].append(new_nav)
                weights_hist[s].append(w.copy())

                sigma_p   = float(np.sqrt(w @ (train_ret.cov().values * 252) @ w) / np.sqrt(252))
                mu_p      = float(train_ret.mean().values @ w)
                var_today = -(mu_p + sigma_p * norm.ppf(1 - self.alpha))
                var_hist[s].append(var_today)

        dates_nav = [self.returns.index[self.train_window - 1]] + list(test_dates)

        for s in strategies:
            nav_arr = np.array(nav[s])
            ret_arr = np.diff(nav_arr) / nav_arr[:-1]
            var_arr = np.array(var_hist[s])

            self.nav[s] = pd.Series(nav_arr, index=dates_nav)
            self.weights_history[s] = pd.DataFrame(
                weights_hist[s], index=test_dates, columns=self.tickers
            )
            self.results[s] = comprehensive_report(
                strategy_name  = s,
                nav            = nav_arr,
                returns        = ret_arr,
                var_estimates  = var_arr,
                alpha          = self.alpha,
                risk_free_rate = self.rf,
            )

        if verbose:
            self._print_summary()

        return self.results

    def _rebalance(self, train_ret, current_weights, step, verbose):
        new_weights = {}
        new_weights["equal_weight"] = np.ones(self.n) / self.n

        # Classic-Markowitz: вибіркова коваріаційна матриця
        try:
            from models.portfolio.markowitz import MarkowitzOptimizer
            mu    = train_ret.mean().values * 252
            Sigma = train_ret.cov().values  * 252
            opt   = MarkowitzOptimizer(mu, Sigma, tickers=self.tickers,
                                        risk_free_rate=self.rf)
            new_weights["classic_markowitz"] = opt.max_sharpe()
        except Exception as e:
            if verbose and step == 0:
                print(f"  [Classic Markowitz] Помилка: {e}. Використовуємо equal-weight.")
            new_weights["classic_markowitz"] = current_weights["classic_markowitz"]

        # ML-Markowitz: rolling-std як proxy ML-прогнозу σ
        # ВИПРАВЛЕНО: правильний порядок аргументів — (returns, sigma_forecasts)
        # Раніше аргументи були переставлені (sigma_ml, train_ret) замість (train_ret, sigma_ml),
        # що призводило до AttributeError і fallback на equal-weight на кожному кроці.
        try:
            sigma_ml = train_ret.tail(20).std().values   # proxy ML σ: (n,)
            from models.portfolio.ml_optimizer import MLPortfolioOptimizer
            ml_opt = MLPortfolioOptimizer(tickers=self.tickers, risk_free_rate=self.rf)
            result = ml_opt.optimize(
                returns         = train_ret,   # DataFrame (T × n)
                sigma_forecasts = sigma_ml,    # ndarray (n,)
                method          = "max_sharpe",
                cov_method      = "ewma",      # EWMA матриця без ML sigma для стабільності
            )
            new_weights["ml_markowitz"] = result["weights"]
        except Exception as e:
            if verbose and step == 0:
                print(f"  [ML Markowitz] Помилка: {e}. Використовуємо equal-weight.")
            new_weights["ml_markowitz"] = current_weights["ml_markowitz"]

        if verbose and step % (self.rebal_freq * 6) == 0:
            print(f"  Крок {step}: ребалансування виконано")

        return new_weights

    def _print_summary(self):
        print(f"\n{'='*72}")
        print(f"ПІДСУМОК БЕКТЕСТИНГУ")
        print(f"{'='*72}")
        print(f"{'Стратегія':<24} {'CAGR':>7} {'Sharpe':>7} {'MDD':>7} "
              f"{'Calmar':>7} {'VaR%':>6} {'Kupiec':>8}")
        print(f"{'-'*72}")
        for s, r in self.results.items():
            kupiec_ok = "OK" if not r.get("kupiec_reject", True) else "REJECT"
            var_rate  = r.get("var_exceedance_rate", float("nan"))
            print(f"{s:<24} {r['cagr']:>7.3f} {r['sharpe']:>7.3f} "
                  f"{r['max_drawdown']:>7.3f} {r['calmar']:>7.3f} "
                  f"{var_rate:>6.3f} {kupiec_ok:>8}")
        print(f"{'='*72}\n")

    def get_nav_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.nav)

    def get_drawdown_series(self, strategy: str) -> pd.Series:
        nav  = self.nav[strategy].values
        peak = np.maximum.accumulate(nav)
        dd   = (nav - peak) / peak
        return pd.Series(dd, index=self.nav[strategy].index)

    def get_rolling_sharpe(self, strategy: str, window: int = 63) -> pd.Series:
        returns = self.nav[strategy].pct_change().dropna()
        rolling = returns.rolling(window)
        return (rolling.mean() * 252 - self.rf) / (rolling.std() * np.sqrt(252))
