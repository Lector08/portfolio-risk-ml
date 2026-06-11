# =============================================================================
# models/portfolio/markowitz.py
# Оптимізація Марковіца з обмеженнями на ваги і мінімальну доходність
# =============================================================================

import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config import RISK_FREE_RATE

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


class InfeasibleConstraintError(Exception):
    """
    Виникає коли задача оптимізації нерозв'язна через обмеження.
    Наприклад: жоден актив не дає мінімально необхідної доходності.
    """
    pass


class MarkowitzOptimizer:
    """
    Оптимізація портфеля Марковіца з обмеженнями на ваги та мінімальну доходність.

    Підтримує:
        - Обмеження на ваги позицій (lb_i <= w_i <= ub_i)
        - Мінімальну цільову доходність (w^T mu >= r_min)

    Якщо задача нерозв'язна (r_min недосяжний) — кидає InfeasibleConstraintError.
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix:       np.ndarray,
        risk_free_rate:   float = RISK_FREE_RATE,
        tickers:          list[str] = None,
    ):
        self.mu      = np.array(expected_returns, dtype=float)
        self.Sigma   = np.array(cov_matrix,       dtype=float)
        self.rf      = risk_free_rate
        self.n       = len(expected_returns)
        self.tickers = tickers or [f"Asset_{i}" for i in range(self.n)]

        self.min_var_weights    = None
        self.max_sharpe_weights = None
        self.frontier_risks     = None
        self.frontier_returns   = None
        self.frontier_weights   = None

        self._regularize()

    def _regularize(self):
        self.Sigma = (self.Sigma + self.Sigma.T) / 2
        eigvals = np.linalg.eigvalsh(self.Sigma)
        if np.any(eigvals < 1e-8):
            self.Sigma += np.eye(self.n) * (abs(min(eigvals)) + 1e-6)

    def check_min_return_feasibility(
        self,
        r_min:  float,
        lb:     np.ndarray,
        ub:     np.ndarray,
    ) -> dict:
        """
        Перевіряє чи можливо досягти мінімальну доходність r_min.

        Вирішує допоміжну лінійну програму:
            max  mu^T w
            s.t. sum(w) = 1, lb <= w <= ub

        Returns:
            dict: feasible, max_achievable, shortfall, best_single
        """
        # Точний розрахунок через cvxpy або scipy
        max_ret = None
        if CVXPY_AVAILABLE:
            w    = cp.Variable(self.n)
            prob = cp.Problem(cp.Maximize(self.mu @ w),
                              [cp.sum(w) == 1, w >= lb, w <= ub])
            prob.solve(solver=cp.OSQP, verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                max_ret = float(prob.value)

        if max_ret is None:
            try:
                from scipy.optimize import linprog
                res = linprog(-self.mu,
                              A_eq=np.ones((1, self.n)), b_eq=[1],
                              bounds=list(zip(lb, ub)), method="highs")
                if res.success:
                    max_ret = float(-res.fun)
            except Exception:
                pass

        # Fallback (виправлено): вкладаємо максимально можливу частку
        # в найдохідніший актив при дотриманні lb/ub
        if max_ret is None:
            best_idx   = int(np.argmax(self.mu))
            w_fallback = lb.copy().astype(float)
            remaining  = max(0.0, 1.0 - lb.sum())
            added      = min(remaining, ub[best_idx] - lb[best_idx])
            w_fallback[best_idx] += added
            max_ret = float(self.mu @ w_fallback)

        best_single = self.tickers[int(np.argmax(self.mu))]
        return {
            "feasible":       max_ret >= r_min - 1e-6,
            "max_achievable": max_ret,
            "shortfall":      max(0.0, r_min - max_ret),
            "best_single":    best_single,
        }

    def _solve_min_var(self, lb, ub, r_target=None):
        if CVXPY_AVAILABLE:
            return self._solve_cvxpy(lb, ub, r_target)
        return self._solve_scipy(lb, ub, r_target)

    def _solve_cvxpy(self, lb, ub, r_target):
        w    = cp.Variable(self.n)
        obj  = cp.Minimize(cp.quad_form(w, self.Sigma))
        cons = [cp.sum(w) == 1, w >= lb, w <= ub]
        if r_target is not None:
            cons.append(self.mu @ w >= r_target)
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, verbose=False)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return None
        wv = np.clip(np.array(w.value).flatten(), lb, ub)
        s  = wv.sum()
        return wv / s if s > 1e-10 else None

    def _solve_scipy(self, lb, ub, r_target):
        from scipy.optimize import minimize
        def obj(w):
            return float(w @ self.Sigma @ w)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        if r_target is not None:
            constraints.append({"type": "ineq", "fun": lambda w: self.mu @ w - r_target})
        bounds = list(zip(lb, ub))
        w0 = np.clip(np.ones(self.n) / self.n, lb, ub)
        w0 /= w0.sum()
        res = minimize(obj, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-12, "maxiter": 1000})
        if not res.success:
            return None
        wv = np.clip(res.x, lb, ub)
        s  = wv.sum()
        return wv / s if s > 1e-10 else None

    def min_variance(self, lb=None, ub=None, r_min=None):
        lb = np.zeros(self.n) if lb is None else np.array(lb, dtype=float)
        ub = np.ones(self.n)  if ub is None else np.array(ub, dtype=float)
        if r_min is not None:
            check = self.check_min_return_feasibility(r_min, lb, ub)
            if not check["feasible"]:
                raise InfeasibleConstraintError(
                    f"Мінімальна доходність {r_min*100:.1f}% недосяжна. "
                    f"Максимально можливо: {check['max_achievable']*100:.1f}%."
                )
        w = self._solve_min_var(lb, ub, r_target=r_min)
        if w is None:
            w = np.clip(np.ones(self.n) / self.n, lb, ub)
            w /= w.sum()
        self.min_var_weights = w
        ret, risk, sharpe = self._metrics(w)
        print(f"[Markowitz] Min-Variance: ret={ret:.4f} | risk={risk:.4f} | sharpe={sharpe:.4f}")
        return w

    def max_sharpe(self, lb=None, ub=None, r_min=None):
        lb = np.zeros(self.n) if lb is None else np.array(lb, dtype=float)
        ub = np.ones(self.n)  if ub is None else np.array(ub, dtype=float)
        if r_min is not None:
            check = self.check_min_return_feasibility(r_min, lb, ub)
            if not check["feasible"]:
                raise InfeasibleConstraintError(
                    f"Мінімальна доходність {r_min*100:.1f}% недосяжна. "
                    f"Максимально можливо: {check['max_achievable']*100:.1f}%."
                )
        from scipy.optimize import minimize

        def neg_sharpe(w):
            ret  = float(self.mu @ w)
            risk = float(np.sqrt(max(float(w @ self.Sigma @ w), 1e-12)))
            return -(ret - self.rf) / risk

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        if r_min is not None:
            constraints.append({"type": "ineq", "fun": lambda w: self.mu @ w - r_min})
        bounds = list(zip(lb, ub))

        best_sharpe = -np.inf
        best_w      = np.clip(np.ones(self.n) / self.n, lb, ub)
        best_w     /= best_w.sum()

        np.random.seed(42)
        for _ in range(30):
            w0 = np.random.dirichlet(np.ones(self.n))
            w0 = np.clip(w0, lb, ub)
            s  = w0.sum()
            if s < 1e-10:
                continue
            w0 /= s
            res = minimize(neg_sharpe, w0, method="SLSQP",
                           bounds=bounds, constraints=constraints,
                           options={"ftol": 1e-12, "maxiter": 1000})
            if res.success and -res.fun > best_sharpe:
                best_sharpe = -res.fun
                best_w      = res.x

        best_w = np.clip(best_w, lb, ub)
        best_w /= best_w.sum()
        self.max_sharpe_weights = best_w
        ret, risk, sharpe = self._metrics(best_w)
        print(f"[Markowitz] Max-Sharpe: ret={ret:.4f} | risk={risk:.4f} | sharpe={sharpe:.4f}")
        self._print_weights(best_w)
        return best_w

    def efficient_frontier(self, n_points=100, lb=None, ub=None):
        lb = np.zeros(self.n) if lb is None else np.array(lb, dtype=float)
        ub = np.ones(self.n)  if ub is None else np.array(ub, dtype=float)
        if self.min_var_weights is None:
            try:
                self.min_variance(lb, ub)
            except InfeasibleConstraintError:
                self.min_var_weights = np.clip(np.ones(self.n) / self.n, lb, ub)
                self.min_var_weights /= self.min_var_weights.sum()
        r_min_ef = float(self.mu @ self.min_var_weights)
        r_max_ef = float(np.max(self.mu))
        risks, returns, weights_list = [], [], []
        for r_t in np.linspace(r_min_ef, r_max_ef, n_points):
            w = self._solve_min_var(lb, ub, r_target=r_t)
            if w is None:
                continue
            risks.append(float(np.sqrt(w @ self.Sigma @ w)))
            returns.append(float(self.mu @ w))
            weights_list.append(w)
        self.frontier_risks   = np.array(risks)
        self.frontier_returns = np.array(returns)
        self.frontier_weights = np.array(weights_list)
        print(f"[Markowitz] Ефективна межа: {len(risks)} точок")
        return self.frontier_risks, self.frontier_returns, self.frontier_weights

    def _metrics(self, w):
        ret    = float(self.mu @ w)
        var    = float(w @ self.Sigma @ w)
        risk   = float(np.sqrt(max(var, 0)))
        sharpe = (ret - self.rf) / risk if risk > 1e-10 else 0.0
        return ret, risk, sharpe

    def portfolio_metrics(self, weights):
        w = np.array(weights)
        ret, risk, sharpe = self._metrics(w)
        return {"return": ret, "risk": risk, "sharpe": sharpe,
                "weights": dict(zip(self.tickers, w))}

    def _print_weights(self, w, top_n=5):
        for t, wi in sorted(zip(self.tickers, w), key=lambda x: -x[1])[:top_n]:
            if wi > 0.001:
                print(f"    {t}: {wi:.3f} ({wi*100:.1f}%)")

    def compare_with_equal_weight(self):
        ew = np.ones(self.n) / self.n
        results = {"equal_weight": self.portfolio_metrics(ew)}
        if self.min_var_weights  is not None:
            results["min_variance"] = self.portfolio_metrics(self.min_var_weights)
        if self.max_sharpe_weights is not None:
            results["max_sharpe"]  = self.portfolio_metrics(self.max_sharpe_weights)
        print(f"\n{'Портфель':<20} {'Доходність':>12} {'Ризик':>8} {'Sharpe':>8}")
        print("-" * 52)
        for name, m in results.items():
            print(f"{name:<20} {m['return']:>12.4f} {m['risk']:>8.4f} {m['sharpe']:>8.4f}")
        return results
