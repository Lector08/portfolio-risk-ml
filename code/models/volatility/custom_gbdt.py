# =============================================================================
# models/volatility/custom_gbdt.py
#
# Власна реалізація Gradient Boosting Decision Trees (GBDT) на чистому NumPy.
# Мета: порівняти з XGBoost за точністю (RMSE, MAE, QLIKE) і швидкістю.
#
# Алгоритм (Friedman, 2001):
#   1. F_0(x) = mean(y)                       <- константний початковий прогноз
#   2. Для m = 1..M:
#       a. r_i = y_i - F_{m-1}(x_i)           <- псевдо-залишки (негативний градієнт MSE)
#       b. Навчаємо дерево h_m на залишках r
#       c. F_m(x) = F_{m-1}(x) + lr * h_m(x)  <- оновлення ансамблю
#   3. Прогноз: F_M(x)
#
# Дерево рішень (регресійне, CART-подібне):
#   - Жадібний пошук найкращого розбиття по всіх ознаках і порогах
#   - Критерій: мінімум зваженого MSE в дочірніх вузлах
#   - Обмеження глибини (max_depth) для регуляризації
#   - Leaf value = mean(y у листку)
# =============================================================================

import numpy as np
import time
from typing import Optional


# =============================================================================
# Вузол дерева рішень
# =============================================================================

class _Node:
    """Один вузол регресійного дерева рішень."""

    __slots__ = ("feature", "threshold", "left", "right", "value", "n_samples")

    def __init__(self):
        self.feature   = None   # int: індекс ознаки для розбиття
        self.threshold = None   # float: поріг розбиття
        self.left      = None   # _Node: лівий нащадок (x <= threshold)
        self.right     = None   # _Node: правий нащадок (x > threshold)
        self.value     = None   # float: значення листка = mean(y)
        self.n_samples = 0


# =============================================================================
# Регресійне дерево рішень
# =============================================================================

class _RegressionTree:
    """
    Регресійне CART-дерево. Будується жадібним пошуком кращого розбиття.

    Параметри:
        max_depth:        максимальна глибина дерева
        min_samples_leaf: мінімум зразків у кожному листку
        max_features:     частка ознак для розгляду на кожному розбитті (random subspace)
        random_state:     відтворюваність
    """

    def __init__(self, max_depth=4, min_samples_leaf=5, max_features=0.7, random_state=42):
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features     = max_features
        self._rng             = np.random.default_rng(random_state)
        self.root_            = None

    def fit(self, X, y):
        n_features = X.shape[1]
        self._n_features = max(1, int(self.max_features * n_features))
        self.root_ = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        node           = _Node()
        node.n_samples = len(y)
        node.value     = float(np.mean(y))

        # Умови зупинки
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return node

        best_feat, best_thr = self._best_split(X, y)
        if best_feat is None:
            return node

        mask = X[:, best_feat] <= best_thr
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            return node

        node.feature   = best_feat
        node.threshold = best_thr
        node.left  = self._build(X[mask],  y[mask],  depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _best_split(self, X, y):
        """Знаходить розбиття що мінімізує зважений MSE дочірніх вузлів."""
        n = len(y)
        feat_idx   = self._rng.choice(X.shape[1], size=self._n_features, replace=False)
        best_feat  = None
        best_thr   = None
        best_score = float(np.var(y) * n)   # baseline = MSE поточного вузла

        for feat in feat_idx:
            x_col    = X[:, feat]
            sort_idx = np.argsort(x_col)
            xs       = x_col[sort_idx]
            ys       = y[sort_idx]

            # Prefix sums для ефективного розрахунку MSE
            cs  = np.cumsum(ys)
            cs2 = np.cumsum(ys ** 2)
            tot  = cs[-1]
            tot2 = cs2[-1]

            for i in range(self.min_samples_leaf - 1, n - self.min_samples_leaf):
                if xs[i] == xs[i + 1]:
                    continue

                nl = i + 1
                nr = n - nl

                sl  = cs[i];  sl2 = cs2[i]
                sr  = tot - sl; sr2 = tot2 - sl2

                mse_l = sl2 - sl * sl / nl
                mse_r = sr2 - sr * sr / nr
                score = mse_l + mse_r

                if score < best_score:
                    best_score = score
                    best_feat  = feat
                    best_thr   = 0.5 * (xs[i] + xs[i + 1])

        return best_feat, best_thr

    def predict(self, X):
        return np.array([self._traverse(self.root_, x) for x in X])

    def _traverse(self, node, x):
        if node.feature is None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(node.left, x)
        return self._traverse(node.right, x)


# =============================================================================
# Gradient Boosting
# =============================================================================

class CustomGBDT:
    """
    Власна реалізація Gradient Boosting для регресії (цільова функція: MSE).

    Алгоритм Friedman (2001) "Greedy Function Approximation":
        F_0 = mean(y)
        For m = 1..n_estimators:
            r_m = y - F_{m-1}(x)            <- негативний градієнт MSE = залишки
            h_m = RegressionTree.fit(X, r_m) <- дерево на залишках
            F_m = F_{m-1} + lr * h_m         <- оновлення прогнозу

    Підтримує:
        - Early stopping (зупинка якщо val_loss не падає N ітерацій)
        - Subsampling (stochastic gradient boosting, зменшує overfitting)
        - Random subspace (max_features < 1.0, теж регуляризація)
    """

    def __init__(
        self,
        n_estimators          = 200,
        learning_rate         = 0.05,
        max_depth             = 4,
        min_samples_leaf      = 5,
        max_features          = 0.7,
        subsample             = 0.8,
        early_stopping_rounds = 20,
        random_state          = 42,
        verbose               = True,
    ):
        self.n_estimators          = n_estimators
        self.learning_rate         = learning_rate
        self.max_depth             = max_depth
        self.min_samples_leaf      = min_samples_leaf
        self.max_features          = max_features
        self.subsample             = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state          = random_state
        self.verbose               = verbose
        self._rng                  = np.random.default_rng(random_state)

        self._trees          = []
        self._F0             = 0.0
        self.best_iteration_ = 0
        self.train_losses_   = []
        self.val_losses_     = []

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Навчає GBDT з early stopping.

        Args:
            X_train, y_train: навчальна вибірка (numpy arrays)
            X_val,   y_val:   валідаційна вибірка для early stopping

        Returns:
            self
        """
        n_train = len(X_train)

        # F_0: початковий прогноз = mean(y_train)
        self._F0 = float(np.mean(y_train))
        F_train  = np.full(n_train,    self._F0, dtype=np.float64)
        F_val    = np.full(len(X_val), self._F0, dtype=np.float64)

        self._trees        = []
        self.train_losses_ = []
        self.val_losses_   = []

        best_val_loss  = np.inf
        no_improve     = 0
        self.best_iteration_ = 0
        t_start = time.perf_counter()

        for m in range(self.n_estimators):

            # 1. Псевдо-залишки (негативний градієнт MSE = y - F)
            residuals = y_train - F_train

            # 2. Subsampling: навчаємо дерево на підмножині рядків
            if self.subsample < 1.0:
                n_sub   = max(1, int(n_train * self.subsample))
                idx     = self._rng.choice(n_train, size=n_sub, replace=False)
                Xs, rs  = X_train[idx], residuals[idx]
            else:
                Xs, rs  = X_train, residuals

            # 3. Навчаємо дерево на залишках
            tree = _RegressionTree(
                max_depth        = self.max_depth,
                min_samples_leaf = self.min_samples_leaf,
                max_features     = self.max_features,
                random_state     = self.random_state + m,
            )
            tree.fit(Xs, rs)

            # 4. Оновлюємо прогнози: F += lr * h(x)
            F_train += self.learning_rate * tree.predict(X_train)
            F_val   += self.learning_rate * tree.predict(X_val)

            self._trees.append(tree)

            # 5. Логуємо втрати
            tl = float(np.mean((y_train - F_train) ** 2))
            vl = float(np.mean((y_val   - F_val)   ** 2))
            self.train_losses_.append(tl)
            self.val_losses_.append(vl)

            # 6. Early stopping
            if vl < best_val_loss - 1e-8:
                best_val_loss        = vl
                no_improve           = 0
                self.best_iteration_ = m + 1
            else:
                no_improve += 1

            if self.verbose and (m + 1) % 25 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"  [{m+1:>4}/{self.n_estimators}]  "
                      f"train_MSE={tl:.6f}  val_MSE={vl:.6f}  ({elapsed:.1f}s)")

            if no_improve >= self.early_stopping_rounds:
                if self.verbose:
                    print(f"\n  Early stopping на ітерації {m+1} "
                          f"(best_iter={self.best_iteration_}, "
                          f"best_val_RMSE={best_val_loss**0.5:.6f})")
                break

        elapsed = time.perf_counter() - t_start
        if self.verbose:
            print(f"\n[CustomGBDT] Завершено за {elapsed:.2f}с | "
                  f"Дерев: {len(self._trees)} | "
                  f"Best iter: {self.best_iteration_} | "
                  f"Best val_RMSE: {best_val_loss**0.5:.6f}")

        # Залишаємо тільки дерева до best_iteration (як XGBoost)
        self._trees = self._trees[:self.best_iteration_]
        return self

    def predict(self, X):
        """Прогнозує значення для матриці X."""
        if not self._trees:
            raise RuntimeError("Модель не навчена. Виклич fit() спочатку.")

        F = np.full(len(X), self._F0, dtype=np.float64)
        for tree in self._trees:
            F += self.learning_rate * tree.predict(X)

        return np.clip(F, 0, None).astype(np.float32)

    def evaluate(self, X, y_true):
        """
        RMSE, MAE, QLIKE — ідентична реалізація до xgboost_model.py
        для коректного порівняння метрик.
        """
        y_pred = self.predict(X)
        eps    = 1e-8

        rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae   = float(np.mean(np.abs(y_true - y_pred)))
        qlike = float(np.mean(
            np.log(y_pred ** 2 + eps) + y_true ** 2 / (y_pred ** 2 + eps)
        ))

        return {"rmse": rmse, "mae": mae, "qlike": qlike}


# =============================================================================
# Допоміжна функція: 19 ознак + цільова змінна (ідентично до app.py)
# =============================================================================

def build_features_and_target(prices_series):
    """
    Будує матрицю 19 ознак і цільову змінну log1p(RV).
    Повністю ідентично до train_per_ticker() в app.py — для чесного порівняння.
    """
    import pandas as pd

    r = np.log(prices_series / prices_series.shift(1)).dropna()
    p = prices_series.loc[r.index]
    f = {}

    for w in [5, 20]:
        f[f"ma{w}_ratio"] = p / p.rolling(w).mean() - 1

    delta = p.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta).clip(lower=0).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    f["rsi14"] = (100 - 100 / (1 + rs)) / 50 - 1

    ma20  = p.rolling(20).mean()
    std20 = p.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    f["pct_b"]     = ((p - lower) / (upper - lower).replace(0, np.nan)).clip(0, 1)
    f["bandwidth"] = (upper - lower) / ma20.replace(0, np.nan)

    for w in [5, 10, 20]:
        f[f"std{w}"]  = r.rolling(w).std()
        f[f"skew{w}"] = r.rolling(w).skew()
        f[f"kurt{w}"] = r.rolling(w).kurt()

    for lag in [1, 2, 3, 5, 10]:
        f[f"lag{lag}"] = r.shift(lag)

    X_df = pd.DataFrame(f, index=p.index).dropna()

    # Цільова змінна: log1p(realized_vol) — ідентично app.py
    rv   = r.pow(2).rolling(20).sum().apply(np.sqrt) * np.sqrt(252)
    rv   = np.log1p(rv.dropna())

    common = X_df.index.intersection(rv.index)
    return (
        X_df.loc[common].values.astype(np.float32),
        rv.loc[common].values.astype(np.float32),
        list(X_df.columns),
    )


# =============================================================================
# Запуск порівняння:  python3 models/volatility/custom_gbdt.py
# =============================================================================

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

    TICKER = "AAPL"

    print("=" * 65)
    print("  Порівняння: CustomGBDT vs XGBoost")
    print(f"  Тікер: {TICKER}")
    print("=" * 65)

    # ── Завантаження даних ─────────────────────────────────────────────
    try:
        import yfinance as yf
        raw    = yf.download(TICKER, start="2018-01-01",
                             auto_adjust=True, progress=False)
        prices = raw["Close"].squeeze()
        print(f"\nЗавантажено {len(prices)} рядків з yfinance")
    except Exception as e:
        print(f"Помилка завантаження: {e}")
        sys.exit(1)

    X, y, feature_names = build_features_and_target(prices)

    n  = len(X)
    t1 = int(n * 0.80)
    t2 = int(n * 0.90)
    X_tr, X_val, X_te = X[:t1], X[t1:t2], X[t2:]
    y_tr, y_val, y_te = y[:t1], y[t1:t2], y[t2:]

    # Нормалізація (fit тільки на train)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_tr_s  = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)
    X_te_s  = sc.transform(X_te)

    print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_te)}")
    print(f"Ознак: {X.shape[1]}")

    results = {}

    # ─────────────────────────────────────────────────────────────────────
    # 1. CustomGBDT
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("  1. CustomGBDT (numpy, власна реалізація з нуля)")
    print("-" * 65)

    cmodel = CustomGBDT(
        n_estimators          = 300,
        learning_rate         = 0.05,
        max_depth             = 4,
        min_samples_leaf      = 5,
        max_features          = 0.7,
        subsample             = 0.8,
        early_stopping_rounds = 20,
        verbose               = True,
    )
    t0 = time.perf_counter()
    cmodel.fit(X_tr_s, y_tr, X_val_s, y_val)
    custom_time = time.perf_counter() - t0
    results["CustomGBDT"] = {
        "time":    custom_time,
        "n_trees": len(cmodel._trees),
        **cmodel.evaluate(X_te_s, y_te),
    }

    # ─────────────────────────────────────────────────────────────────────
    # 2. XGBoost
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("  2. XGBoost (бібліотечна реалізація, C++ ядро)")
    print("-" * 65)

    try:
        from models.volatility.xgboost_model import XGBoostVolatilityModel
        xmodel = XGBoostVolatilityModel()
        t0 = time.perf_counter()
        xmodel.fit(X_tr_s, y_tr, X_val_s, y_val, feature_names=feature_names)
        xgb_time = time.perf_counter() - t0
        results["XGBoost"] = {
            "time":    xgb_time,
            "n_trees": xmodel.model.best_iteration,
            **xmodel.evaluate(X_te_s, y_te),
        }
    except Exception as e:
        print(f"  Помилка XGBoost: {e}")
        results["XGBoost"] = None

    # ─────────────────────────────────────────────────────────────────────
    # 3. Підсумкова таблиця
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  РЕЗУЛЬТАТИ ПОРІВНЯННЯ  (тестова вибірка)")
    print("=" * 65)
    print(f"\n  {'Модель':<18} {'RMSE':>10} {'MAE':>10} {'QLIKE':>10} "
          f"{'Час (с)':>9} {'Дерев':>7}")
    print(f"  {'-'*62}")

    for name, r in results.items():
        if r is None:
            print(f"  {name:<18}  (помилка)")
            continue
        print(f"  {name:<18} {r['rmse']:>10.6f} {r['mae']:>10.6f} "
              f"{r['qlike']:>10.4f} {r['time']:>9.2f} {r['n_trees']:>7d}")

    if results.get("XGBoost") and results.get("CustomGBDT"):
        xg = results["XGBoost"]
        cg = results["CustomGBDT"]
        print(f"\n  Відношення RMSE (Custom/XGB): "
              f"{cg['rmse']/xg['rmse']:.3f}×  "
              f"({'гірше' if cg['rmse'] > xg['rmse'] else 'краще'})")
        print(f"  Відношення часу (Custom/XGB): "
              f"{cg['time']/xg['time']:.1f}×  "
              f"(XGBoost швидший у {cg['time']/xg['time']:.1f} разів)")

    print("\n" + "=" * 65)
    print("  ЧОМУ XGBoost ШВИДШИЙ:")
    print("  - C++ ядро з паралельним пошуком розбиттів (OpenMP)")
    print("  - Гістограмний алгоритм: O(n_bins) замість O(n) на розбиття")
    print("  - Level-wise побудова дерев з cache-friendly доступом до пам'яті")
    print("  - Векторизовані операції над bin indices")
    print()
    print("  ЧОМУ ТОЧНОСТІ БЛИЗЬКІ:")
    print("  - Обидва алгоритми реалізують один і той самий GBDT фреймворк")
    print("  - Головна відмінність — ефективність обчислень, а не алгоритм")
    print("  - XGBoost має L1/L2 регуляризацію на ваги листків (наш — ні)")
    print("=" * 65)
