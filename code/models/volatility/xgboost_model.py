# =============================================================================
# models/volatility/xgboost_model.py — XGBoost прогноз волатильності
# MILESTONE 2.1
# =============================================================================

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class XGBoostVolatilityModel:
    """
    XGBoost-регресія для прогнозу реалізованої волатильності.

    Цільова змінна: RV_t (реалізована волатильність, вікно=20)
    Ознаки: технічні індикатори + rolling stats + лагові доходності

    Підбір гіперпараметрів: Optuna (байєсівська оптимізація, n_trials=50)
    """

    def __init__(self, params: dict = None):
        """
        Args:
            params: гіперпараметри XGBoost; якщо None — використовуються дефолтні
        """
        self.default_params = {
            "n_estimators":  500,
            "max_depth":     6,
            "learning_rate": 0.05,
            "subsample":     0.8,
            "colsample_bytree": 0.8,
            "reg_alpha":     0.1,
            "reg_lambda":    1.0,
            "random_state":  42,
            "n_jobs":        -1,
            "verbosity":     0,
        }
        self.params = params or self.default_params
        self.model = None
        self.best_params = {}
        self.feature_names = None

    # ------------------------------------------------------------------
    # Навчання
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        feature_names: list[str] = None,
    ) -> "XGBoostVolatilityModel":
        """
        Навчає XGBoost з early stopping на validation set.

        Args:
            X_train, y_train: навчальна вибірка
            X_val,   y_val:   валідаційна вибірка (для early stopping)
            feature_names:    назви ознак (для feature importance)

        Returns:
            self (для method chaining)
        """
        self.feature_names = feature_names

        self.model = xgb.XGBRegressor(
            **self.params,
            early_stopping_rounds=30,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_pred = self.model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"[XGBoost] Навчено: best_iteration={self.model.best_iteration}, "
              f"val_RMSE={val_rmse:.6f}")

        return self

    # ------------------------------------------------------------------
    # Прогноз
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Прогнозує реалізовану волатильність.

        Args:
            X: матриця ознак (n_samples, n_features)

        Returns:
            np.ndarray: прогноз sigma_hat (n_samples,)
        """
        if self.model is None:
            raise RuntimeError("Модель не навчена. Спочатку викличте fit()")
        pred = self.model.predict(X)
        return np.clip(pred, a_min=0, a_max=None)   # волатильність >= 0

    # ------------------------------------------------------------------
    # Optuna: підбір гіперпараметрів
    # ------------------------------------------------------------------

    def tune_hyperparams(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        n_trials: int = 50,
    ) -> dict:
        """
        Підбирає оптимальні гіперпараметри через Optuna (байєсівська оптимізація).

        Простір пошуку:
            n_estimators:      [100, 1000]
            max_depth:         [3, 10]
            learning_rate:     [0.01, 0.3]   (log scale)
            subsample:         [0.5, 1.0]
            colsample_bytree:  [0.5, 1.0]
            reg_alpha:         [1e-4, 10]    (log scale)
            reg_lambda:        [1e-4, 10]    (log scale)

        Args:
            n_trials: кількість ітерацій Optuna (рекомендовано 50+)

        Returns:
            dict: найкращі знайдені гіперпараметри
        """
        if not OPTUNA_AVAILABLE:
            print("[XGBoost] Optuna не встановлено. pip3 install optuna")
            return self.default_params

        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
                "max_depth":        trial.suggest_int("max_depth", 3, 10),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
            }
            model = xgb.XGBRegressor(**params, early_stopping_rounds=20)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
            pred = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, pred))   # мінімізуємо RMSE

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        self.best_params["random_state"] = 42
        self.best_params["n_jobs"] = -1
        self.best_params["verbosity"] = 0
        self.params = self.best_params

        print(f"[XGBoost] Optuna завершено: best_RMSE={study.best_value:.6f}")
        print(f"[XGBoost] Кращі параметри: {self.best_params}")
        return self.best_params

    # ------------------------------------------------------------------
    # Метрики та feature importance
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Розраховує метрики якості прогнозу.

        Returns:
            dict з ключами: rmse, mae, qlike
        """
        y_pred = self.predict(X)

        rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
        mae   = mean_absolute_error(y_true, y_pred)

        # QLIKE loss — стандартна метрика для волатильності
        # QLIKE = mean( log(sigma^2) + realized^2 / sigma^2 )
        eps = 1e-8
        qlike = np.mean(np.log(y_pred**2 + eps) + y_true**2 / (y_pred**2 + eps))

        metrics = {"rmse": rmse, "mae": mae, "qlike": qlike}
        print(f"[XGBoost] Метрики → RMSE={rmse:.6f} | MAE={mae:.6f} | QLIKE={qlike:.4f}")
        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.Series:
        """
        Повертає топ-N найважливіших ознак (gain importance).

        Returns:
            pd.Series: відсортований Series (ознака → важливість)
        """
        if self.model is None:
            raise RuntimeError("Модель не навчена")

        importance = self.model.feature_importances_
        names = self.feature_names or [f"f{i}" for i in range(len(importance))]

        series = pd.Series(importance, index=names).sort_values(ascending=False)
        return series.head(top_n)

    # ------------------------------------------------------------------
    # Збереження / завантаження
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Зберігає модель у файл .json (нативний формат XGBoost)"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        print(f"[XGBoost] Модель збережено → {path}")

    def load(self, path: str):
        """Завантажує модель з файлу"""
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
        print(f"[XGBoost] Модель завантажено ← {path}")
        return self


# ---------------------------------------------------------------------------
# Швидкий тест: python models/volatility/xgboost_model.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

    from data.loader import get_prices
    from data.preprocessor import compute_log_returns, train_val_test_split, compute_realized_volatility
    from data.feature_engineering import build_feature_matrix

    # 1. Дані
    prices  = get_prices()
    returns = compute_log_returns(prices)
    rv      = compute_realized_volatility(returns)   # цільова змінна

    # 2. Матриця ознак
    X_df = build_feature_matrix(prices, returns)

    # Вирівнюємо індекс X та y (rv може починатись пізніше)
    common_idx = X_df.index.intersection(rv.index)
    X_df = X_df.loc[common_idx]
    y_df = rv.loc[common_idx]

    # Беремо один тикер для демо (AAPL)
    ticker = "AAPL"
    X = X_df.values.astype(np.float32)
    y = y_df[ticker].values.astype(np.float32)

    # 3. Split
    n = len(X)
    t1 = int(n * 0.8)
    t2 = int(n * 0.9)
    X_train, X_val, X_test = X[:t1], X[t1:t2], X[t2:]
    y_train, y_val, y_test = y[:t1], y[t1:t2], y[t2:]

    # 4. Навчання
    model = XGBoostVolatilityModel()
    model.fit(X_train, y_train, X_val, y_val,
              feature_names=list(X_df.columns))

    # 5. Оцінка
    print("\n--- Test set ---")
    model.evaluate(X_test, y_test)

    # 6. Feature importance
    print("\n--- Топ-10 ознак ---")
    print(model.get_feature_importance(top_n=10))
