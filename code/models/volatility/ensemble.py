# =============================================================================
# models/volatility/ensemble.py — Ансамблева модель LSTM + XGBoost + GARCH
# MILESTONE 2.3
# sigma_ensemble = w_lstm * sigma_lstm + w_xgb * sigma_xgb
# Ваги: inverse-MSE weighting на validation set
# =============================================================================

import numpy as np


class VolatilityEnsemble:
    """
    Зважений ансамбль двох ML-моделей прогнозу волатильності.

    Стратегія: inverse-error weighting
        w_i = (1 / MSE_i) / sum(1 / MSE_j)
        Модель з меншою помилкою отримує більшу вагу.

    Також реалізує GARCH(1,1) як baseline для порівняння.
    """

    def __init__(self, lstm_model, xgb_model):
        """
        Args:
            lstm_model: навчений LSTMVolatilityModel
            xgb_model:  навчений XGBoostVolatilityModel
        """
        self.lstm    = lstm_model
        self.xgb     = xgb_model
        self.weights = None      # [w_lstm, w_xgb] після fit_weights()
        self.garch_model = None  # GARCH baseline

    # ------------------------------------------------------------------
    # Підбір ваг
    # ------------------------------------------------------------------

    def fit_weights(
        self,
        X_val_lstm: np.ndarray,
        X_val_xgb:  np.ndarray,
        y_val:      np.ndarray,
    ) -> "VolatilityEnsemble":
        """
        Визначає ваги ансамблю на основі validation MSE.

        Формула:
            MSE_i = mean((y_val - y_hat_i)^2)
            w_i   = (1/MSE_i) / (1/MSE_lstm + 1/MSE_xgb)

        Args:
            X_val_lstm: валідаційні дані для LSTM (3D)
            X_val_xgb:  валідаційні дані для XGBoost (2D)
            y_val:      справжні значення волатильності

        Returns:
            self
        """
        lstm_pred = self.lstm.predict(X_val_lstm)
        xgb_pred  = self.xgb.predict(X_val_xgb)

        mse_lstm = np.mean((y_val - lstm_pred) ** 2)
        mse_xgb  = np.mean((y_val - xgb_pred) ** 2)

        inv_sum  = 1/mse_lstm + 1/mse_xgb
        w_lstm   = (1/mse_lstm) / inv_sum
        w_xgb    = (1/mse_xgb)  / inv_sum

        self.weights = [w_lstm, w_xgb]

        print(f"[Ensemble] Ваги: LSTM={w_lstm:.3f}, XGBoost={w_xgb:.3f}")
        print(f"[Ensemble] Val MSE: LSTM={mse_lstm:.6f}, XGBoost={mse_xgb:.6f}")
        return self

    # ------------------------------------------------------------------
    # Прогноз
    # ------------------------------------------------------------------

    def predict(
        self,
        X_lstm: np.ndarray,
        X_xgb:  np.ndarray,
    ) -> np.ndarray:
        """
        Зважений прогноз ансамблю.

        Args:
            X_lstm: 3D масив для LSTM (samples, seq_len, features)
            X_xgb:  2D масив для XGBoost (samples, features)

        Returns:
            np.ndarray: sigma_ensemble (samples,)
        """
        if self.weights is None:
            raise RuntimeError("Спочатку викличте fit_weights()")

        w_lstm, w_xgb = self.weights
        lstm_pred = self.lstm.predict(X_lstm)
        xgb_pred  = self.xgb.predict(X_xgb)

        # Вирівнюємо довжини (LSTM sequences коротші через seq_len offset)
        min_len   = min(len(lstm_pred), len(xgb_pred))
        lstm_pred = lstm_pred[-min_len:]
        xgb_pred  = xgb_pred[-min_len:]

        ensemble  = w_lstm * lstm_pred + w_xgb * xgb_pred
        return np.clip(ensemble, a_min=0, a_max=None)

    def evaluate(
        self,
        X_lstm: np.ndarray,
        X_xgb:  np.ndarray,
        y_true: np.ndarray,
    ) -> dict:
        """Метрики ансамблю: RMSE, MAE, QLIKE"""
        y_pred = self.predict(X_lstm, X_xgb)
        y_true = y_true[-len(y_pred):]   # вирівнювання

        rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae   = float(np.mean(np.abs(y_true - y_pred)))
        eps   = 1e-8
        qlike = float(np.mean(np.log(y_pred**2 + eps) + y_true**2 / (y_pred**2 + eps)))

        metrics = {"rmse": rmse, "mae": mae, "qlike": qlike}
        print(f"[Ensemble] Метрики → RMSE={rmse:.6f} | MAE={mae:.6f} | QLIKE={qlike:.4f}")
        return metrics

    # ------------------------------------------------------------------
    # GARCH(1,1) baseline
    # ------------------------------------------------------------------

    def fit_garch(self, returns: np.ndarray) -> "VolatilityEnsemble":
        """
        Навчає GARCH(1,1) baseline на тренувальних доходностях.

        Модель: sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}

        Args:
            returns: 1D масив лог-доходностей (тренувальна вибірка)

        Returns:
            self
        """
        try:
            from arch import arch_model
        except ImportError:
            print("[Ensemble] arch не встановлено. pip3 install arch")
            return self

        am = arch_model(returns * 100, vol="Garch", p=1, q=1,
                        dist="normal", rescale=False)
        self.garch_model = am.fit(disp="off")
        print(f"[Ensemble] GARCH(1,1) навчено: "
              f"omega={self.garch_model.params['omega']:.4f}, "
              f"alpha={self.garch_model.params['alpha[1]']:.4f}, "
              f"beta={self.garch_model.params['beta[1]']:.4f}")
        return self

    def predict_garch(self, horizon: int = 1) -> np.ndarray:
        """
        Прогноз волатильності GARCH на horizon кроків вперед.

        Returns:
            np.ndarray: sigma_garch (horizon,) — денна волатильність
        """
        if self.garch_model is None:
            raise RuntimeError("Спочатку викличте fit_garch()")

        forecast = self.garch_model.forecast(horizon=horizon, reindex=False)
        variance = forecast.variance.values[-1]     # остання точка прогнозу
        sigma    = np.sqrt(variance) / 100           # повертаємо масштаб
        return sigma

    # ------------------------------------------------------------------
    # Порівняльна таблиця всіх моделей
    # ------------------------------------------------------------------

    def compare_models(
        self,
        X_lstm: np.ndarray,
        X_xgb:  np.ndarray,
        y_true: np.ndarray,
    ) -> dict:
        """
        Порівнює метрики LSTM, XGBoost, Ensemble на тестових даних.

        Returns:
            dict: {'lstm': {...}, 'xgboost': {...}, 'ensemble': {...}}
        """
        min_len = min(
            len(self.lstm.predict(X_lstm)),
            len(self.xgb.predict(X_xgb)),
            len(y_true),
        )

        results = {}
        for name, model, X in [
            ("lstm",     self.lstm, X_lstm),
            ("xgboost",  self.xgb,  X_xgb),
        ]:
            pred = model.predict(X)[-min_len:]
            y    = y_true[-min_len:]
            rmse  = float(np.sqrt(np.mean((y - pred)**2)))
            mae   = float(np.mean(np.abs(y - pred)))
            eps   = 1e-8
            qlike = float(np.mean(np.log(pred**2+eps) + y**2/(pred**2+eps)))
            results[name] = {"rmse": rmse, "mae": mae, "qlike": qlike}

        ens_pred = self.predict(X_lstm, X_xgb)[-min_len:]
        y        = y_true[-min_len:]
        rmse  = float(np.sqrt(np.mean((y - ens_pred)**2)))
        mae   = float(np.mean(np.abs(y - ens_pred)))
        qlike = float(np.mean(np.log(ens_pred**2+1e-8) + y**2/(ens_pred**2+1e-8)))
        results["ensemble"] = {"rmse": rmse, "mae": mae, "qlike": qlike}

        print("\n[Ensemble] Порівняння моделей:")
        print(f"{'Модель':<12} {'RMSE':>10} {'MAE':>10} {'QLIKE':>10}")
        print("-" * 45)
        for name, m in results.items():
            print(f"{name:<12} {m['rmse']:>10.6f} {m['mae']:>10.6f} {m['qlike']:>10.4f}")

        return results
