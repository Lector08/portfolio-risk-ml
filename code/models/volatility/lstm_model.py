# =============================================================================
# models/volatility/lstm_model.py — LSTM прогноз волатильності
# MILESTONE 2.2
# Архітектура: Input → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(1)
# =============================================================================

import os
import numpy as np


class LSTMVolatilityModel:
    """
    LSTM-мережа для прогнозу реалізованої волатильності.

    Вхід:  3D тензор (samples, seq_len=60, n_features)
    Вихід: sigma_hat_{t+1} — прогноз волатильності на наступний день

    Keras завантажується лише при виклику build_model(),
    щоб не гальмувати імпорт у модулях які не використовують LSTM.
    """

    def __init__(self, seq_len: int, n_features: int):
        """
        Args:
            seq_len:    розмір вхідного вікна (кількість timesteps), default=60
            n_features: кількість ознак на кожному timestep
        """
        self.seq_len    = seq_len
        self.n_features = n_features
        self.model      = None
        self.history    = None

    # ------------------------------------------------------------------
    # Побудова архітектури
    # ------------------------------------------------------------------

    def build_model(
        self,
        units: list[int] = [128, 64],
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
    ) -> "LSTMVolatilityModel":
        """
        Будує Keras Sequential модель.

        Архітектура:
            LSTM(units[0], return_sequences=True)   ← запам'ятовує всю послідовність
            Dropout(dropout)
            LSTM(units[1], return_sequences=False)  ← повертає лише останній стан
            Dropout(dropout)
            Dense(1, activation='relu')             ← relu: волатильність >= 0

        Args:
            units:         кількість нейронів у кожному LSTM шарі
            dropout:       ймовірність dropout (регуляризація)
            learning_rate: крок навчання для Adam

        Returns:
            self
        """
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.optimizers.schedules import CosineDecay

        tf.get_logger().setLevel("ERROR")

        # Cosine annealing: LR поступово зменшується
        lr_schedule = CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            alpha=1e-5,
        )

        self.model = Sequential([
            Input(shape=(self.seq_len, self.n_features)),
            LSTM(units[0], return_sequences=True,
                 kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            Dropout(dropout),
            LSTM(units[1], return_sequences=False,
                 kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            Dropout(dropout),
            Dense(1, activation="relu"),   # волатильність невід'ємна
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss="mse",
            metrics=["mae"],
        )

        total_params = self.model.count_params()
        print(f"[LSTM] Модель побудована: {total_params:,} параметрів")
        self.model.summary(print_fn=lambda x: None)   # тихо
        return self

    # ------------------------------------------------------------------
    # Навчання
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        epochs:     int = 50,
        batch_size: int = 32,
        save_path:  str = None,
    ) -> "LSTMVolatilityModel":
        """
        Навчає LSTM з EarlyStopping та ModelCheckpoint.

        Callbacks:
            EarlyStopping: зупиняє навчання якщо val_loss не покращується 10 епох
            ReduceLROnPlateau: знижує LR у 2× якщо val_loss стоїть 5 епох
            ModelCheckpoint: зберігає найкращу модель (за val_loss)

        Args:
            X_train, y_train: тренувальні дані
            X_val,   y_val:   валідаційні дані
            epochs:           максимальна кількість епох
            batch_size:       розмір батчу
            save_path:        шлях для збереження найкращої моделі (опційно)

        Returns:
            self
        """
        from tensorflow.keras.callbacks import (
            EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        )

        if self.model is None:
            self.build_model()

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0,
            ),
        ]

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    filepath=save_path,
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=0,
                )
            )

        print(f"[LSTM] Навчання: epochs={epochs}, batch={batch_size}, "
              f"train={len(X_train)}, val={len(X_val)}")

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        best_epoch = np.argmin(self.history.history["val_loss"]) + 1
        best_val   = min(self.history.history["val_loss"])
        print(f"[LSTM] Завершено: best_epoch={best_epoch}, best_val_loss={best_val:.6f}")
        return self

    # ------------------------------------------------------------------
    # Прогноз та оцінка
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: 3D тензор (samples, seq_len, n_features)

        Returns:
            np.ndarray: sigma_hat (samples,)
        """
        if self.model is None:
            raise RuntimeError("Модель не навчена. Спочатку викличте fit()")
        return self.model.predict(X, verbose=0).flatten()

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """Метрики якості: RMSE, MAE, QLIKE"""
        y_pred = self.predict(X)

        rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae   = float(np.mean(np.abs(y_true - y_pred)))
        eps   = 1e-8
        qlike = float(np.mean(np.log(y_pred**2 + eps) + y_true**2 / (y_pred**2 + eps)))

        metrics = {"rmse": rmse, "mae": mae, "qlike": qlike}
        print(f"[LSTM] Метрики → RMSE={rmse:.6f} | MAE={mae:.6f} | QLIKE={qlike:.4f}")
        return metrics

    def get_learning_curves(self) -> dict:
        """Повертає train/val loss по епохах для побудови графіку"""
        if self.history is None:
            return {}
        return {
            "train_loss": self.history.history["loss"],
            "val_loss":   self.history.history["val_loss"],
            "train_mae":  self.history.history.get("mae", []),
            "val_mae":    self.history.history.get("val_mae", []),
        }

    # ------------------------------------------------------------------
    # Збереження / завантаження
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Зберігає модель у форматі .keras"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"[LSTM] Модель збережено → {path}")

    def load(self, path: str) -> "LSTMVolatilityModel":
        """Завантажує модель з файлу"""
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
        print(f"[LSTM] Модель завантажено ← {path}")
        return self


# ---------------------------------------------------------------------------
# Швидкий тест: python models/volatility/lstm_model.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

    from data.loader import get_prices
    from data.preprocessor import (compute_log_returns, compute_realized_volatility)
    from data.feature_engineering import build_feature_matrix, create_lstm_sequences
    from sklearn.preprocessing import StandardScaler

    # 1. Дані
    prices  = get_prices()
    returns = compute_log_returns(prices)
    rv      = compute_realized_volatility(returns)

    X_df = build_feature_matrix(prices, returns)
    common_idx = X_df.index.intersection(rv.index)
    X_df = X_df.loc[common_idx]
    y_df = rv.loc[common_idx]["AAPL"]

    # 2. Масштабування ознак
    X_raw = X_df.values.astype(np.float32)
    y_raw = y_df.values.astype(np.float32)

    n  = len(X_raw)
    t1, t2 = int(n * 0.8), int(n * 0.9)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 3. LSTM sequences
    SEQ = 60
    X_seq, y_seq = create_lstm_sequences(X_scaled, y_raw, seq_len=SEQ)

    # Хронологічне розбиття після формування sequences
    X_tr, X_vl, X_te = X_seq[:t1-SEQ], X_seq[t1-SEQ:t2-SEQ], X_seq[t2-SEQ:]
    y_tr, y_vl, y_te = y_seq[:t1-SEQ], y_seq[t1-SEQ:t2-SEQ], y_seq[t2-SEQ:]

    print(f"\nShapes: train={X_tr.shape}, val={X_vl.shape}, test={X_te.shape}")

    # 4. Навчання
    lstm = LSTMVolatilityModel(seq_len=SEQ, n_features=X_scaled.shape[1])
    lstm.build_model()
    lstm.fit(X_tr, y_tr, X_vl, y_vl, epochs=10, batch_size=32)  # 10 епох для тесту

    # 5. Оцінка
    print("\n--- Test set ---")
    lstm.evaluate(X_te, y_te)
