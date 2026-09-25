"""
Real Bi-LSTM Residual Correction, trained on REAL quarterly data
(39 quarters, 2016 Q1 - 2025 Q3). This is the first version of this
codebase where the Bi-LSTM is actually trained on genuine data rather
than synthetic placeholders or being blocked by data scarcity.

Design:
    residual(t) = real_quarterly_cases(t) - sica_quarterly_baseline(t)
    Bi-LSTM learns to predict residual(t) from residual(t-w..t-1)
    hybrid_forecast(t) = sica_baseline(t) + bilstm_predicted_residual(t)

Chronological train/test split (NEVER shuffle time series data):
    Train: 2016 Q1 - ~2023 Q4 (about 32 quarters)
    Test:  ~2024 Q1 - 2025 Q3 (about 7 quarters) -- genuine out-of-
           sample forecast evaluation on the most recent real data.
"""

import numpy as np

from sica_model import SICAParams
from quarterly_data import get_quarterly_dataframe
from calibration_quarterly import calibrate_quarterly, sica_quarterly_predictions


WINDOW_SIZE = 4  # one year of lookback (4 quarters)
TEST_QUARTERS = 7  # roughly the last ~1.75 years held out


def build_bilstm_model(window_size: int, units: int = 16, dropout: float = 0.2):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(window_size, 1)),
        layers.Bidirectional(layers.LSTM(units)),
        layers.Dropout(dropout),
        layers.Dense(8, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-3), loss="mse", metrics=["mae"])
    return model


def create_sequences(series: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X).reshape(-1, window_size, 1), np.array(y)


def normalize(series, ref_min=None, ref_max=None):
    ref_min = series.min() if ref_min is None else ref_min
    ref_max = series.max() if ref_max is None else ref_max
    if ref_max - ref_min == 0:
        return series * 0.0, (ref_min, ref_max)
    return (series - ref_min) / (ref_max - ref_min), (ref_min, ref_max)


def denormalize(scaled, min_max):
    ref_min, ref_max = min_max
    return scaled * (ref_max - ref_min) + ref_min


def run():
    df = get_quarterly_dataframe()
    real_values = df["new_cases"].values.astype(float)

    n_test = TEST_QUARTERS
    n_train_raw = len(real_values) - n_test
    train_values = real_values[:n_train_raw]

    # Strict train-only calibration: no leakage from the held-out test period.
    calibrated_params, y0, calib_result, _, _ = calibrate_quarterly(train_values=train_values)
    sica_baseline = sica_quarterly_predictions(calibrated_params, y0, n_quarters=len(real_values))

    residual = real_values - sica_baseline

    # 2. Chronological train/test split BEFORE windowing, so no
    #    test-period information leaks into training sequences.
    train_residual_raw = residual[:n_train_raw]
    # include WINDOW_SIZE points of context before the test period so
    # the first test window has real lookback data (not synthetic)
    test_residual_raw = residual[n_train_raw - WINDOW_SIZE:]

    # 3. Normalize using TRAIN stats only (avoid test-set leakage)
    train_scaled, min_max = normalize(train_residual_raw)
    test_scaled, _ = normalize(test_residual_raw, *min_max)

    X_train, y_train = create_sequences(train_scaled, WINDOW_SIZE)
    X_test, y_test = create_sequences(test_scaled, WINDOW_SIZE)

    print(f"Total quarters: {len(real_values)} | Train sequences: {len(X_train)} | "
          f"Test sequences: {len(X_test)}")

    # 4. Train
    from tensorflow import keras
    model = build_bilstm_model(WINDOW_SIZE)
    early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=25, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=300, batch_size=4, callbacks=[early_stop], verbose=0)

    # 5. Predict on held-out real test quarters
    pred_scaled = model.predict(X_test, verbose=0).flatten()
    pred_residual = denormalize(pred_scaled, min_max)

    sica_test = sica_baseline[n_train_raw:]
    real_test = real_values[n_train_raw:]
    hybrid_test = sica_test + pred_residual

    # 6. Honest out-of-sample metrics -- real data only, real test period
    def metrics(y_true, y_pred, name):
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print(f"{name:<25} MAE={mae:6.2f}  RMSE={rmse:6.2f}  MAPE={mape:6.2f}%")
        return mae, rmse, mape

    print(f"\n--- Held-out test quarters ({n_test} quarters, real data only) ---")
    test_period = df.iloc[n_train_raw:][["year", "quarter"]].values
    for i, (yr, q) in enumerate(test_period):
        print(f"  {yr} Q{q}:  Real={real_test[i]:.0f}  SICA-only={sica_test[i]:.1f}  "
              f"Hybrid={hybrid_test[i]:.1f}")

    print()
    metrics(real_test, sica_test, "SICA-only baseline:")
    metrics(real_test, hybrid_test, "Hybrid SICA+Bi-LSTM:")

    return {
        "df": df, "sica_baseline": sica_baseline, "real_values": real_values,
        "n_train_raw": n_train_raw, "hybrid_test": hybrid_test,
        "sica_test": sica_test, "real_test": real_test, "model": model,
    }


if __name__ == "__main__":
    results = run()