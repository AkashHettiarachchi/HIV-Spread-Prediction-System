"""
Bi-LSTM Residual Correction Model.

STATUS: Architecture ready, but NOT yet trainable in a statistically
meaningful way -- we currently only have ~7-10 REAL annual data
points (see real_data.py), which is far too few to window into
train/test sequences for a neural network.

This module is intentionally kept separate and import-safe (it does
NOT run any training on import) so the rest of the codebase --
sica_model.py, calibration.py, leave_one_out_validation.py -- can be
used, tested, and presented to your supervisor RIGHT NOW using only
real, defensible data, while the NSACP data request is pending.

Once monthly or quarterly data arrives from NSACP (or another
verified source), plug it into `data_utils.load_real_series()` below
and re-run this module -- no other files need to change.
"""

import numpy as np


def create_sequences(series: np.ndarray, window_size: int):
    """Turn a 1-D series into (X, y) supervised windows for the Bi-LSTM."""
    if len(series) <= window_size:
        raise ValueError(
            f"Series has only {len(series)} points, need > {window_size} "
            f"(window_size) to create even a single training sequence. "
            f"This is exactly the real-data-scarcity issue -- see module docstring."
        )
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X).reshape(-1, window_size, 1), np.array(y)


def train_test_split_chronological(X, y, train_ratio: float = 0.8):
    """Chronological split -- never shuffle time series data."""
    split_idx = max(1, int(len(X) * train_ratio))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def build_bilstm_model(window_size: int, units: int = 16, dropout: float = 0.2):
    """
    Small Bi-LSTM by design -- with limited real data, a large network
    (64-128 units, as in the earlier prototype) will overfit almost
    immediately. Keep it small until dataset size justifies scaling up.
    Requires tensorflow (not installed in this sandbox -- run locally).
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(window_size, 1)),
        layers.Bidirectional(layers.LSTM(units)),
        layers.Dropout(dropout),
        layers.Dense(8, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


def train_bilstm(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=4, patience=20):
    from tensorflow import keras
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                                restore_best_weights=True)
    return model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=1)


def normalize(series: np.ndarray):
    s_min, s_max = series.min(), series.max()
    if s_max - s_min == 0:
        return series * 0.0, (s_min, s_max)
    return (series - s_min) / (s_max - s_min), (s_min, s_max)


def denormalize(scaled: np.ndarray, min_max):
    s_min, s_max = min_max
    return scaled * (s_max - s_min) + s_min


if __name__ == "__main__":
    from quarterly_data import get_dataframe

    df = get_dataframe().dropna(subset=["new_infections_spectrum"])
    series = df["new_infections_spectrum"].values.astype(float)

    print(f"Current real data points available: {len(series)}")
    try:
        X, y = create_sequences(series, window_size=3)
        print(f"Sequences created: {len(X)} -- likely still too few for reliable training.")
    except ValueError as e:
        print(f"Cannot train yet: {e}")
        print("\nNext step: once NSACP data request returns monthly/quarterly figures,"
              " update real_data.py with the new series and re-run this module.")
