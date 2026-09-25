import numpy as np

from sica_model import SICAParams, simulate_sica
from quarterly_data import get_quarterly_dataframe
from calibration_quarterly import calibrate_quarterly, sica_quarterly_predictions
from hybrid_pipeline_quarterly import build_bilstm_model, create_sequences, normalize, denormalize, WINDOW_SIZE


def forecast_future(n_future_quarters: int = 20, start_year: int = 2026, start_quarter: int = 1):
    """
    n_future_quarters: how many quarters to forecast forward.
                        Default 20 = 5 years (2026 Q1 - 2030 Q4).
    """
    df = get_quarterly_dataframe()
    real_values = df["new_cases"].values.astype(float)
    n_real = len(real_values)

    # --- 1. SICA baseline, extended all the way to the end of the forecast horizon ---
    calibrated_params, y0, calib_result, _, _ = calibrate_quarterly()
    total_quarters = n_real + n_future_quarters
    sica_full = sica_quarterly_predictions(calibrated_params, y0, n_quarters=total_quarters)
    sica_historical = sica_full[:n_real]
    sica_future = sica_full[n_real:]

    # --- 2. Train Bi-LSTM on ALL real residuals (no held-out split --
    #        this is the deployed/production model, not the validation one) ---
    residual_historical = real_values - sica_historical
    scaled_residual, min_max = normalize(residual_historical)
    X_all, y_all = create_sequences(scaled_residual, WINDOW_SIZE)

    from tensorflow import keras
    model = build_bilstm_model(WINDOW_SIZE)
    early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=25, restore_best_weights=True)
    model.fit(X_all, y_all, epochs=300, batch_size=4, callbacks=[early_stop], verbose=0)

    # Estimate training fit RMSE to quantify residual uncertainty.
    train_pred = model.predict(X_all, verbose=0).reshape(-1)
    residual_fit_error = y_all - train_pred
    sigma_e = float(np.sqrt(np.mean(residual_fit_error ** 2)))

    # --- 3. Autoregressive rollout for future residuals ---
    window = list(scaled_residual[-WINDOW_SIZE:])  # last real window, scaled
    future_residuals_scaled = []
    for _ in range(n_future_quarters):
        x_input = np.array(window[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 1)
        next_scaled = model.predict(x_input, verbose=0)[0, 0]
        future_residuals_scaled.append(next_scaled)
        window.append(next_scaled)

    future_residuals = denormalize(np.array(future_residuals_scaled), min_max)
    hybrid_future = sica_future + future_residuals
    hybrid_future = np.clip(hybrid_future, a_min=0, a_max=None)  # cases can't be negative

    # 95% confidence intervals for the forecast horizon.
    alpha = 0.05
    horizon_idx = np.arange(1, n_future_quarters + 1, dtype=float)
    se = sigma_e * np.sqrt(1.0 + alpha * (horizon_idx - 1.0))
    hybrid_lower = np.maximum(0.0, hybrid_future - 1.96 * se)
    hybrid_upper = hybrid_future + 1.96 * se

    # --- 4. Build a labeled future quarters table ---
    future_labels = []
    yr, q = start_year, start_quarter
    for _ in range(n_future_quarters):
        future_labels.append((yr, q))
        q += 1
        if q > 4:
            q = 1
            yr += 1

    return {
        "df_historical": df,
        "real_historical": real_values,
        "sica_historical": sica_historical,
        "sica_future": sica_future,
        "hybrid_future": hybrid_future,
        "hybrid_lower": hybrid_lower,
        "hybrid_upper": hybrid_upper,
        "future_labels": future_labels,
        "calibrated_params": calibrated_params,
        "sigma_e": sigma_e,
    }


if __name__ == "__main__":
    results = forecast_future(n_future_quarters=20)

    print("HIV New Case Forecasts: 2026 Q1 - 2030 Q4")
    print("=" * 55)
    print(f"{'Year':<8}{'Q':<4}{'SICA-only':<14}{'Hybrid Forecast':<18}{'95% CI':<20}")
    for (yr, q), sica_val, hybrid_val, lower_val, upper_val in zip(
        results["future_labels"],
        results["sica_future"],
        results["hybrid_future"],
        results["hybrid_lower"],
        results["hybrid_upper"],
    ):
        print(f"{yr:<8}{q:<4}{sica_val:<14.1f}{hybrid_val:<18.1f}{lower_val:.0f} - {upper_val:.0f}")

    print("=" * 55)
    annual = {}
    for (yr, q), hybrid_val in zip(results["future_labels"], results["hybrid_future"]):
        annual[yr] = annual.get(yr, 0) + hybrid_val
    print("\nAnnual totals (hybrid forecast):")
    for yr, total in annual.items():
        print(f"  {yr}: {total:.0f} new HIV cases (forecast)")

    print(f"\nLast real data point: {results['df_historical'].iloc[-1]['year']:.0f} "
          f"Q{results['df_historical'].iloc[-1]['quarter']:.0f} "
          f"= {results['real_historical'][-1]:.0f} cases")