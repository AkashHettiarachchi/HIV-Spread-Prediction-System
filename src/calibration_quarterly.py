"""
Calibrate SICA model against the REAL QUARTERLY data (quarterly_data.py)
-- supersedes the earlier annual-only calibration.py, which is kept
for reference but should no longer be treated as primary.

Same non-identifiability caveat as before applies: only beta, rho,
alpha are calibrated; other SICA parameters remain fixed at
literature-informed values (see sica_model.py). With 39 quarterly
points (vs. 7 annual points before), the fit is considerably better
constrained, but this limitation should still be stated explicitly.
"""

import numpy as np
from scipy.optimize import least_squares

from sica_model import SICAParams, simulate_sica
from quarterly_data import get_quarterly_dataframe


CALIBRATION_BOUNDS = ([0.001, 0.001, 0.001], [2.0, 0.99, 0.99])


def initial_conditions_2008q1():
    """
    Approximate compartment split at 2008 Q1, using the PLHIV estimate
    as of end 2007 (per NSACP quarterly reports: 4,000 adults + 50
    children = ~4,050) and a rough ART coverage assumption for that
    period. ART rollout in Sri Lanka was minimal in 2007-2008 (national
    ART programme was newly established around this period), so a low
    ART-coverage assumption (~10-15%) is used here vs. the ~30%+ used
    for later periods in earlier versions of this calibration.
    This split is an APPROXIMATION -- state this limitation explicitly;
    a precise 2008 I/C/A breakdown was not available in the source
    reports.
    """
    plhiv_2008 = 4050
    C0 = 500    # on ART / chronic-controlled (low ART coverage assumption, ~2008)
    I0 = 2500   # undiagnosed / not yet linked to care
    A0 = max(plhiv_2008 - C0 - I0, 50)
    S0 = 16_000_000  # order-of-magnitude Sri Lanka adult population, ~2008
    return (S0, I0, C0, A0)


# Backward-compatible alias in case other scripts still reference the old name
initial_conditions_2016q1 = initial_conditions_2008q1


def sica_quarterly_predictions(params: SICAParams, y0, n_quarters: int):
    """Run SICA and return quarterly new-infection increments (length n_quarters)."""
    t, y = simulate_sica(params, y0, t_span=(0, n_quarters / 4),
                          n_points=n_quarters * 4 + 1)  # dense grid, 4x oversample per quarter
    cuminf = y[4]
    # sample cumulative infections at each quarter boundary
    quarter_times = np.arange(0, n_quarters + 1) / 4.0
    cuminf_at_quarters = np.interp(quarter_times, t, cuminf)
    quarterly_new = np.diff(cuminf_at_quarters)  # length n_quarters
    return quarterly_new


def residuals_fn(free_params, fixed: SICAParams, target_values, y0):
    beta, rho, alpha = free_params
    p = SICAParams(
        Lambda=fixed.Lambda, beta=beta, eta=fixed.eta, mu=fixed.mu,
        phi=fixed.phi, rho=rho, omega=fixed.omega, gamma=fixed.gamma,
        alpha=alpha, delta=fixed.delta,
    )
    predicted = sica_quarterly_predictions(p, y0, n_quarters=len(target_values))
    return predicted - target_values


def calibrate_quarterly(train_values=None):
    """Calibrate SICA against the full series by default, or against a supplied
    training slice when a strict train-only fit is required.

    Passing a train-only array keeps the app and validation pipeline leak-free,
    while the production forecast script continues to use the full dataset.
    """
    df = get_quarterly_dataframe()
    if train_values is None:
        target_values = df["new_cases"].values.astype(float)
        used_values = target_values
    else:
        target_values = np.asarray(train_values, dtype=float)
        used_values = target_values

    fixed = SICAParams()
    y0 = initial_conditions_2008q1()

    x0 = [fixed.beta, fixed.rho, fixed.alpha]
    result = least_squares(residuals_fn, x0, bounds=CALIBRATION_BOUNDS,
                            args=(fixed, used_values, y0))

    calibrated = SICAParams(
        Lambda=fixed.Lambda, beta=result.x[0], eta=fixed.eta, mu=fixed.mu,
        phi=fixed.phi, rho=result.x[1], omega=fixed.omega, gamma=fixed.gamma,
        alpha=result.x[2], delta=fixed.delta,
    )
    return calibrated, y0, result, df, target_values


if __name__ == "__main__":
    calibrated, y0, result, df, target_values = calibrate_quarterly()

    predicted = sica_quarterly_predictions(calibrated, y0, n_quarters=len(target_values))

    mae = np.mean(np.abs(predicted - target_values))
    mape = np.mean(np.abs((predicted - target_values) / target_values)) * 100
    rmse = np.sqrt(np.mean((predicted - target_values) ** 2))

    print("Calibration converged:", result.success, "| cost:", round(result.cost, 4))
    print(f"\nCalibrated: beta={calibrated.beta:.5f}  rho={calibrated.rho:.5f}  alpha={calibrated.alpha:.5f}")
    print(f"\nIn-sample fit (all {len(target_values)} quarters, NOT a held-out test):")
    print(f"  MAE:  {mae:.2f} cases/quarter")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: {rmse:.2f}")

    print(f"\n{'Year':<6}{'Q':<4}{'Real':<8}{'SICA':<10}")
    for i, row in df.iterrows():
        print(f"{row['year']:<6}{row['quarter']:<4}{row['new_cases']:<8}{predicted[i]:<10.1f}")