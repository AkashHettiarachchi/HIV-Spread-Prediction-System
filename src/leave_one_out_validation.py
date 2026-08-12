"""
Leave-One-Year-Out Cross-Validation of the calibrated SICA model.

Why this approach: we only have 7 real annual data points
(2019-2025). That is too few to do a conventional train/test split
(e.g. 80/20 would leave 1-2 test points, not statistically
meaningful). Leave-one-out CV is a well-established technique for
exactly this small-N situation: for each year, calibrate on the
other 6 years and predict the held-out year. This gives 7 honest
out-of-sample error estimates using only real data -- no synthetic
data involved anywhere in this script.

This is legitimate, defensible evidence of the SICA model's
forecasting accuracy that you can put directly in your dissertation's
evaluation section (4.6), independent of whatever happens with the
Bi-LSTM component.
"""

import numpy as np
from scipy.optimize import least_squares

from sica_model import SICAParams, simulate_sica, annual_new_infections
from quarterly_data import get_dataframe
from calibration_quarterly import initial_conditions_2019, residuals_fn


def leave_one_out_cv():
    df = get_dataframe()
    df_fit = df[df["year"] >= 2019].dropna(subset=["new_infections_spectrum"])
    years = df_fit["year"].values
    values = df_fit["new_infections_spectrum"].values

    fixed = SICAParams()
    y0 = initial_conditions_2019()

    results = []
    for i in range(len(years)):
        train_years = np.delete(years, i)
        train_values = np.delete(values, i)
        held_out_year = years[i]
        held_out_value = values[i]

        x0 = [fixed.beta, fixed.rho, fixed.alpha]
        bounds = ([0.01, 0.01, 0.01], [1.0, 0.9, 0.9])

        # Calibrate on the remaining years only
        res = least_squares(
            residuals_fn, x0, bounds=bounds,
            args=(fixed, train_years, train_values, y0),
        )
        calibrated = SICAParams(
            Lambda=fixed.Lambda, beta=res.x[0], eta=fixed.eta, mu=fixed.mu,
            phi=fixed.phi, rho=res.x[1], omega=fixed.omega, gamma=fixed.gamma,
            alpha=res.x[2], delta=fixed.delta,
        )

        # Predict across the full span, including the held-out year
        n_years = years[-1] - years[0] + 1
        t, y = simulate_sica(calibrated, y0, t_span=(0, n_years), n_points=n_years * 12 + 1)
        annual = dict(annual_new_infections(t, y[4], t_start_year=years[0]))
        predicted = annual.get(held_out_year, np.nan)

        results.append({
            "held_out_year": held_out_year,
            "actual": held_out_value,
            "predicted": predicted,
            "abs_error": abs(predicted - held_out_value),
            "pct_error": abs(predicted - held_out_value) / held_out_value * 100,
        })
    return results


if __name__ == "__main__":
    results = leave_one_out_cv()

    print("Leave-One-Year-Out Cross-Validation (real data only, no synthetic values)")
    print("=" * 70)
    print(f"{'Year':<8}{'Actual':<10}{'Predicted':<12}{'Abs Error':<12}{'% Error':<10}")
    for r in results:
        print(f"{r['held_out_year']:<8}{r['actual']:<10.0f}{r['predicted']:<12.1f}"
              f"{r['abs_error']:<12.2f}{r['pct_error']:<10.2f}")

    mape = np.mean([r["pct_error"] for r in results])
    mae = np.mean([r["abs_error"] for r in results])
    print("=" * 70)
    print(f"Mean Absolute Error (MAE):      {mae:.2f} cases/year")
    print(f"Mean Absolute % Error (MAPE):   {mape:.2f}%")
    print("\nUse these honest, real-data-only numbers as your SICA-alone baseline")
    print("in the model comparison table -- they replace the fabricated")
    print("'52.1 cases / -72%' figures that were hardcoded in the old dashboard.")