"""
Scenario Simulation Engine: Policy "What-If" Analysis for 2026-2027
==================================================================
Backend logic for the Scenario Simulation & Policy Lab page. This module
does NOT modify any existing project file -- it only imports and reuses
calibration_quarterly.py, sica_model.py, hybrid_pipeline_quarterly.py, and
forecast_future.py exactly as they already exist.

Three policy levers, applied on top of the existing calibrated SICA model
and the existing Bi-LSTM's own 2026-2027 residual forecast:

    Lever 1 -- Transmission Risk Reduction (0-50%, default 10%)
        beta_sim = beta_calibrated * (1 - lever1 / 100)

    Lever 2 -- Surveillance & Testing Expansion (0-50%, default 15%)
        E_simulated = E_BiLSTM * (1 + lever2 / 100)
        (More testing surfaces more existing undiagnosed cases, so this
         can INCREASE reported/detected incidence even while the
         underlying epidemic shrinks -- this is intentional and is
         surfaced explicitly in the UI, not a bug.)

    Lever 3 -- ART Treatment Acceleration (0-50%, default 15%)
        alpha_sim = alpha_calibrated * (1 + lever3 / 100)

Modelling simplification (documented, not hidden): the adjusted SICA
parameters are applied across the model's full solved trajectory (the
same single continuous-ODE-solve convention already used by
forecast_future.py, which does not split the integration at the
present day either). The 2026-2027 quarters are then read off that
trajectory. This assumes the adjusted transmission/treatment dynamic
characterizes the model consistently for projection purposes -- a
standard simplification for compartmental scenario dashboards, and
consistent with how the rest of this project already computes its
baseline forecast.

Quarter-labeling note: this reuses forecast_future.py's own labeling
convention verbatim (its first forecast quarter is labelled "2026 Q1"
immediately after the last real quarter, 2025 Q3) so that quarter
labels here always match the existing Forecast Horizon page exactly.

--------------------------------------------------------------------
CACHING (new): forecast_future() re-fits SICA (a least_squares solve)
AND retrains a fresh, unseeded Bi-LSTM (up to 300 epochs) every time
it's called. Calling it fresh inside compute_scenario_forecast() -- i.e.
on every slider move -- causes two problems:

  1. Speed: each scenario would take as long as a full model retrain,
     making the page feel frozen on every interaction.
  2. Stability: with no random seed, the "baseline" forecast changes
     slightly every time it's recomputed. Since this page reports the
     gap between baseline and simulated totals as the policy effect,
     an unstable baseline mixes real intervention impact with random
     retraining noise.

_baseline_forecast_cached() and _calibration_cached() fix both: the
expensive baseline forecast and the SICA calibration each run once per
process (seeded, for the Bi-LSTM) and are reused for every lever
combination. Only the cheap part -- re-solving the SICA ODE with the
adjusted beta/alpha, and rescaling the residual -- re-runs per call.
Call reset_baseline_cache() to force a fresh baseline and calibration
(e.g. after new quarterly data is added).
"""

from dataclasses import replace
from functools import lru_cache

import numpy as np

from quarterly_data import get_quarterly_dataframe
from calibration_quarterly import calibrate_quarterly, sica_quarterly_predictions
from forecast_future import forecast_future

N_FUTURE_QUARTERS_NEEDED = 8  # covers labelled 2026 Q1 - 2027 Q4; 2027 is the last 4 of these


@lru_cache(maxsize=1)
def _baseline_forecast_cached():
    """The existing, unmodified hybrid pipeline's 2026-2027 forecast -- computed
    once per process and reused for every scenario. See module docstring."""
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(42)  # makes the cached baseline reproducible
    except Exception:
        np.random.seed(42)

    baseline_future = forecast_future(n_future_quarters=N_FUTURE_QUARTERS_NEEDED)
    future_labels = list(baseline_future["future_labels"])
    sica_baseline_all = np.asarray(baseline_future["sica_future"])
    hybrid_baseline_all = np.asarray(baseline_future["hybrid_future"])

    future_labels = future_labels[:N_FUTURE_QUARTERS_NEEDED]
    forecast_indices = list(range(len(future_labels)))
    return {
        "future_labels": future_labels,
        "forecast_indices": forecast_indices,
        "sica_baseline": sica_baseline_all[forecast_indices],
        "hybrid_baseline": hybrid_baseline_all[forecast_indices],
    }


@lru_cache(maxsize=1)
def _calibration_cached():
    """calibrate_quarterly() also runs a least_squares fit -- cache it too, so
    a scenario run only re-solves the (cheap) SICA ODE with adjusted
    parameters rather than re-fitting beta/rho/alpha from scratch."""
    calibrated_params, y0, _calib_result, _, _ = calibrate_quarterly()
    return calibrated_params, y0


def reset_baseline_cache():
    """Force the next compute_scenario_forecast() call to recompute the baseline
    forecast and SICA calibration from scratch (e.g. after the quarterly
    dataset is updated with new NSACP reports)."""
    _baseline_forecast_cached.cache_clear()
    _calibration_cached.cache_clear()


def compute_scenario_forecast(lever1_pct: float, lever2_pct: float, lever3_pct: float):
    """
    Runs the baseline hybrid pipeline's own 2026-2027 projection (unmodified,
    cached -- see module docstring) alongside a policy-adjusted SICA +
    scaled-residual projection for the same eight quarters, using the three
    sliders described above.

    Returns a dict with baseline and simulated per-quarter values for
    2026 Q1-2027 Q4, yearly and two-year totals, and the recent
    real history (2022 onward) for charting context.
    """
    df = get_quarterly_dataframe()

    # ---- 1. Baseline: the existing, unmodified forecast pipeline (cached) ----
    baseline = _baseline_forecast_cached()
    future_labels = baseline["future_labels"][:N_FUTURE_QUARTERS_NEEDED]
    forecast_indices = baseline["forecast_indices"]
    sica_baseline = baseline["sica_baseline"]
    hybrid_baseline = baseline["hybrid_baseline"]
    # E_BiLSTM: the Bi-LSTM's own residual correction, backed out
    # from the existing pipeline's own baseline outputs (no re-derivation).
    residual_baseline = hybrid_baseline - sica_baseline

    # ---- 2. Simulated SICA trajectory with policy-adjusted parameters ----
    calibrated_params, y0 = _calibration_cached()
    beta_sim = calibrated_params.beta * (1 - lever1_pct / 100.0)
    alpha_sim = calibrated_params.alpha * (1 + lever3_pct / 100.0)
    sim_params = replace(calibrated_params, beta=beta_sim, alpha=alpha_sim)

    n_real = len(df)
    sica_sim_all = sica_quarterly_predictions(sim_params, y0, n_quarters=n_real + N_FUTURE_QUARTERS_NEEDED)
    sica_sim_future = sica_sim_all[n_real:]  # same slicing convention as forecast_future.py
    sica_sim = sica_sim_future[forecast_indices]

    # ---- 3. Simulated residual (surveillance/testing lever) ----
    residual_sim = residual_baseline * (1 + lever2_pct / 100.0)

    # ---- 4. Simulated hybrid forecast ----
    hybrid_sim = sica_sim + residual_sim
    hybrid_sim = np.clip(hybrid_sim, a_min=0, a_max=None)

    mask_2026 = np.asarray([year == 2026 for year, _quarter in future_labels])
    mask_2027 = np.asarray([year == 2027 for year, _quarter in future_labels])

    def totals(mask):
        baseline_total = float(np.sum(hybrid_baseline[mask]))
        simulated_total = float(np.sum(hybrid_sim[mask]))
        return baseline_total, simulated_total, baseline_total - simulated_total

    baseline_total_2026, simulated_total_2026, delta_total_2026 = totals(mask_2026)
    baseline_total_2027, simulated_total_2027, delta_total_2027 = totals(mask_2027)
    baseline_total_2_year, simulated_total_2_year, delta_total_2_year = totals(mask_2026 | mask_2027)

    # ---- 5. Recent real history for chart context ----
    recent_history = df[df["year"] >= 2022][["year", "quarter", "new_cases"]].copy()

    return {
        "future_labels": future_labels,
        "sica_baseline": sica_baseline,
        "hybrid_baseline": hybrid_baseline,
        "residual_baseline": residual_baseline,
        "sica_sim": sica_sim,
        "residual_sim": residual_sim,
        "hybrid_sim": hybrid_sim,
        "beta_calibrated": calibrated_params.beta,
        "beta_sim": beta_sim,
        "alpha_calibrated": calibrated_params.alpha,
        "alpha_sim": alpha_sim,
        "recent_history": recent_history,
        "lever1_pct": lever1_pct,
        "lever2_pct": lever2_pct,
        "lever3_pct": lever3_pct,
        "baseline_total_2026": baseline_total_2026,
        "simulated_total_2026": simulated_total_2026,
        "delta_total_2026": delta_total_2026,
        "baseline_total_2027": baseline_total_2027,
        "simulated_total_2027": simulated_total_2027,
        "delta_total_2027": delta_total_2027,
        "baseline_total_2_year": baseline_total_2_year,
        "simulated_total_2_year": simulated_total_2_year,
        "delta_total_2_year": delta_total_2_year,
    }


if __name__ == "__main__":
    result = compute_scenario_forecast(lever1_pct=10, lever2_pct=15, lever3_pct=15)

    print("2026-2027 Scenario Simulation (Lever 1=10%, Lever 2=15%, Lever 3=15%)")
    print("=" * 70)
    print(f"{'Quarter':<12}{'Baseline Hybrid':<18}{'Simulated Hybrid':<18}")
    for (yr, q), base, sim in zip(result["future_labels"], result["hybrid_baseline"], result["hybrid_sim"]):
        print(f"{yr} Q{q:<8}{base:<18.1f}{sim:<18.1f}")

    print("=" * 70)
    print(f"Baseline 2026-2027 total:  {result['baseline_total_2_year']:.0f} cases")
    print(f"Simulated 2026-2027 total: {result['simulated_total_2_year']:.0f} cases")
    print(f"Delta (baseline - simulated): {result['delta_total_2_year']:.0f} cases")
    print(f"\nbeta:  {result['beta_calibrated']:.5f} -> {result['beta_sim']:.5f}")
    print(f"alpha: {result['alpha_calibrated']:.5f} -> {result['alpha_sim']:.5f}")