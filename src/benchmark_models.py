"""
benchmark_models.py -- backend for the "Model Benchmark" page.

Compares the proposed hybrid (SICA + Bi-LSTM) against SICA, ARIMA and a
Vanilla LSTM on the SAME chronological hold-out (last 7 quarters), using
the same real NSACP data as the rest of the project.

Fairness rules (applied to every model):
  1. Chronological split only -- the last `n_test` quarters are never seen
     during fitting.
  2. SICA is calibrated on the TRAINING quarters only, then run forward
     through the test period. (The overview page currently calibrates SICA
     on all 71 quarters, so numbers here will differ from that page --
     these are the leak-free ones.)
  3. Every neural model uses the same window, units, dropout, optimiser,
     epochs, batch size, early stopping and validation split.
  4. Scaling statistics come from the training partition only.
  5. Neural models are trained with several random seeds; results are
     reported as mean +/- std across seeds.
  6. Two evaluation modes are reported side by side:
        one_step   -- each test quarter is predicted from the TRUE history
                      (what a surveillance team would do quarter by quarter)
        multi_step -- all test quarters are forecast blind from the end of
                      training (what forecast_future.py does for 2026-2030)

Models:
  proposed   : Hybrid SICA + Bi-LSTM
  baselines  : SICA, ARIMA(2,1,2), Vanilla LSTM (pure data-driven)
  references : SICA + LSTM (one-directional hybrid), Naive last-value

Nothing here edits existing project files. It only imports from them.
Extra dependency: statsmodels  (pip install statsmodels)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from quarterly_data import get_quarterly_dataframe
from sica_model import SICAParams
from calibration_quarterly import (
    initial_conditions_2008q1,
    residuals_fn,
    sica_quarterly_predictions,
)
from hybrid_pipeline_quarterly import create_sequences, normalize, denormalize


# --------------------------------------------------------------------------
# Names / labels
# --------------------------------------------------------------------------
HYBRID = "Hybrid SICA + Bi-LSTM"
SICA = "SICA"
ARIMA = "ARIMA (2,1,2)"
VLSTM = "Vanilla LSTM"
SICA_LSTM = "SICA + LSTM"
NAIVE = "Naive (last value)"

MODEL_ROLES = {
    HYBRID: "proposed",
    SICA: "baseline",
    ARIMA: "baseline",
    VLSTM: "baseline",
    SICA_LSTM: "reference",
    NAIVE: "reference",
}
MODEL_ORDER = [HYBRID, SICA, ARIMA, VLSTM, SICA_LSTM, NAIVE]

METRIC_LABELS = {"mae": "MAE", "rmse": "RMSE", "mape": "MAPE (%)", "r2": "R²"}
MODE_LABELS = {
    "one_step": "One-step-ahead (true history each quarter)",
    "multi_step": "Multi-step (blind 7-quarter forecast)",
}
_LOWER_IS_BETTER = {"mae", "rmse", "mape"}


@dataclass
class BenchmarkConfig:
    n_test: int = 7                 # held-out quarters (2024 Q1 - 2025 Q3)
    window: int = 4                 # lookback quarters (same as the paper)
    units: int = 16                 # LSTM units (per direction for Bi-LSTM)
    dropout: float = 0.2
    learning_rate: float = 5e-3
    epochs: int = 300
    batch_size: int = 4
    patience: int = 35
    val_sequences: int = 4          # last N training windows used for early stopping (0 = monitor train loss)
    n_seeds: int = 3
    base_seed: int = 42
    arima_order: Tuple[int, int, int] = (2, 1, 2)
    include_reference: bool = True  # also train SICA+LSTM and add the naive baseline


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err / y_true)) * 100.0),
        "r2": float(1.0 - np.sum(err ** 2) / sst) if sst > 0 else float("nan"),
    }


# --------------------------------------------------------------------------
# SICA (train-only calibration)
# --------------------------------------------------------------------------
def calibrate_sica_train_only(train_values: np.ndarray):
    """Same 3 free parameters / bounds as calibration_quarterly.py, but fitted
    on the training quarters only."""
    fixed = SICAParams()
    y0 = initial_conditions_2008q1()
    x0 = [fixed.beta, fixed.rho, fixed.alpha]
    bounds = ([0.001, 0.001, 0.001], [2.0, 0.99, 0.99])
    res = least_squares(residuals_fn, x0, bounds=bounds, args=(fixed, train_values, y0))
    params = SICAParams(
        Lambda=fixed.Lambda, beta=res.x[0], eta=fixed.eta, mu=fixed.mu,
        phi=fixed.phi, rho=res.x[1], omega=fixed.omega, gamma=fixed.gamma,
        alpha=res.x[2], delta=fixed.delta,
    )
    return params, y0, res


def _bound_hits(x, bounds_lo=(0.01, 0.01, 0.01), bounds_hi=(1.0, 0.9, 0.9)) -> List[str]:
    names = ("beta", "rho", "alpha")
    hits = []
    for n, v, lo, hi in zip(names, x, bounds_lo, bounds_hi):
        if abs(v - lo) < 1e-4 or abs(v - hi) < 1e-4:
            hits.append(n)
    return hits


# --------------------------------------------------------------------------
# ARIMA
# --------------------------------------------------------------------------
def _arima_forecasts(train: np.ndarray, test: np.ndarray, order) -> Tuple[np.ndarray, np.ndarray]:
    """Return (one_step, multi_step) forecasts for the test quarters.
    Parameters are estimated on the training data only; for one-step the
    true test observations are appended WITHOUT re-estimating parameters."""
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = _ARIMA(np.asarray(train, float), order=order).fit()
        multi = np.asarray(fit.forecast(len(test)), dtype=float)
        one, cur = [], fit
        for i in range(len(test)):
            one.append(float(np.asarray(cur.forecast(1))[0]))
            cur = cur.append(np.array([test[i]], dtype=float), refit=False)
    return np.array(one), multi


# --------------------------------------------------------------------------
# Neural models (Vanilla LSTM, SICA+LSTM, SICA+Bi-LSTM share one recipe)
# --------------------------------------------------------------------------
def _build_recurrent(cfg: BenchmarkConfig, bidirectional: bool):
    from tensorflow import keras
    from tensorflow.keras import layers

    rnn = layers.LSTM(cfg.units)
    if bidirectional:
        rnn = layers.Bidirectional(rnn)
    model = keras.Sequential([
        layers.Input(shape=(cfg.window, 1)),
        rnn,
        layers.Dropout(cfg.dropout),
        layers.Dense(8, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate), loss="mse")
    return model


def _fit_recurrent(scaled_train: np.ndarray, cfg: BenchmarkConfig, bidirectional: bool, seed: int):
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    X, y = create_sequences(scaled_train, cfg.window)
    v = cfg.val_sequences
    if v and len(X) > v + 20:
        data = dict(x=X[:-v], y=y[:-v], validation_data=(X[-v:], y[-v:]))
        monitor = "val_loss"
    else:
        data = dict(x=X, y=y)
        monitor = "loss"
    model = _build_recurrent(cfg, bidirectional)
    stop = keras.callbacks.EarlyStopping(monitor=monitor, patience=cfg.patience, restore_best_weights=True)
    model.fit(epochs=cfg.epochs, batch_size=cfg.batch_size, callbacks=[stop], verbose=0, **data)
    return model


def _predict(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X, verbose=0), dtype=float).reshape(-1)


def _one_step(model, scaled_full: np.ndarray, n_train: int, cfg: BenchmarkConfig) -> np.ndarray:
    X = np.stack([
        scaled_full[n_train + i - cfg.window: n_train + i] for i in range(cfg.n_test)
    ]).reshape(cfg.n_test, cfg.window, 1)
    return _predict(model, X)


def _multi_step(model, scaled_train: np.ndarray, cfg: BenchmarkConfig) -> np.ndarray:
    win = list(scaled_train[-cfg.window:])
    out = []
    for _ in range(cfg.n_test):
        x = np.array(win[-cfg.window:], dtype=float).reshape(1, cfg.window, 1)
        p = float(_predict(model, x)[0])
        out.append(p)
        win.append(p)
    return np.array(out)


def _run_neural(
    series_full: np.ndarray,
    offset_test: np.ndarray,
    n_train: int,
    cfg: BenchmarkConfig,
    bidirectional: bool,
    on_seed_done: Optional[Callable[[int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train one network per seed on series_full[:n_train] (scaled with train
    statistics) and forecast the test quarters.

    series_full : the series the network models (raw cases for the vanilla
                  LSTM, SICA residuals for the hybrids).
    offset_test : added back after inverse scaling (zeros for vanilla LSTM,
                  the SICA baseline for the hybrids).
    Returns two arrays of shape (n_seeds, n_test): one-step and multi-step.
    """
    scaled_train, mm = normalize(series_full[:n_train])
    scaled_full, _ = normalize(series_full, *mm)

    ones, multis = [], []
    for k in range(cfg.n_seeds):
        model = _fit_recurrent(scaled_train, cfg, bidirectional, cfg.base_seed + k)
        one = denormalize(_one_step(model, scaled_full, n_train, cfg), mm) + offset_test
        multi = denormalize(_multi_step(model, scaled_train, cfg), mm) + offset_test
        ones.append(np.clip(one, 0, None))
        multis.append(np.clip(multi, 0, None))
        if on_seed_done:
            on_seed_done(k)
    return np.array(ones), np.array(multis)


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------
def _pack(name: str, one: np.ndarray, multi: np.ndarray, y_test: np.ndarray) -> dict:
    """one / multi: arrays (n_runs, n_test). Deterministic models pass n_runs=1."""
    out = {"role": MODEL_ROLES[name], "n_runs": int(one.shape[0])}
    for mode, P in (("one_step", one), ("multi_step", multi)):
        per_run = [compute_metrics(y_test, p) for p in P]
        keys = list(per_run[0].keys())
        mean_pred = P.mean(axis=0)
        out[mode] = {
            "pred": mean_pred,
            "pred_std": P.std(axis=0),
            "abs_err": np.abs(y_test - mean_pred),
            "metrics": {k: float(np.mean([m[k] for m in per_run])) for k in keys},
            "metrics_std": {k: float(np.std([m[k] for m in per_run])) for k in keys},
            "per_run_metrics": per_run,
        }
    return out


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def run_benchmark(
    cfg: Optional[BenchmarkConfig] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """Run the full benchmark. `progress(fraction, message)` is optional."""
    cfg = cfg or BenchmarkConfig()

    df = get_quarterly_dataframe()
    y = df["new_cases"].values.astype(float)
    n_total = len(y)
    n_train = n_total - cfg.n_test
    y_train, y_test = y[:n_train], y[n_train:]
    labels_full = [f"{int(r.year)} Q{int(r.quarter)}" for r in df.itertuples()]

    neural_jobs = 2 + (1 if cfg.include_reference else 0)   # vanilla, hybrid, (sica_lstm)
    total_steps = 3 + neural_jobs * cfg.n_seeds
    state = {"done": 0}

    def tick(msg: str):
        state["done"] += 1
        if progress:
            progress(min(state["done"] / total_steps, 1.0), msg)

    models: Dict[str, dict] = {}
    errors: Dict[str, str] = {}

    # ---- SICA (train-only calibration) ----
    sica_full = None
    sica_info = {}
    try:
        sica_params, y0, res = calibrate_sica_train_only(y_train)
        sica_full = sica_quarterly_predictions(sica_params, y0, n_quarters=n_total)
        sica_test = sica_full[n_train:]
        models[SICA] = _pack(SICA, sica_test[None, :], sica_test[None, :], y_test)
        sica_info = {
            "beta": float(res.x[0]), "rho": float(res.x[1]), "alpha": float(res.x[2]),
            "cost": float(res.cost), "converged": bool(res.success),
            "at_bounds": _bound_hits(res.x),
            "in_sample_mape": float(np.mean(np.abs((y_train - sica_full[:n_train]) / y_train)) * 100),
        }
    except Exception as exc:  # noqa: BLE001
        errors[SICA] = f"{type(exc).__name__}: {exc}"
    tick("SICA calibrated on training quarters")

    # ---- ARIMA ----
    try:
        one, multi = _arima_forecasts(y_train, y_test, cfg.arima_order)
        models[ARIMA] = _pack(ARIMA, one[None, :], multi[None, :], y_test)
    except Exception as exc:  # noqa: BLE001
        errors[ARIMA] = f"{type(exc).__name__}: {exc}"
    tick("ARIMA fitted")

    # ---- Naive (reference) ----
    if cfg.include_reference:
        one = np.concatenate([[y_train[-1]], y_test[:-1]])
        multi = np.full(cfg.n_test, y_train[-1])
        models[NAIVE] = _pack(NAIVE, one[None, :], multi[None, :], y_test)
    tick("Naive baseline")

    # ---- Neural models ----
    def neural(name: str, series_full: np.ndarray, offset: np.ndarray, bidirectional: bool):
        try:
            ones, multis = _run_neural(
                series_full, offset, n_train, cfg, bidirectional,
                on_seed_done=lambda k: tick(f"{name}: seed {k + 1}/{cfg.n_seeds}"),
            )
            models[name] = _pack(name, ones, multis, y_test)
        except Exception as exc:  # noqa: BLE001
            errors[name] = f"{type(exc).__name__}: {exc}"
            state["done"] += cfg.n_seeds  # keep the progress bar honest
            if progress:
                progress(min(state["done"] / total_steps, 1.0), f"{name} skipped")

    neural(VLSTM, y, np.zeros(cfg.n_test), bidirectional=False)

    if sica_full is not None:
        residual = y - sica_full
        sica_test = sica_full[n_train:]
        neural(HYBRID, residual, sica_test, bidirectional=True)
        if cfg.include_reference:
            neural(SICA_LSTM, residual, sica_test, bidirectional=False)
    else:
        for name in (HYBRID, SICA_LSTM) if cfg.include_reference else (HYBRID,):
            errors[name] = "Skipped: SICA calibration failed"

    ordered = {n: models[n] for n in MODEL_ORDER if n in models}
    return {
        "config": cfg,
        "labels_full": labels_full,
        "labels_test": labels_full[n_train:],
        "actual_full": y,
        "actual_test": y_test,
        "n_train": n_train,
        "n_test": cfg.n_test,
        "sica_full": sica_full,
        "sica_info": sica_info,
        "models": ordered,
        "errors": errors,
    }


# --------------------------------------------------------------------------
# Views on the results (used by the frontend)
# --------------------------------------------------------------------------
def results_to_dataframe(results: dict, mode: str = "one_step") -> pd.DataFrame:
    rows = []
    for name, m in results["models"].items():
        r = {"Model": name, "Role": m["role"], "Runs": m["n_runs"]}
        for k in ("mae", "rmse", "mape", "r2"):
            r[k] = m[mode]["metrics"][k]
            r[k + "_std"] = m[mode]["metrics_std"][k]
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(results: dict, mode: str = "one_step", metric: str = "rmse") -> dict:
    models = results["models"]
    sign = 1.0 if metric in _LOWER_IS_BETTER else -1.0
    value = lambda n: models[n][mode]["metrics"][metric]      # noqa: E731
    spread = lambda n: models[n][mode]["metrics_std"][metric]  # noqa: E731

    ranking = sorted(models, key=lambda n: sign * value(n))
    out = {
        "ranking": [(n, value(n)) for n in ranking],
        "hybrid_present": HYBRID in models,
    }
    if HYBRID not in models:
        return out

    out["hybrid_rank"] = ranking.index(HYBRID) + 1
    out["n_models"] = len(ranking)
    others = [n for n in models if n != HYBRID]
    hv = value(HYBRID)

    def hybrid_better(n):
        return sign * hv < sign * value(n)

    out["beats"] = [n for n in others if hybrid_better(n)]
    out["loses_to"] = [n for n in others if not hybrid_better(n)]
    out["ties"] = [n for n in others if abs(hv - value(n)) <= max(spread(HYBRID), spread(n))
                   and (spread(HYBRID) > 0 or spread(n) > 0)]

    baselines = [n for n in others if models[n]["role"] == "baseline"]
    out["baselines"] = baselines
    if baselines:
        best_base = min(baselines, key=lambda n: sign * value(n))
        out["best_baseline"] = best_base
        bv = value(best_base)
        if metric in _LOWER_IS_BETTER and bv != 0:
            out["improvement_pct"] = (bv - hv) / abs(bv) * 100.0   # >0: hybrid better

    h_err = models[HYBRID][mode]["abs_err"]
    out["quarter_wins"] = {
        n: (int(np.sum(h_err < models[n][mode]["abs_err"])), len(h_err)) for n in others
    }
    return out


def verdict_lines(results: dict, mode: str, metric: str) -> List[Tuple[str, str]]:
    """Plain, neutral statements about what the numbers do and do not show.
    Returns [(level, text)] with level in {"good", "neutral", "warn"}."""
    s = summarize(results, mode, metric)
    lines: List[Tuple[str, str]] = []
    if not s["hybrid_present"]:
        return [("warn", "The hybrid model did not run, so no comparison is possible.")]

    label = METRIC_LABELS[metric]
    lines.append(("neutral", f"On {label} ({MODE_LABELS[mode].split(' (')[0].lower()}), the hybrid ranks "
                             f"#{s['hybrid_rank']} of {s['n_models']} models."))

    baselines = s.get("baselines", [])
    if baselines:
        beaten = [n for n in baselines if n in s["beats"]]
        if len(beaten) == len(baselines):
            txt = f"The hybrid has a lower {label} than all requested baselines ({', '.join(baselines)})."
            if "improvement_pct" in s:
                txt += f" Versus the best baseline ({s['best_baseline']}): {s['improvement_pct']:+.1f}%."
            lines.append(("good", txt))
        else:
            missed = [n for n in baselines if n not in s["beats"]]
            lines.append(("warn", f"The hybrid does NOT beat: {', '.join(missed)}."))

    ties = [n for n in s.get("ties", []) if n in s["beats"] or n in s["loses_to"]]
    if ties:
        lines.append(("neutral", "Difference from " + ", ".join(ties) +
                                 " is within seed-to-seed variation -- treat as a tie, not a win."))

    refs_better = [n for n in s.get("loses_to", []) if results["models"][n]["role"] == "reference"]
    if refs_better:
        lines.append(("warn", "Reference model(s) doing at least as well as the hybrid: "
                              + ", ".join(refs_better) + "."))

    lines.append(("neutral", f"Only {results['n_test']} test quarters: small differences can flip with a "
                             f"different split. Use this as evidence, not proof."))
    return lines
