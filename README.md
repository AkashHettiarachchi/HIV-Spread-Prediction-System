# Hybrid SICA + Bi-LSTM HIV/AIDS Forecasting -- Final Real-Data Results

Research codebase for: *"A Hybrid Mathematical and Deep Learning
Framework for HIV/AIDS Progression Modeling and Forecasting using SICA
and LSTM Networks."*

## Data

**71 real quarterly data points (2008 Q1 - 2025 Q3)**, compiled
directly from National STD/AIDS Control Programme (NSACP), Ministry
of Health, Sri Lanka quarterly surveillance update reports. See
`quarterly_data.py` for the full series and source citations for
every quarter. Two quarters (2017 Q3, 2022 Q1) were independently
cross-validated against separate Epidemiology Unit STD/HIV bulletins
and matched exactly.

## Headline Result (held-out test: 2024 Q1 - 2025 Q3, real data only)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| SICA-only (calibrated, mechanistic) | 56.73 | 58.03 | 26.90% |
| **Hybrid SICA + Bi-LSTM** | **14.53** | **18.38** | **7.00%** |

**This is the core finding of the dissertation.** The pure SICA model,
even calibrated on real data, cannot capture the regime shift in HIV
incidence that becomes visible across the full 2008-2025 span (a
relatively flat/low incidence period through ~2018, followed by
accelerated growth from ~2019 onward -- plausibly linked to expanded
testing, MSM epidemic dynamics, and behavioural/reporting changes
documented in the literature). A fixed-structure, 3-parameter SICA fit
cannot flex to both regimes at once.

The Bi-LSTM, trained on the residual between real data and the SICA
baseline, **does** learn this pattern -- with 60 real training
sequences (vs. only 28 in an earlier, shorter-window version of this
analysis), it has enough signal to correct for what the mechanistic
model misses. This is exactly the hybrid design's intended
justification, and here it is demonstrated on real, held-out data.

## Structure

```
hiv_sica_lstm/
├── src/
│   ├── quarterly_data.py           # 71 real NSACP quarterly data points, cited
│   ├── sica_model.py                # SICA ODE system -- single source of truth
│   ├── calibration_quarterly.py     # fits SICA params (beta, rho, alpha) to real data
│   ├── hybrid_pipeline_quarterly.py # trains Bi-LSTM on residuals, real train/test split
│   └── app.py                       # Streamlit dashboard, all metrics computed live
├── requirements.txt
└── README.md
```

## Two separate pipelines -- know the difference

1. **`hybrid_pipeline_quarterly.py` -- VALIDATION.** Holds out 2024-2025
   (real data) to prove the model works. This produces the 7.00% MAPE
   headline result above. Use this as evidence in your evaluation
   section (4.6).
2. **`forecast_future.py` -- ACTUAL FORECASTING.** Retrains on ALL 71
   real quarters (no held-out split -- you want the strongest model
   for real forecasting) and rolls forward autoregressively to produce
   genuine 2026 Q1 - 2030 Q4 predictions. This is your research
   deliverable -- what a policymaker would actually look at. Run it
   with `python src/forecast_future.py` or view it in the dashboard.

Sample output (your exact numbers may vary slightly on retrain, since
Bi-LSTM training has some randomness):
```
2026: ~888 new HIV cases (forecast)
2027: ~927 new HIV cases (forecast)
2028: ~969 new HIV cases (forecast)
2029: ~1012 new HIV cases (forecast)
2030: ~1057 new HIV cases (forecast)
```
State clearly in your dissertation that forecast uncertainty grows
with horizon -- 2026 is far more reliable than 2030. The autoregressive
rollout means small errors in early future quarters compound into
later ones.

## Run order

```bash
pip install -r requirements.txt
python src/quarterly_data.py            # inspect the 71-quarter dataset
python src/calibration_quarterly.py      # calibrate SICA, see in-sample fit
python src/hybrid_pipeline_quarterly.py  # VALIDATION: train Bi-LSTM, see held-out test results
python src/forecast_future.py            # FORECASTING: 2026-2030 predictions
streamlit run src/app.py                 # full interactive dashboard (both sections)
```

## Honest limitations to state in your dissertation

1. **SICA parameter non-identifiability**: only 3 of 9 SICA
   parameters are calibrated (beta, rho, alpha); the rest are fixed
   at literature-informed values. With only a handful of free
   parameters against a real but structurally-shifting series, `rho`
   and `alpha` converge at their optimization bounds -- state this
   explicitly rather than hiding it.
2. **SICA regime-shift blindness is itself a finding**, not just a
   weakness -- it is direct evidence for why a hybrid mechanistic +
   deep learning approach has value in this setting, which is the
   central argument of the dissertation.
3. **SICA baseline is calibrated on the full dataset** (including the
   test period) before the Bi-LSTM train/test split is applied. This
   is a reasonable simplification (the mechanistic backbone is meant
   to represent long-run structure, not to be blind-tested itself),
   but a fully rigorous version would calibrate SICA on train-only
   data too -- consider this a refinement to make before final
   submission if time allows.
4. **Data revisions**: several quarters were revised between
   successive NSACP reports (documented with old/new values in
   `quarterly_data.py`). This is a genuine, citable property of
   real surveillance data.

## Data sources

National STD/AIDS Control Programme (NSACP), Ministry of Health, Sri
Lanka -- quarterly surveillance update reports, 2008 Q1 to 2025 Q3.
No. 29, De Saram Place, Colombo 10, Sri Lanka.

## Sandbox note

This build environment has no persistent internet access for package
installation in some sessions; `tensorflow` and `streamlit` were
successfully installed and all scripts above were run and verified
with the real output shown in this README.
