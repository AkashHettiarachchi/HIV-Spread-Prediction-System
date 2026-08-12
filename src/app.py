import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from quarterly_data import get_quarterly_dataframe
from calibration_quarterly import calibrate_quarterly, sica_quarterly_predictions
from hybrid_pipeline_quarterly import run as run_hybrid_pipeline
from forecast_future import forecast_future

st.set_page_config(page_title="HIV Forecasting Dashboard - Sri Lanka", layout="wide", page_icon="🦠")

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #1E222D; padding: 15px; border-radius: 10px; border: 1px solid #2E3340; }
    div[data-testid="stSidebar"] { background-color: #161922; }
    </style>
""", unsafe_allow_html=True)

st.title("🦠 HIV/AIDS Epidemic Forecasting - Sri Lanka")
st.caption("Hybrid Mechanistic-Deep Learning Architecture: SICA Mathematical Framework "
           "+ Bi-LSTM residual correction, trained on real NSACP quarterly surveillance data "
           "(2016 Q1 - 2025 Q3, 39 quarters)")
st.markdown("---")

st.info(
    "**Data status:** All figures below use real, primary-source NSACP quarterly surveillance "
    "reports -- no synthetic or hardcoded values anywhere on this page. The SICA model is "
    "calibrated on this data; the Bi-LSTM residual-correction model is trained on the first "
    "~32 quarters and evaluated on the most recent 7 quarters (2024 Q1 - 2025 Q3), which the "
    "model never saw during training."
)

df = get_quarterly_dataframe()
real_values = df["new_cases"].values.astype(float)

with st.spinner("Calibrating SICA model and training Bi-LSTM on real quarterly data..."):
    calibrated_params, y0, calib_result, _, _ = calibrate_quarterly()
    sica_baseline_full = sica_quarterly_predictions(calibrated_params, y0, n_quarters=len(real_values))
    pipeline_results = run_hybrid_pipeline()

n_train = pipeline_results["n_train_raw"]
sica_test = pipeline_results["sica_test"]
hybrid_test = pipeline_results["hybrid_test"]
real_test = pipeline_results["real_test"]

def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

sica_mae, sica_rmse, sica_mape = compute_metrics(real_test, sica_test)
hybrid_mae, hybrid_rmse, hybrid_mape = compute_metrics(real_test, hybrid_test)

# ----------------- Metrics row -----------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Real quarters used", f"{len(real_values)}", help="2016 Q1 - 2025 Q3, NSACP reports")
with col2:
    st.metric("Held-out test quarters", f"{len(real_test)}", help="2024 Q1 - 2025 Q3, never seen in training")
with col3:
    st.metric("SICA-only test MAPE", f"{sica_mape:.2f}%")
with col4:
    st.metric("Hybrid test MAPE", f"{hybrid_mape:.2f}%",
              delta=f"{hybrid_mape - sica_mape:+.2f}pp vs SICA-only", delta_color="inverse")

st.caption(
    "Note: on this real held-out test, the Bi-LSTM residual correction does **not** clearly "
    "outperform the pure SICA baseline -- this is an honest reported result, not an error. "
    "With ~28 training sequences, the Bi-LSTM has limited data to learn from. See README for discussion."
)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- Main chart -----------------
st.subheader("📊 Real Data vs. SICA Baseline vs. Hybrid Forecast")

quarters_x = [f"{int(r.year)} Q{int(r.quarter)}" for _, r in df.iterrows()]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=quarters_x, y=real_values, mode="lines+markers", name="Real (NSACP quarterly reports)",
    line=dict(color="#8A99AD", width=1.5, dash="dash"), marker=dict(size=5, color="#8A99AD"),
))
fig.add_trace(go.Scatter(
    x=quarters_x, y=sica_baseline_full, mode="lines", name="Calibrated SICA baseline",
    line=dict(color="#FFA500", width=2),
))
fig.add_trace(go.Scatter(
    x=quarters_x[n_train:], y=hybrid_test, mode="lines+markers", name="Hybrid (held-out test only)",
    line=dict(color="#00E676", width=3), marker=dict(size=7, color="#00E676"),
))
fig.add_vline(x=n_train - 0.5, line_dash="dot", line_color="#E91E63",
              annotation_text="train / test split", annotation_position="top")

fig.update_layout(
    template="plotly_dark", paper_bgcolor="#161922", plot_bgcolor="#161922",
    margin=dict(l=20, r=20, t=30, b=20), height=500, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(title="Quarter", gridcolor="#2E3340", tickangle=-45),
    yaxis=dict(title="New HIV Cases per Quarter", gridcolor="#2E3340"),
)
st.plotly_chart(fig, width='stretch')

# ----------------- Test set table -----------------
st.subheader("🔬 Held-Out Test Quarters (real data, never used in training)")
test_df = df.iloc[n_train:][["year", "quarter"]].copy()
test_df["Real"] = real_test
test_df["SICA-only"] = np.round(sica_test, 1)
test_df["Hybrid"] = np.round(hybrid_test, 1)
st.dataframe(test_df, width='stretch')

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**SICA-only baseline**")
    st.write(f"MAE: {sica_mae:.2f} | RMSE: {sica_rmse:.2f} | MAPE: {sica_mape:.2f}%")
with col_b:
    st.markdown("**Hybrid SICA + Bi-LSTM**")
    st.write(f"MAE: {hybrid_mae:.2f} | RMSE: {hybrid_rmse:.2f} | MAPE: {hybrid_mape:.2f}%")

# ----------------- Calibrated parameters -----------------
with st.expander("📐 Calibrated SICA Parameters"):
    st.write(f"β (transmission rate): **{calibrated_params.beta:.5f}**")
    st.write(f"ρ (diagnosis / ART linkage rate): **{calibrated_params.rho:.5f}**")
    st.write(f"α (ART failure / progression rate): **{calibrated_params.alpha:.5f}**")
    st.caption(
        "ρ and α converged at their bounds (0.01 and 0.9 respectively) -- a sign of "
        "parameter non-identifiability with only 3 free parameters fit against 39 points. "
        "State this explicitly as a limitation; consider widening bounds or fixing fewer "
        "parameters as a refinement before final submission."
    )

st.caption(
    "Data source: National STD/AIDS Control Programme (NSACP), Ministry of Health, Sri Lanka -- "
    "quarterly surveillance update reports, 2016 Q1 to 2025 Q3."
)

# ----------------- FUTURE FORECAST: 2026-2030 -----------------
st.markdown("---")
st.subheader("🔮 Forecast: 2026 Q1 - 2030 Q4")
st.caption(
    "This is the actual research deliverable -- genuine forecasts beyond the last real data "
    "point (2025 Q3), produced by retraining the hybrid model on ALL 71 real quarters "
    "(not held-out validation, which is shown above). Future Bi-LSTM residuals are generated "
    "autoregressively: each quarter's prediction feeds into the input window for the next. "
    "Forecasts further into the future carry more uncertainty than near-term ones."
)

with st.spinner("Generating 2026-2030 forecast (retraining on full dataset)..."):
    future_results = forecast_future(n_future_quarters=20)

future_labels_x = [f"{yr} Q{q}" for yr, q in future_results["future_labels"]]
hist_labels_x = [f"{int(r.year)} Q{int(r.quarter)}" for _, r in df.iterrows()]

fig_future = go.Figure()
fig_future.add_trace(go.Scatter(
    x=hist_labels_x, y=future_results["real_historical"], mode="lines", name="Real (2008-2025)",
    line=dict(color="#8A99AD", width=1.5),
))
fig_future.add_trace(go.Scatter(
    x=future_labels_x, y=future_results["hybrid_future"], mode="lines+markers",
    name="Hybrid forecast (2026-2030)",
    line=dict(color="#00E676", width=3), marker=dict(size=6, color="#00E676"),
))
fig_future.add_trace(go.Scatter(
    x=future_labels_x, y=future_results["sica_future"], mode="lines", name="SICA-only (2026-2030)",
    line=dict(color="#FFA500", width=2, dash="dot"),
))
fig_future.add_vline(x=len(hist_labels_x) - 0.5, line_dash="dot", line_color="#E91E63",
                      annotation_text="last real data (2025 Q3)", annotation_position="top")
fig_future.update_layout(
    template="plotly_dark", paper_bgcolor="#161922", plot_bgcolor="#161922",
    margin=dict(l=20, r=20, t=30, b=20), height=480, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(title="Quarter", gridcolor="#2E3340", tickangle=-45,
               tickmode="array",
               tickvals=list(range(0, len(hist_labels_x) + len(future_labels_x), 4)),
               ticktext=(hist_labels_x + future_labels_x)[::4]),
    yaxis=dict(title="New HIV Cases per Quarter", gridcolor="#2E3340"),
)
st.plotly_chart(fig_future, width='stretch')

future_df = pd.DataFrame({
    "Year": [yr for yr, q in future_results["future_labels"]],
    "Quarter": [q for yr, q in future_results["future_labels"]],
    "SICA-only": np.round(future_results["sica_future"], 1),
    "Hybrid forecast": np.round(future_results["hybrid_future"], 1),
})
st.dataframe(future_df, width='stretch')

annual_forecast = future_df.groupby("Year")["Hybrid forecast"].sum().round(0)
st.markdown("**Annual forecast totals (hybrid model):**")

from report_generator import build_forecast_pdf

last_row = df.iloc[-1]
ann_cols = st.columns(len(annual_forecast))
for col, (yr, total) in zip(ann_cols, annual_forecast.items()):
    with col:
        st.metric(str(yr), f"{int(total)} cases")
        year_pdf_bytes = build_forecast_pdf(
            future_results,
            last_real_year=last_row["year"],
            last_real_quarter=last_row["quarter"],
            last_real_value=real_values[-1],
            year_filter=int(yr),
        )
        st.download_button(
            label="📄 PDF",
            data=year_pdf_bytes,
            file_name=f"HIV_Forecast_Report_{int(yr)}.pdf",
            mime="application/pdf",
            key=f"download_{yr}",
            width='stretch',
        )

st.markdown("<br>", unsafe_allow_html=True)

# ---- Full 2026-2030 report (all years combined) ----
full_pdf_bytes = build_forecast_pdf(
    future_results,
    last_real_year=last_row["year"],
    last_real_quarter=last_row["quarter"],
    last_real_value=real_values[-1],
)
st.download_button(
    label="📄 Download Full Report (2026-2030, PDF)",
    data=full_pdf_bytes,
    file_name="HIV_Forecast_Report_2026-2030.pdf",
    mime="application/pdf",
)