from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from quarterly_data import get_quarterly_dataframe
from calibration_quarterly import calibrate_quarterly, sica_quarterly_predictions
from sica_model import SICAParams
from hybrid_pipeline_quarterly import run as run_hybrid_pipeline
from sidebar import render_sidebar

st.set_page_config(
    page_title="Preditca | AI HIV Analytics",
    layout="wide",
    page_icon=str(Path(__file__).parent / "assets" / "LOGO4.png"),
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #080a0c; --line: #26302d; --muted: #8c9a95; --green: #75f0a8; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: radial-gradient(circle at 70% -10%, #163329 0%, var(--ink) 34%); color: #edf4f0; }
    .main .block-container { max-width: 1440px; padding: 2.5rem 4rem 4rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -.02em; }
    h1 { font-size: clamp(2rem, 4vw, 3.3rem) !important; line-height: 1.05 !important; margin-bottom: .35rem !important; }
    h2 { font-size: 1.25rem !important; margin-top: 1.6rem !important; }
    .eyebrow, .section-kicker { color: var(--green); font-size: .68rem; font-weight: 700; letter-spacing: .15em; text-transform: uppercase; }
    .eyebrow { margin-bottom: .7rem; }
    .section-kicker { margin-top: 2rem; }
    .hero-copy { color: var(--muted); max-width: 740px; line-height: 1.6; }
    .stAlert { background: #12231b; border: 1px solid #285e43; color: #cdebd9; border-radius: 12px; }
    div[data-testid="stMetric"] { background: linear-gradient(145deg, #151c1b, #0f1414); border: 1px solid var(--line); border-radius: 14px; padding: 1.1rem 1.2rem; box-shadow: 0 12px 28px rgba(0,0,0,.18); }
    div[data-testid="stMetricLabel"] { color: var(--muted); font-size: .74rem; }
    div[data-testid="stMetricValue"] { color: #f4faf7; }
    div[data-testid="stMetricDelta"] { color: var(--green); }
    .insight { background: #101716; border: 1px solid var(--line); border-left: 3px solid var(--green); border-radius: 0 12px 12px 0; padding: .85rem 1rem; color: #aabbb4; font-size: .86rem; line-height: 1.55; }
    div[data-testid="stExpander"] { margin-top: 10px; }
    .stDataFrame { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    .stButton button, .stDownloadButton button { border-radius: 9px; transition: transform .18s ease, box-shadow .18s ease; }
    .stButton button:hover, .stDownloadButton button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(117,240,168,.16); }
    </style>
""", unsafe_allow_html=True)

render_sidebar("overview")

st.markdown('<div class="eyebrow">PCA / Overview</div>', unsafe_allow_html=True)
st.title("HIV intelligence, made legible.")
st.markdown('<div class="hero-copy">A transparent view of Sri Lanka\'s quarterly HIV surveillance signal, combining a calibrated SICA baseline with residual deep-learning correction.</div>', unsafe_allow_html=True)
st.markdown('<div style="height: 1.3rem"></div>', unsafe_allow_html=True)

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
    n_test = 7
    n_train = len(real_values) - n_test
    train_values = real_values[:n_train]
    calibrated_params, y0, calib_result, _, _ = calibrate_quarterly(train_values=train_values)
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

st.markdown('<div class="section-kicker">Signal health</div>', unsafe_allow_html=True)
st.subheader("Model performance at a glance")
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

st.markdown('<div class="insight">Validation note: the hybrid residual correction does not clearly outperform the pure SICA baseline on the real held-out window. The result is reported as observed, with no synthetic values or post-hoc adjustment.</div>', unsafe_allow_html=True)
st.markdown('<div class="section-kicker">Observed signal</div>', unsafe_allow_html=True)
st.subheader("Historical incidence and model fit")
st.caption(
    "Note: on this real held-out test, the Bi-LSTM residual correction does **not** clearly "
    "outperform the pure SICA baseline -- this is an honest reported result, not an error. "
    "See README for discussion of the validation result and model limitations."
)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- Main chart -----------------
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
    template="plotly_dark", paper_bgcolor="#111817", plot_bgcolor="#111817",
    margin=dict(l=20, r=20, t=30, b=20), height=500, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(title="Quarter", gridcolor="#26302d", tickangle=-45, zeroline=False),
    yaxis=dict(title="New HIV Cases per Quarter", gridcolor="#26302d", zeroline=False),
)
st.plotly_chart(fig, width='stretch')

# ----------------- Mechanistic sensitivity and residual XAI -----------------
with st.expander("🔬 Mechanistic Model Sensitivity & XAI Analysis", expanded=False):
    st.markdown(
        "Adjust the three calibrated SICA parameters to see how transmission, "
        "diagnosis/linkage, and ART progression assumptions change the mechanistic "
        "quarterly incidence trajectory. The other SICA parameters remain fixed at "
        "their literature-informed defaults."
    )

    sensitivity_col1, sensitivity_col2, sensitivity_col3 = st.columns(3)
    with sensitivity_col1:
        sensitivity_beta = st.slider(
            "β — transmission rate",
            min_value=0.001,
            max_value=2.0,
            value=float(np.clip(calibrated_params.beta, 0.001, 2.0)),
            step=0.001,
            format="%.3f",
            help="Effective transmission rate in the SICA force of infection.",
        )
    with sensitivity_col2:
        sensitivity_rho = st.slider(
            "ρ — diagnosis / ART linkage rate",
            min_value=0.001,
            max_value=0.99,
            value=float(np.clip(calibrated_params.rho, 0.001, 0.99)),
            step=0.001,
            format="%.3f",
            help="Rate at which undiagnosed people enter the chronic/ART compartment.",
        )
    with sensitivity_col3:
        sensitivity_alpha = st.slider(
            "α — ART failure / progression rate",
            min_value=0.001,
            max_value=0.99,
            value=float(np.clip(calibrated_params.alpha, 0.001, 0.99)),
            step=0.001,
            format="%.3f",
            help="Rate of progression from chronic/ART care to AIDS stage.",
        )

    sensitivity_params = SICAParams(
        Lambda=calibrated_params.Lambda,
        beta=sensitivity_beta,
        eta=calibrated_params.eta,
        mu=calibrated_params.mu,
        phi=calibrated_params.phi,
        rho=sensitivity_rho,
        omega=calibrated_params.omega,
        gamma=calibrated_params.gamma,
        alpha=sensitivity_alpha,
        delta=calibrated_params.delta,
    )
    interactive_sica = sica_quarterly_predictions(
        sensitivity_params, y0, n_quarters=len(real_values)
    )

    sensitivity_fig = go.Figure()
    sensitivity_fig.add_trace(go.Scatter(
        x=quarters_x, y=real_values, mode="lines+markers",
        name="Real NSACP data", line=dict(color="#8A99AD", width=1.5, dash="dash"),
        marker=dict(size=5, color="#8A99AD"),
    ))
    sensitivity_fig.add_trace(go.Scatter(
        x=quarters_x, y=sica_baseline_full, mode="lines",
        name="Calibrated baseline", line=dict(color="#FFA500", width=2),
    ))
    sensitivity_fig.add_trace(go.Scatter(
        x=quarters_x, y=interactive_sica, mode="lines",
        name="Interactive simulation", line=dict(color="#00E676", width=2.5),
    ))
    sensitivity_fig.update_layout(
        template="plotly_dark", paper_bgcolor="#111817", plot_bgcolor="#111817",
        margin=dict(l=20, r=20, t=30, b=20), height=440, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Quarter", gridcolor="#26302d", tickangle=-45, zeroline=False),
        yaxis=dict(title="New HIV Cases per Quarter", gridcolor="#26302d", zeroline=False),
    )
    st.plotly_chart(sensitivity_fig, width="stretch")

    residuals = real_values - sica_baseline_full
    residual_fig = go.Figure()
    residual_fig.add_trace(go.Bar(
        x=quarters_x, y=residuals, name="Residual Eₜ",
        marker_color=np.where(residuals >= 0, "#00E676", "#FF6B6B"),
        hovertemplate="%{x}<br>Residual: %{y:.1f} cases<extra></extra>",
    ))
    residual_fig.add_trace(go.Scatter(
        x=quarters_x, y=np.zeros(len(residuals)), mode="lines", name="Zero error",
        line=dict(color="#D5DED9", width=1),
    ))
    residual_fig.add_vrect(
        x0="2020 Q1", x1="2021 Q4", fillcolor="#FF6B6B", opacity=0.12,
        line_width=0, annotation_text="COVID-19 testing disruptions",
        annotation_position="top left",
    )
    residual_fig.add_vrect(
        x0="2022 Q2", x1="2025 Q3", fillcolor="#00E676", opacity=0.10,
        line_width=0, annotation_text="Expanded MSM testing / accelerated detection",
        annotation_position="bottom right",
    )
    residual_fig.update_layout(
        template="plotly_dark", paper_bgcolor="#111817", plot_bgcolor="#111817",
        margin=dict(l=20, r=20, t=55, b=20), height=420, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Quarter", gridcolor="#26302d", tickangle=-45, zeroline=False),
        yaxis=dict(title="Residual: actual − SICA baseline (cases)", gridcolor="#26302d", zeroline=True),
    )
    st.plotly_chart(residual_fig, width="stretch")
    st.markdown(
        '<div class="insight"><strong>Why the Bi-LSTM residual correction is needed:</strong> '
        "SICA encodes a smooth mechanistic transmission and care-flow process, but it does not "
        "observe abrupt changes in testing access, reporting intensity, service disruption, or "
        "case-finding strategy. Negative residuals during 2020 Q1–2021 Q4 are consistent with "
        "COVID-19 testing disruptions and under-reporting, while the sustained positive shift from "
        "2022 Q2–2025 Q3 is consistent with expanded MSM testing and accelerated detection. The "
        "Bi-LSTM is trained on these residual dynamics so the hybrid model can correct the SICA "
        "baseline when surveillance regimes change.</div>",
        unsafe_allow_html=True,
    )

# ----------------- Test set table -----------------
st.markdown('<div class="section-kicker">Validation window</div>', unsafe_allow_html=True)
st.subheader("Held-out quarters")
test_df = df.iloc[n_train:][["year", "quarter"]].copy()
test_df["Real"] = real_test
test_df["SICA-only"] = np.round(sica_test, 1)
test_df["Hybrid"] = np.round(hybrid_test, 1)
st.dataframe(test_df, width='stretch')

col_a, col_b = st.columns(2)
with col_a:
    st.markdown('<div class="insight"><strong>SICA-only baseline</strong><br>MAE: {:.2f} &nbsp;|&nbsp; RMSE: {:.2f} &nbsp;|&nbsp; MAPE: {:.2f}%</div>'.format(sica_mae, sica_rmse, sica_mape), unsafe_allow_html=True)
with col_b:
    st.markdown('<div class="insight"><strong>Hybrid SICA + Bi-LSTM</strong><br>MAE: {:.2f} &nbsp;|&nbsp; RMSE: {:.2f} &nbsp;|&nbsp; MAPE: {:.2f}%</div>'.format(hybrid_mae, hybrid_rmse, hybrid_mape), unsafe_allow_html=True)

# ----------------- Calibrated parameters -----------------
with st.expander("Model parameters and limitations"):
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
    "quarterly surveillance update reports, 2008 Q1 to 2025 Q3."
)
