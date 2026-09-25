from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scenario_simulation import compute_scenario_forecast
from sidebar import render_sidebar


st.set_page_config(
    page_title="Preditca | Scenario Simulation & Policy Lab",
    layout="wide",
    page_icon=str(Path(__file__).parent.parent / "assets" / "LOGO4.png"),
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
    .eyebrow, .section-kicker { color: var(--green); font-size: .68rem; font-weight: 700; letter-spacing: .15em; text-transform: uppercase; }
    .eyebrow { margin-bottom: .7rem; }
    .section-kicker { margin-top: 2rem; }
    .hero-copy { color: var(--muted); max-width: 760px; line-height: 1.6; }
    .stAlert { background: #12231b; border: 1px solid #285e43; color: #cdebd9; border-radius: 12px; }
    div[data-testid="stMetric"] { background: linear-gradient(145deg, #151c1b, #0f1414); border: 1px solid var(--line); border-radius: 14px; padding: 1.1rem 1.2rem; }
    div[data-testid="stMetricLabel"] { color: var(--muted); font-size: .74rem; }
    div[data-testid="stMetricValue"] { color: #f4faf7; }
    div[data-testid="stMetricDelta"] { color: var(--green); }
    .insight { background: #101716; border: 1px solid var(--line); border-left: 3px solid var(--green); border-radius: 0 12px 12px 0; padding: .85rem 1rem; color: #aabbb4; font-size: .86rem; line-height: 1.55; }
    .lever-card { background: linear-gradient(145deg, #151c1b, #0d1211); border: 1px solid var(--line); border-radius: 15px; padding: 1.1rem 1.2rem .6rem; height: 100%; }
    .lever-title { color: #f2f8f4; font-family: 'Space Grotesk', sans-serif; font-size: 1.02rem; font-weight: 600; margin-bottom: .3rem; }
    .lever-copy { color: #8fa298; font-size: .8rem; line-height: 1.5; margin-bottom: .6rem; }
    .stDataFrame { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    .stSlider { padding-top: .2rem; }
    </style>
""", unsafe_allow_html=True)

render_sidebar("scenario")

# ----------------------------- header -----------------------------
st.markdown('<div class="eyebrow">PCA / Policy lab</div>', unsafe_allow_html=True)
st.title("What if we intervened?")
st.markdown(
    '<div class="hero-copy">Simulate how three public health interventions could reshape the 2026–2027 '
    'HIV incidence forecast, by adjusting the calibrated SICA parameters and the Bi-LSTM residual '
    'correction rather than editing raw model numbers by hand.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div style="height: 1.1rem"></div>', unsafe_allow_html=True)

st.info(
    "**How this simulation works.** The baseline 2026–2027 forecast is the existing, unmodified hybrid "
    "pipeline. The simulated forecast re-solves the SICA model with adjusted \u03b2 (transmission) and "
    "\u03b1 (ART progression), and scales the Bi-LSTM's own 2026–2027 residual correction to represent wider "
    "testing coverage. Testing expansion can raise *detected* cases even while it lowers *underlying* "
    "transmission \u2014 that is expected, not an error."
)

# ----------------------------- levers -----------------------------
st.markdown('<div class="section-kicker">Policy levers</div>', unsafe_allow_html=True)
st.subheader("Set the 2026–2027 intervention scenario")

l1, l2, l3 = st.columns(3)
with l1:
    st.markdown(
        '<div class="lever-card"><div class="lever-title">Transmission risk reduction</div>'
        '<div class="lever-copy">Safe-behavior uptake \u2014 PrEP, condom distribution, awareness '
        'campaigns. Lowers the calibrated transmission rate \u03b2.</div>',
        unsafe_allow_html=True,
    )
    lever1_pct = st.slider("Transmission Risk Reduction / Safe Behavior (%)", 0, 50, 10, key="lever1")
    st.markdown('</div>', unsafe_allow_html=True)
with l2:
    st.markdown(
        '<div class="lever-card"><div class="lever-title">Surveillance &amp; testing expansion</div>'
        '<div class="lever-copy">Expanded MSM / high-risk group screening campaigns. Scales up the '
        'Bi-LSTM residual, so more of the undiagnosed population is surfaced.</div>',
        unsafe_allow_html=True,
    )
    lever2_pct = st.slider("Surveillance & Testing Coverage Expansion (%)", 0, 50, 15, key="lever2")
    st.markdown('</div>', unsafe_allow_html=True)
with l3:
    st.markdown(
        '<div class="lever-card"><div class="lever-title">ART treatment acceleration</div>'
        '<div class="lever-copy">Faster linkage to antiretroviral therapy and viral suppression '
        '(U=U). Raises the calibrated progression rate \u03b1 to the chronic, controlled stage.</div>',
        unsafe_allow_html=True,
    )
    lever3_pct = st.slider("ART Treatment Acceleration Rate (%)", 0, 50, 15, key="lever3")
    st.markdown('</div>', unsafe_allow_html=True)

with st.spinner("Re-solving the SICA model and applying the scenario..."):
    result = compute_scenario_forecast(lever1_pct, lever2_pct, lever3_pct)

# ----------------------------- chart -----------------------------
st.markdown('<div class="section-kicker">Projected trajectory</div>', unsafe_allow_html=True)
st.subheader("2026–2027 Outlook: Baseline vs. Simulated Interventions")

history = result["recent_history"]
history_x = [f"{int(r.year)} Q{int(r.quarter)}" for r in history.itertuples()]
future_labels = result["future_labels"][:8]
future_x = [f"{yr} Q{q}" for yr, q in future_labels]
last_row = history.iloc[-1]
last_period = f"{int(last_row['year'])} Q{int(last_row['quarter'])}"

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=history_x, y=history["new_cases"], mode="lines+markers", name="Real history (2022 onward)",
    line=dict(color="#8A99AD", width=1.5), marker=dict(size=5, color="#8A99AD"),
))
fig.add_trace(go.Scatter(
    x=future_x, y=result["hybrid_baseline"], mode="lines+markers", name="Baseline hybrid forecast",
    line=dict(color="#FFA500", width=2.5), marker=dict(size=7, color="#FFA500"),
))
fig.add_trace(go.Scatter(
    x=future_x, y=result["hybrid_sim"], mode="lines+markers", name="Simulated (with interventions)",
    line=dict(color="#00E676", width=3, dash="dash"), marker=dict(size=8, color="#00E676"),
))
fig.add_vline(x=len(history_x) - 0.5, line_dash="dot", line_color="#E91E63",
              annotation_text=f"Last real data ({last_period}) / Forecast Start", annotation_position="top")
fig.update_layout(
    template="plotly_dark", paper_bgcolor="#111817", plot_bgcolor="#111817",
    margin=dict(l=20, r=20, t=30, b=20), height=480, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(title="Quarter", gridcolor="#26302d", tickangle=-45, zeroline=False),
    yaxis=dict(title="New HIV cases per quarter", gridcolor="#26302d", zeroline=False),
)
st.plotly_chart(fig, width="stretch")

# ----------------------------- impact metrics -----------------------------
st.markdown('<div class="section-kicker">Impact</div>', unsafe_allow_html=True)
st.subheader("2026, 2027, and two-year totals")

metric_columns = st.columns(3)
for column, label, period in zip(
    metric_columns,
    ("2026 total cases", "2027 total cases", "2026–2027 total cases"),
    ("2026", "2027", "2_year"),
):
    delta = result[f"delta_total_{period}"]
    outcome = "prevented" if delta >= 0 else "more detected"
    with column:
        st.metric(
            label,
            f"{result[f'baseline_total_{period}']:,.0f} / {result[f'simulated_total_{period}']:,.0f}",
            delta=f"{abs(delta):,.0f} {outcome}",
            delta_color="inverse",
        )

st.markdown(
    '<div class="insight">A positive prevented count means the simulated total is lower than baseline. '
    'If testing expansion is set high relative to the transmission and ART levers, the simulated total '
    'can rise instead \u2014 more of the true epidemic is being detected, not more people being infected.'
    '</div>',
    unsafe_allow_html=True,
)

# ----------------------------- quarterly table -----------------------------
st.markdown('<div class="section-kicker">Detail</div>', unsafe_allow_html=True)
st.subheader("Quarterly breakdown: 2026 Q1 through 2027 Q4")

quarter_df = pd.DataFrame({
    "Quarter": future_x,
    "Baseline hybrid": np.round(result["hybrid_baseline"], 1),
    "Simulated hybrid": np.round(result["hybrid_sim"], 1),
    "Net delta (baseline - simulated)": np.round(result["hybrid_baseline"] - result["hybrid_sim"], 1),
})
st.dataframe(quarter_df, width="stretch", hide_index=True)

with st.expander("Model parameters behind this scenario"):
    st.write(f"\u03b2 (transmission rate): calibrated **{result['beta_calibrated']:.5f}** "
             f"\u2192 simulated **{result['beta_sim']:.5f}**")
    st.write(f"\u03b1 (ART progression rate): calibrated **{result['alpha_calibrated']:.5f}** "
             f"\u2192 simulated **{result['alpha_sim']:.5f}**")
    st.caption(
        "Adjusted parameters are applied across the model's full solved trajectory, the same "
        "convention the existing forecast already uses \u2014 2026-2027 quarters are read off that "
        "trajectory rather than re-solved from a fresh starting point."
    )
    st.download_button(
        "Download this scenario (CSV)",
        data=quarter_df.to_csv(index=False).encode("utf-8"),
        file_name=f"scenario_2026-2027_L1-{lever1_pct}_L2-{lever2_pct}_L3-{lever3_pct}.csv",
        mime="text/csv",
    )

st.caption(
    "Data source: National STD/AIDS Control Programme (NSACP), Ministry of Health, Sri Lanka. "
    "Scenario projections are for research and planning discussion, not a substitute for clinical or "
    "policy decisions on their own."
)