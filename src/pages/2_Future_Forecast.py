from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from quarterly_data import get_quarterly_dataframe
from forecast_future import forecast_future
from report_generator import build_forecast_pdf
from sidebar import render_sidebar


st.set_page_config(
    page_title="Preditca | Future Forecast",
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
    .hero-copy { color: var(--muted); max-width: 740px; line-height: 1.6; }
    .forecast-intro { display: flex; align-items: flex-end; justify-content: space-between; gap: 2rem; margin: 0 0 1.6rem; }
    .forecast-intro-copy { max-width: 680px; }
    .forecast-badge { flex: 0 0 auto; padding: .55rem .8rem; border: 1px solid #315840; border-radius: 999px; color: #a8efbd; background: rgba(117,240,168,.07); font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .outlook-summary { display: flex; align-items: center; justify-content: space-between; gap: 1.5rem; margin: .5rem 0 1rem; padding: 1rem 1.15rem; border: 1px solid var(--line); border-radius: 14px; background: linear-gradient(100deg, rgba(117,240,168,.10), rgba(16,23,22,.62)); }
    .outlook-summary-label { color: var(--muted); font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; }
    .outlook-summary-value { margin-top: .2rem; color: #f1f7f3; font-family: 'Space Grotesk', sans-serif; font-size: 1.7rem; font-weight: 600; }
    .outlook-summary-note { color: #9fb0a7; font-size: .8rem; text-align: right; }
    .annual-card { min-height: 166px; box-sizing: border-box; padding: 1.1rem 1.15rem .95rem; border: 1px solid var(--line); border-radius: 15px; background: linear-gradient(145deg, #151c1b, #0d1211); transition: border-color .18s ease, transform .18s ease; }
    .annual-card:hover { border-color: #487a59; transform: translateY(-2px); }
    .annual-year { color: #aabbb3; font-size: .78rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
    .annual-value { margin: .7rem 0 .45rem; color: #f2f8f4; font-family: 'Space Grotesk', sans-serif; font-size: clamp(1.65rem, 2.5vw, 2.35rem); font-weight: 600; letter-spacing: -.045em; white-space: nowrap; }
    .annual-meta { color: #7f9087; font-size: .74rem; }
    .annual-meta.up { color: #85eaa7; }
    .annual-meta.down { color: #f0ad77; }
    .full-report { margin-top: 1.3rem; padding: 1.1rem 1.2rem; border: 1px solid #2e513b; border-radius: 14px; background: #101a14; }
    .full-report-title { color: #e9f5ed; font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 600; }
    .full-report-copy { margin: .2rem 0 .7rem; color: #91a59a; font-size: .8rem; }
    .stAlert { background: #12231b; border: 1px solid #285e43; border-radius: 12px; }
    div[data-testid="stMetric"] { background: linear-gradient(145deg, #151c1b, #0f1414); border: 1px solid var(--line); border-radius: 14px; padding: 1.1rem 1.2rem; }
    div[data-testid="stMetricLabel"] { color: var(--muted); font-size: .74rem; }
    div[data-testid="stMetricValue"] { color: #f4faf7; }
    .stDataFrame { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    .stDownloadButton button { background: var(--green); color: #07120c; border: 0; border-radius: 9px; font-weight: 700; }
    .stDownloadButton button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(117,240,168,.16); }
    </style>
""", unsafe_allow_html=True)

render_sidebar("forecast")

st.markdown('<div class="eyebrow">PCA / Forecast lab</div>', unsafe_allow_html=True)
st.markdown('<div class="forecast-intro"><div class="forecast-intro-copy"><h1>The next signal, projected.</h1><div class="hero-copy">A forward-looking view of the hybrid model\'s 20-quarter projection, grounded in the last real surveillance point and clearly separated from observed history.</div></div><div class="forecast-badge">20-quarter horizon</div></div>', unsafe_allow_html=True)
st.markdown('<div style="height: 1.3rem"></div>', unsafe_allow_html=True)
st.caption(
    "Genuine forecasts beyond the last real data point (2025 Q3), produced by retraining "
    "the hybrid model on all 71 real quarters. Future Bi-LSTM residuals are generated "
    "autoregressively, so forecasts further into the future carry more uncertainty."
)
st.markdown("---")

df = get_quarterly_dataframe()
real_values = df["new_cases"].values.astype(float)

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
    template="plotly_dark", paper_bgcolor="#111817", plot_bgcolor="#111817",
    margin=dict(l=20, r=20, t=30, b=20), height=480, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(title="Quarter", gridcolor="#26302d", tickangle=-45, zeroline=False,
               tickmode="array",
               tickvals=list(range(0, len(hist_labels_x) + len(future_labels_x), 4)),
               ticktext=(hist_labels_x + future_labels_x)[::4]),
    yaxis=dict(title="New HIV Cases per Quarter", gridcolor="#26302d", zeroline=False),
)
st.plotly_chart(fig_future, width="stretch")

future_df = pd.DataFrame({
    "Year": [yr for yr, q in future_results["future_labels"]],
    "Quarter": [q for yr, q in future_results["future_labels"]],
    "SICA-only": np.round(future_results["sica_future"], 1),
    "Hybrid forecast": np.round(future_results["hybrid_future"], 1),
})
st.markdown('<div class="section-kicker">Projection detail</div>', unsafe_allow_html=True)
st.subheader("Quarterly forecast")
st.dataframe(future_df, width="stretch")

annual_forecast = future_df.groupby("Year")["Hybrid forecast"].sum().round(0)
st.markdown('<div class="section-kicker">Annual outlook</div>', unsafe_allow_html=True)
st.subheader("Forecast totals by year")

total_forecast = int(annual_forecast.sum())
first_year = int(annual_forecast.index[0])
last_year = int(annual_forecast.index[-1])
st.markdown(
    f'<div class="outlook-summary"><div><div class="outlook-summary-label">Projected new cases · {first_year}-{last_year}</div><div class="outlook-summary-value">{total_forecast:,}</div></div><div class="outlook-summary-note">Hybrid model estimate<br>Aggregated from 20 quarterly projections</div></div>',
    unsafe_allow_html=True,
)

last_row = df.iloc[-1]
ann_cols = st.columns(len(annual_forecast))
previous_total = None
for col, (yr, total) in zip(ann_cols, annual_forecast.items()):
    with col:
        change_markup = "Baseline year"
        change_class = ""
        if previous_total is not None:
            change = ((total - previous_total) / previous_total) * 100
            change_markup = f"{change:+.1f}% vs {int(yr) - 1}"
            change_class = "up" if change >= 0 else "down"
        st.markdown(
            f'<div class="annual-card"><div class="annual-year">{int(yr)}</div><div class="annual-value">{int(total):,}</div><div class="annual-meta {change_class}">{change_markup}</div></div>',
            unsafe_allow_html=True,
        )
        year_pdf_bytes = build_forecast_pdf(
            future_results,
            last_real_year=last_row["year"],
            last_real_quarter=last_row["quarter"],
            last_real_value=real_values[-1],
            year_filter=int(yr),
        )
        st.download_button(
            label="Download PDF",
            data=year_pdf_bytes,
            file_name=f"HIV_Forecast_Report_{int(yr)}.pdf",
            mime="application/pdf",
            key=f"download_{yr}",
            width="stretch",
        )
        previous_total = total

full_pdf_bytes = build_forecast_pdf(
    future_results,
    last_real_year=last_row["year"],
    last_real_quarter=last_row["quarter"],
    last_real_value=real_values[-1],
)
st.markdown('<div class="full-report"><div class="full-report-title">Need the complete outlook?</div><div class="full-report-copy">Download the full 2026-2030 forecast with quarterly detail and methodology notes.</div></div>', unsafe_allow_html=True)
st.download_button(
    label="Download full report · PDF",
    data=full_pdf_bytes,
    file_name="HIV_Forecast_Report_2026-2030.pdf",
    mime="application/pdf",
    width="stretch",
)