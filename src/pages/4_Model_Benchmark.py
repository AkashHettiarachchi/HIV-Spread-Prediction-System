from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from benchmark_models import (
    HYBRID, SICA, ARIMA, VLSTM, SICA_LSTM, NAIVE,
    METRIC_LABELS, MODE_LABELS,
    BenchmarkConfig, run_benchmark, results_to_dataframe, summarize, verdict_lines,
)
from quarterly_data import get_quarterly_dataframe
from sidebar import render_sidebar


st.set_page_config(
    page_title="Preditca | Model Benchmark",
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
    .insight { background: #101716; border: 1px solid var(--line); border-left: 3px solid var(--green); border-radius: 0 12px 12px 0; padding: .85rem 1rem; color: #aabbb4; font-size: .86rem; line-height: 1.55; }
    .verdict { background: #101716; border: 1px solid var(--line); border-left: 3px solid #5f7a6c; border-radius: 0 12px 12px 0; padding: .7rem 1rem; margin-bottom: .55rem; color: #c3d1ca; font-size: .9rem; line-height: 1.5; }
    .verdict.good { border-left-color: #75f0a8; }
    .verdict.warn { border-left-color: #f0ad77; }
    .stDataFrame { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    .stButton button, .stDownloadButton button { border-radius: 9px; }
    .stButton button[kind="primary"] { background: var(--green); color: #07120c; border: 0; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

render_sidebar("benchmark")

df_scope = get_quarterly_dataframe()
total_quarters = len(df_scope)
first_period = f"{int(df_scope.iloc[0]['year'])} Q{int(df_scope.iloc[0]['quarter'])}"
last_period = f"{int(df_scope.iloc[-1]['year'])} Q{int(df_scope.iloc[-1]['quarter'])}"
benchmark_n_test = BenchmarkConfig().n_test
benchmark_n_train = total_quarters - benchmark_n_test
test_start_period = f"{int(df_scope.iloc[-benchmark_n_test]['year'])} Q{int(df_scope.iloc[-benchmark_n_test]['quarter'])}"

MODEL_COLORS = {
    HYBRID: "#00E676",
    SICA: "#FFA500",
    ARIMA: "#4FC3F7",
    VLSTM: "#B39DDB",
    SICA_LSTM: "#F06292",
    NAIVE: "#8A99AD",
}
ROLE_LABELS = {"proposed": "Proposed", "baseline": "Baseline", "reference": "Reference"}
LOWER_IS_BETTER = {"mae", "rmse", "mape"}

# ----------------------------- header -----------------------------
st.markdown('<div class="eyebrow">PCA / Benchmark lab</div>', unsafe_allow_html=True)
st.title("How does the hybrid compare?")
st.markdown(
    '<div class="hero-copy">The hybrid SICA + Bi-LSTM model and its competitors, scored on the same '
    'held-out NSACP quarters. Everything below is computed live from the data, so it shows what the '
    'data supports, whichever model comes out ahead.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div style="height: 1.1rem"></div>', unsafe_allow_html=True)

st.info(
    f"**How the comparison is kept fair.** All models are fitted on {first_period} - "
    f"the quarter before {test_start_period} ({benchmark_n_train} quarters) and scored on "
    f"{test_start_period} - {last_period} ({benchmark_n_test} quarters, never seen while fitting). SICA is calibrated on the training quarters only. "
    "The three neural models share one recipe (window, units, dropout, optimiser, early stopping on a "
    "chronological validation slice) and are trained with several random seeds, so scores are "
    "mean \u00b1 spread, not a single lucky run. Because the SICA calibration here excludes the test "
    "quarters, these numbers will differ from the overview page."
)

# ----------------------------- controls -----------------------------
st.markdown('<div class="section-kicker">Run the comparison</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1.3, 1.7, 1])
with c1:
    n_seeds = st.slider(
        "Training seeds per neural model", 1, 10, 3,
        help="Each seed trains a fresh network. More seeds give steadier averages but take longer.",
    )
with c2:
    include_ref = st.checkbox(
        "Add reference models (SICA + LSTM, naive last-value)", value=True,
        help="SICA + LSTM isolates what the bidirectional layer adds. The naive forecast repeats the last "
             "observed quarter and shows whether any model is doing real work.",
    )
with c3:
    st.markdown('<div style="height: 1.7rem"></div>', unsafe_allow_html=True)
    run_clicked = st.button("Run benchmark", type="primary", width="stretch")
st.caption("Each seed trains up to three networks, so a run can take a few minutes. Results stay on the page until you run again.")

if run_clicked:
    bar = st.progress(0.0, text="Starting benchmark...")

    def _progress(frac: float, msg: str) -> None:
        bar.progress(float(min(max(frac, 0.0), 1.0)), text=msg)

    try:
        st.session_state["bench_results"] = run_benchmark(
            BenchmarkConfig(n_seeds=n_seeds, include_reference=include_ref),
            progress=_progress,
        )
    finally:
        bar.empty()

res = st.session_state.get("bench_results")
if res is None:
    st.markdown(
        '<div class="insight">No results yet. Choose the number of seeds and press <strong>Run benchmark</strong>.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ----------------------------- warnings for models that did not run -----------------------------
for name, msg in res["errors"].items():
    hint = ""
    if "statsmodels" in msg:
        hint = " Install it with `pip install statsmodels`."
    elif "tensorflow" in msg.lower():
        hint = " Install it with `pip install tensorflow`."
    st.warning(f"{name} did not run: {msg}.{hint}")

if not res["models"]:
    st.error("No model produced results.")
    st.stop()

# ----------------------------- view options -----------------------------
v1, v2 = st.columns([2, 1])
with v1:
    mode = st.radio(
        "Evaluation mode", list(MODE_LABELS), format_func=lambda k: MODE_LABELS[k], horizontal=True,
        help="One-step: each quarter is predicted from the real history before it. "
               f"Multi-step: all {benchmark_n_test} quarters are forecast blind from the end of training, "
               "which matches the forward-forecast evaluation design.",
    )
with v2:
    metric = st.selectbox("Ranking metric", list(METRIC_LABELS), index=1, format_func=lambda k: METRIC_LABELS[k])

# ----------------------------- verdict -----------------------------
st.markdown('<div class="section-kicker">What the numbers say</div>', unsafe_allow_html=True)
st.subheader("Summary")
for level, text in verdict_lines(res, mode, metric):
    css = {"good": "good", "warn": "warn"}.get(level, "")
    st.markdown(f'<div class="verdict {css}">{text}</div>', unsafe_allow_html=True)

# ----------------------------- scorecard -----------------------------
st.markdown('<div class="section-kicker">Scorecard</div>', unsafe_allow_html=True)
st.subheader("Error on the held-out quarters")
df = results_to_dataframe(res, mode)

best = {}
for k in METRIC_LABELS:
    best[k] = df[k].min() if k in LOWER_IS_BETTER else df[k].max()


def _cell(row, k: str) -> str:
    v, s = row[k], row[k + "_std"]
    txt = f"{v:.2f} \u00b1 {s:.2f}" if s > 0 else f"{v:.2f}"
    return txt + (" \u2605" if np.isclose(v, best[k]) else "")


table = pd.DataFrame({
    "Model": df["Model"],
    "Role": df["Role"].map(ROLE_LABELS),
    "Runs": df["Runs"],
    **{METRIC_LABELS[k]: df.apply(lambda r, k=k: _cell(r, k), axis=1) for k in METRIC_LABELS},
})
st.dataframe(table, width="stretch", hide_index=True)
st.caption("\u2605 marks the best value in each column. MAE and RMSE are in cases per quarter. "
           "\u00b1 is the spread across training seeds (deterministic models have none). "
           f"R\u00b2 is computed on only {benchmark_n_test} points, so it can be negative.")

fig_bar = go.Figure(go.Bar(
    x=df["Model"], y=df[metric],
    error_y=dict(type="data", array=df[metric + "_std"], visible=True, color="#c3d1ca"),
    marker_color=[MODEL_COLORS.get(n, "#8A99AD") for n in df["Model"]],
    text=[f"{v:.2f}" for v in df[metric]], textposition="outside",
))
fig_bar.update_layout(
    template="plotly_dark", paper_bgcolor="#111817", plot_bgcolor="#111817",
    margin=dict(l=20, r=20, t=30, b=20), height=380, showlegend=False,
    xaxis=dict(title="", gridcolor="#26302d"),
    yaxis=dict(title=METRIC_LABELS[metric] + (" (lower is better)" if metric in LOWER_IS_BETTER else " (higher is better)"),
               gridcolor="#26302d", zeroline=False),
)
st.plotly_chart(fig_bar, width="stretch")

# ----------------------------- forecast lines -----------------------------
st.markdown('<div class="section-kicker">Test quarters</div>', unsafe_allow_html=True)
st.subheader("Predictions against reported cases")

n_train = res["n_train"]
ctx = 12
x_ctx = res["labels_full"][n_train - ctx:]
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=x_ctx, y=res["actual_full"][n_train - ctx:], mode="lines+markers", name="Reported (NSACP)",
    line=dict(color="#EEF5F1", width=2), marker=dict(size=6),
))
for name, m in res["models"].items():
    is_hybrid = name == HYBRID
    fig_line.add_trace(go.Scatter(
        x=res["labels_test"], y=m[mode]["pred"], mode="lines+markers", name=name,
        line=dict(color=MODEL_COLORS.get(name, "#8A99AD"), width=3.5 if is_hybrid else 1.8,
                  dash="solid" if m["role"] != "reference" else "dot"),
        marker=dict(size=7 if is_hybrid else 5),
    ))
fig_line.add_vline(x=ctx - 0.5, line_dash="dot", line_color="#E91E63",
                   annotation_text="train / test split", annotation_position="top")
fig_line.update_layout(
    template="plotly_dark", paper_bgcolor="#111817", plot_bgcolor="#111817",
    margin=dict(l=20, r=20, t=30, b=20), height=460, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(title="Quarter", gridcolor="#26302d", tickangle=-45),
    yaxis=dict(title="New HIV cases per quarter", gridcolor="#26302d", zeroline=False),
)
st.plotly_chart(fig_line, width="stretch")

qdf = pd.DataFrame({"Quarter": res["labels_test"], "Reported": res["actual_test"].astype(int)})
for name, m in res["models"].items():
    qdf[name] = np.round(m[mode]["pred"], 1)
st.dataframe(qdf, width="stretch", hide_index=True)

summary = summarize(res, mode, metric)
wins = summary.get("quarter_wins", {})
if wins:
    win_df = pd.DataFrame({
        "Compared with": list(wins.keys()),
        "Quarters where the hybrid was closer": [f"{w} of {n}" for w, n in wins.values()],
    })
    st.markdown("**Quarter by quarter**")
    st.dataframe(win_df, width="stretch", hide_index=True)
    st.caption("Uses the average prediction across seeds. A model that wins on total error can still lose most individual quarters, and the reverse.")

# ----------------------------- diagnostics -----------------------------
with st.expander("Diagnostics and settings"):
    info = res.get("sica_info") or {}
    if info:
        st.write(f"SICA calibrated on training quarters only: "
                 f"\u03b2 = **{info['beta']:.5f}**, \u03c1 = **{info['rho']:.5f}**, \u03b1 = **{info['alpha']:.5f}**")
        st.write(f"In-sample MAPE on the training quarters: **{info['in_sample_mape']:.1f}%**")
        if info["at_bounds"]:
            st.warning(
                "These parameters sit on their calibration bounds: " + ", ".join(info["at_bounds"]) +
                ". That usually means the parameters are not identifiable from the data, so the SICA "
                "baseline is a weak anchor and the neural network may be doing most of the work."
            )
    cfg = res["config"]
    st.write(
        f"Window {cfg.window} quarters, {cfg.units} units, dropout {cfg.dropout}, Adam {cfg.learning_rate}, "
        f"batch {cfg.batch_size}, up to {cfg.epochs} epochs, patience {cfg.patience}, "
        f"validation slice {cfg.val_sequences} windows, {cfg.n_seeds} seeds, ARIMA order {cfg.arima_order}."
    )

both = pd.concat(
    [results_to_dataframe(res, m).assign(Mode=m) for m in MODE_LABELS], ignore_index=True
)
st.download_button(
    "Download scorecard (CSV)", data=both.to_csv(index=False).encode("utf-8"),
    file_name="model_benchmark_scorecard.csv", mime="text/csv",
)

st.markdown(
    '<div class="insight"><strong>Reading these results.</strong> Seven test quarters is a small sample. '
    'Treat gaps smaller than the \u00b1 spread as ties, and prefer a model that wins in both evaluation '
    'modes over one that wins in only one.</div>',
    unsafe_allow_html=True,
)
st.caption("Data source: National STD/AIDS Control Programme (NSACP), Ministry of Health, Sri Lanka, quarterly surveillance reports.")