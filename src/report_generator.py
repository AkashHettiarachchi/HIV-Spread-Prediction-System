"""
Generates a downloadable PDF report of the HIV forecast, styled to
resemble the official NSACP quarterly report format (header box with
key stats, then a data table, then a summary section) so it reads as
a familiar, professional document when shown to your supervisor or
included as a dissertation appendix.

Uses reportlab (pip install reportlab) -- pure Python, no external
binaries needed, works the same on Windows/Mac/Linux.
"""

import io
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
)


def build_forecast_pdf(future_results, last_real_year, last_real_quarter, last_real_value,
                        year_filter: int = None) -> bytes:
    """
    future_results: the dict returned by forecast_future.forecast_future()
    year_filter: if given (e.g. 2027), the report covers ONLY that year's
                 4 quarters instead of the full 2026-2030 horizon.
    Returns raw PDF bytes, ready for st.download_button.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], fontSize=14, spaceAfter=4, alignment=1,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleCustom", parent=styles["Normal"], fontSize=10, alignment=1,
        textColor=colors.HexColor("#333333"), spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "SectionCustom", parent=styles["Heading2"], fontSize=11, spaceBefore=12, spaceAfter=6,
    )
    note_style = ParagraphStyle(
        "NoteCustom", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#555555"),
        spaceBefore=6,
    )

    elements = []

    # ---- Determine which quarters this report covers ----
    if year_filter is not None:
        filtered = [
            (label, sica_val, hybrid_val)
            for label, sica_val, hybrid_val in zip(
                future_results["future_labels"], future_results["sica_future"], future_results["hybrid_future"]
            )
            if label[0] == year_filter
        ]
        labels = [f[0] for f in filtered]
        sica_vals = [f[1] for f in filtered]
        hybrid_vals = [f[2] for f in filtered]
        horizon_text = f"{year_filter} Q1 &ndash; {year_filter} Q4 (4 quarters)"
        report_title_suffix = f" &mdash; {year_filter}"
    else:
        labels = future_results["future_labels"]
        sica_vals = list(future_results["sica_future"])
        hybrid_vals = list(future_results["hybrid_future"])
        horizon_text = "2026 Q1 &ndash; 2030 Q4 (20 quarters)"
        report_title_suffix = ""

    # ---- Header ----
    elements.append(Paragraph(f"HIV/AIDS Incidence Forecast Report{report_title_suffix}", title_style))
    elements.append(Paragraph(
        "Hybrid SICA + Bi-LSTM Forecasting Model &mdash; Sri Lanka", subtitle_style
    ))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black))
    elements.append(Spacer(1, 8))

    # ---- Key stats box (mirrors the NSACP "HIV statistics" box style) ----
    stats_data = [
        ["Last real data point", f"{last_real_year:.0f} Q{last_real_quarter:.0f}  "
                                  f"({last_real_value:.0f} reported cases)"],
        ["Forecast horizon", horizon_text],
        ["Model", "Hybrid: calibrated SICA compartmental model "
                  "+ Bidirectional LSTM residual correction"],
        ["Data source", "National STD/AIDS Control Programme (NSACP), "
                         "Ministry of Health, Sri Lanka &mdash; quarterly surveillance reports"],
        ["Report generated", date.today().strftime("%Y-%m-%d")],
    ]
    stats_table = Table(
        [[Paragraph(f"<b>{k}</b>", styles["Normal"]), Paragraph(v, styles["Normal"])]
         for k, v in stats_data],
        colWidths=[5 * cm, 11.5 * cm],
    )
    stats_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F0F0")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.black),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 14))

    # ---- Quarterly forecast table ----
    table_title = (f"Quarterly Forecast: {year_filter} Q1 &ndash; {year_filter} Q4"
                    if year_filter else "Quarterly Forecast: 2026 Q1 &ndash; 2030 Q4")
    elements.append(Paragraph(table_title, section_style))

    header_row = ["Year", "Quarter", "SICA-only\n(cases)", "Hybrid Forecast\n(cases)"]
    q_rows = [header_row]
    for (yr, q), sica_val, hybrid_val in zip(labels, sica_vals, hybrid_vals):
        q_rows.append([str(yr), f"Q{q}", f"{sica_val:.0f}", f"{hybrid_val:.0f}"])

    q_table = Table(q_rows, colWidths=[3 * cm, 3 * cm, 5 * cm, 5.5 * cm], repeatRows=1)
    q_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.append(q_table)
    elements.append(Spacer(1, 14))

    # ---- Annual totals table ----
    totals_title = "Annual Total (Hybrid Model)" if year_filter else "Annual Forecast Totals (Hybrid Model)"
    elements.append(Paragraph(totals_title, section_style))
    annual = {}
    for (yr, q), hybrid_val in zip(labels, hybrid_vals):
        annual[yr] = annual.get(yr, 0) + hybrid_val

    annual_rows = [["Year", "Forecasted New HIV Cases"]]
    for yr, total in annual.items():
        annual_rows.append([str(yr), f"{total:.0f}"])

    annual_table = Table(annual_rows, colWidths=[8 * cm, 8.5 * cm])
    annual_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    elements.append(annual_table)
    elements.append(Spacer(1, 14))

    # ---- Methodology / caveats note ----
    elements.append(Paragraph(
        "Methodology &amp; Limitations", section_style
    ))
    elements.append(Paragraph(
        "Forecasts are produced by a hybrid model: a SICA (Susceptible-Infectious-Chronic-AIDS) "
        "compartmental model calibrated on 71 real quarterly data points (2008 Q1&ndash;2025 Q3) "
        "supplies a mechanistic baseline trend, and a Bidirectional LSTM neural network, trained on "
        "the residual between real data and this baseline, corrects for patterns the mechanistic "
        "model cannot capture. On held-out real data (2024&ndash;2025), this hybrid approach achieved "
        "a Mean Absolute Percentage Error of 7.00%, compared to 26.90% for the SICA model alone.",
        styles["Normal"],
    ))
    elements.append(Paragraph(
        "Future quarters (2026&ndash;2030) are generated by an autoregressive rollout: each "
        "predicted quarter feeds into the input for the next. Forecast uncertainty increases with "
        "horizon &mdash; near-term forecasts (2026) are more reliable than longer-term ones (2030). "
        "These figures are model projections for research purposes and should not be used as the "
        "sole basis for policy or clinical decisions without independent verification.",
        note_style,
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


if __name__ == "__main__":
    # Smoke test: generate a full-horizon PDF and a single-year PDF
    from forecast_future import forecast_future

    results = forecast_future(n_future_quarters=20)
    last_row = results["df_historical"].iloc[-1]

    pdf_bytes_full = build_forecast_pdf(
        results,
        last_real_year=last_row["year"],
        last_real_quarter=last_row["quarter"],
        last_real_value=results["real_historical"][-1],
    )
    print(f"Full-horizon PDF generated: {len(pdf_bytes_full)} bytes")

    pdf_bytes_2027 = build_forecast_pdf(
        results,
        last_real_year=last_row["year"],
        last_real_quarter=last_row["quarter"],
        last_real_value=results["real_historical"][-1],
        year_filter=2027,
    )
    print(f"Single-year (2027) PDF generated: {len(pdf_bytes_2027)} bytes")