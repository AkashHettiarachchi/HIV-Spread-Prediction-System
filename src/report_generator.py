"""Create thesis-ready PDF reports for the quarterly HIV forecast."""

import io
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


NAVY = colors.HexColor("#1E3A5F")
BLUE = colors.HexColor("#DCE8F5")
PALE_BLUE = colors.HexColor("#F3F7FB")
GREY = colors.HexColor("#555555")


def _paragraph_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
            fontSize=15, leading=18, alignment=TA_CENTER, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=styles["Normal"], fontSize=9.5,
            leading=12, alignment=TA_CENTER, textColor=GREY, spaceAfter=8,
        ),
        "section": ParagraphStyle(
            "SectionHeading", parent=styles["Heading2"], fontName="Helvetica-Bold",
            fontSize=11, leading=14, textColor=NAVY, spaceBefore=10, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyTextAcademic", parent=styles["BodyText"], fontSize=8.5,
            leading=11, alignment=TA_LEFT, spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "SmallTextAcademic", parent=styles["BodyText"], fontSize=7.5,
            leading=9.5, textColor=GREY, spaceAfter=3,
        ),
        "table_header": ParagraphStyle(
            "TableHeader", parent=styles["Normal"], fontName="Helvetica-Bold",
            fontSize=7.5, leading=9, textColor=colors.white, alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "TableCell", parent=styles["Normal"], fontSize=7.5, leading=9,
            alignment=TA_CENTER,
        ),
    }


def _academic_table(rows, col_widths, repeat_rows=1):
    table = Table(rows, colWidths=col_widths, repeatRows=repeat_rows, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#8A9AAA")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_BLUE]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _selected_forecast_rows(future_results, year_filter):
    rows = zip(
        future_results["future_labels"],
        future_results["sica_future"],
        future_results["hybrid_future"],
        future_results["hybrid_lower"],
        future_results["hybrid_upper"],
    )
    if year_filter is not None:
        rows = (row for row in rows if row[0][0] == year_filter)
    return list(rows)


def _technical_specification(styles, horizon_text, last_real_text):
    rows = [
        [Paragraph("Technical specification", styles["table_header"]), ""],
        [Paragraph("Model architecture", styles["table_cell"]),
         Paragraph("Hybrid SICA (compartmental ODE) + Bidirectional LSTM residual correction.", styles["table_cell"])],
        [Paragraph("Data source and scope", styles["table_cell"]),
         Paragraph("National STD/AIDS Control Programme (NSACP), Ministry of Health, Sri Lanka; 71 quarters (2008 Q1 – 2025 Q3).", styles["table_cell"])],
        [Paragraph("Hyperparameters", styles["table_cell"]),
         Paragraph("Bi-LSTM units = 16; lookback window <i>w</i> = 4 quarters; dropout = 0.2; optimizer = Adam (learning rate = 5 × 10<super>−3</super>).", styles["table_cell"])],
        [Paragraph("Forecast horizon", styles["table_cell"]),
         Paragraph(f"{horizon_text} with 95% confidence intervals.", styles["table_cell"])],
        [Paragraph("Reference observation", styles["table_cell"]),
         Paragraph(last_real_text, styles["table_cell"])],
    ]
    table = Table(rows, colWidths=[4.2 * cm, 12.3 * cm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("SPAN", (0, 0), (-1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (0, -1), BLUE),
        ("BOX", (0, 0), (-1, -1), 0.75, NAVY),
        ("INNERGRID", (0, 1), (-1, -1), 0.4, colors.HexColor("#9AA9B8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def build_forecast_pdf(future_results, last_real_year, last_real_quarter, last_real_value,
                       year_filter: int = None) -> bytes:
    """Return a thesis-appendix PDF for the full horizon or one selected year."""
    styles = _paragraph_styles()
    selected_rows = _selected_forecast_rows(future_results, year_filter)
    horizon_text = (
        f"{year_filter} Q1 – {year_filter} Q4 (4 quarters)"
        if year_filter is not None else "2026 Q1 – 2030 Q4 (20 quarters)"
    )
    title_suffix = f" — {year_filter}" if year_filter is not None else ""
    last_real_text = (
        f"{int(last_real_year)} Q{int(last_real_quarter)} "
        f"({int(last_real_value)} reported cases)"
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, topMargin=1.35 * cm, bottomMargin=1.35 * cm,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        title=f"HIV/AIDS Incidence Forecast Report{title_suffix}",
        author="HIV Spread Prediction System",
        subject="Thesis appendix forecast report",
    )
    elements = [
        Paragraph(f"HIV/AIDS Incidence Forecast Report{title_suffix}", styles["title"]),
        Paragraph("Hybrid SICA + Bi-LSTM Forecasting Model — Sri Lanka", styles["subtitle"]),
        HRFlowable(width="100%", thickness=1, color=NAVY),
        Spacer(1, 7),
        _technical_specification(styles, horizon_text, last_real_text),
        Spacer(1, 9),
    ]

    elements.append(Paragraph("1. Methodology and mathematical formulation", styles["section"]))
    elements.append(Paragraph(
        "The SICA model partitions the population into Susceptible (S), Undiagnosed/Infected (I), "
        "Chronic/ART (C), and AIDS-stage (A) compartments. Its calibrated compartmental ordinary "
        "differential equations provide the mechanistic incidence baseline. The neural component is "
        "trained on the discrepancy between observed incidence and that baseline.", styles["body"]
    ))
    equation_rows = [
        [Paragraph("Quantity", styles["table_header"]), Paragraph("Definition", styles["table_header"])],
        [Paragraph("Residual", styles["table_cell"]), Paragraph("E<sub>t</sub> = Y<sub>actual,t</sub> − Y<sub>SICA,t</sub>", styles["table_cell"])],
        [Paragraph("Hybrid prediction", styles["table_cell"]), Paragraph("Ŷ<sub>t</sub> = Y<sub>SICA,t</sub> + Ê<sub>BiLSTM,t</sub>", styles["table_cell"])],
        [Paragraph("Compounding uncertainty", styles["table_cell"]), Paragraph("SE(t) = σ<sub>e</sub> √[1 + α(t − 1)], where α = 0.05; 95% CI = Ŷ<sub>t</sub> ± 1.96 SE(t)", styles["table_cell"])],
    ]
    elements.append(_academic_table(equation_rows, [4.2 * cm, 12.3 * cm]))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("2. Quarterly forecast results", styles["section"]))
    quarterly_rows = [[
        Paragraph("Year", styles["table_header"]),
        Paragraph("Quarter", styles["table_header"]),
        Paragraph("SICA Baseline", styles["table_header"]),
        Paragraph("Hybrid Forecast", styles["table_header"]),
        Paragraph("95% CI (Lower – Upper)", styles["table_header"]),
    ]]
    for (year, quarter), sica, hybrid, lower, upper in selected_rows:
        quarterly_rows.append([
            Paragraph(str(int(year)), styles["table_cell"]),
            Paragraph(f"Q{int(quarter)}", styles["table_cell"]),
            Paragraph(f"{sica:.0f}", styles["table_cell"]),
            Paragraph(f"{hybrid:.0f}", styles["table_cell"]),
            Paragraph(f"{lower:.0f} – {upper:.0f}", styles["table_cell"]),
        ])
    elements.append(_academic_table(
        quarterly_rows, [2.1 * cm, 2.1 * cm, 3.5 * cm, 3.7 * cm, 6.1 * cm]
    ))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("3. Aggregated annual forecast totals", styles["section"]))
    annual = {}
    baseline_by_year = {}
    for (year, _), sica, hybrid, lower, upper in selected_rows:
        year = int(year)
        baseline_by_year[year] = baseline_by_year.get(year, 0.0) + sica
        totals = annual.setdefault(year, [0.0, 0.0, 0.0])
        totals[0] += hybrid
        totals[1] += lower
        totals[2] += upper
    annual_rows = [[
        Paragraph("Year", styles["table_header"]),
        Paragraph("SICA Baseline Total", styles["table_header"]),
        Paragraph("Hybrid Forecast Total", styles["table_header"]),
        Paragraph("95% CI (Lower – Upper)", styles["table_header"]),
    ]]
    for year, totals in annual.items():
        annual_rows.append([
            Paragraph(str(year), styles["table_cell"]),
            Paragraph(f"{baseline_by_year[year]:.0f}", styles["table_cell"]),
            Paragraph(f"{totals[0]:.0f}", styles["table_cell"]),
            Paragraph(f"{totals[1]:.0f} – {totals[2]:.0f}", styles["table_cell"]),
        ])
    elements.append(_academic_table(
        annual_rows, [2.5 * cm, 4.4 * cm, 4.6 * cm, 6.0 * cm]
    ))

    elements.append(Paragraph("4. Validation summary", styles["section"]))
    elements.append(Paragraph(
        "Chronological out-of-sample validation used a held-out 2024–2025 test window, with SICA "
        "calibration and residual training performed using the training portion only. The reported "
        "reference metrics are SICA MAPE = 26.90% and Hybrid MAPE = 7.00%. These values summarize "
        "the model comparison used by the forecasting study and are presented as validation context, "
        "not as a re-estimate from the future projection arrays in this report.", styles["body"]
    ))

    elements.append(Paragraph("5. Data citation, interpretation, and limitations", styles["section"]))
    elements.append(Paragraph(
        "Data source: National STD/AIDS Control Programme, Ministry of Health, Sri Lanka, quarterly "
        "HIV surveillance reports covering 2008 Q1–2025 Q3. The source series represents reported "
        "cases and should be interpreted in the context of testing coverage, reporting practices, "
        "diagnostic delays, treatment access, and other surveillance-system changes. The exact source "
        "report or data-file identifier should be added to the dissertation reference list alongside "
        "the archived dataset used for model fitting.", styles["body"]
    ))
    elements.append(Paragraph(
        "The 2026–2030 values are model-based projections, not observed counts. Uncertainty increases "
        "with forecast horizon because the residual model is rolled forward autoregressively. These "
        "estimates are suitable for research interpretation and scenario planning, but must not be used "
        "as the sole basis for clinical, funding, or public-health policy decisions without independent "
        "epidemiological review and updated surveillance data.", styles["small"]
    ))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(
        f"Report generated: {date.today().isoformat()} | Forecast horizon: {horizon_text}",
        styles["small"],
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


if __name__ == "__main__":
    from forecast_future import forecast_future

    results = forecast_future(n_future_quarters=20)
    last_row = results["df_historical"].iloc[-1]
    pdf_bytes = build_forecast_pdf(
        results,
        last_real_year=last_row["year"],
        last_real_quarter=last_row["quarter"],
        last_real_value=results["real_historical"][-1],
    )
    print(f"Full-horizon PDF generated: {len(pdf_bytes)} bytes")
