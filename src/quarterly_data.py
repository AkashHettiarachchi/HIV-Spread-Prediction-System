"""
Real Sri Lanka HIV quarterly surveillance data, compiled directly from
National STD/AIDS Control Programme (NSACP), Ministry of Health,
quarterly surveillance update reports (2008 Q1 - 2025 Q3, 71 quarters).

This REPLACES the earlier real_data.py, which only had ~10 annual
anchor points from a single secondary source (Suranga et al. 2020).
This is primary-source, quarterly-resolution data direct from NSACP's
own published quarterly reports -- a substantially stronger dataset,
now extended back to 2008 with a second batch of source documents.

Source documents (all NSACP official quarterly updates, No. 29, De
Saram Place, Colombo 10):
    - Update 1st Quarter 2009 through Update 4th Quarter 2016
      (a combined 2009-2016 PDF covering 2008 Q1 - 2016 Q3, with
      successive quarterly reports each restating and sometimes
      revising prior quarters) -> 2008 Q1 - 2016 Q3
    - UPDATE 4th QUARTER 2017 (2018-01-09) -> 2016 Q1-Q4, 2017 Q1-Q4
    - UPDATE 4th QUARTER 2018 (2019-03-08) -> 2017 Q1-Q4, 2018 Q1-Q4
    - UPDATE 2nd/3rd/4th QUARTER 2019 -> 2018-2019 data
    - UPDATE 2nd/3rd/4th QUARTER 2020 -> 2019-2020 data
    - UPDATE 2nd/3rd/4th QUARTER 2021 -> 2020-2021 data
    - HIV/AIDS Quarterly Report 4th Quarter 2023 (V3) -> 2022-2023 data
    - HIV-AIDS Quarterly Report Q1-Q4 2024 (Cleaned data) -> 2023-2024
    - HIV-AIDS Quarterly Report Q1 2025 Final, Q2 2025, Q3 2025 -> 2025
    - Cross-validated against independent Epidemiology Unit STD/HIV
      quarterly bulletins (separate report series, same underlying
      clinic data) for 2017 Q3, 2022 Q1 -- figures matched exactly.

A NOTE ON DATA REVISIONS (state this in your methodology --
it is a real, citable characteristic of surveillance data, not a
flaw you introduced): a small number of quarters were revised
between one NSACP report and a later one (e.g. 2019 Q3 was reported
as 114 in the Q3 2019 update itself, then revised to 112 in the
subsequent Q4 2019 and Q3 2020 updates; 2020 Q3 was reported as 104
in the Q3 2020 update, then revised to 105 in later updates). Where
this happened, the LATER report's figure is used here, since later
reports reflect corrected/reconciled case counts. This is itself a
legitimate methodological point: real surveillance data is revised
over time, which is part of why a hybrid model that doesn't overfit
to any single data vintage is valuable.
"""

import numpy as np
import pandas as pd

# (year, quarter, new_cases_reported, source_report)
QUARTERLY_DATA = [
    (2008, 1, 16, "Update 1st Quarter 2009 (2008-2016 combined PDF)"),
    (2008, 2, 23, "Update 1st Quarter 2009 (2008-2016 combined PDF)"),
    (2008, 3, 33, "Update 1st Quarter 2009 (2008-2016 combined PDF)"),
    (2008, 4, 30, "Update 1st Quarter 2009 (2008-2016 combined PDF)"),

    (2009, 1, 40, "Update 4th Quarter 2009 / 1st Quarter 2010"),
    (2009, 2, 28, "Update 4th Quarter 2009 / 1st Quarter 2010"),
    (2009, 3, 34, "Update 4th Quarter 2009 / 1st Quarter 2010"),
    (2009, 4, 35, "Update 1st Quarter 2010"),

    (2010, 1, 27, "Update 2nd Quarter 2010"),
    (2010, 2, 26, "Update 2nd Quarter 2010"),
    (2010, 3, 36, "Update 4th Quarter 2010"),
    (2010, 4, 32, "Update 4th Quarter 2010"),

    (2011, 1, 32, "Update 1st Quarter 2011"),
    (2011, 2, 38, "Update 3rd Quarter 2011 (revised from 39)"),
    (2011, 3, 42, "Update 4th Quarter 2011"),
    (2011, 4, 34, "Update 1st Quarter 2012"),

    (2012, 1, 40, "Update 2nd Quarter 2012"),
    (2012, 2, 41, "Update 3rd Quarter 2012"),
    (2012, 3, 53, "Update 4th Quarter 2012"),
    (2012, 4, 52, "Update 1st Quarter 2013"),

    (2013, 1, 44, "Update 2nd Quarter 2013"),
    (2013, 2, 46, "Update 3rd Quarter 2013"),
    (2013, 3, 69, "Update 4th Quarter 2013"),
    (2013, 4, 37, "Update 1st Quarter 2014"),

    (2014, 1, 51, "Update 2nd Quarter 2014"),
    (2014, 2, 59, "Update 3rd Quarter 2014"),
    (2014, 3, 59, "Update 4th Quarter 2014"),
    (2014, 4, 60, "Update 1st Quarter 2015"),

    (2015, 1, 59, "Update 2nd Quarter 2015"),
    (2015, 2, 47, "Update 2nd Quarter 2015"),
    (2015, 3, 61, "Update 1st Quarter 2016"),
    (2015, 4, 67, "Update 4th Quarter 2016 (revised from 68)"),

    (2016, 1, 66, "UPDATE 4th Quarter 2017"),
    (2016, 2, 58, "UPDATE 4th Quarter 2017"),
    (2016, 3, 67, "UPDATE 4th Quarter 2017"),
    (2016, 4, 58, "UPDATE 4th Quarter 2017"),

    (2017, 1, 73, "UPDATE 4th Quarter 2018"),
    (2017, 2, 58, "UPDATE 4th Quarter 2018"),
    (2017, 3, 78, "UPDATE 4th Quarter 2018"),
    (2017, 4, 76, "UPDATE 4th Quarter 2018"),

    (2018, 1, 90, "UPDATE 4th Quarter 2019"),
    (2018, 2, 86, "UPDATE 4th Quarter 2019"),
    (2018, 3, 99, "UPDATE 4th Quarter 2019"),
    (2018, 4, 75, "UPDATE 4th Quarter 2019"),

    (2019, 1, 100, "UPDATE 4th Quarter 2019"),
    (2019, 2, 101, "UPDATE 4th Quarter 2019"),
    (2019, 3, 112, "UPDATE 4th Quarter 2019 / 3rd Quarter 2020 (revised from 114)"),
    (2019, 4, 126, "UPDATE 4th Quarter 2019"),

    (2020, 1, 109, "UPDATE 4th Quarter 2021"),
    (2020, 2, 76, "UPDATE 4th Quarter 2021"),
    (2020, 3, 105, "UPDATE 4th Quarter 2021 (revised from 104)"),
    (2020, 4, 73, "UPDATE 4th Quarter 2021"),

    (2021, 1, 79, "UPDATE 4th Quarter 2021"),
    (2021, 2, 69, "UPDATE 4th Quarter 2021"),
    (2021, 3, 103, "UPDATE 4th Quarter 2021"),
    (2021, 4, 159, "UPDATE 4th Quarter 2021"),

    (2022, 1, 152, "HIV/AIDS Quarterly Report 4th Quarter 2023 V3"),
    (2022, 2, 130, "HIV/AIDS Quarterly Report 4th Quarter 2023 V3"),
    (2022, 3, 145, "HIV/AIDS Quarterly Report 4th Quarter 2023 V3"),
    (2022, 4, 180, "HIV/AIDS Quarterly Report 4th Quarter 2023 V3"),

    (2023, 1, 165, "HIV-AIDS Quarterly Report Q4 2024 (Cleaned)"),
    (2023, 2, 181, "HIV-AIDS Quarterly Report Q4 2024 (Cleaned)"),
    (2023, 3, 139, "HIV-AIDS Quarterly Report Q4 2024 (Cleaned)"),
    (2023, 4, 209, "HIV-AIDS Quarterly Report Q4 2024 (Cleaned)"),

    (2024, 1, 186, "HIV-AIDS Quarterly Report Q4 2024 (Cleaned)"),
    (2024, 2, 214, "HIV-AIDS Quarterly Report Q4 2024 (Cleaned)"),
    (2024, 3, 205, "HIV-AIDS Quarterly Report Q4 2024 (Cleaned)"),
    (2024, 4, 219, "HIV-AIDS Quarterly Report Q4 2024 (Cleaned)"),

    (2025, 1, 230, "Q3 2025 report"),
    (2025, 2, 200, "Q3 2025 report"),
    (2025, 3, 209, "Q3 2025 report"),
]

# Point-estimate context (from the surveillance reports, useful for
# sanity-checking / discussion, not used directly as model input):
PLHIV_ESTIMATES_BY_YEAR = {
    2016: 3900, 2017: 3500, 2018: 3500, 2019: 3600, 2020: 3700,
    2022: 4100, 2023: 4700, 2024: 5700,
}


def get_quarterly_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(QUARTERLY_DATA, columns=["year", "quarter", "new_cases", "source"])
    df["period_index"] = range(len(df))  # 0-based sequential quarter index, for modelling
    df["year_frac"] = df["year"] + (df["quarter"] - 1) / 4.0  # decimal year, for SICA time axis
    return df


if __name__ == "__main__":
    df = get_quarterly_dataframe()
    print(df[["year", "quarter", "new_cases", "year_frac"]].to_string(index=False))
    print(f"\nTotal quarters: {len(df)}")
    print(f"Date range: {df['year'].min()} Q{df.iloc[0]['quarter']} "
          f"to {df['year'].max()} Q{df.iloc[-1]['quarter']}")
    yearly = df.groupby("year")["new_cases"].sum()
    print("\nAnnual totals (for cross-check against Suranga et al. 2020 style annual figures):")
    print(yearly.to_string())