"""
SICA Compartmental Model -- SINGLE SOURCE OF TRUTH.
Compartments:
    S - Susceptible
    I - Infected, undiagnosed / not yet linked to ART
    C - Chronic, diagnosed and on ART ("Controlled")
    A - AIDS stage (advanced disease)
    CumInf - cumulative new infections (auxiliary state, NOT a real
             compartment -- tracks the integral of the S->I flow so
             we can report "new infections per year", which is the
             actual quantity reported in NSACP/AEM/Spectrum data,
             rather than a compartment size that is never directly
             observed.

Reference: Silva, C.J. & Torres, D.F.M. (2016), "A SICA compartmental
model in epidemiology with application to HIV/AIDS in Cape Verde."
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass


@dataclass
class SICAParams:
    Lambda: float = 1000.0   # recruitment rate into susceptible pool (per year)
    beta: float = 0.25       # effective transmission rate
    eta: float = 0.1         # relative infectiousness of chronic/ART stage vs. I
    mu: float = 0.0079       # natural (non-HIV) annual death rate (~1/127 yrs, Sri Lanka life expectancy-based)
    phi: float = 0.02        # rate of undiagnosed (I) stabilizing/being diagnosed without full ART linkage
    rho: float = 0.15        # rate of I -> C (diagnosis + ART linkage)
    omega: float = 0.05      # rate of relapse/A -> C (re-engagement in care)
    gamma: float = 0.2       # rate of C -> C (viral suppression maintenance; reduces effective C->A)
    alpha: float = 0.08      # rate of C -> A (ART failure / progression despite treatment)
    delta: float = 0.25      # additional AIDS-induced death rate


def sica_rhs(t, y, p: SICAParams):
    S, I, C, A, _CumInf = y
    N = S + I + C + A
    if N <= 0:
        N = 1.0

    force_of_infection = p.beta * S * (I + p.eta * C) / N

    dS = p.Lambda - force_of_infection - p.mu * S
    dI = force_of_infection - (p.mu + p.phi + p.rho) * I
    dC = p.rho * I + p.omega * A - (p.mu + p.gamma + p.alpha) * C
    dA = p.alpha * C - (p.mu + p.delta) * A - p.omega * A
    dCumInf = force_of_infection  # tracks total new infections accumulated

    return [dS, dI, dC, dA, dCumInf]


def simulate_sica(params: SICAParams, y0_no_cuminf, t_span, n_points=200):
    """
    y0_no_cuminf: (S0, I0, C0, A0) -- CumInf always starts at 0.
    Returns t (n_points,), y (5, n_points) rows = S, I, C, A, CumInf.
    """
    y0 = list(y0_no_cuminf) + [0.0]
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        fun=lambda t, y: sica_rhs(t, y, params),
        t_span=t_span, y0=y0, t_eval=t_eval,
        method="RK45", rtol=1e-8, atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    return sol.t, sol.y


def annual_new_infections(t: np.ndarray, cuminf: np.ndarray, t_start_year: int):
    """
    Convert the cumulative-infections curve into per-year new-infection
    counts, aligned to calendar years, so it can be compared directly
    against the real AEM/Spectrum annual series.
    """
    years = np.floor(t_start_year + t).astype(int)
    unique_years = np.unique(years)
    annual = []
    for yr in unique_years:
        mask = years == yr
        if mask.sum() < 2:
            continue
        annual.append((yr, cuminf[mask][-1] - cuminf[mask][0]))
    return annual


if __name__ == "__main__":
    params = SICAParams()
    y0 = (16_700_000, 1500, 1668, 437)  # see calibration.py for how these are derived
    t, y = simulate_sica(params, y0, t_span=(0, 7), n_points=7 * 12 + 1)
    S, I, C, A, CumInf = y
    print("Final compartments (S, I, C, A):", y[:4, -1])
    print("Total new infections over 7 years (uncalibrated defaults):", round(CumInf[-1], 1))