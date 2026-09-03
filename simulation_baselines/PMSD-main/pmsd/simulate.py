"""
Recursive stock-flow simulation over the test horizon (Definition 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .equations import FittedEquation


def simulate_test_horizon(
    history: pd.DataFrame,
    arrival_rate_forecast: pd.Series,
    finish_rate_eq: FittedEquation,
    throughput_time_eq: FittedEquation,
) -> pd.DataFrame:
    """
    Walk the fitted equations forward over `arrival_rate_forecast`'s index.

    Returns a dataframe indexed like arrival_rate_forecast with columns
    concurrent_cases, arrival_rate, finish_rate, throughput_time.
    """
    sim = history.copy()
    last_cc = float(history["concurrent_cases"].iloc[-1])
    last_finish = float(history["finish_rate"].iloc[-1])

    rows = []
    for date, arrival in arrival_rate_forecast.items():
        sim.loc[date] = np.nan
        sim.loc[date, "arrival_rate"] = float(arrival)

        # concurrent_cases first: it only ever needs arrival_rate[i]
        # (exogenous, just set above) and finish_rate[i-1] (already known
        # from the previous step) — never a same-day value — so it can
        # always go first and be safely used as a same-day (shift=0)
        # predictor by the two equations below.
        cc = max(0.0, last_cc + float(arrival) - last_finish)  # a case count can't go negative
        sim.loc[date, "concurrent_cases"] = cc

        tt = throughput_time_eq.predict_row(sim)
        sim.loc[date, "throughput_time"] = tt

        finish = finish_rate_eq.predict_row(sim)
        sim.loc[date, "finish_rate"] = finish

        rows.append({
            "date": date,
            "concurrent_cases": cc,
            "arrival_rate": float(arrival),
            "finish_rate": finish,
            "throughput_time": tt,
        })
        last_cc, last_finish = cc, finish

    return pd.DataFrame(rows).set_index("date")
