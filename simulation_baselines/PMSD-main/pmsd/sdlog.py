"""
Build the SD-Log (Pourbafrani & van der Aalst, Definition 9) for one event log.

"""

from __future__ import annotations

import pandas as pd

_RANGE_FREQ = {"hours": "h", "days": "D", "weeks": "W-MON", "months": "MS"}


def _floor(ts: pd.Series, window: str) -> pd.Series:
    if window == "hours":
        return ts.dt.floor("h")
    if window == "days":
        return ts.dt.normalize()
    if window == "weeks":
        return ts.dt.to_period("W").dt.start_time
    return ts.dt.to_period("M").dt.start_time


def _floor_ts(ts: pd.Timestamp, window: str) -> pd.Timestamp:
    return _floor(pd.Series([ts]), window).iloc[0]


def build_sd_log(
    log: pd.DataFrame,
    time_col: str = "time:timestamp",
    start_col: str = "start:timestamp",
    case_col: str = "case:concept:name",
    activity_col: str = "concept:name",
    resource_col: str = "org:resource",
    window: str = "days",
    cut_date=None,
) -> pd.DataFrame:
    """
    Build the full 9-column SD-Log, indexed by time bucket, up through
    `cut_date` (inclusive). Pass the FULL log — see module docstring.
    """
    from time_series_creation import (
        create_concurrent_cases_timeseries,
        create_avg_throughtput_time_timeseries,
    )

    t = pd.to_datetime(log[time_col])
    if t.dt.tz is not None:
        t = t.dt.tz_convert(None)

    cut_ts = None
    if cut_date is not None:
        cut_ts = pd.Timestamp(cut_date)
        if cut_ts.tzinfo is not None:
            cut_ts = cut_ts.tz_convert(None)

    has_resource = resource_col in log.columns
    ev = pd.DataFrame({
        "_case": log[case_col].values,
        "_act":  log[activity_col].values,
        "_t":    t.values,
    })
    if has_resource:
        ev["_res"] = log[resource_col].values
    has_start = start_col in log.columns
    if has_start:
        s = pd.to_datetime(log[start_col])
        if s.dt.tz is not None:
            s = s.dt.tz_convert(None)
        ev["_dur_h"] = (ev["_t"] - s.values).dt.total_seconds() / 3600.0
    else:
        ev["_dur_h"] = 0.0

    ev_hist = ev[ev["_t"] <= cut_ts] if cut_ts is not None else ev
    ev_hist = ev_hist.assign(_bucket=_floor(ev_hist["_t"], window))

    per_case = ev.groupby("_case").agg(first=("_t", "min"), true_last=("_t", "max"))
    if cut_ts is not None:
        per_case = per_case[per_case["first"] <= cut_ts]  # not yet arrived -> unknown to history

    finished = per_case["true_last"] <= cut_ts if cut_ts is not None else pd.Series(True, index=per_case.index)

    per_case["first_bucket"] = _floor(per_case["first"], window)
    per_case["last_bucket"] = _floor(per_case["true_last"], window)
    per_case["dur_sum_h"] = ev_hist.groupby("_case")["_dur_h"].sum().reindex(per_case.index, fill_value=0.0)

    end_bucket = _floor_ts(cut_ts, window) if cut_ts is not None else max(
        per_case["last_bucket"].max(), ev_hist["_bucket"].max()
    )
    idx = pd.date_range(
        min(per_case["first_bucket"].min(), ev_hist["_bucket"].min()),
        end_bucket,
        freq=_RANGE_FREQ[window],
    )

    arrival_rate = per_case["first_bucket"].value_counts().reindex(idx, fill_value=0).astype(float)
    finish_rate = (
        per_case.loc[finished, "last_bucket"].value_counts().reindex(idx, fill_value=0).astype(float)
    )

    by_bucket = ev_hist.groupby("_bucket")
    if has_resource:
        num_unique_resources = by_bucket["_res"].nunique().reindex(idx, fill_value=0).astype(float)
    else:

        num_unique_resources = pd.Series(0.0, index=idx)
    num_unique_activities = by_bucket["_act"].nunique().reindex(idx, fill_value=0).astype(float)
    process_active_time = by_bucket["_dur_h"].sum().reindex(idx, fill_value=0.0)

    service_time = (
        per_case.loc[finished].groupby("last_bucket")["dur_sum_h"].mean().reindex(idx).ffill().fillna(0.0)
    )

    concurrent_cases = create_concurrent_cases_timeseries(
        log, time_col=time_col, case_col=case_col, window=window, plot=False, cut_date=cut_date,
    ).reindex(idx).ffill().fillna(0.0)
    throughput_time = create_avg_throughtput_time_timeseries(
        log, time_col=time_col, case_col=case_col, window=window, plot=False, cut_date=cut_date,
    ).reindex(idx).ffill().fillna(0.0)

    if has_start:
        waiting_time = (throughput_time - service_time).clip(lower=0.0)
    else:

        service_time = pd.Series(0.0, index=idx)
        waiting_time = pd.Series(0.0, index=idx)

    sd_log = pd.DataFrame({
        "concurrent_cases":      concurrent_cases,
        "arrival_rate":          arrival_rate,
        "finish_rate":           finish_rate,
        "throughput_time":       throughput_time,
        "service_time":          service_time,
        "waiting_time":          waiting_time,
        "num_unique_resources":  num_unique_resources,
        "num_unique_activities": num_unique_activities,
        "process_active_time":  process_active_time,
    }, index=idx)
    sd_log.index.name = "date"
    return sd_log
