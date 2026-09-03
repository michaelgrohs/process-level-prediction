from typing import Literal

import pandas as pd
import matplotlib.pyplot as plt

_VALID_WINDOWS = ("hours", "days", "weeks", "months")

_RANGE_FREQ = {
    "hours":  "h",
    "days":   "D",
    "weeks":  "W-MON",
    "months": "MS",
}

_TITLES_CC = {
    "hours":  "Concurrent Cases per Hour",
    "days":   "Concurrent Cases per Day",
    "weeks":  "Concurrent Cases per Week",
    "months": "Concurrent Cases per Month",
}

_TITLES_TT = {
    "hours":  "Avg Throughput Time per Hour",
    "days":   "Avg Throughput Time per Day",
    "weeks":  "Avg Throughput Time per Week",
    "months": "Avg Throughput Time per Month",
}


def _floor_to_window(ts: pd.Series, window: str) -> pd.Series:
    if window == "hours":
        return ts.dt.floor("h")
    if window == "days":
        return ts.dt.normalize()
    if window == "weeks":
        return ts.dt.to_period("W").dt.start_time
    # months
    return ts.dt.to_period("M").dt.start_time


def create_concurrent_cases_timeseries(
    log: pd.DataFrame,
    time_col: str = "time:timestamp",
    case_col: str = "case:concept:name",
    window: str = "days",
    plot: bool = True,
    cut_date=None,
) -> pd.Series:
    """
    Create a time series of concurrently active cases per time window.

    Parameters
    ----------
    log      : event log as a DataFrame
    time_col : timestamp column name
    case_col : case ID column name
    window   : granularity – one of 'hours', 'days', 'weeks', 'months'
    plot     : whether to plot the result
    cut_date : optional upper bound for the time axis (str or Timestamp).
               Cases still in-flight at cut_date (last event > cut_date) have
               their last bucket capped at cut_date so they are correctly counted
               as active at cut_date.  The series does not extend past cut_date,
               eliminating the sparse fade-out tail.

    Returns
    -------
    pd.Series with the bucket timestamps as index and active-case counts as values
    """
    if window not in _VALID_WINDOWS:
        raise ValueError(f"window must be one of {_VALID_WINDOWS}, got {window!r}")

    t = pd.to_datetime(log[time_col])
    if t.dt.tz is not None:
        t = t.dt.tz_convert(None)

    first_last = (
        log.assign(_t=t)
        .groupby(case_col)["_t"]
        .agg(first="min", last="max")
    )

    if cut_date is not None:
        cut_ts = pd.Timestamp(cut_date)
        if cut_ts.tzinfo is not None:
            cut_ts = cut_ts.tz_convert(None)
        # In-flight cases: cap their last timestamp at cut_date so they
        # contribute to CC at cut_date instead of disappearing early.
        first_last["last"] = first_last["last"].clip(upper=cut_ts)

    first_bucket = _floor_to_window(first_last["first"], window)
    last_bucket  = _floor_to_window(first_last["last"],  window)

    end_bucket = last_bucket.max()
    if cut_date is not None:
        end_bucket = min(end_bucket, _floor_to_window(pd.Series([cut_ts]), window).iloc[0])

    buckets = pd.date_range(
        start=first_bucket.min(),
        end=end_bucket,
        freq=_RANGE_FREQ[window],
    )

    counts = pd.Series(
        [(first_bucket <= b).sum() - (last_bucket < b).sum() for b in buckets],
        index=buckets,
        name="concurrent_cases",
    )

    if plot:
        counts.plot(figsize=(12, 4), title=_TITLES_CC[window])
        plt.xlabel("Date")
        plt.ylabel("Active cases")
        plt.tight_layout()
        plt.show()

    return counts


def create_avg_throughtput_time_timeseries(
    log: pd.DataFrame,
    time_col: str = "time:timestamp",
    case_col: str = "case:concept:name",
    window: str = "days",
    fill: Literal["ffill", "interpolate", None] = "ffill",
    plot: bool = True,
    cut_date=None,
) -> pd.Series:
    """
    Create a time series of average throughput time per time window.

    For each bucket, the average throughput time (in hours) is computed
    over all cases whose last event falls within that bucket.
    Buckets with no completing cases are filled according to `fill`.

    Parameters
    ----------
    log      : event log as a DataFrame
    time_col : timestamp column name
    case_col : case ID column name
    window   : granularity – one of 'hours', 'days', 'weeks', 'months'
    fill     : how to handle empty buckets – 'ffill', 'interpolate', or None (keep NaN)
    plot     : whether to plot the result
    cut_date : optional upper bound for the time axis (str or Timestamp).
               Cases completing after cut_date are excluded from TT; the series
               does not extend past cut_date.

    Returns
    -------
    pd.Series with bucket timestamps as index and average throughput time (hours) as values
    """
    if window not in _VALID_WINDOWS:
        raise ValueError(f"window must be one of {_VALID_WINDOWS}, got {window!r}")

    t = pd.to_datetime(log[time_col])
    if t.dt.tz is not None:
        t = t.dt.tz_convert(None)

    first_last = (
        log.assign(_t=t)
        .groupby(case_col)["_t"]
        .agg(first="min", last="max")
    )

    if cut_date is not None:
        cut_ts = pd.Timestamp(cut_date)
        if cut_ts.tzinfo is not None:
            cut_ts = cut_ts.tz_convert(None)
        first_last = first_last[first_last["last"] <= cut_ts]

    first_last["duration_h"] = (
        (first_last["last"] - first_last["first"]).dt.total_seconds() / 3600
    )
    first_last["last_bucket"] = _floor_to_window(first_last["last"], window)

    avg_tt = (
        first_last
        .groupby("last_bucket")["duration_h"]
        .mean()
        .rename("avg_throughput_time_h")
    )

    end_idx = avg_tt.index.max()
    if cut_date is not None:
        end_idx = min(end_idx, _floor_to_window(pd.Series([cut_ts]), window).iloc[0])

    buckets = pd.date_range(
        start=avg_tt.index.min(),
        end=end_idx,
        freq=_RANGE_FREQ[window],
    )
    avg_tt = avg_tt.reindex(buckets)

    if fill == "ffill":
        avg_tt = avg_tt.ffill()
    elif fill == "interpolate":
        avg_tt = avg_tt.interpolate(method="time")

    if plot:
        avg_tt.plot(figsize=(12, 4), title=_TITLES_TT[window])
        plt.xlabel("Date")
        plt.ylabel("Avg throughput time (hours)")
        plt.tight_layout()
        plt.show()
        #print(avg_tt.describe())

    return avg_tt