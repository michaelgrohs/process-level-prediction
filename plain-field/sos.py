"""
Start-Of-Sequence helpers for the plain-field prediction scenario.

Plain-field: at test-horizon start we observe
  – fresh arrivals : only arrival timestamp, no case history
  – in-flight cases: partial event history up to val_split

Helpers in this module prepare the *static* per-log quantities (most-frequent
activity / resource, empirical hour distribution) used by runner.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Activity / resource selectors ─────────────────────────────────────────────

def most_frequent_first_activity(
    train_df: pd.DataFrame,
    case_col: str = "caseid",
    task_col: str = "task",
    time_col: str = "end_timestamp",
) -> str:
    """Most common first activity across training cases."""
    first_acts = (
        train_df.sort_values(time_col)
                .groupby(case_col)[task_col]
                .first()
    )
    return str(first_acts.value_counts().index[0])


def most_frequent_first_resource(
    train_df: pd.DataFrame,
    case_col: str = "caseid",
    task_col: str = "task",
    user_col: str = "user",
    time_col: str = "end_timestamp",
) -> str:
    """
    Most common resource (user) paired with the most frequent first activity.

    Finds the most common first activity, then — among all training cases whose
    first event has that activity — returns the most frequent resource.
    """
    first_events = (
        train_df.sort_values(time_col)
                .groupby(case_col)
                .first()
                .reset_index()
    )
    most_freq_act = first_events[task_col].value_counts().index[0]
    act_rows = first_events[first_events[task_col] == most_freq_act]
    return str(act_rows[user_col].value_counts().index[0])


# ── Empirical arrival-hour sampler ─────────────────────────────────────────────

def empirical_arrival_hour_sampler(
    train_df: pd.DataFrame,
    case_col: str = "caseid",
    time_col: str = "end_timestamp",
    seed: int | None = 0,
):
    """
    Return a callable that samples a random intra-day offset (pd.Timedelta)
    from the empirical distribution of first-event hours in training data.


    Usage:
        sampler = empirical_arrival_hour_sampler(train_df)
        offset  = sampler()   # e.g. Timedelta('09:23:47')
        ts      = day_ts + offset
    """
    ts = pd.to_datetime(train_df[time_col])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(None)

    first_ts = (
        train_df.assign(_ts=ts)
                .sort_values("_ts")
                .groupby(case_col)["_ts"]
                .first()
    )
    # Seconds from midnight for each first event
    secs = (
        first_ts.dt.hour   * 3600 +
        first_ts.dt.minute * 60   +
        first_ts.dt.second
    ).values.astype(int)

    rng = np.random.default_rng(seed)

    def _sample() -> pd.Timedelta:
        s = int(rng.choice(secs))
        # Add ±15 min jitter so identical-hour cases don't all land at :00
        jitter = int(rng.uniform(-900, 900))
        return pd.Timedelta(seconds=max(0, min(86399, s + jitter)))

    return _sample


# ── Case-duration fallback ─────────────────────────────────────────────────────

def mean_case_duration(
    train_df: pd.DataFrame,
    time_col: str = "end_timestamp",
) -> float:
    """
    Mean case duration (days) from train_df.


    """
    durations = []
    for _, grp in train_df.groupby("caseid"):
        ts = pd.to_datetime(grp[time_col]).sort_values()
        if len(ts) >= 2:
            dur = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400
            durations.append(dur)
    return float(np.mean(durations)) if durations else 0.0


def amiri_sos_rem_time(
    train_df: pd.DataFrame,
    time_col: str = "end_timestamp",
) -> float:
    """
    Scalar SOS rem_time for Amiri (days) = mean case duration.

    Kept for backward compatibility; the per-arrival pipeline now uses
    predict_amiri_plain_field() in runner.py which handles both SOS and
    in-flight cases via the full GPS inference pipeline.
    """
    return mean_case_duration(train_df, time_col)
