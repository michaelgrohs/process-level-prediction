"""
Steady-state-detection-based log truncation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ATTR_CASE = "case:concept:name"
ATTR_TIME = "time:timestamp"

KERNEL_SIGMA = 4
KERNEL_CONSENSUS = 0.7
ROLLING_LONG = 25
ROLLING_SHORT = 5
ROLLING_STD_THRESHOLD = 0.5
ACTIVITY_FLOOR_FRAC = 0.1

FEATURE_COLS = ["case_arrivals", "case_completions", "active_cases", "avg_lead_time"]
# case_arrivals counted 2x in the aggregation vote, per project request to
# prioritize it — it's the clearest independent signal that a quiet window
# is genuinely ended rather than merely calm.
AGGREGATION_WEIGHTS = {"case_arrivals": 2.0, "case_completions": 1.0,
                       "active_cases": 1.0, "avg_lead_time": 1.0}


# ── Process features (ported from approach/process_features.py) ────────────

def _get_timeframe(df: pd.DataFrame, window_step: str) -> pd.DatetimeIndex:
    start, end = df[ATTR_TIME].min(), df[ATTR_TIME].max()
    timeframe = pd.date_range(start=start, end=end, freq=window_step, normalize=True)
    timeframe = timeframe.union(pd.date_range(end=timeframe[0], periods=2, freq=window_step))
    timeframe = timeframe.union(pd.date_range(start=timeframe[-1], periods=2, freq=window_step))
    return timeframe


def _get_period(timestamp, timeframe) -> float:
    for i, (start, end) in enumerate(zip(timeframe[:-1], timeframe[1:])):
        if start <= timestamp < end:
            return i + 1
    return np.nan


def derive_process_features(df: pd.DataFrame, window_step: str = "D") -> pd.DataFrame:
    """
    Ports approach/process_features.py::derive_process_feature, computing
    case_arrivals, case_completions, active_cases, avg_lead_time. The repo's
    own default scope (ssd_methods_parameters.yml) has arrivals off; it's
    added here — see module docstring for why.
    """
    t = pd.to_datetime(df[ATTR_TIME])
    if t.dt.tz is not None:
        t = t.dt.tz_convert(None)
    df = df.assign(**{ATTR_TIME: t})

    timeframe = _get_timeframe(df, window_step)
    df = df.assign(window=df[ATTR_TIME].apply(lambda x: _get_period(x, timeframe)))

    windows = pd.DataFrame({"window": range(1, len(timeframe))})

    def _count_by(idx_fn, col_name):
        ev = df.loc[idx_fn(df.groupby(ATTR_CASE)[ATTR_TIME])][[ATTR_CASE, "window"]]
        counted = ev.groupby("window")[ATTR_CASE].count().reset_index(name=col_name)
        merged = windows.merge(counted, on="window", how="left")
        merged[col_name] = merged[col_name].fillna(0)
        return merged

    arrivals = _count_by(lambda g: g.idxmin(), "case_arrivals")
    completions = _count_by(lambda g: g.idxmax(), "case_completions")

    # active_cases: distinct cases with an event in each window
    active = df.groupby("window")[ATTR_CASE].nunique().reset_index(name="active_cases")
    active = windows.merge(active, on="window", how="left")
    active["active_cases"] = active["active_cases"].fillna(0)

    # avg_lead_time: mean case duration (as a fraction of one window's length),
    # attributed to each case's last window
    dur = df.groupby(ATTR_CASE)[ATTR_TIME].agg(lambda x: x.max() - x.min())
    end_window = df.groupby(ATTR_CASE)["window"].max()
    merged = pd.DataFrame({"case_duration": dur, "case_end_window": end_window})
    avg_lead = merged.groupby("case_end_window")["case_duration"].mean()
    avg_lead = windows.merge(avg_lead.rename("avg_lead_time"), left_on="window",
                             right_index=True, how="left")
    avg_lead["avg_lead_time"] = avg_lead["avg_lead_time"].fillna(pd.Timedelta(0))
    window_seconds = (timeframe[1] - timeframe[0]).total_seconds()
    avg_lead["avg_lead_time"] = avg_lead["avg_lead_time"].apply(lambda x: x.total_seconds()) / window_seconds

    features = arrivals.merge(completions, on="window").merge(active, on="window").merge(avg_lead, on="window")
    features = features.set_index("window")
    features.index = pd.to_datetime(timeframe[1:])
    return features


# ── SSD: rolling-window drift detector (ported from offline_ssd_methods.py) ─

def apply_ssd_rolling_window(
    data: pd.Series,
    long_interval: int = ROLLING_LONG,
    short_interval: int = ROLLING_SHORT,
    std_threshold: float = ROLLING_STD_THRESHOLD,
    detect: str = "both",
    activity_floor_frac: float = ACTIVITY_FLOOR_FRAC,
) -> pd.Series:
    """Steady state = 1, drift/transient/dead = 0.


    """
    long_average = data.rolling(long_interval).mean()
    long_std = data.rolling(long_interval).std()
    short_average = data.rolling(short_interval).mean()

    increase = short_average > long_average + std_threshold * long_std
    decrease = short_average < long_average - std_threshold * long_std
    if detect == "increase":
        drift = increase
    elif detect == "decrease":
        drift = decrease
    else:
        drift = increase | decrease

    no_drift = ~drift.astype(bool)

    if activity_floor_frac > 0:
        nonzero = data[data > 0]
        reference_level = nonzero.median() if not nonzero.empty else 0.0
        is_active = long_average >= activity_floor_frac * reference_level
        steady = no_drift & is_active.fillna(False)
    else:
        steady = no_drift

    return steady.astype(int)


# ── Aggregation across features (ported from utilities.py kernel_N method) ──

def aggregate_ssd_curves(curves: pd.DataFrame, sigma: int = KERNEL_SIGMA,
                         consensus: float = KERNEL_CONSENSUS,
                         weights: dict | None = None) -> pd.Series:
    """curves: one 0/1 column per process feature. Returns the aggregated 0/1
    steady-state curve: (weighted) mean across features -> Gaussian smoothing
    -> threshold. weights defaults to AGGREGATION_WEIGHTS (uniform 1.0 for any
    column not listed there)."""
    weights = weights if weights is not None else AGGREGATION_WEIGHTS
    w = pd.Series({col: weights.get(col, 1.0) for col in curves.columns})
    mean_series = curves.mul(w, axis=1).sum(axis=1) / w.sum()

    smoothed = pd.Series(gaussian_filter1d(mean_series.to_numpy(), sigma=sigma), index=mean_series.index)

    lo, hi = mean_series.min(), mean_series.max()
    s_lo, s_hi = smoothed.min(), smoothed.max()
    if s_hi == s_lo:
        normalized = mean_series
    else:
        normalized = (smoothed - s_lo) / (s_hi - s_lo) * (hi - lo) + lo

    return (normalized > consensus).astype(int)


# ── Segment the curve, find the cutoff, truncate ────────────────────────────

def find_segments(curve: pd.Series) -> pd.DataFrame:
    """Contiguous constant-value runs. Returns columns: label, start, end (dates)."""
    values = curve.to_numpy()
    change = np.r_[True, values[1:] != values[:-1]]
    seg_id = np.cumsum(change)
    df = pd.DataFrame({"label": values, "date": curve.index, "seg_id": seg_id})
    segments = df.groupby("seg_id").agg(label=("label", "first"),
                                         start=("date", "min"),
                                         end=("date", "max"))
    return segments.reset_index(drop=True)


def find_cutoff(curve: pd.Series) -> pd.Timestamp | None:
    """
    Cut at the start of the LAST unstable (non-steady) segment anywhere in
    the curve — not "the last steady segment" reasoned about case by case.


    Returns None if the curve has no unstable segment at all.
    """
    segments = find_segments(curve)
    unstable = segments[segments["label"] == 0]
    if unstable.empty:
        return None
    return unstable.iloc[-1]["start"]


def run_ssd_trim(log: pd.DataFrame, window_step: str = "D",
                 long_interval: int = ROLLING_LONG, short_interval: int = ROLLING_SHORT,
                 std_threshold: float = ROLLING_STD_THRESHOLD,
                 activity_floor_frac: float = ACTIVITY_FLOOR_FRAC,
                 weights: dict | None = None) -> dict:
    """
    End-to-end: compute the process features, run the rolling-window SSD
    (with the activity-floor guard) on each, aggregate, find the cutoff,
    truncate the log.

    Returns dict: features (raw per-window values), curves (per-feature 0/1),
    aggregated (the combined 0/1 curve), segments, cutoff (Timestamp or None),
    truncated_log (events with time:timestamp < cutoff; the full log if
    cutoff is None).
    """
    features = derive_process_features(log, window_step=window_step)

    curves = pd.DataFrame({
        col: apply_ssd_rolling_window(features[col], long_interval=long_interval,
                                      short_interval=short_interval, std_threshold=std_threshold,
                                      activity_floor_frac=activity_floor_frac)
        for col in FEATURE_COLS
    }, index=features.index)

    aggregated = aggregate_ssd_curves(curves, weights=weights)
    segments = find_segments(aggregated)
    cutoff = find_cutoff(aggregated)

    t = pd.to_datetime(log[ATTR_TIME])
    if t.dt.tz is not None:
        t = t.dt.tz_convert(None)
    if cutoff is not None:
        truncated_log = log[t < cutoff].copy()
    else:
        truncated_log = log.copy()

    return dict(
        features=features,
        curves=curves,
        aggregated=aggregated,
        segments=segments,
        cutoff=cutoff,
        truncated_log=truncated_log,
    )
