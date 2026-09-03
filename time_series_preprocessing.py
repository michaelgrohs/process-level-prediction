import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def trim_tail_pct(series: pd.Series, pct: float = 0.10) -> pd.Series:
    """Drop the last `pct` fraction of observations (default 10%)."""
    keep = max(1, int(len(series) * (1 - pct)))
    return series.iloc[:keep]


def trim_tail_magnitude(
    series: pd.Series,
    k: float = 1.5,
    window: int = 7,
) -> pd.Series:
    """Drop a fading-out tail using adaptive moving statistics.

    Threshold at t: expanding_mean[t] - k * expanding_std[t].
    Cuts after the last index where the rolling mean exceeds this threshold.
    """
    rolling_mean   = series.rolling(window=window, min_periods=1).mean()
    expanding_mean = series.expanding(min_periods=1).mean()
    expanding_std  = series.expanding(min_periods=1).std().fillna(0)
    threshold      = expanding_mean - k * expanding_std
    above = rolling_mean[rolling_mean >= threshold]
    if above.empty:
        return series
    cut = series.index.get_loc(above.index[-1])
    return series.iloc[: cut + 1]


def trim_tail_peak(
    series: pd.Series,
    frac: float = 0.60,
    window: int = 7,
) -> pd.Series:
    """Drop a fading-out tail by comparing to the global peak.

    Threshold: frac * series.max() — a fixed, stable anchor that the tail
    can't drag down.  Cuts after the last index where the rolling mean still
    exceeds this fraction of the peak.  Tune `frac` between 0 and 1:
    higher = more aggressive trimming.
    """
    peak      = series.max()
    threshold = frac * peak
    rolling   = series.rolling(window=window, min_periods=1).mean()
    above = rolling[rolling >= threshold]
    if above.empty:
        return series
    cut = series.index.get_loc(above.index[-1])
    return series.iloc[: cut + 1]


def apply_trim(series: pd.Series, method: str, **kw) -> pd.Series:
    if method == "pct":
        return trim_tail_pct(series, pct=kw.get("pct", 0.10))
    if method == "magnitude":
        return trim_tail_magnitude(series, k=kw.get("k", 1.5), window=kw.get("window", 7))
    if method == "peak":
        return trim_tail_peak(series, frac=kw.get("frac", 0.60), window=kw.get("window", 7))
    return series   # method=None → no trimming

class Split3WayConfig:
    """
    Three-way split configuration for train/validation/test set.

    """
    train_frac: float = 0.7
    val_frac: float = 0.1
    test_frac: float = 0.2

    def __init__(self, train_frac, val_frac, test_frac):
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        s = self.train_frac + self.val_frac + self.test_frac
        if s != 1.0:
            raise ValueError(f"train_frac+val_frac+test_frac must sum to 1.0; got {s}")
        if min(self.train_frac, self.val_frac, self.test_frac) <= 0:
            raise ValueError("All split fractions must be positive.")

def split_by_dates(
    series: pd.Series,
    train_split: pd.Timestamp,
    val_split: pd.Timestamp,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Slice a series into train/val/test using pre-computed split timestamps.

    Used to enforce the same cut dates across all series (CC and TT) so that
    trimming one series cannot shift the boundary for another.
    """
    idx = series.index
    lo = train_split.tz_convert(None) if idx.tz is None else train_split
    hi = val_split.tz_convert(None)   if idx.tz is None else val_split
    return series[idx <= lo], series[(idx > lo) & (idx <= hi)], series[idx > hi]


def split_timeseries(ts: pd.Series, cfg: Split3WayConfig):
    n = len(ts)
    train_end = int(n * cfg.train_frac)
    val_end   = train_end + int(n * cfg.val_frac)
    train, val, test = ts.iloc[:train_end], ts.iloc[train_end:val_end], ts.iloc[val_end:]

    def _to_utc(t):
        t = pd.Timestamp(t)
        return t.tz_convert("UTC") if t.tzinfo is not None else t.tz_localize("UTC")

    return train, val, test, _to_utc(train.index[-1]), _to_utc(val.index[-1])


def ts_splits_from_log(
    log: pd.DataFrame,
    trim_method: str   = "pct",
    trim_pct:    float = 0.25,
    trim_k:      float = 1.5,
    trim_frac:   float = 0.60,
    trim_window: int   = 7,
    train_frac:  float = 0.70,
    val_frac:    float = 0.10,
    cut_date=None,
) -> dict:
    """
    Run the full TS pipeline on a pm4py event log and return split timestamps
    alongside the train/val/test series for both metrics.

    Mirrors the manual notebook workflow:
        avg_cc  = create_concurrent_cases_timeseries(log)
        trimmed = trim_tail_pct(avg_cc, pct=0.15)
        train, val, test, train_split, val_split = split_timeseries(trimmed, cfg)

    Returns a dict with keys "concurrent_cases" and "throughput_time".
    Each value is a dict with:
        raw         : pd.Series  — untrimmed series
        trimmed     : pd.Series  — after trim
        train       : pd.Series
        val         : pd.Series
        test        : pd.Series
        train_split : pd.Timestamp  — last date of train period
        val_split   : pd.Timestamp  — last date of val period
    """
    from time_series_creation import (  # local import avoids circular dependency
        create_concurrent_cases_timeseries,
        create_avg_throughtput_time_timeseries,
    )

    cfg = Split3WayConfig(train_frac, val_frac, round(1 - train_frac - val_frac, 10))
    trim_kw = dict(pct=trim_pct, k=trim_k, frac=trim_frac, window=trim_window)

    # Canonical timestamps from CC (the more complete series — no missing days).
    cc_raw     = create_concurrent_cases_timeseries(log, cut_date=cut_date, plot=False)
    cc_trimmed    = apply_trim(cc_raw, trim_method, **trim_kw)
    canonical_end = cc_trimmed.index[-1]
    _, _, _, train_split, val_split = split_timeseries(cc_trimmed, cfg)

    def _slice(raw: pd.Series) -> dict:
        trimmed = raw[raw.index <= canonical_end]
        idx = trimmed.index
        # Match timezone of split timestamps to the series index.
        if idx.tz is None:
            lo = train_split.tz_convert(None)
            hi = val_split.tz_convert(None)
        else:
            lo, hi = train_split, val_split
        return {
            "raw":         raw,
            "trimmed":     trimmed,
            "train":       trimmed[idx <= lo],
            "val":         trimmed[(idx > lo) & (idx <= hi)],
            "test":        trimmed[idx > hi],
            "train_split": train_split,
            "val_split":   val_split,
        }

    return {
        "concurrent_cases": _slice(cc_raw),
        "throughput_time":  _slice(create_avg_throughtput_time_timeseries(log, cut_date=cut_date, plot=False)),
    }


def compute_split_timestamps(
    df: pd.DataFrame,
    time_col: str,
    trim_method: str  = "pct",
    trim_pct:    float = 0.25,
    trim_k:      float = 1.5,
    trim_frac:   float = 0.60,
    trim_window: int   = 7,
    train_frac:  float = 0.70,
    val_frac:    float = 0.10,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Derive train/val split boundaries from an event log DataFrame.

    Builds a daily event-count series, trims the tail, then returns the
    timestamps at the train/val boundaries.

    Returns (train_split, val_split) as UTC-aware Timestamps.
    """
    daily = (
        df.set_index(time_col)
        .resample("D")
        .size()
        .rename("count")
    )
    daily      = daily[daily > 0]
    daily_trim = apply_trim(
        daily, trim_method,
        pct=trim_pct, k=trim_k, frac=trim_frac, window=trim_window,
    )
    cfg = Split3WayConfig(train_frac, val_frac, round(1 - train_frac - val_frac, 10))
    _, _, _, train_split, val_split = split_timeseries(daily_trim, cfg)
    return train_split, val_split