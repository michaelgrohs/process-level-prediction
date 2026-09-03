"""
Utilities for splitting event logs into temporal train/test windows.

Two split modes
---------------
full_traces  : only cases fully completed within a period are included;
               cases straddling a boundary are excluded from both sides.
prefix_split : a case spanning a boundary contributes its prefix (all
               events up to the split) to the earlier set and its suffix
               (events after the split) to the later set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pm4py


def load_event_log(
    path: str | Path,
    time_col: str,
    case_col: str,
    columns_map: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Load an event log from .xes or .csv.

    - path        : path to event log file
    - time_col    : timestamp column name (before renaming)
    - case_col    : case ID column name (before renaming)
    - columns_map : optional rename map applied after loading
    """
    path   = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".xes":
        log = pm4py.read_xes(str(path))
        df  = pm4py.convert_to_dataframe(log)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    df[time_col] = pd.to_datetime(df[time_col], errors="raise", utc=True)
    df = df.dropna(subset=[case_col])
    df = df[df[case_col] != ""]

    if columns_map:
        df = df.rename(columns=columns_map)

    return df


def make_window(
    df: pd.DataFrame,
    case_col: str,
    time_col: str,
    split_ts: pd.Timestamp,
    last_ts:  Optional[pd.Timestamp] = None,
    full_traces: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split an event log at split_ts into a training set and a test set.

    Parameters
    ----------
    df          : event log DataFrame (time_col must be UTC-aware datetime)
    case_col    : case ID column name
    time_col    : timestamp column name
    split_ts    : boundary between training and test periods
    last_ts     : upper bound for test period; None means no upper bound
    full_traces : see modes below

    Modes
    -----
    full_traces=True
        train : cases whose last event is <= split_ts
        test  : cases whose first event is > split_ts (and last <= last_ts if given)

    full_traces=False  (prefix split)
        train : all events with timestamp <= split_ts
                (includes prefixes of running cases)
        test  : for cases that have at least one event after split_ts,
                only the events after split_ts (up to last_ts if given)
                — i.e. the suffix the model must predict

    Returns
    -------
    (train_df, test_df)
    """
    ts = pd.to_datetime(df[time_col], utc=True)

    if full_traces:
        case_max = df.groupby(case_col)[time_col].max()
        case_min = df.groupby(case_col)[time_col].min()

        train_cases = case_max[case_max <= split_ts].index
        train = df[df[case_col].isin(train_cases)].copy()

        if last_ts is not None:
            test_cases = case_max[
                (case_min > split_ts) & (case_max <= last_ts)
            ].index
        else:
            test_cases = case_max[case_min > split_ts].index
        test = df[df[case_col].isin(test_cases)].copy()

    else:
        train = df[ts <= split_ts].copy()

        running_cases = df[ts > split_ts][case_col].unique()
        test = df[df[case_col].isin(running_cases) & (ts > split_ts)].copy()
        if last_ts is not None:
            test = test[pd.to_datetime(test[time_col], utc=True) <= last_ts].copy()

    return train, test


def make_three_way_split(
    df: pd.DataFrame,
    case_col: str,
    time_col: str,
    train_split: pd.Timestamp,
    val_split: pd.Timestamp,
    full_traces: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Three-way temporal split into (train_df, val_df, test_df).

    test_df is identical regardless of full_traces — it encodes the correct
    prediction context at val_split without leaking future information:
        in-flight cases (case_min ≤ val_split < case_max) → full prefix,
            i.e. all events with timestamp ≤ val_split
        fresh cases (case_min > val_split) → first event only

    full_traces=True
        train_df : complete cases with last event ≤ train_split
        val_df   : complete cases whose first AND last event fall in
                   (train_split, val_split]

    full_traces=False  (prefix split)
        train_df : all events with timestamp ≤ train_split
        val_df   : full prefix up to val_split for every case that has at
                   least one event after train_split (includes pre-train_split
                   events of ongoing cases, giving the model full context)

    Parameters
    ----------
    df          : event log DataFrame (time_col must be UTC-aware datetime)
    case_col    : case ID column
    time_col    : timestamp column
    train_split : boundary between train and val periods
    val_split   : boundary between val and test periods
    full_traces : see above
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    train_ts = pd.Timestamp(train_split)
    val_ts   = pd.Timestamp(val_split)
    if train_ts.tzinfo is None:
        train_ts = train_ts.tz_localize("UTC")
    if val_ts.tzinfo is None:
        val_ts = val_ts.tz_localize("UTC")

    ts       = df[time_col]
    case_min = df.groupby(case_col)[time_col].min()
    case_max = df.groupby(case_col)[time_col].max()

    # ── train_df and val_df depend on full_traces ──────────────────────────────
    if full_traces:
        train_cases = case_max[case_max <= train_ts].index
        train_df    = df[df[case_col].isin(train_cases)].copy()

        val_cases = case_max[
            (case_min > train_ts) & (case_max <= val_ts)
        ].index
        val_df = df[df[case_col].isin(val_cases)].copy()

    else:
        train_df = df[ts <= train_ts].copy()

        active_after_train = set(df[ts > train_ts][case_col])
        val_df = df[
            df[case_col].isin(active_after_train) & (ts <= val_ts)
        ].copy()

    # ── test_df is mode-agnostic ───────────────────────────────────────────────
    in_flight_ids = set(
        case_min[(case_min <= val_ts) & (case_max > val_ts)].index
    )
    fresh_ids = set(case_min[case_min > val_ts].index)

    in_flight_df = df[
        df[case_col].isin(in_flight_ids) & (ts <= val_ts)
    ].copy()

    fresh_df = (
        df[df[case_col].isin(fresh_ids)]
        .sort_values(time_col)
        .groupby(case_col, as_index=False)
        .first()
    )

    test_df = pd.concat([in_flight_df, fresh_df], ignore_index=True)

    return train_df, val_df, test_df
