# -*- coding: utf-8 -*-
"""
Inter-case ("load state") feature computation for the Camargo suffix-
prediction model -- same feature definitions as inter-case-bukhsh/features.py
(see that module's docstring and inter-case-bukhsh/README.md for the full
derivation from Gunnarsson, vanden Broucke, De Weerdt 2024, "LS-ICE").


"""
from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

DEFAULT_TOP_N = 5


def ic_feature_cols(top_n: int = DEFAULT_TOP_N) -> list[str]:
    """Column names, in the fixed order the model expects them in."""
    return ["current_load", "prev_load"] + [
        f"next_likely_load_{i + 1}" for i in range(top_n)
    ]


def build_transition_table(events: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> dict:
    """
    Static "which activities most likely follow this one" lookup, built
    ONLY from train+val cases (never test), so it never encodes anything
    about held-out behaviour.

    Parameters
    ----------
    events : DataFrame[caseid, activity, timestamp]
        Restricted to train+val cases. Order doesn't matter here -- we
        group by caseid and re-sort within each group by timestamp.

    Returns
    -------
    dict[activity] -> list of up to `top_n` most likely next activities,
    ranked by historical transition frequency (most frequent first).
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for _, grp in events.sort_values("timestamp", kind="mergesort").groupby(
        "caseid", sort=False
    ):
        acts = grp["activity"].tolist()
        for a, b in zip(acts[:-1], acts[1:]):
            counts[a][b] += 1
    return {act: [nxt for nxt, _ in ctr.most_common(top_n)] for act, ctr in counts.items()}


def load_state_features_for(
    activity: str,
    prev_activity: str | None,
    activity_case_count: Counter,
    transition_table: dict,
    top_n: int = DEFAULT_TOP_N,
) -> dict:
    cur_count = activity_case_count[activity]
    if prev_activity == activity:
        cur_count -= 1

    if prev_activity is None:
        prev_count = 0
    else:
        prev_count = activity_case_count[prev_activity] - 1

    next_counts = []
    for nxt in transition_table.get(activity, [])[:top_n]:
        c = activity_case_count[nxt]
        if prev_activity == nxt:
            c -= 1
        next_counts.append(max(0, c))
    next_counts += [0] * (top_n - len(next_counts))

    return {
        "current_load": max(0, cur_count),
        "prev_load": max(0, prev_count),
        **{f"next_likely_load_{i + 1}": v for i, v in enumerate(next_counts)},
    }


def build_load_state_table(
    events: pd.DataFrame,
    transition_table: dict,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """
    Single chronological sweep that produces one feature row per (caseid, k),
    where k is the 1-indexed position of that event within its own case.

    Parameters
    ----------
    events : DataFrame[caseid, activity, timestamp]
        MUST be every case's genuine, COMPLETE event history (the full,
        untruncated historical log -- see
        InterCaseCamargoTrainer._build_load_state_table, which passes
        `full_df`).
    transition_table : dict
        From build_transition_table(), computed on train+val only.
    top_n : int
        Must match the top_n used to build transition_table.

    Returns
    -------
    DataFrame[caseid, k, current_load, prev_load,
              next_likely_load_1 .. next_likely_load_{top_n}]
    """
    events = events.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    case_total_len: dict = events.groupby("caseid").size().to_dict()

    activity_case_count: Counter = Counter()
    case_state: dict = {}
    case_k: Counter = Counter()

    rows = []
    for caseid, activity in zip(events["caseid"], events["activity"]):
        prev_activity = case_state.get(caseid)

        feats = load_state_features_for(
            activity, prev_activity, activity_case_count, transition_table, top_n
        )

        case_k[caseid] += 1
        rows.append({"caseid": caseid, "k": case_k[caseid], **feats})

        if prev_activity is not None:
            activity_case_count[prev_activity] -= 1
        activity_case_count[activity] += 1
        case_state[caseid] = activity

        if case_k[caseid] == case_total_len[caseid]:
            activity_case_count[activity] -= 1
            del case_state[caseid]

    return pd.DataFrame(rows)


def sweep_state_upto(events: pd.DataFrame, cutoff_ts) -> tuple[Counter, dict]:
    """
    Run the same chronological sweep as build_load_state_table, but stop as
    soon as events past `cutoff_ts` are reached, and return the LIVE state
    at that point instead of a per-row feature table:

        (activity_case_count, case_state)

    This is the seed state for the joint generation simulation:
    activity_case_count / case_state reflect every case's true, real
    position in the system as of the anchor time (`cutoff_ts`), exactly as
    an observer standing at that moment would see it. Generation then
    continues the SAME state forward turn-by-turn using
    `load_state_features_for`, so cases whose future is being generated
    are, from that point on, driven by the model's own predictions rather
    than real events -- honestly reflecting that we cannot know the true
    future, while still being causally consistent with the real past.

    `events` must be the full, untruncated log (see build_load_state_table's
    docstring) so case-completion detection (a case genuinely ending at or
    before cutoff_ts) is accurate.
    """
    events = events.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    case_total_len: dict = events.groupby("caseid").size().to_dict()

    activity_case_count: Counter = Counter()
    case_state: dict = {}
    case_k: Counter = Counter()

    for caseid, activity, ts in zip(
        events["caseid"], events["activity"], events["timestamp"]
    ):
        if ts > cutoff_ts:
            break

        prev_activity = case_state.get(caseid)
        case_k[caseid] += 1
        if prev_activity is not None:
            activity_case_count[prev_activity] -= 1
        activity_case_count[activity] += 1
        case_state[caseid] = activity

        if case_k[caseid] == case_total_len[caseid]:
            activity_case_count[activity] -= 1
            del case_state[caseid]

    return activity_case_count, case_state


def attach_ic_features(
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    load_state_table: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:

    def _naive(series):
        if pd.api.types.is_datetime64_any_dtype(series):
            ts = series
        else:
            ts = pd.to_datetime(series, utc=True, format="mixed")
        return ts.dt.tz_convert(None) if ts.dt.tz is not None else ts

    full_k = full_df.copy()
    full_k["end_timestamp"] = _naive(full_k["end_timestamp"])
    full_k = full_k.sort_values(["caseid", "end_timestamp"], kind="mergesort")
    full_k["caseid"] = full_k["caseid"].astype(str)
    full_k["k"] = full_k.groupby("caseid").cumcount() + 1
    full_k = full_k.drop_duplicates(subset=["caseid", "task", "end_timestamp"])

    d = df.copy()
    d["caseid"] = d["caseid"].astype(str)
    d["end_timestamp"] = _naive(d["end_timestamp"])
    merged = d.merge(
        full_k[["caseid", "task", "end_timestamp", "k"]],
        on=["caseid", "task", "end_timestamp"],
        how="left",
    )

    table = load_state_table.copy()
    table["caseid"] = table["caseid"].astype(str)
    merged = merged.merge(table, on=["caseid", "k"], how="left")

    ic_cols = ic_feature_cols(top_n)
    merged[ic_cols] = merged[ic_cols].fillna(0.0)
    return merged.drop(columns=["k"])
