# -*- coding: utf-8 -*-
"""
Inter-case ("load state") feature computation for the Bukhsh remaining-time
model, based on:

    Gunnarsson, vanden Broucke, De Weerdt (2024). "LS-ICE: A Load State
    Intercase Encoding framework for improved predictive monitoring of
    business processes." Information Systems 123, 102432.
    https://doi.org/10.1016/j.is.2024.102432

We implement the paper's *case-based* encoding using the *number of active
cases* derivation of load state (their simpler, consistently best-performing
variant across datasets -- no per-activity "optimal time window" search
needed, unlike their alternative windowed-count derivation):

  current_load        -- # OTHER cases whose most recently completed
                          activity, as of this event's own timestamp,
                          equals THIS event's own activity ("current load
                          point": how busy is the place this case just
                          arrived at?).
  prev_load            -- # OTHER cases at the load point this case ITSELF
                          last visited (its own previous activity). 0 for
                          a case's first event (no previous activity yet).
  next_likely_load_i   -- # OTHER cases at each of the top-N activities
                          historically most likely to follow the current
                          activity (i=1..N, ranked by transition frequency
                          in train+val data only -- a static lookup table,
                          not something recomputed from future events).

All three quantities are, by construction, computed only from timestamps
<= the event's own timestamp (every OTHER case's *already observed*
history) -- there is no future information involved anywhere. That is what
makes it safe to use the exact same function to build every training-time
feature row and the single anchor-point feature vector needed at inference
time: nothing about "leakage" changes between the two call sites.
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


def build_load_state_table(
    events: pd.DataFrame,
    transition_table: dict,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """
    Single chronological sweep that produces one feature row per (caseid, k),
    where k is the 1-indexed position of that event within its own case --
    the same prefix-length indexing processtransformer's own helper
    functions use.

    Parameters
    ----------
    events : DataFrame[caseid, activity, timestamp]
        MUST be every case's genuine, COMPLETE event history (i.e. built
        from the full, untruncated historical log -- see
        InterCaseBukhshTrainer._build_load_state_table, which passes
        `full_df`, not the observation-truncated train/val/test concat).
        This is safe and does not leak anything into any individual
        prediction: the sweep only ever reads a row's own features off the
        counters *before* that row's own event is registered, so nothing
        with a later timestamp than the event being scored is ever visible
        to it, regardless of whose case it belongs to. Truncating `events`
        to each case's *observed* prefix (rather than its true, complete
        lifecycle) does not add any safety here -- it only breaks the
        "case has terminated" detection below, since a case's last row in a
        truncated frame means "our observation window ends here", not
        "this case is actually done" (see git history for the two
        incorrect attempts at handling this before settling on "just use
        the full log").
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

    activity_case_count: Counter = Counter()  # activity -> # cases currently "at" it
    case_state: dict = {}  # caseid -> current (most recent) activity
    case_k: Counter = Counter()  # caseid -> running prefix length seen so far

    rows = []
    for caseid, activity in zip(events["caseid"], events["activity"]):
        prev_activity = case_state.get(caseid)


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

        case_k[caseid] += 1
        rows.append(
            {
                "caseid": caseid,
                "k": case_k[caseid],
                "current_load": max(0, cur_count),
                "prev_load": max(0, prev_count),
                **{f"next_likely_load_{i + 1}": v for i, v in enumerate(next_counts)},
            }
        )


        if prev_activity is not None:
            activity_case_count[prev_activity] -= 1
        activity_case_count[activity] += 1
        case_state[caseid] = activity

        if case_k[caseid] == case_total_len[caseid]:
            activity_case_count[activity] -= 1
            del case_state[caseid]

    return pd.DataFrame(rows)
