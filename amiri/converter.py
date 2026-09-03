"""Convert DataFrames (caseid, task, user, end_timestamp) to PyG graph datasets."""
from __future__ import annotations

import bisect
import pickle
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def _to_naive(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _active_cases(sorted_starts, sorted_ends, t) -> int:
    return bisect.bisect_right(sorted_starts, t) - bisect.bisect_right(sorted_ends, t)


def _build_global_stats(train_df: pd.DataFrame, val_df: pd.DataFrame,
                        test_df: Optional[pd.DataFrame] = None) -> dict:
    """Compute statistics from train+val (and node vocab from all data)."""
    tv_df = pd.concat([train_df, val_df], ignore_index=True)
    all_df = pd.concat(
        [tv_df] + ([test_df] if test_df is not None else []), ignore_index=True
    )

    # Node class dict from all data (so OOV test activities have a valid index)
    # Cast to str first so integer-typed resources/activities don't cause sort failure
    # when mixed with string-typed SOS cases in the plain-field scenario.
    all_acts = sorted(all_df["task"].dropna().astype(str).unique())
    node_class_dict = {act: i for i, act in enumerate(all_acts)}

    # User encoder from all data (handle_unknown='ignore' for OOV)
    all_users = sorted(all_df["user"].dropna().astype(str).unique())
    user_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    try:
        user_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        user_encoder.fit(np.array(all_users).reshape(-1, 1))
    except TypeError:
        user_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        user_encoder.fit(np.array(all_users).reshape(-1, 1))

    # Case-level stats from train+val only
    case_starts, case_ends = [], []
    max_case_df = 0
    max_time_norm = 0.0

    for cid, grp in tv_df.groupby("caseid"):
        grp = grp.sort_values("end_timestamp")
        acts  = [str(a) for a in grp["task"].tolist()]
        times = [_to_naive(t) for t in grp["end_timestamp"].tolist()]
        N = len(acts)
        if N < 2:
            continue

        case_starts.append(times[0])
        case_ends.append(times[-1])
        dur = (times[-1] - times[0]).total_seconds() / 86400
        if dur > max_time_norm:
            max_time_norm = dur

        df_dict: dict = {}
        for s, t in zip(acts[:-1], acts[1:]):
            df_dict[(s, t)] = df_dict.get((s, t), 0) + 1
        if df_dict:
            max_case_df = max(max_case_df, max(df_dict.values()))

    sorted_starts = sorted(case_starts)
    sorted_ends = sorted(case_ends)

    max_active_cases = 0
    for t in [_to_naive(t) for t in tv_df["end_timestamp"].unique()]:
        ac = _active_cases(sorted_starts, sorted_ends, t)
        if ac > max_active_cases:
            max_active_cases = ac

    edge_dim = 7 + len(all_users)

    return {
        "node_class_dict": node_class_dict,
        "user_encoder": user_encoder,
        "max_case_df": max(max_case_df, 1),
        "max_active_cases": max(max_active_cases, 1),
        "max_time_norm": max(max_time_norm, 1.0),
        "sorted_starts": sorted_starts,
        "sorted_ends": sorted_ends,
        "edge_dim": edge_dim,
        "num_node_types": len(node_class_dict),
    }


def _convert_split(df: pd.DataFrame, stats: dict, test_mode: bool = False):
    """Convert a DataFrame split to a list of PyG Data objects.

    train/val (test_mode=False): all prefix lengths 2..N-1 per case.
    test (test_mode=True): one graph per case using the full prefix.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as e:
        raise ImportError(
            "torch and torch_geometric are required. "
            "Install them in your server environment."
        ) from e

    ncd = stats["node_class_dict"]
    user_enc = stats["user_encoder"]
    max_case_df = stats["max_case_df"]
    max_active_cases = stats["max_active_cases"]
    max_time_norm = stats["max_time_norm"]
    sorted_starts = stats["sorted_starts"]
    sorted_ends = stats["sorted_ends"]
    edge_dim = stats["edge_dim"]

    data_list = []
    removed = []
    test_case_order = []  # only populated in test_mode

    for cid, grp in df.groupby("caseid"):
        grp = grp.sort_values("end_timestamp")
        acts  = [str(a) for a in grp["task"].tolist()]
        users = [str(u) for u in grp["user"].tolist()]
        times = [_to_naive(t) for t in grp["end_timestamp"].tolist()]
        N = len(acts)

        if test_mode:
            if N < 2:
                removed.append(cid)
                continue
            prefix_lengths = [N]
        else:
            if N < 3:
                removed.append(cid)
                continue
            prefix_lengths = range(2, N)

        case_start = times[0]
        case_end = times[-1]

        for pl in prefix_lengths:
            p_acts = acts[:pl]
            p_times = times[:pl]
            p_users = users[:pl]

            unique_acts = list(dict.fromkeys(p_acts))  # preserves first-seen order

            # Node features: class index
            x_arr = np.array([[ncd.get(a, 0)] for a in unique_acts], dtype=np.int64)
            x = torch.from_numpy(x_arr).long()

            # Build edge index from direct-follows frequencies
            pair_freq: dict = {}
            for sa, ta in zip(p_acts[:-1], p_acts[1:]):
                si = unique_acts.index(sa)
                ti = unique_acts.index(ta)
                pair_freq[(si, ti)] = pair_freq.get((si, ti), 0) + 1

            if not pair_freq:
                continue
            edges_list = list(pair_freq.keys())
            edge_index = torch.tensor(edges_list, dtype=torch.long)

            # Edge features
            edge_feature = np.zeros((len(edges_list), edge_dim), dtype=np.float64)
            for eidx, (si, ti) in enumerate(edges_list):
                src_act = unique_acts[si]
                tgt_act = unique_acts[ti]

                occ = [
                    (i, i + 1)
                    for i in range(len(p_acts) - 1)
                    if p_acts[i] == src_act and p_acts[i + 1] == tgt_act
                ]

                num_occ = len(occ) / max_case_df
                last_dur = sum_dur = 0.0
                for ai, aj in occ:
                    dur = (p_times[aj] - p_times[ai]).total_seconds() / 86400 / max_time_norm
                    sum_dur += dur
                    last_dur = dur

                last_ai, last_aj = occ[-1]
                if last_aj == pl - 1:
                    t_last = p_times[last_aj]
                    temp1 = (t_last - case_start).total_seconds() / 86400 / max_time_norm
                    _frac = (t_last.hour * 3600 + t_last.minute * 60 + t_last.second) / 86400
                    temp2 = _frac
                    temp3 = (t_last.weekday() + _frac) / 7
                else:
                    temp1 = temp2 = temp3 = 0.0

                t_edge = p_times[occ[-1][1]]
                num_cases = _active_cases(sorted_starts, sorted_ends, t_edge) / max_active_cases

                last_user = p_users[occ[-1][1]]
                user_oh = user_enc.transform([[last_user]]).reshape(-1)

                special = np.array([num_occ, last_dur, sum_dur, temp1, temp2, temp3, num_cases])
                edge_feature[eidx, :] = np.concatenate([special, user_oh])

            edge_attr = torch.from_numpy(edge_feature).float()

            if test_mode:
                y_val = 0.0
            else:
                y_val = (case_end - p_times[-1]).total_seconds() / 86400 / max_time_norm

            y = torch.tensor([max(0.0, y_val)], dtype=torch.float)
            graph = Data(
                x=x,
                edge_index=edge_index.t().contiguous(),
                edge_attr=edge_attr,
                y=y,
                cid=cid,
                pl=pl,
            )
            data_list.append(graph)

            if test_mode:
                test_case_order.append(cid)

    if removed:
        print(f"  skipped {len(removed)} cases with fewer than 2 events")

    return data_list, test_case_order


def convert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_dir: str | Path,
    seed: int = 42,
    ref_meta_path: Optional[str | Path] = None,
) -> dict:
    """Convert DataFrames to PyG graph pickles.

    Saves to:
      {dataset_dir}/AMIRI/raw/train.pickle
      {dataset_dir}/AMIRI/raw/val.pickle
      {dataset_dir}/AMIRI/raw/test.pickle
      {dataset_dir}/AMIRI/raw/meta.pkl
      {dataset_dir}/AMIRI/raw/test_case_order.pkl

    ref_meta_path : path to an existing meta.pkl (e.g. from the training run).
        When provided the stats (user_encoder, edge_dim, normalization constants)
        are reused instead of being recomputed from data.  This ensures that
        inference datasets are encoded in the same feature space as training.

    Returns the meta dict.
    """
    raw_dir = Path(dataset_dir) / "AMIRI" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = Path(dataset_dir) / "AMIRI" / "processed"
    if proc_dir.exists():
        import shutil
        shutil.rmtree(proc_dir)

    if ref_meta_path is not None and Path(ref_meta_path).exists():
        print(f"[converter] reusing training stats from {ref_meta_path} …")
        with open(ref_meta_path, "rb") as f:
            stats = pickle.load(f)
    else:
        print("[converter] building global stats …")
        stats = _build_global_stats(train_df, val_df, test_df)

    print("[converter] converting train split …")
    train_graphs, _ = _convert_split(train_df, stats, test_mode=False)
    random.seed(seed)
    random.shuffle(train_graphs)

    print("[converter] converting val split …")
    val_graphs, _ = _convert_split(val_df, stats, test_mode=False)
    random.shuffle(val_graphs)

    print("[converter] converting test split (full prefix only) …")
    test_graphs, test_case_order = _convert_split(test_df, stats, test_mode=True)

    print(f"  train: {len(train_graphs):,} graphs | val: {len(val_graphs):,} | test: {len(test_graphs):,}")

    for name, graphs in [("train", train_graphs), ("val", val_graphs), ("test", test_graphs)]:
        path = raw_dir / f"{name}.pickle"
        with open(path, "wb") as f:
            pickle.dump(graphs, f)
        print(f"  saved {path}")

    meta = {k: v for k, v in stats.items() if k != "user_encoder"}
    meta["user_encoder"] = stats["user_encoder"]  # include encoder for denorm in predict

    with open(raw_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    with open(raw_dir / "test_case_order.pkl", "wb") as f:
        pickle.dump(test_case_order, f)

    print(f"[converter] done. node_types={stats['num_node_types']}, edge_dim={stats['edge_dim']}, "
          f"max_time_norm={stats['max_time_norm']:.2f}d")
    return meta
