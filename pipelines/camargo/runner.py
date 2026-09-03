"""
Plain-field prediction pipeline.

Plain-field scenario
--------------------
At the start of the test horizon we observe:
  (a) in-flight cases — cases that started before val_split and are still
      running; we have their full event prefix up to val_split.
  (b) fresh arrivals — new cases arriving during the test period; we only
      know the arrival timestamp.

For each day t in the test horizon Prophet forecasts N_t new arrivals.
Those N_t cases are initialised with (first_act, first_resource, random
intra-day timestamp) and their suffixes are predicted by the process model.
The in-flight cases are predicted once at the start of the horizon using
their full observed prefix.

CC and TT time series are then derived from the union of both groups.

Results are saved under:
    results/plain_field/<model>/<trim>/<run_name>/
        metrics_<run_name>.csv      (CC + TT MAE / MSE)
        arrivals_metrics.csv        (Prophet arrival MAE / MSE)
        arrivals_forecast.png       (arrival rate plot)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent.parent
PF_DIR  = Path(__file__).resolve().parent
_REPO_DIR = ROOT / "GenerativeLSTM" / "GenerativeLSTM"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PF_DIR))

from time_series_preprocessing    import ts_splits_from_log          # noqa: E402
from time_series_creation          import (                           # noqa: E402
    create_concurrent_cases_timeseries,
    create_avg_throughtput_time_timeseries,
)
from create_prefixes_from_windows  import make_three_way_split, load_event_log  # noqa: E402
from arrival                       import compute_arrival_series, ProphetArrivalModel  # noqa: E402
from sos                           import (                           # noqa: E402
    most_frequent_first_activity,
    most_frequent_first_resource,
    empirical_arrival_hour_sampler,
)

_RESULTS = ROOT / "results"


def _load_amiri_params(amiri_dir: Path) -> dict:
    """Load saved best_params.json; fall back to default_params() if absent."""
    import json
    from amiri.params import default_params
    bp = amiri_dir / "best_params.json"
    if bp.exists():
        return json.loads(bp.read_text())
    return default_params()


# ── Data helpers ───────────────────────────────────────────────────────────────

def get_inflight_cases(
    df: pd.DataFrame,
    val_split,
    case_col: str = "caseid",
    time_col: str = "end_timestamp",
) -> pd.DataFrame:
    """
    Return the observable prefix (events before val_split) for all cases that
    started before val_split but haven't finished by val_split.

    These are the "in-flight" cases visible at the start of the test horizon.
    """
    ts = pd.to_datetime(df[time_col])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(None)

    vs = pd.Timestamp(val_split)
    if vs.tzinfo is not None:
        vs = vs.tz_convert(None)

    df2 = df.copy()
    df2["_ts"] = ts

    case_first = df2.groupby(case_col)["_ts"].min()
    case_last  = df2.groupby(case_col)["_ts"].max()
    inflight_ids = case_first.index[(case_first < vs) & (case_last >= vs)]

    return (df2[df2[case_col].isin(inflight_ids) & (df2["_ts"] < vs)]
              .drop(columns=["_ts"])
              .copy())


def build_sos_cases(
    predicted_arrivals: pd.Series,
    first_act: str,
    first_resource: str,
    hour_sampler,
) -> pd.DataFrame:
    """
    One-row-per-predicted-arrival DataFrame (fresh SOS test cases).

    Each case gets task=first_act, user=first_resource, and a random intra-day
    timestamp drawn from the empirical first-event hour distribution.
    caseids are 'pf_000000', 'pf_000001', … so they never collide with real
    case IDs.
    """
    rows, n = [], 0
    for day, count in predicted_arrivals.items():
        day_ts  = pd.Timestamp(day).normalize()
        n_cases = max(0, int(round(float(count))))
        for _ in range(n_cases):
            arrival_ts = day_ts + hour_sampler()
            rows.append({
                "caseid":          f"pf_{n:06d}",
                "task":            first_act,
                "user":            first_resource,
                "end_timestamp":   arrival_ts,
                "start_timestamp": arrival_ts,
            })
            n += 1
    if not rows:
        return pd.DataFrame(
            columns=["caseid", "task", "user", "end_timestamp", "start_timestamp"])
    return pd.DataFrame(rows)


def _rem_time_to_event_log(rt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert rem_time_df → 2-event-per-case DataFrame for CC/TT computation.

    Columns needed in rt_df:
        caseid, start_timestamp, anchor_timestamp, rem_time_days

    Each case gets:
        start event : end_timestamp = start_timestamp
        end   event : end_timestamp = anchor_timestamp + rem_time_days
    """
    start_ts = pd.to_datetime(rt_df["start_timestamp"])
    anchor   = pd.to_datetime(rt_df["anchor_timestamp"])
    end_ts   = anchor + pd.to_timedelta(rt_df["rem_time_days"].astype(float), unit="D")
    if start_ts.dt.tz is not None:
        start_ts = start_ts.dt.tz_convert(None)
    if end_ts.dt.tz is not None:
        end_ts = end_ts.dt.tz_convert(None)
    return pd.concat([
        pd.DataFrame({"caseid": rt_df["caseid"].values, "end_timestamp": start_ts.values}),
        pd.DataFrame({"caseid": rt_df["caseid"].values, "end_timestamp": end_ts.values}),
    ], ignore_index=True)


# ── Per-model plain-field predictors ──────────────────────────────────────────

def predict_bukhsh_plain_field(
    trainer,
    sos_df: pd.DataFrame,
    inflight_df: pd.DataFrame,
) -> tuple:
    """
    Predict remaining time AND suffix for SOS + in-flight cases with BukhshTrainer.

    SOS cases        : acts = [first_act],  time_feats = [0]*5
    In-flight cases  : acts = actual prefix, time_feats from real timestamps

    Returns (rem_time_df, suffix_event_log):
      rem_time_df      — caseid, start_timestamp, anchor_timestamp,
                         prefix_len, rem_time_days
      suffix_event_log — caseid, end_timestamp  (start + predicted-suffix events
                         per case; suitable for CC/TT via compute_cc_tt_metrics)
    """
    from bukhsh.trainer import _compute_time_feats, _normalize
    from tqdm import tqdm

    if not trainer._models:
        trainer._load_all_models()
    trainer._build_role_map()

    _END = "<END>"

    def _strip(t):
        t = pd.Timestamp(t)
        return t.tz_convert(None) if t.tzinfo else t

    n_sos = len(sos_df)
    n_if  = inflight_df["caseid"].nunique()
    rt_rows     = []
    suffix_rows = []

    with tqdm(total=n_sos + n_if, desc="Bukhsh predicting", unit=" cases") as pbar:
        # ── SOS cases ──────────────────────────────────────────────────────────
        for _, row in sos_df.iterrows():
            cid    = row["caseid"]
            acts   = [_normalize(str(row["task"]))]
            roles  = [trainer._get_role(str(row.get("user", "unk")))]
            anchor = _strip(row["end_timestamp"])
            feats  = [0.0, 0.0, 0.0, 0.0, 0.0]

            rem = trainer._predict_rem_time(acts, feats)
            rt_rows.append({
                "caseid":           cid,
                "start_timestamp":  anchor,
                "anchor_timestamp": anchor,
                "prefix_len":       1,
                "rem_time_days":    max(0.0, rem) / 86400,
            })

            suffix_rows.append({"caseid": cid, "end_timestamp": anchor})
            for act, role, abs_ts in trainer._hallucinate(acts, roles, anchor, feats):
                if act == _END or abs_ts is None:
                    continue
                suffix_rows.append({"caseid": cid, "end_timestamp": abs_ts})

            pbar.update(1)

        # ── In-flight cases ────────────────────────────────────────────────────
        for caseid, grp in inflight_df.groupby("caseid"):
            grp    = grp.sort_values("end_timestamp")
            acts   = [_normalize(str(a)) for a in grp["task"]]
            users  = [str(u) for u in grp["user"]]
            roles  = [trainer._get_role(u) for u in users]
            ts     = [_strip(t) for t in grp["end_timestamp"]]
            feats  = _compute_time_feats(ts)

            rem = trainer._predict_rem_time(acts, feats)
            rt_rows.append({
                "caseid":           caseid,
                "start_timestamp":  ts[0],
                "anchor_timestamp": ts[-1],
                "prefix_len":       len(acts),
                "rem_time_days":    max(0.0, rem) / 86400,
            })

            suffix_rows.append({"caseid": caseid, "end_timestamp": ts[0]})
            suffix_rows.append({"caseid": caseid, "end_timestamp": ts[-1]})
            for act, role, abs_ts in trainer._hallucinate(acts, roles, ts[-1], feats):
                if act == _END or abs_ts is None:
                    continue
                suffix_rows.append({"caseid": caseid, "end_timestamp": abs_ts})

            pbar.update(1)

    rem_time_df      = pd.DataFrame(rt_rows)
    suffix_event_log = pd.DataFrame(suffix_rows)
    return rem_time_df, suffix_event_log


def predict_amiri_plain_field(
    trainer,
    sos_df: pd.DataFrame,
    inflight_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict remaining time for SOS + in-flight cases with AmiriTrainer.

    Both groups are combined into a single test_df and passed to
    trainer.predict() (GPS subprocess).  trainer.test_df is restored
    afterwards.  The inference outputs in trainer.work_dir are overwritten
    (training weights in gps_results/ are NOT affected).

    SOS cases (1 event) fall back to mean_case_duration inside GPS.
    In-flight cases (≥ 2 events) get real GPS predictions.

    Returns rem_time_df (caseid, start_timestamp, anchor_timestamp,
                         prefix_len, rem_time_days).
    """
    def _to_naive(df):
        d = df[["caseid", "task", "user", "end_timestamp"]].copy()
        ts = pd.to_datetime(d["end_timestamp"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(None)
        d["end_timestamp"] = ts
        d["task"] = d["task"].astype(str)
        d["user"] = d["user"].astype(str)
        return d

    combined = pd.concat([_to_naive(sos_df), _to_naive(inflight_df)], ignore_index=True)

    import shutil as _shutil

    orig_work_dir    = trainer.work_dir
    orig_dataset_dir = trainer.dataset_dir
    orig_raw_dir     = trainer.raw_dir
    orig_ref_dir     = trainer.ref_dataset_dir
    orig_test        = trainer.test_df

    # Use a dedicated plain-field subdir so inference outputs don't overwrite
    # the original gps_results_infer / rem_time.csv from HPO.
    pf_work_dir = orig_work_dir / "pf"
    pf_work_dir.mkdir(parents=True, exist_ok=True)

    # Symlink gps_results so _write_infer_yaml resolves the training checkpoint.
    pf_gps = pf_work_dir / "gps_results"
    if not pf_gps.exists():
        pf_gps.symlink_to(orig_work_dir / "gps_results")

    # Build a validated ref_meta to ensure edge_dim matches the checkpoint.
    # The meta.pkl on disk may be corrupted (wrong edge_dim) from a previous
    # bad plain-field run.  We validate against gps_results config.yaml and
    # rebuild from training data when there is a mismatch.
    import pickle as _pickle, yaml as _yaml
    from amiri.converter import _build_global_stats

    ref_copy_dir = pf_work_dir / "_ref_meta" / "AMIRI" / "raw"
    ref_copy_dir.mkdir(parents=True, exist_ok=True)
    ref_meta_path = ref_copy_dir / "meta.pkl"

    # Read expected edge_dim: try config.yaml first, fall back to checkpoint
    # weight shape (needed when config.yaml lacks the two_layer_linear_edge_encoder
    # section, as is the case for bpic17-o).
    import torch as _torch
    gps_cfg = orig_work_dir / "gps_results" / "amiri_gps" / "config.yaml"
    expected_edge_dim = None
    if gps_cfg.exists():
        try:
            cfg = _yaml.safe_load(gps_cfg.read_text())
            expected_edge_dim = cfg.get("two_layer_linear_edge_encoder", {}).get("in_dim")
        except Exception:
            pass
    if expected_edge_dim is None:
        ckpts = list((orig_work_dir / "gps_results" / "amiri_gps").rglob("ckpt_best_mae.ckpt"))
        if ckpts:
            try:
                _ckpt = _torch.load(ckpts[0], map_location="cpu")
                _w = _ckpt.get("model_state", {}).get(
                    "model.encoder.edge_encoder.encoder1.weight"
                )
                if _w is not None:
                    expected_edge_dim = int(_w.shape[1])
            except Exception:
                pass

    # Try to reuse existing meta; rebuild if absent or edge_dim is wrong.
    existing_meta_src = orig_dataset_dir / "AMIRI" / "raw" / "meta.pkl"
    need_rebuild = True
    if existing_meta_src.exists():
        try:
            with open(existing_meta_src, "rb") as _f:
                _m = _pickle.load(_f)
            if expected_edge_dim is None or _m.get("edge_dim") == expected_edge_dim:
                _shutil.copy2(existing_meta_src, ref_meta_path)
                need_rebuild = False
        except Exception:
            pass

    if need_rebuild:
        print(f"[pf] rebuilding ref_meta from train+val+orig_test "
              f"(existing edge_dim mismatch, expected {expected_edge_dim})")
        # Use the original test split (not the plain-field combined) so the
        # vocabulary includes all users/activities seen during HPO training.
        ref_stats = _build_global_stats(trainer.train_df, trainer.val_df, orig_test)
        with open(ref_meta_path, "wb") as _f:
            _pickle.dump(ref_stats, _f)

    trainer.ref_dataset_dir = pf_work_dir / "_ref_meta"

    # Use a plain-field-specific dataset dir so training pickles are not overwritten.
    pf_dataset_dir = pf_work_dir / "datasets"
    trainer.work_dir    = pf_work_dir
    trainer.dataset_dir = pf_dataset_dir
    trainer.raw_dir     = pf_dataset_dir / "AMIRI" / "raw"
    trainer.test_df     = combined
    try:
        rt_df = trainer.predict()
    finally:
        trainer.work_dir    = orig_work_dir
        trainer.dataset_dir = orig_dataset_dir
        trainer.raw_dir     = orig_raw_dir
        trainer.ref_dataset_dir = orig_ref_dir
        trainer.test_df     = orig_test
    return rt_df


def predict_camargo_plain_field(
    trainer,
    sos_df: pd.DataFrame,
    inflight_df: pd.DataFrame,
    probe_suffix: str = "_pf",
) -> pd.DataFrame:
    """
    Run GLSTM suffix prediction for SOS + in-flight cases.

    A combined test_df is built (SOS: 1 row each, in-flight: N rows each)
    and predict() is called ONCE.  Camargo's predict() naturally handles both:
    – 1-row cases → fresh; uses prefix_mode='start_plus_first'
    – N-row cases → in-flight; uses full prefix regardless of prefix_mode

    A probe directory is used so real gen_*.csv files are not overwritten.

    Returns pred_log (caseid, task, user, end_timestamp) from to_event_log().
    """
    import shutil
    from camargo.trainer import CamargoTrainer as _CT

    # ── Build combined test_df ─────────────────────────────────────────────────
    def _to_naive(df):
        d = df[["caseid", "task", "user", "end_timestamp"]].copy()
        ts = pd.to_datetime(d["end_timestamp"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(None)
        d["end_timestamp"] = ts
        return d

    parts = []
    if not sos_df.empty:
        parts.append(_to_naive(sos_df))
    if not inflight_df.empty:
        parts.append(_to_naive(inflight_df))

    if not parts:
        return pd.DataFrame(columns=["caseid", "end_timestamp"])

    combined_raw = pd.concat(parts, ignore_index=True)

    # Apply role + index mapping
    combined = _CT._apply_roles(combined_raw.copy(), trainer._resource_map)
    combined["ac_index"] = combined["task"].map(trainer.ac_index).fillna(0).astype(int)
    combined["rl_index"] = combined["role"].map(trainer.rl_index).fillna(0).astype(int)

    # ── Probe directory ────────────────────────────────────────────────────────
    probe_name   = trainer.run_name + probe_suffix
    orig_out     = _REPO_DIR / trainer._out_base()
    probe_out    = orig_out.parent / probe_name
    probe_params = probe_out / "parameters"
    probe_params.mkdir(parents=True, exist_ok=True)
    orig_params  = orig_out / "parameters"
    for fname in ("model_parameters.json", "resource_map.csv"):
        src = orig_params / fname
        if src.exists():
            shutil.copy2(src, probe_params / fname)
    for h5 in orig_out.glob("*.h5"):
        shutil.copy2(h5, probe_out / h5.name)

    # ── Predict ────────────────────────────────────────────────────────────────
    # output_dir must be overridden alongside run_name/test_df: to_event_log()
    # saves to self.output_dir (set once at construction, NOT derived from
    # self.run_name at call time), so without this override it silently
    # overwrites the original full-trace event_log.csv with the plain-field
    # probe's event log — discovered the hard way when a test run clobbered
    # loan_flat's full-trace predictions (recovered by re-decoding the
    # untouched gen_*.csv against the original test_df).
    orig_test       = trainer.test_df
    orig_name       = trainer.run_name
    orig_output_dir = trainer.output_dir
    trainer.test_df    = combined
    trainer.run_name    = probe_name
    trainer.output_dir = (str(Path(orig_output_dir).parent / probe_name)
                          if orig_output_dir else orig_output_dir)
    try:
        pred_paths = trainer.predict(
            prefix_mode="start_plus_first",   # applies to 1-row (SOS) cases only
            full_prefix_only=True,
        )
        pred_log = trainer.to_event_log(pred_paths)
    finally:
        trainer.test_df    = orig_test
        trainer.run_name    = orig_name
        trainer.output_dir = orig_output_dir

    return pred_log


# ── Arrival-rate evaluation helpers ───────────────────────────────────────────

def save_arrival_eval(
    predicted_arrivals: pd.Series,
    arrival_series: pd.Series,
    test_index,
    out_dir: Path,
    dataset: str,
    trim: str,
) -> dict:
    """
    Save Prophet arrival-rate metrics CSV + forecast PNG to out_dir.

    Returns dict with mae, mse, total_pred, total_actual.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Align actual arrivals to test index (tz-naive)
    test_idx_naive = pd.DatetimeIndex(test_index)
    if test_idx_naive.tz is not None:
        test_idx_naive = test_idx_naive.tz_convert(None)
    actual = arrival_series.reindex(test_idx_naive).fillna(0)

    mae = mean_absolute_error(actual.values, predicted_arrivals.values)
    mse = mean_squared_error( actual.values, predicted_arrivals.values)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metrics CSV
    pd.DataFrame([dict(dataset=dataset, trim=trim, mae=round(mae, 6),
                       mse=round(mse, 6),
                       total_predicted=round(float(predicted_arrivals.sum()), 1),
                       total_actual=round(float(actual.sum()), 1))
                  ]).to_csv(out_dir / "arrivals_metrics.csv", index=False)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=False)

    ax = axes[0]
    arrival_series[arrival_series.index < test_idx_naive[0]].plot(
        ax=ax, color="steelblue", alpha=0.5, lw=0.9, label="train (actual)")
    actual.plot(ax=ax, color="green", lw=1.2, label="test (actual)")
    predicted_arrivals.plot(ax=ax, color="darkorange", lw=1.4, ls="--",
                            label="test (Prophet)")
    ax.axvline(test_idx_naive[0], color="gray", ls=":", lw=0.8)
    ax.set_title(f"{dataset} / {trim} — arrival rate forecast (Prophet)")
    ax.set_ylabel("cases / day")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    err = predicted_arrivals.values - actual.values
    ax2.bar(predicted_arrivals.index, err,
            color=["#d62728" if e > 0 else "#1f77b4" for e in err],
            alpha=0.7, width=0.9, label="error (pred − actual)")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title(f"Arrival forecast error   MAE={mae:.2f}   MSE={mse:.2f}")
    ax2.set_ylabel("pred − actual")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "arrivals_forecast.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return dict(mae=mae, mse=mse,
                total_pred=float(predicted_arrivals.sum()),
                total_actual=float(actual.sum()))


# ── CC/TT metrics helper ───────────────────────────────────────────────────────

def compute_cc_tt_metrics(
    pred_event_log: pd.DataFrame,
    cc_test: pd.Series,
    tt_test: pd.Series,
) -> tuple[pd.Series, pd.Series, dict]:
    """
    Derive CC and TT predictions from pred_event_log and compute MAE/MSE.

    pred_event_log must have columns: caseid, end_timestamp.
    The log can be either a 2-event-per-case synthetic log (from
    _rem_time_to_event_log) or a full predicted event log (from
    predict_camargo_plain_field).

    Returns (cc_pred_series, tt_pred_series, metrics_dict).
    """
    log = pred_event_log.copy()
    ts  = pd.to_datetime(log["end_timestamp"], utc=True).dt.tz_convert(None)
    log["end_timestamp"] = ts

    cc_p = (create_concurrent_cases_timeseries(
                log, time_col="end_timestamp", case_col="caseid",
                window="days", plot=False)
            .reindex(cc_test.index).ffill().bfill().fillna(0))
    tt_p = (create_avg_throughtput_time_timeseries(
                log, time_col="end_timestamp", case_col="caseid",
                window="days", plot=False)
            .reindex(tt_test.index).ffill().bfill().fillna(0))
    m = dict(
        cc_mae=round(mean_absolute_error(cc_test.to_numpy(), cc_p.to_numpy()), 6),
        cc_mse=round(mean_squared_error( cc_test.to_numpy(), cc_p.to_numpy()), 6),
        tt_mae=round(mean_absolute_error(tt_test.to_numpy(), tt_p.to_numpy()), 6),
        tt_mse=round(mean_squared_error( tt_test.to_numpy(), tt_p.to_numpy()), 6),
    )
    return cc_p, tt_p, m


def save_pf_metrics(
    dataset: str,
    model_label: str,
    metrics: dict,
    out_dir: Path,
) -> None:
    """Save plain-field metrics CSV to out_dir/metrics_<dataset>_test_full.csv."""
    run_name = f"{dataset}_test_full"
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        dict(dataset=dataset, series="concurrent_cases", model=model_label,
             mse=metrics["cc_mse"], mae=metrics["cc_mae"]),
        dict(dataset=dataset, series="throughput_time",  model=model_label,
             mse=metrics["tt_mse"], mae=metrics["tt_mae"]),
    ]).to_csv(out_dir / f"metrics_{run_name}.csv", index=False)


def save_pf_plot(
    cc_pred: pd.Series,
    tt_pred: pd.Series,
    cc_test: pd.Series,
    tt_test: pd.Series,
    out_dir: Path,
    dataset: str,
    trim: str,
    model_label: str,
) -> None:
    """Save 2-panel CC + TT prediction vs actual plot to out_dir/predictions.png."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, actual, pred, title, ylabel in [
        (axes[0], cc_test, cc_pred, "Concurrent Cases",    "# cases"),
        (axes[1], tt_test, tt_pred, "Avg Throughput Time", "hours"),
    ]:
        actual.plot(ax=ax, color="green",     lw=1.4,          label="actual")
        pred.plot(  ax=ax, color="steelblue", lw=1.4, ls="--", label="predicted")
        ax.set_title(f"{dataset} / {trim} — {title}\n({model_label})")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _trim_kw(trim_str: str) -> dict:
    base = dict(trim_pct=0.25, trim_k=1.5, trim_frac=0.6, trim_window=7)
    if trim_str == "none":
        return {**base, "trim_method": None}
    method, val = trim_str.rsplit("_", 1)
    val = float(val)
    if method == "magnitude": return {**base, "trim_method": "magnitude", "trim_k": val}
    if method == "pct":       return {**base, "trim_method": "pct",       "trim_pct": val}
    if method == "peak":      return {**base, "trim_method": "peak",      "trim_frac": val}
    raise ValueError(f"Unknown trim: {trim_str}")
