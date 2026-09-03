"""
run_experiments_real.py
=======================

"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure the project root is on the Python path so local modules (bukhsh,
# camargo, time_series_creation, …) can be imported regardless of where the
# script is invoked from.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "steady_state_detection"))

# Use the non-interactive Agg backend so plots can be saved to disk without a
# display (required for headless servers / SSH sessions).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pm4py
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_absolute_error, mean_squared_error

from bukhsh.params import default_params as bukhsh_default_params
from bukhsh.trainer import BukhshTrainer
from camargo.params import default_params as camargo_params
from camargo.trainer import CamargoTrainer
from create_prefixes_from_windows import make_three_way_split
from setttings import set_global_seed
from time_series_creation import (
    create_avg_throughtput_time_timeseries,
    create_concurrent_cases_timeseries,
)
from time_series_preprocessing import Split3WayConfig, split_timeseries, ts_splits_from_log

# =============================================================================
# Paths
# =============================================================================

DATA_DIR    = ROOT / "data" / "real-life"   # XES input logs
RESULTS_DIR = ROOT / "results"              # metric CSVs + plots
BEST_MODELS = ROOT / "best_models"          # HPO trial artefacts + final models

# =============================================================================
# Trim configurations
# =============================================================================

# Rolling-window size (days) used by all trimming methods to smooth the
# concurrent-cases series before detecting the tail.
TRIM_WINDOW: int = 7

# All seven trim configurations that will be applied to each log.
# Each entry is a dict accepted by ts_splits_from_log / apply_trim:
#   method : None | "peak" | "magnitude"   trimming strategy
#   frac   : float   peak threshold as fraction of global max (used by "peak")
#   k      : float   number of std-devs below expanding mean (used by "magnitude")
#   pct    : float   fraction of tail to drop (used by "pct", not active here)
TRIM_CONFIGS: list[dict] = [
    {"method": None,        "frac": 0.60, "k": 1.5, "pct": 0.25},  # no trimming
    {"method": "peak",      "frac": 0.60, "k": 1.5, "pct": 0.25},  # gentle peak trim
    {"method": "peak",      "frac": 0.70, "k": 1.5, "pct": 0.25},  # moderate peak trim
    {"method": "peak",      "frac": 0.80, "k": 1.5, "pct": 0.25},  # aggressive peak trim
    {"method": "magnitude", "frac": 0.60, "k": 1.0, "pct": 0.25},  # aggressive magnitude trim
    {"method": "magnitude", "frac": 0.60, "k": 2.0, "pct": 0.25},  # moderate magnitude trim
    {"method": "magnitude", "frac": 0.60, "k": 3.0, "pct": 0.25},  # gentle magnitude trim
]

# =============================================================================
# HPO settings
# =============================================================================

# Bukhsh hyperparameter search space. Each key maps to a list of candidate
# values. hyperopt will choose from these lists using Bayesian TPE search.
BUKHSH_SPACE: dict[str, list] = {
    "num_heads":     [2, 4],               # number of attention heads
    "batch_size":    [16, 32],             # training batch size
    "epochs":        [20, 50],             # training epochs per trial
    "learning_rate": [0.0005, 0.001, 0.005],
}
# Number of Bayesian HPO trials to run for Bukhsh.
# Each trial trains a full model, so this is the dominant time cost.
BUKHSH_MAX_EVAL: int = 12

# Number of Bayesian HPO trials inside Camargo's trainer.train() call.
# Camargo handles its own hyperopt loop internally.
CAMARGO_MAX_EVAL: int = 10

# Whether to use only fully-completed traces for train/val/test splits.
# True  → a case is included in a split only if its last event falls within
#          that period (clean, no partial traces).
# False → prefix/suffix split (prefix goes to earlier period, suffix to later).
FULL_TRACES: bool = True

# Column renaming map from pm4py / XES standard names to the project-internal
# names expected by BukhshTrainer, CamargoTrainer, and the KPI functions.
# Columns not present in the DataFrame (e.g. lifecycle:transition) are ignored
# by pandas rename(), so this map is safe for all real-life logs.
_COLS: dict[str, str] = {
    "case:concept:name":    "caseid",        # unique case identifier
    "concept:name":         "task",          # activity label
    "lifecycle:transition": "event_type",    # e.g. "complete" (often absent)
    "org:resource":         "user",          # resource / employee ID
    "time:timestamp":       "end_timestamp", # event completion timestamp
}

_COLS_GROUP: dict[str, str] = {
    "case:concept:name":    "caseid",        # unique case identifier
    "concept:name":         "task",          # activity label
    "lifecycle:transition": "event_type",    # e.g. "complete" (often absent)
    "org:group":         "user",          # resource / employee ID
    "time:timestamp":       "end_timestamp", # event completion timestamp
}

# =============================================================================
# Internal helpers
# =============================================================================

def _trim_dir(method, pct, k, frac) -> str:
    """Return the canonical folder-name string for a trim configuration.

    This name is used consistently across the directory layout so that every
    (log, trim_config) combination gets its own isolated output folder.

    Examples
    --------
    >>> _trim_dir(None,         0.25, 1.5, 0.60)  # 'none'
    >>> _trim_dir("peak",       0.25, 1.5, 0.60)  # 'peak_0.6'
    >>> _trim_dir("peak",       0.25, 1.5, 0.80)  # 'peak_0.8'
    >>> _trim_dir("magnitude",  0.25, 1.0, 0.60)  # 'magnitude_1'
    >>> _trim_dir("magnitude",  0.25, 2.0, 0.60)  # 'magnitude_2'
    """
    if method is None:        return "none"
    if method == "pct":       return f"pct_{pct:g}"
    if method == "magnitude": return f"magnitude_{k:g}"
    if method == "peak":      return f"peak_{frac:g}"
    return str(method)


def _trim_cfg_by_name(name: str) -> dict:
    """Look up a trim config dict by its canonical name string.

    Parameters
    ----------
    name:
        One of the strings returned by _trim_dir, e.g. ``'peak_0.6'``.

    Returns
    -------
    The matching dict from TRIM_CONFIGS.

    Raises
    ------
    ValueError
        If no config with that name exists.
    """
    for cfg in TRIM_CONFIGS:
        if _trim_dir(cfg["method"], cfg["pct"], cfg["k"], cfg["frac"]) == name:
            return cfg
    available = [_trim_dir(c["method"], c["pct"], c["k"], c["frac"]) for c in TRIM_CONFIGS]
    raise ValueError(f"Unknown trim config {name!r}. Available: {available}")


def _resolve_log(log: str | Path) -> Path:
    """Resolve a log name or path to an absolute Path.

    Accepts either a full/relative path to an existing XES file, or just the
    file stem (e.g. ``'helpdesk'``), in which case the file is looked up in
    DATA_DIR.

    Raises
    ------
    FileNotFoundError
        If the log cannot be found by either strategy.
    """
    p = Path(log)
    if p.exists():
        return p.resolve()
    # Try appending .xes and searching DATA_DIR
    candidate = DATA_DIR / f"{p.stem}.xes"
    if candidate.exists():
        return candidate
    available = ", ".join(f.stem for f in sorted(DATA_DIR.glob("*.xes")))
    raise FileNotFoundError(
        f"Log not found: {log!r}. "
        f"Provide a full path or a stem from DATA_DIR. Available: {available}"
    )


def _load_data(log_path: Path, trim_cfg: dict, seed: int = 1904):
    """Load an XES log and produce all data structures needed for HPO + evaluation.

    This function is the single data-loading entry point. It reads the XES file
    once and derives everything else from that single parse:

      1. **Time-series splits** — concurrent-cases and throughput-time series
         are trimmed according to trim_cfg and split into train / val / test
         periods. The split *dates* (train_split, val_split) are derived from
         the concurrent-cases series because it is the more complete and
         less noisy of the two metrics.

      2. **Event-log splits** — the raw event log is filtered into three
         DataFrames using the same split dates. Only fully-completed traces
         are included in each partition (FULL_TRACES=True).

      3. **HPO evaluation set** — a separate test set (val_as_test_df) that
         covers cases active at train_split. During HPO, the model is trained
         on train_df only and evaluated on this set, so the val period is
         never seen during HPO. This avoids data leakage while still
         providing a representative evaluation window.

    Parameters
    ----------
    log_path:
        Absolute path to the XES file.
    trim_cfg:
        One entry from TRIM_CONFIGS, e.g.
        ``{"method": "peak", "frac": 0.6, "k": 1.5, "pct": 0.25}``.
    seed:
        Random seed passed to set_global_seed for reproducibility.

    Returns
    -------
    cc : dict
        Time-series split for concurrent cases with keys:
        ``raw, trimmed, train, val, test, train_split, val_split``.
    tt : dict
        Same structure for average throughput time.
    train_df : pd.DataFrame
        Event log rows for the training period.
        Columns: caseid, task, user, end_timestamp (+ any extra XES columns).
    val_df : pd.DataFrame
        Event log rows for the validation period.
    test_df : pd.DataFrame
        Event log rows for the test period (in-flight + fresh cases).
    val_as_test_df : pd.DataFrame
        Cases active at train_split, used as the HPO evaluation target.
    empty_val : pd.DataFrame
        Empty DataFrame with the same columns as train_df. Passed to
        BukhshTrainer as val_df during HPO trials (no validation data
        is needed inside a single trial — we evaluate manually after predict).
    """
    set_global_seed(seed)

    # Read the XES file. pm4py returns a pandas DataFrame with standard
    # column names (case:concept:name, concept:name, time:timestamp, …).
    log = pm4py.read_xes(str(log_path))

    if trim_cfg["method"] == "ssd":
        # log_path is the FULL, untruncated data/real-life/<name>.xes.
        # ssd is structurally identical to every other trim: one log read
        # (the full one, already done above as `log`), a cutoff (found via
        # steady_state_detection instead of a fixed formula), and
        # everything -- splits, ground truth, AND train/val/test_df --
        # derived from that single full log. The cutoff is used solely to
        # (a) place train_split/val_split and (b) bound the CC/TT scoring
        # window -- never to censor which events are visible to
        # make_three_way_split.
        #
        # NOTE: deliberately NOT using ts_splits_from_log(..., cut_date=...).
        # create_avg_throughtput_time_timeseries's cut_date filters cases at
        # exact-timestamp granularity (`first_last["last"] <= cut_ts`), while
        # create_concurrent_cases_timeseries's cut_date clips at day-bucket
        # granularity -- since canonical_end (the ssd cutoff) is a midnight
        # timestamp, the TT instant-level filter would silently drop every
        # case completing anywhere during the cutoff's own calendar day.
        # Instead: raw series, no cut_date, sliced by day-bucket index
        # (`raw.index <= canonical_end`), so TT matches CC bucket-for-bucket.
        from ssd_trim import run_ssd_trim
        full_cc_raw = create_concurrent_cases_timeseries(log, plot=False)
        ssd_result = run_ssd_trim(log, window_step="D")
        canonical_end = ssd_result["cutoff"] if ssd_result["cutoff"] is not None else full_cc_raw.index[-1]
        full_cc_trimmed = full_cc_raw[full_cc_raw.index <= canonical_end]
        split_cfg = Split3WayConfig(train_frac=0.70, val_frac=0.10, test_frac=0.20)
        _, _, _, train_split, val_split = split_timeseries(full_cc_trimmed, split_cfg)

        full_tt_raw = create_avg_throughtput_time_timeseries(log, plot=False)
        full_tt_trimmed = full_tt_raw[full_tt_raw.index <= canonical_end]

        def _slice(raw, trimmed):
            idx = trimmed.index
            lo = train_split.tz_convert(None) if idx.tz is None else train_split
            hi = val_split.tz_convert(None) if idx.tz is None else val_split
            return {
                "raw": raw, "trimmed": trimmed,
                "train": trimmed[idx <= lo],
                "val": trimmed[(idx > lo) & (idx <= hi)],
                "test": trimmed[idx > hi],
                "train_split": train_split, "val_split": val_split,
            }

        ts = {
            "concurrent_cases": _slice(full_cc_raw, full_cc_trimmed),
            "throughput_time": _slice(full_tt_raw, full_tt_trimmed),
        }
    else:
        # Derive train/val/test split timestamps from the concurrent-cases series.
        # ts_splits_from_log applies the chosen trim method, computes the series,
        # and returns both the split timestamps and the pre-split series segments.
        ts = ts_splits_from_log(
            log,
            trim_method=trim_cfg["method"],
            trim_pct=trim_cfg["pct"],
            trim_k=trim_cfg["k"],
            trim_frac=trim_cfg["frac"],
            trim_window=TRIM_WINDOW,
            cut_date=None,  # no artificial cut-off for real-life logs
        )
    cc = ts["concurrent_cases"]  # dict: raw, trimmed, train, val, test, train_split, val_split
    tt = ts["throughput_time"]   # same structure
    train_split = cc["train_split"]  # last date of the training period (pd.Timestamp, UTC)
    val_split   = cc["val_split"]    # last date of the validation period (pd.Timestamp, UTC)

    # Convert the pm4py DataFrame to the project's internal column names.
    # We reuse the already-loaded DataFrame instead of calling load_event_log()
    # to avoid parsing the XES file a second time.
    df = pm4py.convert_to_dataframe(log)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)

    df = df.dropna(subset=["case:concept:name"])
    if not "org:resource" in df.columns:
        df = df.rename(columns=_COLS_GROUP)
    else:
        df = df.rename(columns=_COLS)
    # Fill missing activity / resource labels with a sentinel so models don't
    # receive NaN tokens.
    df["task"] = df["task"].fillna("unk")
    df["user"] = df["user"].fillna("unk")

    # Standard 3-way split: a case belongs to the partition in which its last
    # event falls. FULL_TRACES=True means only complete traces are included;
    # cases straddling a boundary are dropped from both sides.
    train_df, val_df, test_df = make_three_way_split(
        df, case_col="caseid", time_col="end_timestamp",
        train_split=train_split, val_split=val_split, full_traces=FULL_TRACES,
    )

    # HPO evaluation set: pass val_split=train_split so that make_three_way_split
    # treats cases active at train_split as the "test" set. This gives us the
    # in-flight cases that will exist at the moment of prediction without ever
    # touching the true val/test data during HPO.
    _, _, val_as_test_df = make_three_way_split(
        df, case_col="caseid", time_col="end_timestamp",
        train_split=train_split, val_split=train_split, full_traces=FULL_TRACES,
    )

    # BukhshTrainer requires a val_df argument even when we don't want to use
    # one during HPO. Pass an empty DataFrame with matching columns.
    empty_val = pd.DataFrame(columns=train_df.columns)

    return cc, tt, train_df, val_df, test_df, val_as_test_df, empty_val


def _kpi_series(event_log: pd.DataFrame, cc_index, tt_index):
    """Compute concurrent-cases and throughput-time arrays from a predicted event log.

    Converts the model's output event log into the two KPI time series used
    for evaluation, then aligns them to the ground-truth test index so that
    element-wise comparison (MSE / MAE) is valid.

    Alignment strategy:
      - reindex() to the test index (adds NaN for missing dates)
      - ffill()   propagates the last known value forward (no sudden drops)
      - bfill()   fills leading NaNs (model predicts later than test starts)
      - fillna(0) safety net for any remaining NaN (no cases predicted at all)

    Parameters
    ----------
    event_log:
        DataFrame with columns ``caseid`` and ``end_timestamp`` (timezone-naive).
        Typically the output of BukhshTrainer.predict() or
        CamargoTrainer.to_event_log(), after stripping the timezone.
    cc_index:
        DatetimeIndex of the ground-truth concurrent-cases test series.
    tt_index:
        DatetimeIndex of the ground-truth throughput-time test series.

    Returns
    -------
    cc_arr : np.ndarray  shape (len(cc_index),)
    tt_arr : np.ndarray  shape (len(tt_index),)
    """
    pred_cc = create_concurrent_cases_timeseries(
        event_log, time_col="end_timestamp", case_col="caseid", window="days", plot=False)
    pred_tt = create_avg_throughtput_time_timeseries(
        event_log, time_col="end_timestamp", case_col="caseid", window="days", plot=False)

    cc_arr = pred_cc.reindex(cc_index).ffill().bfill().fillna(0).to_numpy()
    tt_arr = pred_tt.reindex(tt_index).ffill().bfill().fillna(0).to_numpy()
    return cc_arr, tt_arr


def _save_plots(cc_actual, cc_pred, tt_actual, tt_pred,
                cc_index, tt_index, model_name: str, run_name: str, out_dir: Path):
    """Save prediction-vs-actual PNG plots for both KPI series.

    Produces two files in out_dir:
      concurrent_cases.png
      throughput_time.png

    Each plot shows the actual test series in green and the model prediction
    in purple (dashed), with MSE and MAE in the title.

    Parameters
    ----------
    cc_actual, cc_pred : array-like
        Ground-truth and predicted concurrent-cases values (same length).
    tt_actual, tt_pred : array-like
        Ground-truth and predicted throughput-time values (same length).
    cc_index, tt_index : DatetimeIndex
        Date index for x-axis labels.
    model_name : str
        Label used in the legend, e.g. ``'bukhsh_hpo'``.
    run_name : str
        Dataset + split identifier used in the plot title.
    out_dir : Path
        Directory where the PNG files are written.
    """
    for series_name, actual, pred, index in [
        ("concurrent_cases", cc_actual, cc_pred, cc_index),
        ("throughput_time",  tt_actual, tt_pred, tt_index),
    ]:
        mse = mean_squared_error(actual, pred)
        mae = mean_absolute_error(actual, pred)
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(index, actual, color="green",  label="actual",               linewidth=1.5)
        ax.plot(index, pred,   color="purple", label=f"{model_name} (pred)", linestyle="--")
        ax.set_title(f"{run_name} — {series_name}  MSE={mse:.2f}  MAE={mae:.2f}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{series_name}.png", dpi=150, bbox_inches="tight")
        plt.close()  # release memory; important when iterating over many configs


# =============================================================================
# Bukhsh HPO
# =============================================================================

def run_bukhsh(log_path: str | Path, trim_cfg: dict) -> dict:
    """Run the full Bukhsh experiment for one log and one trim configuration.

    Steps
    -----
    1. Load data and compute splits via _load_data().
    2. Run Bayesian HPO (hyperopt TPE) with BUKHSH_MAX_EVAL trials.
       Each trial trains BukhshTrainer on train_df and evaluates on the
       val-period KPIs (concurrent-cases MAE is the HPO loss).
    3. Select the trial with the lowest val CC MAE as the winner.
    4. Retrain the winning configuration on train_df + val_df combined.
    5. Predict event suffixes for all test-period cases.
    6. Compute CC and TT MSE / MAE against the ground-truth test series.
    7. Save best_params.json, hpo_results.csv, metrics CSV, timing CSV,
       and prediction plots.

    Checkpointing
    -------------
    * Each trial saves its result to ``hpo_trials/trial_NNN/result.json``
      immediately after finishing. On re-run, trials with an existing
      result.json are skipped entirely (no retraining, no prediction).
    * The hyperopt Trials object is pickled to ``hyperopt_trials.pkl``
      at the start of each trial call. Even if the process crashes mid-trial,
      all previously completed trials are preserved and will be resumed.
    * The final model checks for ``meta.pkl``; if found, only prediction
      is re-run (much faster than retraining).

    Parameters
    ----------
    log_path:
        Path to the XES file or log stem name (resolved via _resolve_log).
    trim_cfg:
        One entry from TRIM_CONFIGS. The trim method and its parameter
        determine which part of the time series is used for splitting.

    Returns
    -------
    dict with keys ``'concurrent_cases'`` and ``'throughput_time'``, each
    mapping to ``{'mse': float, 'mae': float}`` on the test period.
    """
    log_path  = _resolve_log(log_path)
    dataset   = log_path.stem
    trim_name = _trim_dir(trim_cfg["method"], trim_cfg["pct"], trim_cfg["k"], trim_cfg["frac"])
    # run_name identifies the final model artefacts (dataset + split mode).
    run_name  = f"{dataset}_test_full"

    print(f"\n{'='*60}")
    print(f"Bukhsh HPO | {dataset} | {trim_name}")
    print(f"{'='*60}")

    cc, tt, train_df, val_df, test_df, val_as_test_df, empty_val = _load_data(log_path, trim_cfg)

    # HPO trial artefacts are stored under best_models/{dataset}/{trim}/bukhsh/hpo_trials/
    # Final model goes to best_models/{dataset}/{trim}/bukhsh/{run_name}/
    hpo_dir  = BEST_MODELS / dataset / trim_name / "bukhsh" / "hpo_trials"
    best_dir = BEST_MODELS / dataset / trim_name / "bukhsh"
    hpo_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Bayesian HPO
    # ------------------------------------------------------------------

    # Try to resume from a previous interrupted run.
    _trials_pkl = hpo_dir / "hyperopt_trials.pkl"
    if _trials_pkl.exists():
        with open(_trials_pkl, "rb") as f:
            _hpo_trials = pickle.load(f)
        # Count only completed (status=ok) trials — partial trials don't count.
        _completed = len([t for t in _hpo_trials.trials
                          if t["result"].get("status") == "ok"])
        print(f"Resuming: {_completed}/{BUKHSH_MAX_EVAL} trials already done")
    else:
        _hpo_trials = Trials()
        _completed  = 0
        print(f"Starting Bukhsh HPO: {BUKHSH_MAX_EVAL} trials")

    hpo_results: list[dict] = []
    # Mutable counter so the closure can track which trial index it is on.
    # Starts at _completed so new trial IDs continue the existing numbering.
    _trial_counter = [_completed]

    def _objective(trial_cfg):
        """hyperopt objective: train one Bukhsh trial and return val CC MAE.

        This closure is called by hyperopt's fmin() for each trial. It:
          - assigns a unique ID (trial_000, trial_001, …)
          - checkpoints the Trials state before training
          - skips training if a cached result.json already exists
          - handles partially-trained models (meta.pkl exists but no result)
          - trains BukhshTrainer and evaluates KPIs on the val period
          - saves result.json and returns the loss to hyperopt
        """
        i = _trial_counter[0]
        _trial_counter[0] += 1
        trial_id  = f"trial_{i:03d}"
        trial_dir = hpo_dir / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Persist the Trials object before doing any work so that a crash
        # after this point still preserves all previously completed trials.
        with open(_trials_pkl, "wb") as f:
            pickle.dump(_hpo_trials, f)

        # Write config only on first creation — never overwrite.  If weights already
        # exist from a previous run, config.json holds the original num_heads so
        # _train_task rebuilds the right architecture and load_weights succeeds.
        _cfg_path = trial_dir / "config.json"
        if not _cfg_path.exists():
            _cfg_path.write_text(json.dumps(trial_cfg))

        result_path = trial_dir / "result.json"
        _tasks   = ("next_act", "next_role", "next_time", "rem_time")
        _w_dir   = trial_dir / "results"
        _n_weights = sum(
            (_w_dir / f"{trial_id}_{t}.weights.h5").exists() for t in _tasks
        )

        # Full cache hit: all 4 weights present and result already computed.
        if result_path.exists() and _n_weights == 4:
            cached = json.loads(result_path.read_text())
            print(f"  [{i+1}/{BUKHSH_MAX_EVAL}] cached  CC MAE={cached['val_cc_mae']:.4f}  ({trial_id})")
            hpo_results.append(cached)
            return {"loss": cached["val_cc_mae"], "status": STATUS_OK}

        # Stale result.json without all weights — clean up.
        if result_path.exists():
            result_path.unlink()

        print(f"\n  [{i+1}/{BUKHSH_MAX_EVAL}] {trial_cfg}  ({_n_weights}/4 weights already on disk)")
        params  = bukhsh_default_params(**trial_cfg)
        trainer = BukhshTrainer(
            train_df,
            val_df=empty_val,         # no val data needed inside a single HPO trial
            test_df=val_as_test_df,   # predict on HPO eval set (cases at train_split)
            run_name=trial_id,
            params=params,
            output_dir=trial_dir,
        )

        # Always call run(): it skips tasks whose weight files already exist
        # and calls _save_meta() at the end — predict() needs meta.pkl.
        t0 = time.perf_counter()
        trainer.run()
        train_s = round(time.perf_counter() - t0, 2)

        # Predict event suffixes for the HPO eval set.
        event_log, _ = trainer.predict()
        event_log = event_log.copy()
        # BukhshTrainer returns timezone-aware timestamps; strip tz for the KPI
        # functions which expect naive datetimes.
        event_log["end_timestamp"] = (
            pd.to_datetime(event_log["end_timestamp"], utc=True).dt.tz_convert(None))

        # Evaluate against the val-period KPI series.
        cc_pred, tt_pred = _kpi_series(event_log, cc["val"].index, tt["val"].index)
        cc_mae = mean_absolute_error(cc["val"].to_numpy(), cc_pred)
        tt_mae = mean_absolute_error(tt["val"].to_numpy(), tt_pred)

        # Persist result so this trial is skipped on re-run.
        row = {**trial_cfg, "trial": trial_id,
               "val_cc_mae": round(cc_mae, 4), "val_tt_mae": round(tt_mae, 4),
               "train_s": train_s}
        hpo_results.append(row)
        result_path.write_text(json.dumps(row))
        print(f"    CC MAE={cc_mae:.4f}  TT MAE={tt_mae:.4f}  time={train_s:.0f}s")

        # hyperopt minimises loss; use CC MAE as the primary HPO objective.
        return {"loss": cc_mae, "status": STATUS_OK}

    # Run (or resume) Bayesian optimisation.
    _space = {k: hp.choice(k, v) for k, v in BUKHSH_SPACE.items()}
    fmin(fn=_objective, space=_space, algo=tpe.suggest,
         max_evals=BUKHSH_MAX_EVAL, trials=_hpo_trials, show_progressbar=False)

    # fmin() skips calling _objective when all max_evals trials are already in
    # _hpo_trials, so hpo_results may be empty on a full cache hit. Load from
    # the individual result.json files in that case.
    if not hpo_results:
        for rp in sorted(hpo_dir.glob("trial_*/result.json")):
            hpo_results.append(json.loads(rp.read_text()))
        print(f"Loaded {len(hpo_results)} cached results from disk")

    # Rank trials by val CC MAE; the first row is the winner.
    hpo_df = pd.DataFrame(hpo_results).sort_values("val_cc_mae")
    hpo_df.to_csv(best_dir / "hpo_results.csv", index=False)

    best_row    = hpo_df.iloc[0].to_dict()
    best_params = {k: best_row[k] for k in BUKHSH_SPACE}
    # Cast to proper Python types (hyperopt returns numpy scalars).
    best_params = {
        k: int(v) if k in ("num_heads", "batch_size", "epochs") else float(v)
        for k, v in best_params.items()
    }
    (best_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))
    print(f"Best params: {best_params}")

    # ------------------------------------------------------------------
    # Step 2: Final retrain on train + val with the best hyperparameters
    # ------------------------------------------------------------------
    # Retraining on the full train+val set (rather than just train) gives the
    # model access to more data before it has to generalise to the test period.
    final_dir    = best_dir / run_name
    trainer_final = BukhshTrainer(
        train_df, val_df=val_df, test_df=test_df,
        run_name=run_name,
        params=bukhsh_default_params(**best_params),
        output_dir=final_dir,
    )

    _final_tasks = ("next_act", "next_role", "next_time", "rem_time")
    _final_w_dir = final_dir / "results"
    _final_complete = all(
        (_final_w_dir / f"{run_name}_{t}.weights.h5").exists()
        for t in _final_tasks
    )
    if _final_complete:
        print("Final model exists (4/4 weights), skipping retrain")
        final_train_s = 0.0
    else:
        t0 = time.perf_counter()
        trainer_final.run()
        final_train_s = round(time.perf_counter() - t0, 2)
        print(f"Final retrain: {final_train_s:.1f}s")

    # Record training time for later analysis.
    pd.DataFrame([
        dict(model="bukhsh_hpo", phase="train", params_json=json.dumps(best_params),
             val_mse=float("nan"), time_s=final_train_s, is_best=True,
             dataset=run_name, series="all"),
    ]).to_csv(best_dir / f"time_{run_name}.csv", index=False)

    # ------------------------------------------------------------------
    # Step 3: Predict on the test period and evaluate KPIs
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    event_log_b, _ = trainer_final.predict()
    predict_s = round(time.perf_counter() - t0, 2)
    print(f"Prediction: {predict_s:.1f}s  ({event_log_b['caseid'].nunique():,} cases)")

    # Strip timezone: KPI functions expect timezone-naive timestamps.
    event_log_b = event_log_b.copy()
    event_log_b["end_timestamp"] = (
        pd.to_datetime(event_log_b["end_timestamp"], utc=True).dt.tz_convert(None))

    cc_pred, tt_pred = _kpi_series(event_log_b, cc["test"].index, tt["test"].index)
    cc_actual = cc["test"].to_numpy()
    tt_actual = tt["test"].to_numpy()

    results = {
        "concurrent_cases": {
            "mse": mean_squared_error(cc_actual, cc_pred),
            "mae": mean_absolute_error(cc_actual, cc_pred),
        },
        "throughput_time": {
            "mse": mean_squared_error(tt_actual, tt_pred),
            "mae": mean_absolute_error(tt_actual, tt_pred),
        },
    }

    # ------------------------------------------------------------------
    # Step 4: Save metrics, timing, and plots
    # ------------------------------------------------------------------
    out_dir = RESULTS_DIR / "bukhsh_hpo" / trim_name / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([
        dict(dataset=run_name, series=name, model="bukhsh_hpo", mse=m["mse"], mae=m["mae"])
        for name, m in results.items()
    ]).to_csv(out_dir / f"metrics_{run_name}.csv", index=False)

    pd.DataFrame([
        dict(model="bukhsh_hpo", phase="predict", params_json=json.dumps(best_params),
             val_mse=float("nan"), time_s=predict_s, is_best=True, dataset=run_name, series="all"),
    ]).to_csv(out_dir / f"time_{run_name}.csv", index=False)

    _save_plots(cc_actual, cc_pred, tt_actual, tt_pred,
                cc["test"].index, tt["test"].index, "bukhsh_hpo", run_name, out_dir)

    print(f"Saved → {out_dir}")
    for name, m in results.items():
        print(f"  {name:<22}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}")

    return results


# =============================================================================
# Camargo HPO
# =============================================================================

def run_camargo(log_path: str | Path, trim_cfg: dict) -> dict:
    """Run the full Camargo experiment for one log and one trim configuration.

    Steps
    -----
    1. Load data and compute splits via _load_data().
    2. Initialise CamargoTrainer and preprocess (role clustering, encoding).
    3. Run Bayesian HPO internally via trainer_c.train().
       Camargo's trainer runs its own hyperopt loop (CAMARGO_MAX_EVAL trials).
    4. Save an hpo_summary.json so the model can be reloaded later.
    5. Predict event suffixes for all test cases (autoregressive sampling).
    6. Compute CC and TT MSE / MAE against ground-truth test series.
    7. Save metrics, timing, and plots.

    Checkpointing
    -------------
    CamargoTrainer.__init__ automatically detects an existing trained model
    in the output directory and sets best_model_path. If it is set when
    run_camargo() is called, the training step is skipped and the existing
    model is used directly for prediction.

    Note on model directories
    -------------------------
    Camargo stores its model weights inside GenerativeLSTM/GenerativeLSTM/
    output_files/{trim_dir}/{run_name}/, which is managed internally by
    CamargoTrainer. The hpo_summary.json saved to best_models/ records the
    run_name and trim_dir so the model can be reloaded in a separate script.
    A unique run_name is constructed per (log, trim_config) pair to prevent
    different configurations from overwriting each other's weights.

    Parameters
    ----------
    log_path:
        Path to the XES file or log stem name (resolved via _resolve_log).
    trim_cfg:
        One entry from TRIM_CONFIGS.

    Returns
    -------
    dict with keys ``'concurrent_cases'`` and ``'throughput_time'``, each
    mapping to ``{'mse': float, 'mae': float}`` on the test period.
    """
    log_path  = _resolve_log(log_path)
    dataset   = log_path.stem
    trim_name = _trim_dir(trim_cfg["method"], trim_cfg["pct"], trim_cfg["k"], trim_cfg["frac"])
    run_name  = f"{dataset}_test_full"

    print(f"\n{'='*60}")
    print(f"Camargo HPO | {dataset} | {trim_name}")
    print(f"{'='*60}")

    cc, tt, train_df, val_df, test_df, val_as_test_df, _ = _load_data(log_path, trim_cfg)

    # Build a unique run_name per (log, trim_config) so that multiple configs
    # can coexist in Camargo's output_files/ directory without collisions.
    # E.g.: "helpdesk_peak_0.6_hpo"
    camargo_run  = f"{dataset}_{trim_name}_hpo"
    camargo_trim = trim_name   # used as the subfolder inside output_files/

    best_dir = BEST_MODELS / dataset / trim_name / "camargo"
    best_dir.mkdir(parents=True, exist_ok=True)

    # Initialise trainer. If a trained model already exists in output_files/,
    # CamargoTrainer.__init__ loads it and sets best_model_path automatically.
    # val_as_test_df/cc["val"]/tt["val"] let train() select the HPO winner by
    # real validation-period CC MAE (same protocol bukhsh's own HPO already
    # uses) instead of camargo's internal vectorized next-event loss -- see
    # camargo/trainer.py::train()'s winner-selection block.
    params_c  = camargo_params(camargo_run, max_eval=CAMARGO_MAX_EVAL, epochs=200)
    trainer_c = CamargoTrainer(
        train_df, val_df, test_df,
        camargo_run, params_c, trim_dir=camargo_trim,
        val_as_test_df=val_as_test_df, cc_val=cc["val"], tt_val=tt["val"],
    )

    # Preprocess: mine resource roles, build activity / role index, compute
    # embeddings. Must be called even when loading a pre-trained model.
    trainer_c.preprocess()
    print(f"Activities: {len(trainer_c.ac_index)}  Roles: {len(trainer_c.rl_index)}")

    # ------------------------------------------------------------------
    # Step 1: HPO (skipped if a fully-trained model file exists on disk)
    # ------------------------------------------------------------------
    # best_model_path is an absolute path set by load_trained_model() in
    # __init__.  We also verify the .h5 file is actually on disk — it could
    # be missing if a previous run was interrupted after writing
    # model_parameters.json but before the model file was fully flushed.
    _model_on_disk = (
        trainer_c.best_model_path is not None
        and Path(trainer_c.best_model_path).exists()
    )

    if _model_on_disk:
        print(f"Model exists, skipping training (loss={trainer_c.best_loss:.4f})")
        camargo_train_s = 0.0
    else:
        if trainer_c.best_model_path is not None:
            # Path was set but the file is gone — warn and retrain cleanly.
            print(f"  model path set but file missing "
                  f"({trainer_c.best_model_path}) — will retrain")
            trainer_c.best_model_path = None
        # train() internally runs Bayesian HPO (hyperopt TPE) over model_type,
        # lstm_act, learning_rate, and dropout. The best model is saved to disk.
        t0 = time.perf_counter()
        trainer_c.train()
        camargo_train_s = round(time.perf_counter() - t0, 2)
        print(f"Best loss  : {trainer_c.best_loss:.4f}  Train time: {camargo_train_s:.1f}s")

    # Save a summary so the model can be reloaded without retraining, e.g.:
    #   summary = json.loads((best_dir / "hpo_summary.json").read_text())
    #   trainer = CamargoTrainer.from_saved(train_df, val_df, test_df,
    #                 summary["run_name"], params, trim_dir=summary["trim_dir"])
    (best_dir / "hpo_summary.json").write_text(json.dumps({
        "run_name":     camargo_run,
        "trim_dir":     camargo_trim,
        "best_loss":    trainer_c.best_loss,
        "train_time_s": camargo_train_s,
        "output_dir":   trainer_c.output_dir,  # absolute path for reference
    }, indent=2))

    # ------------------------------------------------------------------
    # Step 2: Predict event suffixes for test cases
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    # predict() samples complete case suffixes autoregressively.
    # Returns a list of paths to the raw gen_*.csv prediction files.
    pred_paths  = trainer_c.predict()
    # to_event_log() converts the raw predictions into the standard
    # event-log format (caseid, task, user, end_timestamp).
    event_log_c = trainer_c.to_event_log(pred_paths)
    predict_s   = round(time.perf_counter() - t0, 2)
    print(f"Prediction: {predict_s:.1f}s  ({event_log_c['caseid'].nunique():,} cases)")

    # ------------------------------------------------------------------
    # Step 3: Evaluate KPIs
    # ------------------------------------------------------------------
    cc_pred, tt_pred = _kpi_series(event_log_c, cc["test"].index, tt["test"].index)
    cc_actual = cc["test"].to_numpy()
    tt_actual = tt["test"].to_numpy()

    results = {
        "concurrent_cases": {
            "mse": mean_squared_error(cc_actual, cc_pred),
            "mae": mean_absolute_error(cc_actual, cc_pred),
        },
        "throughput_time": {
            "mse": mean_squared_error(tt_actual, tt_pred),
            "mae": mean_absolute_error(tt_actual, tt_pred),
        },
    }

    # ------------------------------------------------------------------
    # Step 4: Save metrics, timing, and plots
    # ------------------------------------------------------------------
    out_dir = RESULTS_DIR / "camargo_hpo" / trim_name / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([
        dict(dataset=run_name, series=name, model="camargo_hpo", mse=m["mse"], mae=m["mae"])
        for name, m in results.items()
    ]).to_csv(out_dir / f"metrics_{run_name}.csv", index=False)

    pd.DataFrame([
        dict(model="camargo_hpo", phase="predict", params_json="{}",
             val_mse=float("nan"), time_s=predict_s, is_best=True, dataset=run_name, series="all"),
    ]).to_csv(out_dir / f"time_{run_name}.csv", index=False)

    _save_plots(cc_actual, cc_pred, tt_actual, tt_pred,
                cc["test"].index, tt["test"].index, "camargo_hpo", run_name, out_dir)

    print(f"Saved → {out_dir}")
    for name, m in results.items():
        print(f"  {name:<22}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}")

    return results


MODEL_RUNNERS = {
    "bukhsh": run_bukhsh,
    "camargo": run_camargo,
}


def _resolve_models(models: str | list[str] | None = None) -> list[str]:
    """Return selected model names. None means run all models."""
    if models is None:
        return list(MODEL_RUNNERS)

    if isinstance(models, str):
        models = [models]

    invalid = sorted(set(models) - set(MODEL_RUNNERS))
    if invalid:
        available = ", ".join(MODEL_RUNNERS)
        raise ValueError(f"Unknown model(s): {invalid}. Available: {available}")

    return list(models)


# =============================================================================
# Combined runners
# =============================================================================

# def run_config(log_path: str | Path, trim_cfg: dict) -> dict:
#     """Run both Bukhsh and Camargo for one (log, trim_config) pair.

#     Errors in one model are caught and printed without aborting the other,
#     so a Camargo failure does not prevent Bukhsh results from being saved
#     and vice versa.

#     Parameters
#     ----------
#     log_path:
#         Path or stem name of the XES log file.
#     trim_cfg:
#         One entry from TRIM_CONFIGS.

#     Returns
#     -------
#     dict with keys ``'bukhsh'`` and ``'camargo'``, each mapping to the
#     metrics dict returned by the respective run function (or absent if
#     that model errored).
#     """
#     log_path  = _resolve_log(log_path)
#     trim_name = _trim_dir(trim_cfg["method"], trim_cfg["pct"], trim_cfg["k"], trim_cfg["frac"])
#     results   = {}

#     for model_name, fn in [("bukhsh", run_bukhsh), ("camargo", run_camargo)]:
#         try:
#             results[model_name] = fn(log_path, trim_cfg)
#         except Exception:
#             print(f"\n[ERROR] {model_name} failed for {log_path.stem} / {trim_name}:")
#             traceback.print_exc()

#     return results

def run_config(
    log_path: str | Path,
    trim_cfg: dict,
    models: str | list[str] | None = None,
) -> dict:
    """Run selected model(s) for one (log, trim_config) pair.

    models:
        None = run all models.
        "bukhsh" = only Bukhsh.
        "camargo" = only Camargo.
    """
    log_path  = _resolve_log(log_path)
    trim_name = _trim_dir(trim_cfg["method"], trim_cfg["pct"], trim_cfg["k"], trim_cfg["frac"])
    results   = {}

    selected_models = _resolve_models(models)

    for model_name in selected_models:
        fn = MODEL_RUNNERS[model_name]
        try:
            results[model_name] = fn(log_path, trim_cfg)
        except Exception:
            print(f"\n[ERROR] {model_name} failed for {log_path.stem} / {trim_name}:")
            traceback.print_exc()

    return results


# def run_log(
#     log: str | Path,
#     trim_configs: list[dict] | None = None,
# ) -> dict:
#     """Run both models across all trim configurations for a single log.

#     This is the main entry point for running a complete set of experiments
#     on one dataset. All seven trim configurations are run in sequence; errors
#     in one configuration do not abort the rest.

#     Example
#     -------
#     >>> from run_experiments_real import run_log, TRIM_CONFIGS
#     >>> # All 7 trim configs:
#     >>> results = run_log("helpdesk")
#     >>> # Only peak configs:
#     >>> peak_cfgs = [c for c in TRIM_CONFIGS if c["method"] == "peak"]
#     >>> results = run_log("helpdesk", peak_cfgs)
#     >>> # Access a specific result:
#     >>> results["peak_0.6"]["bukhsh"]["concurrent_cases"]["mae"]

#     Parameters
#     ----------
#     log:
#         Log stem name (e.g. ``'helpdesk'``, ``'bpic12-a'``) or a full or
#         relative path to an XES file. The stem is looked up in DATA_DIR if
#         a plain name is given.
#     trim_configs:
#         List of trim config dicts to run. Each must be a dict with keys
#         ``method``, ``frac``, ``k``, ``pct``. Defaults to all 7 entries
#         in TRIM_CONFIGS.

#     Returns
#     -------
#     Nested dict: ``{trim_name: {"bukhsh": metrics, "camargo": metrics}}``.
#     """
#     if trim_configs is None:
#         trim_configs = TRIM_CONFIGS

#     log_path = _resolve_log(log)
#     print(f"\n{'#'*60}")
#     print(f"# Log: {log_path.stem}  ({len(trim_configs)} trim config(s))")
#     print(f"{'#'*60}")

#     all_results: dict[str, dict] = {}
#     for cfg in trim_configs:
#         trim_name = _trim_dir(cfg["method"], cfg["pct"], cfg["k"], cfg["frac"])
#         all_results[trim_name] = run_config(log_path, cfg)

#     return all_results

def run_log(
    log: str | Path,
    trim_configs: list[dict] | None = None,
    models: str | list[str] | None = None,
) -> dict:
    if trim_configs is None:
        trim_configs = TRIM_CONFIGS

    selected_models = _resolve_models(models)

    log_path = _resolve_log(log)
    print(f"\n{'#'*60}")
    print(f"# Log: {log_path.stem}  ({len(trim_configs)} trim config(s), {len(selected_models)} model(s))")
    print(f"{'#'*60}")

    all_results: dict[str, dict] = {}
    for cfg in trim_configs:
        trim_name = _trim_dir(cfg["method"], cfg["pct"], cfg["k"], cfg["frac"])
        all_results[trim_name] = run_config(log_path, cfg, selected_models)

    return all_results


# def run_all_logs(
#     trim_configs: list[dict] | None = None,
#     data_dir: Path | None = None,
# ) -> dict:
#     """Run both models across all trim configs for every XES log in data/real-life/.

#     This is the top-level entry point for a full experiment run. Logs are
#     processed in alphabetical order. Errors within a single (log, config,
#     model) combination are printed and skipped without aborting the rest.

#     Example
#     -------
#     >>> from run_experiments_real import run_all_logs, TRIM_CONFIGS
#     >>> # Full experiment (all logs × all configs × both models):
#     >>> results = run_all_logs()
#     >>> # Only the "none" config across all logs (fastest, no trimming):
#     >>> results = run_all_logs([TRIM_CONFIGS[0]])

#     Parameters
#     ----------
#     trim_configs:
#         List of trim config dicts to run. Defaults to all 7 entries in
#         TRIM_CONFIGS. Pass a subset to run fewer configurations.
#     data_dir:
#         Directory containing ``*.xes`` files. Defaults to ``data/real-life/``.
#         Override to point at a different set of logs.

#     Returns
#     -------
#     Nested dict:
#     ``{log_stem: {trim_name: {"bukhsh": metrics, "camargo": metrics}}}``.
#     """
#     if trim_configs is None:
#         trim_configs = TRIM_CONFIGS
#     if data_dir is None:
#         data_dir = DATA_DIR

#     xes_files = sorted(Path(data_dir).glob("*.xes"))
#     print(f"Found {len(xes_files)} log(s) in {data_dir}")
#     print(f"Trim configs : {[_trim_dir(c['method'],c['pct'],c['k'],c['frac']) for c in trim_configs]}")
#     print(f"Total runs   : {len(xes_files)} logs × {len(trim_configs)} configs × 2 models "
#           f"= {len(xes_files) * len(trim_configs) * 2}")

#     all_results: dict[str, dict] = {}
#     for log_path in xes_files:
#         all_results[log_path.stem] = run_log(log_path, trim_configs)

#     return all_results

def run_all_logs(
    trim_configs: list[dict] | None = None,
    data_dir: Path | None = None,
    models: str | list[str] | None = None,
) -> dict:
    if trim_configs is None:
        trim_configs = TRIM_CONFIGS
    if data_dir is None:
        data_dir = DATA_DIR

    selected_models = _resolve_models(models)

    xes_files = sorted(Path(data_dir).glob("*.xes"))
    print(f"Found {len(xes_files)} log(s) in {data_dir}")
    print(f"Trim configs : {[_trim_dir(c['method'],c['pct'],c['k'],c['frac']) for c in trim_configs]}")
    print(f"Models       : {selected_models}")
    print(f"Total runs   : {len(xes_files)} logs × {len(trim_configs)} configs × {len(selected_models)} model(s) "
          f"= {len(xes_files) * len(trim_configs) * len(selected_models)}")

    all_results: dict[str, dict] = {}
    for log_path in xes_files:
        all_results[log_path.stem] = run_log(log_path, trim_configs, selected_models)

    return all_results


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the CLI entry point."""
    available_trims = [
        _trim_dir(c["method"], c["pct"], c["k"], c["frac"]) for c in TRIM_CONFIGS
    ]
    parser = argparse.ArgumentParser(
        description="Run Bukhsh + Camargo HPO on real-life event logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
examples:
  # Run all logs × all 7 trim configs (full experiment):
  python run_experiments_real.py

  # Run one log × all trim configs:
  python run_experiments_real.py --log helpdesk

  # Run one log × one specific trim config:
  python run_experiments_real.py --log helpdesk --trim peak_0.6

  # Run one log × one config with a full path:
  python run_experiments_real.py --log data/real-life/bpic12-a.xes --trim none

available trim configs:
  {", ".join(available_trims)}

available logs (stems in data/real-life/):
  bpic12-a  bpic15-1  bpic15-2  bpic17-o
  bpic20-dom  bpic20-int  helpdesk  sepsis
        """,
    )
    parser.add_argument(
        "--log", type=str, default=None,
        metavar="LOG",
        help=(
            "Log stem name (e.g. 'helpdesk') or path to an XES file. "
            "Omit to run all logs in data/real-life/."
        ),
    )
    parser.add_argument(
        "--trim", type=str, default=None,
        metavar="TRIM",
        help=(
            "Trim configuration name (e.g. 'peak_0.6'). "
            "Omit to run all 7 configurations. "
            f"Available: {', '.join(available_trims)}"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["bukhsh", "camargo"],
        metavar="MODEL",
        help=(
            "Model to run. "
            "Omit to run both models. "
            "Available: bukhsh, camargo"
        ),
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    # Resolve trim config(s) from the CLI argument.
    trim_cfgs: list[dict]
    if args.trim is not None:
        trim_cfgs = [_trim_cfg_by_name(args.trim)]  # raises ValueError if unknown
    else:
        trim_cfgs = TRIM_CONFIGS

    # # Dispatch to the appropriate runner.
    # if args.log is not None:
    #     run_log(args.log, trim_cfgs)
    # else:
    #     run_all_logs(trim_cfgs)
    if args.log is not None:
        run_log(args.log, trim_cfgs, models=args.model)
    else:
        run_all_logs(trim_cfgs, models=args.model)
