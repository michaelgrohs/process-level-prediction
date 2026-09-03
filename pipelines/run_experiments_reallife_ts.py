"""
run_experiments_reallife_ts.py
================================
Run time series forecasting models on real-life event logs.



Trim strategies (mirrors run_experiments_real.py TRIM_CONFIGS):
  none          No trimming
  peak_0.6      Cut when rolling mean drops below 60% of peak
  peak_0.7      Cut when rolling mean drops below 70% of peak
  peak_0.8      Cut when rolling mean drops below 80% of peak
  magnitude_1   Cut when series drops >1 std below expanding mean
  magnitude_2   Cut when series drops >2 std below expanding mean
  magnitude_3   Cut when series drops >3 std below expanding mean
  ssd           Steady-state detection (rolling-window drift detector)

Usage (run from the repo root)
-------------------------------
  # All datasets, all trims, both new models (default):
  python simulation_baselines/run_experiments_reallife_ts.py

  # Single dataset:
  python simulation_baselines/run_experiments_reallife_ts.py --dataset helpdesk

  # Single dataset + single trim:
  python simulation_baselines/run_experiments_reallife_ts.py --dataset helpdesk --trim peak_0.7

  # Specific models only:
  python simulation_baselines/run_experiments_reallife_ts.py --models tabpfn chronos

  # All models in the default set:
  python simulation_baselines/run_experiments_reallife_ts.py --models all

  # Use large hyperparameter grids:
  python simulation_baselines/run_experiments_reallife_ts.py --tuning large

Available datasets (stems in data/real-life/):
  helpdesk  sepsis  bpic12-a  bpic15-1  bpic15-2
  bpic17-o  bpic20-dom  bpic20-int

Available trim configs:
  none  peak_0.6  peak_0.7  peak_0.8
  magnitude_1  magnitude_2  magnitude_3  ssd
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent  # pipelines/ -> repo root (moved 2026-09-03, was simulation_baselines/, same depth)
SSD_DIR = ROOT / "steady_state_detection"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SSD_DIR))

import pandas as pd
import pm4py
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm.auto import tqdm

from setttings import set_global_seed
from ssd_trim import run_ssd_trim
from time_series_creation import (
    create_avg_throughtput_time_timeseries,
    create_concurrent_cases_timeseries,
)
from time_series_prediction import run_pipeline
from time_series_preprocessing import Split3WayConfig, apply_trim, split_by_dates, split_timeseries

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR    = ROOT / "data" / "real-life"
RESULTS_DIR = ROOT / "results" / "reallife_ts"

# ─────────────────────────────────────────────────────────────────────────────
# Trim configurations (mirrors TRIM_CONFIGS in run_experiments_real.py)
# ─────────────────────────────────────────────────────────────────────────────

TRIM_CONFIGS: list[dict] = [
    {"name": "none",        "method": None,          "frac": 0.70, "k": 1.0, "pct": 0.25},
    {"name": "peak_0.6",    "method": "peak",        "frac": 0.60, "k": 1.5, "pct": 0.25},
    {"name": "peak_0.7",    "method": "peak",        "frac": 0.70, "k": 1.5, "pct": 0.25},
    {"name": "peak_0.8",    "method": "peak",        "frac": 0.80, "k": 1.5, "pct": 0.25},
    {"name": "magnitude_1", "method": "magnitude",   "frac": 0.70, "k": 1.0, "pct": 0.25},
    {"name": "magnitude_2", "method": "magnitude",   "frac": 0.70, "k": 2.0, "pct": 0.25},
    {"name": "magnitude_3", "method": "magnitude",   "frac": 0.70, "k": 3.0, "pct": 0.25},
    {"name": "ssd",         "method": "ssd",         "frac": 0.70, "k": 1.5, "pct": 0.25},
]

TRIM_CONFIG_BY_NAME = {c["name"]: c for c in TRIM_CONFIGS}

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODELS = ["tabpfn", "chronos"]

ALL_MODELS = [
    "naive", "seasonal_naive",
    "ets", "sarimax", "theta", "stl",
    "ridge", "ridge_mimo",
    "gru", "gru_mimo",
    "nbeats", "nhits", "tft",
    "tabpfn", "chronos",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _trim_dir_name(trim_cfg: dict) -> str:
    return trim_cfg["name"]


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation (mirrors evaluate_single_dataset from pipeline_real.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(
    xes_path: Path,
    results_dir: Path,
    trim_cfg: dict,
    models: list[str],
    tuning: str = "small",
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    trim_window: int = 7,
) -> pd.DataFrame:
    """Evaluate *models* on a single real-life XES dataset with one trim config."""
    dataset_name = xes_path.stem
    trim_name    = trim_cfg["name"]
    out_dir      = results_dir / trim_name / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg     = Split3WayConfig(train_frac=train_frac, val_frac=val_frac, test_frac=test_frac)
    trim_kw = dict(
        pct=trim_cfg["pct"], k=trim_cfg["k"],
        frac=trim_cfg["frac"], window=trim_window,
    )

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Dataset: {dataset_name}  |  trim: {trim_name}  |  models: {models}")
    tqdm.write(f"{'='*60}")

    try:
        log = pm4py.read_xes(str(xes_path))
    except Exception:
        tqdm.write(f"[FAIL] could not load {xes_path}:\n{traceback.format_exc()}")
        return pd.DataFrame()

    # Canonical trim end and split dates from CC; TT capped at the same end
    _cc_raw = create_concurrent_cases_timeseries(log, plot=False)
    if trim_cfg["method"] == "ssd":
        ssd_result = run_ssd_trim(log, window_step="D")
        if ssd_result["cutoff"] is not None:
            canonical_end = ssd_result["cutoff"]
        else:
            canonical_end = _cc_raw.index[-1]
        _cc_trimmed = _cc_raw[_cc_raw.index <= canonical_end]
    else:
        _cc_trimmed = apply_trim(_cc_raw, trim_cfg["method"], **trim_kw)
        canonical_end = _cc_trimmed.index[-1]
    _, _, _, canonical_train_split, canonical_val_split = split_timeseries(_cc_trimmed, cfg)
    tqdm.write(
        f"trim_end={canonical_end.date()}  "
        f"train_split={canonical_train_split.date()}  "
        f"val_split={canonical_val_split.date()}"
    )

    series_map = {
        "concurrent_cases": create_concurrent_cases_timeseries,
        "throughput_time":  create_avg_throughtput_time_timeseries,
    }

    all_rows:      list[dict]         = []
    all_timing:    list[pd.DataFrame] = []
    all_pred_rows: list[dict]         = []

    for series_name, series_fn in series_map.items():
        tqdm.write(f"\n--- {series_name} ---")
        try:
            raw     = series_fn(log, plot=False)
            trimmed = raw[raw.index <= canonical_end]
            train, val, test = split_by_dates(trimmed, canonical_train_split, canonical_val_split)
            tqdm.write(
                f"trimmed {len(raw)}→{len(trimmed)}, "
                f"train={len(train)}  val={len(val)}  test={len(test)}"
            )

            preds, timing_df = run_pipeline(
                train, val, test,
                label=f"{dataset_name}/{trim_name}/{series_name}",
                models=models,
                tuning=tuning,
            )
            timing_df["dataset"] = dataset_name
            timing_df["trim"]    = trim_name
            timing_df["series"]  = series_name
            all_timing.append(timing_df)

            y_test     = test.to_numpy()
            test_dates = test.index
            for model_name, yhat in preds.items():
                try:
                    mse = mean_squared_error(y_test, yhat)
                    mae = mean_absolute_error(y_test, yhat)
                except Exception:
                    tqdm.write(f"[METRIC FAIL] {series_name}/{model_name}:\n{traceback.format_exc()}")
                    mse, mae = float("nan"), float("nan")
                all_rows.append(dict(
                    dataset=dataset_name, trim=trim_name, series=series_name,
                    model=model_name, mse=mse, mae=mae,
                ))
                for date, actual, predicted in zip(test_dates, y_test, yhat):
                    all_pred_rows.append(dict(
                        dataset=dataset_name, trim=trim_name, series=series_name,
                        model=model_name, date=date,
                        actual=actual, predicted=predicted,
                    ))

        except Exception:
            tqdm.write(f"[SKIP] {series_name}:\n{traceback.format_exc()}")
            continue

    results_df = pd.DataFrame(all_rows)
    if not results_df.empty:
        metrics_path = out_dir / f"metrics_{dataset_name}.csv"
        if metrics_path.exists():
            existing = pd.read_csv(metrics_path)
            existing = existing[~existing["model"].isin(models)]
            results_df = pd.concat([existing, results_df], ignore_index=True)
        results_df.to_csv(metrics_path, index=False)
        tqdm.write(f"\nSaved → {metrics_path}")
        if all_timing:
            time_path = out_dir / f"time_{dataset_name}.csv"
            timing_df = pd.concat(all_timing, ignore_index=True)
            if time_path.exists():
                existing_t = pd.read_csv(time_path)
                existing_t = existing_t[~existing_t["model"].isin(models)]
                timing_df = pd.concat([existing_t, timing_df], ignore_index=True)
            timing_df.to_csv(time_path, index=False)
            tqdm.write(f"Saved → {time_path}")

    if all_pred_rows:
        preds_df   = pd.DataFrame(all_pred_rows)
        preds_path = out_dir / f"predictions_{dataset_name}.csv"
        if preds_path.exists():
            existing_p = pd.read_csv(preds_path, parse_dates=["date"])
            existing_p = existing_p[~existing_p["model"].isin(models)]
            preds_df = pd.concat([existing_p, preds_df], ignore_index=True)
        preds_df.to_csv(preds_path, index=False)
        tqdm.write(f"Saved → {preds_path}")

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    available_datasets = sorted(p.stem for p in DATA_DIR.glob("*.xes"))
    available_trims    = list(TRIM_CONFIG_BY_NAME.keys())

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default=None, metavar="NAME",
        help=(
            f"Dataset stem to run (e.g. helpdesk). "
            f"Omit to run all. Available: {', '.join(available_datasets)}"
        ),
    )
    parser.add_argument(
        "--trim", type=str, default=None, metavar="TRIM",
        help=(
            f"Trim config to use. Omit to run all 7. "
            f"Available: {', '.join(available_trims)}"
        ),
    )
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="MODEL",
        help=(
            f"Models to run. Use 'all' for all default models. "
            f"Default: {DEFAULT_MODELS}."
        ),
    )
    parser.add_argument(
        "--tuning", choices=["small", "large"], default="small",
        help="Hyperparameter grid size (default: small).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve models
    if args.models is None:
        models = DEFAULT_MODELS
    elif args.models == ["all"]:
        models = ALL_MODELS
    else:
        unknown = [m for m in args.models if m not in ALL_MODELS]
        if unknown:
            print(f"Unknown model(s): {unknown}\nAvailable: {ALL_MODELS}", file=sys.stderr)
            sys.exit(1)
        models = args.models

    # Resolve datasets
    if args.dataset is not None:
        xes_path = DATA_DIR / f"{args.dataset}.xes"
        if not xes_path.exists():
            available = sorted(p.stem for p in DATA_DIR.glob("*.xes"))
            print(
                f"Dataset '{args.dataset}' not found in {DATA_DIR}.\n"
                f"Available: {', '.join(available)}",
                file=sys.stderr,
            )
            sys.exit(1)
        xes_files = [xes_path]
    else:
        xes_files = sorted(DATA_DIR.glob("*.xes"))
        if not xes_files:
            print(f"No .xes files found in {DATA_DIR}", file=sys.stderr)
            sys.exit(1)

    # Resolve trim configs
    if args.trim is not None:
        if args.trim not in TRIM_CONFIG_BY_NAME:
            print(
                f"Unknown trim '{args.trim}'.\n"
                f"Available: {', '.join(TRIM_CONFIG_BY_NAME)}",
                file=sys.stderr,
            )
            sys.exit(1)
        trim_configs = [TRIM_CONFIG_BY_NAME[args.trim]]
    else:
        trim_configs = TRIM_CONFIGS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seed(1904)

    print(f"Datasets : {[f.stem for f in xes_files]}")
    print(f"Trims    : {[c['name'] for c in trim_configs]}")
    print(f"Models   : {models}")
    print(f"Tuning   : {args.tuning}")
    print(f"Results  : {RESULTS_DIR}\n")

    all_results: list[pd.DataFrame] = []

    for xes_path in xes_files:
        for trim_cfg in trim_configs:
            df = evaluate_dataset(
                xes_path    = xes_path,
                results_dir = RESULTS_DIR,
                trim_cfg    = trim_cfg,
                models      = models,
                tuning      = args.tuning,
            )
            all_results.append(df)

    # Save combined results across all datasets and trims
    combined = pd.concat([r for r in all_results if not r.empty], ignore_index=True)
    if not combined.empty:
        combined_path = RESULTS_DIR / "metrics_all.csv"
        if combined_path.exists():
            existing = pd.read_csv(combined_path)
            existing = existing[~existing["model"].isin(models)]
            combined = pd.concat([existing, combined], ignore_index=True)
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined metrics saved → {combined_path}")


if __name__ == "__main__":
    main()
