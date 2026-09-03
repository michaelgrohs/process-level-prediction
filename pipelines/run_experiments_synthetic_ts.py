"""
run_experiments_synthetic_ts.py
================================
Run time series forecasting models on synthetic event logs.

Usage (run from the repo root)
-------------------------------
  # All datasets, both new models (default):
  python simulation_baselines/run_experiments_synthetic_ts.py

  # Single dataset:
  python simulation_baselines/run_experiments_synthetic_ts.py --dataset loan_combined

  # Specific models only:
  python simulation_baselines/run_experiments_synthetic_ts.py --models tabpfn chronos

  # All models in the default set:
  python simulation_baselines/run_experiments_synthetic_ts.py --models all

  # Combine flags:
  python simulation_baselines/run_experiments_synthetic_ts.py --dataset o2c_flat --models chronos

  # Use large hyperparameter grids:
  python simulation_baselines/run_experiments_synthetic_ts.py --tuning large

  # Custom train/val/test split (val is always 0.1):
  python simulation_baselines/run_experiments_synthetic_ts.py --split 0.2 0.1 0.7
  python simulation_baselines/run_experiments_synthetic_ts.py --split 0.3 0.1 0.6

Available datasets (stems in data/synthetic/):
  loan_flat  loan_seasonal  loan_trend  loan_recency  loan_drift  loan_combined
  o2c_flat   o2c_seasonal   o2c_trend   o2c_recency   o2c_drift   o2c_combined
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent  # pipelines/ -> repo root (moved 2026-09-03, was simulation_baselines/, same depth)
sys.path.insert(0, str(ROOT))

import pandas as pd
import pm4py
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm.auto import tqdm

from setttings import set_global_seed
from time_series_creation import (
    create_avg_throughtput_time_timeseries,
    create_concurrent_cases_timeseries,
)
from time_series_prediction import run_pipeline
from time_series_preprocessing import Split3WayConfig, apply_trim, split_by_dates, split_timeseries

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR    = ROOT / "data" / "synthetic"
RESULTS_DIR = ROOT / "results" / "synthetic"
CUT_DATE    = "2023-07-01"

# ─────────────────────────────────────────────────────────────────────────────
# Default models (the two new foundation models)
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
# Helpers (mirror pipeline_synthetic.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def _trim_dir_name(trim_method, trim_pct, trim_k, trim_frac) -> str:
    if trim_method is None:
        return "none"
    if trim_method == "pct":
        return f"pct_{trim_pct:g}"
    if trim_method == "magnitude":
        return f"magnitude_{trim_k:g}"
    if trim_method == "peak":
        return f"peak_{trim_frac:g}"
    return str(trim_method)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation (mirrors evaluate_single_dataset from the notebook)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(
    xes_path: Path,
    results_dir: Path,
    models: list[str],
    tuning: str = "small",
    trim_method: str | None = None,
    trim_pct: float = 0.25,
    trim_k: float = 1.5,
    trim_frac: float = 0.70,
    trim_window: int = 7,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    cut_date: str | None = CUT_DATE,
) -> pd.DataFrame:
    """Evaluate *models* on a single XES dataset and save results."""
    dataset_name = xes_path.stem
    out_dir      = results_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg     = Split3WayConfig(train_frac=train_frac, val_frac=val_frac, test_frac=test_frac)
    trim_kw = dict(pct=trim_pct, k=trim_k, frac=trim_frac, window=trim_window)

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Dataset: {dataset_name}  |  models: {models}  |  tuning: {tuning}")
    tqdm.write(f"{'='*60}")

    try:
        log = pm4py.read_xes(str(xes_path))
    except Exception:
        tqdm.write(f"[FAIL] could not load {xes_path}:\n{traceback.format_exc()}")
        return pd.DataFrame()

    # Canonical split dates from CC series (TT is capped at the same end)
    _cc_raw     = create_concurrent_cases_timeseries(log, cut_date=cut_date, plot=False)
    _cc_trimmed = apply_trim(_cc_raw, trim_method, **trim_kw)
    canonical_end                              = _cc_trimmed.index[-1]
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

    all_rows:    list[dict]         = []
    all_timing:  list[pd.DataFrame] = []
    all_pred_rows: list[dict]       = []

    for series_name, series_fn in series_map.items():
        tqdm.write(f"\n--- {series_name} ---")
        try:
            raw     = series_fn(log, cut_date=cut_date, plot=False)
            trimmed = raw[raw.index <= canonical_end]
            train, val, test = split_by_dates(trimmed, canonical_train_split, canonical_val_split)
            tqdm.write(
                f"trimmed {len(raw)}→{len(trimmed)}, "
                f"train={len(train)}  val={len(val)}  test={len(test)}"
            )

            preds, timing_df = run_pipeline(
                train, val, test,
                label=f"{dataset_name}/{series_name}",
                models=models,
                tuning=tuning,
            )
            timing_df["dataset"] = dataset_name
            timing_df["series"]  = series_name
            all_timing.append(timing_df)

            y_test = test.to_numpy()
            test_dates = test.index
            for model_name, yhat in preds.items():
                try:
                    mse = mean_squared_error(y_test, yhat)
                    mae = mean_absolute_error(y_test, yhat)
                except Exception:
                    tqdm.write(f"[METRIC FAIL] {series_name}/{model_name}:\n{traceback.format_exc()}")
                    mse, mae = float("nan"), float("nan")
                all_rows.append(dict(
                    dataset=dataset_name, series=series_name,
                    model=model_name, mse=mse, mae=mae,
                ))
                for date, actual, predicted in zip(test_dates, y_test, yhat):
                    all_pred_rows.append(dict(
                        dataset=dataset_name, series=series_name,
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default=None, metavar="NAME",
        help=(
            f"Dataset stem to run (e.g. loan_combined). "
            f"Omit to run all. Available: {', '.join(available_datasets)}"
        ),
    )
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="MODEL",
        help=(
            f"Models to run. Use 'all' for all default models. "
            f"Default: {DEFAULT_MODELS}. "
            f"Available: {ALL_MODELS}"
        ),
    )
    parser.add_argument(
        "--tuning", choices=["small", "large"], default="small",
        help="Hyperparameter grid size (default: small).",
    )
    parser.add_argument(
        "--split", nargs=3, type=float, default=[0.7, 0.1, 0.2],
        metavar=("TRAIN", "VAL", "TEST"),
        help=(
            "Train/val/test fractions (must sum to 1.0). "
            "Default: 0.7 0.1 0.2. "
            "Example: --split 0.2 0.1 0.7"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Validate and unpack split fractions
    train_frac, val_frac, test_frac = args.split
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        print(
            f"--split fractions must sum to 1.0; got {train_frac}+{val_frac}+{test_frac}"
            f"={train_frac+val_frac+test_frac:.4f}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve models list
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
    print("models passed")

    # Resolve dataset(s)
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
        print(f"xes files: {xes_files}")
    else:
        xes_files = sorted(DATA_DIR.glob("*.xes"))
        if not xes_files:
            print(f"No .xes files found in {DATA_DIR}", file=sys.stderr)
            sys.exit(1)

    # Results folder encodes trim method (none) and split fractions
    split_tag = f"split_{int(train_frac*100):02d}_{int(val_frac*100):02d}_{int(test_frac*100):02d}"
    results_dir = RESULTS_DIR / _trim_dir_name(None, 0.25, 1.5, 0.70) / split_tag
    results_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(1904)

    print(f"Datasets : {[f.stem for f in xes_files]}")
    print(f"Models   : {models}")
    print(f"Tuning   : {args.tuning}")
    print(f"Split    : train={train_frac}  val={val_frac}  test={test_frac}")
    print(f"Results  : {results_dir}\n")

    all_results: list[pd.DataFrame] = []
    for xes_path in xes_files:
        df = evaluate_dataset(
            xes_path    = xes_path,
            results_dir = results_dir,
            models      = models,
            tuning      = args.tuning,
            train_frac  = train_frac,
            val_frac    = val_frac,
            test_frac   = test_frac,
        )
        all_results.append(df)

    # Save combined results
    non_empty = [r for r in all_results if not r.empty]
    if not non_empty:
        return
    combined = pd.concat(non_empty, ignore_index=True)
    if not combined.empty:
        combined_path = results_dir / "metrics_all.csv"
        if combined_path.exists():
            existing_all = pd.read_csv(combined_path)
            existing_all = existing_all[~existing_all["model"].isin(models)]
            combined = pd.concat([existing_all, combined], ignore_index=True)
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined metrics saved → {combined_path}")


if __name__ == "__main__":
    print("Start script")
    main()
