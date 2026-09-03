"""
End-to-end PMSD baseline: fit on train+val, simulate the test horizon, score.

Reuses this project's existing infrastructure rather than reimplementing it:
  - time_series_preprocessing.ts_splits_from_log for the canonical
    train/val/test split (same boundaries every other baseline uses).
Everything specific to PMSD lives in sdlog.py / relations.py / equations.py /
simulate.py — see README "Design decisions" for what changed vs. the paper
and why.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PF_DIR = ROOT / "plain-field"
if str(PF_DIR) not in sys.path:
    sys.path.insert(0, str(PF_DIR))

from time_series_preprocessing import ts_splits_from_log  # noqa: E402
from arrival import ProphetArrivalModel  # noqa: E402

from .sdlog import build_sd_log  # noqa: E402
from .relations import discover_relations  # noqa: E402
from .equations import fit_best_equation_pair  # noqa: E402
from .simulate import simulate_test_horizon  # noqa: E402


SIMULATE_COLS = ["concurrent_cases", "arrival_rate", "finish_rate", "throughput_time"]
ALL_SD_LOG_COLS = SIMULATE_COLS + [
    "service_time", "waiting_time", "num_unique_resources",
    "num_unique_activities", "process_active_time",
]

RESULTS_ROOT = ROOT / "results" / "ts_predictions" / "pmsd"


_TRIM_KW = {
    "none":        dict(trim_method=None),
    "peak_0.6":    dict(trim_method="peak",      trim_frac=0.6),
    "peak_0.7":    dict(trim_method="peak",      trim_frac=0.7),
    "peak_0.8":    dict(trim_method="peak",      trim_frac=0.8),
    "magnitude_1": dict(trim_method="magnitude", trim_k=1.0),
    "magnitude_2": dict(trim_method="magnitude", trim_k=2.0),
    "magnitude_3": dict(trim_method="magnitude", trim_k=3.0),
}
_BASE_TRIM_KW = dict(trim_pct=0.25, trim_k=1.5, trim_frac=0.6, trim_window=7)


def _internal_train_val_split(sd_log: pd.DataFrame, val_frac: float = 0.15):
    n = len(sd_log)
    cut = max(1, int(n * (1 - val_frac)))
    return sd_log.iloc[:cut], sd_log.iloc[cut:]


def run_pmsd(
    dataset: str,
    log: pd.DataFrame,
    trim: str = "none",
    is_real: bool = False,
    time_col: str = "time:timestamp",
    start_col: str = "start:timestamp",
    case_col: str = "case:concept:name",
    max_shift: int = 7,
    corr_threshold: float = 0.3,
    save: bool = True,
    precomputed_splits: dict | None = None,
) -> dict:
    """
    Run the full PMSD baseline on one event log.

    """
    if precomputed_splits is not None:
        splits = precomputed_splits
    else:
        trim_kw = _TRIM_KW.get(trim, dict(trim_method=None)) if is_real else dict(trim_method=None)
        splits = ts_splits_from_log(log, train_frac=0.70, val_frac=0.10, **{**_BASE_TRIM_KW, **trim_kw})
    cc_split = splits["concurrent_cases"]
    tt_split = splits["throughput_time"]
    val_split_ts = cc_split["val_split"]


    sd_log = build_sd_log(
        log, time_col=time_col, start_col=start_col, case_col=case_col, cut_date=val_split_ts,
    )


    relations = discover_relations(
        sd_log, targets=SIMULATE_COLS, predictor_pool=ALL_SD_LOG_COLS,
        max_shift=max_shift, threshold=corr_threshold,
    )


    sim_relations = discover_relations(
        sd_log, targets=["finish_rate", "throughput_time"], predictor_pool=SIMULATE_COLS,
        max_shift=max_shift, threshold=corr_threshold,
    )

    eq_train, eq_val = _internal_train_val_split(sd_log)
    finish_rate_eq, throughput_time_eq = fit_best_equation_pair(
        eq_train, eq_val, sim_relations["finish_rate"], sim_relations["throughput_time"],
    )


    arrival_model = ProphetArrivalModel(yearly_seasonality=False).fit(sd_log["arrival_rate"])
    test_index = cc_split["test"].index
    arrival_forecast = arrival_model.predict(pd.DatetimeIndex(test_index).tz_localize(None)
                                             if pd.DatetimeIndex(test_index).tz is not None
                                             else pd.DatetimeIndex(test_index))
    arrival_forecast.index = test_index  # keep test index's own tz/dtype for downstream alignment

    simulated = simulate_test_horizon(sd_log, arrival_forecast, finish_rate_eq, throughput_time_eq)

    cc_actual = cc_split["test"].reindex(simulated.index).to_numpy()
    tt_actual = tt_split["test"].reindex(simulated.index).to_numpy()
    cc_pred = simulated["concurrent_cases"].to_numpy()
    tt_pred = simulated["throughput_time"].to_numpy()

    metrics = dict(
        cc_mae=round(mean_absolute_error(cc_actual, cc_pred), 6),
        cc_mse=round(mean_squared_error(cc_actual, cc_pred), 6),
        tt_mae=round(mean_absolute_error(tt_actual, tt_pred), 6),
        tt_mse=round(mean_squared_error(tt_actual, tt_pred), 6),
    )

    if save:
        sub = "synthetic/none" if not is_real else trim
        _save_predictions(dataset, sub, simulated)
        _save_metrics(dataset, trim if is_real else "none", is_real, metrics)

    return dict(
        sd_log=sd_log,
        relations=relations,
        sim_relations=sim_relations,
        finish_rate_eq=finish_rate_eq,
        throughput_time_eq=throughput_time_eq,
        simulated=simulated,
        metrics=metrics,
        cc_test=cc_split["test"],
        tt_test=tt_split["test"],
    )


def run_pmsd_synthetic(dataset: str, log: pd.DataFrame, **kwargs) -> dict:
    """Back-compat thin wrapper: synthetic datasets always use trim='none'."""
    return run_pmsd(dataset, log, trim="none", is_real=False, **kwargs)


def _save_predictions(dataset: str, sub: str, simulated: pd.DataFrame) -> None:
    out_dir = RESULTS_ROOT / sub / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for series_name, col in [("concurrent_cases", "concurrent_cases"), ("throughput_time", "throughput_time")]:
        for date, val in simulated[col].items():
            rows.append({"series": series_name, "date": str(pd.Timestamp(date).date()), "predicted": float(val)})
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)


def _save_metrics(dataset: str, trim_dir: str, is_real: bool, metrics: dict) -> None:

    out_dir = (ROOT / "results" / trim_dir / dataset) if is_real else (ROOT / "results" / "synthetic" / "none" / dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"metrics_{dataset}_pmsd.csv"
    new_rows = pd.DataFrame([
        dict(dataset=dataset, series="concurrent_cases", model="pmsd", mae=metrics["cc_mae"], mse=metrics["cc_mse"]),
        dict(dataset=dataset, series="throughput_time", model="pmsd", mae=metrics["tt_mae"], mse=metrics["tt_mse"]),
    ])
    new_rows.to_csv(path, index=False)
