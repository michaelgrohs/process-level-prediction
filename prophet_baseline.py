"""
Prophet as a direct time-series baseline for CC and TT forecasting.

"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

import sys
sys.path.insert(0, str(ROOT))

from time_series_preprocessing import ts_splits_from_log


# Small grid over Prophet's main trend/seasonality flexibility knobs.
# yearly/weekly seasonality stay on and daily off (fixed) — these are structural
# choices appropriate for all the daily-granularity KPI series here, not really
# "tunable" in the same sense; changepoint/seasonality *prior scale* and mode are
# the standard Prophet HPO surface (see Prophet's own tuning guide).
PROPHET_GRID: list[dict] = [
    {"changepoint_prior_scale": cps, "seasonality_prior_scale": sps, "seasonality_mode": mode}
    for cps in [0.01, 0.05, 0.5]
    for sps in [1.0, 10.0]
    for mode in ["additive", "multiplicative"]
]


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


def _fit_prophet(train_series: pd.Series, target_index: pd.DatetimeIndex,
                 params: dict | None = None) -> pd.Series:
    """Fit Prophet on train_series and predict target_index. Returns predicted Series."""
    from prophet import Prophet

    p = params or {}
    full_idx = pd.date_range(train_series.index.min(), train_series.index.max(), freq="D")
    s = train_series.reindex(full_idx).interpolate("time")

    df = pd.DataFrame({"ds": s.index.tz_localize(None), "y": s.values})
    m = Prophet(
        yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
        changepoint_prior_scale=p.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=p.get("seasonality_prior_scale", 10.0),
        seasonality_mode=p.get("seasonality_mode", "additive"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df)

    future = pd.DataFrame({"ds": pd.DatetimeIndex(target_index).tz_localize(None)})
    forecast = m.predict(future)
    pred = pd.Series(
        forecast["yhat"].values,
        index=pd.DatetimeIndex(forecast["ds"]),
    ).clip(lower=0)
    return pred


def _tune_prophet(train_series: pd.Series, val_series: pd.Series, grid: list[dict]):
    """Grid-search PROPHET_GRID on val_series; returns (best_params, best_val_mse,
    timing_rows). timing_rows: one {params_json, val_mse, time_s} dict per
    candidate — same convention as time_series_prediction.py's tune_on_val()."""
    y_val = val_series.to_numpy()
    best_params, best_mse = None, float("inf")
    rows = []
    for params in grid:
        t0 = time.perf_counter()
        try:
            pred = _fit_prophet(train_series, val_series.index, params)
            yhat = pred.reindex(val_series.index).ffill().bfill().fillna(0).to_numpy()
            mse = mean_squared_error(y_val, yhat) if np.all(np.isfinite(yhat)) else float("nan")
        except Exception:
            mse = float("nan")
        elapsed = time.perf_counter() - t0
        rows.append({"params_json": json.dumps(params), "val_mse": mse, "time_s": round(elapsed, 4)})
        if np.isfinite(mse) and mse < best_mse:
            best_mse, best_params = mse, params
    return best_params, best_mse, rows


def run_prophet_baseline(
    dataset: str,
    xes_path: str | Path,
    trim: str = "none",
    is_real: bool = False,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Tune + fit Prophet on CC and TT for one dataset/trim; write metrics + timing.

    Parameters
    ----------
    dataset    : log stem name, e.g. 'loan_flat' or 'bpic12-a'
    xes_path   : path to the .xes file
    trim       : trim config string, e.g. 'none', 'peak_0.6', 'magnitude_1'
    is_real    : True for real-life logs
    output_dir : where to write the metrics CSV; defaults to the standard location
    overwrite  : if False, skip if 'prophet' row already present in the CSV.
                 If True, re-run and replace both the metrics and timing rows —
                 use this to refresh old fixed-config (pre-HPO) results.

    Returns
    -------
    DataFrame with the two new rows (cc and tt).
    """
    import pm4py

    if output_dir is None:
        if is_real:
            output_dir = RESULTS / trim / dataset
        else:
            output_dir = RESULTS / "synthetic" / trim / dataset
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"metrics_{dataset}.csv"
    time_path    = output_dir / f"time_prophet_{dataset}.csv"

    if not overwrite and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        if "prophet" in existing["model"].values:
            print(f"[skip] 'prophet' already in {metrics_path}")
            return existing[existing["model"] == "prophet"]

    print(f"[prophet] loading {xes_path} …")
    log = pm4py.read_xes(str(xes_path))
    if trim == "ssd":
        # ssd's cutoff comes from steady-state detection (run_ssd_trim), not a
        # pct/magnitude/peak formula -- _trim_kw()/apply_trim() have no "ssd"
        # case at all (a bare `trim="ssd"` would crash _trim_kw's rsplit("_", 1)
        # unpack). Reuse analysis/ts_comparison.py's own load_splits(), which
        # already implements the identical, project-wide ssd split
        # (run_ssd_trim + a plain train/val/test slice of the full log) rather
        # than re-deriving it a second time here.
        import sys as _sys
        from pathlib import Path as _Path
        _analysis_dir = _Path(__file__).resolve().parent / "analysis"
        if str(_analysis_dir) not in _sys.path:
            _sys.path.insert(0, str(_analysis_dir))
        from ts_comparison import load_splits as _load_splits_ts
        _split = _load_splits_ts(dataset, "ssd", is_real=True)
        cc = _split["cc"]
        tt = _split["tt"]
    else:
        ts  = ts_splits_from_log(log, **_trim_kw(trim), train_frac=0.7, val_frac=0.1, cut_date=None)
        cc  = ts["concurrent_cases"]
        tt  = ts["throughput_time"]

    metric_rows = []
    timing_rows = []

    for series_name, s in [("concurrent_cases", cc), ("throughput_time", tt)]:
        print(f"[prophet] tuning {series_name} ({len(PROPHET_GRID)} candidates)…")
        best_params, best_val_mse, tune_rows = _tune_prophet(s["train"], s["val"], PROPHET_GRID)
        for row in tune_rows:
            timing_rows.append({
                "model": "prophet", "phase": "tune", **row, "is_best": False,
                "dataset": dataset, "trim": trim, "series": series_name,
            })
        if best_params is None:
            print(f"[prophet]   all candidates failed for {series_name} — falling back to defaults")
            best_params = {}
        else:
            best_json = json.dumps(best_params)
            for row in timing_rows[-len(tune_rows):]:
                if row["params_json"] == best_json:
                    row["is_best"] = True
        print(f"[prophet]   best: {best_params}  (val_mse={best_val_mse:.4f})")

        train_val = pd.concat([s["train"], s["val"]])
        test_idx  = s["test"].index

        t0 = time.perf_counter()
        pred = _fit_prophet(train_val, test_idx, best_params)
        fit_elapsed = time.perf_counter() - t0
        timing_rows.append({
            "model": "prophet", "phase": "fit",
            "params_json": json.dumps(best_params), "val_mse": best_val_mse,
            "time_s": round(fit_elapsed, 4), "is_best": True,
            "dataset": dataset, "trim": trim, "series": series_name,
        })

        actual = s["test"].reindex(test_idx).ffill().bfill().fillna(0)
        yhat   = pred.reindex(test_idx).ffill().bfill().fillna(0)
        metric_rows.append({
            "dataset": dataset, "series": series_name, "model": "prophet",
            "mse": mean_squared_error(actual.values, yhat.values),
            "mae": mean_absolute_error(actual.values, yhat.values),
        })

    new_rows = pd.DataFrame(metric_rows)

    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        existing = existing[existing["model"] != "prophet"]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_csv(metrics_path, index=False)

    pd.DataFrame(timing_rows).to_csv(time_path, index=False)

    print(f"[prophet] cc_mae={metric_rows[0]['mae']:.4f}  tt_mae={metric_rows[1]['mae']:.4f}  "
          f"→ {metrics_path}")
    print(f"[prophet] timing ({len(timing_rows)} rows) → {time_path}")
    return new_rows
