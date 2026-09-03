"""
ts_comparison.py
"""
from __future__ import annotations

import copy
import functools
import gc
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # project root — this module lives in analysis/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plain-field"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # analysis/ itself — for extra_forecasters

from extra_forecasters import EXTRA_FORECASTERS

RESULTS = ROOT / "results"
BEST = ROOT / "best_models"


_NA_VALUES_NO_NA = {
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NULL", "NaN", "None",
    "n/a", "nan", "null",
}


def _read_csv_safe_caseid(path) -> pd.DataFrame:
    """pd.read_csv with "NA" excluded from the recognized missing-value
    strings for the caseid column specifically -- see _NA_VALUES_NO_NA."""
    return pd.read_csv(path, keep_default_na=False, na_values=_NA_VALUES_NO_NA,
                       dtype={"caseid": str})

STAT_CANDIDATES = ["ets", "sarimax", "theta", "stl"]
ML_CANDIDATES = ["ridge", "ridge_mimo", "gru", "gru_mimo"]
PPM_CANDIDATES = ["amiri", "bukhsh_rt", "bukhsh_suffix", "camargo"]
PPM_REGIMES = ["plain_field", "half", "full"]

GLSTM = ROOT / "GenerativeLSTM" / "GenerativeLSTM"


_TRIM_KW = {
    "none":        dict(trim_method=None),
    "peak_0.6":    dict(trim_method="peak",      trim_frac=0.6),
    "peak_0.7":    dict(trim_method="peak",      trim_frac=0.7),
    "peak_0.8":    dict(trim_method="peak",      trim_frac=0.8),
    "magnitude_1": dict(trim_method="magnitude", trim_k=1.0),
    "magnitude_2": dict(trim_method="magnitude", trim_k=2.0),
    "magnitude_3": dict(trim_method="magnitude", trim_k=3.0),
}
_BASE = dict(trim_pct=0.25, trim_k=1.5, trim_frac=0.6, trim_window=7,
             train_frac=0.7, val_frac=0.1, cut_date=None)

_COLS = {
    "case:concept:name":    "caseid",
    "concept:name":         "task",
    "lifecycle:transition": "event_type",
    "org:resource":         "user",
    "time:timestamp":       "end_timestamp",
}


# ── Loading + splitting ──────────────────────────────────────────────────────

def _xes_path(dataset: str, is_real: bool) -> Path:
    sub = "real-life" if is_real else "synthetic"
    return ROOT / "data" / sub / f"{dataset}.xes"


@functools.lru_cache(maxsize=None)
def _read_xes_cached(dataset: str, is_real: bool) -> pd.DataFrame:
    import pm4py
    xes = _xes_path(dataset, is_real)
    if not xes.exists():
        raise FileNotFoundError(f"No XES found: {xes}")
    return pm4py.read_xes(str(xes))


def _ssd_split(log: pd.DataFrame) -> tuple[dict, dict]:
    from time_series_preprocessing import Split3WayConfig, split_timeseries
    from time_series_creation import create_concurrent_cases_timeseries, create_avg_throughtput_time_timeseries
    _ssd_dir = str(ROOT / "steady_state_detection")
    if _ssd_dir not in sys.path:
        sys.path.insert(0, _ssd_dir)
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

    return _slice(full_cc_raw, full_cc_trimmed), _slice(full_tt_raw, full_tt_trimmed)


@functools.lru_cache(maxsize=None)
def _load_splits_cached(dataset: str, trim: str, is_real: bool) -> dict:
    from create_prefixes_from_windows import make_three_way_split

    log = _read_xes_cached(dataset, is_real).copy()

    if trim == "ssd":
        if not is_real:
            raise ValueError("'ssd' trim is real-life only")
        cc, tt = _ssd_split(log)
    else:
        from time_series_preprocessing import ts_splits_from_log
        trim_kw = _TRIM_KW.get(trim, dict(trim_method=None)) if is_real else dict(trim_method=None)
        ts = ts_splits_from_log(log, **{**_BASE, **trim_kw})
        cc, tt = ts["concurrent_cases"], ts["throughput_time"]

    df = log.copy()
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="raise", utc=True)
    df = df.dropna(subset=["case:concept:name"])
    df = df[df["case:concept:name"] != ""]
    df = df.rename(columns=_COLS)
    df["task"] = df["task"].fillna("unk")
    df["user"] = df.get("user", pd.Series("unk", index=df.index)).fillna("unk")

    train_, val_, test_ = make_three_way_split(
        df, case_col="caseid", time_col="end_timestamp",
        train_split=cc["train_split"], val_split=cc["val_split"], full_traces=True,
    )
    return dict(df=df, cc=cc, tt=tt, train=train_, val=val_, test=test_)


def load_splits(dataset: str, trim: str, is_real: bool) -> dict:
    return copy.deepcopy(_load_splits_cached(dataset, trim, is_real))



def forecast_naive(train: pd.Series, horizon: int, params: dict | None = None) -> np.ndarray:
    return np.full(horizon, float(train.iloc[-1]), dtype=float)


def forecast_sarimax(train: pd.Series, horizon: int, params: dict | None = None) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    p = params or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(train, order=tuple(p.get("order", (1, 0, 0))),
                         seasonal_order=tuple(p.get("seasonal_order", (0, 0, 0, 0))), trend="c")
        res = model.fit(disp=False, method="lbfgs", maxiter=200)
    yhat = np.asarray(res.get_forecast(steps=horizon).predicted_mean, dtype=float)
    res.remove_data(); del res, model; gc.collect()
    return yhat


def forecast_ets(train: pd.Series, horizon: int, params: dict | None = None) -> np.ndarray:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    p = params or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ETSModel(
                train, error=p.get("error", "add"), trend=p.get("trend", None),
                damped_trend=bool(p.get("damped_trend", False)),
                seasonal=p.get("seasonal", None), seasonal_periods=int(p.get("seasonal_periods", 7)),
            )
            res = model.fit(maxiter=1000, disp=False)
            yhat = np.asarray(res.forecast(horizon), dtype=float)
            if np.all(np.isfinite(yhat)):
                return yhat
        except Exception:
            pass
        model = ETSModel(train, error="add", trend=None, seasonal=None)
        res = model.fit(maxiter=1000, disp=False)
    return np.asarray(res.forecast(horizon), dtype=float)


def forecast_theta(train: pd.Series, horizon: int, params: dict | None = None) -> np.ndarray:
    from statsmodels.tsa.forecasting.theta import ThetaModel
    p = params or {}
    theta = float(p.get("theta", 2.0))
    deseasonalize = bool(p.get("deseasonalize", True))
    period = int(p.get("period", 7))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ThetaModel(train, period=period, deseasonalize=deseasonalize, use_test=False)
        result = model.fit(disp=False)
    return np.asarray(result.forecast(horizon, theta=theta), dtype=float)


def forecast_stl(train: pd.Series, horizon: int, params: dict | None = None) -> np.ndarray:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.forecasting.stl import STLForecast
    p = params or {}
    period = int(p.get("period", 7))
    error = p.get("error", "add")
    trend = p.get("trend", None)
    damped_trend = bool(p.get("damped_trend", False))

    def _fit(err_type):
        stlf = STLForecast(
            train, ETSModel, period=period,
            model_kwargs={"error": err_type, "trend": trend,
                          "damped_trend": damped_trend, "seasonal": None},
        )
        return stlf.fit(fit_kwargs={"maxiter": 1000, "disp": False})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = _fit(error)
        except ValueError:
            result = _fit("add")
    return np.asarray(result.forecast(horizon), dtype=float)


STAT_FORECASTERS = {
    "naive": forecast_naive, "ets": forecast_ets, "sarimax": forecast_sarimax,
    "theta": forecast_theta, "stl": forecast_stl,
}


# ── Model selection ──────────────────────────────────────────────────────────

def _metrics_dir(dataset: str, trim: str, is_real: bool) -> Path:
    return RESULTS / trim / dataset if is_real else RESULTS / "synthetic" / "none" / dataset


def _find_baseline_csv(directory: Path, dataset: str):
    p = directory / f"metrics_{dataset}.csv"
    if p.exists():
        return p
    p = directory / "metrics.csv"
    if p.exists():
        return p
    candidates = sorted(f for f in directory.glob("metrics_*.csv")
                        if not f.stem.endswith("_val_mean"))
    return candidates[0] if candidates else None


def _pick_time_csv(dir_path: Path):
    candidates = [c for c in sorted(dir_path.glob("time*.csv"))
                  if not c.name.startswith("time_prophet_")]
    if not candidates:
        return None
    named = [c for c in candidates if c.name != "time.csv"]
    pool = named or candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def _pick_best_by_recorded_mae(dataset: str, trim: str, is_real: bool, candidates: list[str]):
    d = _metrics_dir(dataset, trim, is_real)
    metrics_path = _find_baseline_csv(d, dataset)
    time_path = _pick_time_csv(d)
    if metrics_path is None:
        raise FileNotFoundError(f"No metrics_{dataset}.csv (or variant) found in {d}")

    metrics = pd.read_csv(metrics_path)
    ranks: dict[str, list[int]] = {}
    for series in ("concurrent_cases", "throughput_time"):
        sub = metrics[(metrics["series"] == series) & (metrics["model"].isin(candidates))]
        sub = sub.sort_values("mae")
        for rank, model in enumerate(sub["model"], start=1):
            ranks.setdefault(model, []).append(rank)
    if not ranks:
        raise ValueError(f"None of {candidates} found in {metrics_path}")
    best_model = min(ranks, key=lambda m: sum(ranks[m]))

    best_params, mae = {}, {}
    time_df = pd.read_csv(time_path) if time_path is not None else pd.DataFrame()
    for series in ("concurrent_cases", "throughput_time"):
        if not time_df.empty:
            row = time_df[(time_df["model"] == best_model) & (time_df["series"] == series)
                          & (time_df["phase"] == "fit")]
            best_params[series] = json.loads(row.iloc[0]["params_json"]) if not row.empty else {}
        else:
            best_params[series] = {}
        m = metrics[(metrics.series == series) & (metrics.model == best_model)]
        mae[series] = float(m["mae"].iloc[0]) if not m.empty else None

    return best_model, best_params, mae


def pick_best_statistical_model(dataset: str, trim: str, is_real: bool):
    return _pick_best_by_recorded_mae(dataset, trim, is_real, STAT_CANDIDATES)


def pick_best_ml_model(dataset: str, trim: str, is_real: bool):
    return _pick_best_by_recorded_mae(dataset, trim, is_real, ML_CANDIDATES)


def pick_best_foundation_model(dataset: str, trim: str, is_real: bool):
    foundation = load_foundation_predictions(dataset, trim, is_real)
    present = [m for m in ("chronos", "tabpfn") if m in foundation]
    if not present:
        return None, None, None
    recorded = get_recorded_foundation_mae(dataset, trim, is_real)
    if len(present) == 1:
        best = present[0]
    else:
        ranks: dict[str, list[int]] = {}
        for series in ("concurrent_cases", "throughput_time"):
            ordered = sorted(present, key=lambda m: recorded.get(m, {}).get(series, float("inf")))
            for rank, m in enumerate(ordered, start=1):
                ranks.setdefault(m, []).append(rank)
        best = min(ranks, key=lambda m: sum(ranks[m]))
    return best, foundation[best], recorded.get(best, {})


def _ppm_metrics_path(dataset: str, trim: str, model_name: str):
    p = RESULTS / "plain_field" / model_name / trim / run / f"metrics_{run}.csv"
    if p.exists():
        return p, model_name
    if model_name == "bukhsh_rt":
        p2 = RESULTS / "plain_field" / "bukhsh" / trim / run / f"metrics_{run}.csv"
        if p2.exists():
            return p2, "bukhsh"
    return None, None


def _pick_best_plain_field_ppm(dataset: str, trim: str, is_real: bool, split: dict | None = None):
    candidates = list(PPM_CANDIDATES)
    cc_actual, tt_actual = (split["cc"]["test"], split["tt"]["test"]) if split is not None else (None, None)

    found: dict[str, dict] = {}
    for model_name in candidates:
        raw = None
        if split is not None:
            raw = _ppm_candidate_series(model_name, "plain_field", dataset, trim, is_real, cc_actual, tt_actual)
        if raw is not None:
            cc_pred, tt_pred, mae = raw
            found[model_name] = {"mae": mae, "series": {"concurrent_cases": cc_pred, "throughput_time": tt_pred}}
            continue
        p, _ = _ppm_metrics_path(dataset, trim, model_name)
        if p is None:
            continue
        df = pd.read_csv(p)
        mae = {r["series"]: r["mae"] for _, r in df.iterrows()}
        if mae:
            found[model_name] = {"mae": mae, "series": None}

    if not found:
        return None, None, None

    ranks: dict[str, list[int]] = {}
    for series in ("concurrent_cases", "throughput_time"):
        ordered = sorted(
            ((m, d["mae"][series]) for m, d in found.items() if series in d["mae"]),
            key=lambda kv: kv[1],
        )
        for rank, (model_name, _) in enumerate(ordered, start=1):
            ranks.setdefault(model_name, []).append(rank)
    if not ranks:
        return None, None, None
    best = min(ranks, key=lambda m: sum(ranks[m]))
    return best, found[best]["mae"], found[best]["series"]



def _rem_time_to_event_log(rt_df: pd.DataFrame) -> pd.DataFrame:
    start_ts = pd.to_datetime(rt_df["start_timestamp"], utc=True, format="mixed")
    anchor = pd.to_datetime(rt_df["anchor_timestamp"], utc=True, format="mixed")
    end_ts = anchor + pd.to_timedelta(rt_df["rem_time_days"].astype(float), unit="D")
    start_ts = start_ts.dt.tz_convert(None)
    end_ts = end_ts.dt.tz_convert(None)
    return pd.concat([
        pd.DataFrame({"caseid": rt_df["caseid"].values, "end_timestamp": start_ts.values}),
        pd.DataFrame({"caseid": rt_df["caseid"].values, "end_timestamp": end_ts.values}),
    ], ignore_index=True)


def load_ppm_raw_predictions(model: str, regime: str, dataset: str, trim: str, is_real: bool):
    run = f"{dataset}_test_full"

    if model == "amiri":
        base = (BEST / dataset / trim / "amiri" / f"{dataset}_full") if is_real else \
               (BEST / dataset / "amiri" / "none" / f"{dataset}_full")
        if regime == "plain_field":
            p = base / "pf" / "rem_time.csv"
        elif regime == "half":
            p = base / "half_prefix" / "rem_time.csv"
        else:
            p = base / "rem_time.csv"
        if not p.exists():
            return None
        return _rem_time_to_event_log(_read_csv_safe_caseid(p))

    if model in ("bukhsh_rt", "bukhsh_suffix"):
        base = (BEST / dataset / trim / "bukhsh" / run) if is_real else (BEST / dataset / "bukhsh" / run)
        if regime == "plain_field":
            sub = base / "pf"
        elif regime == "half":
            sub = base / "half_prefix"
        else:
            sub = base
        if model == "bukhsh_rt":
            p = sub / "rem_time.csv"
            if not p.exists():
                return None
            return _rem_time_to_event_log(_read_csv_safe_caseid(p))
        else:
            p = sub / "event_log.csv"
            if not p.exists():
                return None
            df = _read_csv_safe_caseid(p)
            df["end_timestamp"] = pd.to_datetime(df["end_timestamp"], utc=True, format="mixed").dt.tz_convert(None)
            return df

    if model == "camargo":
        if regime == "plain_field":
            p = (BEST / dataset / trim / "camargo" / "pf" / "event_log.csv") if is_real else \
                (BEST / dataset / "camargo" / "pf" / "event_log.csv")
            if not p.exists():
                return None
            df = _read_csv_safe_caseid(p)
        elif regime == "half":
            p = RESULTS / "camargo_hpo_half" / trim / run / f"event_log_{run}.csv"
            if not p.exists():
                cs, working_trim_dir = _camargo_summary_and_trim_dir(dataset, trim, is_real)
                if cs is None:
                    return None
                base_dir = GLSTM / "output_files" / working_trim_dir / cs["run_name"]
                p = base_dir / "event_log_half_prefix.csv"
                if not p.exists():
                    return None
            df = _read_csv_safe_caseid(p)
        else:
            p = _camargo_full_trace_event_log_path(dataset, trim, is_real)
            if p is None:
                return None
            df = _read_csv_safe_caseid(p)
        df["end_timestamp"] = pd.to_datetime(df["end_timestamp"], utc=True, format="mixed").dt.tz_convert(None)
        return df

    raise ValueError(f"Unknown PPM model: {model!r}")


def _camargo_full_trace_event_log_path(dataset: str, trim: str, is_real: bool):
    summary_candidates = []
    if is_real:
        summary_candidates.append(BEST / dataset / trim / "camargo" / "hpo_summary.json")
    summary_candidates.append(BEST / dataset / "camargo" / "hpo_summary.json")

    for summary_path in summary_candidates:
        if not summary_path.exists():
            continue
        cs = json.loads(summary_path.read_text())
        output_dir = Path(cs["output_dir"])
        for candidate_dir in (
            GLSTM / output_dir,
            GLSTM / output_dir.parent / "hpo" / output_dir.name,
        ):
            p = candidate_dir / "event_log.csv"
            if p.exists():
                return p
    return None


def _camargo_summary_and_trim_dir(dataset: str, trim: str, is_real: bool):
    summary_candidates = []
    if is_real:
        summary_candidates.append(BEST / dataset / trim / "camargo" / "hpo_summary.json")
    summary_candidates.append(BEST / dataset / "camargo" / "hpo_summary.json")

    for summary_path in summary_candidates:
        if not summary_path.exists():
            continue
        cs = json.loads(summary_path.read_text())
        base_trim_dir = cs["trim_dir"]
        for trim_dir_candidate in (base_trim_dir, f"{base_trim_dir}/hpo" if base_trim_dir else "hpo"):
            params_path = (GLSTM / "output_files" / trim_dir_candidate / cs["run_name"]
                          / "parameters" / "model_parameters.json")
            if params_path.exists():
                return cs, trim_dir_candidate
    return None, None


def _ppm_candidate_series(model: str, regime: str, dataset: str, trim: str, is_real: bool,
                          cc_actual: pd.Series, tt_actual: pd.Series):
    from time_series_creation import (
        create_concurrent_cases_timeseries, create_avg_throughtput_time_timeseries,
    )
    try:
        pred_log = load_ppm_raw_predictions(model, regime, dataset, trim, is_real)
    except Exception as e:
        print(f"  [warn] {dataset}/{trim} {model}/{regime}: raw prediction read failed: {e}")
        return None
    if pred_log is None or pred_log.empty:
        return None

    pred_cc = create_concurrent_cases_timeseries(
        pred_log, time_col="end_timestamp", case_col="caseid", window="days", plot=False)
    pred_tt = create_avg_throughtput_time_timeseries(
        pred_log, time_col="end_timestamp", case_col="caseid", window="days", plot=False)

    cc_aligned = pred_cc.reindex(cc_actual.index).ffill().bfill().fillna(0)
    tt_aligned = pred_tt.reindex(tt_actual.index).ffill().bfill().fillna(0)
    mae = {
        "concurrent_cases": float(np.mean(np.abs(cc_aligned.to_numpy() - cc_actual.to_numpy()))),
        "throughput_time": float(np.mean(np.abs(tt_aligned.to_numpy() - tt_actual.to_numpy()))),
    }
    return cc_aligned, tt_aligned, mae


def _pick_best_half_or_full_ppm(dataset: str, trim: str, is_real: bool, regime: str, split: dict):
    cc_actual, tt_actual = split["cc"]["test"], split["tt"]["test"]
    found: dict[str, tuple] = {}
    for model_name in PPM_CANDIDATES:
        if model_name == "camargo":
            if regime == "half" and not is_real:
                continue
        result = _ppm_candidate_series(model_name, regime, dataset, trim, is_real, cc_actual, tt_actual)
        if result is not None:
            found[model_name] = result
    if not found:
        return None, None, None

    ranks: dict[str, list[int]] = {}
    for series in ("concurrent_cases", "throughput_time"):
        ordered = sorted(found.items(), key=lambda kv: kv[1][2][series])
        for rank, (model_name, _) in enumerate(ordered, start=1):
            ranks.setdefault(model_name, []).append(rank)
    best = min(ranks, key=lambda m: sum(ranks[m]))
    cc_pred, tt_pred, mae = found[best]
    return best, mae, {"concurrent_cases": cc_pred, "throughput_time": tt_pred}


def pick_best_ppm_model(dataset: str, trim: str, is_real: bool,
                        ppm_regime: str = "plain_field", split: dict | None = None):
    if ppm_regime == "plain_field":
        return _pick_best_plain_field_ppm(dataset, trim, is_real, split)
    if ppm_regime in ("half", "full"):
        if split is None:
            raise ValueError("split is required for ppm_regime in {'half', 'full'}")
        return _pick_best_half_or_full_ppm(dataset, trim, is_real, ppm_regime, split)
    raise ValueError(f"Unknown ppm_regime: {ppm_regime!r} (expected one of {PPM_REGIMES})")




def _amiri_dir(dataset: str, trim: str, is_real: bool) -> Path:
    if is_real:
        return BEST / dataset / trim / "amiri" / f"{dataset}_full"
    return BEST / dataset / "amiri" / "none" / f"{dataset}_full"


def _bukhsh_dir(dataset: str, trim: str, is_real: bool) -> Path:
    if is_real:
        return BEST / dataset / trim / "bukhsh" / f"{dataset}_test_full"
    return BEST / dataset / "bukhsh" / f"{dataset}_test_full"


def _pf_dir(ppm_model: str, dataset: str, trim: str, is_real: bool) -> Path:
    if ppm_model == "amiri":
        return _amiri_dir(dataset, trim, is_real) / "pf"
    if ppm_model in ("bukhsh_rt", "bukhsh_suffix"):
        return _bukhsh_dir(dataset, trim, is_real) / "pf"
    if ppm_model == "camargo":
        return (BEST / dataset / trim / "camargo" / "pf") if is_real else (BEST / dataset / "camargo" / "pf")
    raise ValueError(f"Unknown ppm_model: {ppm_model!r}")




def ensure_ppm_plain_field_predictions(dataset: str, trim: str, is_real: bool,
                                       ppm_model: str, split: dict):
    existing = _ppm_candidate_series(ppm_model, "plain_field", dataset, trim, is_real,
                                     split["cc"]["test"], split["tt"]["test"])
    if existing is not None:
        cc_pred, tt_pred, _mae = existing
        return cc_pred, tt_pred

    print(f"      [regen] no plain-field predictions saved for {ppm_model} — running the live "
          f"pipeline now (will be cached to disk afterwards, no regen needed next time)...")
    cc_pred, tt_pred, m = regenerate_ppm(dataset, trim, is_real, ppm_model, split)
    print(f"      [regen] done — reconstructed metrics: {m}")
    return cc_pred, tt_pred


# ── Foundation models (Chronos, TabPFN) ─────────────────────────────────────


def load_foundation_predictions(dataset: str, trim: str, is_real: bool) -> dict:
    d = (RESULTS / "reallife_ts" / trim / dataset) if is_real else (RESULTS / "synthetic" / "none" / dataset)
    p = d / f"predictions_{dataset}.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out: dict[str, dict] = {}
    for model in ("chronos", "tabpfn"):
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        out[model] = {}
        for series in ("concurrent_cases", "throughput_time"):
            s_sub = sub[sub["series"] == series]
            if not s_sub.empty:
                out[model][series] = dict(zip(s_sub["date"], s_sub["predicted"]))
    return out


def get_recorded_foundation_mae(dataset: str, trim: str, is_real: bool) -> dict:
    d = (RESULTS / "reallife_ts" / trim / dataset) if is_real else (RESULTS / "synthetic" / "none" / dataset)
    p = _find_baseline_csv(d, dataset)
    if p is None:
        return {}
    df = pd.read_csv(p)
    out: dict[str, dict] = {}
    for model in ("chronos", "tabpfn"):
        sub = df[df["model"] == model]
        if not sub.empty:
            out[model] = dict(zip(sub["series"], sub["mae"]))
    return out


def regenerate_chronos(cc: dict, tt: dict, model_size: str = "small") -> dict:
    import torch
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(
        f"amazon/chronos-t5-{model_size}", device_map="cpu", torch_dtype=torch.float32)
    out = {}
    for name, s in [("concurrent_cases", cc), ("throughput_time", tt)]:
        train_val = pd.concat([s["train"], s["val"]])
        horizon = len(s["test"])
        context = torch.tensor(train_val.to_numpy(dtype="float32"), dtype=torch.float32)
        forecast = pipeline.predict(context, prediction_length=horizon, num_samples=20)
        out[name] = np.median(forecast.squeeze(0).numpy(), axis=0)
    return out


# ── Orchestration ────────────────────────────────────────────────────────────

def _cache_paths(dataset: str, trim: str, is_real: bool):
    d = _metrics_dir(dataset, trim, is_real)
    return d / "ts_comparison_predictions.csv", d / "ts_comparison_meta.json"


def build_comparison(dataset: str, trim: str = "none", is_real: bool = False,
                     run_ppm: bool = True, ppm_regime: str = "plain_field",
                     regenerate_if_missing: bool = False,
                     force_regenerate: bool = False):

    if ppm_regime not in PPM_REGIMES:
        raise ValueError(f"Unknown ppm_regime: {ppm_regime!r} (expected one of {PPM_REGIMES})")

    cache_csv, cache_meta = _cache_paths(dataset, trim, is_real)
    if cache_csv.exists() and cache_meta.exists() and not force_regenerate:
        cached = json.loads(cache_meta.read_text())
        # A cache written for a different ppm_regime, or by an older schema
        # (pre-ml_model / pre-single-best-foundation-model) doesn't cover this call.
        satisfies = "ml_model" in cached and ((not run_ppm) or cached.get("ppm_regime") == ppm_regime)
        if satisfies:
            print(f"[cache] loading {cache_csv}")
            return pd.read_csv(cache_csv, parse_dates=["date"]), cached
        print(f"[cache] found but for a different ppm_regime — regenerating")

    print(f"[1/5] loading {dataset} ({'real-life' if is_real else 'synthetic'}, trim={trim})...")
    split = load_splits(dataset, trim, is_real)
    cc_, tt_ = split["cc"], split["tt"]

    print("[2/5] picking best statistical model...")
    stat_model, stat_params, stat_mae = pick_best_statistical_model(dataset, trim, is_real)
    print(f"      -> {stat_model}  (recorded MAE: {stat_mae})")

    print("[3/5] picking best ML model...")
    ml_model, ml_params, ml_mae = pick_best_ml_model(dataset, trim, is_real)
    print(f"      -> {ml_model}  (recorded MAE: {ml_mae})")

    rows = []
    for series_name, s in [("concurrent_cases", cc_), ("throughput_time", tt_)]:
        train_val = pd.concat([s["train"], s["val"]])
        horizon = len(s["test"])
        idx = s["test"].index
        y_true = s["test"].to_numpy()
        y_naive = forecast_naive(train_val, horizon)
        y_stat = STAT_FORECASTERS[stat_model](train_val, horizon, stat_params[series_name])
        y_ml = EXTRA_FORECASTERS[ml_model](train_val, horizon, ml_params[series_name])
        for model_name, yhat in [("y_true", y_true), ("naive", y_naive),
                                 (stat_model, y_stat), (ml_model, y_ml)]:
            for d, v in zip(idx, yhat):
                rows.append({"date": d, "series": series_name, "model": model_name, "value": float(v)})

    print("[4/5] picking best foundation model (Chronos/TabPFN, from disk only)...")
    foundation_model, foundation_preds, foundation_mae = pick_best_foundation_model(dataset, trim, is_real)
    if foundation_model is None:
        print("      none found on disk for this dataset/trim yet")
    else:
        print(f"      -> {foundation_model}  (recorded MAE: {foundation_mae})")
        for series_name, s in [("concurrent_cases", cc_), ("throughput_time", tt_)]:
            preds_by_date = foundation_preds.get(series_name, {})
            matched_true, matched_pred = [], []
            for d in s["test"].index:
                v = preds_by_date.get(str(d.date()))
                if v is not None:
                    rows.append({"date": d, "series": series_name, "model": foundation_model, "value": float(v)})
                    matched_true.append(s["test"][d])
                    matched_pred.append(v)
            # Sanity check: MAE from these raw values should match results_auto.xlsx's
            # recorded MAE for this (model, series) — same file, zero transformation.
            rec = foundation_mae.get(series_name)
            if matched_true and rec is not None:
                computed = float(np.mean(np.abs(np.array(matched_true) - np.array(matched_pred))))
                match = "OK" if abs(computed - rec) < max(1e-6, 0.01 * abs(rec)) else "MISMATCH"
                print(f"      [sanity] {series_name}: computed MAE={computed:.2f}  "
                      f"recorded (results_auto)={rec:.2f}  [{match}]"
                      + ("" if len(matched_true) == len(s['test']) else
                         f"  (only {len(matched_true)}/{len(s['test'])} dates matched)"))

    ppm_model, ppm_mae, ppm_series = None, None, None
    if run_ppm:
        print(f"[5/5] picking best {ppm_regime} PPM model (from disk only)...")
        ppm_model, ppm_mae, ppm_series = pick_best_ppm_model(dataset, trim, is_real, ppm_regime, split)
        if ppm_model is None:
            print(f"      no {ppm_regime} PPM data found on disk for {dataset}/{trim}")
        elif ppm_series is not None:
            print(f"      -> {ppm_model}  (MAE: {ppm_mae}) — loading predictions from disk...")
            for series_name, pred in ppm_series.items():
                for d, v in pred.items():
                    rows.append({"date": d, "series": series_name, "model": ppm_model, "value": float(v)})
        elif ppm_regime == "plain_field" and regenerate_if_missing:
            print(f"      -> {ppm_model} is the best {ppm_regime} PPM model by recorded MAE "
                  f"(MAE: {ppm_mae}) — no raw predictions on disk yet, regenerating...")
            cc_pred, tt_pred = ensure_ppm_plain_field_predictions(dataset, trim, is_real, ppm_model, split)
            for series_name, pred in [("concurrent_cases", cc_pred), ("throughput_time", tt_pred)]:
                for d, v in pred.items():
                    rows.append({"date": d, "series": series_name, "model": ppm_model, "value": float(v)})
        else:
            print(f"      -> {ppm_model} is the best {ppm_regime} PPM model by recorded MAE "
                  f"(MAE: {ppm_mae}), but no raw per-case predictions are saved on disk for it "
                  f"— nothing to plot (pass ppm_regime='half' or 'full' for an actual plotted "
                  f"line where available, or regenerate_if_missing=True to run the live "
                  f"plain-field pipeline and cache it for next time).")
    else:
        print("[5/5] skipping PPM (run_ppm=False)")

    df = pd.DataFrame(rows)
    meta = {
        "dataset": dataset, "trim": trim, "is_real": is_real,
        "stat_model": stat_model, "stat_mae": stat_mae,
        "ml_model": ml_model, "ml_mae": ml_mae,
        "foundation_model": foundation_model, "foundation_mae": foundation_mae,
        "ppm_regime": ppm_regime, "ppm_model": ppm_model, "ppm_mae": ppm_mae,
    }
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    cache_meta.write_text(json.dumps(meta, indent=2, default=str))
    print(f"[done] cached -> {cache_csv}")
    return df, meta


# ── Plotting ──────────────────────────────────────────────────────────────
# Colors from the project's validated categorical palette (dataviz skill,
# references/palette.md), restricted to slots with >=3:1 contrast on white.

COLOR_TRUE = "#0b0b0b"     # text-primary — reference series, not a categorical slot
COLOR_NAIVE = "#8a8a86"    # muted gray — trivial baseline, deliberately recessive
COLOR_STAT = "#2a78d6"     # categorical slot 1 (blue)
COLOR_ML = "#eb6834"       # categorical slot 8 (orange)
COLOR_CHRONOS = "#4a3aa7"  # categorical slot 5 (violet)
COLOR_TABPFN = "#008300"   # categorical slot 4 (green)
COLOR_PPM = "#e34948"      # categorical slot 6 (red) — best PPM, made to stand out

SERIES_INFO = [
    ("concurrent_cases", "Concurrent cases"),
    ("throughput_time", "Avg. throughput time (h)"),
]


SERIES_FILE_SUFFIX = {"concurrent_cases": "cc", "throughput_time": "tt"}


def plot_comparison(df: pd.DataFrame, meta: dict, save_dir: Path | None = None):
    import matplotlib.pyplot as plt

    stat_model = meta["stat_model"]
    ml_model = meta.get("ml_model")
    foundation_model = meta.get("foundation_model")
    ppm_model = meta.get("ppm_model")

    colors = {"y_true": COLOR_TRUE, "naive": COLOR_NAIVE, stat_model: COLOR_STAT}
    labels = {"y_true": "Actual", "naive": "Naive", stat_model: f"{stat_model.upper()} (best statistical)"}
    if ml_model:
        colors[ml_model] = COLOR_ML
        labels[ml_model] = f"{ml_model} (best ML)"
    if foundation_model:
        colors[foundation_model] = COLOR_CHRONOS if foundation_model == "chronos" else COLOR_TABPFN
        labels[foundation_model] = f"{foundation_model} (best foundation)"
    regime_label = {"plain_field": "plain-field", "half": "half-prefix", "full": "full-trace"}.get(
        meta.get("ppm_regime", "plain_field"), meta.get("ppm_regime"))
    if ppm_model:
        colors[ppm_model] = COLOR_PPM
        labels[ppm_model] = f"{ppm_model} (best {regime_label} PPM)"

    present = set(df["model"].unique())
    order = [m for m in ["naive", stat_model, ml_model, foundation_model, ppm_model, "y_true"]
             if m and m in present]

    title = f"{meta['dataset']}" + (f" ({meta['trim']})" if meta.get("is_real") else "")
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    figs = {}
    for series_key, series_label in SERIES_INFO:
        fig, ax = plt.subplots(figsize=(10, 4))
        sub = df[df["series"] == series_key]
        for model in order:
            s = sub[sub["model"] == model].sort_values("date")
            if s.empty:
                continue
            ax.plot(s["date"], s["value"], color=colors.get(model, "#999999"),
                    label=labels.get(model, model),
                    linewidth=2.5 if model == "y_true" else 1.8,
                    zorder=3 if model == "y_true" else 2)
        ax.set_ylabel(series_label)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.set_title(f"{title} — {series_label} (actual vs. predicted, test period)")
        ax.set_xlabel("date")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        plt.tight_layout()
        if save_dir:
            suffix = SERIES_FILE_SUFFIX[series_key]
            fig.savefig(save_dir / f"ts_comparison_{meta['dataset']}_{meta['trim']}_{suffix}.pdf",
                       bbox_inches="tight")
        plt.show()
        figs[series_key] = fig
    return figs
