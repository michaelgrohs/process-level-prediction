"""
split_correlation.py


"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # project root — this module lives in analysis/
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"

REAL_TRIM_NAMES = ["none", "peak_0.6", "peak_0.7", "peak_0.8",
                   "magnitude_1", "magnitude_2", "magnitude_3"]
real_logs = [p.stem for p in sorted((ROOT / "data" / "real-life").glob("*.xes"))]

NAIVE_BASELINES = ["naive", "val_mean"]
STATISTICAL = ["prophet", "ets", "sarimax", "stl"]
ML_MODELS = ["ridge", "ridge_mimo", "nhits", "tft"]
FOUNDATION_MODELS = ["tabpfn", "chronos"]
SIMULATION_MODELS = ["pmsd", "simod", "agentsimulator"]
PROCESS_MODELS = ["camargo", "amiri", "bukhsh_rt", "bukhsh_suffix"]
APPROACH_ORDER = (NAIVE_BASELINES + STATISTICAL + ML_MODELS + FOUNDATION_MODELS
                  + SIMULATION_MODELS + PROCESS_MODELS)


PPM_REGIMES = ["plain_field", "half", "full"]

DISPLAY_NAMES = {
    "naive": "Naive", "val_mean": "Validation Average",
    "prophet": "Prophet", "ets": "ETS", "sarimax": "SARIMAX", "stl": "STL",
    "ridge": "Ridge", "ridge_mimo": "Ridge_M", "nhits": "N-HiTS", "tft": "TFT",
    "tabpfn": "TabPFN", "chronos": "Chronos", "pmsd": "PMSD",
    "simod": "Simod", "agentsimulator": "AgentSimulator",
    "camargo": "GLSTM", "amiri": "PGTNet", "bukhsh_rt": "PT_RT", "bukhsh_suffix": "PT_su",
}
REGIME_DISPLAY = {"plain_field": "plain-field", "half": "half prefix", "full": "full trace"}

SERIES = ["concurrent_cases", "throughput_time"]

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
    "case:concept:name": "caseid", "concept:name": "task",
    "lifecycle:transition": "event_type", "org:resource": "user",
    "time:timestamp": "end_timestamp",
}


# ── Split measures ───────────────────────────────────────────────────────────

def _cv(x: np.ndarray) -> float:
    m = np.mean(x)
    return float(np.std(x) / m) if m != 0 else np.nan


def _autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag:
        return np.nan
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def _trend_pct_per_day(x: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    t = np.arange(len(x))
    slope = np.polyfit(t, x, 1)[0]
    m = np.mean(x)
    return float(slope / m) if m != 0 else np.nan


def _series_measures(prefix: str, train: np.ndarray, test: np.ndarray) -> dict:
    train_mean, train_std = np.mean(train), np.std(train)
    return {
        f"{prefix}_train_cv":             _cv(train),
        f"{prefix}_test_cv":              _cv(test),
        f"{prefix}_train_test_shift":     (float(np.mean(test)) - float(train_mean)) / train_mean if train_mean != 0 else np.nan,
        f"{prefix}_test_train_std_ratio": float(np.std(test)) / train_std if train_std != 0 else np.nan,
        f"{prefix}_train_trend":          _trend_pct_per_day(train),
        f"{prefix}_train_autocorr1":      _autocorr(train, 1),
        f"{prefix}_train_autocorr7":      _autocorr(train, 7),
    }


def compute_split_measures(cc: dict, tt: dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """cc/tt: the dicts returned by time_series_preprocessing.ts_splits_from_log()['concurrent_cases'/'throughput_time']."""
    measures = {}
    measures.update(_series_measures("cc", cc["train"].to_numpy(dtype=float), cc["test"].to_numpy(dtype=float)))
    measures.update(_series_measures("tt", tt["train"].to_numpy(dtype=float), tt["test"].to_numpy(dtype=float)))

    measures["train_len_days"] = len(cc["train"])
    measures["val_len_days"] = len(cc["val"])
    measures["test_len_days"] = len(cc["test"])
    measures["test_train_len_ratio"] = len(cc["test"]) / len(cc["train"]) if len(cc["train"]) else np.nan

    measures["n_train_cases"] = train_df["caseid"].nunique()
    measures["n_test_cases"] = test_df["caseid"].nunique()
    measures["n_activities"] = train_df["task"].nunique()
    measures["avg_trace_length"] = float(train_df.groupby("caseid").size().mean())

    dur = train_df.groupby("caseid")["end_timestamp"].agg(["min", "max"])
    dur_h = (pd.to_datetime(dur["max"]) - pd.to_datetime(dur["min"])).dt.total_seconds() / 3600
    measures["case_duration_cv"] = _cv(dur_h.to_numpy(dtype=float))

    return measures


MEASURE_NAMES = (list(_series_measures("cc", np.array([1.0, 2.0]), np.array([1.0, 2.0])).keys())
                  + list(_series_measures("tt", np.array([1.0, 2.0]), np.array([1.0, 2.0])).keys())
                  + ["train_len_days", "val_len_days", "test_len_days", "test_train_len_ratio",
                     "n_train_cases", "n_test_cases", "n_activities", "avg_trace_length", "case_duration_cv"])


# ── MAE parsing (no re-prediction — reads existing metrics_*.csv) ───────────

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


_VAL_MEAN_ALL = None  # lazy-loaded cache of results/val_mean_all.csv


def _val_mean_mae(dataset: str, trim: str, is_real: bool) -> dict:
    global _VAL_MEAN_ALL
    if _VAL_MEAN_ALL is None:
        p = RESULTS / "val_mean_all.csv"
        _VAL_MEAN_ALL = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if _VAL_MEAN_ALL.empty:
        return {}
    log_group = "real" if is_real else "synthetic"
    sub = _VAL_MEAN_ALL[(_VAL_MEAN_ALL["log_group"] == log_group)
                        & (_VAL_MEAN_ALL["dataset"] == dataset)
                        & (_VAL_MEAN_ALL["trim"] == trim)]
    return dict(zip(sub["series"], sub["mae"])) if not sub.empty else {}


def _pmsd_mae(dataset: str, trim: str, is_real: bool) -> dict:
    d = (RESULTS / trim / dataset) if is_real else (RESULTS / "synthetic" / "none" / dataset)
    p = d / f"metrics_{dataset}_pmsd.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    sub = df[df["model"] == "pmsd"]
    return dict(zip(sub["series"], sub["mae"])) if not sub.empty else {}


def _simulator_mae(model: str, dataset: str, trim: str, is_real: bool) -> dict:
    suffix = "_real" if is_real else ""
    p = RESULTS / model / f"metrics_{model}{suffix}.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    mask = df["dataset"] == dataset
    if is_real and "trim" in df.columns:
        mask &= df["trim"] == trim
    sub = df[mask]
    return dict(zip(sub["series"], sub["mae"])) if not sub.empty else {}


def parse_approach_maes(dataset: str, trim: str, is_real: bool) -> dict:
    out: dict[str, dict] = {}

    d_main = (RESULTS / trim / dataset) if is_real else (RESULTS / "synthetic" / "none" / dataset)
    p = _find_baseline_csv(d_main, dataset)
    if p is not None:
        df = pd.read_csv(p)
        for model in ["naive"] + STATISTICAL + ML_MODELS:
            sub = df[df["model"] == model]
            if not sub.empty:
                out[model] = dict(zip(sub["series"], sub["mae"]))

    pmsd = _pmsd_mae(dataset, trim, is_real)
    if pmsd:
        out["pmsd"] = pmsd

    for sim_model in ("simod", "agentsimulator"):
        sim_mae = _simulator_mae(sim_model, dataset, trim, is_real)
        if sim_mae:
            out[sim_model] = sim_mae

    vm = _val_mean_mae(dataset, trim, is_real)
    if vm:
        out["val_mean"] = vm

    d_ts = (RESULTS / "reallife_ts" / trim / dataset) if is_real else d_main
    p2 = _find_baseline_csv(d_ts, dataset)
    if p2 is not None:
        df2 = pd.read_csv(p2)
        for model in FOUNDATION_MODELS:
            sub = df2[df2["model"] == model]
            if not sub.empty:
                out[model] = dict(zip(sub["series"], sub["mae"]))

    return out


def _ppm_metrics_path(model: str, regime: str, dataset: str, trim: str):
    run = f"{dataset}_test_full"
    if regime in ("full", "half"):
        base_map = {
            "camargo": "camargo_hpo", "amiri": "amiri_hpo",
            "bukhsh_suffix": "bukhsh_hpo", "bukhsh_rt": "bukhsh_hpo",
        }
        base = base_map[model] + ("_half" if regime == "half" else "")
        suffix = "_rt" if model == "bukhsh_rt" else ""
        return RESULTS / base / trim / run / f"metrics_{run}{suffix}.csv"
    if regime == "plain_field":
        dirname = model
        p = RESULTS / "plain_field" / dirname / trim / run / f"metrics_{run}.csv"
        if not p.exists() and model == "bukhsh_rt":
            p = RESULTS / "plain_field" / "bukhsh" / trim / run / f"metrics_{run}.csv"
        return p
    raise ValueError(f"Unknown regime: {regime!r}")


def parse_ppm_maes(dataset: str, trim: str, regime: str) -> dict:
    """Returns {model: {'concurrent_cases': mae, 'throughput_time': mae}} for one regime."""
    out: dict[str, dict] = {}
    for model in PROCESS_MODELS:
        p = _ppm_metrics_path(model, regime, dataset, trim)
        if p.exists():
            df = pd.read_csv(p)
            out[model] = dict(zip(df["series"], df["mae"]))
    return out


# ── Dataset assembly ─────────────────────────────────────────────────────────

def _load_and_split(log, df, cc_all_kw, trim: str):
    from time_series_preprocessing import ts_splits_from_log
    from create_prefixes_from_windows import make_three_way_split

    trim_kw = _TRIM_KW.get(trim, dict(trim_method=None))
    ts = ts_splits_from_log(log, **{**_BASE, **trim_kw})
    cc, tt = ts["concurrent_cases"], ts["throughput_time"]

    train_, val_, test_ = make_three_way_split(
        df, case_col="caseid", time_col="end_timestamp",
        train_split=cc["train_split"], val_split=cc["val_split"], full_traces=True,
    )
    return cc, tt, train_, test_


def build_dataset(datasets: list[str] | None = None,
                  trims: list[str] | None = None,
                  verbose: bool = True) -> pd.DataFrame:
    """One row per (dataset, trim): split measures + every approach's NMAE per series."""
    import pm4py
    from create_prefixes_from_windows import load_event_log

    datasets = datasets if datasets is not None else real_logs
    trims = trims if trims is not None else REAL_TRIM_NAMES

    rows = []
    for ds in datasets:
        xes = ROOT / "data" / "real-life" / f"{ds}.xes"
        if not xes.exists():
            if verbose:
                print(f"[skip] {ds}: no XES")
            continue
        if verbose:
            print(f"=== {ds}: parsing XES ===")
        log = pm4py.read_xes(str(xes))
        df = load_event_log(xes, time_col="time:timestamp", case_col="case:concept:name")
        df = df.rename(columns=_COLS)
        df["task"] = df["task"].fillna("unk")
        df["user"] = df.get("user", pd.Series("unk", index=df.index)).fillna("unk")

        for trim in trims:
            try:
                cc, tt, train_, test_ = _load_and_split(log, df, None, trim)
            except Exception as e:
                if verbose:
                    print(f"  [{trim}] split ERROR: {e}")
                continue

            measures = compute_split_measures(cc, tt, train_, test_)
            maes = parse_approach_maes(ds, trim, True)
            ppm_maes = {regime: parse_ppm_maes(ds, trim, regime) for regime in PPM_REGIMES}
            test_means = {"concurrent_cases": float(cc["test"].mean()),
                          "throughput_time": float(tt["test"].mean())}

            def _nmae(mae, series):
                tm = test_means[series]
                return (mae / tm) if (mae is not None and tm not in (0, None) and not pd.isna(tm)) else np.nan

            row = {"dataset": ds, "trim": trim, **measures}
            n_found = 0
            for series in SERIES:
                for approach in NAIVE_BASELINES + STATISTICAL + ML_MODELS + FOUNDATION_MODELS + SIMULATION_MODELS:
                    mae = maes.get(approach, {}).get(series)
                    row[f"nmae__{series}__{approach}"] = _nmae(mae, series)
                    if mae is not None:
                        n_found += 1
                for regime in PPM_REGIMES:
                    for model in PROCESS_MODELS:
                        mae = ppm_maes[regime].get(model, {}).get(series)
                        row[f"nmae__{series}__{model}__{regime}"] = _nmae(mae, series)
                        if mae is not None:
                            n_found += 1
            rows.append(row)
            if verbose:
                n_expected = len(SERIES) * (len(NAIVE_BASELINES + STATISTICAL + ML_MODELS + FOUNDATION_MODELS + SIMULATION_MODELS)
                                            + len(PPM_REGIMES) * len(PROCESS_MODELS))
                print(f"  [{trim}] {n_found}/{n_expected} (approach, series[, regime]) values found")

    return pd.DataFrame(rows)


# ── Correlation ───────────────────────────────────────────────────────────

def correlation_matrix(df: pd.DataFrame, series: str, ppm_regime: str = "full",
                       method: str = "pearson") -> pd.DataFrame:
    if ppm_regime not in PPM_REGIMES:
        raise ValueError(f"ppm_regime must be one of {PPM_REGIMES}, got {ppm_regime!r}")

    prefix = "cc" if series == "concurrent_cases" else "tt"
    shared = ["train_len_days", "val_len_days", "test_len_days", "test_train_len_ratio",
              "n_train_cases", "n_test_cases", "n_activities", "avg_trace_length", "case_duration_cv"]
    measure_cols = [c for c in MEASURE_NAMES if c.startswith(f"{prefix}_")] + shared

    non_ppm = NAIVE_BASELINES + STATISTICAL + ML_MODELS + FOUNDATION_MODELS + SIMULATION_MODELS
    columns = [(a, f"nmae__{series}__{a}") for a in non_ppm]
    columns += [(m, f"nmae__{series}__{m}__{ppm_regime}") for m in PROCESS_MODELS]

    out = pd.DataFrame(index=measure_cols,
                       columns=[DISPLAY_NAMES.get(a, a) for a, _ in columns], dtype=float)
    for measure in measure_cols:
        for approach, col in columns:
            if col not in df.columns:
                continue
            sub = df[[measure, col]].dropna()
            if len(sub) < 4:
                continue
            out.loc[measure, DISPLAY_NAMES.get(approach, approach)] = sub[measure].corr(sub[col], method=method)
    return out
