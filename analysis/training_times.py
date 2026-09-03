"""

Scrapes the training time recorded for every approach

Every row reports up to two numbers, always labelled so they're never confused:

  single_run_s  — time to train the ONE model that was actually deployed
                  (the winning HPO config, or the only config if there's no HPO).
  full_space_s  — total time spent training every config tried during HPO
                  (i.e. the whole search), or equal to single_run_s if there
                  was nothing to search over (e.g. a naive baseline).

"""

from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # project root — this module lives in analysis/
BEST = ROOT / "best_models"
RESULTS = ROOT / "results"
RESULTS_TS = RESULTS / "reallife_ts"
GLSTM = ROOT / "GenerativeLSTM" / "GenerativeLSTM"

REAL_TRIM_NAMES = ["none", "peak_0.6", "peak_0.7", "peak_0.8",
                   "magnitude_1", "magnitude_2", "magnitude_3"]

real_logs = [p.stem for p in sorted((ROOT / "data" / "real-life").glob("*.xes"))]

AMIRI_MAX_TRIALS   = 12
CAMARGO_MAX_EVAL   = 10

_AMIRI_TIME_RE = re.compile(r"Total train loop time:\s*([\d.]+)\s*h")

INFERENCE_APPROACHES = {"ts_chronos", "ts_tabpfn"}

APPROACH_ORDER = [
    "ts_naive", "ts_seasonal_naive", "ts_ets", "ts_sarimax", "ts_theta", "ts_stl",
    "ts_ridge", "ts_ridge_mimo", "ts_gru", "ts_gru_mimo", "ts_nbeats", "ts_nhits",
    "ts_tft", "ts_prophet", "ts_chronos", "ts_tabpfn",
    "camargo", "amiri", "bukhsh",
]


def _regime(approach: str) -> str:
    return "inference" if approach in INFERENCE_APPROACHES else "training"


def _fmt_duration(seconds):
    if seconds is None or pd.isna(seconds):
        return "—"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _row(dataset, trim, approach, single_s, single_kind, full_s, full_kind,
         series=None, source=None):
    return {
        "dataset":         dataset,
        "trim":            trim,
        "approach":        approach,
        "regime":          _regime(approach),
        "series":          series,
        "single_run_s":    single_s,
        "single_run":      _fmt_duration(single_s),
        "single_run_kind": single_kind,
        "full_space_s":    full_s,
        "full_space":      _fmt_duration(full_s),
        "full_space_kind": full_kind,
        "source":          source,
    }


# ── Amiri ──────────────────────────────────────────────────────────────────

def get_amiri_time(dataset: str, trim: str):
    base = BEST / dataset / trim / "amiri" / f"{dataset}_full" / "gps_results" / "amiri_gps"
    logs = sorted(base.glob("*/logging.log"))
    if not logs:
        return _row(dataset, trim, "amiri", None, None, None, None, source=None)

    text = logs[0].read_text()
    m = _AMIRI_TIME_RE.search(text)
    single_s = float(m.group(1)) * 3600 if m else None
    source = str(logs[0].relative_to(ROOT))

    hpo_dir = BEST / dataset / trim / "amiri" / "hpo_trials"
    trial_dirs = sorted(hpo_dir.glob("trial_*"))
    full_s, full_kind = None, None
    if trial_dirs:
        total, n_found = 0.0, 0
        for td in trial_dirs:
            log = td / "training.log"
            if not log.exists():
                continue
            tm = _AMIRI_TIME_RE.search(log.read_text())
            if tm:
                total += float(tm.group(1)) * 3600
                n_found += 1
        if n_found == len(trial_dirs) and n_found > 0:
            full_s, full_kind = total, "exact"
        elif n_found > 0:
            full_s, full_kind = total / n_found * len(trial_dirs), "approx"
    if full_s is None and single_s is not None:
        full_s, full_kind = single_s * AMIRI_MAX_TRIALS, "approx"

    return _row(dataset, trim, "amiri", single_s, "exact" if single_s is not None else None,
                full_s, full_kind, source=source)


# ── Bukhsh ─────────────────────────────────────────────────────────────────

def get_bukhsh_time(dataset: str, trim: str):
    bukhsh_dir = BEST / dataset / trim / "bukhsh"

    single_s, single_source = None, None
    for time_csv in sorted(bukhsh_dir.glob("time_*.csv")):
        df = pd.read_csv(time_csv)
        best_rows = df[df.get("is_best", False) == True]  # noqa: E712
        if not best_rows.empty:
            single_s = float(best_rows.iloc[0]["time_s"])
            single_source = str(time_csv.relative_to(ROOT))
            break

    csv_path = bukhsh_dir / "hpo_results.csv"
    full_s, full_kind, full_source = None, None, None
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if not df.empty and "train_s" in df.columns:
            full_s, full_kind = float(df["train_s"].sum()), "exact"
            full_source = str(csv_path.relative_to(ROOT))
            if single_s is None:
                best = df.sort_values("val_cc_mae").iloc[0]
                single_s, single_source = float(best["train_s"]), full_source

    source = single_source or full_source
    if single_source and full_source and single_source != full_source:
        source = f"{single_source} | {full_source}"

    return _row(dataset, trim, "bukhsh",
                single_s, "exact" if single_s is not None else None,
                full_s, full_kind, source=source)


# ── Camargo ────────────────────────────────────────────────────────────────

def get_camargo_time(dataset: str, trim: str):
    json_path = BEST / dataset / trim / "camargo" / "hpo_summary.json"
    if not json_path.exists():
        return _row(dataset, trim, "camargo", None, None, None, None, source=None)

    data = json.loads(json_path.read_text())
    full_s = data.get("train_time_s")
    full_kind = "exact" if full_s is not None else None
    source = str(json_path.relative_to(ROOT))

    n_trials = None
    out_dir = data.get("output_dir")
    if out_dir:
        trials_dir = GLSTM / f"{out_dir}_trials"
        if trials_dir.exists():
            n_trials = len(list(trials_dir.glob("trial_*"))) or None
    n_trials = n_trials or CAMARGO_MAX_EVAL

    single_s = (full_s / n_trials) if full_s is not None else None
    single_kind = "approx" if single_s is not None else None

    return _row(dataset, trim, "camargo", single_s, single_kind,
                float(full_s) if full_s is not None else None, full_kind, source=source)


EVENT_LOG_GETTERS = {
    "amiri":   get_amiri_time,
    "bukhsh":  get_bukhsh_time,
    "camargo": get_camargo_time,
}


# ── Time-series baselines (results/ and results/reallife_ts/) ──────────────

def _pick_time_csv(dir_path: Path):
    candidates = [c for c in sorted(dir_path.glob("time*.csv"))
                  if not c.name.startswith("time_prophet_")]
    if not candidates:
        return None
    named = [c for c in candidates if c.name != "time.csv"]
    pool = named or candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def _ts_rows_from_csv(csv_path: Path, dataset: str, trim: str, source_pipeline: str):
    df = pd.read_csv(csv_path)
    rows = []
    for (model, series), grp in df.groupby(["model", "series"]):
        fit_rows = grp[grp["phase"] == "fit"]
        single_s = float(fit_rows.iloc[0]["time_s"]) if not fit_rows.empty else None
        full_s = float(grp["time_s"].sum())
        rows.append(_row(
            dataset, trim, f"ts_{model}", single_s,
            "exact" if single_s is not None else None,
            full_s, "exact", series=series,
            source=f"{source_pipeline}:{csv_path.relative_to(ROOT)}",
        ))
    return rows


def get_ts_model_rows(dataset: str, trim: str):
    rows = []

    main_dir = RESULTS / trim / dataset
    if main_dir.exists():
        csv_path = _pick_time_csv(main_dir)
        if csv_path is not None:
            rows.extend(_ts_rows_from_csv(csv_path, dataset, trim, "main"))

    ts_dir = RESULTS_TS / trim / dataset
    if ts_dir.exists():
        csv_path = _pick_time_csv(ts_dir)
        if csv_path is not None:
            rows.extend(_ts_rows_from_csv(csv_path, dataset, trim, "reallife_ts"))

    return rows


def get_prophet_rows(dataset: str, trim: str):
    metrics_path = RESULTS / trim / dataset / f"metrics_{dataset}.csv"
    if not metrics_path.exists():
        return []
    df = pd.read_csv(metrics_path)
    if "prophet" not in df.get("model", pd.Series(dtype=str)).values:
        return []

    time_path = RESULTS / trim / dataset / f"time_prophet_{dataset}.csv"
    times = {}
    if time_path.exists():
        tdf = pd.read_csv(time_path)
        times = dict(zip(tdf["series"], tdf["time_s"]))

    rows = []
    for series in ("concurrent_cases", "throughput_time"):
        t = times.get(series)
        source = str(time_path.relative_to(ROOT)) if time_path.exists() else str(metrics_path.relative_to(ROOT))
        rows.append(_row(
            dataset, trim, "ts_prophet",
            float(t) if t is not None else None, "exact" if t is not None else None,
            float(t) if t is not None else None, "exact" if t is not None else None,
            series=series, source=source,
        ))
    return rows


def build_timing_table() -> pd.DataFrame:
    rows = []
    for dataset in real_logs:
        for trim in REAL_TRIM_NAMES:
            for getter in EVENT_LOG_GETTERS.values():
                rows.append(getter(dataset, trim))
            rows.extend(get_ts_model_rows(dataset, trim))
            rows.extend(get_prophet_rows(dataset, trim))
    return pd.DataFrame(rows)


# ── Averaged summary tables ─────────────────────────────────────────────────

def _with_regime_col(pivot: pd.DataFrame) -> pd.DataFrame:
    pivot = pivot.reindex(APPROACH_ORDER)
    pivot.insert(0, "regime", [_regime(a) for a in pivot.index])
    return pivot


def build_avg_by_trim(timing_df: pd.DataFrame, value_col: str = "single_run_s") -> pd.DataFrame:
    """Rows = approach (canonical order), columns = trim, values = mean(value_col)
    across datasets (+series). First column flags training vs. inference."""
    pivot = timing_df.groupby(["approach", "trim"])[value_col].mean().unstack("trim")
    pivot = pivot.reindex(columns=REAL_TRIM_NAMES)
    pivot["avg_all_trims"] = timing_df.groupby("approach")[value_col].mean()
    return _with_regime_col(pivot)


def build_avg_by_dataset(timing_df: pd.DataFrame, value_col: str = "single_run_s") -> pd.DataFrame:
    """Rows = approach (canonical order), columns = dataset, values = mean(value_col)
    across trims (+series). First column flags training vs. inference."""
    pivot = timing_df.groupby(["approach", "dataset"])[value_col].mean().unstack("dataset")
    pivot = pivot.reindex(columns=real_logs)
    pivot["avg_all_datasets"] = timing_df.groupby("approach")[value_col].mean()
    return _with_regime_col(pivot)


def _fmt_table(pivot: pd.DataFrame) -> pd.DataFrame:
    fmt = pivot.copy()
    value_cols = [c for c in fmt.columns if c != "regime"]
    fmt[value_cols] = fmt[value_cols].map(_fmt_duration)
    return fmt


def build_avg_training_time_row(timing_df: pd.DataFrame,
                                 value_col: str = "single_run_s") -> pd.DataFrame:
    means = timing_df.groupby("approach")[value_col].mean()
    row = {}
    for a in APPROACH_ORDER:
        row[a] = "-" if a in INFERENCE_APPROACHES else _fmt_duration(means.get(a))
    return pd.DataFrame([row])


def save_avg_training_time_excel(timing_df: pd.DataFrame,
                                  out_path: Path = None,
                                  value_col: str = "single_run_s") -> Path:
    out_path = out_path or (RESULTS / "training_times_avg.xlsx")
    df = build_avg_training_time_row(timing_df, value_col)
    df.to_excel(out_path, index=False)
    return out_path


if __name__ == "__main__":
    timing_df = build_timing_table()
    timing_df["approach"] = pd.Categorical(timing_df["approach"],
                                            categories=APPROACH_ORDER, ordered=True)
    timing_df = timing_df.sort_values(["approach", "dataset", "trim"])

    n_approaches = timing_df["approach"].nunique()
    print(f"Datasets: {len(real_logs)}   Trims: {len(REAL_TRIM_NAMES)}   "
          f"Approaches: {n_approaches} ({APPROACH_ORDER})")
    print()
    print(timing_df[["dataset", "trim", "approach", "regime", "series", "single_run",
                      "single_run_kind", "full_space", "full_space_kind"]]
          .to_string(index=False))

    print()
    found = timing_df.dropna(subset=["single_run_s"])
    missing = timing_df[timing_df["single_run_s"].isna()]
    print(f"Found  : {len(found)}/{len(timing_df)} rows (single_run_s)")
    print(f"Missing: {len(missing)}/{len(timing_df)} rows")

    print()
    print("Per-approach summary — single_run_s (seconds):")
    summary = (found.groupby("approach", observed=True)["single_run_s"]
               .agg(["count", "mean", "min", "max"]).round(1))
    summary.insert(0, "regime", [_regime(a) for a in summary.index])
    print(summary.to_string())

    print()
    print("=" * 78)
    print("Average single_run_s BY TRIM (mean across datasets):")
    print("=" * 78)
    print(_fmt_table(build_avg_by_trim(timing_df, "single_run_s")).to_string())

    print()
    print("=" * 78)
    print("Average single_run_s BY DATASET (mean across trims):")
    print("=" * 78)
    print(_fmt_table(build_avg_by_dataset(timing_df, "single_run_s")).to_string())

    print()
    print("=" * 78)
    print("Overall average training time — one row, one column per approach:")
    print("=" * 78)
    avg_row = build_avg_training_time_row(timing_df, "single_run_s")
    print(avg_row.to_string(index=False))
    xlsx_path = save_avg_training_time_excel(timing_df, value_col="single_run_s")
    print(f"\nSaved → {xlsx_path.relative_to(ROOT)}")
