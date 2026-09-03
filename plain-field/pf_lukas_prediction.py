"""
pf_lukas_prediction.py

"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PF_DIR = Path(__file__).resolve().parent
ROOT   = PF_DIR.parent
BEST   = ROOT / "best_models"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PF_DIR))

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import pm4py

from arrival import ProphetArrivalModel, compute_arrival_series
from create_prefixes_from_windows import load_event_log, make_three_way_split
from runner import (
    _load_amiri_params,
    _rem_time_to_event_log,
    build_sos_cases,
    compute_cc_tt_metrics,
    get_inflight_cases,
    predict_amiri_plain_field,
    predict_bukhsh_plain_field,
    predict_camargo_plain_field,
    save_arrival_eval,
    save_pf_metrics,
    save_pf_plot,
)
from sos import (
    empirical_arrival_hour_sampler,
    most_frequent_first_activity,
    most_frequent_first_resource,
)
from time_series_preprocessing import ts_splits_from_log

# ── Constants ─────────────────────────────────────────────────────────────────
SYNTH_LOGS = [p.stem for p in sorted((ROOT / "data" / "synthetic").glob("*.xes"))
              if "recency" not in p.stem]
REAL_LOGS  = [p.stem for p in sorted((ROOT / "data" / "real-life").glob("*.xes"))]
REAL_TRIMS = [
    "none", "peak_0.6", "peak_0.7", "peak_0.8",
    "magnitude_1", "magnitude_2", "magnitude_3",
]

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pf_out(ds: str, model: str, trim: str) -> Path:
    return ROOT / "results" / "plain_field" / model / trim / f"{ds}_test_full"


def _already_done(ds: str, model: str, trim: str) -> bool:
    return (_pf_out(ds, model, trim) / f"metrics_{ds}_test_full.csv").exists()


def _bukhsh_both_done(ds: str, trim: str) -> bool:
    return _already_done(ds, "bukhsh_rt", trim) and _already_done(ds, "bukhsh_suffix", trim)


def _amiri_dir(ds: str, trim: str, is_real: bool) -> Path:
    if is_real:
        return BEST / ds / trim / "amiri" / f"{ds}_full"
    return BEST / ds / "amiri" / "none" / f"{ds}_full"


def _bukhsh_dir(ds: str, trim: str, is_real: bool) -> Path:
    if is_real:
        return BEST / ds / trim / "bukhsh" / f"{ds}_test_full"
    return BEST / ds / "bukhsh" / f"{ds}_test_full"


# ── Per-model runners ─────────────────────────────────────────────────────────

def _run_amiri(ds, trim, is_real, train_, val_, test_, sos_, if_, cc_, tt_,
               pred_arr_, arr_s_):
    if _already_done(ds, "amiri", trim):
        print("    [amiri]   skip")
        return
    adir = _amiri_dir(ds, trim, is_real)
    if not adir.exists():
        print("    [amiri]   no model weights — skip")
        return
    from amiri.trainer import AmiriTrainer
    try:
        ta = AmiriTrainer(train_, val_, test_, f"{ds}_test_full",
                          _load_amiri_params(adir), output_dir=adir)
        rt = predict_amiri_plain_field(ta, sos_, if_)
        el = _rem_time_to_event_log(rt)
        out = _pf_out(ds, "amiri", trim)
        save_arrival_eval(pred_arr_, arr_s_, cc_["test"].index, out, ds, trim)
        cc_p, tt_p, m = compute_cc_tt_metrics(el, cc_["test"], tt_["test"])
        save_pf_metrics(ds, "plain_field_amiri", m, out)
        save_pf_plot(cc_p, tt_p, cc_["test"], tt_["test"], out, ds, trim, "plain_field_amiri")
        print(f"    [amiri]   CC MAE={m['cc_mae']:.3f}  TT MAE={m['tt_mae']:.3f}")
    except Exception as e:
        print(f"    [amiri]   ERROR: {e}")
        traceback.print_exc()


def _run_bukhsh(ds, trim, is_real, train_, val_, test_, sos_, if_, cc_, tt_,
                pred_arr_, arr_s_):
    rt_done  = _already_done(ds, "bukhsh_rt", trim) or _already_done(ds, "bukhsh", trim)
    sfx_done = _already_done(ds, "bukhsh_suffix", trim)
    if rt_done and sfx_done:
        print("    [bukhsh]  skip")
        return
    bdir = _bukhsh_dir(ds, trim, is_real)
    if not (bdir / "meta.pkl").exists():
        print("    [bukhsh]  no model weights — skip")
        return
    from bukhsh.params  import default_params as bukhsh_dp
    from bukhsh.trainer import BukhshTrainer
    try:
        tb = BukhshTrainer(train_, val_, test_, ds,
                           bukhsh_dp(epochs=50), output_dir=bdir)
        tb._load_all_models()
        rt, sfx = predict_bukhsh_plain_field(tb, sos_, if_)
        if not rt_done:
            el_rt = _rem_time_to_event_log(rt)
            out_rt = _pf_out(ds, "bukhsh_rt", trim)
            save_arrival_eval(pred_arr_, arr_s_, cc_["test"].index, out_rt, ds, trim)
            cc_p, tt_p, m = compute_cc_tt_metrics(el_rt, cc_["test"], tt_["test"])
            save_pf_metrics(ds, "plain_field_bukhsh_rt", m, out_rt)
            save_pf_plot(cc_p, tt_p, cc_["test"], tt_["test"], out_rt, ds, trim, "plain_field_bukhsh_rt")
            print(f"    [bukhsh RT]     CC MAE={m['cc_mae']:.3f}  TT MAE={m['tt_mae']:.3f}")
        if not sfx_done:
            out_sfx = _pf_out(ds, "bukhsh_suffix", trim)
            save_arrival_eval(pred_arr_, arr_s_, cc_["test"].index, out_sfx, ds, trim)
            cc_ps, tt_ps, ms = compute_cc_tt_metrics(sfx, cc_["test"], tt_["test"])
            save_pf_metrics(ds, "plain_field_bukhsh_suffix", ms, out_sfx)
            save_pf_plot(cc_ps, tt_ps, cc_["test"], tt_["test"], out_sfx, ds, trim, "plain_field_bukhsh_suffix")
            print(f"    [bukhsh suffix] CC MAE={ms['cc_mae']:.3f}  TT MAE={ms['tt_mae']:.3f}")
    except Exception as e:
        print(f"    [bukhsh]  ERROR: {e}")
        traceback.print_exc()


def _run_camargo(ds, trim, is_real, train_, val_, test_, sos_, if_, cc_, tt_,
                 pred_arr_, arr_s_):
    from analysis.ts_comparison import _camargo_summary_and_trim_dir
    cs, resolved_trim_dir = _camargo_summary_and_trim_dir(ds, trim, is_real)
    if cs is None:
        return  # no trained camargo model for this dataset/trim at all
    if _already_done(ds, "camargo", trim):
        print("    [camargo] skip")
        return
    from camargo.params  import default_params as camargo_dp
    from camargo.trainer import CamargoTrainer
    try:
        tc  = CamargoTrainer.from_saved(
            train_, val_, test_, cs["run_name"],
            camargo_dp(cs["run_name"], epochs=50), trim_dir=resolved_trim_dir)
        pl  = predict_camargo_plain_field(tc, sos_, if_)
        out = _pf_out(ds, "camargo", trim)
        save_arrival_eval(pred_arr_, arr_s_, cc_["test"].index, out, ds, trim)
        cc_p, tt_p, m = compute_cc_tt_metrics(pl, cc_["test"], tt_["test"])
        save_pf_metrics(ds, "plain_field_camargo", m, out)
        save_pf_plot(cc_p, tt_p, cc_["test"], tt_["test"], out, ds, trim, "plain_field_camargo")
        print(f"    [camargo] CC MAE={m['cc_mae']:.3f}  TT MAE={m['tt_mae']:.3f}")
    except Exception as e:
        print(f"    [camargo] ERROR: {e}")
        traceback.print_exc()


# ── Single-job entry point (importable) ─────────────────────────────────────────

def run_pf_job(ds: str, trim: str, is_real: bool,
                do_amiri: bool = True, do_bukhsh: bool = True, do_camargo: bool = True,
                skip_if_done: bool = True) -> str:
    """Run plain-field prediction for one (dataset, trim) job.

    Returns "ok", "skipped" (metrics already exist), or "missing" (XES not found).
    Can be called directly (e.g. from a notebook) instead of going through the
    CLI in `main()` below.
    """
    xes = ROOT / "data" / ("real-life" if is_real else "synthetic") / f"{ds}.xes"
    if not xes.exists():
        print(f"  [skip] {ds} / {trim} — XES not found")
        return "missing"

    models_to_run = (
        (["amiri"]                          if do_amiri   else []) +
        (["bukhsh_rt", "bukhsh_suffix"]     if do_bukhsh  else []) +
        (["camargo"]                        if do_camargo else [])
    )
    if skip_if_done and all(
        _already_done(ds, m, trim) or (m == "bukhsh_rt" and _already_done(ds, "bukhsh", trim))
        for m in models_to_run
    ):
        print(f"  [skip] {ds} / {trim} — all done")
        return "skipped"

    print(f"\n  {ds} / {trim}")

    # ── Load & split ───────────────────────────────────────────────────────────
    log_ = pm4py.read_xes(str(xes))
    ts_  = ts_splits_from_log(log_, **{**_BASE, **_TRIM_KW.get(trim, {})})
    cc_  = ts_["concurrent_cases"]
    tt_  = ts_["throughput_time"]

    df_ = load_event_log(xes, time_col="time:timestamp",
                         case_col="case:concept:name")
    df_ = df_.rename(columns=_COLS)
    df_["task"] = df_["task"].fillna("unk")
    df_["user"] = df_.get("user", pd.Series("unk", index=df_.index)).fillna("unk")

    train_, val_, test_ = make_three_way_split(
        df_, case_col="caseid", time_col="end_timestamp",
        train_split=cc_["train_split"], val_split=cc_["val_split"],
        full_traces=True,
    )

    # ── Arrival model ──────────────────────────────────────────────────────────
    arr_s_ = compute_arrival_series(df_)
    vs_    = pd.Timestamp(cc_["val_split"])
    if vs_.tzinfo:
        vs_ = vs_.tz_convert(None)
    am_ = ProphetArrivalModel()
    am_.fit(arr_s_[arr_s_.index < vs_])
    pred_arr_ = am_.predict(cc_["test"].index)

    # ── SOS + in-flight ────────────────────────────────────────────────────────
    fa_  = most_frequent_first_activity(train_)
    fr_  = most_frequent_first_resource(train_)
    hs_  = empirical_arrival_hour_sampler(train_)
    sos_ = build_sos_cases(pred_arr_, fa_, fr_, hs_)
    if_  = get_inflight_cases(df_, cc_["val_split"])

    n_sos = len(sos_)
    n_if  = if_["caseid"].nunique()
    print(f"    {n_sos} SOS + {n_if} in-flight")

    # ── Models ─────────────────────────────────────────────────────────────────
    if do_amiri:
        _run_amiri(ds, trim, is_real, train_, val_, test_,
                   sos_, if_, cc_, tt_, pred_arr_, arr_s_)
    if do_bukhsh:
        _run_bukhsh(ds, trim, is_real, train_, val_, test_,
                    sos_, if_, cc_, tt_, pred_arr_, arr_s_)
    if do_camargo:
        _run_camargo(ds, trim, is_real, train_, val_, test_,
                     sos_, if_, cc_, tt_, pred_arr_, arr_s_)

    return "ok"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plain-field predictions (Amiri / Bukhsh / Camargo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python plain-field/pf_lukas_prediction.py
  python plain-field/pf_lukas_prediction.py --dataset loan_flat
  python plain-field/pf_lukas_prediction.py --trim peak_0.6
  python plain-field/pf_lukas_prediction.py --model amiri
  python plain-field/pf_lukas_prediction.py --dataset bpic15-1 --trim none --model amiri
  python plain-field/pf_lukas_prediction.py --real-only
  python plain-field/pf_lukas_prediction.py --synth-only
        """,
    )
    parser.add_argument("--dataset",    default=None, help="Run only this dataset")
    parser.add_argument("--trim",       default=None, help="Run only this trim config")
    parser.add_argument("--model",      default=None, choices=["amiri", "bukhsh", "camargo"],
                        help="Run only this model (default: all)")
    parser.add_argument("--real-only",  action="store_true", help="Skip synthetic logs")
    parser.add_argument("--synth-only", action="store_true", help="Skip real-life logs")
    args = parser.parse_args()

    # ── Build job list ─────────────────────────────────────────────────────────
    jobs: list[tuple[str, str, bool]] = []
    if not args.real_only:
        for ds in SYNTH_LOGS:
            if args.dataset and ds != args.dataset:
                continue
            if args.trim and args.trim != "none":
                continue
            jobs.append((ds, "none", False))
    if not args.synth_only:
        for ds in REAL_LOGS:
            if args.dataset and ds != args.dataset:
                continue
            for trim in REAL_TRIMS:
                if args.trim and trim != args.trim:
                    continue
                jobs.append((ds, trim, True))

    if not jobs:
        print("No jobs match the specified filters.")
        return

    do_amiri   = args.model in (None, "amiri")
    do_bukhsh  = args.model in (None, "bukhsh")
    do_camargo = args.model in (None, "camargo")

    print(f"{len(jobs)} job(s) to process.\n")
    ok = skipped = errors = 0

    for ds, trim, is_real in jobs:
        try:
            status = run_pf_job(ds, trim, is_real, do_amiri, do_bukhsh, do_camargo)
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            # "missing" (XES not found) is neither counted nor an error
        except Exception as e:
            print(f"  [ERROR] {ds} / {trim}: {e}")
            traceback.print_exc()
            errors += 1

    print(f"\nDone: {ok}  Skipped: {skipped}  Errors: {errors}")


if __name__ == "__main__":
    main()
