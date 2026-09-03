"""
run_predictions_real.py
=======================

"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pm4py
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
from time_series_preprocessing import ts_splits_from_log

# =============================================================================
# Paths and constants
# =============================================================================

BEST_MODELS = ROOT / 'best_models'
DATA_DIR    = ROOT / 'data' / 'real-life'
RESULTS_DIR = ROOT / 'results'
TRIM_WINDOW = 7

# Column rename maps for pm4py → internal names.
# Some logs use org:group instead of org:resource — both are handled.
_COLS: dict[str, str] = {
    'case:concept:name':    'caseid',
    'concept:name':         'task',
    'lifecycle:transition': 'event_type',
    'org:resource':         'user',
    'time:timestamp':       'end_timestamp',
}
_COLS_GROUP: dict[str, str] = {
    'case:concept:name':    'caseid',
    'concept:name':         'task',
    'lifecycle:transition': 'event_type',
    'org:group':            'user',
    'time:timestamp':       'end_timestamp',
}

# =============================================================================
# Utility helpers
# =============================================================================

def _parse_trim(name: str) -> dict:
    """Convert a trim folder name to ts_splits_from_log kwargs."""
    base = dict(trim_pct=0.25, trim_k=1.5, trim_frac=0.60, trim_window=TRIM_WINDOW)
    if name == 'none':
        return {**base, 'trim_method': None}
    method, val = name.rsplit('_', 1)
    val = float(val)
    if method == 'magnitude':
        return {**base, 'trim_method': 'magnitude', 'trim_k': val}
    if method == 'peak':
        return {**base, 'trim_method': 'peak', 'trim_frac': val}
    if method == 'pct':
        return {**base, 'trim_method': 'pct', 'trim_pct': val}
    raise ValueError(f'Unknown trim folder name: {name!r}')


def _kpi_series(event_log: pd.DataFrame, cc_index, tt_index):
    """Compute CC and TT arrays from a predicted event log, aligned to test index."""
    pred_cc = create_concurrent_cases_timeseries(
        event_log, time_col='end_timestamp', case_col='caseid', window='days', plot=False)
    pred_tt = create_avg_throughtput_time_timeseries(
        event_log, time_col='end_timestamp', case_col='caseid', window='days', plot=False)
    cc_arr = pred_cc.reindex(cc_index).ffill().bfill().fillna(0).to_numpy()
    tt_arr = pred_tt.reindex(tt_index).ffill().bfill().fillna(0).to_numpy()
    return cc_arr, tt_arr


def _save_results(
    cc_actual, cc_pred, tt_actual, tt_pred,
    cc_index, tt_index,
    run_name: str,
    model_label: str,
    out_dir: Path,
    predict_s: float,
    params_json: str = '{}',
    metrics_suffix: str = '',
) -> dict:
    """Save metrics CSV, time CSV, and plots to out_dir. Returns results dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        'concurrent_cases': {
            'mse': mean_squared_error(cc_actual, cc_pred),
            'mae': mean_absolute_error(cc_actual, cc_pred),
        },
        'throughput_time': {
            'mse': mean_squared_error(tt_actual, tt_pred),
            'mae': mean_absolute_error(tt_actual, tt_pred),
        },
    }

    pd.DataFrame([
        dict(dataset=run_name, series=s, model=model_label, mse=m['mse'], mae=m['mae'])
        for s, m in results.items()
    ]).to_csv(out_dir / f'metrics_{run_name}{metrics_suffix}.csv', index=False)

    # Time CSV and plots only for the primary (non-RT) metrics to avoid duplication.
    if not metrics_suffix:
        pd.DataFrame([
            dict(model=model_label, phase='predict', params_json=params_json,
                 val_mse=float('nan'), time_s=predict_s, is_best=True,
                 dataset=run_name, series='all'),
        ]).to_csv(out_dir / f'time_{run_name}.csv', index=False)

        for series_name, actual, pred, index in [
            ('concurrent_cases', cc_actual, cc_pred, cc_index),
            ('throughput_time',  tt_actual, tt_pred, tt_index),
        ]:
            m = results[series_name]
            fig, ax = plt.subplots(figsize=(13, 4))
            ax.plot(index, actual, color='green',  label='actual',                linewidth=1.5)
            ax.plot(index, pred,   color='purple', label=f'{model_label} (pred)', linestyle='--')
            ax.set_title(f'{run_name} — {series_name}  MSE={m["mse"]:.4f}  MAE={m["mae"]:.4f}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f'{series_name}.png', dpi=150, bbox_inches='tight')
            plt.close()

    for s, m in results.items():
        print(f'      {s:<22}  MSE={m["mse"]:.4f}  MAE={m["mae"]:.4f}')
    return results


def _build_rt_log(rem_time_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a 2-event-per-case event log from rem_time predictions.

    Each case contributes a start event (first test event) and an end event
    (anchor_timestamp + rem_time_days). This is the simplest representation of
    the remaining-time prediction that can be fed to the KPI series functions.
    """
    rt_df = rem_time_df.copy()
    if 'anchor_timestamp' in rt_df.columns:
        rt_df['start_timestamp']  = pd.to_datetime(rt_df['start_timestamp'])
        rt_df['anchor_timestamp'] = pd.to_datetime(rt_df['anchor_timestamp'])
    else:
        # Older BukhshTrainer versions may not save these columns → derive from test_df.
        _ts     = pd.to_datetime(test_df['end_timestamp'], utc=True).dt.tz_convert(None)
        _tdf    = test_df.assign(_ts=_ts).sort_values(['caseid', '_ts'])
        _start  = _tdf.groupby('caseid')['_ts'].first().rename('start_timestamp').reset_index()
        _anchor = _tdf.groupby('caseid')['_ts'].last().rename('anchor_timestamp').reset_index()
        rt_df   = rt_df.merge(_start, on='caseid').merge(_anchor, on='caseid')

    rt_df['predicted_end'] = (rt_df['anchor_timestamp']
                              + pd.to_timedelta(rt_df['rem_time_days'], unit='D'))
    return pd.concat([
        rt_df[['caseid', 'start_timestamp']].rename(columns={'start_timestamp': 'end_timestamp'}),
        rt_df[['caseid', 'predicted_end']].rename(columns={'predicted_end': 'end_timestamp'}),
    ], ignore_index=True)


def _make_half_prefix_test_df(df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Return test cases truncated to ceil(N/2) events each."""
    test_cids = set(test_df['caseid'].astype(str))
    subset = (df[df['caseid'].astype(str).isin(test_cids)]
              .copy()
              .sort_values(['caseid', 'end_timestamp']))
    parts = [grp.iloc[: max(1, math.ceil(len(grp) / 2))]
             for _, grp in subset.groupby('caseid', sort=False)]
    return pd.concat(parts, ignore_index=True)


def _setup_half_bukhsh_dir(full_dir: Path) -> Path:
    """
    Create half_prefix/ subdir under full_dir and link the files that
    BukhshTrainer.predict() needs: meta.pkl (has absolute weight paths),
    train.csv, and vocab_ref.csv (needed by _build_role_map).

    Falls back to shutil.copy2 on systems where symlinks fail (e.g. some
    Windows configurations).
    """
    half_dir = full_dir / 'half_prefix'
    half_dir.mkdir(exist_ok=True)
    for fname in ('meta.pkl', 'train.csv', 'vocab_ref.csv'):
        src = (full_dir / fname).resolve()
        dst = half_dir / fname
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
    return half_dir


# =============================================================================
# Data loading
# =============================================================================

def _load_data(xes_path: Path, trim_name: str):
    """
    Load an XES log, compute time-series splits and 3-way event-log splits.

    Returns
    -------
    df          : full event DataFrame (all cases)
    cc          : concurrent-cases split dict (train / val / test / train_split / val_split)
    tt          : throughput-time split dict
    train_df    : event-log training partition
    val_df      : event-log validation partition
    test_df     : event-log test partition (full traces)
    """
    trim_kw = _parse_trim(trim_name)
    log = pm4py.read_xes(str(xes_path))

    ts = ts_splits_from_log(log, cut_date=None, **trim_kw)
    cc = ts['concurrent_cases']
    tt = ts['throughput_time']

    df = pm4py.convert_to_dataframe(log)
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)
    df = df.dropna(subset=['case:concept:name'])
    df = df.rename(columns=_COLS if 'org:resource' in df.columns else _COLS_GROUP)
    df['task'] = df['task'].fillna('unk')
    df['user'] = df['user'].fillna('unk')

    train_df, val_df, test_df = make_three_way_split(
        df, case_col='caseid', time_col='end_timestamp',
        train_split=cc['train_split'], val_split=cc['val_split'],
        full_traces=True,
    )
    return df, cc, tt, train_df, val_df, test_df


# =============================================================================
# Model discovery
# =============================================================================

def _discover_entries(
    dataset_filter: str | None = None,
    trim_filter: str | None = None,
) -> list[dict]:
    """
    Return one dict per (dataset, trim) pair that has at least one trained model.

    Only considers real-life models (best_models/{dataset}/{trim}/{model}/),
    i.e. 2-level-deep paths from BEST_MODELS. Synthetic models have a different
    layout (best_models/{dataset}/{model}/) and are intentionally excluded.

    Each dict has keys:
      dataset, trim, xes, bukhsh_bmd (Path or None), camargo_summary (Path or None)
    """
    entries: dict[tuple, dict] = {}

    for bp in sorted(BEST_MODELS.glob('*/*/bukhsh/best_params.json')):
        # bp: best_models/{dataset}/{trim}/bukhsh/best_params.json
        trim    = bp.parent.parent.name
        dataset = bp.parent.parent.parent.name
        xes     = DATA_DIR / f'{dataset}.xes'
        if not xes.exists():
            continue
        if dataset_filter and dataset != dataset_filter:
            continue
        if trim_filter and trim != trim_filter:
            continue
        key = (dataset, trim)
        if key not in entries:
            entries[key] = dict(dataset=dataset, trim=trim, xes=xes,
                                bukhsh_bmd=None, camargo_summary=None)
        entries[key]['bukhsh_bmd'] = bp.parent  # .../bukhsh/

    for sp in sorted(BEST_MODELS.glob('*/*/camargo/hpo_summary.json')):
        trim    = sp.parent.parent.name
        dataset = sp.parent.parent.parent.name
        xes     = DATA_DIR / f'{dataset}.xes'
        if not xes.exists():
            continue
        if dataset_filter and dataset != dataset_filter:
            continue
        if trim_filter and trim != trim_filter:
            continue
        key = (dataset, trim)
        if key not in entries:
            entries[key] = dict(dataset=dataset, trim=trim, xes=xes,
                                bukhsh_bmd=None, camargo_summary=None)
        entries[key]['camargo_summary'] = sp

    return list(entries.values())


# =============================================================================
# Per-model runners
# =============================================================================

def _run_bukhsh(
    mode: str,          # 'full' or 'half'
    bmd: Path,          # best_models/{ds}/{trim}/bukhsh/
    run_name: str,      # e.g. 'helpdesk_test_full'
    trim_name: str,
    best_params: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    df_full: pd.DataFrame,  # entire log (needed for half-prefix truncation)
    cc: dict,
    tt: dict,
) -> bool:
    """
    Run Bukhsh predictions for one mode. Returns True if skipped (already done).

    Saves raw logs to:
      mode=full  →  bmd/{run_name}/event_log.csv  and  rem_time.csv
      mode=half  →  bmd/{run_name}/half_prefix/event_log.csv  and  rem_time.csv

    Saves KPI metrics to:
      results/bukhsh_hpo{_half}/{trim}/{run_name}/metrics_{run_name}.csv      (suffix)
      results/bukhsh_hpo{_half}/{trim}/{run_name}/metrics_{run_name}_rt.csv   (RT)
    """
    results_dir = 'bukhsh_hpo' if mode == 'full' else 'bukhsh_hpo_half'
    out_dir = RESULTS_DIR / results_dir / trim_name / run_name

    already_done = (
        (out_dir / f'metrics_{run_name}.csv').exists()
        and (out_dir / f'metrics_{run_name}_rt.csv').exists()
    )
    if already_done:
        print(f'  [Bukhsh/{mode}] SKIP — metrics already exist')
        return True

    if mode == 'half':
        test_df_use = _make_half_prefix_test_df(df_full, test_df)
        model_dir   = _setup_half_bukhsh_dir(bmd / run_name)
    else:
        test_df_use = test_df
        model_dir   = bmd / run_name

    params_b  = bukhsh_default_params(**best_params)
    trainer_b = BukhshTrainer(
        train_df, val_df=val_df, test_df=test_df_use,
        run_name=run_name, params=params_b, output_dir=model_dir,
    )

    t0 = time.perf_counter()
    event_log_b, rem_time_df_b = trainer_b.predict()
    predict_s = round(time.perf_counter() - t0, 2)

    event_log_b = event_log_b.copy()
    event_log_b['end_timestamp'] = (
        pd.to_datetime(event_log_b['end_timestamp'], utc=True).dt.tz_convert(None))

    cc_actual = cc['test'].to_numpy()
    tt_actual = tt['test'].to_numpy()

    # --- suffix metrics ---
    cc_pred_s, tt_pred_s = _kpi_series(event_log_b, cc['test'].index, tt['test'].index)
    print(f'  [Bukhsh/{mode}] {predict_s:.1f}s  — suffix metrics:')
    _save_results(
        cc_actual, cc_pred_s, tt_actual, tt_pred_s,
        cc['test'].index, tt['test'].index,
        run_name, results_dir, out_dir, predict_s,
        params_json=json.dumps(best_params),
    )

    # --- RT metrics ---
    rt_log = _build_rt_log(rem_time_df_b, test_df_use)
    cc_pred_rt, tt_pred_rt = _kpi_series(rt_log, cc['test'].index, tt['test'].index)
    print(f'  [Bukhsh/{mode}] RT metrics:')
    _save_results(
        cc_actual, cc_pred_rt, tt_actual, tt_pred_rt,
        cc['test'].index, tt['test'].index,
        run_name, f'{results_dir}_rt', out_dir, predict_s,
        params_json=json.dumps(best_params),
        metrics_suffix='_rt',
    )

    print(f'  [Bukhsh/{mode}] saved → {out_dir}')
    return False


def _run_camargo(
    mode: str,              # 'full' or 'half'
    summary_path: Path,     # best_models/{ds}/{trim}/camargo/hpo_summary.json
    run_name: str,
    trim_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    df_full: pd.DataFrame,
    cc: dict,
    tt: dict,
) -> bool:
    """
    Run Camargo predictions for one mode. Returns True if skipped (already done).

    Raw gen_*.csv files are written by CamargoTrainer.predict() to its internal
    output_files/ directory. This function additionally saves the derived
    event_log.csv to the results folder.

    Saves KPI metrics to:
      results/camargo_hpo{_half}/{trim}/{run_name}/metrics_{run_name}.csv
    """
    results_dir = 'camargo_hpo' if mode == 'full' else 'camargo_hpo_half'
    out_dir = RESULTS_DIR / results_dir / trim_name / run_name

    if (out_dir / f'metrics_{run_name}.csv').exists():
        print(f'  [Camargo/{mode}] SKIP — metrics already exist')
        return True

    summary    = json.loads(summary_path.read_text())
    run_name_c = summary['run_name']
    trim_dir_c = summary['trim_dir']

    test_df_use = (_make_half_prefix_test_df(df_full, test_df)
                   if mode == 'half' else test_df)

    params_c  = camargo_params(run_name_c, max_eval=1, epochs=200)
    trainer_c = CamargoTrainer.from_saved(
        train_df, val_df, test_df_use,
        run_name_c, params_c, trim_dir=trim_dir_c,
    )

    t0 = time.perf_counter()

    if mode == 'half':
        # Full and half modes otherwise reuse the exact same
        # output_files/<trim_dir_c>/<run_name_c>/ directory (same trained
        # model, both read trim_dir/run_name straight from hpo_summary.json)
        # -- the top-level, unqualified event_log.csv there is the FULL
        # regime's own file (load_ppm_raw_predictions's full-regime lookup
        # depends on exactly that path). Point half's prediction at a
        # separate <run_name_c>_half/ directory instead, mirroring
        # regen_camargo_predictions.py's regenerate_half(), so half's own
        # predict()/to_event_log() writes can never clobber full's.
        from analysis.ts_comparison import GLSTM
        orig_name, orig_out = trainer_c.run_name, trainer_c.output_dir
        new_name = orig_name + '_half'
        orig_out_abs = GLSTM / orig_out
        new_out_abs = orig_out_abs.parent / new_name
        new_params = new_out_abs / 'parameters'
        new_params.mkdir(parents=True, exist_ok=True)
        orig_params = orig_out_abs / 'parameters'
        for fname in ('model_parameters.json', 'resource_map.csv'):
            src = orig_params / fname
            if src.exists():
                shutil.copy2(src, new_params / fname)
        for h5 in orig_out_abs.glob('*.h5'):
            shutil.copy2(h5, new_out_abs / h5.name)

        trainer_c.run_name = new_name
        trainer_c.output_dir = str(Path(orig_out).parent / new_name)
        try:
            pred_paths  = trainer_c.predict()
            event_log_c = trainer_c.to_event_log(pred_paths)
        finally:
            trainer_c.run_name, trainer_c.output_dir = orig_name, orig_out
    else:
        pred_paths  = trainer_c.predict()
        event_log_c = trainer_c.to_event_log(pred_paths)

    predict_s = round(time.perf_counter() - t0, 2)

    cc_actual = cc['test'].to_numpy()
    tt_actual = tt['test'].to_numpy()

    cc_pred, tt_pred = _kpi_series(event_log_c, cc['test'].index, tt['test'].index)
    print(f'  [Camargo/{mode}] {predict_s:.1f}s  — suffix metrics:')
    _save_results(
        cc_actual, cc_pred, tt_actual, tt_pred,
        cc['test'].index, tt['test'].index,
        run_name, results_dir, out_dir, predict_s,
    )

    # Save event_log so it can be reloaded without re-running predict().
    out_dir.mkdir(parents=True, exist_ok=True)
    event_log_c.to_csv(out_dir / f'event_log_{run_name}.csv', index=False)

    print(f'  [Camargo/{mode}] saved → {out_dir}')
    return False


# =============================================================================
# Main loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate Bukhsh + Camargo HPO predictions for real-life models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_predictions_real.py                           # all models
  python run_predictions_real.py --dataset helpdesk        # one dataset
  python run_predictions_real.py --trim peak_0.6           # one trim config
  python run_predictions_real.py --model bukhsh            # only Bukhsh
  python run_predictions_real.py --dataset bpic20-dom --trim none
        """,
    )
    parser.add_argument('--dataset', default=None,
                        help='Process only this dataset (e.g. helpdesk)')
    parser.add_argument('--trim', default=None,
                        help='Process only this trim config (e.g. peak_0.6)')
    parser.add_argument('--model', default=None, choices=['bukhsh', 'camargo'],
                        help='Process only this model type (default: both)')
    args = parser.parse_args()

    set_global_seed(1904)

    entries = _discover_entries(
        dataset_filter=args.dataset,
        trim_filter=args.trim,
    )

    if not entries:
        print('No trained real-life models found matching the specified filters.')
        print(f'Searched in: {BEST_MODELS}')
        print('Expected path layout: best_models/{dataset}/{trim}/bukhsh/ or .../camargo/')
        return

    do_bukhsh  = args.model in (None, 'bukhsh')
    do_camargo = args.model in (None, 'camargo')

    print(f'Found {len(entries)} (dataset, trim) pair(s).\n')

    for entry in entries:
        dataset  = entry['dataset']
        trim     = entry['trim']
        xes_path = entry['xes']
        bmd      = entry['bukhsh_bmd']      # Path or None
        summary  = entry['camargo_summary']  # Path or None

        run_name = f'{dataset}_test_full'

        has_bukhsh  = bmd is not None
        has_camargo = summary is not None

        print(f'\n{"="*60}')
        print(f'  {dataset}  |  trim={trim}  |  run={run_name}')
        print(f'  bukhsh={has_bukhsh}  camargo={has_camargo}')
        print(f'{"="*60}')

        try:
            df, cc, tt, train_df, val_df, test_df = _load_data(xes_path, trim)
        except Exception:
            print('[ERROR] Failed to load data for this entry:')
            traceback.print_exc()
            continue

        print(f'  train={train_df["caseid"].nunique()}'
              f'  val={val_df["caseid"].nunique()}'
              f'  test={test_df["caseid"].nunique()} cases')

        for mode in ('full', 'half'):
            print(f'\n  --- mode: {mode} ---')

            if do_bukhsh and has_bukhsh:
                try:
                    best_params = json.loads((bmd / 'best_params.json').read_text())
                    _run_bukhsh(
                        mode, bmd, run_name, trim, best_params,
                        train_df, val_df, test_df, df, cc, tt,
                    )
                except Exception:
                    print(f'[ERROR] Bukhsh/{mode} failed:')
                    traceback.print_exc()

            if do_camargo and has_camargo:
                try:
                    _run_camargo(
                        mode, summary, run_name, trim,
                        train_df, val_df, test_df, df, cc, tt,
                    )
                except Exception:
                    print(f'[ERROR] Camargo/{mode} failed:')
                    traceback.print_exc()


if __name__ == '__main__':
    main()
