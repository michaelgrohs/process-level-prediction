"""
CamargoTrainer — file-based bridge between our train/val/test splits and the
GenerativeLSTM (Camargo et al.) pipeline.

Usage
-----
from camargo.trainer import CamargoTrainer
from camargo.params  import default_params

params  = default_params("bpic12_val_full", max_eval=5)
trainer = CamargoTrainer(train_df, val_df, test_df, "bpic12_val_full", params)
trainer.run()                       # preprocess + HPO train
pred_path = trainer.predict()       # suffix prediction on test_df
"""
import ast
import copy
import csv
import json
import re
import os
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import traceback

import numpy as np
import pandas as pd


_NA_VALUES_NO_NA = {
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NULL", "NaN", "None",
    "n/a", "nan", "null",
}

# ── Repo path setup ────────────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent / "GenerativeLSTM" / "GenerativeLSTM"

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# Evict broken PyPI 'readers' package so GLSTM's local modules win.
for _m in [m for m in sys.modules if m == "readers" or m.startswith("readers.")]:
    del sys.modules[_m]

# ── GLSTM imports (resolved after sys.path is set) ────────────────────────────
from model_training import embedding_training as em            # noqa: E402
from model_training.features_manager import FeaturesMannager  # noqa: E402
from model_prediction.model_predictor import ModelPredictor   # noqa: E402
import utils.support as sup                                    # noqa: E402

from camargo.optimizer import CamargoOptimizer                # noqa: E402
from hpo_val_scoring import kpi_mae                            # noqa: E402


def _shift_list(lst):
    """Shift a time list so it starts at 0 (fixes negative-leading offsets)."""
    if not lst:
        return lst
    first = int(lst[0])
    return [x - first for x in lst] if first < 0 else lst


def _cumulate_list(lst):
    if not lst:
        return lst
    total, result = 0, []
    for x in lst:
        if x is not None:
            total += x
        result.append(total)
    return result


@contextmanager
def _repo_cwd():
    """Temporarily change cwd to REPO_DIR so GLSTM relative paths resolve."""
    old = os.getcwd()
    os.chdir(str(REPO_DIR))
    try:
        yield
    finally:
        os.chdir(old)


class CamargoTrainer:
    """
    Orchestrates the full GenerativeLSTM pipeline using our canonical
    train / val / test splits.

    Steps
    -----
    1. preprocess()  — role assignment, activity/role indexing, embeddings
    2. train()       — Bayesian HPO on (train, val); exports model + test_log.csv
    3. predict()     — suffix prediction on test_df; returns prediction CSV path(s)

    The output folder inside the GLSTM repo is:
        GenerativeLSTM/GenerativeLSTM/output_files/<trim_dir>/<run_name>/
    where trim_dir is empty string → folder collapses to output_files/<run_name>/.
    """

    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                 test_df: pd.DataFrame, run_name: str, params: dict,
                 trim_dir: str = "", val_as_test_df: pd.DataFrame | None = None,
                 cc_val: pd.Series | None = None, tt_val: pd.Series | None = None):
        self.train_df = train_df.copy()
        self.val_df   = val_df.copy()
        self.test_df  = test_df.copy()
        self.run_name = run_name
        self.trim_dir = trim_dir   # e.g. "pct_0.25", "peak_0.6", ""
        self.params   = copy.deepcopy(params)

        self.val_as_test_df = val_as_test_df.copy() if val_as_test_df is not None else None
        self.cc_val = cc_val
        self.tt_val = tt_val

        self.ac_index:  dict  = {}
        self.index_ac:  dict  = {}
        self.rl_index:  dict  = {}
        self.index_rl:  dict  = {}
        self.ac_weights       = np.array([])
        self.rl_weights       = np.array([])

        self.output_dir:      str | None = None
        self.best_model_path: str | None = None
        self.best_loss:       float      = float("inf")

        if self.has_trained_model():
            self.load_trained_model()

    def _out_base(self) -> str:
        """output_files[/trim_dir]/run_name — relative path inside REPO_DIR."""
        parts = ["output_files"]
        if self.trim_dir:
            parts.append(self.trim_dir)
        parts.append(self.run_name)
        return os.path.join(*parts)

    # ── Public API ─────────────────────────────────────────────────────────────

    @classmethod
    def from_saved(
        cls,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        run_name: str,
        params: dict,
        trim_dir: str = "",
    ) -> "CamargoTrainer":
        """
        Reconstruct a trainer for prediction only, loading indexes and model
        path from a previously saved run.  Skips preprocess() and train().

        The DataFrames must still be provided (they are needed to build the
        prediction prefix and to anchor timestamps in to_event_log()).
        Everything else — ac_index / rl_index / best_model_path — is read from
        output_files/<trim_dir>/<run_name>/parameters/model_parameters.json.

        Usage
        -----
        trainer = CamargoTrainer.from_saved(train_df, val_df, test_df,
                                             RUN_NAME, params, trim_dir="pct_0.25")
        pred_paths = trainer.predict(...)
        """
        trainer = cls(train_df, val_df, test_df, run_name, params, trim_dir=trim_dir)
        with _repo_cwd():
            params_dir  = os.path.join(trainer._out_base(), "parameters")
            params_path = os.path.join(params_dir, "model_parameters.json")
            with open(params_path) as f:
                saved = json.load(f)

            # JSON keys are always strings; restore integer keys for index_ac/rl.
            trainer.index_ac = {int(k): v for k, v in saved["index_ac"].items()}
            trainer.ac_index = {v: int(k) for k, v in saved["index_ac"].items()}
            trainer.index_rl = {int(k): v for k, v in saved["index_rl"].items()}
            trainer.rl_index = {v: int(k) for k, v in saved["index_rl"].items()}

            trainer.output_dir      = trainer._out_base()
            trainer.best_model_path = trainer._find_model_file(trainer.output_dir)
            trainer.best_loss       = saved.get("best_loss", float("inf"))

            # Restore one_timestamp from the saved run so to_event_log() reads
            # the correct time columns (tm_pred vs dur_pred/wait_pred).
            if "one_timestamp" in saved:
                trainer.params["one_timestamp"] = saved["one_timestamp"]
                trainer.params["read_options"]["one_timestamp"] = saved["one_timestamp"]

            # Apply the saved user→role mapping and index columns to all splits,
            # replacing the role-discovery + _apply_indexes() calls in preprocess().
            resource_map_path = os.path.join(params_dir, "resource_map.csv")
            resources = pd.read_csv(resource_map_path)
            trainer._resource_map = resources
            for attr in ("train_df", "val_df", "test_df"):
                df = cls._apply_roles(getattr(trainer, attr), resources)
                df["ac_index"] = df["task"].map(trainer.ac_index).fillna(0).astype(int)
                df["rl_index"] = df["role"].map(trainer.rl_index).fillna(0).astype(int)
                setattr(trainer, attr, df)

        print(f"[from_saved] model  → {trainer.best_model_path}")
        print(f"[from_saved] activities: {len(trainer.ac_index)}  roles: {len(trainer.rl_index)}")
        return trainer

    def run(self):
        """Convenience: preprocess then train."""
        self.preprocess()
        self.train()

    def preprocess(self):
        """Add roles, build indexes, train/load embeddings. Runs in REPO_DIR."""
        for attr in ("train_df", "val_df", "test_df"):
            df = getattr(self, attr)
            if "start_timestamp" not in df.columns:
                df = df.copy()
                df["start_timestamp"] = df["end_timestamp"]
                setattr(self, attr, df)
        with _repo_cwd():
            self._add_roles()
            self._build_indexes()
            self._apply_indexes()
            self._load_or_train_embeddings()

    def has_trained_model(self) -> bool:
        """Return True if a trained model already exists on disk for this run."""
        try:
            self._find_model_file(str(REPO_DIR / self._out_base()))
            return True
        except FileNotFoundError:
            return False

    def load_trained_model(self) -> None:
        """Load an existing trained model from disk without retraining."""
        out_base = REPO_DIR / self._out_base()
        params_path = out_base / "parameters" / "model_parameters.json"
        with open(params_path) as f:
            saved = json.load(f)
        self.best_loss = saved.get("best_loss", float("inf"))
        self.output_dir = str(self._out_base())
        self.best_model_path = self._find_model_file(str(out_base))
        print(f"[load_trained_model] model → {self.best_model_path}")
        print(f"[load_trained_model] loss  → {self.best_loss:.4f}")

    def train(self):
        """Bayesian HPO on (train_df, val_df). Exports best model and test_log.csv."""
        with _repo_cwd():
            self.params["output"] = os.path.join("output_files", sup.folder_id())

            # Persistent per-trial checkpoint directory alongside the named output.
            hpo_dir = str(REPO_DIR / (self._out_base() + "_trials"))

            optimizer = CamargoOptimizer(
                self.params,
                self.train_df, self.val_df,
                self.ac_index, self.ac_weights,
                self.rl_index, self.rl_weights,
                hpo_dir=hpo_dir,
            )
            _t0 = time.perf_counter()
            optimizer.execute_trials()
            total_train_time_s = time.perf_counter() - _t0
            finish_time = datetime.now(timezone.utc).isoformat()

            if optimizer.best_output is None:
                raise RuntimeError("HPO produced no successful trial.")

            # ── Winner selection ────────────────────────────────────────────
            winner_output = optimizer.best_output
            winner_params = optimizer.best_params
            winner_loss   = optimizer.best_loss
            selection_metric = "vectorized_next_event_loss"

            if self.val_as_test_df is not None and self.cc_val is not None and self.tt_val is not None:
                rescored = self._select_trial_by_val_cc_mae(hpo_dir)
                if rescored is not None:
                    winner_output, winner_params, winner_loss = rescored
                    selection_metric = "val_cc_mae"
                    print(f"[train] HPO winner selected by real val CC MAE = {winner_loss:.4f} "
                          f"(vectorized-loss winner would have scored differently -- see rescore log above)")
                else:
                    print("[train] WARNING: val-CC-MAE rescoring produced no usable trial "
                          "(e.g. no per-trial checkpoints on disk) -- falling back to the "
                          "vectorized next-event loss winner.")
            else:
                print("[train] WARNING: val_as_test_df/cc_val/tt_val not provided to "
                      "CamargoTrainer -- selecting the HPO winner by vectorized next-event "
                      "loss, NOT by validation-period CC MAE. This violates the project's "
                      "'equal chance for all HPOs' requirement (bukhsh's own HPO already "
                      "selects by val CC MAE). Pass these three at construction time for "
                      "the correct, fair selection.")

            # Copy winning trial folder → named output directory
            self.output_dir = self._out_base()
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
            shutil.copytree(winner_output, self.output_dir)
            shutil.copy(optimizer.file_name, self.output_dir)

            # Remove temp HPO folder
            shutil.rmtree(self.params["output"], ignore_errors=True)

            winner_params["dataset"]            = self.run_name
            winner_params["finish_time"]        = finish_time
            winner_params["total_train_time_s"] = round(total_train_time_s, 2)
            winner_params["best_loss"]          = winner_loss
            winner_params["selection_metric"]   = selection_metric

            self._export_params(self.output_dir, winner_params)
            self.best_model_path = self._find_model_file(self.output_dir)
            self.best_loss = winner_loss
            print(f"[train] best model → {self.best_model_path}")
            print(f"[train] total time  {total_train_time_s:.1f}s  finished {finish_time}")

    def _select_trial_by_val_cc_mae(self, hpo_dir: str):
        """Score every completed HPO trial checkpoint under hpo_dir on
        self.val_as_test_df and return the lowest-val-CC-MAE trial as
        (output_dir, params_dict, cc_mae) -- the same shape train() expects
        from optimizer.best_output/best_params/best_loss. Returns None if no
        trial could be scored (e.g. hpo_dir has no persisted checkpoints).

        """
        trial_dirs = sorted(Path(hpo_dir).glob("trial_*"))
        if not trial_dirs:
            return None

        val_prepped = self.val_as_test_df.copy()
        if "start_timestamp" not in val_prepped.columns:
            val_prepped["start_timestamp"] = val_prepped["end_timestamp"]
        val_prepped = self._apply_roles(val_prepped, self._resource_map)
        val_prepped["ac_index"] = val_prepped["task"].map(self.ac_index).fillna(0).astype(int)
        val_prepped["rl_index"] = val_prepped["role"].map(self.rl_index).fillna(0).astype(int)

        rescore_trim_dir = (f"{self.trim_dir}/{self.run_name}_trials_rescore"
                             if self.trim_dir else f"{self.run_name}_trials_rescore")

        best = None  # (cc_mae, output_dir, params_dict)
        for tdir in trial_dirs:
            result_path = tdir / "result.json"
            h5_files = list(tdir.glob("*.h5"))
            if not result_path.exists() or not h5_files:
                continue
            trial_result = json.loads(result_path.read_text())
            trial_params = {k: trial_result.get(k) for k in
                            ("model_type", "n_size", "l_size", "lstm_act",
                             "dense_act", "norm_method", "optim")}
            if "scale_args" not in trial_result:
                print(f"  [rescore {tdir.name}] skipped: no scale_args in result.json")
                continue
            trial_params["scale_args"] = trial_result["scale_args"]

            scratch = copy.copy(self)
            scratch.run_name = f"{self.run_name}_hpo_rescore_{tdir.name}"
            scratch.trim_dir = rescore_trim_dir
            scratch.test_df = val_prepped
            scratch.best_model_path = str(h5_files[0])
            scratch.output_dir = scratch._out_base()

            try:
                event_log_path = Path(scratch.output_dir) / "event_log.csv"
                if event_log_path.exists():
                    event_log = self._read_csv_safe_caseid(event_log_path)
                    print(f"  [rescore {tdir.name}] reusing cached event_log.csv")
                else:
                    scratch._export_params(scratch.output_dir, dict(trial_params))
                    h5_link = Path(scratch.output_dir) / Path(h5_files[0]).name
                    if not h5_link.exists():
                        h5_link.symlink_to(Path(h5_files[0]).resolve())
                    pred_paths = scratch.predict(full_prefix_only=True)
                    event_log = scratch.to_event_log(pred_paths)
                event_log["end_timestamp"] = pd.to_datetime(
                    event_log["end_timestamp"], utc=True, format="mixed").dt.tz_convert(None)
                cc_mae, tt_mae = kpi_mae(event_log, self.cc_val, self.tt_val)
            except Exception as exc:
                print(f"  [rescore {tdir.name}] FAILED: {exc}")
                continue

            print(f"  [rescore {tdir.name}] val_cc_mae={cc_mae:.4f}  val_tt_mae={tt_mae:.4f}")
            if best is None or cc_mae < best[0]:
                params_out = dict(trial_params)
                params_out["val_cc_mae"] = round(float(cc_mae), 4)
                params_out["val_tt_mae"] = round(float(tt_mae), 4)
                best = (cc_mae, str(tdir), params_out)

        if best is None:
            return None
        cc_mae, output_dir, params_out = best
        return output_dir, params_out, cc_mae

    @staticmethod
    def _read_csv_safe_caseid(path):
        """caseid-preserving CSV read (avoids pandas' default 'NA' string ->
        NaN coercion, which silently corrupts real case ids like sepsis's
        case 'NA'). Matches analysis/ts_comparison.py's _read_csv_safe_caseid."""
        return pd.read_csv(path, keep_default_na=False, na_values=_NA_VALUES_NO_NA,
                            dtype={"caseid": str})

    def predict(
        self,
        variant: str = "random_choice",
        reps: int = 1,
        prefix_mode: str = "start_plus_first",
        full_prefix_only: bool = False,
    ) -> list[Path]:
        """
        Run suffix prediction on test_df.

        test_df must be structured by make_three_way_split() (or equivalent):
            in-flight cases  → all events ≤ val_split  (multiple rows per case)
            fresh cases      → first event only         (one row per case)

        This encoding is self-describing: cases with >1 row in test_df are
        in-flight and get their max pref_size prediction row (full history
        conditioning); cases with exactly 1 row are fresh and get
        pref_size 2 (start_plus_first) or 1 (start_only).

        Parameters
        ----------
        prefix_mode : "start_plus_first" (default) | "start_only"
            Applies only to fresh cases (single-event prefix).
            In-flight cases always use their full prefix regardless.
        full_prefix_only : bool, default False
            When True, only generate the single prediction conditioned on the
            full prefix per case (skipping all shorter prefix lengths).
            Produces identical results to False but avoids the O(N) wasted
            predictions that are discarded by to_event_log() anyway.

        Returns
        -------
        list[Path]  paths to filtered gen_*.csv files (one per repetition).
        """
        if prefix_mode not in ("start_only", "start_plus_first"):
            raise ValueError(f"prefix_mode must be 'start_only' or 'start_plus_first', got {prefix_mode!r}")
        if self.best_model_path is None:
            raise RuntimeError("Call train() before predict().")

        with _repo_cwd():
            params_dir = os.path.join(self._out_base(), "parameters")

            _test_for_csv = self.test_df.copy()
            _test_for_csv["caseid"] = _test_for_csv["caseid"].astype(str)
            _test_for_csv.to_csv(
                os.path.join(params_dir, "test_log.csv"),
                index=False, encoding="utf-8",
            )

            _folder = (os.path.join(self.trim_dir, self.run_name)
                       if self.trim_dir else self.run_name)
            parms = {
                "folder":         _folder,
                "model_file":     os.path.basename(self.best_model_path),
                "activity":       "pred_sfx",
                "variant":        variant,
                "rep":            reps,
                "one_timestamp":  self.params["one_timestamp"],
                "read_options":   self.params["read_options"],
                "is_single_exec": False,
                "scale_args":     {},
            }
            parms["scale_args"] = self._compute_scale_args(parms)
            parms["checkpoint_path"] = str(
                REPO_DIR / self._out_base() / "prediction_checkpoint.pkl"
            )
            parms["checkpoint_interval"] = 500

            from tensorflow.keras import Model as _KerasModel
            from model_prediction.suffix_predictor import SuffixPredictor as _SP


            if full_prefix_only:
                n_total = int(self.test_df["caseid"].nunique())
            else:
                n_total = int(
                    self.test_df.groupby("caseid")["caseid"].count().add(1).sum()
                )

            _done = 0
            _last_print = time.monotonic()

            _orig_create_result = _SP.create_result_record

            def _patched_create_result(self_sp, index, spl, preds, parms_sp, pref_size):
                nonlocal _done, _last_print
                result = _orig_create_result(self_sp, index, spl, preds, parms_sp, pref_size)
                _done += 1
                now = time.monotonic()
                if now - _last_print >= 10.0 or _done == n_total:
                    print(f"Predicting traces: {_done}/{n_total}", flush=True)
                    _last_print = now
                return result

            _orig_predict = _KerasModel.predict

            def _fast_predict(mdl, x, **kwargs):
                outputs = mdl(x, training=False)
                if isinstance(outputs, (list, tuple)):
                    return [o.numpy() for o in outputs]
                return outputs.numpy()

            _SP.create_result_record = _patched_create_result
            _KerasModel.predict = _fast_predict

            if full_prefix_only:
                from model_prediction.suffix_samples_creator import (
                    SuffixSamplesCreator as _SSC,
                )
                _orig_sample_suffix       = _SSC._sample_suffix
                _orig_sample_suffix_inter = _SSC._sample_suffix_inter

                def _full_prefix_sample_suffix(self_ssc, columns, parms_ssc):
                    import numpy as np, pandas as pd
                    times = ['dur_norm'] if parms_ssc['one_timestamp'] else ['dur_norm', 'wait_norm']
                    equi  = {'ac_index': 'activities', 'rl_index': 'roles'}
                    vec   = {'prefixes': dict(), 'next_evt': dict()}
                    x_times_dict, y_times_dict = dict(), dict()
                    case_ids = list()
                    self_ssc.log = self_ssc.reformat_events(columns, parms_ssc['one_timestamp'])
                    for i, _ in enumerate(self_ssc.log):
                        N = len(self_ssc.log[i][columns[0]])
                        case_ids.append(self_ssc.log[i]['caseid'])
                        for x in columns:
                            idx  = N - 1
                            serie   = [self_ssc.log[i][x][:idx]]
                            y_serie = [self_ssc.log[i][x][idx:]]
                            if x in equi:
                                vec['prefixes'][equi[x]] = (vec['prefixes'][equi[x]] + serie if i > 0 else serie)
                                vec['next_evt'][equi[x]] = (vec['next_evt'][equi[x]] + y_serie if i > 0 else y_serie)
                            elif x in times:
                                x_times_dict[x] = (x_times_dict[x] + serie if i > 0 else serie)
                                y_times_dict[x] = (y_times_dict[x] + y_serie if i > 0 else y_serie)
                    vec['prefixes']['times'] = []
                    for row in pd.DataFrame(x_times_dict).values:
                        nr = np.dstack([np.array(v) for v in row])
                        vec['prefixes']['times'].append(nr.reshape(nr.shape[1], nr.shape[2]))
                    vec['next_evt']['times'] = []
                    for row in pd.DataFrame(y_times_dict).values:
                        nr = np.dstack([np.array(v) for v in row])
                        vec['next_evt']['times'].append(nr.reshape(nr.shape[1], nr.shape[2]))
                    vec['prefixes']['caseids'] = case_ids
                    return vec

                def _full_prefix_sample_suffix_inter(self_ssc, columns, parms_ssc):
                    import numpy as np, pandas as pd
                    equi = {'ac_index': 'activities', 'rl_index': 'roles', 'dur_norm': 'times'}
                    spl  = {'prefixes': dict(), 'suffixes': dict()}
                    x_inter_dict, y_inter_dict = dict(), dict()
                    case_ids = list()
                    self_ssc.log = self_ssc.reformat_events(columns, parms_ssc['one_timestamp'])
                    for i, _ in enumerate(self_ssc.log):
                        N = len(self_ssc.log[i][columns[0]])
                        case_ids.append(self_ssc.log[i]['caseid'])
                        for x in columns:
                            idx     = N - 1
                            serie   = [self_ssc.log[i][x][:idx]]
                            y_serie = [self_ssc.log[i][x][idx:]]
                            if x in equi:
                                spl['prefixes'][equi[x]] = (spl['prefixes'][equi[x]] + serie if i > 0 else serie)
                                spl['suffixes'][equi[x]] = (spl['suffixes'][equi[x]] + y_serie if i > 0 else y_serie)
                            else:
                                x_inter_dict[x] = (x_inter_dict[x] + serie if i > 0 else serie)
                                y_inter_dict[x] = (y_inter_dict[x] + y_serie if i > 0 else y_serie)
                    spl['prefixes']['inter_attr'] = []
                    for row in pd.DataFrame(x_inter_dict).values:
                        nr = np.dstack([np.array(v) for v in row])
                        spl['prefixes']['inter_attr'].append(nr.reshape(nr.shape[1], nr.shape[2]))
                    spl['suffixes']['inter_attr'] = []
                    for row in pd.DataFrame(y_inter_dict).values:
                        nr = np.dstack([np.array(v) for v in row])
                        spl['suffixes']['inter_attr'].append(nr.reshape(nr.shape[1], nr.shape[2]))
                    spl['prefixes']['caseids'] = case_ids
                    return spl

                _SSC._sample_suffix       = _full_prefix_sample_suffix
                _SSC._sample_suffix_inter = _full_prefix_sample_suffix_inter

            try:
                ModelPredictor(parms)
            except Exception as exc:
                # export_predictions() is called before the evaluator, so gen_*.csv
                # files are already saved even when the post-prediction step raises.
                print(f"[predict] post-prediction step raised: {exc}")
                traceback.print_exc()
                # Re-raise if no gen_*.csv were produced — the exception is the root cause.
                _check_dir = REPO_DIR / self._out_base()
                if not list(_check_dir.glob("gen_*.csv")):
                    raise RuntimeError(
                        f"ModelPredictor failed before writing gen_*.csv. "
                        f"Root cause above."
                    ) from exc
            finally:
                _SP.create_result_record = _orig_create_result
                _KerasModel.predict = _orig_predict
                if full_prefix_only:
                    _SSC._sample_suffix       = _orig_sample_suffix
                    _SSC._sample_suffix_inter = _orig_sample_suffix_inter

        pred_dir = REPO_DIR / self._out_base()
        raw_paths = sorted(pred_dir.glob("gen_*.csv"))
        if not raw_paths:
            raise FileNotFoundError(f"No gen_*.csv found in {pred_dir}")

        # Classify cases by event count in test_df:
        #   >1 event → in-flight → keep max pref_size (full-history conditioning)
        #   =1 event → fresh     → keep pref_size 2 (start_plus_first) or 1 (start_only)
        test_counts   = self.test_df.groupby("caseid")["caseid"].count()
        in_flight_str = {str(c) for c in test_counts[test_counts > 1].index}
        fresh_str     = {str(c) for c in test_counts[test_counts == 1].index}
        fresh_pref_size = 1 if prefix_mode == "start_only" else 2

        filtered_paths = []
        for p in raw_paths:
            df_p = pd.read_csv(p, keep_default_na=False, na_values=_NA_VALUES_NO_NA,
                              dtype={"caseid": str})
            df_p["caseid"] = df_p["caseid"].astype(str)

            parts_f = []
            if in_flight_str:
                parts_f.append(
                    df_p[df_p["caseid"].isin(in_flight_str)]
                    .sort_values("pref_size")
                    .groupby("caseid", as_index=False)
                    .last()
                )
            if fresh_str:
                parts_f.append(
                    df_p[
                        df_p["caseid"].isin(fresh_str) &
                        (df_p["pref_size"] == fresh_pref_size)
                    ]
                )
            filtered = pd.concat(parts_f, ignore_index=True) if parts_f else pd.DataFrame()
            print(f"[predict] filtered {p.name}: {len(filtered)} rows "
                  f"(in_flight={len(in_flight_str)}, fresh={len(fresh_str)}, "
                  f"fresh_pref_size={fresh_pref_size})")
            filtered.to_csv(p, index=False)
            filtered_paths.append(p)

        return filtered_paths

    @staticmethod
    def _parse_list_col(s: str):
        """ast.literal_eval with numpy-scalar stripping (e.g. np.int64(5) → 5)."""
        s = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', str(s))
        return ast.literal_eval(s)

    def to_event_log(self, pred_paths: list[Path]) -> pd.DataFrame:
        """
        Convert gen_*.csv prediction files into an event log with absolute timestamps.

        Reads the predicted suffixes, decodes integer indices to labels, converts
        relative time predictions to absolute timestamps anchored to the last known
        event per case, and returns:

            pre-val_split actual history  +  GLSTM-predicted suffix events

        Columns: caseid, task, user, end_timestamp

        Requires test_df to be structured by make_three_way_split():
            in-flight cases → all pre-val_split events  (anchor = last of these)
            fresh cases     → first event only           (anchor = that event)
        """
        _header = pd.read_csv(pred_paths[0], nrows=0).columns.tolist()
        one_ts = "tm_pred" in _header

        _base_cols = ["ac_prefix", "ac_expect", "ac_pred",
                      "rl_prefix", "rl_expect", "rl_pred"]
        _time_cols = (["tm_prefix", "tm_expect", "tm_pred"] if one_ts
                      else ["dur_prefix", "dur_expect", "dur_pred",
                            "wait_prefix", "wait_expect", "wait_pred"])
        _list_cols = _base_cols + _time_cols

        for attr in ("train_df", "val_df", "test_df"):
            df_a = getattr(self, attr)
            df_a["caseid"] = df_a["caseid"].astype(str)
            setattr(self, attr, df_a)

        frames = [
            pd.read_csv(p, dtype={"caseid": str},
                        keep_default_na=False, na_values=_NA_VALUES_NO_NA,
                        converters={c: self._parse_list_col for c in _list_cols
                                    if c in pd.read_csv(p, nrows=0).columns})
            for p in pred_paths
        ]
        preds = pd.concat(frames, ignore_index=True)

        preds["ac_pred_label"] = preds["ac_pred"].apply(
            lambda lst: [self.index_ac.get(x, "unk") for x in lst])
        preds["rl_pred_label"] = preds["rl_pred"].apply(
            lambda lst: [self.index_rl.get(x, "unk") for x in lst])

        if one_ts:
            preds["_secs"] = preds["tm_pred"].apply(_shift_list).apply(_cumulate_list)
        else:
            def _combine_dur_wait(row):
                dur  = row["dur_pred"]  if isinstance(row["dur_pred"],  list) else []
                wait = row["wait_pred"] if isinstance(row["wait_pred"], list) else []
                combined = [w + d for w, d in zip(wait, dur)]
                return _cumulate_list(_shift_list(combined))
            preds["_secs"] = preds.apply(_combine_dur_wait, axis=1)

        known = pd.concat([self.train_df, self.val_df], ignore_index=True)
        known["caseid"] = known["caseid"].astype(str)


        anchor = (known.groupby("caseid")["end_timestamp"]
                       .max().rename("last_ts").reset_index())

        missing_anchor = set(preds["caseid"].dropna()) - set(anchor["caseid"].dropna())
        if missing_anchor:
            anchor_test = (
                self.test_df[self.test_df["caseid"].isin(missing_anchor)]
                .groupby("caseid")["end_timestamp"]
                .max().rename("last_ts").reset_index()
            )
            anchor = pd.concat([anchor, anchor_test], ignore_index=True)

        anchor["caseid"] = anchor["caseid"].astype(str)
        preds = preds.merge(anchor, on="caseid", how="left")

        lens_ok = (
            preds["ac_pred_label"].str.len().eq(preds["rl_pred_label"].str.len()) &
            preds["ac_pred_label"].str.len().eq(preds["_secs"].str.len())
        )
        preds = preds.loc[
            lens_ok,
            ["caseid", "ac_pred_label", "rl_pred_label", "_secs", "last_ts"]
        ].copy()

        preds = preds.explode(
            ["ac_pred_label", "rl_pred_label", "_secs"], ignore_index=True)
        all_pred_caseids = set(preds["caseid"].dropna())
        preds = preds[~preds["ac_pred_label"].isin(["start", "end"])].copy()

        preds["_secs"] = pd.to_numeric(preds["_secs"], errors="coerce")
        _max_secs = 5 * 365.25 * 24 * 3600  # 5-year cap; anything beyond is a model artifact
        n_clipped = (preds["_secs"].abs() > _max_secs).sum()
        if n_clipped:
            print(f"[to_event_log] clipping {n_clipped} out-of-range duration(s) (> 5 years)")
        preds["_secs"] = preds["_secs"].clip(-_max_secs, _max_secs)
        preds["end_timestamp"] = (
            preds["last_ts"] + pd.to_timedelta(preds["_secs"], unit="s"))

        hist = known[["caseid", "task", "user", "end_timestamp"]].copy()
        missing_hist = all_pred_caseids - set(hist["caseid"].dropna())

        if missing_hist:
            test_hist = self.test_df[self.test_df["caseid"].isin(missing_hist)][
                ["caseid", "task", "user", "end_timestamp"]
            ]
            hist = pd.concat([hist, test_hist], ignore_index=True)

        pred_events = preds.rename(columns={
            "ac_pred_label": "task",
            "rl_pred_label": "user",
        })[["caseid", "task", "user", "end_timestamp"]].copy()

        event_log = pd.concat([hist, pred_events], ignore_index=True)

        if self.output_dir is not None:
            out_path = REPO_DIR / self.output_dir / "event_log.csv"
            event_log.to_csv(out_path, index=False)
            print(f"[to_event_log] saved → {out_path}")

        return event_log

    # ── Preprocessing internals ────────────────────────────────────────────────

    def _add_roles(self):
        """
        Discover resource pools from train_df, then apply the same mapping to
        val_df and test_df.  Users absent from the training pool get role='unk'.
        Stores the user→role table on self._resource_map for later serialization.
        """
        from support_modules import role_discovery as rd

        analyser  = rd.ResourcePoolAnalyser(self.train_df,
                                             sim_threshold=self.params["rp_sim"])
        resources = (
            pd.DataFrame.from_records(analyser.resource_table)
              .rename(columns={"resource": "user"})
        )
        self._resource_map = resources

        self.train_df = self._apply_roles(self.train_df, resources)
        self.val_df   = self._apply_roles(self.val_df,   resources)
        self.test_df  = self._apply_roles(self.test_df,  resources)

    @staticmethod
    def _apply_roles(df: pd.DataFrame, resources: pd.DataFrame) -> pd.DataFrame:
        df["user"] = df["user"].astype(str)
        resources["user"] = resources["user"].astype(str)
        df = (df.merge(resources, on="user", how="left")
                .pipe(lambda d: d[~d["task"].isin(["Start", "End"])])
                .reset_index(drop=True))
        df["role"]   = df["role"].fillna("unk")
        df["caseid"] = df["caseid"].astype(str)
        return df

    def _build_indexes(self):
        """Build ac_index and rl_index from training data only."""
        self.ac_index = self._create_index(self.train_df, "task")
        self.ac_index["start"] = 0
        self.ac_index["end"]   = len(self.ac_index)
        self.index_ac = {v: k for k, v in self.ac_index.items()}

        self.rl_index = self._create_index(self.train_df, "role")
        self.rl_index["start"] = 0
        self.rl_index["end"]   = len(self.rl_index)
        self.index_rl = {v: k for k, v in self.rl_index.items()}

    def _apply_indexes(self):
        """Add ac_index / rl_index columns to all three splits."""
        for attr in ("train_df", "val_df", "test_df"):
            df = getattr(self, attr)
            df["ac_index"] = df["task"].map(self.ac_index).fillna(0).astype(int)
            df["rl_index"] = df["role"].map(self.rl_index).fillna(0).astype(int)
            setattr(self, attr, df)

    def _load_or_train_embeddings(self):
        """Load .emb files if they exist and match vocabulary; otherwise train."""
        stem    = self.params["file_name"].split(".")[0]
        ac_emb  = f"ac_{stem}.emb"
        rl_emb  = f"rl_{stem}.emb"
        emb_dir = os.path.join("input_files", "embedded_matix")

        need_train = True
        if os.path.exists(os.path.join(emb_dir, ac_emb)):
            ac_w = self._load_embedded(self.index_ac, ac_emb)
            rl_w = self._load_embedded(self.index_rl, rl_emb)
            if ac_w.shape[0] == len(self.index_ac) and rl_w.shape[0] == len(self.index_rl):
                self.ac_weights = ac_w
                self.rl_weights = rl_w
                need_train = False
            else:
                print(f"[preprocess] stale embeddings (ac={ac_w.shape[0]}, rl={rl_w.shape[0]}) "
                      f"vs vocab (ac={len(self.index_ac)}, rl={len(self.index_rl)}) — re-training.")

        if need_train:
            em.training_model(
                self.params, self.train_df,
                self.ac_index, self.index_ac,
                self.rl_index, self.index_rl,
            )
            self.ac_weights = self._load_embedded(self.index_ac, ac_emb)
            self.rl_weights = self._load_embedded(self.index_rl, rl_emb)

    # ── Post-training internals ────────────────────────────────────────────────

    def _compute_scale_args(self, parms: dict) -> dict:
        """
        Recompute scale_args from training+val data using the saved model's
        norm_method.  Called from predict() so that models saved without
        scale_args in their JSON still produce correct rescaled timestamps.
        Must be called inside _repo_cwd().
        """
        params_path = os.path.join(
            self._out_base(), "parameters", "model_parameters.json"
        )
        try:
            with open(params_path) as f:
                saved = json.load(f)
            if "scale_args" in saved:
                return {}   # load_parameters will handle it from the JSON
            model_type = saved["model_type"]
            model_def = CamargoOptimizer._read_model_def(model_type)
            fm_parms = {
                "model_type":    model_type,
                "norm_method":   saved["norm_method"],
                "one_timestamp": parms["one_timestamp"],
            }
            fm = FeaturesMannager(fm_parms)
            fm.register_scaler(model_type, model_def["scaler"])
            combined = pd.concat([self.train_df, self.val_df], ignore_index=True)
            _, scale_args = fm.calculate(combined, model_def["additional_columns"])
            return scale_args
        except Exception as exc:
            print(f"[predict] could not compute scale_args: {exc}")
            return {}

    def _export_params(self, output_folder: str, best_params: dict):
        """
        Write model_parameters.json and test_log.csv into output_folder/parameters/.
        test_log.csv contains OUR test_df, not GLSTM's internal split.
        """
        params_dir = os.path.join(output_folder, "parameters")
        os.makedirs(params_dir, exist_ok=True)

        full_train = pd.concat([self.train_df, self.val_df], ignore_index=True)
        best_params["max_trace_size"] = int(
            full_train.groupby("caseid")["task"].count().max()
        )
        best_params["index_ac"] = self.index_ac
        best_params["index_rl"] = self.index_rl
        best_params["one_timestamp"] = self.params["one_timestamp"]
        # suffix_predictor uses parms['dim']['time_dim'] as the LSTM window size
        best_params["dim"] = {"time_dim": best_params["n_size"]}

        sup.create_json(
            best_params,
            os.path.join(params_dir, "model_parameters.json"),
        )
        if hasattr(self, "_resource_map"):
            self._resource_map.to_csv(
                os.path.join(params_dir, "resource_map.csv"),
                index=False,
            )
        # Our test split becomes the prediction target.
        self.test_df.to_csv(
            os.path.join(params_dir, "test_log.csv"),
            index=False, encoding="utf-8",
        )

    # ── Statics ────────────────────────────────────────────────────────────────

    @staticmethod
    def _create_index(log_df: pd.DataFrame, column: str) -> dict:
        vals = sorted(log_df[column].dropna().unique())
        return {v: i + 1 for i, v in enumerate(vals)}

    @staticmethod
    def _load_embedded(index: dict, filename: str) -> np.ndarray:
        weights = []
        path = os.path.join("input_files", "embedded_matix", filename)
        with open(path, "r") as f:
            for row in csv.reader(f):
                cat_ix = int(row[0])
                if index.get(cat_ix) == row[1].strip():
                    weights.append([float(x) for x in row[2:]])
        return np.array(weights)

    @staticmethod
    def _find_model_file(output_folder: str) -> str:
        h5_files = list(Path(output_folder).glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 model file in {output_folder}")
        return str(h5_files[0])
