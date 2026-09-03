"""
CamargoOptimizer — Bayesian HPO that uses pre-split train / val DataFrames.

Drop-in replacement for GenerativeLSTM's ModelOptimizer, with one key
difference: instead of performing an internal timeline split on every trial,
it receives separate train and val DataFrames and uses them directly.

Scaling is computed on (train ∪ val) before splitting back, which is identical
to what the original code does (it scales the full log, then splits).
"""
import ast
import copy
import json
import os
import pickle
import shutil
import time
import traceback

import pandas as pd
import configparser as cp
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK, STATUS_FAIL

# These are resolved at call-time (REPO_DIR is on sys.path by the time
# CamargoTrainer instantiates this class).
from model_training import samples_creator as sc
from model_training import model_loader as mload
from model_training import features_manager as feat
import utils.support as sup


def _to_native(obj):
    """Recursively convert numpy scalars to plain Python types for CSV round-trip."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    try:
        return obj.item()
    except AttributeError:
        return obj


class CamargoOptimizer:
    """
    Bayesian HPO over GenerativeLSTM model architectures.

    Parameters
    ----------
    parms       : dict  – see camargo.params.default_params
    log_train   : DataFrame – training events (after role assignment + indexing)
    log_val     : DataFrame – validation events (same preprocessing)
    ac_index    : dict  – activity → integer index
    ac_weights  : ndarray – pre-trained activity embedding matrix
    rl_index    : dict  – role → integer index
    rl_weights  : ndarray – pre-trained role embedding matrix
    hpo_dir     : str | None – persistent directory for per-trial checkpoints.
                  If None, no checkpointing is done (legacy behaviour).
    """

    def __init__(self, parms, log_train, log_val,
                 ac_index, ac_weights, rl_index, rl_weights,
                 hpo_dir: str | None = None):
        self.parms      = parms
        self.log_train  = copy.deepcopy(log_train)
        self.log_val    = copy.deepcopy(log_val)
        self.ac_index   = ac_index
        self.ac_weights = ac_weights
        self.rl_index   = rl_index
        self.rl_weights = rl_weights

        self.space = self._build_space(parms)

        self.temp_output = parms["output"]
        os.makedirs(self.temp_output, exist_ok=True)
        self.file_name = os.path.join(self.temp_output, sup.file_id(prefix="OP_"))
        open(self.file_name, "w").close()

        # ── Checkpointing setup ───────────────────────────────────────────────
        self.hpo_dir      = hpo_dir
        self._trials_pkl  = (
            os.path.join(hpo_dir, "hyperopt_trials.pkl") if hpo_dir else None
        )
        self._trial_counter = [0]

        if hpo_dir:
            os.makedirs(hpo_dir, exist_ok=True)

        self.bayes_trials = Trials()
        if self._trials_pkl and os.path.exists(self._trials_pkl):
            with open(self._trials_pkl, "rb") as _f:
                self.bayes_trials = pickle.load(_f)
            _done = len(self.bayes_trials.trials)
            self._trial_counter = [_done]
            print(f"[CamargoOptimizer] Resuming: {_done}/{parms['max_eval']} trials already done")
        elif hpo_dir:
            print(f"[CamargoOptimizer] Starting fresh HPO: {parms['max_eval']} trials")

        self.best_output = None
        self.best_params = {}
        self.best_loss   = 1.0

    # ── Public ────────────────────────────────────────────────────────────────

    def execute_trials(self):
        _hpo_dir  = self.hpo_dir
        _pkl_path = self._trials_pkl
        _counter  = self._trial_counter

        def exec_pipeline(trial_stg):
            i = _counter[0]
            _counter[0] += 1

            # Persist Trials state before this trial starts so that a crash
            # mid-trial leaves the pkl one trial behind (recoverable via
            # result.json caching below).
            if _pkl_path:
                with open(_pkl_path, "wb") as _f:
                    pickle.dump(self.bayes_trials, _f)

            # ── Cache check ───────────────────────────────────────────────────
            if _hpo_dir:
                trial_subdir = os.path.join(_hpo_dir, f"trial_{i:03d}")
                result_json  = os.path.join(trial_subdir, "result.json")

                if os.path.exists(result_json):
                    with open(result_json) as _f:
                        cached = json.load(_f)
                    print(f"[{i+1}/{self.parms['max_eval']}] cached  "
                          f"loss={cached['loss']:.4f}  (trial_{i:03d})")
                    return {
                        "output": cached["output"],
                        "status": cached["status"],
                        "loss":   cached["loss"],
                    }

                # Partial folder exists but no result.json → crashed mid-trial.
                if os.path.exists(trial_subdir):
                    shutil.rmtree(trial_subdir)

            # ── Run trial ─────────────────────────────────────────────────────
            try:
                trial_stg = self._redef_output(trial_stg)
                model_def = self._read_model_def(trial_stg["model_type"])

                # Scale on combined log (same as original: scale before split).
                combined = pd.concat([self.log_train, self.log_val], ignore_index=True)
                combined, trial_stg = self._scale(combined, trial_stg, model_def)

                train_ids = set(self.log_train["caseid"])
                val_ids   = set(self.log_val["caseid"])
                log_train = combined[combined["caseid"].isin(train_ids)].reset_index(drop=True)
                log_valdn = combined[combined["caseid"].isin(val_ids)].reset_index(drop=True)
                print(f"  train={len(log_train):,}  val={len(log_valdn):,}")

                # Vectorise
                one_ts = self.parms["read_options"]["one_timestamp"]
                vectorizer = sc.SequencesCreator(one_ts, self.ac_index, self.rl_index)
                vectorizer.register_vectorizer(trial_stg["model_type"], model_def["vectorizer"])
                train_vec = vectorizer.vectorize(
                    trial_stg["model_type"], log_train, trial_stg, model_def["additional_columns"])
                valdn_vec = vectorizer.vectorize(
                    trial_stg["model_type"], log_valdn, trial_stg, model_def["additional_columns"])

                # Train
                m_loader = mload.ModelLoader(trial_stg)
                m_loader.register_model(trial_stg["model_type"], model_def["trainer"])
                _t0 = time.perf_counter()
                model = m_loader.train(
                    trial_stg["model_type"], train_vec, valdn_vec,
                    self.ac_weights, self.rl_weights, trial_stg["output"])
                train_time_s = time.perf_counter() - _t0

                # Evaluate on val
                x_input = {
                    "ac_input": valdn_vec["prefixes"]["activities"],
                    "rl_input": valdn_vec["prefixes"]["roles"],
                    "t_input":  valdn_vec["prefixes"]["times"],
                }
                if "inter_attr" in valdn_vec["prefixes"]:
                    x_input["inter_input"] = valdn_vec["prefixes"]["inter_attr"]

                acc = model.evaluate(
                    x=x_input,
                    y={
                        "act_output":  valdn_vec["next_evt"]["activities"],
                        "role_output": valdn_vec["next_evt"]["roles"],
                        "time_output": valdn_vec["next_evt"]["times"],
                    },
                    return_dict=True,
                )
                print("-- End of trial --")

                response = self._make_response(trial_stg, STATUS_OK, acc["loss"], train_time_s)

                # ── Persist trial to hpo_dir ──────────────────────────────────
                if _hpo_dir:
                    shutil.copytree(trial_stg["output"], trial_subdir)
                    result_data = {
                        "trial":       f"trial_{i:03d}",
                        "output":      trial_subdir,
                        "status":      response["status"],
                        "loss":        response["loss"],
                        "scale_args":  _to_native(trial_stg.get("scale_args", {})),
                        "train_time_s": round(train_time_s, 2),
                        "model_type":  trial_stg.get("model_type"),
                        "lstm_act":    trial_stg.get("lstm_act"),
                        "n_size":      trial_stg.get("n_size"),
                        "l_size":      trial_stg.get("l_size"),
                        "norm_method": trial_stg.get("norm_method"),
                        "optim":       trial_stg.get("optim"),
                    }
                    with open(result_json, "w") as _f:
                        json.dump(result_data, _f, indent=2)
                    # Point response output to the persistent path so Trials
                    # stores it correctly for best-model extraction.
                    response["output"] = trial_subdir

                return response

            except Exception as exc:
                print(exc)
                traceback.print_exc()
                return self._make_response(trial_stg, STATUS_FAIL, 1.0, None)

        best = fmin(
            fn=exec_pipeline,
            space=self.space,
            algo=tpe.suggest,
            max_evals=self.parms["max_eval"],
            trials=self.bayes_trials,
            show_progressbar=False,
        )

        # Save final pkl after all trials complete.
        if _pkl_path:
            with open(_pkl_path, "wb") as _f:
                pickle.dump(self.bayes_trials, _f)

        try:
            results = (
                pd.DataFrame(self.bayes_trials.results)
                  .sort_values("loss", ascending=True)
            )
            result = results[results.status == "ok"].head(1).iloc[0]
            self.best_output = result.output
            self.best_loss   = result.loss
            # Map hyperopt integer choices back to actual values
            self.best_params = {
                k: self.parms[k][v]
                for k, v in best.items()
                if isinstance(self.parms.get(k), list)
            }
            # Read scale_args from persistent result.json (hpo_dir) or CSV (legacy).
            if _hpo_dir:
                _best_result_json = os.path.join(result.output, "result.json")
                with open(_best_result_json) as _f:
                    _best_data = json.load(_f)
                self.best_params["scale_args"] = _best_data["scale_args"]
            else:
                opt_res = pd.read_csv(self.file_name)
                opt_res = opt_res[opt_res.output == result.output].iloc[0]
                self.best_params["scale_args"] = json.loads(opt_res.scale_args)
        except Exception as exc:
            print(f"[CamargoOptimizer] could not extract best params: {exc}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_space(parms):
        return {
            "model_type":  hp.choice("model_type",  parms["model_type"]),
            "n_size":      hp.choice("n_size",       parms["n_size"]),
            "l_size":      hp.choice("l_size",       parms["l_size"]),
            "lstm_act":    hp.choice("lstm_act",     parms["lstm_act"]),
            "dense_act":   hp.choice("dense_act",    parms["dense_act"]),
            "norm_method": hp.choice("norm_method",  parms["norm_method"]),
            "optim":       hp.choice("optim",        parms["optim"]),
            # constants passed through unchanged
            "imp":           parms["imp"],
            "file":          parms["file_name"],
            "batch_size":    parms["batch_size"],
            "epochs":        parms["epochs"],
            "one_timestamp": parms["one_timestamp"],
        }

    def _redef_output(self, settings):
        settings = copy.deepcopy(settings)
        settings["output"] = os.path.join(self.temp_output, sup.folder_id())
        os.makedirs(settings["output"], exist_ok=True)
        return settings

    @staticmethod
    def _scale(log, params, model_def):
        inp = feat.FeaturesMannager(params)
        inp.register_scaler(params["model_type"], model_def["scaler"])
        log, params["scale_args"] = inp.calculate(log, model_def["additional_columns"])
        return log, params

    def _make_response(self, parms, status, loss, train_time_s):
        response = {
            "output": parms.get("output", ""),
            "status": status,
            "loss":   loss if status == STATUS_OK and loss > 0 else 1.0,
        }
        if status == STATUS_OK and loss <= 0:
            response["status"] = STATUS_FAIL

        row = {
            "loss":          response["loss"],
            "sim_metric":    "val_loss",
            "status":        response["status"],
            "n_size":        parms.get("n_size"),
            "l_size":        parms.get("l_size"),
            "lstm_act":      parms.get("lstm_act"),
            "dense_act":     parms.get("dense_act"),
            "optim":         parms.get("optim"),
            "train_time_s":  round(train_time_s, 2) if train_time_s is not None else None,
            "scale_args":    json.dumps(_to_native(parms.get("scale_args"))),
            "output":        parms.get("output"),
        }
        if os.path.getsize(self.file_name) > 0:
            sup.create_csv_file([row], self.file_name, mode="a")
        else:
            sup.create_csv_file_header([row], self.file_name)
        return response

    @staticmethod
    def _read_model_def(model_type):
        config = cp.ConfigParser(interpolation=None)
        config.read("models_spec.ini")
        return {
            "additional_columns": sup.reduce_list(
                config.get(model_type, "additional_columns"), dtype="str"),
            "scaler":     config.get(model_type, "scaler"),
            "vectorizer": config.get(model_type, "vectorizer"),
            "trainer":    config.get(model_type, "trainer"),
        }
