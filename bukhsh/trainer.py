"""
BukhshTrainer — bridge between our train/val/test splits and the
ProcessTransformer (Bukhsh et al. 2021) pipeline.

Usage
-----
from bukhsh.trainer import BukhshTrainer
from bukhsh.params  import default_params

params  = default_params(epochs=50)
trainer = BukhshTrainer(train_df, val_df, test_df, "loan_flat", params,
                         output_dir="bukhsh_output/loan_flat")
trainer.run()                               # preprocess + train all 4 tasks
event_log, rem_time_df = trainer.predict()  # suffix + remaining-time prediction
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── ProcessTransformer repo path ──────────────────────────────────────────────
PT_DIR = Path(__file__).resolve().parent.parent / "processtransformer"
if str(PT_DIR) not in sys.path:
    sys.path.insert(0, str(PT_DIR))

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import tensorflow as tf  # noqa: E402

from pre_process.transformer_processor import LogsDataProcessor  # noqa: E402
from pre_process.transformer_loader    import LogsDataLoader      # noqa: E402
from models                            import transformer as _tf_models  # noqa: E402

_LOG = logging.getLogger("BukhshTrainer")
_END = "end"


def _normalize(s: str) -> str:
    return str(s).lower().replace(" ", "-")


class BukhshTrainer:
    """
    Orchestrates the full ProcessTransformer pipeline using our canonical
    train / val / test splits.

    Steps
    -----
    1. run()     — write CSVs, train next_act / next_role / next_time / rem_time
    2. predict() — hallucinate suffix per test case + predict remaining time

    Output
    ------
    predict() returns (event_log_df, rem_time_df) where
      event_log_df : caseid, task, user, end_timestamp
                     (history from train+val+test_prefix + predicted suffix)
      rem_time_df  : caseid, rem_time_days
                     (remaining time in days predicted by the rem_time model)

    The trainer also saves both files under output_dir/event_log.csv and
    output_dir/rem_time.csv.
    """

    def __init__(
        self,
        train_df:   pd.DataFrame,
        val_df:     pd.DataFrame,
        test_df:    pd.DataFrame,
        run_name:   str,
        params:     dict,
        output_dir: Optional[str | Path] = None,
    ):
        self.train_df = train_df.copy()
        self.val_df   = val_df.copy()
        self.test_df  = test_df.copy()
        self.run_name = run_name
        self.params   = dict(params)

        self.work_dir = (Path(output_dir) if output_dir
                        else Path("bukhsh_output") / run_name)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self._models:   dict = {}   # task → loaded TF model
        self._meta:     dict = {}   # task → {x_word_dict, y_word_dict, ...}
        self._role_map: Optional[dict] = None  # user → normalised role

        self._train_csv: Optional[Path] = None
        self._vocab_csv: Optional[Path] = None  # combined all-splits for PT vocab
        self._test_csv:  Optional[Path] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self):
        """Write CSVs and train all four ProcessTransformer tasks."""
        self._prepare_csvs()
        _results = self.work_dir / "results"
        for task in ("next_act", "next_role", "next_time", "rem_time"):
            _w = _results / f"{self.run_name}_{task}.weights.h5"
            if _w.exists():
                print(f"[{task}] weights found on disk — skipping training")
                self._train_task(task)   # loads weights + populates self._meta
            else:
                print(f"\n{'='*60}\nTraining task: {task}\n{'='*60}")
                self._train_task(task)
        self._save_meta()

    def predict(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predict suffix and remaining time for every case in test_df.

        Returns
        -------
        event_log_df : DataFrame  (caseid, task, user, end_timestamp)
        rem_time_df  : DataFrame  (caseid, rem_time_days)
        """
        if not self._models:
            self._load_all_models()

        self._build_role_map()

        suffix_rows:   list = []
        rem_time_rows: list = []

        cases = self.test_df.groupby("caseid")
        n = len(cases)
        for idx, (case_id, case_df) in enumerate(cases):
            if (idx + 1) % 200 == 0:
                print(f"  predicting case {idx+1}/{n}")

            case_df = case_df.sort_values("end_timestamp")
            acts  = [_normalize(a) for a in case_df["task"]]
            users = list(case_df["user"])
            roles = [self._get_role(u) for u in users]
            ts    = pd.to_datetime(case_df["end_timestamp"]).tolist()
            ts    = [_strip_tz(t) for t in ts]

            time_feats = _compute_time_feats(ts)
            anchor_ts  = ts[-1]

            # ── remaining time ────────────────────────────────────────────────
            rem_secs = self._predict_rem_time(acts, time_feats)
            rem_time_rows.append({
                "caseid":            case_id,
                "start_timestamp":   ts[0],       # first prefix event (tz-naive)
                "anchor_timestamp":  anchor_ts,   # last prefix event; predicted_end = anchor + rem
                "prefix_len":        len(acts),
                "rem_time_days":     max(0.0, rem_secs) / 86400,
            })

            # ── suffix hallucination ──────────────────────────────────────────
            suffix = self._hallucinate(acts, roles, anchor_ts, time_feats)
            for act, role, abs_ts in suffix:
                if act == _END:
                    continue
                suffix_rows.append({
                    "caseid":        case_id,
                    "task":          act,
                    "user":          role,
                    "end_timestamp": abs_ts,
                })

        # ── build full event log ───────────────────────────────────────────────
        known = pd.concat([self.train_df, self.val_df], ignore_index=True)
        known["caseid"] = known["caseid"].astype(str)

        # For cases not in train/val (rare in full_traces mode, common in prefix
        # mode), use the test_df prefix events as history.
        missing = set(self.test_df["caseid"].astype(str)) - set(known["caseid"])
        if missing:
            extra = self.test_df[self.test_df["caseid"].astype(str).isin(missing)]
            known = pd.concat([
                known,
                extra[["caseid", "task", "user", "end_timestamp"]],
            ], ignore_index=True)

        hist = known[["caseid", "task", "user", "end_timestamp"]].copy()
        # Strip timezone so hist (tz-aware from XES) and pred (tz-naive from
        # _hallucinate) can be concatenated without a mixed-tz error.
        hist["end_timestamp"] = (pd.to_datetime(hist["end_timestamp"], utc=True)
                                   .dt.tz_convert(None))
        pred = pd.DataFrame(suffix_rows)

        event_log = (pd.concat([hist, pred], ignore_index=True)
                       if len(pred) else hist)

        rem_df = pd.DataFrame(rem_time_rows)

        # ── save ──────────────────────────────────────────────────────────────
        el_path = self.work_dir / "event_log.csv"
        rt_path = self.work_dir / "rem_time.csv"
        event_log.to_csv(el_path, index=False)
        rem_df.to_csv(rt_path, index=False)
        print(f"[predict] event_log → {el_path}  ({len(event_log):,} rows)")
        print(f"[predict] rem_time  → {rt_path}  ({len(rem_df):,} rows)")

        return event_log, rem_df

    # ── CSV preparation ────────────────────────────────────────────────────────

    def _prepare_csvs(self):
        """
        Write three CSVs for ProcessTransformer:
        - train.csv : train+val with appended 'end' events (used for model fitting)
        - vocab_ref.csv : train+val+test combined WITHOUT end events (used as
          the PT 'test' path so that role discovery always has enough users and
          the vocabulary covers all activities seen at inference time)
        - test.csv : actual test split WITHOUT end events (kept for reference /
          role-map building in _build_role_map)
        """
        full_train = pd.concat([self.train_df, self.val_df], ignore_index=True)
        full_all   = pd.concat(
            [self.train_df, self.val_df, self.test_df], ignore_index=True)

        train_csv  = self._to_pt_csv(full_train, add_end_events=True)
        vocab_csv  = self._to_pt_csv(full_all,   add_end_events=False)
        test_csv   = self._to_pt_csv(self.test_df, add_end_events=False)

        self._train_csv = self.work_dir / "train.csv"
        self._vocab_csv = self.work_dir / "vocab_ref.csv"
        self._test_csv  = self.work_dir / "test.csv"

        train_csv.to_csv(self._train_csv, index=False)
        vocab_csv.to_csv(self._vocab_csv, index=False)
        test_csv.to_csv(self._test_csv,   index=False)

        print(f"[prepare] train: {len(train_csv):,} rows "
              f"| vocab_ref: {len(vocab_csv):,} rows "
              f"| test: {len(test_csv):,} rows")

    @staticmethod
    def _to_pt_csv(df: pd.DataFrame, *, add_end_events: bool) -> pd.DataFrame:
        """
        Convert a DataFrame to ProcessTransformer CSV format.
        Expected columns in df: caseid, task, user, end_timestamp.
        Output columns: case_id, activity_name, end_timestamp, start_timestamp, resource.
        start_timestamp is set to end_timestamp (zero processing time).
        """
        out = df[["caseid", "task", "user", "end_timestamp"]].copy()
        out = out.rename(columns={
            "caseid": "case_id",
            "task":   "activity_name",
            "user":   "resource",
        })
        out["end_timestamp"] = pd.to_datetime(out["end_timestamp"])
        if out["end_timestamp"].dt.tz is not None:
            out["end_timestamp"] = out["end_timestamp"].dt.tz_localize(None)
        out["start_timestamp"] = out["end_timestamp"]

        if add_end_events:
            rows = []
            for cid, grp in out.groupby("case_id"):
                last = grp["end_timestamp"].max()
                end_ts = last + pd.Timedelta(seconds=1)
                rows.append({
                    "case_id":         cid,
                    "activity_name":   _END,
                    "resource":        _END,
                    "end_timestamp":   end_ts,
                    "start_timestamp": end_ts,
                })
            out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)
            out = (out.sort_values(["case_id", "end_timestamp"])
                      .reset_index(drop=True))

        # Format explicitly AFTER sorting so midnight timestamps always appear as
        # '2023-01-02 00:00:00' in the CSV (pandas omits the time for pure-date
        # midnight values when writing datetime64 columns).
        out["end_timestamp"]   = pd.to_datetime(out["end_timestamp"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S")
        out["start_timestamp"] = pd.to_datetime(out["start_timestamp"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S")

        return out[["case_id", "activity_name", "end_timestamp",
                    "start_timestamp", "resource"]]

    # ── Task training ──────────────────────────────────────────────────────────

    def _make_args(self, task: str) -> argparse.Namespace:
        """Build an argparse.Namespace that LogsDataProcessor / LogsDataLoader expect."""
        processed = self.work_dir / "processed"
        processed.mkdir(exist_ok=True)
        result   = self.work_dir / "results"
        result.mkdir(exist_ok=True)

        args = argparse.Namespace(
            dataset       = self.run_name,
            task          = task,
            sim           = "none",
            model         = "transformer",
            root_path     = str(self.work_dir),
            processed_path = str(processed),
            train_path    = str(self._train_csv),
            # Use vocab_ref.csv as the "test" path for ProcessTransformer so that
            # role discovery always succeeds (enough users) and the vocabulary
            # covers every activity seen at inference time.
            test_path     = str(self._vocab_csv),
            result_path   = str(result),
            time_format   = self.params.get("time_format", "%Y-%m-%d %H:%M:%S"),
            rp_sim        = self.params.get("rp_sim", 0.85),
            temp_files    = [],
        )
        args.model_path = str(result / f"{self.run_name}_{task}.weights.h5")
        return args

    def _train_task(self, task: str):
        args = self._make_args(task)

        _weights_exist = Path(args.model_path).exists()


        _proc_prefix   = f"{args.sim}_{args.model}_{task}_"
        _train_proc    = Path(args.processed_path) / f"{_proc_prefix}train_run_0_seed_42.csv"
        _test_proc     = Path(args.processed_path) / f"{_proc_prefix}test_run_0_seed_42.csv"
        _meta_json     = (Path(args.processed_path) /
                          f"{args.dataset}#{args.sim}#{args.task}#run0#metadata.json")

        _proc_done  = Path(args.processed_path) / f"{_proc_prefix}train_run_0_seed_42.done"
        if (_train_proc.exists() and _test_proc.exists()
                and _meta_json.exists() and not _proc_done.exists()):
            _proc_done.touch()  # migrate pre-sentinel runs
        _proc_ready = (_train_proc.exists() and _test_proc.exists()
                       and _meta_json.exists() and _proc_done.exists())

        if _proc_ready:
            args.processed_train_path = str(_train_proc)
            args.processed_test_path  = str(_test_proc)
            args.meta_data_path       = str(_meta_json)
            print(f"[{task}] cached: processed CSVs exist → skipping process_logs()")
        else:
            processor = LogsDataProcessor(
                args    = args,
                run     = 0,
                seed    = 42,
                columns = ["case_id", "activity_name", "end_timestamp",
                           "start_timestamp", "resource"],
                logger  = _LOG,
                pool    = 4,
            )
            processor.process_logs()
            args = processor.return_args()
            _proc_done.touch()

        loader = LogsDataLoader(args=args)
        (train_df, test_df, x_word_dict, y_word_dict,
         max_case_length, vocab_size, num_output) = loader.load_data()

        ep = self.params.get("epochs", 50)
        bs = self.params.get("batch_size", 12)
        lr = self.params.get("learning_rate", 0.001)
        nh = self.params.get("num_heads", 4)


        _config_path = self.work_dir / "config.json"
        if _weights_exist and _config_path.exists():
            try:
                _saved_nh = json.loads(_config_path.read_text()).get("num_heads")
                if _saved_nh is not None:
                    nh = int(_saved_nh)
            except Exception:
                pass

        _done_json = Path(args.model_path + ".json")
        if _weights_exist and not _done_json.exists():
            _done_json.write_text(json.dumps({"epochs_done": ep, "epochs_requested": ep}))
        _weights_complete = (
            _weights_exist
            and _done_json.exists()
            and json.loads(_done_json.read_text()).get("epochs_done", 0) >= ep
        )
        if _weights_exist and not _weights_complete:
            print(f"[{task}] incomplete training detected — deleting weights, will retrain")
            Path(args.model_path).unlink(missing_ok=True)
            _done_json.unlink(missing_ok=True)

        if task == "next_act":
            model = _tf_models.get_next_activity_model(
                max_case_length=max_case_length,
                vocab_size=vocab_size, output_dim=num_output, num_heads=nh)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
            if _weights_complete:
                try:
                    model.load_weights(args.model_path)
                except Exception as _e:
                    print(f"[{task}] weights incompatible ({type(_e).__name__}), retraining")
                    Path(args.model_path).unlink(missing_ok=True)
                    _done_json.unlink(missing_ok=True)
                    train_x, train_y = loader.prepare_data_next_activity(
                        train_df, x_word_dict, y_word_dict, max_case_length)
                    _fit(model, train_x, train_y, ep, bs, args.model_path, "sparse_categorical_accuracy", "max")
            else:
                train_x, train_y = loader.prepare_data_next_activity(
                    train_df, x_word_dict, y_word_dict, max_case_length)
                _fit(model, train_x, train_y, ep, bs, args.model_path, "sparse_categorical_accuracy", "max")

            self._meta["next_act"] = {
                "x_word_dict":     x_word_dict,
                "inv_y":           {v: k for k, v in y_word_dict.items()},
                "max_case_length": max_case_length,
                "vocab_size":      vocab_size,
                "num_output":      num_output,
                "num_heads":       nh,
                "model_path":      args.model_path,
            }

        elif task == "next_role":
            model = _tf_models.get_next_role_model(
                max_case_length=max_case_length,
                vocab_size=vocab_size, output_dim=num_output, num_heads=nh)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
            if _weights_complete:
                try:
                    model.load_weights(args.model_path)
                except Exception as _e:
                    print(f"[{task}] weights incompatible ({type(_e).__name__}), retraining")
                    Path(args.model_path).unlink(missing_ok=True)
                    _done_json.unlink(missing_ok=True)
                    train_x, train_y = loader.prepare_data_next_role(
                        train_df, x_word_dict, y_word_dict, max_case_length)
                    _fit(model, train_x, train_y, ep, bs, args.model_path, "sparse_categorical_accuracy", "max")
            else:
                train_x, train_y = loader.prepare_data_next_role(
                    train_df, x_word_dict, y_word_dict, max_case_length)
                _fit(model, train_x, train_y, ep, bs, args.model_path, "sparse_categorical_accuracy", "max")

            self._meta["next_role"] = {
                "x_word_dict":     x_word_dict,   # role vocab
                "inv_y":           {v: k for k, v in y_word_dict.items()},
                "max_case_length": max_case_length,
                "vocab_size":      vocab_size,
                "num_output":      num_output,
                "num_heads":       nh,
                "model_path":      args.model_path,
            }

        elif task == "next_time":
            train_x, train_tx, train_y, t_sc, y_sc = loader.prepare_data_next_time(
                train_df, x_word_dict, max_case_length)
            model = _tf_models.get_next_time_model(
                max_case_length=max_case_length, vocab_size=vocab_size, num_heads=nh)
            model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                          loss=tf.keras.losses.LogCosh())
            if _weights_complete:
                try:
                    model.load_weights(args.model_path)
                except Exception as _e:
                    print(f"[{task}] weights incompatible ({type(_e).__name__}), retraining")
                    Path(args.model_path).unlink(missing_ok=True)
                    _done_json.unlink(missing_ok=True)
                    _fit(model, [train_x, train_tx], train_y, ep, bs, args.model_path, "loss", "min")
            else:
                _fit(model, [train_x, train_tx], train_y, ep, bs, args.model_path, "loss", "min")

            self._meta["next_time"] = {
                "x_word_dict":     x_word_dict,
                "max_case_length": max_case_length,
                "vocab_size":      vocab_size,
                "num_heads":       nh,
                "time_scaler":     t_sc,
                "y_scaler":        y_sc,
                "model_path":      args.model_path,
            }

        elif task == "rem_time":
            train_x, train_tx, train_y, t_sc, y_sc = loader.prepare_data_remaining_time(
                train_df, x_word_dict, max_case_length)
            model = _tf_models.get_remaining_time_model(
                max_case_length=max_case_length, vocab_size=vocab_size, num_heads=nh)
            model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                          loss=tf.keras.losses.LogCosh())
            if _weights_complete:
                try:
                    model.load_weights(args.model_path)
                except Exception as _e:
                    print(f"[{task}] weights incompatible ({type(_e).__name__}), retraining")
                    Path(args.model_path).unlink(missing_ok=True)
                    _done_json.unlink(missing_ok=True)
                    _fit(model, [train_x, train_tx], train_y, ep, bs, args.model_path, "loss", "min")
            else:
                _fit(model, [train_x, train_tx], train_y, ep, bs, args.model_path, "loss", "min")

            self._meta["rem_time"] = {
                "x_word_dict":     x_word_dict,
                "max_case_length": max_case_length,
                "vocab_size":      vocab_size,
                "num_heads":       nh,
                "time_scaler":     t_sc,
                "y_scaler":        y_sc,
                "model_path":      args.model_path,
            }

        self._models[task] = model
        print(f"[{task}] done. weights → {args.model_path}")

    # ── Model persistence ──────────────────────────────────────────────────────

    def _save_meta(self):
        """Persist non-TF meta (vocabs, scalers) so predict() can reload after kernel restart."""
        import pickle
        meta_path = self.work_dir / "meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(self._meta, f)
        print(f"[save_meta] → {meta_path}")

    def _load_all_models(self):
        """Reload saved model weights into fresh TF model instances."""
        meta_path = self.work_dir / "meta.pkl"
        if not meta_path.exists():
            raise RuntimeError(
                f"No meta.pkl found in {self.work_dir}. Call run() first.")
        import pickle
        with open(meta_path, "rb") as f:
            self._meta = pickle.load(f)

        for task, meta in self._meta.items():
            mcl = meta["max_case_length"]
            vs  = meta["vocab_size"]
            nh  = meta.get("num_heads", self.params.get("num_heads", 4))
            if task == "next_act":
                m = _tf_models.get_next_activity_model(
                    max_case_length=mcl, vocab_size=vs,
                    output_dim=meta["num_output"], num_heads=nh)
            elif task == "next_role":
                m = _tf_models.get_next_role_model(
                    max_case_length=mcl, vocab_size=vs,
                    output_dim=meta["num_output"], num_heads=nh)
            elif task == "next_time":
                m = _tf_models.get_next_time_model(
                    max_case_length=mcl, vocab_size=vs, num_heads=nh)
            elif task == "rem_time":
                m = _tf_models.get_remaining_time_model(
                    max_case_length=mcl, vocab_size=vs, num_heads=nh)
            else:
                continue


            model_path = Path(meta["model_path"])
            if not model_path.exists():
                candidates = [
                    self.work_dir / "results" / model_path.name,
                    self.work_dir.parent / "results" / model_path.name,
                ]
                local_path = next((c for c in candidates if c.exists()), None)
                if local_path is None:
                    raise FileNotFoundError(
                        f"Weights for task {task!r} not found at recorded path "
                        f"{model_path} or local fallbacks {candidates}")
                model_path = local_path
            m.load_weights(str(model_path))
            self._models[task] = m
            print(f"[load] {task} ← {model_path}")

        # Restore CSV paths so _build_role_map() works without run() being called.
        if self._train_csv is None:
            self._train_csv = self.work_dir / "train.csv"
        if self._vocab_csv is None:
            self._vocab_csv = self.work_dir / "vocab_ref.csv"
        if self._test_csv is None:
            self._test_csv = self.work_dir / "test.csv"

    # ── Role mapping ───────────────────────────────────────────────────────────

    def _build_role_map(self):
        """Discover user→role mapping from training CSV (one-time)."""
        if self._role_map is not None:
            return
        from pre_process import lstm_role_discovery as rl

        train_csv = pd.read_csv(self._train_csv)
        df_roles  = train_csv.rename(columns={
            "activity_name": "task",
            "resource":      "user",
        })
        analyser  = rl.ResourcePoolAnalyser(
            df_roles, sim_threshold=self.params.get("rp_sim", 0.85))
        resources = pd.DataFrame.from_records(analyser.resource_table)
        self._role_map = {
            str(r): _normalize(role)
            for r, role in zip(resources["resource"], resources["role"])
        }

    def _get_role(self, user) -> str:
        if self._role_map is None:
            return "unk"
        return self._role_map.get(str(user), "unk")

    # ── Inference helpers ──────────────────────────────────────────────────────

    def _encode_prefix(self, tokens_list: list, x_word_dict: dict,
                       max_case_length: int) -> np.ndarray:
        """Encode a list of activity/role names as a padded (1, max_case_length) array."""
        unk = x_word_dict.get("[UNK]", 1)
        ids = [x_word_dict.get(t, unk) for t in tokens_list]
        # Pre-pad with zeros; truncate from left if too long
        if len(ids) < max_case_length:
            ids = [0] * (max_case_length - len(ids)) + ids
        else:
            ids = ids[-max_case_length:]
        return np.array([ids], dtype=np.float32)

    def _predict_rem_time(self, acts: list, time_feats: list) -> float:
        """Predict remaining time in seconds for a prefix."""
        meta  = self._meta["rem_time"]
        model = self._models["rem_time"]

        token_x = self._encode_prefix(acts, meta["x_word_dict"],
                                      meta["max_case_length"])
        time_x  = meta["time_scaler"].transform(
            np.array([time_feats], dtype=np.float32))
        pred    = model([token_x, time_x], training=False)
        return float(
            meta["y_scaler"].inverse_transform(
                np.array(pred).reshape(-1, 1))[0][0])

    def _hallucinate(
        self,
        acts:       list,
        roles:      list,
        anchor_ts:  datetime.datetime,
        time_feats: list,
    ) -> list:
        """
        Recursively predict the suffix of a case.

        Parameters
        ----------
        acts       : normalised activity names seen so far (prefix)
        roles      : normalised role names corresponding to acts
        anchor_ts  : absolute datetime of the last known event (tz-naive)
        time_feats : [recent_wait, recent_proc, latest_wait, latest_proc, time_passed]
                     all in seconds

        Returns
        -------
        list of (activity_str, role_str, absolute_datetime)
        """
        meta_act  = self._meta["next_act"]
        meta_role = self._meta["next_role"]
        meta_time = self._meta["next_time"]
        mdl_act   = self._models["next_act"]
        mdl_role  = self._models["next_role"]
        mdl_time  = self._models["next_time"]

        max_steps = self.params.get("max_suffix_length", 50)

        cur_acts  = list(acts)
        cur_roles = list(roles)
        recent_wait, recent_proc, latest_wait, latest_proc, time_passed = time_feats

        events = []
        for _ in range(max_steps):
            # ── predict next activity ─────────────────────────────────────────
            tx_act = self._encode_prefix(cur_acts, meta_act["x_word_dict"],
                                         meta_act["max_case_length"])
            logits_act = mdl_act(tx_act, training=False)
            next_act   = meta_act["inv_y"].get(int(np.argmax(logits_act[0])), _END)

            if next_act == _END:
                events.append((_END, _END, None))
                break

            # ── predict next role ─────────────────────────────────────────────
            tx_role    = self._encode_prefix(cur_roles, meta_role["x_word_dict"],
                                             meta_role["max_case_length"])
            logits_role = mdl_role(tx_role, training=False)
            next_role   = meta_role["inv_y"].get(int(np.argmax(logits_role[0])), "unk")

            # ── predict next time ─────────────────────────────────────────────
            tx_time = self._encode_prefix(cur_acts, meta_time["x_word_dict"],
                                          meta_time["max_case_length"])
            time_x_raw = np.array(
                [[recent_wait, recent_proc, latest_wait, latest_proc, time_passed]],
                dtype=np.float32)
            time_x_sc  = meta_time["time_scaler"].transform(time_x_raw)
            time_pred  = meta_time["y_scaler"].inverse_transform(
                np.array(mdl_time([tx_time, time_x_sc], training=False)))[0]

            next_wait = max(0.0, float(time_pred[0]))
            next_proc = max(0.0, float(time_pred[1]))
            delta     = next_wait + next_proc
            time_passed += delta

            abs_ts = anchor_ts + datetime.timedelta(seconds=time_passed)
            events.append((next_act, next_role, abs_ts))

            # update prefix & time features
            cur_acts.append(next_act)
            cur_roles.append(next_role)
            recent_wait, recent_proc = latest_wait, latest_proc
            latest_wait, latest_proc = next_wait, next_proc

        return events


# ── Module-level helpers ───────────────────────────────────────────────────────

def _strip_tz(ts) -> datetime.datetime:
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if isinstance(ts, datetime.datetime) and ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts


def _compute_time_feats(timestamps: list) -> list:
    """
    Compute time features from a list of naive datetime objects.

    Returns [recent_wait, recent_proc, latest_wait, latest_proc, time_passed]
    in seconds.  Processing time is always 0 (start_ts == end_ts).
    """
    times = [_strip_tz(t) for t in timestamps]
    gaps  = []
    for i in range(1, len(times)):
        gaps.append(max(0.0, (times[i] - times[i - 1]).total_seconds()))

    time_passed = sum(gaps)
    latest_wait = gaps[-1] if gaps else 0.0
    recent_wait = gaps[-2] if len(gaps) >= 2 else 0.0
    return [recent_wait, 0.0, latest_wait, 0.0, time_passed]


def _fit(model, x, y, epochs, batch_size, ckpt_path, monitor, mode):
    """Compile-and-fit wrapper with model checkpoint."""
    cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path, save_weights_only=True,
        monitor=monitor, mode=mode, save_best_only=True)
    history = model.fit(x, y, epochs=epochs, batch_size=batch_size,
                        verbose=1, shuffle=True, callbacks=[cb])
    if Path(ckpt_path).exists():
        model.load_weights(ckpt_path)
    else:
        # save_best_only=True never fired (metric was NaN every epoch);
        # persist current weights so _load_all_models can always find the file.
        model.save_weights(ckpt_path)
    # Record completion so the next run can distinguish fully-trained models
    # from ones where training was interrupted after the first checkpoint save.
    epochs_done = len(history.history.get("loss", []))
    Path(ckpt_path + ".json").write_text(
        json.dumps({"epochs_done": epochs_done, "epochs_requested": epochs}))
