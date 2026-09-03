# -*- coding: utf-8 -*-
"""
InterCaseBukhshTrainer -- the Bukhsh `rem_time` model augmented with
LS-ICE-style inter-case ("load state") features, computed identically at
training and inference time (see features.py for the feature definitions
and README.md for the full rationale).

Usage
-----
    from importlib import import_module
    import sys
    sys.path.insert(0, "inter-case-bukhsh")
    from trainer import InterCaseBukhshTrainer

    trainer = InterCaseBukhshTrainer(
        train_df, val_df, test_df, "helpdesk_ic", params,
        output_dir="inter_case_bukhsh_output/helpdesk/intercase",
        use_intercase_features=True,
    )
    trainer.run()
    metrics = trainer.evaluate()          # MAE / RMSE across all test prefixes
    rem_df  = trainer.predict_rem_time()  # per-case anchor-point prediction
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
PT_DIR = _ROOT / "processtransformer"
for _p in (str(PT_DIR), str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402
from sklearn import preprocessing, utils as sk_utils  # noqa: E402

from pre_process.transformer_processor import LogsDataProcessor  # noqa: E402
from pre_process.transformer_loader import LogsDataLoader  # noqa: E402

from models_ic import get_remaining_time_model_ic  # noqa: E402
from features import build_transition_table, build_load_state_table, ic_feature_cols  # noqa: E402

# Reuse Bukhsh's own tiny, pure-Python helpers instead of duplicating them.
from bukhsh.trainer import (  # noqa: E402
    BukhshTrainer as _Base,
    _compute_time_feats,
    _strip_tz,
    _normalize,
)

BASE_TIME_COLS = [
    "recent_wait_time", "recent_proc_time",
    "latest_wait_time", "latest_proc_time", "time_passed",
]


def _fit(model, x, y, epochs, batch_size, ckpt_path, monitor, mode):
    """
    Same as bukhsh.trainer._fit, except verbose=2 (one clean line per epoch)
    instead of verbose=1 (a per-step progress bar). v
    """
    cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path, save_weights_only=True,
        monitor=monitor, mode=mode, save_best_only=True)
    history = model.fit(x, y, epochs=epochs, batch_size=batch_size,
                        verbose=2, shuffle=True, callbacks=[cb])
    if Path(ckpt_path).exists():
        model.load_weights(ckpt_path)
    else:
        model.save_weights(ckpt_path)
    epochs_done = len(history.history.get("loss", []))
    Path(ckpt_path + ".json").write_text(
        json.dumps({"epochs_done": epochs_done, "epochs_requested": epochs}))


def _prepare_data_remaining_time(
    df: pd.DataFrame,
    x_word_dict: dict,
    max_case_length: int,
    feature_cols: list[str],
    time_scaler=None,
    y_scaler=None,
    inference: bool = False,
):
    """
    """
    x = df["prefix"].values
    time_x = df[feature_cols].values.astype(np.float32)
    y = df["remaining_time"].values.astype(np.float32)
    if not inference:
        x, time_x, y = sk_utils.shuffle(x, time_x, y)

    unk = x_word_dict.get("[UNK]", 1)
    token_x = [[x_word_dict.get(s, unk) for s in _x.split()] for _x in x]

    if time_scaler is None:
        time_scaler = preprocessing.StandardScaler()
        time_x = time_scaler.fit_transform(time_x).astype(np.float32)
    else:
        time_x = time_scaler.transform(time_x).astype(np.float32)

    if y_scaler is None:
        y_scaler = preprocessing.StandardScaler()
        y = y_scaler.fit_transform(y.reshape(-1, 1)).astype(np.float32)
    else:
        y = y_scaler.transform(y.reshape(-1, 1)).astype(np.float32)

    token_x = tf.keras.preprocessing.sequence.pad_sequences(token_x, maxlen=max_case_length)
    token_x = np.array(token_x, dtype=np.float32)
    time_x = np.array(time_x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return token_x, time_x, y, time_scaler, y_scaler


class InterCaseBukhshTrainer:
    """
    Trains (and predicts with) the Bukhsh ``rem_time`` model only,
    optionally augmented with LS-ICE-style inter-case load-state features.

    Parameters
    ----------
    use_intercase_features : bool
        True  -> BASE_TIME_COLS + ic_feature_cols(top_n_next)  (12 feats by default)
        False -> BASE_TIME_COLS only (5 feats) -- an otherwise IDENTICAL
                 code path, so a baseline run and an inter-case run differ
                 ONLY in whether the load-state columns are computed and
                 fed in, nothing else (same data, same architecture, same
                 hyperparameters).
    top_n_next : int
        How many most-likely-next load points to track (LS-ICE default: 5).
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        run_name: str,
        params: dict,
        output_dir: Optional[str | Path] = None,
        use_intercase_features: bool = True,
        top_n_next: int = 5,
        full_df: Optional[pd.DataFrame] = None,
    ):
        self.train_df = train_df.copy()
        self.val_df = val_df.copy()
        self.test_df = test_df.copy()
        self.run_name = run_name
        self.params = dict(params)
        self.use_ic = use_intercase_features
        self.top_n = top_n_next
        self.full_df = full_df

        self.feature_cols = list(BASE_TIME_COLS)
        if self.use_ic:
            self.feature_cols += ic_feature_cols(self.top_n)

        self.work_dir = (
            Path(output_dir) if output_dir else Path("inter_case_bukhsh_output") / run_name
        )
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._meta: dict = {}
        self._load_state_table: Optional[pd.DataFrame] = None
        self._ic_index: dict = {}

        self._train_csv: Optional[Path] = None
        self._vocab_csv: Optional[Path] = None

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self):
        """Prepare CSVs, build inter-case features, train the rem_time model."""
        self._prepare_csvs()
        self._build_load_state_table()

        args = self._make_args()
        weights_path = Path(args.model_path)

        proc_prefix = f"{args.sim}_{args.model}_rem_time_"
        train_proc = Path(args.processed_path) / f"{proc_prefix}train_run_0_seed_42.csv"
        test_proc = Path(args.processed_path) / f"{proc_prefix}test_run_0_seed_42.csv"
        meta_json = (
            Path(args.processed_path)
            / f"{args.dataset}#{args.sim}#rem_time#run0#metadata.json"
        )

        if train_proc.exists() and test_proc.exists() and meta_json.exists():
            args.processed_train_path = str(train_proc)
            args.processed_test_path = str(test_proc)
            args.meta_data_path = str(meta_json)
            print("[rem_time] cached processed CSVs found -> skipping process_logs()")
        else:
            processor = LogsDataProcessor(
                args=args, run=0, seed=42,
                columns=["case_id", "activity_name", "end_timestamp",
                        "start_timestamp", "resource"],
                logger=None, pool=4,
            )
            processor.process_logs()
            args = processor.return_args()

        loader = LogsDataLoader(args=args)
        (train_df, _test_df_processed, x_word_dict, _y_word_dict,
         max_case_length, vocab_size, _num_output) = loader.load_data()

        train_df = self._merge_ic(train_df)

        ep = self.params.get("epochs", 50)
        bs = self.params.get("batch_size", 12)
        lr = self.params.get("learning_rate", 0.001)
        nh = self.params.get("num_heads", 4)

        train_x, train_tx, train_y, t_sc, y_sc = _prepare_data_remaining_time(
            train_df, x_word_dict, max_case_length, self.feature_cols
        )

        model = get_remaining_time_model_ic(
            max_case_length=max_case_length, vocab_size=vocab_size,
            num_time_feats=len(self.feature_cols), num_heads=nh,
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=tf.keras.losses.LogCosh())

        if weights_path.exists():
            print(f"[rem_time] weights found on disk ({weights_path}) -> loading")
            model.load_weights(str(weights_path))
        else:
            _fit(model, [train_x, train_tx], train_y, ep, bs, str(weights_path), "loss", "min")

        self._model = model
        self._meta = {
            "x_word_dict": x_word_dict,
            "max_case_length": max_case_length,
            "vocab_size": vocab_size,
            "num_heads": nh,
            "time_scaler": t_sc,
            "y_scaler": y_sc,
            "feature_cols": self.feature_cols,
            "model_path": str(weights_path),
            "use_ic": self.use_ic,
            "top_n": self.top_n,
        }
        self._save_meta()
        print(f"[rem_time] done. weights -> {weights_path}  "
              f"({len(self.feature_cols)} numeric features: {self.feature_cols})")

    def evaluate(self) -> dict:
        """
        """
        if self._model is None:
            self._load_model()

        args = self._make_args()
        proc_prefix = f"{args.sim}_{args.model}_rem_time_"
        test_proc = Path(args.processed_path) / f"{proc_prefix}test_run_0_seed_42.csv"
        test_df = pd.read_csv(test_proc)

        test_ids = set(self.test_df["caseid"].astype(str))
        test_df["case_id"] = test_df["case_id"].astype(str)
        test_df = test_df[test_df["case_id"].isin(test_ids)].copy()
        test_df = self._merge_ic(test_df)

        meta = self._meta
        token_x, time_x, y_true, _, _ = _prepare_data_remaining_time(
            test_df, meta["x_word_dict"], meta["max_case_length"], self.feature_cols,
            time_scaler=meta["time_scaler"], y_scaler=meta["y_scaler"], inference=True,
        )

        pred = self._model([token_x, time_x], training=False)
        y_pred_secs = meta["y_scaler"].inverse_transform(np.array(pred).reshape(-1, 1)).ravel()
        y_true_secs = meta["y_scaler"].inverse_transform(y_true.reshape(-1, 1)).ravel()

        err = y_pred_secs - y_true_secs
        mae_days = float(np.mean(np.abs(err))) / 86400
        rmse_days = float(np.sqrt(np.mean(err ** 2))) / 86400

        result = {
            "run_name": self.run_name,
            "use_intercase_features": self.use_ic,
            "n_prefixes": int(len(test_df)),
            "n_cases": int(test_df["case_id"].nunique()),
            "mae_days": mae_days,
            "rmse_days": rmse_days,
        }
        (self.work_dir / "eval_metrics.json").write_text(json.dumps(result, indent=2))
        print(f"[eval] {self.run_name}  n_prefixes={len(test_df):,}  "
              f"MAE={mae_days:.3f}d  RMSE={rmse_days:.3f}d")
        return result

    def predict_rem_time(self) -> pd.DataFrame:
        """
        """
        if self._model is None:
            self._load_model()

        meta = self._meta
        ic_cols = ic_feature_cols(self.top_n) if self.use_ic else []
        rows = []
        for case_id, case_df in self.test_df.groupby("caseid"):
            case_df = case_df.sort_values("end_timestamp")
            acts = [_normalize(a) for a in case_df["task"]]
            ts = [_strip_tz(t) for t in pd.to_datetime(case_df["end_timestamp"]).tolist()]
            time_feats = _compute_time_feats(ts)
            anchor_ts = ts[-1]
            k = len(acts)

            ic_feats = self._lookup_ic(str(case_id), k) if self.use_ic else []
            full_feats = list(time_feats) + list(ic_feats)

            token_x = self._encode_prefix(acts, meta["x_word_dict"], meta["max_case_length"])
            time_x = meta["time_scaler"].transform(np.array([full_feats], dtype=np.float32))
            pred = self._model([token_x, time_x], training=False)
            rem_secs = float(
                meta["y_scaler"].inverse_transform(np.array(pred).reshape(-1, 1))[0][0]
            )

            rows.append({
                "caseid": case_id,
                "start_timestamp": ts[0],
                "anchor_timestamp": anchor_ts,
                "prefix_len": k,
                "rem_time_days": max(0.0, rem_secs) / 86400,
            })

        df = pd.DataFrame(rows)
        out = self.work_dir / "rem_time.csv"
        df.to_csv(out, index=False)
        print(f"[predict] rem_time -> {out} ({len(df):,} rows)")
        return df

    # ── CSV preparation (reuses BukhshTrainer's own static CSV writer) ─────────

    def _prepare_csvs(self):
        full_train = pd.concat([self.train_df, self.val_df], ignore_index=True)
        full_all = pd.concat([self.train_df, self.val_df, self.test_df], ignore_index=True)

        train_csv = _Base._to_pt_csv(full_train, add_end_events=True)
        vocab_csv = _Base._to_pt_csv(full_all, add_end_events=False)

        self._train_csv = self.work_dir / "train.csv"
        self._vocab_csv = self.work_dir / "vocab_ref.csv"
        train_csv.to_csv(self._train_csv, index=False)
        vocab_csv.to_csv(self._vocab_csv, index=False)
        print(f"[prepare] train: {len(train_csv):,} rows | vocab_ref: {len(vocab_csv):,} rows")

    def _make_args(self):
        # BukhshTrainer._make_args only touches attributes this class also
        # defines (work_dir, run_name, params, _train_csv, _vocab_csv), so
        # it can be called unbound against our own instance.
        return _Base._make_args(self, "rem_time")

    def _encode_prefix(self, tokens, x_word_dict, max_case_length):
        return _Base._encode_prefix(self, tokens, x_word_dict, max_case_length)

    # ── Inter-case ("load state") features ─────────────────────────────────────

    def _events_df(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df[["caseid", "task", "end_timestamp"]].copy()
        d["caseid"] = d["caseid"].astype(str)
        d["activity"] = d["task"].map(_normalize)
        d["timestamp"] = pd.to_datetime(d["end_timestamp"]).map(_strip_tz)
        return d[["caseid", "activity", "timestamp"]]

    def _build_load_state_table(self):
        if not self.use_ic:
            self._load_state_table = None
            self._ic_index = {}
            return

        cache = self.work_dir / "load_state_features.csv"
        if cache.exists():
            self._load_state_table = pd.read_csv(cache)
        else:
            trainval_events = pd.concat(
                [self._events_df(self.train_df), self._events_df(self.val_df)],
                ignore_index=True,
            )
            if self.full_df is not None:
                all_events = self._events_df(self.full_df)
            else:
                all_events = pd.concat(
                    [trainval_events, self._events_df(self.test_df)], ignore_index=True,
                )
            transition_table = build_transition_table(trainval_events, top_n=self.top_n)
            self._load_state_table = build_load_state_table(
                all_events, transition_table, top_n=self.top_n
            )
            self._load_state_table.to_csv(cache, index=False)

        self._index_load_state_table()
        print(f"[ic] load-state table ready: {len(self._load_state_table):,} (case,k) rows")

    def _index_load_state_table(self):
        if self._load_state_table is None:
            self._ic_index = {}
            return
        cols = ic_feature_cols(self.top_n)
        t = self._load_state_table
        self._ic_index = {
            (str(row[0]), int(row[1])): list(row[2:])
            for row in t[["caseid", "k"] + cols].itertuples(index=False, name=None)
        }

    def _lookup_ic(self, case_id: str, k: int) -> list[float]:
        default = [0.0] * len(ic_feature_cols(self.top_n))
        return self._ic_index.get((case_id, k), default)

    def _merge_ic(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.use_ic:
            return df
        d = df.copy()
        d["case_id"] = d["case_id"].astype(str)
        table = self._load_state_table.copy()
        table["caseid"] = table["caseid"].astype(str)
        merged = d.merge(table, left_on=["case_id", "k"], right_on=["caseid", "k"], how="left")
        ic_cols = ic_feature_cols(self.top_n)
        merged[ic_cols] = merged[ic_cols].fillna(0.0)
        return merged

    # ── Persistence ─────────────────────────────────────────────────────────────

    def _save_meta(self):
        import pickle

        with open(self.work_dir / "meta.pkl", "wb") as f:
            pickle.dump(self._meta, f)
        print(f"[save_meta] -> {self.work_dir / 'meta.pkl'}")

    def _load_model(self):
        import pickle

        meta_path = self.work_dir / "meta.pkl"
        if not meta_path.exists():
            raise RuntimeError(f"No meta.pkl found in {self.work_dir}. Call run() first.")
        with open(meta_path, "rb") as f:
            self._meta = pickle.load(f)

        self.use_ic = self._meta.get("use_ic", self.use_ic)
        self.top_n = self._meta.get("top_n", self.top_n)
        self.feature_cols = self._meta.get("feature_cols", self.feature_cols)

        model = get_remaining_time_model_ic(
            max_case_length=self._meta["max_case_length"],
            vocab_size=self._meta["vocab_size"],
            num_time_feats=len(self.feature_cols),
            num_heads=self._meta.get("num_heads", 4),
        )
        model.load_weights(self._meta["model_path"])
        self._model = model

        self._build_load_state_table()
        print(f"[load] rem_time (intercase={self.use_ic}) <- {self._meta['model_path']}")
