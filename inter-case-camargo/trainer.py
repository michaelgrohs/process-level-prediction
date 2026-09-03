# -*- coding: utf-8 -*-
"""
InterCaseCamargoTrainer -- adds the 7 LS-ICE-style inter-case ("load state")
features to the Camargo suffix-prediction pipeline.

"""
from __future__ import annotations

import copy
import heapq
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from camargo.trainer import CamargoTrainer, REPO_DIR, _repo_cwd  # noqa: E402

from features import (  # noqa: E402
    DEFAULT_TOP_N,
    attach_ic_features,
    build_load_state_table,
    build_transition_table,
    ic_feature_cols,
    load_state_features_for,
    sweep_state_upto,
)


class InterCaseCamargoTrainer(CamargoTrainer):

    def __init__(self, train_df, val_df, test_df, run_name, params,
                 trim_dir: str = "", full_df: pd.DataFrame | None = None,
                 top_n_next: int = DEFAULT_TOP_N,
                 val_as_test_df: pd.DataFrame | None = None,
                 cc_val: pd.Series | None = None, tt_val: pd.Series | None = None,
                 train_split=None):
        self.full_df = full_df.copy() if full_df is not None else None
        self.top_n = top_n_next
        self._transition_table: dict | None = None
        self._load_state_table: pd.DataFrame | None = None
        self._ic_scale_max: dict | None = None
        self._full_df_by_case_cache: dict | None = None
        self._val_train_split = train_split
        super().__init__(train_df, val_df, test_df, run_name, params, trim_dir=trim_dir,
                         val_as_test_df=val_as_test_df, cc_val=cc_val, tt_val=tt_val)

    # ── inter-case feature wiring ──────────────────────────────────────────────

    @staticmethod
    def _events_df(df: pd.DataFrame) -> pd.DataFrame:
        d = df[["caseid", "task", "end_timestamp"]].copy()
        d["caseid"] = d["caseid"].astype(str)
        d["activity"] = d["task"]
        ts = pd.to_datetime(d["end_timestamp"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(None)
        d["timestamp"] = ts
        return d[["caseid", "activity", "timestamp"]]

    def _ensure_ic_features(self):
        """
        """
        if self._load_state_table is not None:
            return

        work_dir = REPO_DIR / self._out_base()
        work_dir.mkdir(parents=True, exist_ok=True)
        cache = work_dir / "load_state_features.csv"
        ic_meta_path = work_dir / "ic_meta.json"

        if ic_meta_path.exists():
            ic_meta = json.loads(ic_meta_path.read_text())
            self._transition_table = ic_meta["transition_table"]
            self._ic_scale_max = ic_meta["ic_scale_max"]
        else:
            trainval_events = pd.concat(
                [self._events_df(self.train_df), self._events_df(self.val_df)],
                ignore_index=True,
            )
            self._transition_table = build_transition_table(trainval_events, top_n=self.top_n)

        if cache.exists():
            self._load_state_table = pd.read_csv(cache)
        else:
            if self.full_df is not None:
                all_events = self._events_df(self.full_df)
            else:
                all_events = pd.concat(
                    [self._events_df(self.train_df), self._events_df(self.val_df),
                     self._events_df(self.test_df)], ignore_index=True,
                )
            self._load_state_table = build_load_state_table(
                all_events, self._transition_table, top_n=self.top_n
            )
            self._load_state_table.to_csv(cache, index=False)
        print(f"[ic] load-state table ready: {len(self._load_state_table):,} (case,k) rows")

        full_for_merge = (
            self.full_df if self.full_df is not None
            else pd.concat([self.train_df, self.val_df, self.test_df], ignore_index=True)
        )
        for attr in ("train_df", "val_df", "test_df"):
            df = getattr(self, attr)
            merged = attach_ic_features(
                df, full_for_merge, self._load_state_table, top_n=self.top_n
            )
            setattr(self, attr, merged)

        if not ic_meta_path.exists():
            ic_cols = ic_feature_cols(self.top_n)
            trainval_ic = pd.concat(
                [self.train_df[ic_cols], self.val_df[ic_cols]], ignore_index=True
            )
            self._ic_scale_max = {
                col: (float(trainval_ic[col].max()) if trainval_ic[col].max() > 0 else 0.0)
                for col in ic_cols
            }
            ic_meta_path.write_text(json.dumps({
                "transition_table": self._transition_table,
                "ic_scale_max": self._ic_scale_max,
            }))

    def train(self):
        super().train()
        self._persist_ic_meta()

    def _persist_ic_meta(self):
        work_dir = REPO_DIR / self._out_base()
        work_dir.mkdir(parents=True, exist_ok=True)
        self._load_state_table.to_csv(work_dir / "load_state_features.csv", index=False)
        (work_dir / "ic_meta.json").write_text(json.dumps({
            "transition_table": self._transition_table,
            "ic_scale_max": self._ic_scale_max,
        }))

    def _select_trial_by_val_cc_mae(self, hpo_dir: str):
        if self._val_train_split is None:
            print("[train] WARNING: InterCaseCamargoTrainer has no train_split "
                  "(needed as predict()'s anchor_ts) -- val-CC-MAE rescoring "
                  "skipped, falling back to vectorized-loss winner.")
            return None

        trial_dirs = sorted(Path(hpo_dir).glob("trial_*"))
        if not trial_dirs:
            return None

        from hpo_val_scoring import kpi_mae

        anchor_ts = self._val_train_split
        val_raw = self.val_as_test_df.copy()
        rescore_trim_dir = (f"{self.trim_dir}/{self.run_name}_trials_rescore"
                             if self.trim_dir else f"{self.run_name}_trials_rescore")

        best = None  # (cc_mae, output_dir, params_dict)
        for tdir in trial_dirs:
            result_path = tdir / "result.json"
            h5_files = list(tdir.glob("*.h5"))
            if not result_path.exists() or not h5_files:
                continue
            trial_result = json.loads(result_path.read_text())
            if "scale_args" not in trial_result:
                print(f"  [rescore {tdir.name}] skipped: no scale_args in result.json")
                continue
            trial_params = {k: trial_result.get(k) for k in
                            ("model_type", "n_size", "l_size", "lstm_act",
                             "dense_act", "norm_method", "optim")}
            trial_params["scale_args"] = trial_result["scale_args"]

            scratch = copy.copy(self)
            scratch.run_name = f"{self.run_name}_hpo_rescore_{tdir.name}"
            scratch.trim_dir = rescore_trim_dir
            scratch.best_model_path = str(h5_files[0])
            scratch.output_dir = scratch._out_base()

            try:
                event_log_path = Path(scratch.output_dir) / "event_log.csv"
                if event_log_path.exists():
                    event_log = self._read_csv_safe_caseid(event_log_path)
                else:
                    with _repo_cwd():
                        work_dir = Path(scratch.output_dir)
                        work_dir.mkdir(parents=True, exist_ok=True)
                        scratch._export_params(scratch.output_dir, dict(trial_params))
                        h5_link = work_dir / Path(h5_files[0]).name
                        if not h5_link.exists():
                            h5_link.symlink_to(Path(h5_files[0]).resolve())

                        test_df = val_raw.copy()
                        if "start_timestamp" not in test_df.columns:
                            test_df["start_timestamp"] = test_df["end_timestamp"]
                        test_df = self._apply_roles(test_df, scratch._resource_map)
                        test_df["ac_index"] = test_df["task"].map(scratch.ac_index).fillna(0).astype(int)
                        test_df["rl_index"] = test_df["role"].map(scratch.rl_index).fillna(0).astype(int)
                        scratch.test_df = attach_ic_features(
                            test_df, scratch.full_df, scratch._load_state_table, top_n=scratch.top_n)

                    gen_path = scratch.predict(anchor_ts)
                    event_log = scratch.to_event_log([gen_path])
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

    def preprocess(self):
        self._ensure_ic_features()
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

    @classmethod
    def from_saved(cls, train_df, val_df, test_df, run_name, params,
                   trim_dir: str = "", full_df: pd.DataFrame | None = None,
                   top_n_next: int = DEFAULT_TOP_N) -> "InterCaseCamargoTrainer":
        trainer = cls(train_df, val_df, test_df, run_name, params,
                       trim_dir=trim_dir, full_df=full_df, top_n_next=top_n_next)
        with _repo_cwd():
            params_dir = os.path.join(trainer._out_base(), "parameters")
            params_path = os.path.join(params_dir, "model_parameters.json")
            with open(params_path) as f:
                saved = json.load(f)

            trainer.index_ac = {int(k): v for k, v in saved["index_ac"].items()}
            trainer.ac_index = {v: int(k) for k, v in saved["index_ac"].items()}
            trainer.index_rl = {int(k): v for k, v in saved["index_rl"].items()}
            trainer.rl_index = {v: int(k) for k, v in saved["index_rl"].items()}

            trainer.output_dir = trainer._out_base()
            trainer.best_model_path = trainer._find_model_file(trainer.output_dir)
            trainer.best_loss = saved.get("best_loss", float("inf"))

            if "one_timestamp" in saved:
                trainer.params["one_timestamp"] = saved["one_timestamp"]
                trainer.params["read_options"]["one_timestamp"] = saved["one_timestamp"]

            resource_map_path = os.path.join(params_dir, "resource_map.csv")
            resources = pd.read_csv(resource_map_path)
            trainer._resource_map = resources

            trainer._ensure_ic_features()
            for attr in ("train_df", "val_df", "test_df"):
                df = cls._apply_roles(getattr(trainer, attr), resources)
                df["ac_index"] = df["task"].map(trainer.ac_index).fillna(0).astype(int)
                df["rl_index"] = df["role"].map(trainer.rl_index).fillna(0).astype(int)
                setattr(trainer, attr, df)

        print(f"[from_saved] model  → {trainer.best_model_path}")
        print(f"[from_saved] activities: {len(trainer.ac_index)}  roles: {len(trainer.rl_index)}")
        return trainer

    # ── Joint, time-synchronized suffix generation ─────────────────────────────

    def _full_df_by_case(self) -> dict:
        if self._full_df_by_case_cache is None:
            fdf = self.full_df.copy()
            fts = pd.to_datetime(fdf["end_timestamp"])
            if fts.dt.tz is not None:
                fts = fts.dt.tz_convert(None)
            fdf["end_timestamp"] = fts
            fdf["caseid"] = fdf["caseid"].astype(str)
            self._full_df_by_case_cache = dict(tuple(fdf.groupby("caseid", sort=False)))
        return self._full_df_by_case_cache

    def _true_continuation(self, caseid: str, anchor_real_ts) -> tuple[list, list, list]:

        if self.full_df is None:
            return [], [], []

        case_df = self._full_df_by_case().get(str(caseid))
        future = (self.full_df.iloc[0:0] if case_df is None
                 else case_df[case_df["end_timestamp"] > anchor_real_ts]).copy()
        future = future.sort_values("end_timestamp", kind="mergesort")

        end_ac_idx = self.ac_index.get("end", 0)
        end_rl_idx = self.rl_index.get("end", 0)

        if future.empty:
            return [end_ac_idx], [end_rl_idx], [0.0]

        if getattr(self, "_resource_map", None) is not None and "user" in future.columns:
            resources = self._resource_map.copy()
            future["user"] = future["user"].astype(str)
            resources["user"] = resources["user"].astype(str)
            future = future.merge(resources, on="user", how="left")
            future["role"] = future["role"].fillna("unk")
        else:
            # Plain-field's full_df_for_plain is deliberately stripped to
            # caseid/task/end_timestamp only (the load-state sweep never
            # needed "user") -- no resource identity to look a role up from,
            # so every future event's role is unknown.
            future["role"] = "unk"

        ac_expect = future["task"].map(self.ac_index).fillna(0).astype(int).tolist() + [end_ac_idx]
        rl_expect = future["role"].map(self.rl_index).fillna(0).astype(int).tolist() + [end_rl_idx]

        ts = [anchor_real_ts] + future["end_timestamp"].tolist()
        tm_expect = [max(0.0, (ts[i] - ts[i - 1]).total_seconds()) for i in range(1, len(ts))] + [0.0]

        return ac_expect, rl_expect, tm_expect

    def predict(self, anchor_ts, variant: str = "arg_max",
                prefix_mode: str = "start_plus_first") -> Path:
        """
        Generate suffixes for every case in self.test_df TOGETHER, advancing
        whichever case's predicted next event is chronologically soonest,
        so each case's inter-case features are read off a shared global
        load-state that reflects everyone else's own generated progress --
        not a per-case-isolated autoregressive loop.

        Parameters
        ----------
        anchor_ts : the "now" moment predictions are being made from (e.g.
            the val_split boundary). Explicit, not inferred from the data,
            since in-flight test cases' own last real event can be anywhere
            strictly before this point while fresh cases start at/after it
            -- only the caller (which built the train/val/test split) knows
            the true, uniform anchor for the whole batch.
        variant : "arg_max" (default) | "random_choice" -- same semantics
            as camargo's own SuffixPredictor.
        prefix_mode : "start_plus_first" (default) | "start_only" -- applies
            only to fresh (single-real-event) cases, matching
            CamargoTrainer.predict()'s own convention.

        Returns
        -------
        Path to a gen_*.csv-schema-compatible CSV (ac_prefix/ac_expect/
        ac_pred/... columns), directly usable by the inherited
        to_event_log([path]) unmodified.
        """
        if prefix_mode not in ("start_only", "start_plus_first"):
            raise ValueError(f"prefix_mode must be 'start_only' or 'start_plus_first', got {prefix_mode!r}")
        if self.best_model_path is None:
            raise RuntimeError("Call train() before predict().")
        self._ensure_ic_features()

        ic_cols = ic_feature_cols(self.top_n)
        n_ic = len(ic_cols)
        anchor_ts = pd.Timestamp(anchor_ts)
        if anchor_ts.tzinfo is not None:
            anchor_ts = anchor_ts.tz_convert(None)

        with _repo_cwd():
            params_path = os.path.join(self._out_base(), "parameters", "model_parameters.json")
            with open(params_path) as f:
                saved = json.load(f)
            time_dim = int(saved["dim"]["time_dim"])
            max_trace_size = int(saved["max_trace_size"])
            dur_max = float(saved["scale_args"].get("max_value", 1.0)) or 1.0

            from tensorflow.keras.models import load_model
            try:
                from keras.src.layers.layer import Layer as _KLayer
                _orig_layer_init = _KLayer.__init__

                def _compat_layer_init(self_layer, *a, **kw):
                    kw.pop("quantization_config", None)
                    _orig_layer_init(self_layer, *a, **kw)
                _KLayer.__init__ = _compat_layer_init
            except Exception:
                pass
            model = load_model(self.best_model_path, compile=False)

            def _fast_predict(inputs):
                out = model(inputs, training=False)
                return [o.numpy() for o in out]

            # ── Per-case rolling history, seeded from the real prefix ──────────
            test = self.test_df.copy()
            test["caseid"] = test["caseid"].astype(str)
            ts_col = pd.to_datetime(test["end_timestamp"])
            if ts_col.dt.tz is not None:
                ts_col = ts_col.dt.tz_convert(None)
            test["end_timestamp"] = ts_col
            test = test.sort_values(["caseid", "end_timestamp"], kind="mergesort")

            start_ac_idx = self.ac_index.get("start", 0)
            start_rl_idx = self.rl_index.get("start", 0)

            cases: dict = {}
            for caseid, grp in test.groupby("caseid", sort=False):
                grp = grp.reset_index(drop=True)
                n = len(grp)
                ac_hist = grp["ac_index"].astype(int).tolist()
                rl_hist = grp["rl_index"].astype(int).tolist()
                ts_hist = grp["end_timestamp"].tolist()
                dur_hist = [0.0] + [
                    max(0.0, (ts_hist[i] - ts_hist[i - 1]).total_seconds())
                    for i in range(1, n)
                ]
                dur_norm_hist = [min(1.0, d / dur_max) for d in dur_hist]
                ic_norm_hist = []
                for j in range(n):
                    row = []
                    for col in ic_cols:
                        cmax = self._ic_scale_max.get(col, 0.0)
                        raw = float(grp[col].iloc[j])
                        row.append(raw / cmax if cmax > 0 else 0.0)
                    ic_norm_hist.append(row)

                if n == 1 and prefix_mode == "start_only":
                    ac_hist = [start_ac_idx]
                    rl_hist = [start_rl_idx]
                    dur_norm_hist = [0.0]
                    ic_norm_hist = [[0.0] * n_ic]
                else:
                    ac_hist = [start_ac_idx] + ac_hist
                    rl_hist = [start_rl_idx] + rl_hist
                    dur_norm_hist = [0.0] + dur_norm_hist
                    ic_norm_hist = [[0.0] * n_ic] + ic_norm_hist

                ac_expect, rl_expect, tm_expect = self._true_continuation(caseid, ts_hist[-1])

                cases[caseid] = {
                    "ac_hist": ac_hist, "rl_hist": rl_hist,
                    "dur_norm_hist": dur_norm_hist, "ic_norm_hist": ic_norm_hist,
                    "last_activity": grp["task"].iloc[-1],
                    "last_real_ts": ts_hist[-1],
                    "pref_size": len(ac_hist),
                    "ac_expect": ac_expect, "rl_expect": rl_expect, "tm_expect": tm_expect,
                    "ac_suf": [], "rl_suf": [], "dur_suf": [],
                    "n_steps": 0,
                }

            if not cases:
                raise ValueError("test_df has no cases to predict.")


            sweep_source = (
                self.full_df if self.full_df is not None
                else pd.concat([self.train_df, self.val_df, self.test_df], ignore_index=True)
            )
            activity_case_count, case_state = sweep_state_upto(
                self._events_df(sweep_source), anchor_ts
            )
            for caseid, c in cases.items():
                la = c["last_activity"]
                prev = case_state.get(caseid)
                if prev == la:
                    continue
                if prev is not None:
                    activity_case_count[prev] -= 1
                activity_case_count[la] += 1
                case_state[caseid] = la

            transition_table = self._transition_table

            def _build_inputs(c):
                ac_w = np.array(c["ac_hist"][-time_dim:])
                rl_w = np.array(c["rl_hist"][-time_dim:])
                dur_w = np.array(c["dur_norm_hist"][-time_dim:])
                ic_w = np.array(c["ic_norm_hist"][-time_dim:])

                pad = time_dim - len(ac_w)
                ac_ngram = np.concatenate([np.zeros(pad), ac_w])[-time_dim:].reshape(1, time_dim)
                rl_ngram = np.concatenate([np.zeros(pad), rl_w])[-time_dim:].reshape(1, time_dim)
                dur_pad = np.zeros((pad, 1))
                t_ngram = np.concatenate(
                    [dur_pad, dur_w.reshape(-1, 1)], axis=0
                )[-time_dim:].reshape(1, time_dim, 1)
                ic_pad = np.zeros((pad, n_ic))
                inter_ngram = np.concatenate(
                    [ic_pad, ic_w.reshape(-1, n_ic)], axis=0
                )[-time_dim:].reshape(1, time_dim, n_ic)
                return [ac_ngram, rl_ngram, t_ngram, inter_ngram]

            counter = itertools.count()
            heap: list = []
            pending: dict = {}

            def _schedule_next(caseid, from_ts):
                c = cases[caseid]
                if c["n_steps"] >= max_trace_size - 1:
                    c["n_steps"] = -1  # sentinel: force-finalize, see main loop
                    heapq.heappush(heap, (from_ts, next(counter), caseid))
                    pending[caseid] = None
                    return
                preds = _fast_predict(_build_inputs(c))
                if variant == "random_choice":
                    pos = int(np.random.choice(len(preds[0][0]), p=preds[0][0]))
                    pos1 = int(np.random.choice(len(preds[1][0]), p=preds[1][0]))
                else:
                    pos = int(np.argmax(preds[0][0]))
                    pos1 = int(np.argmax(preds[1][0]))
                dur_real = float(np.rint(float(preds[2][0][0]) * dur_max))
                dur_real = max(0.0, dur_real)
                new_ts = from_ts + pd.Timedelta(seconds=dur_real)
                pending[caseid] = (pos, pos1, dur_real)
                heapq.heappush(heap, (new_ts, next(counter), caseid))

            for caseid, c in cases.items():
                # anchor per case: its own last real event time (same
                # convention as the inherited to_event_log()'s anchor).
                last_ts = c["last_real_ts"]
                c["_ts"] = last_ts
                _schedule_next(caseid, last_ts)

            index_ac = self.index_ac
            _total = len(cases)
            _done = 0
            _last_print = time.monotonic()

            def _progress_tick():
                nonlocal _done, _last_print
                _done += 1
                now = time.monotonic()
                if now - _last_print >= 10.0 or _done == _total:
                    print(f"[{self.run_name}] generating suffixes: {_done}/{_total}", flush=True)
                    _last_print = now
            while heap:
                ts, _, caseid = heapq.heappop(heap)
                c = cases[caseid]
                step = pending.pop(caseid)

                if step is None:
                    # max_trace_size reached -- release from global state and stop.
                    prev = case_state.get(caseid)
                    if prev is not None:
                        activity_case_count[prev] -= 1
                        del case_state[caseid]
                    _progress_tick()
                    continue

                pos, pos1, dur_real = step
                activity_label = index_ac.get(pos, "unk")

                c["ac_suf"].append(pos)
                c["rl_suf"].append(pos1)
                c["dur_suf"].append(dur_real)
                c["n_steps"] += 1

                if activity_label == "end":
                    prev = case_state.get(caseid)
                    if prev is not None:
                        activity_case_count[prev] -= 1
                        del case_state[caseid]
                    _progress_tick()
                    continue

                prev_activity = case_state.get(caseid)
                feats = load_state_features_for(
                    activity_label, prev_activity, activity_case_count,
                    transition_table, self.top_n
                )

                c["ac_hist"].append(pos)
                c["rl_hist"].append(pos1)
                c["dur_hist_seconds"] = dur_real
                c["dur_norm_hist"].append(min(1.0, dur_real / dur_max))
                ic_vec = []
                for col in ic_cols:
                    cmax = self._ic_scale_max.get(col, 0.0)
                    ic_vec.append(feats[col] / cmax if cmax > 0 else 0.0)
                c["ic_norm_hist"].append(ic_vec)
                c["last_activity"] = activity_label
                c["_ts"] = ts

                if prev_activity is not None:
                    activity_case_count[prev_activity] -= 1
                activity_case_count[activity_label] += 1
                case_state[caseid] = activity_label

                _schedule_next(caseid, ts)

            # ── Emit gen_*.csv-schema-compatible rows ──────────────────────────
            rows = []
            for caseid, c in cases.items():
                rows.append({
                    "caseid": caseid,
                    "pref_size": c["pref_size"],
                    "ac_prefix": [], "ac_expect": c["ac_expect"], "ac_pred": c["ac_suf"],
                    "rl_prefix": [], "rl_expect": c["rl_expect"], "rl_pred": c["rl_suf"],
                    "tm_prefix": [], "tm_expect": c["tm_expect"], "tm_pred": c["dur_suf"],
                })
            out_df = pd.DataFrame(rows)
            out_path = REPO_DIR / self._out_base() / "gen_ic_joint.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(out_path, index=False)
            print(f"[predict] joint-sim suffixes written → {out_path} ({len(out_df)} cases)")

        return out_path
