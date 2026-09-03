"""AmiriTrainer — bridge between our DataFrame splits and the PGTNet/GPS pipeline.

Usage
-----
from amiri.trainer import AmiriTrainer
from amiri.params  import default_params

params  = default_params(max_epoch=100)
trainer = AmiriTrainer(train_df, val_df, test_df, "loan_flat", params,
                        output_dir="amiri_output/loan_flat")
trainer.run()               # convert data + train GPS model
rem_time_df = trainer.predict()  # run inference → rem_time_df
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
_RUN_GPS = _THIS_DIR / "run_gps.py"

_TRAIN_CFG_NAME = "amiri_gps.yaml"
_INFER_CFG_NAME = "amiri_gps_infer.yaml"
_GPS_CFG_STEM = "amiri_gps"        # cfg.out_dir = out_dir / _GPS_CFG_STEM
_GPS_INFER_STEM = "amiri_gps_infer"


class AmiriTrainer:
    """Orchestrates data conversion, GPS training, and remaining-time inference."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        run_name: str,
        params: dict,
        output_dir: Optional[str | Path] = None,
        dataset_dir: Optional[str | Path] = None,
        ref_dataset_dir: Optional[str | Path] = None,
    ):
        self.train_df = train_df.copy()
        self.val_df = val_df.copy()
        self.test_df = test_df.copy()
        self.run_name = run_name
        self.params = dict(params)

        self.work_dir = (
            Path(output_dir) if output_dir else Path("amiri_output") / run_name
        )
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # When provided, stats from this dataset's meta.pkl are reused during
        # data conversion so inference uses the same feature space as training.
        self.ref_dataset_dir = Path(ref_dataset_dir) if ref_dataset_dir else None

        # Allow sharing a pre-converted dataset across HPO trials
        self.dataset_dir = Path(dataset_dir) if dataset_dir else self.work_dir / "datasets"
        self.raw_dir = self.dataset_dir / "AMIRI" / "raw"

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self):
        """Convert data to PyG graphs and train the GPS model."""
        self._convert_data()
        cfg_path = self._write_train_yaml()
        self._run_subprocess(cfg_path, label="training")

    def predict(self) -> pd.DataFrame:
        """Run GPS inference and return rem_time_df.

        Returns
        -------
        rem_time_df : DataFrame with columns
            caseid, start_timestamp, anchor_timestamp, prefix_len, rem_time_days
        """
        dataset_rebuilt = self._convert_data()
        if dataset_rebuilt:
            # Stale inference outputs are invalid; remove them so they get regenerated.
            import shutil
            infer_out = self.work_dir / "gps_results_infer"
            if infer_out.exists():
                shutil.rmtree(str(infer_out))
            rem_time_csv = self.work_dir / "rem_time.csv"
            if rem_time_csv.exists():
                rem_time_csv.unlink()
        cfg_path = self._write_infer_yaml()
        self._run_subprocess(cfg_path, label="inference")
        return self._parse_predictions()

    def get_best_val_mae(self) -> float:
        """Return the best (minimum) val MAE from the most recent training run.

        Reads from the GPS stats file produced during train.mode=custom.
        The MAE is in normalised units (divide max_time_norm to convert to days).
        """
        seed = self.params.get("seed", 42)
        stats_path = (
            self.work_dir / "gps_results" / _GPS_CFG_STEM / str(seed) / "val" / "stats.json"
        )
        if not stats_path.exists():
            raise FileNotFoundError(f"Val stats not found: {stats_path}")

        min_mae = float("inf")
        with open(stats_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats = json.loads(line)
                mae = stats.get("mae", float("inf"))
                if mae < min_mae:
                    min_mae = mae
        return min_mae

    # ── Data conversion ────────────────────────────────────────────────────────

    def _convert_data(self) -> bool:
        """Convert data; returns True if the dataset was (re-)built."""
        import shutil

        if (self.raw_dir / "meta.pkl").exists():
            test_order_file = self.raw_dir / "test_case_order.pkl"
            if test_order_file.exists():
                with open(test_order_file, "rb") as f:
                    cached_order = pickle.load(f)
                # Only cases with ≥ 2 events are convertible; fresh cases (1 event)
                # are skipped by the converter, so compare against that count.
                predictable = sum(
                    1 for _, grp in self.test_df.groupby("caseid")
                    if len(grp) >= 2
                )
                if len(cached_order) == predictable:
                    print("[trainer] raw pickles already exist — skipping conversion")
                    return False
                print(
                    f"[trainer] test set size mismatch: {len(cached_order)} cached "
                    f"vs {predictable} predictable in test_df — rebuilding dataset"
                )
            shutil.rmtree(str(self.dataset_dir / "AMIRI"), ignore_errors=True)

        from amiri.converter import convert
        ref_meta = None
        if self.ref_dataset_dir is not None:
            candidate = self.ref_dataset_dir / "AMIRI" / "raw" / "meta.pkl"
            if candidate.exists():
                ref_meta = candidate
        convert(
            self.train_df,
            self.val_df,
            self.test_df,
            self.dataset_dir,
            seed=self.params.get("seed", 42),
            ref_meta_path=ref_meta,
        )
        return True

    # ── YAML config generation ─────────────────────────────────────────────────

    def _read_meta(self) -> dict:
        with open(self.raw_dir / "meta.pkl", "rb") as f:
            return pickle.load(f)

    def _write_train_yaml(self) -> Path:
        meta = self._read_meta()
        cfg = self._base_cfg(meta)
        cfg["out_dir"] = str(self.work_dir / "gps_results")
        cfg["train"]["mode"] = "custom"
        cfg["train"]["batch_size"] = self.params.get("batch_size", 128)
        cfg["train"]["eval_period"] = 1
        cfg["train"]["enable_ckpt"] = True
        cfg["train"]["ckpt_best"] = True

        path = self.work_dir / _TRAIN_CFG_NAME
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"[trainer] train cfg → {path}")
        return path

    def _write_infer_yaml(self) -> Path:
        meta = self._read_meta()
        cfg = self._base_cfg(meta)
        cfg["out_dir"] = str(self.work_dir / "gps_results_infer")
        cfg["train"]["mode"] = "event-inference"
        cfg["train"]["batch_size"] = 1
        cfg["train"]["eval_period"] = 1
        cfg["train"]["enable_ckpt"] = False
        cfg["pretrained"] = {
            "dir": str(self.work_dir / "gps_results" / _GPS_CFG_STEM),
            "freeze_main": False,
            "reset_prediction_head": False,
        }

        path = self.work_dir / _INFER_CFG_NAME
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"[trainer] infer cfg → {path}")
        return path

    def _base_cfg(self, meta: dict) -> dict:
        num_node_types = meta["num_node_types"]
        seed = self.params.get("seed", 42)
        rwse_k = min(self.params.get("posenc_rwse_k", 36), max(num_node_types, 2))
        dim_h = self.params.get("gt_dim_hidden", 64)

        return {
            "out_dir": "results",
            "metric_best": "mae",
            "metric_agg": "argmin",
            "wandb": {"use": False},
            "seed": seed,
            "dataset": {
                "format": "PyG-AMIRI",
                "name": "amiri",
                "dir": str(self.dataset_dir),
                "task": "graph",
                "task_type": "regression",
                "transductive": False,
                "split_mode": "standard",
                "node_encoder": True,
                "node_encoder_name": "TypeDictNode+LapPE+RWSE",
                "node_encoder_num_types": num_node_types,
                "node_encoder_bn": True,
                "edge_encoder": True,
                "edge_encoder_name": "TwoLayerLinearEdge",
                "edge_encoder_bn": True,
                "edge_dim": meta["edge_dim"],
            },
            "posenc_LapPE": {
                "enable": True,
                "eigen": {"laplacian_norm": "none", "eigvec_norm": "L2", "max_freqs": 1},
                "model": "DeepSet",
                "dim_pe": self.params.get("posenc_lapPE_dim_pe", 8),
                "layers": 2,
                "raw_norm_type": "none",
            },
            "posenc_RWSE": {
                "enable": True,
                "kernel": {"times_func": f"range(1,{rwse_k + 1})"},
                "model": "Linear",
                "dim_pe": self.params.get("posenc_rwse_dim_pe", 8),
                "raw_norm_type": "BatchNorm",
            },
            "train": {},
            "model": {
                "type": "GPSModel",
                "loss_fun": "l1",
                "edge_decoding": "dot",
                "graph_pooling": "mean",
            },
            "gt": {
                "layer_type": "GINE+Transformer",
                "layers": self.params.get("gt_layers", 5),
                "n_heads": self.params.get("gt_n_heads", 8),
                "dim_hidden": dim_h,
                "dropout": self.params.get("gt_dropout", 0.2),
                "attn_dropout": self.params.get("gt_attn_dropout", 0.5),
                "layer_norm": False,
                "batch_norm": True,
            },
            "gnn": {
                "head": "san_graph",
                "layers_pre_mp": 0,
                "layers_post_mp": 3,
                "dim_inner": dim_h,
                "batchnorm": True,
                "act": "relu",
                "dropout": self.params.get("gt_dropout", 0.2),
                "agg": "mean",
                "normalize_adj": False,
            },
            "two_layer_linear_edge_encoder": {
                "in_dim": meta["edge_dim"],
            },
            "optim": {
                "clip_grad_norm": True,
                "optimizer": "adamW",
                "weight_decay": self.params.get("weight_decay", 1e-2),
                "base_lr": self.params.get("base_lr", 0.0005),
                "max_epoch": self.params.get("max_epoch", 100),
                "scheduler": "cosine_with_warmup",
                "num_warmup_epochs": max(1, self.params.get("max_epoch", 100) // 2),
            },
        }

    # ── Subprocess execution ───────────────────────────────────────────────────

    def _run_subprocess(self, cfg_path: Path, label: str):
        cmd = [sys.executable, str(_RUN_GPS), "--cfg", str(cfg_path), "--repeat", "1"]
        log_path = self.work_dir / f"{label}.log"
        max_epoch = self.params.get("max_epoch", 100)
        is_training = label == "training"
        print(f"[trainer] {label} → {log_path}")
        if is_training:
            print(f"[trainer] training  0/{max_epoch} epochs", end="", flush=True)
        with open(log_path, "w") as _log:
            proc = subprocess.Popen(
                cmd, cwd=str(_ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                _log.write(line)
                _log.flush()
                if is_training and line.startswith("> Epoch "):
                    # "> Epoch 3: took 12.1s (avg 11.8s) | Best so far: ..."
                    parts = line.split(":")
                    epoch_num = parts[0].replace("> Epoch", "").strip()
                    best_part = line.split("Best so far:")[-1].strip().rstrip("\n")
                    print(f"\r[trainer] training  {epoch_num}/{max_epoch} epochs  |  best: {best_part}", end="", flush=True)
                elif not is_training and "[inference]" in line:
                    print(line.rstrip(), flush=True)
            proc.wait()
        if is_training:
            print()  # newline after progress line
        if proc.returncode != 0:
            try:
                lines = log_path.read_text().splitlines()
                print("\n".join(lines[-30:]))
            except Exception:
                pass
            raise RuntimeError(f"GPS {label} failed (see {log_path})")

    # ── Prediction parsing ─────────────────────────────────────────────────────

    def _parse_predictions(self) -> pd.DataFrame:
        meta = self._read_meta()
        max_time_norm = meta["max_time_norm"]

        with open(self.raw_dir / "test_case_order.pkl", "rb") as f:
            test_case_order = pickle.load(f)

        pred_csv = (
            self.work_dir
            / "gps_results_infer"
            / _GPS_INFER_STEM
            / "amiri_predictions.csv"
        )
        if not pred_csv.exists():
            raise FileNotFoundError(f"Inference output not found: {pred_csv}")

        pred_df = pd.read_csv(pred_csv)
        if len(pred_df) != len(test_case_order):
            raise ValueError(
                f"Prediction count mismatch: {len(pred_df)} predictions vs "
                f"{len(test_case_order)} test cases."
            )

        # Reconstruct anchor timestamps from test_df
        anchor_map = {}
        for cid, grp in self.test_df.groupby("caseid"):
            grp = grp.sort_values("end_timestamp")
            ts = grp["end_timestamp"].tolist()
            anchor_map[cid] = {
                "start_timestamp": pd.Timestamp(ts[0]),
                "anchor_timestamp": pd.Timestamp(ts[-1]),
                "prefix_len": len(grp),
            }

        rows = []
        for cid, pred_norm in zip(test_case_order, pred_df["predicted_normalized"]):
            anchor = anchor_map.get(cid, {})
            rows.append(
                {
                    "caseid": cid,
                    "start_timestamp": anchor.get("start_timestamp"),
                    "anchor_timestamp": anchor.get("anchor_timestamp"),
                    "prefix_len": anchor.get("prefix_len"),
                    "rem_time_days": max(0.0, float(pred_norm) * max_time_norm),
                }
            )

        # Fresh cases (single event in test_df) can't be processed by GPS; add
        # them with a fallback prediction (mean case duration from training data)
        # so that the CC timeseries covers the full test period.
        gps_cids = set(test_case_order)
        case_durations = []
        for _, grp in self.train_df.groupby("caseid"):
            ts = grp["end_timestamp"].sort_values()
            if len(ts) >= 2:
                dur = (pd.Timestamp(ts.iloc[-1]) - pd.Timestamp(ts.iloc[0])).total_seconds() / 86400
                case_durations.append(dur)
        mean_case_dur = float(np.mean(case_durations)) if case_durations else max_time_norm / 2

        for cid, info in anchor_map.items():
            if cid in gps_cids:
                continue
            if info["prefix_len"] < 2:
                rows.append(
                    {
                        "caseid": cid,
                        "start_timestamp": info["start_timestamp"],
                        "anchor_timestamp": info["anchor_timestamp"],
                        "prefix_len": info["prefix_len"],
                        "rem_time_days": mean_case_dur,
                    }
                )

        rem_df = pd.DataFrame(rows)

        el_path = self.work_dir / "rem_time.csv"
        rem_df.to_csv(el_path, index=False)
        print(f"[predict] rem_time → {el_path}  ({len(rem_df):,} rows)")

        return rem_df


def select_amiri_hpo_winner(hpo_dir, train_df, val_df, val_as_test_df,
                             cc_val, tt_val, default_params_fn, rescore_root=None):
    """Score every completed amiri HPO trial checkpoint under hpo_dir by
    real validation-period CC MAE, and return the winning trial.



    Parameters
    ----------
    hpo_dir : Path
        The HPO trial-checkpoint directory (contains trial_NNN/ subdirs and
        a shared_dataset/ conversion cache, exactly as every
        hpo_training_amiri*.ipynb notebook already lays it out).
    train_df, val_df : DataFrame
        Same splits used for HPO training.
    val_as_test_df : DataFrame
        Cases observable at train_split -- build via
        hpo_val_scoring.build_val_as_test_df(full_df, train_split).
    cc_val, tt_val : Series
        Real validation-period concurrent-cases / throughput-time actuals.
    default_params_fn : callable
        amiri.params.default_params (passed in, not imported here, to avoid
        a hard dependency direction from this shared helper onto one
        specific params module shape).
    rescore_root : Path, optional
        Scratch directory for per-trial rescoring work (default:
        hpo_dir/_val_cc_rescore). Never touches the original trial
        checkpoints -- each trial's gps_results/ is symlinked in read-only.

    Returns
    -------
    dict with keys 'trial_id', 'cfg', 'val_cc_mae', 'val_tt_mae',
    'gps_results_dir' (the ORIGINAL trial's checkpoint -- copy this into the
    final deployed model location, same as before, just pointed at the
    correct trial) -- or None if no trial could be scored (e.g. hpo_dir has
    no checkpoints yet).
    """
    from hpo_val_scoring import kpi_mae, rem_time_to_event_log

    hpo_dir = Path(hpo_dir)
    trial_paths = sorted(hpo_dir.glob("trial_*"))
    if not trial_paths:
        return None

    shared_dataset_dir = hpo_dir / "shared_dataset"
    rescore_root = Path(rescore_root) if rescore_root else hpo_dir / "_val_cc_rescore"
    rescore_root.mkdir(parents=True, exist_ok=True)

    best = None  # (cc_mae, trial_id, cfg, tt_mae, gps_results_dir)
    for tdir in trial_paths:
        cfg_path = tdir / "config.json"
        gps_dir = tdir / "gps_results"
        if not cfg_path.exists() or not gps_dir.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        params = default_params_fn(**cfg)
        params["seed"] = 42

        work_dir = rescore_root / tdir.name
        work_dir.mkdir(parents=True, exist_ok=True)
        gps_link = work_dir / "gps_results"
        if not gps_link.exists():
            gps_link.symlink_to(gps_dir.resolve())

        try:
            trainer = AmiriTrainer(
                train_df, val_df, val_as_test_df, run_name=f"{tdir.name}_val_cc_rescore",
                params=params, output_dir=work_dir,
                dataset_dir=work_dir / "datasets", ref_dataset_dir=shared_dataset_dir,
            )
            rem_time_df = trainer.predict()
            event_log = rem_time_to_event_log(rem_time_df)
            cc_mae, tt_mae = kpi_mae(event_log, cc_val, tt_val)
        except Exception as exc:
            print(f"  [rescore {tdir.name}] FAILED: {exc}")
            continue

        print(f"  [rescore {tdir.name}] val_cc_mae={cc_mae:.4f}  val_tt_mae={tt_mae:.4f}")
        if best is None or cc_mae < best[0]:
            best = (cc_mae, tdir.name, cfg, tt_mae, gps_dir)

    if best is None:
        return None
    cc_mae, trial_id, cfg, tt_mae, gps_dir = best
    return {
        "trial_id": trial_id, "cfg": cfg,
        "val_cc_mae": round(float(cc_mae), 4), "val_tt_mae": round(float(tt_mae), 4),
        "gps_results_dir": gps_dir,
    }
