"""GPS training / inference entry point for the Amiri wrapper.

Usage (training):
    python amiri/run_gps.py --cfg <train_cfg.yaml> --repeat 1

Usage (inference):
    python amiri/run_gps.py --cfg <infer_cfg.yaml> --repeat 1
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure pgtnet/scripts (for graphgps) and the project root (for amiri) are importable
_this = Path(__file__).resolve().parent       # .../amiri/
_root = _this.parent                          # .../06_System_level_prediction/
_pgtnet_scripts = _root / "pgtnet" / "scripts"

for _p in [str(_pgtnet_scripts), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import datetime
import logging

import numpy as np
import pandas as pd
import torch

import graphgps  # noqa — registers custom modules
from amiri.loader_patch import patch_loader

patch_loader()

from graphgps.agg_runs import agg_runs
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg, dump_cfg, set_cfg, load_cfg, makedirs_rm_exist,
)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import (
    create_optimizer, create_scheduler, OptimizerConfig,
)
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from graphgps.finetuning import load_pretrained_model_cfg, init_model_from_pretrained
from graphgps.logger import create_logger

from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('two_layer_linear_edge_encoder_keys')
def _two_layer_enc_cfg(cfg):
    """Register YACS schema for the TwoLayerLinearEdge encoder config."""
    cfg.two_layer_linear_edge_encoder = CN()
    cfg.two_layer_linear_edge_encoder.in_dim = 0


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def new_optimizer_config(cfg):
    return OptimizerConfig(
        optimizer=cfg.optim.optimizer,
        base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum,
    )


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps,
        lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch,
        reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience,
        min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode,
        eval_period=cfg.train.eval_period,
    )


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    if len(cfg.run_multiple_splits) == 0:
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        if args.repeat != 1:
            raise NotImplementedError
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


if __name__ == "__main__":
    args = parse_args()
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    torch.set_num_threads(cfg.num_threads)

    inference_dataframes = []

    for run_id, seed, split_index in zip(*run_loop_settings()):
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()

        if cfg.pretrained.dir and cfg.train.mode != "event-inference":
            cfg = load_pretrained_model_cfg(cfg)

        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, split_index={cfg.dataset.split_index}")

        if cfg.train.mode == "event-inference":
            inference_start_time = datetime.datetime.now()

        logging.info(f"    Starting now: {datetime.datetime.now()}")

        loaders = create_loader()
        loggers = create_logger()
        model = create_model()

        if cfg.pretrained.dir and cfg.train.mode != "event-inference":
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head, seed=cfg.seed,
            )

        optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info("Num parameters: %s", cfg.params)

        if cfg.train.mode == "standard":
            datamodule = GraphGymDataModule()
            train(model, datamodule, logger=True)

        elif cfg.train.mode == "event-inference":
            fold_address = cfg.pretrained.dir + f"/{run_id}/ckpt"
            if os.path.exists(fold_address) and os.path.isdir(fold_address):
                ckpt_files = os.listdir(fold_address)
                if len(ckpt_files) == 1:
                    ckpt_path = os.path.join(fold_address, ckpt_files[0])
                    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    model.load_state_dict(loaded["model_state"])
                    print(f"[inference] loaded checkpoint: {ckpt_path}")
                else:
                    print("Error: expected exactly one checkpoint file.")
            else:
                print(f"Error: checkpoint folder not found: {fold_address}")

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            predictions = []
            real_values = []
            test_loader = loaders[2]
            with torch.no_grad():
                for graph in test_loader:
                    graph.to(device)
                    pred = model(graph)
                    predictions.append(float(np.array(pred[0].cpu())))
                    real_values.append(float(np.array(graph.y[0].cpu())))

            fold_df = pd.DataFrame({
                "predicted_normalized": predictions,
                "real_normalized": real_values,
            })
            inference_dataframes.append(fold_df)

        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)

    if cfg.train.mode != "event-inference":
        try:
            agg_runs(cfg.out_dir, cfg.metric_best)
        except Exception as e:
            logging.info(f"Failed to aggregate runs: {e}")

    if cfg.train.mode == "event-inference":
        prediction_df = pd.concat(inference_dataframes, ignore_index=True)
        out_path = os.path.join(cfg.out_dir, "amiri_predictions.csv")
        prediction_df.to_csv(out_path, index=False)
        print(f"[inference] predictions saved → {out_path}  ({len(prediction_df)} rows)")
        inference_end = datetime.datetime.now()
        elapsed_ms = (inference_end - inference_start_time).total_seconds() * 1000
        print(f"[inference] elapsed: {elapsed_ms:.0f} ms  "
              f"({elapsed_ms / max(len(prediction_df), 1):.2f} ms/graph)")

    if args.mark_done:
        os.rename(args.cfg_file, f"{args.cfg_file}_done")
    logging.info(f"[*] All done: {datetime.datetime.now()}")
