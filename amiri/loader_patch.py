"""Monkey-patch the GraphGPS master loader to support the PyG-AMIRI format.

Call patch_loader() once before training or inference. This adds 'PyG-AMIRI'
as a recognized dataset format without modifying any existing files.
"""
from __future__ import annotations

import os.path as osp
from functools import partial


def patch_loader() -> None:
    """Register AmiriDataset as a valid format in the GPS master loader."""
    from torch_geometric.graphgym.register import loader_dict

    # Avoid double-patching
    existing = loader_dict.get("custom_master_loader")
    if existing is not None and getattr(existing, "_amiri_patched", False):
        return
    if loader_dict.get("amiri_loader") is not None:
        return

    def _load_amiri(dataset_dir):
        import torch
        from amiri.dataset import preformat_AMIRI
        from graphgps.transform.posenc_stats import compute_posenc_stats
        from graphgps.transform.transforms import pre_transform_in_memory
        from torch_geometric.graphgym.config import cfg

        amiri_dir = osp.join(dataset_dir, "AMIRI")
        dataset = preformat_AMIRI(amiri_dir)

        n_train = int(dataset._data['train_graph_index'].size(0))
        n_val   = int(dataset._data['val_graph_index'].size(0))
        n_test  = int(dataset._data['test_graph_index'].size(0))

        pe_enabled_list = []
        for key, pecfg in cfg.items():
            if key.startswith('posenc_') and pecfg.enable:
                pe_name = key.split('_', 1)[1]
                pe_enabled_list.append(pe_name)
                if hasattr(pecfg, 'kernel') and pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))

        if pe_enabled_list:
            is_undirected = all(
                d.is_undirected() for d in dataset[:min(10, len(dataset))]
            )
            pre_transform_in_memory(
                dataset,
                partial(compute_posenc_stats,
                        pe_types=pe_enabled_list,
                        is_undirected=is_undirected,
                        cfg=cfg),
                show_progress=False,
            )
            dataset._data['train_graph_index'] = torch.arange(n_train)
            dataset._data['val_graph_index']   = torch.arange(n_train, n_train + n_val)
            dataset._data['test_graph_index']  = torch.arange(
                n_train + n_val, n_train + n_val + n_test)

        return dataset

    if existing is not None:
        # Wrap the existing custom_master_loader so PyG-AMIRI is handled first
        def patched_loader(format, name, dataset_dir):
            if format == "PyG-AMIRI":
                return _load_amiri(dataset_dir)
            return existing(format, name, dataset_dir)

        patched_loader._amiri_patched = True
        loader_dict["custom_master_loader"] = patched_loader
    else:
        # No master loader registered yet — add a standalone one
        def amiri_only_loader(format, name, dataset_dir):
            if format == "PyG-AMIRI":
                return _load_amiri(dataset_dir)
            return None

        loader_dict["amiri_loader"] = amiri_only_loader
