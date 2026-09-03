import os
import os.path as osp
import pickle

import torch
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data


class AmiriDataset(InMemoryDataset):
    """PyG InMemoryDataset backed by locally pre-computed pickle files.

    Expects raw_dir to contain train.pickle, val.pickle, test.pickle produced
    by amiri.converter.convert().
    """

    def __init__(self, root, split="train", transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "AMIRI"
        assert split in ["train", "val", "test"]
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f"{split}.pt")
        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self):
        return ["train.pickle", "val.pickle", "test.pickle"]

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    def download(self):
        raise RuntimeError(
            "AmiriDataset requires pre-computed pickle files in raw_dir. "
            "Run amiri.converter.convert() first."
        )

    def process(self):
        for split in ["train", "val", "test"]:
            with open(osp.join(self.raw_dir, f"{split}.pickle"), "rb") as f:
                graphs = pickle.load(f)

            pbar = tqdm(total=len(graphs), desc=f"Processing {split}")
            data_list = []
            for graph in graphs:
                data = Data(
                    x=graph.x,
                    edge_index=graph.edge_index,
                    edge_attr=graph.edge_attr,
                    y=graph.y,
                    cid=graph.cid,
                    pl=graph.pl,
                )
                if self.pre_filter is not None and not self.pre_filter(data):
                    pbar.update(1)
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
                pbar.update(1)
            pbar.close()

            if data_list:
                out = self.collate(data_list)
            else:
                out = (Data(), {})  # empty split: len() returns 0 via the for-loop fallthrough
            torch.save(out, osp.join(self.processed_dir, f"{split}.pt"))


def _join_splits(datasets):
    assert len(datasets) == 3
    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = (
        [datasets[0].get(i) for i in range(n1)]
        + [datasets[1].get(i) for i in range(n2)]
        + [datasets[2].get(i) for i in range(n3)]
    )
    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    datasets[0].split_idxs = [
        list(range(n1)),
        list(range(n1, n1 + n2)),
        list(range(n1 + n2, n1 + n2 + n3)),
    ]
    # PyG >= 2.4 GraphGym loader expects these index tensors in dataset._data
    datasets[0]._data['train_graph_index'] = torch.arange(n1)
    datasets[0]._data['val_graph_index']   = torch.arange(n1, n1 + n2)
    datasets[0]._data['test_graph_index']  = torch.arange(n1 + n2, n1 + n2 + n3)
    return datasets[0]


def preformat_AMIRI(dataset_dir: str):
    """Load and join train/val/test AmiriDataset splits (used by loader_patch)."""
    return _join_splits(
        [AmiriDataset(root=dataset_dir, split=split) for split in ["train", "val", "test"]]
    )
