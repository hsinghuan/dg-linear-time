import numpy as np
from lightning import LightningDataModule
from tgb.linkproppred.dataset import LinkPropPredDataset

from src.utils.data import Data, get_idx_dataloader

data_num_nodes_map = {
    "tgbl-wiki": 9227,
    "tgbl-review": 352637,
    "tgbl-coin": 638486,
    "tgbl-comment": 994790,
    "tgbl-flight": 18143,
}

data_num_edges_map = {
    "tgbl-wiki": 157474,
    "tgbl-review": 4873540,
    "tgbl-coin": 22809486,
    "tgbl-comment": 44314507,
    "tgbl-flight": 67169570,
}


class TGBLDataModule(LightningDataModule):
    """LightningDataModule for tgbl datasets."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        sample_neighbor_strategy: str = "uniform",
        time_scaling_factor: float = 0.0,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.dataset_name = dataset_name
        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data."""
        LinkPropPredDataset(name=self.hparams.dataset_name, preprocess=True)

    def setup(self, stage=None) -> None:
        """Load data."""
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        dataset = LinkPropPredDataset(name=self.hparams.dataset_name, preprocess=True)
        self.data = dataset.full_data
        self.src_node_ids = self.data["sources"].astype(np.longlong)
        self.dst_node_ids = self.data["destinations"].astype(np.longlong)
        self.node_interact_times = self.data["timestamps"].astype(np.float64)
        self.edge_ids = self.data["edge_idxs"].astype(np.longlong)
        self.labels = self.data["edge_label"]
        self.edge_raw_features = self.data["edge_feat"].astype(np.float64)

        # deal with edge features whose shape has only one dimension
        if len(self.edge_raw_features.shape) == 1:
            self.edge_raw_features = self.edge_raw_features[:, np.newaxis]

        # check node set and edge set
        num_edges = self.edge_raw_features.shape[0]
        assert (
            num_edges == data_num_edges_map[self.hparams.dataset_name]
        ), "Number of edges are not matched!"
        # union to get node set
        num_nodes = len(set(self.src_node_ids) | set(self.dst_node_ids))
        assert (
            num_nodes == data_num_nodes_map[self.hparams.dataset_name]
        ), "Number of nodes are not matched!"

        assert (
            self.src_node_ids.min() == 0 or self.dst_node_ids.min() == 0
        ), "Node index should start from 0!"
        assert (
            self.edge_ids.min() == 0 or self.edge_ids.min() == 1
        ), "Edge index should start from 0 or 1!"
        if self.edge_ids.min() == 1:
            print(f"Manually minus the edge indices by 1 on {self.hparams.dataset_name}")
            self.edge_ids = self.edge_ids - 1
        assert self.edge_ids.min() == 0, "After correction, edge index should start from 0!"

        self.eval_metric_name = dataset.eval_metric

        self.train_mask = dataset.train_mask
        self.val_mask = dataset.val_mask
        self.test_mask = dataset.test_mask
        self.eval_neg_edge_sampler = dataset.negative_sampler
        dataset.load_val_ns()
        dataset.load_test_ns()

        # note that in DyGLib's data preprocess pipeline, they add an extra node and edge with index 0 as the padded node/edge for convenience of model computation,
        # therefore, for TGB, they also manually add the extra node and edge with index 0
        self.src_node_ids = self.src_node_ids + 1
        self.dst_node_ids = self.dst_node_ids + 1
        self.edge_ids = self.edge_ids + 1

        max_feat_dim = 172
        if "node_feat" not in self.data.keys():
            print("No node feat")
            self.node_raw_features = np.zeros(
                (num_nodes + 1, 1)
            )  # TODO: isn't the feature of padded node and padded edge already included in the np.vstack operation below? check in the future
        else:
            self.node_raw_features = self.data["node_feat"].astype(np.float64)
            # deal with node features whose shape has only one dimension
            if len(self.node_raw_features.shape) == 1:
                self.node_raw_features = self.node_raw_features[:, np.newaxis]
        # add feature of padded node and padded edge
        self.node_raw_features = np.vstack(
            [np.zeros(self.node_raw_features.shape[1])[np.newaxis, :], self.node_raw_features]
        )
        self.edge_raw_features = np.vstack(
            [np.zeros(self.edge_raw_features.shape[1])[np.newaxis, :], self.edge_raw_features]
        )

        assert (
            max_feat_dim >= self.node_raw_features.shape[1]
        ), f"Node feature dimension in dataset {self.hparams.dataset_name} is bigger than {max_feat_dim}!"
        assert (
            max_feat_dim >= self.edge_raw_features.shape[1]
        ), f"Edge feature dimension in dataset {self.hparams.dataset_name} is bigger than {max_feat_dim}!"

        self.full_data = Data(
            src_node_ids=self.src_node_ids,
            dst_node_ids=self.dst_node_ids,
            node_interact_times=self.node_interact_times,
            edge_ids=self.edge_ids,
            labels=self.labels,
        )
        self.train_data = Data(
            src_node_ids=self.src_node_ids[self.train_mask],
            dst_node_ids=self.dst_node_ids[self.train_mask],
            node_interact_times=self.node_interact_times[self.train_mask],
            edge_ids=self.edge_ids[self.train_mask],
            labels=self.labels[self.train_mask],
        )
        self.val_data = Data(
            src_node_ids=self.src_node_ids[self.val_mask],
            dst_node_ids=self.dst_node_ids[self.val_mask],
            node_interact_times=self.node_interact_times[self.val_mask],
            edge_ids=self.edge_ids[self.val_mask],
            labels=self.labels[self.val_mask],
        )
        self.test_data = Data(
            src_node_ids=self.src_node_ids[self.test_mask],
            dst_node_ids=self.dst_node_ids[self.test_mask],
            node_interact_times=self.node_interact_times[self.test_mask],
            edge_ids=self.edge_ids[self.test_mask],
            labels=self.labels[self.test_mask],
        )
        print(
            f"Full dataset has {self.full_data.num_interactions} interactions and {self.full_data.num_unique_nodes} unique nodes."
        )
        print(
            f"Train dataset has {self.train_data.num_interactions} interactions and {self.train_data.num_unique_nodes} unique nodes."
        )
        print(
            f"Validation dataset has {self.val_data.num_interactions} interactions and {self.val_data.num_unique_nodes} unique nodes."
        )
        print(
            f"Test dataset has {self.test_data.num_interactions} interactions and {self.test_data.num_unique_nodes} unique nodes."
        )

    def train_dataloader(self):
        """Create and return train dataloader."""
        return get_idx_dataloader(
            indices_list=list(range(len(self.train_data.src_node_ids))),
            batch_size=self.batch_size_per_device,
            shuffle=False,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return get_idx_dataloader(
            indices_list=list(range(len(self.val_data.src_node_ids))),
            batch_size=self.batch_size_per_device,
            shuffle=False,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return get_idx_dataloader(
            indices_list=list(range(len(self.test_data.src_node_ids))),
            batch_size=self.batch_size_per_device,
            shuffle=False,
        )


if __name__ == "__main__":
    datamodule = TGBLDataModule(dataset_name="tgbl-wiki", data_dir="datasets", batch_size=200)
    datamodule.setup()
    print(datamodule.node_raw_features.shape)
    print(len(set(datamodule.src_node_ids) | set(datamodule.dst_node_ids)))
    train_dataloader = datamodule.train_dataloader()
    for batch in train_dataloader:
        print(batch)
        break
