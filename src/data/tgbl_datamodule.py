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
        train_batch_size: int,
        eval_batch_size: int,
        fast_eval_batch_size: int,
        fast_eval_neg_num: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        sample_neighbor_strategy: str = "uniform",
        time_scaling_factor: float = 0.0,
        partition: str = "full",
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.dataset_name = dataset_name
        self.fast_eval_neg_num = fast_eval_neg_num
        self.fast_eval = False
        self.train_batch_size_per_device = train_batch_size
        self.eval_batch_size_per_device = eval_batch_size
        self.fast_eval_batch_size_per_device = fast_eval_batch_size

    def prepare_data(self) -> None:
        """Download data."""
        LinkPropPredDataset(name=self.hparams.dataset_name, preprocess=True)

    def setup(self, stage=None) -> None:
        """Load data."""
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.train_batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Train batch size ({self.hparams.train_batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            if self.hparams.eval_batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Eval batch size ({self.hparams.eval_batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            if self.hparams.fast_eval_batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Eval batch size ({self.hparams.fast_eval_batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.train_batch_size_per_device = (
                self.hparams.train_batch_size // self.trainer.world_size
            )
            self.eval_batch_size_per_device = (
                self.hparams.eval_batch_size // self.trainer.world_size
            )
            self.fast_eval_batch_size_per_device = (
                self.hparams.fast_eval_batch_size // self.trainer.world_size
            )

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

        if self.hparams.partition == "full":
            self.train_mask = dataset.train_mask
            self.val_mask = dataset.val_mask
            self.test_mask = dataset.test_mask
        elif self.hparams.partition == "earlier" or self.hparams.partition == "later":
            # partition == "earlier": use the first half of original train as train and val (80/20 split), and the original val as test
            # partition == "later": use the second half of original train as train and val (80/20 split), and the original val as test
            original_train_idx = np.nonzero(dataset.train_mask)[0]
            original_train_num = len(original_train_idx)
            new_train_num = int(original_train_num * 0.4)
            new_val_num = int(original_train_num * 0.1)
            train_mask = np.zeros_like(dataset.train_mask)
            val_mask = np.zeros_like(dataset.val_mask)
            if self.hparams.partition == "earlier":
                train_mask[original_train_idx[:new_train_num]] = True
                val_mask[original_train_idx[new_train_num : new_train_num + new_val_num]] = True
            else:
                train_mask[
                    original_train_idx[
                        new_train_num + new_val_num : 2 * new_train_num + new_val_num
                    ]
                ] = True
                val_mask[original_train_idx[2 * new_train_num + new_val_num :]] = True
            self.train_mask = train_mask
            self.val_mask = val_mask
            self.test_mask = dataset.val_mask

        self.eval_neg_edge_sampler = dataset.negative_sampler
        dataset.load_val_ns()
        dataset.load_test_ns()
        # missing 1 negative sample of (src: 267, dst: 8806, t: 2057208) and (src: 267, dst: 9059, t: 2057208) in tgbl-wiki val negative set (should be 999)
        # and missing 1 negative sample of (src: 2134, dst: 8314, t: 2461224) and (src: 2134, dst: 8545, t: 2461224) in tgbl-wiki test negative set (should be 999)
        # repeat the last negative sample to make the negative set complete
        if self.hparams.dataset_name == "tgbl-wiki":
            arr = self.eval_neg_edge_sampler.eval_set["val"][(267, 8806, 2057208)]
            self.eval_neg_edge_sampler.eval_set["val"][(267, 8806, 2057208)] = np.append(
                arr, arr[-1]
            )

            arr = self.eval_neg_edge_sampler.eval_set["val"][(267, 9059, 2057208)]
            self.eval_neg_edge_sampler.eval_set["val"][(267, 9059, 2057208)] = np.append(
                arr, arr[-1]
            )

            arr = self.eval_neg_edge_sampler.eval_set["test"][(2134, 8314, 2461224)]
            self.eval_neg_edge_sampler.eval_set["test"][(2134, 8314, 2461224)] = np.append(
                arr, arr[-1]
            )

            arr = self.eval_neg_edge_sampler.eval_set["test"][(2134, 8545, 2461224)]
            self.eval_neg_edge_sampler.eval_set["test"][(2134, 8545, 2461224)] = np.append(
                arr, arr[-1]
            )

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
            batch_size=self.train_batch_size_per_device,
            shuffle=False,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        batch_size = (
            self.fast_eval_batch_size_per_device
            if self.fast_eval
            else self.eval_batch_size_per_device
        )
        return get_idx_dataloader(
            indices_list=list(range(len(self.val_data.src_node_ids))),
            batch_size=batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        batch_size = (
            self.fast_eval_batch_size_per_device
            if self.fast_eval
            else self.eval_batch_size_per_device
        )
        return get_idx_dataloader(
            indices_list=list(range(len(self.test_data.src_node_ids))),
            batch_size=batch_size,
            shuffle=False,
        )


# if __name__ == "__main__":
#     import sys
#     sys.path.append("../..")
#     from src.utils.data import Data, get_idx_dataloader
#     datamodule = TGBLDataModule(dataset_name="tgbl-wiki", train_batch_size=200, eval_batch_size=10, fast_eval_batch_size=200, fast_eval_neg_num=19, partition="full")
#     datamodule.setup()
#     print(datamodule.node_raw_features.shape)
#     print(len(set(datamodule.src_node_ids) | set(datamodule.dst_node_ids)))
#     train_dataloader = datamodule.train_dataloader()
#     for batch in train_dataloader:
#         print(batch)
#         break

#     neg_num = None
#     for (pos_s, pos_d, pos_t), items in datamodule.eval_neg_edge_sampler.eval_set["val"].items():
#         if neg_num is None:
#             neg_num = len(items)
#             print(f"neg num: {neg_num}")
#         if neg_num is not None and len(items) != neg_num:
#             print(f"pos s: {pos_s}, pos d: {pos_d}, pos t: {pos_t} does not have {neg_num} negative examples, have {len(items)} instead")

#     neg_num = None
#     for (pos_s, pos_d, pos_t), items in datamodule.eval_neg_edge_sampler.eval_set["test"].items():
#         if neg_num is None:
#             neg_num = len(items)
#             print(f"neg num: {neg_num}")
#         if neg_num is not None and len(items) != neg_num:
#             print(f"pos s: {pos_s}, pos d: {pos_d}, pos t: {pos_t} does not have {neg_num} negative examples, have {len(items)} instead")


#     datamodule = TGBLDataModule(dataset_name="tgbl-wiki", train_batch_size=200, eval_batch_size=10, fast_eval_batch_size=200, fast_eval_neg_num=19, partition="earlier")
#     datamodule.setup()

#     datamodule = TGBLDataModule(dataset_name="tgbl-wiki", train_batch_size=200, eval_batch_size=10, fast_eval_batch_size=200, fast_eval_neg_num=19, partition="later")
#     datamodule.setup()
