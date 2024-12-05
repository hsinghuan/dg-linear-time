import os
import random
from pathlib import Path
from shutil import copytree

import numpy as np
import pandas as pd
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from pandas.testing import assert_frame_equal

from src.utils.data import Data, get_idx_dataloader


class NonTGBLDataModule(LightningDataModule):
    """DataModule for non-TGBL datasets."""

    def __init__(
        self,
        dataset_name: str,
        original_data_dir: str,
        preprocessed_data_dir: str,
        batch_size: int,
        val_ratio: float,
        test_ratio: float,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_shuffle: bool = False,
        train_negative_sample_strategy: str = "random",
        eval_negative_sample_strategy: str = "random",
    ) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.dataset_name = dataset_name
        self.dataset_type = "non_tgbl"
        self.original_data_dir = original_data_dir
        self.preprocessed_data_dir = preprocessed_data_dir
        self.batch_size_per_device = batch_size

        if dataset_name in ["wikipedia", "reddit", "mooc", "lastfm", "myket"]:
            self.bipartite = True
        else:
            self.bipartite = False

        self.node_feat_dim = 172
        self.edge_feat_dim = 172
        self.train_negative_sample_strategy = train_negative_sample_strategy
        self.eval_negative_sample_strategy = eval_negative_sample_strategy

    def prepare_data(self) -> None:
        # name of original and preprocessed paths
        path = os.path.join(self.original_data_dir, self.dataset_name, f"{self.dataset_name}.csv")
        out_df = os.path.join(
            self.preprocessed_data_dir, self.dataset_name, f"ml_{self.dataset_name}.csv"
        )
        out_feat = os.path.join(
            self.preprocessed_data_dir, self.dataset_name, f"ml_{self.dataset_name}.npy"
        )
        out_node_feat = os.path.join(
            self.preprocessed_data_dir, self.dataset_name, f"ml_{self.dataset_name}_node.npy"
        )

        # skip preprocess if the processed data already exists
        if os.path.exists(out_df) and os.path.exists(out_feat) and os.path.exists(out_node_feat):
            print(f"Already preprocessed dataset {self.dataset_name}")
            return

        # Directly use processed dataset by previous works for Enron, SocialEvo, and UCI datasets
        if self.dataset_name in ["enron", "SocialEvo", "uci"]:
            copytree(
                os.path.join(self.original_data_dir, self.dataset_name),
                os.path.join(self.preprocessed_data_dir, self.dataset_name),
                dirs_exist_ok=True,
            )
            print(
                f"the original dataset of {self.dataset_name} is unavailable, directly use the processed dataset by previous works."
            )
            return

        # make directory for preprocessed dataset
        Path(os.path.join(self.preprocessed_data_dir, self.dataset_name)).mkdir(
            parents=True, exist_ok=True
        )

        # assume that the original data is already downloaded
        assert os.path.exists(path)

        df, edge_feats = self._preprocess(path)
        new_df = self._reindex(df, self.bipartite)

        # edge feature for zero index, which is not used (since edge id starts from 1)
        empty = np.zeros(edge_feats.shape[1])[np.newaxis, :]
        # Stack arrays in sequence vertically(row wise),
        edge_feats = np.vstack([empty, edge_feats])

        # node features with one additional feature for zero index (since node id starts from 1)
        max_idx = max(new_df.u.max(), new_df.i.max())
        node_feats = np.zeros((max_idx + 1, self.node_feat_dim))

        print("number of nodes ", node_feats.shape[0] - 1)
        print("number of node features ", node_feats.shape[1])
        print("number of edges ", edge_feats.shape[0] - 1)
        print("number of edge features ", edge_feats.shape[1])
        print(out_df)
        new_df.to_csv(out_df)  # edge-list
        np.save(out_feat, edge_feats)  # edge features
        np.save(out_node_feat, node_feats)  # node features

        self._check_data()

    def _preprocess(self, path: str):
        """Read the original data file and return the DataFrame that has columns ['u', 'i', 'ts',
        'label', 'idx'] :param dataset_name: str, dataset name :return:"""
        u_list, i_list, ts_list, label_list = [], [], [], []
        feat_l = []
        idx_list = []

        with open(path) as f:
            # skip the first line
            s = next(f)
            previous_time = -1
            for idx, line in enumerate(f):
                e = line.strip().split(",")
                # user_id
                u = int(e[0])
                # item_id
                i = int(e[1])

                # timestamp
                ts = float(e[2])
                # check whether time in ascending order
                assert ts >= previous_time
                previous_time = ts
                # state_label
                label = float(e[3])

                # edge features
                feat = np.array([float(x) for x in e[4:]])

                u_list.append(u)
                i_list.append(i)
                ts_list.append(ts)
                label_list.append(label)
                # edge index
                idx_list.append(idx)

                feat_l.append(feat)
        return pd.DataFrame(
            {"u": u_list, "i": i_list, "ts": ts_list, "label": label_list, "idx": idx_list}
        ), np.array(feat_l)

    def _reindex(self, df, bipartite):
        """Reindex the ids of nodes and edges :param df: DataFrame :param bipartite: boolean,
        whether the graph is bipartite or not :return:"""
        new_df = df.copy()
        if bipartite:
            # check the ids of users and items
            assert df.u.max() - df.u.min() + 1 == len(df.u.unique())
            assert df.i.max() - df.i.min() + 1 == len(df.i.unique())
            assert df.u.min() == df.i.min() == 0

            # if bipartite, discriminate the source and target node by unique ids (target node id is counted based on source node id)
            upper_u = df.u.max() + 1
            new_i = df.i + upper_u

            new_df.i = new_i

        # make the id start from 1
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

        return new_df

    def _check_data(self):
        """Check whether the processed datasets are identical to the given processed datasets
        :param dataset_name: str, dataset name :return:"""
        # original data paths
        original_df = f"{self.original_data_dir}/{self.dataset_name}/ml_{self.dataset_name}.csv"
        original_feat = f"{self.original_data_dir}/{self.dataset_name}/ml_{self.dataset_name}.npy"
        original_node_feat = (
            f"{self.original_data_dir}/{self.dataset_name}/ml_{self.dataset_name}_node.npy"
        )

        # processed data paths
        out_df = f"{self.preprocessed_data_dir}/{self.dataset_name}/ml_{self.dataset_name}.csv"
        out_feat = f"{self.preprocessed_data_dir}/{self.dataset_name}/ml_{self.dataset_name}.npy"
        out_node_feat = (
            f"{self.preprocessed_data_dir}/{self.dataset_name}/ml_{self.dataset_name}_node.npy"
        )

        # Load original data
        origin_g_df = pd.read_csv(original_df)
        origin_e_feat = np.load(original_feat)
        origin_n_feat = np.load(original_node_feat)

        # Load processed data
        g_df = pd.read_csv(out_df)
        e_feat = np.load(out_feat)
        n_feat = np.load(out_node_feat)

        assert_frame_equal(origin_g_df, g_df)
        # check numbers of edges and edge features
        assert (
            origin_e_feat.shape == e_feat.shape
            and origin_e_feat.max() == e_feat.max()
            and origin_e_feat.min() == e_feat.min()
        )
        # check numbers of nodes and node features
        assert (
            origin_n_feat.shape == n_feat.shape
            and origin_n_feat.max() == n_feat.max()
            and origin_n_feat.min() == n_feat.min()
        )

    def setup(self, stage=None) -> None:
        """Load data and split it into train/val/test/new_node_val/new_node_test sets."""
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        # Load data and train/val/test split
        graph_df = pd.read_csv(
            os.path.join(
                self.preprocessed_data_dir, self.dataset_name, f"ml_{self.dataset_name}.csv"
            )
        )
        self.edge_raw_features = np.load(
            os.path.join(
                self.preprocessed_data_dir, self.dataset_name, f"ml_{self.dataset_name}.npy"
            )
        )
        self.node_raw_features = np.load(
            os.path.join(
                self.preprocessed_data_dir, self.dataset_name, f"ml_{self.dataset_name}_node.npy"
            )
        )

        assert (
            self.node_feat_dim >= self.node_raw_features.shape[1]
        ), f"Node feature dimension in dataset {self.dataset_name} is bigger than {self.node_feat_dim}!"
        assert (
            self.edge_feat_dim >= self.edge_raw_features.shape[1]
        ), f"Edge feature dimension in dataset {self.dataset_name} is bigger than {self.edge_feat_dim}!"

        # padding the features of edges and nodes to the same dimension (172 for all the datasets)
        if self.node_raw_features.shape[1] < self.node_feat_dim:
            node_zero_padding = np.zeros(
                (
                    self.node_raw_features.shape[0],
                    self.node_feat_dim - self.node_raw_features.shape[1],
                )
            )
            self.node_raw_features = np.concatenate(
                [self.node_raw_features, node_zero_padding], axis=1
            )
        if self.edge_raw_features.shape[1] < self.edge_feat_dim:
            edge_zero_padding = np.zeros(
                (
                    self.edge_raw_features.shape[0],
                    self.edge_feat_dim - self.edge_raw_features.shape[1],
                )
            )
            self.edge_raw_features = np.concatenate(
                [self.edge_raw_features, edge_zero_padding], axis=1
            )

        assert (
            self.node_feat_dim == self.node_raw_features.shape[1]
            and self.edge_feat_dim == self.edge_raw_features.shape[1]
        ), "Unaligned feature dimensions after feature padding!"

        # get the timestamp of validate and test set
        val_time, test_time = list(
            np.quantile(
                graph_df.ts,
                [
                    (1 - self.hparams.val_ratio - self.hparams.test_ratio),
                    (1 - self.hparams.test_ratio),
                ],
            )
        )

        src_node_ids = graph_df.u.values.astype(np.longlong)
        dst_node_ids = graph_df.i.values.astype(np.longlong)
        node_interact_times = graph_df.ts.values.astype(np.float64)
        edge_ids = graph_df.idx.values.astype(np.longlong)
        labels = graph_df.label.values

        self.full_data = Data(
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
            edge_ids=edge_ids,
            labels=labels,
        )

        # the setting of seed follows previous works
        random.seed(2020)

        # union to get node set
        node_set = set(src_node_ids) | set(dst_node_ids)
        num_total_unique_node_ids = len(node_set)

        # compute nodes which appear at test time
        test_node_set = set(src_node_ids[node_interact_times > val_time]).union(
            set(dst_node_ids[node_interact_times > val_time])
        )
        # sample nodes which we keep as new nodes (to test inductiveness), so then we have to remove all their edges from training
        new_test_node_set = set(random.sample(test_node_set, int(0.1 * num_total_unique_node_ids)))

        # mask for each source and destination to denote whether they are new test nodes
        new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
        new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

        # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
        observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

        # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
        train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

        self.train_data = Data(
            src_node_ids=src_node_ids[train_mask],
            dst_node_ids=dst_node_ids[train_mask],
            node_interact_times=node_interact_times[train_mask],
            edge_ids=edge_ids[train_mask],
            labels=labels[train_mask],
        )

        # define the new nodes sets for testing inductiveness of the model
        train_node_set = set(self.train_data.src_node_ids).union(self.train_data.dst_node_ids)
        assert len(train_node_set & new_test_node_set) == 0
        # new nodes that are not in the training set
        new_node_set = node_set - train_node_set

        val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
        test_mask = node_interact_times > test_time

        # new edges with new nodes in the val and test set (for inductive evaluation)
        edge_contains_new_node_mask = np.array(
            [
                (src_node_id in new_node_set or dst_node_id in new_node_set)
                for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)
            ]
        )
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

        # validation and test data
        self.val_data = Data(
            src_node_ids=src_node_ids[val_mask],
            dst_node_ids=dst_node_ids[val_mask],
            node_interact_times=node_interact_times[val_mask],
            edge_ids=edge_ids[val_mask],
            labels=labels[val_mask],
        )

        self.test_data = Data(
            src_node_ids=src_node_ids[test_mask],
            dst_node_ids=dst_node_ids[test_mask],
            node_interact_times=node_interact_times[test_mask],
            edge_ids=edge_ids[test_mask],
            labels=labels[test_mask],
        )

        # validation and test with edges that at least has one new node (not in training set)
        self.new_node_val_data = Data(
            src_node_ids=src_node_ids[new_node_val_mask],
            dst_node_ids=dst_node_ids[new_node_val_mask],
            node_interact_times=node_interact_times[new_node_val_mask],
            edge_ids=edge_ids[new_node_val_mask],
            labels=labels[new_node_val_mask],
        )

        self.new_node_test_data = Data(
            src_node_ids=src_node_ids[new_node_test_mask],
            dst_node_ids=dst_node_ids[new_node_test_mask],
            node_interact_times=node_interact_times[new_node_test_mask],
            edge_ids=edge_ids[new_node_test_mask],
            labels=labels[new_node_test_mask],
        )

        print(
            "The dataset has {} interactions, involving {} different nodes".format(
                self.full_data.num_interactions, self.full_data.num_unique_nodes
            )
        )
        print(
            "The training dataset has {} interactions, involving {} different nodes".format(
                self.train_data.num_interactions, self.train_data.num_unique_nodes
            )
        )
        print(
            "The validation dataset has {} interactions, involving {} different nodes".format(
                self.val_data.num_interactions, self.val_data.num_unique_nodes
            )
        )
        print(
            "The test dataset has {} interactions, involving {} different nodes".format(
                self.test_data.num_interactions, self.test_data.num_unique_nodes
            )
        )
        print(
            "The new node validation dataset has {} interactions, involving {} different nodes".format(
                self.new_node_val_data.num_interactions, self.new_node_val_data.num_unique_nodes
            )
        )
        print(
            "The new node test dataset has {} interactions, involving {} different nodes".format(
                self.new_node_test_data.num_interactions, self.new_node_test_data.num_unique_nodes
            )
        )
        print(
            "{} nodes were used for the inductive testing, i.e. are never seen during training".format(
                len(new_test_node_set)
            )
        )

    def train_dataloader(self):
        """Return train dataloader."""
        return get_idx_dataloader(
            indices_list=list(range(len(self.train_data.src_node_ids))),
            batch_size=self.batch_size_per_device,
            shuffle=self.hparams.train_shuffle,
        )

    def val_dataloader(self):
        """Return val dataloader."""
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


# if __name__ == "__main__":
#     import sys
#     sys.path.append("../..")
#     from src.utils.data import Data, get_idx_dataloader
#     datamodule = NonTGBLDataModule(
#         dataset_name='mooc',
#         original_data_dir='../../datasets/original',
#         preprocessed_data_dir='../../datasets/preprocessed',
#         batch_size=200,
#         val_ratio=0.15,
#         test_ratio=0.15,
#         num_workers=0,
#         pin_memory=False,
#         train_shuffle=False,
#     )
#     print(type(datamodule))
#     datamodule.prepare_data()
#     datamodule.setup()
#     train_loader = datamodule.train_dataloader()
#     val_loader = datamodule.val_dataloader()
#     test_loader = datamodule.test_dataloader()
#     print('train_loader:', train_loader)
#     print('val_loader:', val_loader)
#     print('test_loader:', test_loader)
