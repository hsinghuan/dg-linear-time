from typing import Any, Dict, List

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from tgb.linkproppred.evaluate import Evaluator

from src.models.modules.graphmixer import GraphMixer
from src.models.modules.mlp import MergeLayer
from src.utils.data import Data, NegativeEdgeSampler, get_neighbor_sampler


class GraphMixerModule(L.LightningModule):
    """GraphMixer Lightning Module."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        time_feat_dim: int,
        num_tokens: int,
        output_dim: int = 172,
        num_layers: int = 2,
        token_dim_expansion_factor: float = 0.5,
        channel_dim_expansion_factor: float = 4.0,
        dropout: float = 0.1,
        num_neighbors: int = 20,
        time_gap: int = 2000,
    ):
        """Initialize GraphMixer Lightning Module."""
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.loss_func = nn.BCELoss()
        self.model = None  # delay model instantiation until setup()
        self.val_perf_list = []
        self.test_perf_list = []

    def setup(self, stage: str) -> None:  # TODO: think about what to do when val or test is called
        """Build model dynamically at the beginning of fit (train + validate), validate, test, or
        predict."""
        self.node_raw_features = self.trainer.datamodule.node_raw_features
        self.edge_raw_features = self.trainer.datamodule.edge_raw_features
        self.full_data = self.trainer.datamodule.full_data
        self.train_data = self.trainer.datamodule.train_data
        self.val_data = self.trainer.datamodule.val_data
        self.test_data = self.trainer.datamodule.test_data
        self.train_neighbor_sampler = get_neighbor_sampler(
            data=self.train_data,
            sample_neighbor_strategy="recent",
            seed=0,
        )
        self.full_neighbor_sampler = get_neighbor_sampler(
            data=self.full_data,
            sample_neighbor_strategy="recent",
            seed=1,
        )
        self.train_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=self.train_data.src_node_ids, dst_node_ids=self.train_data.dst_node_ids
        )
        self.eval_neg_edge_sampler = self.trainer.datamodule.eval_neg_edge_sampler
        self.metric = self.trainer.datamodule.eval_metric_name
        self.evaluator = Evaluator(self.trainer.datamodule.dataset_name)
        backbone = GraphMixer(
            node_raw_features=self.node_raw_features,
            edge_raw_features=self.edge_raw_features,
            neighbor_sampler=self.train_neighbor_sampler,
            time_feat_dim=self.hparams.time_feat_dim,
            num_tokens=self.hparams.num_tokens,
            output_dim=self.hparams.output_dim,
            num_layers=self.hparams.num_layers,
            token_dim_expansion_factor=self.hparams.token_dim_expansion_factor,
            channel_dim_expansion_factor=self.hparams.channel_dim_expansion_factor,
            dropout=self.hparams.dropout,
            device=self.device,
        )
        link_predictor = MergeLayer(
            input_dim1=self.hparams.output_dim,
            input_dim2=self.hparams.output_dim,
            hidden_dim=self.hparams.output_dim,
            output_dim=1,
        )
        self.model = nn.Sequential(backbone, link_predictor).to(self.device)

    def on_train_epoch_start(self) -> None:
        """Set the neighbor sampler for training."""
        self.model[0].set_neighbor_sampler(self.train_neighbor_sampler)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """One batch of training."""
        # print(f"batch device: {batch.device} self device: {self.device}")
        train_data_indices = batch.cpu().numpy()
        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = (
            self.train_data.src_node_ids[train_data_indices],
            self.train_data.dst_node_ids[train_data_indices],
            self.train_data.node_interact_times[train_data_indices],
        )
        _, batch_neg_dst_node_ids = self.train_neg_edge_sampler.sample(
            size=len(batch_src_node_ids)
        )
        batch_neg_src_node_ids = batch_src_node_ids

        # get temporal embedding of source and destination nodes
        # two Tensors, with shape (batch_size, output_dim)
        batch_src_node_embeddings, batch_dst_node_embeddings = self.model[
            0
        ].compute_src_dst_node_temporal_embeddings(
            src_node_ids=batch_src_node_ids,
            dst_node_ids=batch_dst_node_ids,
            node_interact_times=batch_node_interact_times,
            num_neighbors=self.hparams.num_neighbors,
            time_gap=self.hparams.time_gap,
        )
        # get temporal embedding of negative source and negative destination nodes
        # two Tensors, with shape (batch_size, output_dim)
        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = self.model[
            0
        ].compute_src_dst_node_temporal_embeddings(
            src_node_ids=batch_neg_src_node_ids,
            dst_node_ids=batch_neg_dst_node_ids,
            node_interact_times=batch_node_interact_times,
            num_neighbors=self.hparams.num_neighbors,
            time_gap=self.hparams.time_gap,
        )
        # print(f"Time to compute embeddings: {timeit.default_timer() - start_compute_embeddings}")
        # get positive and negative probabilities, shape (batch_size, )
        positive_probabilities = (
            self.model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings)
            .squeeze(dim=-1)
            .sigmoid()
        )
        negative_probabilities = (
            self.model[1](
                input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings
            )
            .squeeze(dim=-1)
            .sigmoid()
        )
        # print(f"embeddings device :{batch_src_node_embeddings.device} {batch_dst_node_embeddings.device}")
        predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
        labels = torch.cat(
            [torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)],
            dim=0,
        )

        loss = self.loss_func(input=predicts, target=labels)

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(
            "train/AP",
            average_precision_score(
                y_true=labels.detach().cpu().numpy(), y_score=predicts.detach().cpu().numpy()
            ),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "train/AUC",
            roc_auc_score(
                y_true=labels.detach().cpu().numpy(), y_score=predicts.detach().cpu().numpy()
            ),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        """Set the neighbor sampler for validation."""
        self.model[0].set_neighbor_sampler(self.full_neighbor_sampler)

    def validation_step(self, batch: torch.Tensor) -> None:
        """One batch of validation."""
        self._eval_step(batch, self.val_data, "val")

    def on_test_epoch_start(self) -> None:
        """Set the neighbor sampler for testing."""
        self.model[0].set_neighbor_sampler(self.full_neighbor_sampler)

    def test_step(self, batch: torch.Tensor) -> None:
        """One batch of testing."""
        self._eval_step(batch, self.test_data, "test")

    def _eval_step(self, batch: torch.Tensor, data: Data, stage: str) -> None:
        """One batch of evaluation, shared by validation_step and test_step."""
        data_indices = batch.cpu().numpy()
        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = (
            data.src_node_ids[data_indices],
            data.dst_node_ids[data_indices],
            data.node_interact_times[data_indices],
        )
        # batch_neg_dst_node_ids_list: a list of list, where each internal list contains the ids of negative destination nodes for a positive source node
        # contain batch lists, each list with length num_negative_samples_per_node (20 in the TGB evaluation)
        # we should pay attention to the mappings of node ids, reduce 1 to convert to the original node ids
        batch_neg_dst_node_ids_list = self.eval_neg_edge_sampler.query_batch(
            pos_src=batch_src_node_ids - 1,
            pos_dst=batch_dst_node_ids - 1,
            pos_timestamp=batch_node_interact_times,
            split_mode=stage,
        )
        predicts = []
        labels = []
        for idx, neg_batch in enumerate(batch_neg_dst_node_ids_list):
            src = np.array([int(batch_src_node_ids[idx]) for _ in range(len(neg_batch) + 1)])
            dst = np.concatenate(
                ([np.array([batch_dst_node_ids[idx]]), np.array(neg_batch) + 1]), axis=0
            )  # add 1 to convert the node ids into dyglib format
            t = np.array([batch_node_interact_times[idx] for _ in range(len(neg_batch) + 1)])
            src_node_embeddings, dst_node_embeddings = self.model[
                0
            ].compute_src_dst_node_temporal_embeddings(
                src_node_ids=src,
                dst_node_ids=dst,
                node_interact_times=t,
                num_neighbors=self.hparams.num_neighbors,
                time_gap=self.hparams.time_gap,
            )
            # print(f"Time to compute embeddings: {timeit.default_timer() - start_compute_embeddings}")
            y_pred = (
                self.model[1](input_1=src_node_embeddings, input_2=dst_node_embeddings)
                .squeeze(dim=-1)
                .sigmoid()
                .cpu()
            )
            # compute MRR
            input_dict = {
                "y_pred_pos": y_pred[0].numpy(),
                "y_pred_neg": y_pred[1:].numpy(),
                "eval_metric": [self.metric],
            }
            if stage == "val":
                self.val_perf_list.append(self.evaluator.eval(input_dict)[self.metric])
            elif stage == "test":
                self.test_perf_list.append(self.evaluator.eval(input_dict)[self.metric])
            else:
                raise ValueError(f"Invalid stage: {stage}")
            predicts.append(y_pred)
            labels.append(torch.cat((torch.ones(1), torch.zeros(len(neg_batch))), dim=0))

        predicts = torch.cat(predicts, dim=0)
        labels = torch.cat(labels, dim=0)
        loss = self.loss_func(input=predicts, target=labels)
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Aggregate validation performance."""
        self._aggregate_eval_log(self.val_perf_list, "val")
        self.val_perf_list = []

    def on_test_epoch_end(self) -> None:
        """Aggregate testing performance."""
        self._aggregate_eval_log(self.test_perf_list, "test")
        self.test_perf_list = []

    def configure_optimizers(self) -> Dict[str, Any]:
        """Return optimizer for training."""
        optimizer = self.hparams.optimizer(params=self.model.parameters())
        return {"optimizer": optimizer}

    def _aggregate_eval_log(self, perf_list: List, stage: str):
        """Aggregate and log the evaluation performance."""
        self.log(
            f"{stage}/{self.metric}",
            np.mean(perf_list),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
