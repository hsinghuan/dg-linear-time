from typing import Any, Dict, List

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from tgb.linkproppred.evaluate import Evaluator

from src.models.linkpredictor import LinkPredictor
from src.models.modules.graphmixer import GraphMixer
from src.models.modules.mlp import MergeLayer
from src.utils.data import Data, NegativeEdgeSampler, get_neighbor_sampler


class GraphMixerModule(LinkPredictor):
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
        sample_neighbor_strategy: str = "recent",
    ):
        """Initialize GraphMixer Lightning Module."""
        super().__init__(sample_neighbor_strategy=sample_neighbor_strategy)

    def setup(self, stage: str) -> None:
        """Build model dynamically at the beginning of fit (train + validate), validate, test, or
        predict."""
        super().setup(stage)

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
        head = MergeLayer(
            input_dim1=self.hparams.output_dim,
            input_dim2=self.hparams.output_dim,
            hidden_dim=self.hparams.output_dim,
            output_dim=1,
        )
        self.model = nn.Sequential(backbone, head).to(self.device)

    def on_train_epoch_start(self) -> None:
        """Set the neighbor sampler for training."""
        self.model[0].set_neighbor_sampler(self.train_neighbor_sampler)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """One batch of training."""
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

        # pos_scores = self._pred_scores(
        #     batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times
        # )
        # neg_scores = self._pred_scores(
        #     batch_neg_src_node_ids, batch_neg_dst_node_ids, batch_node_interact_times
        # )
        pos_scores, neg_scores = self._pred_pos_neg_scores(
            pos_src=batch_src_node_ids,
            pos_dst=batch_dst_node_ids,
            pos_t=batch_node_interact_times,
            neg_src=batch_neg_src_node_ids,
            neg_dst=batch_neg_dst_node_ids,
            neg_t=batch_node_interact_times,
        )
        self.train_pos_scores.append(pos_scores.detach())
        self.train_neg_scores.append(neg_scores.detach())

        predicts = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat(
            [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)],
            dim=0,
        )

        loss = self.loss_func(input=predicts, target=labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        """Set the neighbor sampler for validation."""
        self.model[0].set_neighbor_sampler(self.full_neighbor_sampler)

    def on_test_epoch_start(self) -> None:
        """Set the neighbor sampler for testing."""
        self.model[0].set_neighbor_sampler(self.full_neighbor_sampler)

    def _pred_pos_neg_scores(
        self,
        pos_src: np.ndarray,
        pos_dst: np.ndarray,
        pos_t: np.ndarray,
        neg_src: np.ndarray,
        neg_dst: np.ndarray,
        neg_t: np.ndarray,
        **kwargs
    ) -> torch.Tensor:
        """Predict the probabilities/scores of (pos_src[i], pos_dst[i]) happening at time pos_t[i]
        and (neg_src[i], neg_dst[i]) happening at time neg_t[i]."""
        pos_scores = self._pred_scores(pos_src, pos_dst, pos_t)
        neg_scores = self._pred_scores(neg_src, neg_dst, neg_t)
        return pos_scores, neg_scores

    def _pred_scores(
        self, src: np.ndarray, dst: np.ndarray, t: np.ndarray, **kwargs
    ) -> torch.Tensor:
        """Predict the probability/score of (src[i], dst[i]) happening at time t[i]."""
        src_node_embeddings, dst_node_embeddings = self.model[
            0
        ].compute_src_dst_node_temporal_embeddings(
            src_node_ids=src,
            dst_node_ids=dst,
            node_interact_times=t,
            num_neighbors=self.hparams.num_neighbors,
            time_gap=self.hparams.time_gap,
        )
        scores = (
            self.model[1](input_1=src_node_embeddings, input_2=dst_node_embeddings)
            .squeeze(dim=-1)
            .sigmoid()
        )
        return scores

    def configure_optimizers(self) -> Dict[str, Any]:
        """Return optimizer for training."""
        optimizer = self.hparams.optimizer(params=self.model.parameters())
        return {"optimizer": optimizer}
