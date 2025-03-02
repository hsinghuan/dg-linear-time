from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from src.models.linkpredictor import LinkPredictor
from src.models.modules.mlp import MergeLayer
from src.models.modules.tgat import TGAT
from src.utils.analysis import analyze_target_historical_event_time_diff


class TGATModule(LinkPredictor):
    """LightningModule for TGAT."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        time_feat_dim: int,
        output_dim: int = None,
        num_layers: int = 2,
        num_heads: int = 2,
        num_neighbors: int = 20,
        dropout: float = 0.1,
        sample_neighbor_strategy: str = "recent",
        time_encoding_method: str = "sinusoidal",
        scale_timediff: bool = False,
        use_positional_embedding: bool = False,
    ):
        super().__init__(sample_neighbor_strategy=sample_neighbor_strategy)
        self.save_hyperparameters(logger=False)
        self.loss_func = nn.BCELoss()
        self.model = None  # delay model instantiation until setup()

    def setup(self, stage: str):
        """Build model dynamically at the beginning of fit (train + validate), validate, test, or
        predict."""
        super().setup(stage)
        output_dim = (
            self.hparams.output_dim
            if self.hparams.output_dim is not None
            else self.node_raw_features.shape[1]
        )

        if self.hparams.scale_timediff:
            node_ids = np.concatenate([self.train_data.src_node_ids, self.train_data.dst_node_ids])
            node_interact_times = np.concatenate(
                [self.train_data.node_interact_times, self.train_data.node_interact_times]
            )
            (
                _,
                _,
                nodes_neighbor_times_list,
            ) = self.train_neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=node_ids, node_interact_times=node_interact_times
            )
            (
                avg_time_diffs_per_tgt_edge,
                median_time_diffs_per_tgt_edge,
                _,
                _,
            ) = analyze_target_historical_event_time_diff(
                nodes_neighbor_times_list,
                node_interact_times,
                self.hparams.num_neighbors,
                sample_neighbor_strategy=self.hparams.sample_neighbor_strategy,
            )
            self.avg_time_diff = np.nanmean(avg_time_diffs_per_tgt_edge)
            self.std_time_diff = np.nanstd(avg_time_diffs_per_tgt_edge)
            self.median_time_diff = np.nanmean(median_time_diffs_per_tgt_edge)
        else:
            self.avg_time_diff = 0
            self.median_time_diff = None
            self.std_time_diff = 1

        backbone = TGAT(
            node_raw_features=self.node_raw_features,
            edge_raw_features=self.edge_raw_features,
            neighbor_sampler=self.train_neighbor_sampler,
            time_feat_dim=self.hparams.time_feat_dim,
            num_layers=self.hparams.num_layers,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
            time_encoding_method=self.hparams.time_encoding_method,
            avg_time_diff=self.avg_time_diff,
            std_time_diff=self.std_time_diff,
            device=self.device,
            use_positional_embedding=self.hparams.use_positional_embedding,
        )

        link_predictor = MergeLayer(
            input_dim1=output_dim,
            input_dim2=output_dim,
            hidden_dim=output_dim,
            output_dim=1,
        )
        self.model = nn.Sequential(backbone, link_predictor).to(self.device)

    def on_train_epoch_start(self) -> None:
        """Set the neighbor sampler for training."""
        self.model[0].set_neighbor_sampler(self.train_neighbor_sampler)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        train_data_indices = batch.cpu().numpy()
        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = (
            self.train_data.src_node_ids[train_data_indices],
            self.train_data.dst_node_ids[train_data_indices],
            self.train_data.node_interact_times[train_data_indices],
        )

        if self.train_negative_sample_strategy == "historical":
            _, batch_neg_dst_node_ids = self.train_neg_edge_sampler.sample(
                size=len(batch_src_node_ids),
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids,
                current_batch_start_time=batch_node_interact_times[0],
                current_batch_end_time=batch_node_interact_times[-1],
            )
        elif self.train_negative_sample_strategy == "random":
            _, batch_neg_dst_node_ids = self.train_neg_edge_sampler.sample(
                size=len(batch_src_node_ids)
            )

        batch_neg_src_node_ids = batch_src_node_ids

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
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

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
        **kwargs,
    ) -> torch.Tensor:
        """Predict the probabilities/scores of (pos_src[i], pos_dst[i]) happening at time pos_t[i]
        and (neg_src[i], neg_dst[i]) happening at time neg_t[i]."""
        neg_scores = self._pred_scores(src=neg_src, dst=neg_dst, t=neg_t)
        pos_scores = self._pred_scores(src=pos_src, dst=pos_dst, t=pos_t)
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
