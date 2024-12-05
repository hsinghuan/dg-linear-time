import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from sklearn.metrics import average_precision_score, roc_auc_score
from tgb.linkproppred.evaluate import Evaluator

from src.models.linkpredictor import LinkPredictor
from src.models.modules.dygformer import DyGFormer
from src.models.modules.mlp import MergeLayer
from src.utils.data import Data, NegativeEdgeSampler, get_neighbor_sampler


class DyGFormerModule(LinkPredictor):
    """LightningModule for DyGFormer."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        time_feat_dim: int,
        channel_embedding_dim: int,
        output_dim: int = None,
        patch_size: int = 1,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        max_input_sequence_length: int = 512,
        embed_method: str = "original",
        sample_neighbor_strategy: str = "recent",
    ):
        super().__init__(sample_neighbor_strategy=sample_neighbor_strategy)
        """Initialize DyGFormer LightningModule."""
        self.save_hyperparameters(logger=False)
        self.loss_func = nn.BCELoss()
        self.model = None  # delay model instantiation until setup()

        # lists for length analysis
        self.train_history_length_analysis = {
            "pos": {
                "src": {
                    "avg_time_diffs": [],
                    "median_time_diffs": [],
                    "max_time_diffs": [],
                    "num_temporal_neighbors": [],
                },
                "dst": {
                    "avg_time_diffs": [],
                    "median_time_diffs": [],
                    "max_time_diffs": [],
                    "num_temporal_neighbors": [],
                },
            },
            "neg": {
                "src": {
                    "avg_time_diffs": [],
                    "median_time_diffs": [],
                    "max_time_diffs": [],
                    "num_temporal_neighbors": [],
                },
                "dst": {
                    "avg_time_diffs": [],
                    "median_time_diffs": [],
                    "max_time_diffs": [],
                    "num_temporal_neighbors": [],
                },
            },
        }
        self.val_history_length_analysis = {
            "pos": {
                "src": {
                    "avg_time_diffs": [],
                    "median_time_diffs": [],
                    "max_time_diffs": [],
                    "num_temporal_neighbors": [],
                },
                "dst": {
                    "avg_time_diffs": [],
                    "median_time_diffs": [],
                    "max_time_diffs": [],
                    "num_temporal_neighbors": [],
                },
            },
            "neg": {
                "src": {
                    "avg_time_diffs": [],
                    "median_time_diffs": [],
                    "max_time_diffs": [],
                    "num_temporal_neighbors": [],
                },
                "dst": {
                    "avg_time_diffs": [],
                    "median_time_diffs": [],
                    "max_time_diffs": [],
                    "num_temporal_neighbors": [],
                },
            },
        }
        self.test_history_length_analysis = {
            "pos": {
                "src": {
                    "avg_time_diffs": [],
                    "median_time_diffs": [],
                    "max_time_diffs": [],
                    "num_temporal_neighbors": [],
                },
                "dst": {
                    "avg_time_diffs": [],
                    "median_time_diffs": [],
                    "max_time_diffs": [],
                    "num_temporal_neighbors": [],
                },
            },
            "neg": {
                "src": {
                    "avg_time_diffs": [],
                    "median_time_diffs": [],
                    "max_time_diffs": [],
                    "num_temporal_neighbors": [],
                },
                "dst": {
                    "avg_time_diffs": [],
                    "median_time_diffs": [],
                    "max_time_diffs": [],
                    "num_temporal_neighbors": [],
                },
            },
        }

    def setup(self, stage: str) -> None:
        """Build model dynamically at the beginning of fit (train + validate), validate, test, or
        predict."""
        super().setup(stage)
        output_dim = (
            self.hparams.output_dim
            if self.hparams.output_dim is not None
            else self.node_raw_features.shape[1]
        )
        backbone = DyGFormer(
            node_raw_features=self.node_raw_features,
            edge_raw_features=self.edge_raw_features,
            neighbor_sampler=self.train_neighbor_sampler,
            time_feat_dim=self.hparams.time_feat_dim,
            channel_embedding_dim=self.hparams.channel_embedding_dim,
            output_dim=output_dim,
            patch_size=self.hparams.patch_size,
            num_layers=self.hparams.num_layers,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
            max_input_sequence_length=self.hparams.max_input_sequence_length,
            embed_method=self.hparams.embed_method,
            device=self.device,
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
        """One batch of training."""
        # print(f"batch device: {batch.device} self device: {self.device}")
        train_data_indices = batch.cpu().numpy()
        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = (
            self.train_data.src_node_ids[train_data_indices],
            self.train_data.dst_node_ids[train_data_indices],
            self.train_data.node_interact_times[train_data_indices],
        )
        # _, batch_neg_dst_node_ids = self.train_neg_edge_sampler.sample(
        #     size=len(batch_src_node_ids)
        # )

        if self.train_neg_edge_sampler.negative_sample_strategy == "historical":
            _, batch_neg_dst_node_ids = self.train_neg_edge_sampler.sample(
                size=len(batch_src_node_ids),
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids,
                current_batch_start_time=batch_node_interact_times[0],
                current_batch_end_time=batch_node_interact_times[-1],
            )
        elif self.train_neg_edge_sampler.negative_sample_strategy == "random":
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
        train_kwargs = {"analyze_length": self.current_epoch == 0}
        pred_out = self._pred_pos_neg_scores(
            pos_src=batch_src_node_ids,
            pos_dst=batch_dst_node_ids,
            pos_t=batch_node_interact_times,
            neg_src=batch_neg_src_node_ids,
            neg_dst=batch_neg_dst_node_ids,
            neg_t=batch_node_interact_times,
            **train_kwargs,
        )
        if train_kwargs["analyze_length"]:
            (
                pos_scores,
                neg_scores,
                pos_src_history_length_analysis,
                pos_dst_history_length_analysis,
                neg_src_history_length_analysis,
                neg_dst_history_length_analysis,
            ) = pred_out
            self.train_history_length_analysis["pos"]["src"]["avg_time_diffs"].append(
                pos_src_history_length_analysis["avg_time_diffs"]
            )
            self.train_history_length_analysis["pos"]["src"]["median_time_diffs"].append(
                pos_src_history_length_analysis["median_time_diffs"]
            )
            self.train_history_length_analysis["pos"]["src"]["max_time_diffs"].append(
                pos_src_history_length_analysis["max_time_diffs"]
            )
            self.train_history_length_analysis["pos"]["src"]["num_temporal_neighbors"].append(
                pos_src_history_length_analysis["num_temporal_neighbors"]
            )
            self.train_history_length_analysis["pos"]["dst"]["avg_time_diffs"].append(
                pos_dst_history_length_analysis["avg_time_diffs"]
            )
            self.train_history_length_analysis["pos"]["dst"]["median_time_diffs"].append(
                pos_dst_history_length_analysis["median_time_diffs"]
            )
            self.train_history_length_analysis["pos"]["dst"]["max_time_diffs"].append(
                pos_dst_history_length_analysis["max_time_diffs"]
            )
            self.train_history_length_analysis["pos"]["dst"]["num_temporal_neighbors"].append(
                pos_dst_history_length_analysis["num_temporal_neighbors"]
            )
            self.train_history_length_analysis["neg"]["src"]["avg_time_diffs"].append(
                neg_src_history_length_analysis["avg_time_diffs"]
            )
            self.train_history_length_analysis["neg"]["src"]["median_time_diffs"].append(
                neg_src_history_length_analysis["median_time_diffs"]
            )
            self.train_history_length_analysis["neg"]["src"]["max_time_diffs"].append(
                neg_src_history_length_analysis["max_time_diffs"]
            )
            self.train_history_length_analysis["neg"]["src"]["num_temporal_neighbors"].append(
                neg_src_history_length_analysis["num_temporal_neighbors"]
            )
            self.train_history_length_analysis["neg"]["dst"]["avg_time_diffs"].append(
                neg_dst_history_length_analysis["avg_time_diffs"]
            )
            self.train_history_length_analysis["neg"]["dst"]["median_time_diffs"].append(
                neg_dst_history_length_analysis["median_time_diffs"]
            )
            self.train_history_length_analysis["neg"]["dst"]["max_time_diffs"].append(
                neg_dst_history_length_analysis["max_time_diffs"]
            )
            self.train_history_length_analysis["neg"]["dst"]["num_temporal_neighbors"].append(
                neg_dst_history_length_analysis["num_temporal_neighbors"]
            )
        else:
            pos_scores, neg_scores = pred_out
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

    def _eval_step(self, batch: torch.Tensor, data: Data, stage: str) -> None:
        """One batch of AP and AUC evaluation. Reimplement this because we might want to do length
        analysis here.

        Forward the entire batch through the model. Use this during training for model selection
        due to its efficiency.
        """
        data_indices = batch.cpu().numpy()
        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = (
            data.src_node_ids[data_indices],
            data.dst_node_ids[data_indices],
            data.node_interact_times[data_indices],
            data.edge_ids[data_indices],
        )

        if self.dataset_type == "tgbl":
            # batch_neg_dst_node_ids_list: a list of list, where each internal list contains the ids of negative destination nodes for a positive source node
            # contain batch lists, each list with length num_negative_samples_per_node (20 in the TGB evaluation)
            # we should pay attention to the mappings of node ids, reduce 1 to convert to the original node ids
            batch_neg_dst_node_ids_list = self.eval_neg_edge_sampler.query_batch(
                pos_src=batch_src_node_ids - 1,
                pos_dst=batch_dst_node_ids - 1,
                pos_timestamp=batch_node_interact_times,
                split_mode=stage,
            )
            if self.fast_eval:
                batch_neg_dst_node_ids_list = self._subsample_neg_edges(
                    batch_neg_dst_node_ids_list, portion=self.fast_eval_neg_num
                )
            batch_neg_dst_node_ids = np.array(batch_neg_dst_node_ids_list) + 1
            num_negative_samples_per_node = batch_neg_dst_node_ids.shape[1]
            batch_neg_dst_node_ids = batch_neg_dst_node_ids.flatten()
            # ndarray, shape (batch_size * num_negative_samples_per_node, ),
            # value -> (src_node_1_id, src_node_1_id, ..., src_node_2_id, src_node_2_id, ...)
            batch_neg_src_node_ids = np.repeat(
                batch_src_node_ids, repeats=num_negative_samples_per_node, axis=0
            )

            # ndarray, shape (batch_size * num_negative_samples_per_node, ),
            # value -> (node_1_interact_time, node_1_interact_time, ..., node_2_interact_time, node_2_interact_time, ...)
            batch_neg_node_interact_times = np.repeat(
                batch_node_interact_times, repeats=num_negative_samples_per_node, axis=0
            )
        else:
            if stage == "val":
                eval_neg_edge_sampler = self.val_neg_edge_sampler
            elif stage == "test":
                eval_neg_edge_sampler = self.test_neg_edge_sampler

            if eval_neg_edge_sampler.negative_sample_strategy != "random":
                batch_neg_src_node_ids, batch_neg_dst_node_ids = eval_neg_edge_sampler.sample(
                    size=len(batch_src_node_ids),
                    batch_src_node_ids=batch_src_node_ids,
                    batch_dst_node_ids=batch_dst_node_ids,
                    current_batch_start_time=batch_node_interact_times[0],
                    current_batch_end_time=batch_node_interact_times[-1],
                )
            else:
                _, batch_neg_dst_node_ids = eval_neg_edge_sampler.sample(
                    size=len(batch_src_node_ids)
                )
                batch_neg_src_node_ids = batch_src_node_ids
            batch_neg_node_interact_times = batch_node_interact_times

        inference_kwargs = {"analyze_length": not self.fit}
        pred_out = self._pred_pos_neg_scores(
            pos_src=batch_src_node_ids,
            pos_dst=batch_dst_node_ids,
            pos_t=batch_node_interact_times,
            neg_src=batch_neg_src_node_ids,
            neg_dst=batch_neg_dst_node_ids,
            neg_t=batch_neg_node_interact_times,
            edge_ids=batch_edge_ids,
            **inference_kwargs,
        )
        if inference_kwargs["analyze_length"]:  # with length analysis
            (
                pos_scores,
                neg_scores,
                pos_src_history_length_analysis,
                pos_dst_history_length_analysis,
                neg_src_history_length_analysis,
                neg_dst_history_length_analysis,
            ) = pred_out
        else:
            pos_scores, neg_scores = pred_out
        scores = torch.cat((pos_scores, neg_scores), dim=0)
        labels = torch.cat((torch.ones_like(pos_scores), torch.zeros_like(neg_scores)), dim=0)
        loss = self.loss_func(input=scores, target=labels)
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if stage == "val":
            self.val_pos_scores.append(pos_scores)
            self.val_neg_scores.append(neg_scores)
            if inference_kwargs["analyze_length"]:
                self.val_history_length_analysis["pos"]["src"]["avg_time_diffs"].append(
                    pos_src_history_length_analysis["avg_time_diffs"]
                )
                self.val_history_length_analysis["pos"]["src"]["median_time_diffs"].append(
                    pos_src_history_length_analysis["median_time_diffs"]
                )
                self.val_history_length_analysis["pos"]["src"]["max_time_diffs"].append(
                    pos_src_history_length_analysis["max_time_diffs"]
                )
                self.val_history_length_analysis["pos"]["src"]["num_temporal_neighbors"].append(
                    pos_src_history_length_analysis["num_temporal_neighbors"]
                )
                self.val_history_length_analysis["pos"]["dst"]["avg_time_diffs"].append(
                    pos_dst_history_length_analysis["avg_time_diffs"]
                )
                self.val_history_length_analysis["pos"]["dst"]["median_time_diffs"].append(
                    pos_dst_history_length_analysis["median_time_diffs"]
                )
                self.val_history_length_analysis["pos"]["dst"]["max_time_diffs"].append(
                    pos_dst_history_length_analysis["max_time_diffs"]
                )
                self.val_history_length_analysis["pos"]["dst"]["num_temporal_neighbors"].append(
                    pos_dst_history_length_analysis["num_temporal_neighbors"]
                )
                self.val_history_length_analysis["neg"]["src"]["avg_time_diffs"].append(
                    neg_src_history_length_analysis["avg_time_diffs"]
                )
                self.val_history_length_analysis["neg"]["src"]["median_time_diffs"].append(
                    neg_src_history_length_analysis["median_time_diffs"]
                )
                self.val_history_length_analysis["neg"]["src"]["max_time_diffs"].append(
                    neg_src_history_length_analysis["max_time_diffs"]
                )
                self.val_history_length_analysis["neg"]["src"]["num_temporal_neighbors"].append(
                    neg_src_history_length_analysis["num_temporal_neighbors"]
                )
                self.val_history_length_analysis["neg"]["dst"]["avg_time_diffs"].append(
                    neg_dst_history_length_analysis["avg_time_diffs"]
                )
                self.val_history_length_analysis["neg"]["dst"]["median_time_diffs"].append(
                    neg_dst_history_length_analysis["median_time_diffs"]
                )
                self.val_history_length_analysis["neg"]["dst"]["max_time_diffs"].append(
                    neg_dst_history_length_analysis["max_time_diffs"]
                )
                self.val_history_length_analysis["neg"]["dst"]["num_temporal_neighbors"].append(
                    neg_dst_history_length_analysis["num_temporal_neighbors"]
                )
        elif stage == "test":
            self.test_pos_scores.append(pos_scores)
            self.test_neg_scores.append(neg_scores)
            if inference_kwargs["analyze_length"]:
                self.test_history_length_analysis["pos"]["src"]["avg_time_diffs"].append(
                    pos_src_history_length_analysis["avg_time_diffs"]
                )
                self.test_history_length_analysis["pos"]["src"]["median_time_diffs"].append(
                    pos_src_history_length_analysis["median_time_diffs"]
                )
                self.test_history_length_analysis["pos"]["src"]["max_time_diffs"].append(
                    pos_src_history_length_analysis["max_time_diffs"]
                )
                self.test_history_length_analysis["pos"]["src"]["num_temporal_neighbors"].append(
                    pos_src_history_length_analysis["num_temporal_neighbors"]
                )
                self.test_history_length_analysis["pos"]["dst"]["avg_time_diffs"].append(
                    pos_dst_history_length_analysis["avg_time_diffs"]
                )
                self.test_history_length_analysis["pos"]["dst"]["median_time_diffs"].append(
                    pos_dst_history_length_analysis["median_time_diffs"]
                )
                self.test_history_length_analysis["pos"]["dst"]["max_time_diffs"].append(
                    pos_dst_history_length_analysis["max_time_diffs"]
                )
                self.test_history_length_analysis["pos"]["dst"]["num_temporal_neighbors"].append(
                    pos_dst_history_length_analysis["num_temporal_neighbors"]
                )
                self.test_history_length_analysis["neg"]["src"]["avg_time_diffs"].append(
                    neg_src_history_length_analysis["avg_time_diffs"]
                )
                self.test_history_length_analysis["neg"]["src"]["median_time_diffs"].append(
                    neg_src_history_length_analysis["median_time_diffs"]
                )
                self.test_history_length_analysis["neg"]["src"]["max_time_diffs"].append(
                    neg_src_history_length_analysis["max_time_diffs"]
                )
                self.test_history_length_analysis["neg"]["src"]["num_temporal_neighbors"].append(
                    neg_src_history_length_analysis["num_temporal_neighbors"]
                )
                self.test_history_length_analysis["neg"]["dst"]["avg_time_diffs"].append(
                    neg_dst_history_length_analysis["avg_time_diffs"]
                )
                self.test_history_length_analysis["neg"]["dst"]["median_time_diffs"].append(
                    neg_dst_history_length_analysis["median_time_diffs"]
                )
                self.test_history_length_analysis["neg"]["dst"]["max_time_diffs"].append(
                    neg_dst_history_length_analysis["max_time_diffs"]
                )
                self.test_history_length_analysis["neg"]["dst"]["num_temporal_neighbors"].append(
                    neg_dst_history_length_analysis["num_temporal_neighbors"]
                )

        if self.dataset_type == "tgbl":
            for sample_idx in range(len(batch_src_node_ids)):
                # compute metric
                input_dict = {
                    # use slices instead of index to keep the dimension of y_pred_pos
                    "y_pred_pos": pos_scores[sample_idx : sample_idx + 1].cpu().numpy(),
                    "y_pred_neg": neg_scores[
                        sample_idx
                        * num_negative_samples_per_node : (sample_idx + 1)
                        * num_negative_samples_per_node
                    ]
                    .cpu()
                    .numpy(),
                    "eval_metric": [self.metric],
                }

                if stage == "val":
                    self.val_perf_list.append(self.evaluator.eval(input_dict)[self.metric])
                elif stage == "test":
                    self.test_perf_list.append(self.evaluator.eval(input_dict)[self.metric])
                else:
                    raise ValueError(f"Invalid stage: {stage}")

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
        pos_pred_out = self._pred_scores(pos_src, pos_dst, pos_t, **kwargs)
        analyze_length = "analyze_length" in kwargs and kwargs["analyze_length"]
        if analyze_length:  # with length analysis
            (
                pos_scores,
                pos_src_history_length_analysis,
                pos_dst_history_length_analysis,
            ) = pos_pred_out
        else:
            pos_scores = pos_pred_out
        neg_pred_out = self._pred_scores(neg_src, neg_dst, neg_t, **kwargs)
        if analyze_length:  # with length analysis
            (
                neg_scores,
                neg_src_history_length_analysis,
                neg_dst_history_length_analysis,
            ) = neg_pred_out
        else:
            neg_scores = neg_pred_out

        if analyze_length:
            return (
                pos_scores,
                neg_scores,
                pos_src_history_length_analysis,
                pos_dst_history_length_analysis,
                neg_src_history_length_analysis,
                neg_dst_history_length_analysis,
            )
        else:
            return pos_scores, neg_scores

    def _pred_scores(
        self, src: np.ndarray, dst: np.ndarray, t: np.ndarray, **kwargs
    ) -> torch.Tensor:
        """Predict the probability/score of (src[i], dst[i]) happening at time t[i]."""
        if "analyze_length" in kwargs and kwargs["analyze_length"]:
            (
                src_node_embeddings,
                dst_node_embeddings,
                src_history_length_analysis,
                dst_history_length_analysis,
            ) = self.model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=src, dst_node_ids=dst, node_interact_times=t, analyze_length=True
            )
            scores = (
                self.model[1](input_1=src_node_embeddings, input_2=dst_node_embeddings)
                .squeeze(dim=-1)
                .sigmoid()
            )
            return scores, src_history_length_analysis, dst_history_length_analysis
        else:
            src_node_embeddings, dst_node_embeddings = self.model[
                0
            ].compute_src_dst_node_temporal_embeddings(
                src_node_ids=src, dst_node_ids=dst, node_interact_times=t
            )
            scores = (
                self.model[1](input_1=src_node_embeddings, input_2=dst_node_embeddings)
                .squeeze(dim=-1)
                .sigmoid()
            )
            return scores

    def on_validation_epoch_start(self) -> None:
        """Set the neighbor sampler for validation."""
        self.model[0].set_neighbor_sampler(self.full_neighbor_sampler)

    def on_test_epoch_start(self) -> None:
        """Set the neighbor sampler for testing."""
        self.model[0].set_neighbor_sampler(self.full_neighbor_sampler)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Return optimizer for training."""
        optimizer = self.hparams.optimizer(params=self.model.parameters())
        return {"optimizer": optimizer}

    def on_train_epoch_end(self) -> None:
        """Aggregate training performance.

        Reimplement this because we want to analyze the history length.
        """
        analyze_length = self.current_epoch == 0
        self._aggregate_eval_log(
            "train",
            self.train_pos_scores,
            self.train_neg_scores,
            analyze_length=analyze_length,
            length_analysis=self.train_history_length_analysis,
        )
        self.train_pos_scores = []
        self.train_neg_scores = []
        if analyze_length:
            self.train_history_length_analysis = {
                "pos": {
                    "src": {
                        "avg_time_diffs": [],
                        "median_time_diffs": [],
                        "max_time_diffs": [],
                        "num_temporal_neighbors": [],
                    },
                    "dst": {
                        "avg_time_diffs": [],
                        "median_time_diffs": [],
                        "max_time_diffs": [],
                        "num_temporal_neighbors": [],
                    },
                },
                "neg": {
                    "src": {
                        "avg_time_diffs": [],
                        "median_time_diffs": [],
                        "max_time_diffs": [],
                        "num_temporal_neighbors": [],
                    },
                    "dst": {
                        "avg_time_diffs": [],
                        "median_time_diffs": [],
                        "max_time_diffs": [],
                        "num_temporal_neighbors": [],
                    },
                },
            }

    def on_validation_epoch_end(self) -> None:
        """Aggregate validation performance.

        Reimplement this because we want to analyze the history length.
        """
        analyze_length = not self.fit
        self._aggregate_eval_log(
            "val",
            self.val_pos_scores,
            self.val_neg_scores,
            self.val_perf_list,
            analyze_length=analyze_length,
            length_analysis=self.val_history_length_analysis,
        )
        self.val_pos_scores = []
        self.val_neg_scores = []
        self.val_perf_list = [] if self.dataset_type == "tgbl" else None
        if analyze_length:
            self.val_history_length_analysis = {
                "pos": {
                    "src": {
                        "avg_time_diffs": [],
                        "median_time_diffs": [],
                        "max_time_diffs": [],
                        "num_temporal_neighbors": [],
                    },
                    "dst": {
                        "avg_time_diffs": [],
                        "median_time_diffs": [],
                        "max_time_diffs": [],
                        "num_temporal_neighbors": [],
                    },
                },
                "neg": {
                    "src": {
                        "avg_time_diffs": [],
                        "median_time_diffs": [],
                        "max_time_diffs": [],
                        "num_temporal_neighbors": [],
                    },
                    "dst": {
                        "avg_time_diffs": [],
                        "median_time_diffs": [],
                        "max_time_diffs": [],
                        "num_temporal_neighbors": [],
                    },
                },
            }

    def on_test_epoch_end(self) -> None:
        """Aggregate testing performance.

        Reimplement this because we want to analyze the history length.
        """
        analyze_length = not self.fit
        self._aggregate_eval_log(
            "test",
            self.test_pos_scores,
            self.test_neg_scores,
            self.test_perf_list,
            analyze_length=analyze_length,
            length_analysis=self.test_history_length_analysis,
        )
        self.test_pos_scores = []
        self.test_neg_scores = []
        self.test_perf_list = [] if self.dataset_type == "tgbl" else None
        if analyze_length:
            self.test_history_length_analysis = {
                "pos": {
                    "src": {
                        "avg_time_diffs": [],
                        "median_time_diffs": [],
                        "max_time_diffs": [],
                        "num_temporal_neighbors": [],
                    },
                    "dst": {
                        "avg_time_diffs": [],
                        "median_time_diffs": [],
                        "max_time_diffs": [],
                        "num_temporal_neighbors": [],
                    },
                },
                "neg": {
                    "src": {
                        "avg_time_diffs": [],
                        "median_time_diffs": [],
                        "max_time_diffs": [],
                        "num_temporal_neighbors": [],
                    },
                    "dst": {
                        "avg_time_diffs": [],
                        "median_time_diffs": [],
                        "max_time_diffs": [],
                        "num_temporal_neighbors": [],
                    },
                },
            }

    def _aggregate_eval_log(
        self,
        stage: str,
        pos_scores: List[torch.Tensor],
        neg_scores: List[torch.Tensor],
        perf_list: List[float] = None,
        analyze_length: bool = False,
        length_analysis: Dict[str, Dict[str, Dict[str, List[float]]]] = None,
    ) -> None:
        """Aggregate and log the evaluation performance.

        Reimplement this because we want to analyze the history length.
        """
        pos_scores = torch.cat(pos_scores, dim=0)
        neg_scores = torch.cat(neg_scores, dim=0)
        scores = torch.cat((pos_scores, neg_scores), dim=0)
        labels = torch.cat((torch.ones_like(pos_scores), torch.zeros_like(neg_scores)), dim=0)

        if self.dataset_type == "tgbl":
            if self.fit:
                if self.fast_eval:
                    ap_log_name = f"{stage}/ap_fast"
                    auc_log_name = f"{stage}/auc_fast"
                    metric_log_name = f"{stage}/{self.metric}_fast"
                else:
                    ap_log_name = f"{stage}/ap"
                    auc_log_name = f"{stage}/auc"
                    metric_log_name = f"{stage}/{self.metric}"
            else:
                ap_log_name = f"{stage}/ap_final"
                auc_log_name = f"{stage}/auc_final"
                metric_log_name = f"{stage}/{self.metric}_final"
        else:
            if self.fit:
                if stage != "train":
                    ap_log_name = f"{stage}/{self.eval_negative_sample_strategy}/ap"
                    auc_log_name = f"{stage}/{self.eval_negative_sample_strategy}/auc"
                else:
                    ap_log_name = f"{stage}/{self.train_negative_sample_strategy}/ap"
                    auc_log_name = f"{stage}/{self.train_negative_sample_strategy}/auc"
            else:
                if stage != "train":
                    ap_log_name = f"{stage}/{self.eval_negative_sample_strategy}/ap_final"
                    auc_log_name = f"{stage}/{self.eval_negative_sample_strategy}/auc_final"
                else:
                    ap_log_name = f"{stage}/{self.train_negative_sample_strategy}/ap_final"
                    auc_log_name = f"{stage}/{self.train_negative_sample_strategy}/auc_final"

        self.log(
            ap_log_name,
            average_precision_score(y_true=labels.cpu().numpy(), y_score=scores.cpu().numpy()),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            auc_log_name,
            roc_auc_score(y_true=labels.cpu().numpy(), y_score=scores.cpu().numpy()),
            on_step=False,
            on_epoch=True,
        )

        if perf_list is not None:
            self.log(
                metric_log_name,
                np.mean(perf_list),
                on_step=False,
                on_epoch=True,
            )

        # # finer grained aggregation (aggregate to 4 bins) at trainer.validation() or trainer.test() stages
        # if not self.fit:
        #     pos_scores_num = len(pos_scores)
        #     neg_scores_num = len(neg_scores)
        #     for i in range(4):
        #         if i != 3:
        #             pos_scores_per_quarter = pos_scores[
        #                 i * pos_scores_num // 4 : (i + 1) * pos_scores_num // 4
        #             ]
        #             neg_scores_per_quarter = neg_scores[
        #                 i * neg_scores_num // 4 : (i + 1) * neg_scores_num // 4
        #             ]
        #         else:
        #             pos_scores_per_quarter = pos_scores[i * pos_scores_num // 4 :]
        #             neg_scores_per_quarter = neg_scores[i * neg_scores_num // 4 :]
        #         scores_per_quarter = torch.cat(
        #             (pos_scores_per_quarter, neg_scores_per_quarter), dim=0
        #         )
        #         labels_per_quarter = torch.cat(
        #             (
        #                 torch.ones_like(pos_scores_per_quarter),
        #                 torch.zeros_like(neg_scores_per_quarter),
        #             ),
        #             dim=0,
        #         )
        #         self.log(
        #             ap_log_name + f"_quarter_{i+1}",
        #             average_precision_score(
        #                 y_true=labels_per_quarter.cpu().numpy(),
        #                 y_score=scores_per_quarter.cpu().numpy(),
        #             ),
        #             on_step=False,
        #             on_epoch=True,
        #             prog_bar=progress_bar,
        #         )
        #         self.log(
        #             auc_log_name + f"_quarter_{i+1}",
        #             roc_auc_score(
        #                 y_true=labels_per_quarter.cpu().numpy(),
        #                 y_score=scores_per_quarter.cpu().numpy(),
        #             ),
        #             on_step=False,
        #             on_epoch=True,
        #             prog_bar=progress_bar,
        #         )

        #     if perf_list is not None:
        #         perf_list_num = len(perf_list)
        #         for i in range(4):
        #             if i != 3:
        #                 perf_list_per_quarter = perf_list[
        #                     i * perf_list_num // 4 : (i + 1) * perf_list_num // 4
        #                 ]
        #             else:
        #                 perf_list_per_quarter = perf_list[i * perf_list_num // 4 :]
        #             self.log(
        #                 metric_log_name + f"_quarter_{i+1}",
        #                 np.mean(perf_list_per_quarter),
        #                 on_step=False,
        #                 on_epoch=True,
        #                 prog_bar=progress_bar,
        #             )

        if analyze_length:
            length_analysis["pos"]["src"]["avg_time_diffs"] = np.concatenate(
                length_analysis["pos"]["src"]["avg_time_diffs"]
            )
            length_analysis["pos"]["src"]["median_time_diffs"] = np.concatenate(
                length_analysis["pos"]["src"]["median_time_diffs"]
            )
            length_analysis["pos"]["src"]["max_time_diffs"] = np.concatenate(
                length_analysis["pos"]["src"]["max_time_diffs"]
            )
            length_analysis["pos"]["src"]["num_temporal_neighbors"] = np.concatenate(
                length_analysis["pos"]["src"]["num_temporal_neighbors"]
            )

            length_analysis["pos"]["dst"]["avg_time_diffs"] = np.concatenate(
                length_analysis["pos"]["dst"]["avg_time_diffs"]
            )
            length_analysis["pos"]["dst"]["median_time_diffs"] = np.concatenate(
                length_analysis["pos"]["dst"]["median_time_diffs"]
            )
            length_analysis["pos"]["dst"]["max_time_diffs"] = np.concatenate(
                length_analysis["pos"]["dst"]["max_time_diffs"]
            )
            length_analysis["pos"]["dst"]["num_temporal_neighbors"] = np.concatenate(
                length_analysis["pos"]["dst"]["num_temporal_neighbors"]
            )

            length_analysis["neg"]["src"]["avg_time_diffs"] = np.concatenate(
                length_analysis["neg"]["src"]["avg_time_diffs"]
            )
            length_analysis["neg"]["src"]["median_time_diffs"] = np.concatenate(
                length_analysis["neg"]["src"]["median_time_diffs"]
            )
            length_analysis["neg"]["src"]["max_time_diffs"] = np.concatenate(
                length_analysis["neg"]["src"]["max_time_diffs"]
            )
            length_analysis["neg"]["src"]["num_temporal_neighbors"] = np.concatenate(
                length_analysis["neg"]["src"]["num_temporal_neighbors"]
            )

            length_analysis["neg"]["dst"]["avg_time_diffs"] = np.concatenate(
                length_analysis["neg"]["dst"]["avg_time_diffs"]
            )
            length_analysis["neg"]["dst"]["median_time_diffs"] = np.concatenate(
                length_analysis["neg"]["dst"]["median_time_diffs"]
            )
            length_analysis["neg"]["dst"]["max_time_diffs"] = np.concatenate(
                length_analysis["neg"]["dst"]["max_time_diffs"]
            )
            length_analysis["neg"]["dst"]["num_temporal_neighbors"] = np.concatenate(
                length_analysis["neg"]["dst"]["num_temporal_neighbors"]
            )

            length_analysis["pos"]["scores"] = pos_scores.cpu().numpy()
            length_analysis["neg"]["scores"] = neg_scores.cpu().numpy()
            checkpoint_dir = self.trainer.checkpoint_callback.dirpath
            if checkpoint_dir is not None:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                torch.save(
                    length_analysis,
                    f"{checkpoint_dir}/{stage}_{self.eval_negative_sample_strategy}_length_analysis.pt",
                )

    # def predict_step(self, batch):
    #     """One batch of prediction."""
    #     test_data_indices = batch.cpu().numpy()
    #     batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = (
    #         self.test_data.src_node_ids[test_data_indices],
    #         self.test_data.dst_node_ids[test_data_indices],
    #         self.test_data.node_interact_times[test_data_indices],
    #     )
    #     scores = self._pred_scores(
    #         src=batch_src_node_ids, dst=batch_dst_node_ids, t=batch_node_interact_times
    #     )
    #     return scores
