import random
from typing import Any, Dict, List, Union

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from tgb.linkproppred.evaluate import Evaluator

from src.utils.data import Data, NegativeEdgeSampler, get_neighbor_sampler


class LinkPredictor(L.LightningModule):
    """A common parent class for all learning-based link predictors."""

    def __init__(self, sample_neighbor_strategy: str = "recent"):
        """Initialize the link predictor parent class."""
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.loss_func = nn.BCELoss()

    def setup(self, stage: str) -> None:
        """Get data, samplers, metric and evaluator before fit (train + validate), validate, and
        test."""
        self.node_raw_features = self.trainer.datamodule.node_raw_features
        self.edge_raw_features = self.trainer.datamodule.edge_raw_features
        self.full_data = self.trainer.datamodule.full_data
        self.train_data = self.trainer.datamodule.train_data
        self.val_data = self.trainer.datamodule.val_data
        self.test_data = self.trainer.datamodule.test_data
        self.train_neighbor_sampler = get_neighbor_sampler(
            data=self.train_data,
            sample_neighbor_strategy=self.hparams.sample_neighbor_strategy,
            seed=0,
        )
        self.full_neighbor_sampler = get_neighbor_sampler(
            data=self.full_data,
            sample_neighbor_strategy=self.hparams.sample_neighbor_strategy,
            seed=1,
        )
        self.train_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=self.train_data.src_node_ids, dst_node_ids=self.train_data.dst_node_ids
        )
        self.eval_neg_edge_sampler = self.trainer.datamodule.eval_neg_edge_sampler
        self.metric = self.trainer.datamodule.eval_metric_name
        self.evaluator = Evaluator(self.trainer.datamodule.dataset_name)
        self.fast_eval_neg_num = self.trainer.datamodule.fast_eval_neg_num

        if stage == "fit":
            self.fast_eval = True
            self.trainer.datamodule.fast_eval = True
        elif stage == "validate" or stage == "test":
            self.fast_eval = False
            self.trainer.datamodule.fast_eval = False
        self.train_pos_scores = []
        self.train_neg_scores = []
        self.val_pos_scores = []
        self.val_neg_scores = []
        self.val_perf_list = []
        self.test_pos_scores = []
        self.test_neg_scores = []
        self.test_perf_list = []

    def validation_step(self, batch: torch.Tensor) -> None:
        """One batch of validation."""
        self._eval_step(batch, self.val_data, "val")

    def test_step(self, batch: torch.Tensor) -> None:
        """One batch of testing."""
        self._eval_step(batch, self.test_data, "test")

    def _eval_step(self, batch: torch.Tensor, data: Data, stage: str) -> None:
        """One batch of AP and AUC evaluation.

        Forward the entire batch through the model. Use this during training for model selection
        due to its efficiency.
        """
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
        if self.fast_eval:
            batch_neg_dst_node_ids_list = self._subsample_neg_edges(
                batch_neg_dst_node_ids_list, portion=self.fast_eval_neg_num
            )
        batch_neg_dst_node_ids = np.array(batch_neg_dst_node_ids_list) + 1

        num_negative_samples_per_node = batch_neg_dst_node_ids.shape[1]
        # ndarray, shape (batch_size * num_negative_samples_per_node, ),
        # value -> (src_node_1_id, src_node_1_id, ..., src_node_2_id, src_node_2_id, ...)
        repeated_batch_src_node_ids = np.repeat(
            batch_src_node_ids, repeats=num_negative_samples_per_node, axis=0
        )
        # ndarray, shape (batch_size * num_negative_samples_per_node, ),
        # value -> (node_1_interact_time, node_1_interact_time, ..., node_2_interact_time, node_2_interact_time, ...)
        repeated_batch_node_interact_times = np.repeat(
            batch_node_interact_times, repeats=num_negative_samples_per_node, axis=0
        )

        # forward negative edges first because they do not change the memories of memory-based model
        neg_scores = self._pred_scores(
            src=repeated_batch_src_node_ids,
            dst=batch_neg_dst_node_ids.flatten(),
            t=repeated_batch_node_interact_times,
        )

        pos_scores = self._pred_scores(
            src=batch_src_node_ids, dst=batch_dst_node_ids, t=batch_node_interact_times
        )

        scores = torch.cat((pos_scores, neg_scores), dim=0)
        labels = torch.cat((torch.ones_like(pos_scores), torch.zeros_like(neg_scores)), dim=0)
        loss = self.loss_func(input=scores, target=labels)
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # if self.fast_eval:
        if stage == "val":
            self.val_pos_scores.append(pos_scores)
            self.val_neg_scores.append(neg_scores)
        elif stage == "test":
            self.test_pos_scores.append(pos_scores)
            self.test_neg_scores.append(neg_scores)
        # else:
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

    def _pred_scores(self, src: np.ndarray, dst: np.ndarray, t: np.ndarray) -> torch.Tensor:
        """Predict the probability/score of (src[i], dst[i]) happening at time t[i]."""
        raise NotImplementedError

    def on_train_epoch_end(self) -> None:
        """Aggregate training performance."""
        # print(len(self.train_pos_scores))
        self._aggregate_eval_log("train", self.train_pos_scores, self.train_neg_scores)
        self.train_pos_scores = []
        self.train_neg_scores = []

    def on_validation_epoch_end(self) -> None:
        """Aggregate validation performance."""
        self._aggregate_eval_log(
            "val", self.val_pos_scores, self.val_neg_scores, self.val_perf_list
        )
        self.val_pos_scores = []
        self.val_neg_scores = []
        self.val_perf_list = []

    def on_test_epoch_end(self) -> None:
        """Aggregate testing performance."""
        self._aggregate_eval_log(
            "test", self.test_pos_scores, self.test_neg_scores, self.test_perf_list
        )
        self.test_pos_scores = []
        self.test_neg_scores = []
        self.test_perf_list = []

    def _aggregate_eval_log(
        self,
        stage: str,
        pos_scores: List[torch.Tensor],
        neg_scores: List[torch.Tensor],
        perf_list: List[float] = None,
    ) -> None:
        """Aggregate and log the evaluation performance."""
        pos_scores = torch.cat(pos_scores, dim=0)
        neg_scores = torch.cat(neg_scores, dim=0)
        scores = torch.cat((pos_scores, neg_scores), dim=0)
        labels = torch.cat((torch.ones_like(pos_scores), torch.zeros_like(neg_scores)), dim=0)
        if self.fast_eval:
            ap_log_name = f"{stage}/AP_fast"
            auc_log_name = f"{stage}/AUC_fast"
            metric_log_name = f"{stage}/{self.metric}_fast"
        else:
            ap_log_name = f"{stage}/AP"
            auc_log_name = f"{stage}/AUC"
            metric_log_name = f"{stage}/{self.metric}"
        self.log(
            ap_log_name,
            average_precision_score(y_true=labels.cpu().numpy(), y_score=scores.cpu().numpy()),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            auc_log_name,
            roc_auc_score(y_true=labels.cpu().numpy(), y_score=scores.cpu().numpy()),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if perf_list is not None:
            self.log(
                metric_log_name,
                np.mean(perf_list),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure and return optimizer and scheduler."""
        raise NotImplementedError

    def _subsample_neg_edges(
        self, neg_dst_node_ids_list: List[List[int]], portion: Union[int, float]
    ) -> List[List[int]]:
        """For each list of nodes in neg_dst_node_ids_list, randomly subsample a fixed portion of
        them and return the subsampled neg_dst_node_ids_list."""
        if isinstance(portion, float):
            portion = int(len(neg_dst_node_ids_list[0]) * portion)
        return [
            random.sample(neg_dst_node_ids, portion) for neg_dst_node_ids in neg_dst_node_ids_list
        ]  # randomly sample nodes without replacement
