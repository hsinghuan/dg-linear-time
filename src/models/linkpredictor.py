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

    def __init__(
        self, sample_neighbor_strategy: str = "recent", time_scaling_factor: float = 1e-6
    ):
        """Initialize the link predictor parent class."""
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.loss_func = nn.BCELoss()

    def setup(self, stage: str) -> None:
        """Get data, samplers, metric and evaluator before fit (train + validate), validate, and
        test."""

        self.fit = stage == "fit"
        self.dataset_type = self.trainer.datamodule.dataset_type
        self.node_raw_features = self.trainer.datamodule.node_raw_features
        self.edge_raw_features = self.trainer.datamodule.edge_raw_features
        self.full_data = self.trainer.datamodule.full_data
        self.train_data = self.trainer.datamodule.train_data
        self.val_data = self.trainer.datamodule.val_data
        self.test_data = self.trainer.datamodule.test_data
        self.train_neighbor_sampler = get_neighbor_sampler(
            data=self.train_data,
            sample_neighbor_strategy=self.hparams.sample_neighbor_strategy,
            time_scaling_factor=self.hparams.time_scaling_factor,
            seed=0,
        )
        self.full_neighbor_sampler = get_neighbor_sampler(
            data=self.full_data,
            sample_neighbor_strategy=self.hparams.sample_neighbor_strategy,
            time_scaling_factor=self.hparams.time_scaling_factor,
            seed=1,
        )
        self.train_neg_edge_sampler = NegativeEdgeSampler(
            src_node_ids=self.train_data.src_node_ids, dst_node_ids=self.train_data.dst_node_ids
        )

        self.train_pos_scores = []
        self.train_neg_scores = []
        self.val_pos_scores = []
        self.val_neg_scores = []
        self.test_pos_scores = []
        self.test_neg_scores = []

        if self.dataset_type == "tgbl":
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
            self.val_perf_list = []
            self.test_perf_list = []
        else:
            self.negative_sample_strategy = self.trainer.datamodule.negative_sample_strategy
            if self.negative_sample_strategy != "random":
                self.val_neg_edge_sampler = NegativeEdgeSampler(
                    src_node_ids=self.full_data.src_node_ids,
                    dst_node_ids=self.full_data.dst_node_ids,
                    interact_times=self.full_data.node_interact_times,
                    last_observed_time=self.train_data.node_interact_times[-1],
                    negative_sample_strategy=self.negative_sample_strategy,
                    seed=0,
                )
                self.test_neg_edge_sampler = NegativeEdgeSampler(
                    src_node_ids=self.full_data.src_node_ids,
                    dst_node_ids=self.full_data.dst_node_ids,
                    interact_times=self.full_data.node_interact_times,
                    last_observed_time=self.val_data.node_interact_times[-1],
                    negative_sample_strategy=self.negative_sample_strategy,
                    seed=2,
                )
            else:
                self.val_neg_edge_sampler = NegativeEdgeSampler(
                    src_node_ids=self.full_data.src_node_ids,
                    dst_node_ids=self.full_data.dst_node_ids,
                    seed=0,
                )
                self.test_neg_edge_sampler = NegativeEdgeSampler(
                    src_node_ids=self.full_data.src_node_ids,
                    dst_node_ids=self.full_data.dst_node_ids,
                    seed=2,
                )
            self.val_perf_list = None  # no mrr to be recorded
            self.test_perf_list = None
            self.fast_eval = False  # no need for fast evaluation (because it's already fast)

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

        # forward negative edges first because they do not change the memories of memory-based model
        pos_scores, neg_scores = self._pred_pos_neg_scores(
            pos_src=batch_src_node_ids,
            pos_dst=batch_dst_node_ids,
            pos_t=batch_node_interact_times,
            neg_src=batch_neg_src_node_ids,
            neg_dst=batch_neg_dst_node_ids,
            neg_t=batch_neg_node_interact_times,
            edge_ids=batch_edge_ids,
        )

        scores = torch.cat((pos_scores, neg_scores), dim=0)
        labels = torch.cat((torch.ones_like(pos_scores), torch.zeros_like(neg_scores)), dim=0)
        loss = self.loss_func(input=scores, target=labels)
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if stage == "val":
            self.val_pos_scores.append(pos_scores)
            self.val_neg_scores.append(neg_scores)
        elif stage == "test":
            self.test_pos_scores.append(pos_scores)
            self.test_neg_scores.append(neg_scores)

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
    ):
        """Predict the probabilities/scores of (pos_src[i], pos_dst[i]) happening at time pos_t[i]
        and (neg_src[i], neg_dst[i]) happening at time neg_t[i]."""
        raise NotImplementedError

    def _pred_scores(
        self, src: np.ndarray, dst: np.ndarray, t: np.ndarray, **kwargs
    ) -> torch.Tensor:
        """Predict the probabilities/scores of (src[i], dst[i]) happening at time t[i]."""
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
        self.val_perf_list = [] if self.dataset_type == "tgbl" else None

    def on_test_epoch_end(self) -> None:
        """Aggregate testing performance."""
        self._aggregate_eval_log(
            "test", self.test_pos_scores, self.test_neg_scores, self.test_perf_list
        )
        self.test_pos_scores = []
        self.test_neg_scores = []
        self.test_perf_list = [] if self.dataset_type == "tgbl" else None

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

        if self.dataset_type == "tgbl":
            if self.fit:
                progress_bar = True
                if self.fast_eval:
                    ap_log_name = f"{stage}/ap_fast"
                    auc_log_name = f"{stage}/auc_fast"
                    metric_log_name = f"{stage}/{self.metric}_fast"
                else:
                    ap_log_name = f"{stage}/ap"
                    auc_log_name = f"{stage}/auc"
                    metric_log_name = f"{stage}/{self.metric}"
            else:
                progress_bar = False
                ap_log_name = f"{stage}/ap_final"
                auc_log_name = f"{stage}/auc_final"
                metric_log_name = f"{stage}/{self.metric}_final"
        else:
            if self.fit:
                progress_bar = True
                ap_log_name = f"{stage}/{self.negative_sample_strategy}/ap"
                auc_log_name = f"{stage}/{self.negative_sample_strategy}/auc"
            else:
                progress_bar = False
                ap_log_name = f"{stage}/{self.negative_sample_strategy}/ap_final"
                auc_log_name = f"{stage}/{self.negative_sample_strategy}/auc_final"

        # if self.fit:
        #     if self.fast_eval:
        #         ap_log_name = f"{stage}/ap_fast"
        #         auc_log_name = f"{stage}/auc_fast"
        #         metric_log_name = f"{stage}/{self.metric}_fast"  # must be tgbl dataset
        #     else:
        #         ap_log_name = f"{stage}/ap"
        #         auc_log_name = f"{stage}/auc"
        #         if self.dataset_type == "tgbl":
        #             metric_log_name = f"{stage}/{self.metric}"
        # else:
        #     ap_log_name = f"{stage}/ap_final"
        #     auc_log_name = f"{stage}/auc_final"
        #     if self.dataset_type == "tgbl":
        #         metric_log_name = f"{stage}/{self.metric}_final"

        self.log(
            ap_log_name,
            average_precision_score(y_true=labels.cpu().numpy(), y_score=scores.cpu().numpy()),
            on_step=False,
            on_epoch=True,
            prog_bar=progress_bar,
        )
        self.log(
            auc_log_name,
            roc_auc_score(y_true=labels.cpu().numpy(), y_score=scores.cpu().numpy()),
            on_step=False,
            on_epoch=True,
            prog_bar=progress_bar,
        )

        if perf_list is not None:
            self.log(
                metric_log_name,
                np.mean(perf_list),
                on_step=False,
                on_epoch=True,
                prog_bar=progress_bar,
            )

        # finer grained aggregation (aggregate to 4 bins) at trainer.validation() or trainer.test() stages
        if not self.fit:
            pos_scores_num = len(pos_scores)
            neg_scores_num = len(neg_scores)
            for i in range(4):
                if i != 3:
                    pos_scores_per_quarter = pos_scores[
                        i * pos_scores_num // 4 : (i + 1) * pos_scores_num // 4
                    ]
                    neg_scores_per_quarter = neg_scores[
                        i * neg_scores_num // 4 : (i + 1) * neg_scores_num // 4
                    ]
                else:
                    pos_scores_per_quarter = pos_scores[i * pos_scores_num // 4 :]
                    neg_scores_per_quarter = neg_scores[i * neg_scores_num // 4 :]
                scores_per_quarter = torch.cat(
                    (pos_scores_per_quarter, neg_scores_per_quarter), dim=0
                )
                labels_per_quarter = torch.cat(
                    (
                        torch.ones_like(pos_scores_per_quarter),
                        torch.zeros_like(neg_scores_per_quarter),
                    ),
                    dim=0,
                )
                self.log(
                    ap_log_name + f"_quarter_{i+1}",
                    average_precision_score(
                        y_true=labels_per_quarter.cpu().numpy(),
                        y_score=scores_per_quarter.cpu().numpy(),
                    ),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=progress_bar,
                )
                self.log(
                    auc_log_name + f"_quarter_{i+1}",
                    roc_auc_score(
                        y_true=labels_per_quarter.cpu().numpy(),
                        y_score=scores_per_quarter.cpu().numpy(),
                    ),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=progress_bar,
                )

            if perf_list is not None:
                perf_list_num = len(perf_list)
                for i in range(4):
                    if i != 3:
                        perf_list_per_quarter = perf_list[
                            i * perf_list_num // 4 : (i + 1) * perf_list_num // 4
                        ]
                    else:
                        perf_list_per_quarter = perf_list[i * perf_list_num // 4 :]
                    self.log(
                        metric_log_name + f"_quarter_{i+1}",
                        np.mean(perf_list_per_quarter),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=progress_bar,
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
