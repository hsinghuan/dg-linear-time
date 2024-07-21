import numpy as np
import torch
from lightning import LightningModule
from tgb.linkproppred.evaluate import Evaluator

from src.data.tgbl_datamodule import Data
from src.models.modules.edgebank_predictor import EdgeBankPredictor


class EdgeBankModule(LightningModule):
    """LightningModule for the EdgeBank dynamic link prediction heuristic."""

    def __init__(
        self,
        memory_mode: str = "unlimited",  # could be `unlimited` or `fixed_time_window`
        time_window_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def on_validation_start(self) -> None:
        """Prepare data, negative sampler, evaluator, and edgebank for validation."""
        self.train_data = self.trainer.datamodule.train_data
        self.val_data = self.trainer.datamodule.val_data
        self.test_data = self.trainer.datamodule.test_data
        self.eval_neg_edge_sampler = self.trainer.datamodule.eval_neg_edge_sampler
        self.metric = self.trainer.datamodule.eval_metric_name
        self.evaluator = Evaluator(self.trainer.datamodule.dataset_name)
        self.edgebank = EdgeBankPredictor(
            self.train_data.src_node_ids,
            self.train_data.dst_node_ids,
            self.train_data.node_interact_times,
            memory_mode=self.hparams.memory_mode,
            time_window_ratio=self.hparams.time_window_ratio,
        )
        self.val_perf_list = []

    def on_test_start(self) -> None:
        """Prepare data, negative sampler, evaluator, and edgebank for testing."""
        self.train_data = self.trainer.datamodule.train_data
        self.val_data = self.trainer.datamodule.val_data
        self.test_data = self.trainer.datamodule.test_data
        self.eval_neg_edge_sampler = self.trainer.datamodule.eval_neg_edge_sampler
        self.metric = self.trainer.datamodule.eval_metric_name
        self.evaluator = Evaluator(self.trainer.datamodule.dataset_name)
        self.edgebank = EdgeBankPredictor(
            np.concatenate([self.train_data.src_node_ids, self.val_data.src_node_ids]),
            np.concatenate([self.train_data.dst_node_ids, self.val_data.dst_node_ids]),
            np.concatenate(
                [self.train_data.node_interact_times, self.val_data.node_interact_times]
            ),
            memory_mode=self.hparams.memory_mode,
            time_window_ratio=self.hparams.time_window_ratio,
        )
        self.test_perf_list = []

    def validation_step(self, batch: torch.Tensor) -> None:
        """One batch of validation."""
        self._forward(batch, self.val_data, "val")

    def test_step(self, batch: torch.Tensor) -> None:
        """One batch of testing."""
        self._forward(batch, self.test_data, "test")

    def _forward(self, batch: torch.Tensor, data: Data, stage: str) -> None:
        """Predict and evaluate per val/test batch, forward each sample and its negatives at a
        time."""
        data_indices = batch.numpy()
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

        for idx, neg_batch in enumerate(batch_neg_dst_node_ids_list):
            query_src = np.array([int(batch_src_node_ids[idx]) for _ in range(len(neg_batch) + 1)])
            query_dst = np.concatenate(
                [np.array([int(batch_dst_node_ids[idx])]), np.array(neg_batch) + 1]
            )  # add 1 to convert to the node ids into dyglib format
            y_pred = self.edgebank.predict_link(query_src, query_dst)
            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0]]),
                "y_pred_neg": np.array(y_pred[1:]),
                "eval_metric": [self.metric],
            }
            if stage == "val":
                self.val_perf_list.append(self.evaluator.eval(input_dict)[self.metric])
            elif stage == "test":
                self.test_perf_list.append(self.evaluator.eval(input_dict)[self.metric])
            else:
                raise ValueError(f"Invalid stage: {stage}")

        # update edgebank memory after each positive batch
        self.edgebank.update_memory(
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times
        )

    def on_validation_epoch_end(self) -> None:
        """Aggregate validation performance."""
        self._aggregate_eval_log(self.val_perf_list, "val")
        self.val_perf_list = []

    def on_test_epoch_end(self) -> None:
        """Aggregate testing performance."""
        self._aggregate_eval_log(self.test_perf_list, "test")
        self.test_perf_list = []

    def _aggregate_eval_log(self, perf_list: list, stage: str) -> None:
        """Aggregate and log the evaluation performance."""
        self.log(
            f"{stage}/{self.metric}",
            np.mean(perf_list),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
