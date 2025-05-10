from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.utils.logging_utils import log_hyperparameters
from src.utils.pylogger import RankedLogger
from src.utils.train_utils import get_metric_value, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    print("callbacks", callbacks)
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    def get_ckpt_path(trainer: Trainer, monitor_metric: str) -> Optional[str]:
        """Get the best ckpt path under the given monitor metric. If no ckpt is recorded based on
        the monitor metric, return the first ckpt path.

        Args:
            trainer: The trainer object.
            monitor_metric: The metric to monitor.
        """
        checkpoint_callbacks = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)]
        if len(checkpoint_callbacks) == 0:
            return None
        for callback in checkpoint_callbacks:
            if callback.monitor == monitor_metric:
                log.info(f"Best ckpt under {monitor_metric} found!")
                return callback.best_model_path
        log.info(
            f"No best ckpt under the given monitor metric found! Using {checkpoint_callbacks[0].monitor} instead"
        )
        return checkpoint_callbacks[0].best_model_path

    model_name = cfg.model._target_.split(".")[-2]
    is_memory_based = True if model_name in ["tgn"] else False
    log.info(f"is_memory_based: {is_memory_based} model_name: {model_name}")
    if cfg.get("val"):
        log.info("Starting validation!")

        if (
            "NonTGBLDataModule" in cfg.data._target_ and not is_memory_based
        ):  # directly use past recorded scores of the best ckpt
            checkpoint_callbacks = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)]
            best_metrics = {}

            for strategy in ["random", "historical", "inductive"]:
                for callback in checkpoint_callbacks:
                    if callback.monitor == f"val/{strategy}/ap":
                        best_metrics[f"val/{strategy}/ap_final"] = callback.best_model_score
                        log.info(f"Best val/{strategy}/ap: {callback.best_model_score}")
                        break

            # Log best metrics to wandb
            if logger:
                for metric_name, value in best_metrics.items():
                    logger[0].log_metrics({metric_name: value})
        else:  # for memory-based method, we validate the model again to refresh validation memory with the best ckpt
            ckpt_path = get_ckpt_path(trainer, "val/random/ap")
            # if model_name == "tgn":
            #     print(model.model[0].memory_bank.node_last_updated_times)
            val_results = trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            # if model_name == "tgn":
            #     print(model.model[0].memory_bank.node_last_updated_times)

    val_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")

        if "NonTGBLDataModule" in cfg.data._target_:
            # if model is memory-based, we need to backup and reload the memory up to validation manually because we repeatedly test below
            datamodule.test_negative_sample_strategy = ["random"]
            if is_memory_based:
                model.backup_val_memory_bank()
            trainer: Trainer = hydra.utils.instantiate(
                cfg.trainer, callbacks=callbacks, logger=logger
            )
            log.info("Getting best ckpt under val/random/ap...")
            ckpt_path = get_ckpt_path(trainer, "val/random/ap")
            if ckpt_path is None:
                log.warning("Best ckpt not found! Using current weights for random NS testing...")
            # if model_name == "tgn":
            #     print(model.model[0].memory_bank.node_last_updated_times)
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            # if model_name == "tgn":
            #     print(model.model[0].memory_bank.node_last_updated_times)
            # logger[0].log_metrics({'test/random/ap_final': random_test_results[0]["test/random/ap_final"]})

            if is_memory_based:
                model.reload_val_memory_bank()
            datamodule.test_negative_sample_strategy = ["historical"]
            trainer: Trainer = hydra.utils.instantiate(
                cfg.trainer, callbacks=callbacks, logger=logger
            )
            log.info("Getting best ckpt under val/historical/ap...")
            ckpt_path = get_ckpt_path(trainer, "val/historical/ap")
            if ckpt_path is None:
                log.warning(
                    "Best ckpt not found! Using current weights for historical NS testing..."
                )
            # if model_name == "tgn":
            #     print(model.model[0].memory_bank.node_last_updated_times)
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            # if model_name == "tgn":
            #     print(model.model[0].memory_bank.node_last_updated_times)

            if is_memory_based:
                model.reload_val_memory_bank()
            datamodule.test_negative_sample_strategy = ["inductive"]
            trainer: Trainer = hydra.utils.instantiate(
                cfg.trainer, callbacks=callbacks, logger=logger
            )
            log.info("Getting best ckpt under val/inductive/ap...")
            ckpt_path = get_ckpt_path(trainer, "val/inductive/ap")
            if ckpt_path is None:
                log.warning(
                    "Best ckpt not found! Using current weights for inductive NS testing..."
                )
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        else:
            ckpt_path = get_ckpt_path(trainer, "val/random/ap")
            test_results = trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics
    # merge train and test metrics
    metric_dict = {**train_metrics, **val_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)
    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
