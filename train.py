from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
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
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # object_dict = {
    #     "cfg": cfg,
    #     "datamodule": datamodule,
    #     "model": model,
    #     "callbacks": callbacks,
    #     "logger": logger,
    #     "trainer": trainer,
    # }

    # if logger:
    #     log.info("Logging hyperparameters!")
    #     log_hyperparameters(object_dict)

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

    if cfg.get("val"):
        log.info("Starting validation!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        if "NonTGBLDataModule" in cfg.data._target_:
            # if model is memory-based, we need to backup and reload the memory up to train manually because we repeatedly test below
            is_memory_based = True if cfg.model in ["tgn", "dyrep", "jodie"] else False
            datamodule.negative_sample_strategy = "random"
            if is_memory_based:
                model.backup_train_memory_bank()
            trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            if is_memory_based:
                model.reload_train_memory_bank()
            datamodule.negative_sample_strategy = "historical"
            trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            if is_memory_based:
                model.reload_train_memory_bank()
            datamodule.negative_sample_strategy = "inductive"
            trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        else:
            trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    val_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        if "NonTGBLDataModule" in cfg.data._target_:
            # if model is memory-based, we need to backup and reload the memory up to validation manually because we repeatedly test below
            is_memory_based = True if cfg.model in ["tgn", "dyrep", "jodie"] else False
            datamodule.negative_sample_strategy = "random"
            if is_memory_based:
                model.backup_val_memory_bank()
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            if is_memory_based:
                model.reload_val_memory_bank()
            datamodule.negative_sample_strategy = "historical"
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            if is_memory_based:
                model.reload_val_memory_bank()
            datamodule.negative_sample_strategy = "inductive"
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        else:
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
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
