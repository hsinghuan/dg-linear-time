from typing import Any, Dict, List, Tuple

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils.instantiators import instantiate_loggers
from src.utils.logging_utils import log_hyperparameters
from src.utils.pylogger import RankedLogger
from src.utils.train_utils import task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    print("cfg", cfg)
    if cfg.model._target_ != "src.models.edgebank.EdgeBankModule":
        assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting validation!")
    # trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    if "NonTGBLDataModule" in cfg.data._target_:
        # if model is memory-based, we need to backup and reload the memory up to train manually because we repeatedly test below
        is_memory_based = True if cfg.model in ["tgn", "dyrep", "jodie"] else False
        datamodule.val_negative_sample_strategy = ["random"]
        if is_memory_based:
            model.backup_train_memory_bank()
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
        if is_memory_based:
            model.reload_train_memory_bank()
        datamodule.val_negative_sample_strategy = ["historical"]
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
        if is_memory_based:
            model.reload_train_memory_bank()
        datamodule.val_negative_sample_strategy = ["inductive"]
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    else:
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    log.info("Starting testing!")
    # trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    if "NonTGBLDataModule" in cfg.data._target_:
        # if model is memory-based, we need to backup and reload the memory up to validation manually because we repeatedly test below
        is_memory_based = True if cfg.model in ["tgn", "dyrep", "jodie"] else False
        datamodule.test_negative_sample_strategy = ["random"]
        if is_memory_based:
            model.backup_val_memory_bank()
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
        if is_memory_based:
            model.reload_val_memory_bank()
        datamodule.test_negative_sample_strategy = ["historical"]
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
        if is_memory_based:
            model.reload_val_memory_bank()
        datamodule.test_negative_sample_strategy = ["inductive"]
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    else:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
