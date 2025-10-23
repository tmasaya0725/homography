import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger

from util.util import get_model, get_data
import argparse
import torch

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.get("seed", 42), workers=True)

    model = get_model(cfg.model, cfg.optim)
    data = get_data(cfg.data)

    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.get("callbacks", {}).values()]
    logger_cfg = cfg.get("logger")
    logger: Logger | bool = hydra.utils.instantiate(logger_cfg) if logger_cfg else True

    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule=data)
        
if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    
    main()
