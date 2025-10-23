import pytorch_lightning as pl
import torch.nn as nn
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from STN.SpatialTransformerNetwork import SpatialTransformerNetwork

class STNModule(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, optim_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model = SpatialTransformerNetwork(**model_cfg.params)
        self.criterion = nn.MSELoss()
        self.optim_cfg = optim_cfg

    def forward(self, x):
        return self.model(x)
    
    def step(self, batch, batch_idx): # train,valの共通のloss計算
        src_img, tgt_img = batch
        output = self(src_img)
        loss = self.criterion(output, tgt_img)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train/loss", loss)
        
        # 画像の可視化
        if batch_idx == 0:
            src_img, tgt_img = batch
            output = self(src_img)
            grid = torch.cat([src_img, output, tgt_img], dim=3)  # 横に結合
            self.logger.experiment.add_images("train/comparison", grid, self.current_epoch)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val/loss", loss)
        # 画像の可視化
        if batch_idx == 0:
            src_img, tgt_img = batch
            output = self(src_img)
            grid = torch.cat([src_img, output, tgt_img], dim=3)  # 横に結合
            self.logger.experiment.add_images("val/comparison", grid, self.current_epoch)
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.optim_cfg, params=self.parameters())
        return optimizer
    
    