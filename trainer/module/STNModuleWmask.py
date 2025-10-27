import pytorch_lightning as pl
import torch.nn as nn
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from .model.SpatialTransformerNetworkWmask import SpatialTransformerNetworkWmask

class STNModuleWmask(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, optim_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters({"model_cfg": model_cfg, "optim_cfg": optim_cfg})
        self.model = SpatialTransformerNetworkWmask(**model_cfg.params)
        self.criterion = nn.MSELoss()
        self.optim_cfg = optim_cfg

    def forward(self, x):
        return self.model(x)
    
    def step(self, batch, batch_idx): # train,valの共通のloss計算
        src_img, tgt_img, mask, tgt_mask = batch['source'], batch['target'], batch['mask'], batch['ref_mask']
        output, output_mask = self(src_img, mask)
        output = src_img * (1 - mask) + output_mask * mask
        loss = self.criterion(output, tgt_img)
        
        if batch_idx == 0:
            if src_img.dim() > 4:
                src_img = src_img[:4]
                tgt_img = tgt_img[:4]
                output = output[:4]
                output_mask = output_mask[:4]
                mask = mask[:4]
                tgt_mask = tgt_mask[:4]
            grid = torch.cat([src_img, mask, tgt_img, tgt_mask, output, output_mask], dim=3)  # 横に結合

            self_type = "train" if self.training else "val"
            self.logger.experiment.add_images(f"{self_type}/comparison", grid, self.current_epoch)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.optim_cfg, params=self.parameters())
        return optimizer
    
    