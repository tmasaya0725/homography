import pytorch_lightning as pl
import torch.nn as nn
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from .model.SpatialTransformerNetworkWmaskWpix import SpatialTransformerNetworkWmaskWpix

class STNModuleWmaskWpix(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, optim_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters({"model_cfg": model_cfg, "optim_cfg": optim_cfg})
        self.model = SpatialTransformerNetworkWmaskWpix(**model_cfg.params)
        self.criterion = nn.MSELoss()
        self.optim_cfg = optim_cfg

    def forward(self, x, mask):
        return self.model(x, mask)

    def step(self, batch, batch_idx): # train,valの共通のloss計算
        src_img, tgt_img, mask, tgt_mask = batch['source'], batch['target'], batch['mask'], batch['ref_mask']
        output, output_mask = self(src_img, mask)

        loss_img = self.criterion(output[:, :3], tgt_img)
        loss_mask = self.criterion(output[:, 3:], tgt_mask) * 10
        loss = loss_img + loss_mask

        if batch_idx == 0:
            src_img = src_img[:4, :3]
            mask = mask[:4].repeat(1, 3, 1, 1)
            tgt_img = tgt_img[:4]
            tgt_mask = tgt_mask[:4].repeat(1, 3, 1, 1)
            output = output[:4, :3]
            output_mask = output_mask[:4].repeat(1, 3, 1, 1)
            grid = torch.cat([src_img, mask, tgt_img, tgt_mask, output, output_mask], dim=2)

            self_type = "train" if self.training else "val"
            self.logger.experiment.add_images(f"{self_type}/comparison", grid, self.current_epoch)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train/loss", loss, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val/loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.optim_cfg, params=self.parameters())
        return optimizer

    