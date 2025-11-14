import pytorch_lightning as pl
import torch.nn as nn
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import torch.nn.functional as F
from .model.SpatialTransformerNetworkWmaskWaffine import SpatialTransformerNetworkWmaskWaffine

class STNModuleWmaskWaffine(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, optim_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters({"model_cfg": model_cfg, "optim_cfg": optim_cfg})
        self.model = SpatialTransformerNetworkWmaskWaffine(**model_cfg.params)
        self.criterion = nn.MSELoss()
        self.optim_cfg = optim_cfg

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx): # train,valの共通のloss計算
        src_img, tgt_img, src_mask, tgt_mask = batch['source'], batch['target'], batch['mask'], batch['ref_mask']
        output = self(src_img)
        
        comp = output[:, :3]*output[:, 3:] + src_img[:, :3]*(1 - output[:, 3:])
        # マスク画像

        loss_img = self.criterion(output[:, :3]*output[:, 3:], tgt_img*tgt_mask)
        
        loss_mask = self.criterion(output[:, 3:], tgt_mask)
        
        loss = loss_mask + loss_img

        if batch_idx == 0:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.log(f"model/grad_{name}", param.grad.mean(), sync_dist=True)
            rgb_1 = src_img[:4, :3]
            mask_1 = src_mask[:4].repeat(1,3,1,1)
            rgb_2 = tgt_img[:4]
            mask_2 = tgt_mask[:4].repeat(1,3,1,1)
            rgb_3 = comp[:4]
            mask_3 = output[:4, 3:].repeat(1,3,1,1)
            grid = torch.cat([rgb_1, mask_1, rgb_2, mask_2, rgb_3, mask_3], dim=2)

            self_type = "train" if self.training else "val"
            self.logger.experiment.add_images(f"{self_type}/comparison", grid, self.current_epoch)
            
        return loss_img, loss_mask, loss

    def training_step(self, batch, batch_idx):
        loss_img, loss_mask, loss = self.step(batch, batch_idx)
        self.log("train/loss", loss, sync_dist=True)
        self.log("train/loss_img", loss_img, sync_dist=True)
        self.log("train/loss_mask", loss_mask, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss_img, loss_mask, loss = self.step(batch, batch_idx)
        self.log("val/loss", loss, sync_dist=True)
        self.log("val/loss_img", loss_img, sync_dist=True)
        self.log("val/loss_mask", loss_mask, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.optim_cfg, params=self.parameters())
        return optimizer

    