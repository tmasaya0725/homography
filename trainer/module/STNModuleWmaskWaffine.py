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
        
        # マスクの重心の距離誤差
        # x1, y1 = self.get_object_sentric_coordinates(tgt_mask)
        # x2, y2 = self.get_object_sentric_coordinates(output[:, 3:])
        
        # _, _, H, W = tgt_mask.shape
        # gt_coords = torch.stack([x1 / W, y1 / H], dim=1)
        # pred_coords = torch.stack([x2 / W, y2 / H], dim=1)
        
        # loss_coord = F.l1_loss(pred_coords, gt_coords)

        loss_img = self.criterion(output[:, :3]*output[:, 3:], tgt_img*tgt_mask)
        
        loss_mask = self.criterion(output[:, 3:], tgt_mask)
        
        loss = loss_img + loss_mask # + loss_coord

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
    
    def get_object_sentric_coordinates(self, mask):
        """
        マスク画像から物体中心（重心）の座標を求める。

        引数:
            mask: 形状 (B, 1, H, W) のテンソル（0/1 あるいは [0,1] の連続値想定）

        戻り値:
            x_coords: 形状 (B,) のテンソル。左上原点 (x: 0..W-1) の重心 x 座標
            y_coords: 形状 (B,) のテンソル。左上原点 (y: 0..H-1) の重心 y 座標

        備考:
            - マスクが全て 0（物体なし）の場合は、画像中心 ((W-1)/2, (H-1)/2) を返す。
            - 計算は完全にバッチ化・微分可能な形で実装。
        """
        # 入力は (B, 1, H, W) を想定
        if mask.dim() != 4:
            raise ValueError(f"mask の次元が不正です: {mask.shape} (期待: (B,1,H,W))")

        B, C, H, W = mask.shape
        if C != 1:
            raise ValueError(f"mask のチャンネル数が不正です: {C} (期待: 1)")

        # float に変換
        mask = mask.to(dtype=torch.float32)
        device = mask.device

        # 画素座標グリッド（左上原点）
        ys = torch.arange(H, device=device, dtype=mask.dtype).view(1, 1, H, 1)
        xs = torch.arange(W, device=device, dtype=mask.dtype).view(1, 1, 1, W)

        # 重み総和（各バッチ）
        denom = mask.sum(dim=(2, 3), keepdim=True)  # (B,1,1,1) でも良いが後続合わせて (B,1) にする
        denom = denom.view(B, 1)

        # 分子
        num_x = (mask * xs).sum(dim=(2, 3)).view(B, 1)  # (B,1)
        num_y = (mask * ys).sum(dim=(2, 3)).view(B, 1)  # (B,1)

        # 0 割回避のためのセーフ分母
        denom_safe = denom.clone()
        denom_safe[denom_safe == 0] = 1.0

        x_tmp = num_x / denom_safe  # (B,1)
        y_tmp = num_y / denom_safe  # (B,1)

        # マスクが空（全 0）の場合は画像中心にフォールバック
        center_x = torch.full((B, 1), (W - 1) / 2.0, device=device, dtype=mask.dtype)
        center_y = torch.full((B, 1), (H - 1) / 2.0, device=device, dtype=mask.dtype)
        valid = (denom > 0)

        x_coords = torch.where(valid, x_tmp, center_x).view(B)
        y_coords = torch.where(valid, y_tmp, center_y).view(B)

        return x_coords, y_coords  

    