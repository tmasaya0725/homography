import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import kornia as K
import numpy as np
# import warnings
# warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

@torch._dynamo.disable()
def safe_warp_perspective(img, H, dsize):
    return K.geometry.transform.warp_perspective(img, H, dsize)

class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Localization network パラメータを求めるためのエンコーダ
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # homography変換のパラメータ（8自由度：H[2,2]=1に固定）を出力するための全結合層
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 8)
        )
        
        # 残差パラメータ化のため、最終層をゼロ初期化（出力=0 -> H=I）
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.zero_()

        # 変換の安全範囲（tanh後にスケーリング）
        # 対角・オフ対角の線形項の最大残差量
        self.scale_max: float = 2.0   # diag/off-diag に対する残差のスケール（-2..2）。I+(-2)= -1 で反転が到達可能
        # 透視成分（下段[2,0],[2,1]）の最大残差
        self.persp_max: float = 0.5   # 過度な遠近を抑制
        # 並進は画像サイズに依存してスケール（-W..W, -H..H が到達可能）
        self.trans_scale: float = 1.0
        
    def forward(self, x):
        B, C, H_img, W_img = x.shape

        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)

        # 8パラメータを出力（tanhでクリップ）
        raw = self.fc_loc(xs)                 # (B, 8)
        delta = torch.tanh(raw)               # (B, 8) in (-1, 1)

        # 残差パラメータをスケーリング
        a = delta[:, 0] * self.scale_max      # H[0,0] 残差
        b = delta[:, 1] * self.scale_max      # H[0,1] 残差
        c = delta[:, 2] * self.scale_max      # H[1,0] 残差
        d = delta[:, 3] * self.scale_max      # H[1,1] 残差
        tx = delta[:, 4] * (W_img * self.trans_scale)  # H[0,2] 残差（ピクセル単位）
        ty = delta[:, 5] * (H_img * self.trans_scale)  # H[1,2] 残差（ピクセル単位）
        u = delta[:, 6] * self.persp_max      # H[2,0] 残差
        v = delta[:, 7] * self.persp_max      # H[2,1] 残差

        # 恒等行列に残差を加算して H を構築（H[2,2]=1 を保持）
        I = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
        I[:, 0, 0] = I[:, 0, 0] + a
        I[:, 0, 1] = I[:, 0, 1] + b
        I[:, 1, 0] = I[:, 1, 0] + c
        I[:, 1, 1] = I[:, 1, 1] + d
        I[:, 0, 2] = I[:, 0, 2] + tx
        I[:, 1, 2] = I[:, 1, 2] + ty
        I[:, 2, 0] = I[:, 2, 0] + u
        I[:, 2, 1] = I[:, 2, 1] + v

        theta = I

        x = safe_warp_perspective(x, theta, (x.size(3), x.size(2)))

        return x
    

if __name__ == "__main__":
    stn = SpatialTransformerNetwork()
    # stn = torch.compile(stn)
    input_dummy = torch.randn(1, 1, 28, 28)
    writer = SummaryWriter(comment=f'STN_tutorial')
    writer.add_graph(stn, input_dummy)
    writer.close()
    