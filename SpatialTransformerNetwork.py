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
    def __init__(self, num_network=4, 
                 input_channels=1, hidden_layers=[16, 32], 
                 kernel_size=[3, 3], stride=[2, 2], padding=[1, 1],
                 fc_hidden_size=32, fc_output_size=8):
        super().__init__()

        self.encoder_list = nn.ModuleList()
        self.fc_loc_list = nn.ModuleList()

        for _ in range(num_network):
            # Localization network パラメータを求めるためのエンコーダ
            encoder = nn.Sequential()
            for i in range(len(hidden_layers)):
                in_channels = input_channels if i == 0 else hidden_layers[i - 1]
                out_channels = hidden_layers[i]
                if stride[i] > 0:
                    encoder.add_module(
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size[i], stride=stride[i], padding=padding[i])
                    )
                else:
                    encoder.add_module(
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size[i], padding=padding[i])
                    )
                encoder.add_module(f'act{i+1}', nn.SiLU(True))
            encoder.add_module(nn.AdaptiveAvgPool2d((1, 1)))
            self.encoder_list.append(encoder)

            # homography変換のパラメータ（8自由度：H[2,2]=1に固定）を出力するための全結合層
            fc_loc = nn.Sequential(
                nn.Linear(hidden_layers[-1], fc_output_size)
            )

            # 残差パラメータ化のため、最終層をゼロ初期化（出力=0 -> H=I）
            fc_loc[2].weight.data.zero_()
            fc_loc[2].bias.data.zero_()
            
            self.fc_loc_list.append(fc_loc)

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

        x = K.geometry.transform.warp_perspective(x, theta, (x.size(3), x.size(2)))

        return x
    
import yaml
if __name__ == "__main__":
    # stn = SpatialTransformerNetwork()
    # # stn = torch.compile(stn)
    # input_dummy = torch.randn(1, 1, 28, 28)
    # writer = SummaryWriter(comment=f'STN_tutorial')
    # writer.add_graph(stn, input_dummy)
    # writer.close()
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_params = config["model"]["params"]
    model = SpatialTransformerNetwork(**model_params)