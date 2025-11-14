import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import kornia as K
import numpy as np
import math

class SpatialTransformerNetworkWmaskWaffine(nn.Module):
    def __init__(self, num_network=4, 
                 input_channels=1, hidden_layers=[16, 32], 
                 kernel_size=[3, 3], stride=[2, 2], padding=[1, 1],
                 fc_hidden_size=[32], fc_output_size=4,
                 scale_max=2.0, persp_max=0.5, trans_scale=1.0):
        super().__init__()

        self.encoder_list = nn.ModuleList()
        self.fc_loc_list = nn.ModuleList()
        
        self.persp_max = persp_max

        for i in range(num_network):
            # Localization network パラメータを求めるためのエンコーダ
            encoder = nn.Sequential()
            for j in range(len(hidden_layers)):
                in_channels = input_channels if j == 0 else hidden_layers[j - 1]
                out_channels = hidden_layers[j]
                if stride[j] > 0:
                    encoder.add_module(f"enc{i+1}_conv{j+1}",
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size[j], stride=stride[j], padding=padding[j])
                    )
                else:
                    encoder.add_module(f"enc{i+1}_conv{j+1}",
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size[j], padding=padding[j])
                    )
                encoder.add_module(f'enc{i+1}_act{j+1}', nn.ReLU(True))
            encoder.add_module(f"enc{i+1}_adaptiveavgpool", nn.AdaptiveAvgPool2d((1, 1)))
            self.encoder_list.append(encoder)

            # homography変換のパラメータ（8自由度：H[2,2]=1に固定）を出力するための全結合層
            fc_loc = nn.Sequential()
            if len(fc_hidden_size) > 0:
                for j in range(len(fc_hidden_size)):
                    in_features = hidden_layers[-1] if j == 0 else fc_hidden_size[j - 1]
                    out_features = fc_hidden_size[j]
                    fc_loc.add_module(f'fc{i+1}_linear{j+1}', nn.Linear(in_features, out_features))
                    fc_loc.add_module(f'fc{i+1}_act{j+1}', nn.ReLU(True))
                fc_loc.add_module(f'out{i+1}_linear', nn.Linear(fc_hidden_size[-1], fc_output_size))
            else:
                fc_loc.add_module(f'out{i+1}_linear', nn.Linear(hidden_layers[-1], fc_output_size))

            # 残差パラメータ化のため、最終層を初期化
            # std=0.01に増やして、初期状態でも少し変化が起きるようにする
            nn.init.normal_(fc_loc[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(fc_loc[-1].bias)

            self.fc_loc_list.append(fc_loc)
            
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
        
    def forward(self, x):
        B, C, H_img, W_img = x.shape
        
        for encoder, fc_loc in zip(self.encoder_list, self.fc_loc_list):
            
            xs_mask = x[:, 3:, :, :]
            x_coords, y_coords = self.get_object_sentric_coordinates(xs_mask)
            
            T_center = torch.zeros(B, 3, 3).to(x.device)
            T_center[:, 0, 0] = 1.0
            T_center[:, 1, 1] = 1.0
            T_center[:, 2, 2] = 1.0
            T_center[:, 0, 2] = -x_coords
            T_center[:, 1, 2] = -y_coords
            
            T_center_inv = torch.zeros(B, 3, 3).to(x.device)
            T_center_inv[:, 0, 0] = 1.0
            T_center_inv[:, 1, 1] = 1.0
            T_center_inv[:, 2, 2] = 1.0
            T_center_inv[:, 0, 2] = x_coords
            T_center_inv[:, 1, 2] = y_coords
            
            xs = encoder(x)
            xs = torch.flatten(xs, 1)  # (B, hidden)

            # 4パラメータを出力
            theta = fc_loc(xs)                 # (B, 8)
            theta = torch.tanh(theta)
            
            m = torch.tan(theta[:, 0] * math.pi)
            c = theta[:, 1] * 255.0

            A = torch.zeros(B, 3, 3).to(x.device)
            A[:, 0, 0] = (1-m*m)/(1+m*m)
            A[:, 0, 1] = (2*m)/(1+m*m)
            A[:, 0, 2] = (-2*m*c)/(1+m*m)
            A[:, 1, 0] = (2*m)/(1+m*m)
            A[:, 1, 1] = (m*m-1)/(1+m*m)
            A[:, 1, 2] = (2*c)/(1+m*m)
            A[:, 2, 2] = 1.0
            
            P = torch.zeros(B, 3, 3).to(x.device)
            P[:, 0, 0] = 1.0
            P[:, 1, 1] = 1.0
            P[:, 2, 0] = theta[:, 2] * self.persp_max
            P[:, 2, 1] = theta[:, 3] * self.persp_max
            P[:, 2, 2] = 1.0
            
            H = torch.bmm(torch.bmm(T_center_inv, torch.bmm(P, A)), T_center)

            x = K.geometry.transform.warp_perspective(x, H, (W_img, H_img))

        return x
    
import yaml
try:
    # torchsummary が環境に無い場合はスキップできるようにする
    from torchsummary import summary
    _HAS_TORCHSUMMARY = True
except ImportError:
    _HAS_TORCHSUMMARY = False

if __name__ == "__main__":
    # stn = SpatialTransformerNetwork()
    # # stn = torch.compile(stn)
    # input_dummy = torch.randn(1, 1, 28, 28)
    # writer = SummaryWriter(comment=f'STN_tutorial')
    # writer.add_graph(stn, input_dummy)
    # writer.close()
    config_path = "../config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_params = config["model"]["params"]
    # 正しいクラス名に修正
    model = SpatialTransformerNetworkWmaskWaffine(**model_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if _HAS_TORCHSUMMARY:
        summary(model, (1, 28, 28))
    else:
        print("torchsummary が見つからないため summary をスキップしました。")