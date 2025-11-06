import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import kornia as K
import numpy as np
import math

class SpatialTransformerNetworkWmaskWdec(nn.Module):
    def __init__(self, num_network=4, 
                 input_channels=1, hidden_layers=[16, 32], 
                 kernel_size=[3, 3], stride=[2, 2], padding=[1, 1],
                 fc_hidden_size=[32], fc_output_size=7,
                 scale_max=2.0, persp_max=0.5, trans_scale=1.0):
        super().__init__()

        assert fc_output_size == 7, "fc_output_size must be 7 for WmaskWdec model"

        self.encoder_list = nn.ModuleList()
        self.fc_loc_list = nn.ModuleList()

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
                # encoder.add_module(f'enc{i+1}_ln{j+1}', nn.LayerNorm([out_channels, 1, 1]))
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
                    fc_loc.add_module(f'fc{i+1}_act{j+1}', nn.SiLU(True))
                fc_loc.add_module(f'out{i+1}_linear', nn.Linear(fc_hidden_size[-1], fc_output_size))
            else:
                fc_loc.add_module(f'out{i+1}_linear', nn.Linear(hidden_layers[-1], fc_output_size))

            # 残差パラメータ化のため、最終層を初期化
            # std=0.01に増やして、初期状態でも少し変化が起きるようにする
            nn.init.normal_(fc_loc[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(fc_loc[-1].bias)

            self.fc_loc_list.append(fc_loc)

        # 回転・拡大縮小
        self.scale_max = scale_max   
        # 透視
        self.persp_max = persp_max  
        # 並進
        self.trans_scale = trans_scale
        
    def forward(self, x, mask):
        B, C, H, W = x.shape
        center_x = W / 2
        center_y = H / 2


        for encoder, fc_loc in zip(self.encoder_list, self.fc_loc_list):
            
            xs = encoder(x)
            xs = torch.flatten(xs, 1)  # (B, hidden)

            # 8パラメータを出力
            raw = fc_loc(xs)                 # (B, 8)
            delta =  raw#torch.tanh(raw)          # tanhで-1~1に制限（安定化のため）

            rot =  delta[:, 0] * math.pi      # -π ~ π torch.tensor(math.pi) 
            scale_x = 1.0 + delta[:, 1] * self.scale_max
            scale_y = 1.0 + delta[:, 2] * self.scale_max
            trans_x = delta[:, 3] * W * self.trans_scale
            trans_y = delta[:, 4] * H * self.trans_scale
            persp_u = delta[:, 5] * self.persp_max
            persp_v = delta[:, 6] * self.persp_max

            # 回転の行列
            R_matrix = torch.zeros((B, 3, 3), device=x.device, dtype=x.dtype)
            #print(R_matrix.mean())
            cos_t = torch.cos(rot)
            sin_t = torch.sin(rot)
            R_matrix[:, 0, 0] = cos_t
            R_matrix[:, 0, 1] = -sin_t
            R_matrix[:, 1, 0] = sin_t
            R_matrix[:, 1, 1] = cos_t
            R_matrix[:, 2, 2] = 1.0

            # 拡大縮小の行列
            S_matrix = torch.zeros((B, 3, 3), device=x.device, dtype=x.dtype)
            S_matrix[:, 0, 0] = scale_x
            S_matrix[:, 1, 1] = scale_y
            S_matrix[:, 2, 2] = 1.0

            # 並進の行列
            T_matrix = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
            T_matrix[:, 0, 2] = trans_x
            T_matrix[:, 1, 2] = trans_y
            
            # 透視変換の行列
            P_matrix = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
            P_matrix[:, 2, 0] = persp_u
            P_matrix[:, 2, 1] = persp_v
            
            # 中心合わせの行列
            T_center = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
            T_center[:, 0, 2] = center_x
            T_center[:, 1, 2] = center_y

            T_center_inv = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
            T_center_inv[:, 0, 2] = -center_x
            T_center_inv[:, 1, 2] = -center_y
            
            theta = T_matrix.bmm(T_center).bmm(R_matrix).bmm(S_matrix).bmm(P_matrix).bmm(T_center_inv)  # H行列を計算

            x = K.geometry.transform.warp_perspective(x, theta, (W, H))
            mask = K.geometry.transform.warp_perspective(mask, theta, (W, H), mode="nearest", padding_mode="zeros")
            
        return x, mask, rot
    
import yaml
from torchsummary import summary

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
    model = SpatialTransformerNetwork(**model_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    summary(model, (1, 28, 28))