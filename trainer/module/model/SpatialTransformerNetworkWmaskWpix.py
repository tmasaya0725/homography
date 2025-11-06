import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import kornia as K
import numpy as np

class SpatialTransformerNetworkWmaskWpix(nn.Module):
    def __init__(self, num_network=4, 
                 input_channels=1, hidden_layers=[16, 32], 
                 kernel_size=[3, 3], stride=[2, 2], padding=[1, 1],
                 fc_hidden_size=[32], fc_output_size=8,
                 scale_max=2.0, persp_max=0.5, trans_scale=1.0):
        super().__init__()

        assert fc_output_size == 8, "fc_output_size must be 8 for WmaskWpix model"

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
                encoder.add_module(f'enc{i+1}_act{j+1}', nn.SiLU(True))
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
            # nn.init.normal_(fc_loc[-1].weight, mean=0.0, std=0.01)
            # nn.init.zeros_(fc_loc[-1].bias)

            self.fc_loc_list.append(fc_loc)
        
    def forward(self, x, mask):
        B, C, H_img, W_img = x.shape

        for encoder, fc_loc in zip(self.encoder_list, self.fc_loc_list):
            
            xs = encoder(x)
            xs = torch.flatten(xs, 1)  # (B, hidden)

            # 8パラメータを出力
            raw = fc_loc(xs)                 # (B, 8)
            delta = torch.tanh(raw)          # tanhで-1~1に制限（安定化のため）
            
            # 出力deltaは4点の変位量(dx, dy)なので、元の座標に加算して変換後の4点を求める
            # 座標は(x, y)の順で、画像の4隅を表す
            # 反射のような極端な変形を可能にするため、画像サイズの2倍のスケールを使用
            H_img_f = float(H_img - 1)  # 高さの最大インデックス
            W_img_f = float(W_img - 1)  # 幅の最大インデックス
            
            # 画像の対角線長をベースに、さらに大きなスケールを設定（2.0倍）
            # これにより右上の点が右下まで移動可能になる
            diagonal = (H_img**2 + W_img**2)**0.5
            scale_factor = diagonal * 2.0  # 対角線の2倍の移動範囲
            
            pts1 = torch.tensor([[[0.0, 0.0],              # 左上 (x, y)
                                  [W_img_f, 0.0],          # 右上
                                  [W_img_f, H_img_f],      # 右下
                                  [0.0, H_img_f]]], device=x.device)  # 左下 (1, 4, 2)
            pts1 = pts1.repeat(B, 1, 1)  # (B, 4, 2)
            delta_reshaped = delta.view(B, 4, 2) * scale_factor  # (B, 4, 2) スケール適用
            pts2 = pts1 + delta_reshaped
            
            # ホモグラフィ行列を計算
            H_mat = K.geometry.homography.find_homography_dlt(pts1, pts2)  # (B, 3, 3)

            x = K.geometry.transform.warp_perspective(x, H_mat, (x.size(3), x.size(2)))
            mask = K.geometry.transform.warp_perspective(mask, H_mat, (mask.size(3), mask.size(2)), mode="nearest", padding_mode="zeros")
        
        return x, mask
    
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
    