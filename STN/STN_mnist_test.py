# https://docs.pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html より引用
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from SpatialTransformerNetwork import SpatialTransformerNetwork

from six.moves import urllib

import kornia as K
import torchvision.transforms as T
import cv2
import yaml

# Training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])), batch_size=2048, shuffle=True, num_workers=4
    )

# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=2048, shuffle=True, num_workers=4)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)                    # (B, 1, 28, 28)
        warped, H = random_homography_transform(data)  # 歪み＋反転を加える

        optimizer.zero_grad()
        output = model(warped)   

        loss = F.mse_loss(output, data, reduction='mean')
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            target = data.to(device)
            data , H= random_homography_transform(data.to(device))
            data = data.to(device)
            output = model(data)
            

            # sum up batch loss
            test_loss += F.mse_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}\n'
              .format(test_loss))
            
def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    # inp は (C, H, W) の形状 (make_grid の出力)
    inp = inp.numpy().transpose((1, 2, 0)) # (H, W, C) に変換
    
    # --- 修正点 ---
    # MNIST (1チャンネル) 用の mean と std を使用
    mean = np.array([0.1307])
    std = np.array([0.3081])
    # --- 修正点 ---
    
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def visualize_stn():
    with torch.no_grad():
        model.eval()
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)
        
        # 歪んだ入力画像を作成
        warped_data , H = random_homography_transform(data.clone())
        
        # 歪んだ画像 (warped_data) をモデルに入力し、修復された画像 (transformed_input_tensor) を得る
        transformed_input_tensor = model.forward(warped_data).cpu()
        
        # 入力画像（歪んだもの）
        input_tensor = warped_data.cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))
        
        # 比較のためにオリジナルの画像も保存
        original_grid = convert_image_np(
             torchvision.utils.make_grid(data.cpu()))
        
        print("Saving visualization images...")
        cv2.imwrite("stn_original.png", (original_grid * 255).astype(np.uint8))
        cv2.imwrite("stn_input_warped.png", (in_grid * 255).astype(np.uint8))
        cv2.imwrite("stn_output_restored.png", (out_grid * 255).astype(np.uint8))
        print("Images saved.")

def random_homography_transform(img: torch.Tensor, max_warp=0.0, flip_prob=0.5):
    # B, C, H, W = img.shape
    # # 上下左右反転のホモグラフィー変換の行列を作成
    # H_flip = torch.tensor([[-1, 0, W],
    #                         [0, -1, H],
    #                         [0, 0, 1]], dtype=torch.float32, device=img.device)
    # H_flip = H_flip.unsqueeze(0).repeat(B, 1, 1)  # (B, 3, 3)
    # x = K.geometry.transform.warp_perspective(img, H_flip, (W, H))

    transform_kornia = K.augmentation.RandomPerspective(distortion_scale=0.75, p=1.0)
    x = transform_kornia(img)
    return x, 1

torch.set_float32_matmul_precision('high')

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = "../config/config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

model_params = config["model"]["params"]
model = SpatialTransformerNetwork(**model_params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.compile(model).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4)

for epoch in range(1, 20 + 1):
    train(epoch)
    test()

# Visualize the STN transformation on some input batch
visualize_stn()