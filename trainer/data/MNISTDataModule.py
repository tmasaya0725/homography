import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import kornia as K
import torch

class HomographyMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, _ = self.mnist_dataset[idx]  # img: (1, H, W), tensor
        # num_workers > 0 での非決定性を避けるため、毎回 RandomPerspective を作成
        homography = K.augmentation.RandomPerspective(distortion_scale=0.75, p=1.0)
        img_homo = homography(img.unsqueeze(0)).squeeze(0)  # ゆがませた画像
        return img_homo, img  # (入力, 正解)

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None):
        full_train = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
        self.train_set, self.val_set = random_split(full_train, [55000, 5000])
        self.test_set = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

        # ここでカスタムデータセットにラップ
        self.train_set = HomographyMNIST(self.train_set)
        self.val_set = HomographyMNIST(self.val_set)
        self.test_set = HomographyMNIST(self.test_set)

    def train_dataloader(self):
        # num_workers > 0 でホモグラフィー変換の非決定性が生じるため、0に設定
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)
