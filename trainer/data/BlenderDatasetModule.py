import os
import cv2
import numpy as np
import random
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class BlenderDataset(Dataset):
    def __init__(self, data_dir, type='train', image_size=256):
        self.type = type
        self.on_path = os.path.join(data_dir, type, 'rendered_image/')
        self.off_path = os.path.join(data_dir, type, 'rendered_image_no_reflection/')
        self.mask_path = os.path.join(data_dir, type, 'rendered_image_reflection_mask/')
        self.obj_path = os.path.join(data_dir, type, 'rendered_image_object_mask/')
        self.image_size = image_size
        
        # ファイルリストを取得
        self.data = os.listdir(self.on_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data[idx]

        # 各フォルダから画像を読み込む
        source = cv2.imread(os.path.join(self.off_path, filename))
        mask = cv2.imread(os.path.join(self.obj_path, filename), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(os.path.join(self.on_path, filename))
        ref_mask = cv2.imread(os.path.join(self.mask_path, filename), cv2.IMREAD_GRAYSCALE)
        
        if self.type == 'train':
            # 訓練時はランダムクロップとフリップ
            height, width = target.shape[:2]
            crop_size = random.randint(self.image_size, height)
            crop_positions = self._get_random_crop_positions(width, height, crop_size)
            
            target = self._crop_image(target, crop_positions)
            source = self._crop_image(source, crop_positions)
            mask = self._crop_image(mask, crop_positions)
            ref_mask = self._crop_image(ref_mask, crop_positions)
            
            # ランダムにフリップ
            if random.choice([True, False]):
                source = cv2.flip(source, 1)
                target = cv2.flip(target, 1)
                mask = cv2.flip(mask, 1)
                ref_mask = cv2.flip(ref_mask, 1)
        else:
            # 検証・テスト時はセンタークロップ
            crop_size = random.randint(self.image_size, min(source.shape[0], source.shape[1]))
            source = self._center_crop(source, crop_size)
            target = self._center_crop(target, crop_size)
            mask = self._center_crop(mask, crop_size)
            ref_mask = self._center_crop(ref_mask, crop_size)
            
        # リサイズ
        source = cv2.resize(source, (self.image_size, self.image_size))
        target = cv2.resize(target, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        ref_mask = cv2.resize(ref_mask, (self.image_size, self.image_size))

        # BGRからRGBに変換
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        # 正規化
        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0
        mask = mask.astype(np.float32) / 255.0
        ref_mask = ref_mask.astype(np.float32) / 255.0

        # input画像とマスク画像の結合
        mask = np.expand_dims(mask, axis=2)
        source = np.concatenate((source, mask), axis=2)

        # 辞書形式で返す
        return {
            'target': target,
            'source': source,
            'mask': mask,
            'ref_mask': ref_mask,
        }
    
    def _get_random_crop_positions(self, width, height, crop_size):
        """ランダムなクロップ位置を取得"""
        if crop_size > min(width, height):
            raise ValueError("指定されたクロップサイズが元画像より大きいです")
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
        right = left + crop_size
        bottom = top + crop_size
        return left, top, right, bottom

    def _crop_image(self, image, positions):
        """指定された位置で画像をクロップ"""
        left, top, right, bottom = positions
        cropped_image = image[top:bottom, left:right]
        return cropped_image
    
    def _center_crop(self, img, size):
        """画像を中心からクロップ"""
        height, width = img.shape[:2]
        new_width, new_height = size, size
        start_x = width//2 - (new_width//2)
        start_y = height//2 - (new_height//2)
        return img[start_y:start_y+new_height, start_x:start_x+new_width]
    
    

class BlenderDatasetModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./blender_data/", batch_size: int = 16, num_workers: int = 4, image_size: int = 256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def prepare_data(self):
        assert os.path.exists(self.data_dir), f"Data directory {self.data_dir} does not exist"

    def setup(self, stage: str | None = None):
        if stage == 'fit' or stage is None:
            self.train_set = BlenderDataset(self.data_dir, type='train', image_size=self.image_size)
            self.val_set = BlenderDataset(self.data_dir, type='validation', image_size=self.image_size)
        if stage == 'test' or stage is None:
            self.test_set = BlenderDataset(self.data_dir, type='test', image_size=self.image_size)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)