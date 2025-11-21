import os
import cv2
import numpy as np
import random
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import json

class BlenderDataset(Dataset):
    def __init__(self, data_dir, type='train', image_size=256):
        self.type = type
        self.on_path = os.path.join(data_dir, type, 'rendered_image/')
        self.off_path = os.path.join(data_dir, type, 'rendered_image_no_reflection/')
        self.mask_path = os.path.join(data_dir, type, 'rendered_image_reflection_mask/')
        self.obj_path = os.path.join(data_dir, type, 'rendered_image_object_mask/')
        self.normals_path = os.path.join(data_dir, type, 'normals_image/')
        self.image_size = image_size

        # ファイルリストを取得
        self.data = os.listdir(self.on_path)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # filename = self.data[idx]
        # 仮
        if self.type == 'train':
            if idx == 0:
                filename = "render_6.png"
            else:
                filename = self.data[idx]
        else:
            filename = self.data[idx]

        
        # 各フォルダから画像を読み込む
        source = cv2.imread(os.path.join(self.normals_path, filename))
        mask = cv2.imread(os.path.join(self.obj_path, filename), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(os.path.join(self.on_path, filename))
        ref_mask = cv2.imread(os.path.join(self.mask_path, filename), cv2.IMREAD_GRAYSCALE)
        
        if False:# self.type == 'train':
            target, source, mask, ref_mask = self._crop_until_masks_inside(
                target, source, mask, ref_mask,
                crop_size_min=256, crop_size_max=350, max_attempts=20
            )
        else:
            # 検証・テスト時はセンタークロップ
            crop_size = 256#random.randint(self.image_size, min(source.shape[0], source.shape[1]))
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
        target = target.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        ref_mask = ref_mask.astype(np.float32) / 255.0

        # input画像とマスク画像の結合
        mask = np.expand_dims(mask, axis=2)
        ref_mask = np.expand_dims(ref_mask, axis=2)
        source = np.concatenate((source, mask), axis=2)

        source = np.transpose(source, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        target = np.transpose(target, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        mask = np.transpose(mask, (2, 0, 1))      # (H, W, 1) -> (1, H, W)
        ref_mask = np.transpose(ref_mask, (2, 0, 1))  # (H, W, 1) -> (1, H, W)

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
    
    def _crop_until_masks_inside(self, target, source, mask, ref_mask,
                             crop_size_min=256, crop_size_max=400, max_attempts=10):
        """
        マスクがクロップ領域の境界で切れないようになるまでランダムクロップを繰り返す。
        成功したクロップ画像群 (target, source, mask, ref_mask) を返す。
        最大試行回数後は最後のクロップを返す（フォールバック）。
        """
        H, W = target.shape[:2]
        last = None

        def _mask_touch_border(m):
            """与えられたマスク配列がクロップの境界に触れているかを判定する。
            マスクが全ゼロなら False（切れていない）を返す。"""
            arr = np.asarray(m)
            # チャネルがある場合は空間次元だけを使う
            if arr.ndim == 3:
                arr_sp = arr[..., 0] if arr.shape[2] <= 4 else arr[..., 0]
                arr = arr_sp
            if arr.size == 0:
                return False
            # 二値化（閾値は 0）
            nonzero = (arr > 0)
            if not nonzero.any():
                return False  # マスク空なら切れていないとみなす
            # 上下左右の境界行列に非ゼロがあれば切れている
            if nonzero[0, :].any() or nonzero[-1, :].any() or nonzero[:, 0].any() or nonzero[:, -1].any():
                return True
            return False

        for _ in range(max_attempts):
            crop_size = random.randint(crop_size_min, crop_size_max)
            crop_pos = self._get_random_crop_positions(W, H, crop_size)
            tgt_c = self._crop_image(target, crop_pos)
            src_c = self._crop_image(source, crop_pos)
            mask_c = self._crop_image(mask, crop_pos)
            ref_mask_c = self._crop_image(ref_mask, crop_pos)

            last = (tgt_c, src_c, mask_c, ref_mask_c)

            # どちらのマスクも境界に触れていなければ採用
            if (not _mask_touch_border(mask_c)) and (not _mask_touch_border(ref_mask_c)):
                return tgt_c, src_c, mask_c, ref_mask_c

        # 最大試行回数に達したら最後のクロップを返す（フォールバック）
        return last
    
    

class BlenderDatasetModule_E(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./blender_data/", batch_size: int = 16, num_workers: int = 4, image_size: int = 256, limit_samples: int | None = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        # デバッグ・検証用: データセットの件数を制限（例: 1 で1枚だけ）
        self.limit_samples = limit_samples

    def prepare_data(self):
        assert os.path.exists(self.data_dir), f"Data directory {self.data_dir} does not exist"

    def setup(self, stage: str | None = None):
        if stage == 'fit' or stage is None:
            self.train_set = BlenderDataset(self.data_dir, type='train', image_size=self.image_size)
            self.val_set = BlenderDataset(self.data_dir, type='validation', image_size=self.image_size)
            if self.limit_samples is not None:
                from torch.utils.data import Subset
                n_train = min(self.limit_samples, len(self.train_set))
                n_val = min(self.limit_samples, len(self.val_set))
                self.train_set = Subset(self.train_set, list(range(n_train)))
                self.val_set = Subset(self.val_set, list(range(n_val)))
        if stage == 'test' or stage is None:
            self.test_set = BlenderDataset(self.data_dir, type='test', image_size=self.image_size)
            if self.limit_samples is not None:
                from torch.utils.data import Subset
                n_test = min(self.limit_samples, len(self.test_set))
                self.test_set = Subset(self.test_set, list(range(n_test)))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
        
if __name__ == "__main__":
    dataset = BlenderDataset(data_dir="/home/takanashi.masaya/workspace/dataset/blenderproc_depth_512_E", type='train', image_size=256)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Source shape: {sample['source'].shape}")
    print(f"Target shape: {sample['target'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Ref Mask shape: {sample['ref_mask'].shape}")