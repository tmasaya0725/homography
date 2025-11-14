"""
推論スクリプト

・学習時と同じ config.yaml を読み込み
・チェックポイント（.ckpt）から LightningModule をロード
・データモジュールの val/test ローダーからサンプルを取り、出力画像を保存

出力は以下を横に連結した可視化画像（PNG）
  [src_rgb | src_mask | tgt_rgb | tgt_mask | comp(rgb) | out_mask]

使い方（例）:
	python homography/inference.py \
		--checkpoint /path/to/outputs/2025-11-07/exp/checkpoints/last.ckpt \
		--split validation \
		--max-samples 32 \
		--output-dir ./inference_outputs

注意:
- config.yaml の data: を使ってデータセットを構築します。
- STNModuleWmaskWdec.forward は (output, rot) を返す想定です。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import time
import torch
import torch.nn.functional as F
import numpy as np

import pytorch_lightning as pl
from omegaconf import OmegaConf

from trainer.util.util import get_model, get_data, MODEL_REGISTRY


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Homography STN 推論スクリプト")
	default_cfg = Path(__file__).parent / "config" / "config.yaml"
	parser.add_argument(
		"--config", type=str, default=str(default_cfg), help="設定ファイル(config.yaml)のパス"
	)
	parser.add_argument(
		"--checkpoint", type=str, default=None,
		help="学習済みチェックポイントの指定。未指定かつ --config が実験ディレクトリなら、その配下のcheckpoints/last.ckptを使用。\n"
			 "数値を渡した場合は epoch 番号として実験配下から該当ckptを検索（例: --checkpoint 12）"
	)
	parser.add_argument(
		"--split", type=str, default="validation", choices=["train", "validation", "test"], help="どの分割で推論するか"
	)
	parser.add_argument(
		"--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="推論デバイス"
	)
	parser.add_argument(
		"--max-samples", type=int, default=16, help="保存する最大サンプル数（総枚数）"
	)
	parser.add_argument(
		"--output-dir", type=str, default=None,
		help="出力画像の保存ディレクトリ。未指定かつ --config が実験ディレクトリなら <exp>/test を用いる"
	)
	return parser.parse_args()


def auto_device(device_opt: str) -> torch.device:
	if device_opt == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return torch.device(device_opt)


def load_model_from_ckpt(cfg, ckpt_path: str, device: torch.device) -> pl.LightningModule:
	model_type = cfg.model.type
	assert model_type in MODEL_REGISTRY, f"Unknown model type: {model_type}"
	ModelClass = MODEL_REGISTRY[model_type]

	# Lightning の load_from_checkpoint を使用
	model: pl.LightningModule = ModelClass.load_from_checkpoint(ckpt_path)
	model.eval()
	model.to(device)
	return model


def resolve_experiment_dir(config_arg: str) -> tuple[Path | None, Path | None]:
	"""
	--config で渡された値が実験ディレクトリであれば (exp_dir, saved_cfg_path) を返す。
	実験ディレクトリの判定は <exp_dir>/.hydra/config.yaml が存在するかで行う。
	実験ディレクトリでない場合は (None, None) を返す。
	"""
	p = Path(config_arg).expanduser().resolve()
	if p.is_dir():
		saved_cfg = p / ".hydra" / "config.yaml"
		if saved_cfg.exists():
			return p, saved_cfg
	return None, None


def find_ckpt(exp_dir: Path, checkpoint_arg: str | None) -> Path:
	"""
	checkpoint_arg の解釈:
	- None: <exp_dir>/checkpoints/last.ckpt を返す
	- 末尾が .ckpt の既存ファイルパス: そのまま返す（絶対/相対可）
	- 数字（例: "12"）: <exp_dir>/checkpoints 内で epoch{12:02d} を含む ckpt を検索
	- "last": <exp_dir>/checkpoints/last.ckpt
	見つからなければ例外を投げる。
	"""
	if checkpoint_arg is None or checkpoint_arg == "last":
		ck = exp_dir / "checkpoints" / "last.ckpt"
		if not ck.exists():
			raise FileNotFoundError(f"last.ckpt が見つかりません: {ck}")
		return ck

	arg_path = Path(checkpoint_arg)
	if arg_path.suffix == ".ckpt" and arg_path.exists():
		return arg_path.resolve()

	# エポック番号が来た場合
	if checkpoint_arg.isdigit():
		epoch = int(checkpoint_arg)
		pat = f"epoch{epoch:02d}"
		ckpt_dir = exp_dir / "checkpoints"
		if not ckpt_dir.exists():
			raise FileNotFoundError(f"チェックポイントディレクトリがありません: {ckpt_dir}")
		candidates = sorted([p for p in ckpt_dir.glob("*.ckpt") if pat in p.stem])
		if not candidates:
			raise FileNotFoundError(f"epoch {epoch} を含む ckpt が見つかりません: {ckpt_dir}")
		return candidates[-1].resolve()

	# 実験配下の相対パス指定かもしれない
	maybe = (exp_dir / checkpoint_arg)
	if maybe.exists() and maybe.suffix == ".ckpt":
		return maybe.resolve()

	raise FileNotFoundError(f"チェックポイントの解決に失敗しました: {checkpoint_arg}")


def build_dataloader(cfg, split: str):
	"""cfg.data を使って DataModule を構築し、指定 split の DataLoader を返す。"""
	data_module = get_data(cfg.data)
	# DataModule をセットアップ
	if split in ("train", "validation"):
		data_module.setup(stage="fit")
	else:
		data_module.setup(stage="test")

	if split == "train":
		return data_module.train_dataloader()
	if split == "validation":
		return data_module.val_dataloader()
	return data_module.test_dataloader()


@torch.no_grad()
def run_inference(model: pl.LightningModule, dataloader, device: torch.device, out_dir: Path, max_samples: int):
	out_dir.mkdir(parents=True, exist_ok=True)

	saved = 0
	batch_idx = 0
	for batch in dataloader:
		batch_idx += 1
		# バッチをデバイスへ
		src_img = batch["source"].to(device)  # (B,4,H,W) 先頭3chがRGB, 4ch目が入力マスク
		tgt_img = batch["target"].to(device)  # (B,3,H,W)
		src_mask = batch["mask"].to(device)   # (B,1,H,W)
		tgt_mask = batch["ref_mask"].to(device)  # (B,1,H,W)

		# LightningModule の forward: (output, rot) を想定
		output, rot = model(src_img)

		# 出力の分解
		out_rgb = output[:, :3]                # (B,3,H,W)
		out_mask = output[:, 3:]               # (B,1,H,W)
		comp = out_rgb * out_mask + src_img[:, :3] * (1.0 - out_mask)

		# サンプルごとに保存
		B = src_img.size(0)
		for i in range(B):
			if saved >= max_samples:
				return
			grid = make_vis_grid(
				src_rgb=src_img[i, :3],
				src_m=src_mask[i],
				tgt_rgb=tgt_img[i],
				tgt_m=tgt_mask[i],
				comp=comp[i],
				out_m=out_mask[i],
			)
			save_path = out_dir / f"sample_{saved:05d}.png"
			save_png(grid, save_path)
			saved += 1


def to_uint8(img_chw: torch.Tensor) -> np.ndarray:
	"""[C,H,W] (0..1想定) -> uint8 [H,W,C]"""
	img = img_chw.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
	return img


def to_uint8_1ch(mask_chw: torch.Tensor) -> np.ndarray:
	"""[1,H,W] (0..1想定) -> uint8 [H,W,3] (3chに拡張)"""
	m = mask_chw.detach().clamp(0, 1).mul(255).byte().squeeze(0).cpu().numpy()  # [H,W]
	m3 = np.stack([m, m, m], axis=-1)  # [H,W,3]
	return m3


def make_vis_grid(src_rgb: torch.Tensor, src_m: torch.Tensor, tgt_rgb: torch.Tensor, tgt_m: torch.Tensor,
				  comp: torch.Tensor, out_m: torch.Tensor) -> np.ndarray:
	"""6枚を横連結して可視化グリッドを返す。各引数は [C,H,W]。"""
	a = to_uint8(src_rgb)
	b = to_uint8_1ch(src_m)
	c = to_uint8(tgt_rgb)
	d = to_uint8_1ch(tgt_m)
	e = to_uint8(comp)
	f = to_uint8_1ch(out_m)
	return np.concatenate([a, b, c, d, e, f], axis=1)  # 横方向に連結 (W方向)


def save_png(img_hwc_uint8: np.ndarray, path: Path):
	# imageio または cv2 のどちらかで保存。標準で入っている cv2 を使う。
	import cv2
	bgr = cv2.cvtColor(img_hwc_uint8, cv2.COLOR_RGB2BGR)
	cv2.imwrite(str(path), bgr)


def main():
	args = parse_args()
	# --config の解釈: 実験ディレクトリ or 設定ファイルパス
	exp_dir, saved_cfg_path = resolve_experiment_dir(args.config)
	if saved_cfg_path is not None:
		cfg_path = saved_cfg_path
	else:
		cfg_path = Path(args.config)
		if not cfg_path.exists():
			raise FileNotFoundError(f"設定ファイル/実験ディレクトリが見つかりません: {args.config}")
	cfg = OmegaConf.load(cfg_path)

	device = auto_device(args.device)
	print(f"device: {device}")
	print(f"config: {cfg_path}")
	if exp_dir is not None:
		print(f"experiment_dir: {exp_dir}")
	# チェックポイント解決
	if exp_dir is not None:
		ckpt_path = find_ckpt(exp_dir, args.checkpoint)
	else:
		# 実験ディレクトリがない場合は --checkpoint 必須
		if args.checkpoint is None:
			raise ValueError("--config が実験ディレクトリでない場合、--checkpoint を指定してください")
		p = Path(args.checkpoint)
		if not p.exists():
			raise FileNotFoundError(f"指定されたチェックポイントが存在しません: {p}")
		ckpt_path = p.resolve()
	print(f"checkpoint: {ckpt_path}")
	print(f"split: {args.split}, max_samples: {args.max_samples}")

	# モデルのロード
	model = load_model_from_ckpt(cfg, str(ckpt_path), device)

	# DataLoader の構築
	loader = build_dataloader(cfg, args.split)

	# 出力先
	timestamp = time.strftime("%Y%m%d-%H%M%S")
	if args.output_dir is None and exp_dir is not None:
		base_out = exp_dir / "test"
	else:
		base_out = Path(args.output_dir) if args.output_dir is not None else Path("./inference_outputs")
	out_dir = base_out / f"{ckpt_path.stem}_{args.split}_{timestamp}"

	# 推論実行
	run_inference(model, loader, device, out_dir, args.max_samples)
	print(f"Done. Saved to: {out_dir}")


if __name__ == "__main__":
	torch.set_float32_matmul_precision("high")
	main()

