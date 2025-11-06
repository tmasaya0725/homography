import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os
from pathlib import Path
import sys

from trainer.util.util import get_model, get_data
import torch

# スクリプトのディレクトリから設定ファイルのパスを取得
SCRIPT_DIR = Path(__file__).parent
CONFIG_DIR = SCRIPT_DIR / "config"

def train_from_checkpoint(checkpoint_path: str):
    """チェックポイントから学習を再開"""
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"チェックポイントが見つかりません: {checkpoint_path}")
    
    # チェックポイントのディレクトリ構造: .../outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/xxx.ckpt
    checkpoint_dir = checkpoint_path.parent  # checkpointsディレクトリ
    output_dir = checkpoint_dir.parent  # outputs/YYYY-MM-DD/HH-MM-SS
    saved_config_path = output_dir / ".hydra" / "config.yaml"
    
    if not saved_config_path.exists():
        raise FileNotFoundError(f"保存された設定ファイルが見つかりません: {saved_config_path}")
    
    print(f"チェックポイントから学習を再開します: {checkpoint_path}")
    print(f"ログディレクトリ: {output_dir}")
    
    # 最新のconfig.yamlを読み込む（max_epochなどの更新を反映するため）
    current_config_path = CONFIG_DIR / "config.yaml"
    cfg = OmegaConf.load(current_config_path)
    print(f"最新の設定ファイルを読み込みます: {current_config_path}")
    
    # 保存されていた設定も読み込んで、必要に応じて参照
    saved_cfg = OmegaConf.load(saved_config_path)
    print(f"保存されていた設定ファイル: {saved_config_path}")
    print(f"  - 保存時のmax_epochs: {saved_cfg.trainer.get('max_epochs', 'N/A')}")
    print(f"  - 現在のmax_epochs: {cfg.trainer.get('max_epochs', 'N/A')}")
    
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    
    # チェックポイントコールバックの設定を更新（既存のcheckpointsディレクトリを使用）
    callbacks = []
    for cb_name, cb_cfg in cfg.get("callbacks", {}).items():
        if cb_name == "checkpoint":
            cb_cfg.dirpath = str(checkpoint_dir)  # 既存のcheckpointsディレクトリ
        callbacks.append(hydra.utils.instantiate(cb_cfg))
    
    # TensorBoardのログを既存のディレクトリに追記
    logger_cfg = cfg.get("logger")
    if logger_cfg:
        logger = TensorBoardLogger(
            save_dir=str(output_dir),
            name="",
            version="",
            default_hp_metric=False
        )
    else:
        logger = True
    
    # チェックポイントから直接モデルをロード
    from trainer.util.util import MODEL_REGISTRY
    ModelClass = MODEL_REGISTRY[cfg.model.type]
    model = ModelClass.load_from_checkpoint(str(checkpoint_path))
    print(f"保存されていたハイパーパラメータ: {model.hparams}")
    
    data = get_data(cfg.data)
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule=data, ckpt_path=str(checkpoint_path))

@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.get("seed", 42), workers=True)

    # Hydraの出力ディレクトリを取得
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # チェックポイントコールバックの設定を更新
    callbacks = []
    for cb_name, cb_cfg in cfg.get("callbacks", {}).items():
        if cb_name == "checkpoint":
            # dirkpathをHydraの出力ディレクトリ内に設定
            if cb_cfg.get("dirpath") is None:
                cb_cfg.dirpath = os.path.join(hydra_output_dir, "checkpoints")
        callbacks.append(hydra.utils.instantiate(cb_cfg))
    
    # TensorBoardのログをHydraの出力ディレクトリに保存
    logger_cfg = cfg.get("logger")
    if logger_cfg:
        logger = TensorBoardLogger(
            save_dir=hydra_output_dir,
            name="",  # サブディレクトリを作らない
            version="",  # バージョンディレクトリを作らない
            default_hp_metric=False  # hparams.yamlを作らない
        )
    else:
        logger = True

    # 新規学習
    model = get_model(cfg.model, cfg.optim)
    data = get_data(cfg.data)
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule=data)
        
if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    
    # コマンドライン引数でチェックポイントが指定されているか確認
    # 使い方: python main.py --resume outputs/2025-10-24/01-24-02/checkpoints/last.ckpt
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        if len(sys.argv) < 3:
            print("エラー: チェックポイントのパスを指定してください")
            print("使い方: python main.py --resume <checkpoint_path>")
            sys.exit(1)
        checkpoint_path = sys.argv[2]
        checkpoint_path = os.path.join(checkpoint_path, "checkpoints", "last.ckpt") if not checkpoint_path.endswith(".ckpt") else checkpoint_path
        train_from_checkpoint(checkpoint_path)
    else:
        # 新規学習
        main()
