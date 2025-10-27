#!/bin/bash
# チェックポイントから学習を再開するスクリプト

# スクリプトのディレクトリに移動
SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR

# 使い方例:
# 1. 新規学習の場合
#    bash train_resume.sh
#    bash train_resume.sh <実験名>
#    例: bash train_resume.sh baseline_v1

# 2. チェックポイントから再開する場合
#    bash train_resume.sh --resume <checkpoint_path>
#    例: bash train_resume.sh --resume outputs/2025-10-24/baseline_v1_12-34-56/checkpoints/last.ckpt

if [ "$1" = "--resume" ]; then
    # チェックポイントから再開
    CHECKPOINT_PATH=${2}
    if [ -z "$CHECKPOINT_PATH" ]; then
        echo "エラー: チェックポイントのパスを指定してください"
        echo "使い方: bash train_resume.sh --resume <checkpoint_path>"
        exit 1
    fi
    echo "チェックポイントから学習を再開します: $CHECKPOINT_PATH"
    python main.py --resume "$CHECKPOINT_PATH"
else
    # 新規学習
    EXP_NAME=${1:-"default"}
    echo "新規学習を開始します (実験名: $EXP_NAME)"
    python main.py exp_name="$EXP_NAME"
fi
