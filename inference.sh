#!/bin/bash

# --- 引数の数による分岐 ---

# 1. $1がない場合 (引数0個): 必須引数チェック
if [ $# -eq 0 ]; then
    echo "❌ エラー: configファイル or 実験名を指定してください"
    echo "使用方法: $0 <config_file> [checkpoint] [output_dir]"
    exit 1

# 2. $1のみが渡された場合 (引数1個): $2, $3はなし
elif [ $# -eq 1 ]; then
    echo "✅ configファイル ($1) のみを使用して実行します。"
    
    python inference.py \
    --config "$1" \
    --device cuda \
    --split test \
    --max-samples 32

# 3. $1, $2が渡された場合 (引数2個): $3はなし
elif [ $# -eq 2 ]; then
    echo "✅ config ($1) と checkpoint ($2) を使用して実行します。"
    
    python inference.py \
    --config "$1" \
    --checkpoint "$2" \
    --device cuda \
    --split test \
    --max-samples 32

# 4. $1, $2, $3が渡された場合 (引数3個以上): 全て指定
else [ $# -ge 3 ]
    echo "✅ config ($1)、checkpoint ($2)、出力ディレクトリ ($3) を使用して実行します。"
    
    python inference.py \
    --config "$1" \
    --checkpoint "$2" \
    --device cuda \
    --split test \
    --max-samples 32 \
    --output-dir "$3"
fi