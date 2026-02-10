#!/usr/bin/env bash

# 根目录，里面每个子目录都是一个 user
ROOT_DIR="raw_data"

for user_dir in "$ROOT_DIR"/*; do
    [ -d "$user_dir" ] || continue

    USER=$(basename "$user_dir")
    echo "=============================="
    echo "开始处理用户: $USER"
    echo "=============================="

    python ./transfer/transfer_1.py --user "$USER"
    python ./transfer/transfer_2.py
    python ./transfer/transfer_2chat.py
    python ./transfer/transfer_2qa.py
    python ./transfer/transfer_3.py
    python ./transfer/transfer_3other.py
    python ./transfer/transfer_3qa.py
    python ./transfer/locomo_transfer.py --sample_id "$USER"

    rm -rf date_normalized_data filtered_data sorted_data

    echo "完成用户: $USER"
done

echo "🎉 所有用户处理完成"

# ===== 文件复制到指定目录（保留源文件） =====

SRC_FILE="./dataset/our.json"

DEST_DIR="MemOS/evaluation/data/our"
mkdir -p "$DEST_DIR"
cp -f "$SRC_FILE" "$DEST_DIR/"

echo "文件已复制到 $DEST_DIR 并覆盖同名文件（如存在）"

DEST_DIR="Hindsight/benchmarks/our/datasets"
mkdir -p "$DEST_DIR"
cp -f "$SRC_FILE" "$DEST_DIR/"

echo "文件已复制到 $DEST_DIR 并覆盖同名文件（如存在）"

DEST_DIR="MemU/data"
mkdir -p "$DEST_DIR"
cp -f "$SRC_FILE" "$DEST_DIR/"

echo "文件已复制到 $DEST_DIR 并覆盖同名文件（如存在）"