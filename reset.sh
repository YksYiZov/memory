#!/bin/sh

set -e

TARGETS="
MemOS/evaluation/data/our/our.json
Hindsight/benchmarks/our/datasets/our.json
MemU/data/our.json
MemOS/evaluation/results/our-memos
Hindsight/benchmarks/our/results
MemU/memory
MemU/enhanced_memory_test_results.json
"

for path in $TARGETS; do
    if [ -f "$path" ]; then
        echo "删除文件: $path"
        rm -f "$path"

    elif [ -d "$path" ]; then
        echo "清空目录: $path"
        rm -rf "$path"/* "$path"/.[!.]* "$path"/..?* 2>/dev/null || true

    else
        echo "路径不存在，跳过: $path"
    fi
done

echo "清理完成 ✅"
