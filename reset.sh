#!/bin/sh

set -e

TARGETS="
EverMemOS-main/evaluation/data/our/our.json
hindsight/benchmarks/our/datasets/our.json
memU-experiment-main/data/our.json
EverMemOS-main/evaluation/results/our-memos
hindsight/benchmarks/our/results
memU-experiment-main/memory
memU-experiment-main/enhanced_memory_test_results.json
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
