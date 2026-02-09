# 出错即停止（等价于 set -e 的“严格模式”）
$ErrorActionPreference = "Stop"

$TARGETS = @(
    "MemOS/evaluation/data/our/our.json",
    "Hindsight/benchmarks/our/datasets/our.json",
    "MemU/data/our.json",
    "MemOS/evaluation/results/our-memos",
    "hindsight/benchmarks/our/results",
    "MemU/memory",
    "MemU/enhanced_memory_test_results.json"
)

foreach ($path in $TARGETS) {

    if (Test-Path $path -PathType Leaf) {
        Write-Host "删除文件: $path"
        Remove-Item -Force $path

    } elseif (Test-Path $path -PathType Container) {
        Write-Host "清空目录: $path"

        # 只清空目录内容，不删除目录本身
        Get-ChildItem -Path $path -Force -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    } else {
        Write-Host "路径不存在，跳过: $path"
    }
}

Write-Host "清理完成 ✅"
