# ===============================
# 根目录，里面每个子目录都是一个 user
# ===============================
$ROOT_DIR = "raw_data"

Get-ChildItem -Path $ROOT_DIR -Directory | ForEach-Object {

    $USER = $_.Name

    Write-Host "=============================="
    Write-Host "开始处理用户: $USER"
    Write-Host "=============================="

    python ./transfer/transfer_1.py --user $USER
    python ./transfer/transfer_2.py
    python ./transfer/transfer_2chat.py
    python ./transfer/transfer_2qa.py
    python ./transfer/transfer_3.py
    python ./transfer/transfer_3other.py
    python ./transfer/transfer_3qa.py
    python ./transfer/locomo_transfer.py --sample_id $USER

    # 删除中间目录（如果存在）
    Remove-Item -Recurse -Force `
        date_normalized_data, `
        filtered_data, `
        sorted_data `
        -ErrorAction SilentlyContinue

    Write-Host "完成用户: $USER"
}

Write-Host "🎉 所有用户处理完成"

# ===============================
# 文件复制到指定目录（保留源文件）
# ===============================

$SRC_FILE = "./dataset/our.json"

$DEST_DIRS = @(
    "EverMemOS-main/evaluation/data/our",
    "hindsight/benchmarks/our/datasets",
    "memU-experiment-main/data"
)

foreach ($DEST_DIR in $DEST_DIRS) {
    New-Item -ItemType Directory -Force -Path $DEST_DIR | Out-Null
    Copy-Item -Path $SRC_FILE -Destination $DEST_DIR -Force
    Write-Host "文件已复制到 $DEST_DIR 并覆盖同名文件（如存在）"
}
