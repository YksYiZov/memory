import os
import json
import argparse
from datetime import datetime

# 输入目录和输出目录
parser = argparse.ArgumentParser(
    description="LoCoMo 第一重转换"
)

parser.add_argument(
    "--user", "-u",
    required=True,
    help="输入用户名",
)

args = parser.parse_args()

user = args.user
input_dir = f"./raw_data/{user}"   # 这里换成你的原json目录
output_dir = "./date_normalized_data" # 输出目录
os.makedirs(output_dir, exist_ok=True)

# 指定需要处理的字段名列表
target_fields = ["date", "datetime", "日期", "ask_date"]  # 这里换成你要处理的字段名

def process_value(value):
    """将日期/时间字符串统一为 YYYY-MM-DD"""
    if not isinstance(value, str):
        return value
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value  # 非日期格式保持不变

def rename_fields(obj):
    """
    递归搜索json对象，处理目标字段：
    - 字段名被修改为 'date'
    - 字段值统一为 YYYY-MM-DD
    返回两个值：
        1. 处理后的对象
        2. 是否找到了目标字段
    """
    found = False

    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            # 递归处理子对象
            processed_value, child_found = rename_fields(v)
            if k in target_fields:
                new_obj["date"] = process_value(processed_value)
                found = True
            else:
                new_obj[k] = processed_value
            if child_found:
                found = True
        return new_obj, found

    elif isinstance(obj, list):
        new_list = []
        for item in obj:
            processed_item, item_found = rename_fields(item)
            new_list.append(processed_item)
            if item_found:
                found = True
        return new_list, found

    else:
        return obj, False

# 遍历输入目录
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        new_data, has_target = rename_fields(data)

        # 如果没有找到指定字段，就输出原数据
        final_data = new_data if has_target else data

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

print("处理完成，结果已保存到:", output_dir)