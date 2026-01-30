import os
import json
import shutil

def deep_delete_keys(obj, keys_to_delete):
    """
    使用深度优先搜索递归删除指定键
    :param obj: 要处理的数据结构（字典或列表）
    :param keys_to_delete: 要删除的键列表
    :return: 处理后的数据结构
    """
    if isinstance(obj, dict):
        # 创建新字典，深度处理每个值
        new_dict = {}
        for key, value in obj.items():
            # 如果键在删除列表中，跳过
            if key in keys_to_delete:
                continue
            # 递归处理子结构
            new_dict[key] = deep_delete_keys(value, keys_to_delete)
        return new_dict
    elif isinstance(obj, list):
        # 处理列表中的每个元素
        return [deep_delete_keys(item, keys_to_delete) for item in obj]
    else:
        # 基本数据类型直接返回
        return obj

def process_json_file(input_path, output_dir, delete_keys):
    """
    处理单个JSON文件
    :param input_path: 输入文件路径
    :param output_dir: 输出目录
    :param delete_keys: 要删除的键列表
    """
    try:
        # 读取JSON文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 深度删除指定键
        processed_data = deep_delete_keys(data, delete_keys)
        
        # 创建输出目录结构
        relative_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 写入处理后的数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 处理完成: {input_path} -> {output_path}")
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON解析错误: {input_path} - {e}")
    except Exception as e:
        print(f"✗ 处理失败: {input_path} - {e}")

def process_directory(input_dir, output_dir, delete_keys_map):
    """
    处理整个目录下的JSON文件
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    :param delete_keys_map: 文件名到删除键列表的映射
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计信息
    total_files = 0
    processed_files = 0
    error_files = 0
    
    # 遍历目录树
    for root, dirs, files in os.walk(input_dir):
        # 过滤掉以点开头的隐藏文件和目录
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if not file.endswith('.json'):
                continue
            
            total_files += 1
            file_path = os.path.join(root, file)
            
            # 检查文件是否在映射表中
            if file in delete_keys_map:
                delete_keys = delete_keys_map[file]
                process_json_file(file_path, output_dir, delete_keys)
                processed_files += 1
            else:
                # 如果文件不在映射表中，直接跳过
                continue
    
    print("\n" + "="*50)
    print(f"处理完成!")
    print(f"总文件数: {total_files}")
    print(f"处理文件数: {processed_files}")
    print(f"跳过文件数: {total_files - processed_files}")
    print(f"错误文件数: {error_files}")

def advanced_deep_delete_keys(obj, keys_to_delete, current_path=""):
    """
    高级版本：提供更详细的处理信息和路径跟踪
    :param obj: 要处理的数据结构
    :param keys_to_delete: 要删除的键列表
    :param current_path: 当前路径（用于调试）
    :return: 处理后的数据结构，删除的键统计
    """
    deleted_keys = []
    
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            new_path = f"{current_path}.{key}" if current_path else key
            
            # 检查是否要删除此键
            if key in keys_to_delete:
                deleted_keys.append(new_path)
                continue
            
            # 递归处理子结构
            processed_value, sub_deleted = advanced_deep_delete_keys(
                value, keys_to_delete, new_path
            )
            new_dict[key] = processed_value
            deleted_keys.extend(sub_deleted)
        
        return new_dict, deleted_keys
    
    elif isinstance(obj, list):
        new_list = []
        for i, item in enumerate(obj):
            new_path = f"{current_path}[{i}]" if current_path else f"[{i}]"
            processed_item, sub_deleted = advanced_deep_delete_keys(
                item, keys_to_delete, new_path
            )
            new_list.append(processed_item)
            deleted_keys.extend(sub_deleted)
        
        return new_list, deleted_keys
    
    else:
        return obj, deleted_keys

def process_json_file_with_stats(input_path, output_dir, delete_keys):
    """
    带统计信息的文件处理
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 深度删除指定键并获取统计信息
        processed_data, deleted_paths = advanced_deep_delete_keys(data, delete_keys)
        
        # 创建输出目录结构
        relative_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 写入处理后的数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 处理完成: {input_path}")
        print(f"  删除了 {len(deleted_paths)} 个键")
        if deleted_paths and len(deleted_paths) <= 10:  # 只显示前10个删除的键
            for path in deleted_paths[:10]:
                print(f"    - {path}")
        elif deleted_paths:
            print(f"    ... 以及 {len(deleted_paths) - 10} 个其他键")
        
        return len(deleted_paths)
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON解析错误: {input_path} - {e}")
        return 0
    except Exception as e:
        print(f"✗ 处理失败: {input_path} - {e}")
        return 0
    
# ==========================
# 批量处理目录下 JSON 文件
# ==========================
if __name__ == "__main__":
    input_dir = "date_normalized_data"       # JSON 文件目录
    output_dir = "filtered_data"             # 输出目录
    
    # 针对不同 JSON 文件单独设置要删除的键
    delete_keys_map = {
        "calendar.json": ["title", "description", "start_time", "end_time", "event_id"],
        "call.json": ["call_type", "event_id", "contact_phone_number"],
        "contact.json": ["phone_id", "event_id", "gender", "phoneNumber", "personalEmail", "workEmail", "idNumber"],
        "fitness_health.json": ["phone_id", "城市", "日常活动", "步数", "距离", "热量", "锻炼时长", "活动小时数", "跑步", "运动类型", "运动时间", "天气", "距离统计", "平均心率", "平均步频", "累计爬升", "累计下降", "平均配速", "最佳配速", "总步数", "消耗热量", "骑行", "平均速度", "平均踏频", "平均功率", "最佳速度", "最大踏频", "步行", "步数统计", "睡眠", "入睡时间", "出睡时间", "全部睡眠时长", "浅睡时长", "深睡时长", "快速眼动时长", "清醒时长", "清醒次数", "深睡连续性得分", "睡眠得分", "零星小睡时长", "心率统计", "平均静息心率", "心率变异性", "体温统计", "平均体温", "血糖统计", "平均血糖水平", "体重", "压力", "压力得分", "饮食记录", "摄入热量", "用户交互事件", "时间", "描述", "event_id"],
        "note.json": ["title", "content", "event_id"],
        "photo.json": ["imageTag", "faceRecognition", "event_id", "", "caption", "title", "datetime", "location", "province", "city", "district", "streetName", "streetNumber", "poi", "faceRecognition", "imageTag", "ocrText", "shoot_mode", "image_size"],
        "push.json": ["title", "content", "source", "push_status", "jump_path", "event_id"],
        "sms.json": ["contactName", "contact_phone_number", "event_id"],
        # 你可以继续添加其他文件和它们的删除列表
    }
    
    process_directory(input_dir, output_dir, delete_keys_map)