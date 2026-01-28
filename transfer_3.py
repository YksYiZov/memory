import os
from pathlib import Path
import json
from collections import defaultdict
import re

def process_json_file(file_path):
    """
    处理单个JSON文件，根据三种结构提取事件数据
    :param file_path: JSON文件路径
    :return: 提取到的事件数据列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        events = []
        
        # 确保数据是列表格式
        if not isinstance(data, list):
            print(f"警告: {file_path} 的根元素不是列表，跳过")
            return []
        
        for item in data:
            if not isinstance(item, dict):
                continue
                
            # 检查是否为第一种结构: 包含 date, phone_id, conversation
            if "date" in item and "phone_id" in item and "conversation" in item:
                event = {
                    "date": item["date"],
                    "phone_id": item["phone_id"],
                    "type": item["type"],
                    "conversation": item["conversation"],
                    "sub_events": []  # 第一种结构没有额外的summaried_info
                }
                events.append(event)
            
            # 检查是否为第二种结构: 包含 phone_id, date, summarized_info
            elif "phone_id" in item and "date" in item and "summarized_info" in item:
                event = {
                    "date": item["date"],
                    "phone_id": item["phone_id"],
                    "type": item["type"],
                    "conversation": [],  # 第二种结构没有conversation
                    "sub_events": [item["summarized_info"]]  # 将summarized_info添加到sub_events
                }
                events.append(event)
            
            # 检查是否为第三种结构: 包含 phone_id, message_content, date
            elif "phone_id" in item and "date" in item and "message_content" in item:
                event = {
                    "date": item["date"],
                    "phone_id": item["phone_id"],
                    "type": item["type"],
                    "conversation": [],  # 第三种结构没有conversation
                    "sub_events": [item["message_content"]],  # 将message_content添加到sub_events
                    "message_type": item["message_type"]  # 将message_type添加到sub_events
                }
                events.append(event)
        
        return events
    
    except json.JSONDecodeError as e:
        print(f"✗ JSON解析错误: {file_path} - {e}")
        return []
    except Exception as e:
        print(f"✗ 处理失败: {file_path} - {e}")
        return []

def deduplicate_preserve_order_dict(items):
    """
    去重列表中的字典元素，保持顺序
    :param items: 列表，可能包含字典和其他类型
    :return: 去重后的列表
    """
    seen = []
    result = []
    
    for item in items:
        # 如果item是字典，使用json.dumps转换为字符串进行比较
        if isinstance(item, dict):
            item_str = json.dumps(item, sort_keys=True, ensure_ascii=False)
            if item_str not in seen:
                seen.append(item_str)
                result.append(item)
        # 如果是其他可哈希类型，直接比较
        else:
            if item not in seen:
                seen.append(item)
                result.append(item)
    
    return result

def deduplicate_preserve_order(items):
    """
    去重列表中的元素，保持顺序（支持字典去重）
    :param items: 列表
    :return: 去重后的列表
    """
    if not items:
        return items
    
    # 检查第一个元素是否是字典
    if isinstance(items[0], dict):
        return deduplicate_preserve_order_dict(items)
    else:
        # 对于非字典元素，使用传统方法
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

def extract_phone_id_number(phone_id):
    """
    提取phone_id中的数字部分用于排序
    :param phone_id: 事件ID字符串
    :return: 提取到的数字，如果没有数字则返回0
    """
    if isinstance(phone_id, str):
        # 尝试匹配phone_id中的数字部分
        numbers = re.findall(r'\d+', phone_id)
        if numbers:
            # 返回最后一个数字部分（假设phone_id格式如：event_123 或 event-123）
            return int(numbers[-1])
    elif isinstance(phone_id, (int, float)):
        return int(phone_id)
    return 0

def sort_phone_ids(phone_ids):
    """
    对phone_id列表进行排序
    :param phone_ids: phone_id列表
    :return: 排序后的phone_id列表
    """
    # 使用自定义排序：先按提取的数字排序，然后按字符串排序
    return sorted(phone_ids, key=lambda x: (extract_phone_id_number(x), str(x)))

def combine_and_sort_events(all_events):
    """
    组合和排序事件数据
    :param all_events: 所有事件列表
    :return: 排序后的嵌套结构
    """
    # 按date分组
    date_groups = defaultdict(lambda: defaultdict(list))
    
    for event in all_events:
        date = event["date"]
        tid = event["type"] + event["phone_id"]
        
        # 添加到对应的date和phone_id组
        date_groups[date][tid].append(event)
    
    # 按date排序
    sorted_dates = sorted(date_groups.keys())
    
    # 构建结果结构
    result = []
    
    for date in sorted_dates:
        date_events = []
        tid_dict = date_groups[date]
        
        # 对每个phone_id合并事件（按phone_id排序）
        sorted_phone_ids = sort_phone_ids(tid_dict.keys())
        
        for tid in sorted_phone_ids:
            events_with_same_id = tid_dict[tid]
            
            # 合并同一个phone_id的所有事件
            combined_event = {
                "id": tid,
                "conversation": [],
                "sub_events": []
            }
            
            for event in events_with_same_id:
                # 合并conversation
                if "conversation" in event and event["conversation"]:
                    if isinstance(event["conversation"], list):
                        combined_event["conversation"].extend(event["conversation"])
                    else:
                        combined_event["conversation"].append(event["conversation"])
                
                # 合并sub_events
                if "sub_events" in event and event["sub_events"]:
                    if isinstance(event["sub_events"], list):
                        combined_event["sub_events"].extend(event["sub_events"])
                    else:
                        combined_event["sub_events"].append(event["sub_events"])

                if "message_type" in event and event["message_type"]:
                    combined_event["message_type"] = event["message_type"]
                    
            # 去重（保持顺序）
            combined_event["conversation"] = deduplicate_preserve_order(combined_event["conversation"])
            combined_event["sub_events"] = deduplicate_preserve_order(combined_event["sub_events"])
            
            date_events.append(combined_event)
        
        result.append({
            "date": date,
            "events": date_events
        })
    
    return result

def process_target_files(input_dir, target_files, output_file):
    """
    处理目标文件列表中的JSON文件
    :param input_dir: 输入目录
    :param target_files: 目标文件列表
    :param output_file: 输出文件路径
    """
    all_events = []
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    print(f"开始处理目录: {input_dir}")
    print(f"目标文件列表: {target_files}")
    print("=" * 50)
    
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误: 目录 {input_dir} 不存在")
        return
    
    # 遍历目录
    for root, dirs, files in os.walk(input_dir):
        # 过滤隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if not file.endswith('.json'):
                continue
            
            # 检查是否在目标文件列表中
            if file in target_files:
                file_path = os.path.join(root, file)
                print(f"处理: {file_path}")
                
                try:
                    # 处理JSON文件
                    events = process_json_file(file_path)
                    all_events.extend(events)
                    processed_count += 1
                    
                    print(f"  找到 {len(events)} 个事件")
                except Exception as e:
                    print(f"✗ 处理失败: {file_path} - {e}")
                    error_count += 1
            else:
                skipped_count += 1
    
    print("=" * 50)
    print(f"处理完成!")
    print(f"处理文件数: {processed_count}")
    print(f"跳过文件数: {skipped_count}")
    print(f"错误文件数: {error_count}")
    print(f"总共找到 {len(all_events)} 个事件")
    
    if not all_events:
        print("未找到任何事件数据，输出空结果")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return
    
    # 组合和排序事件
    sorted_events = combine_and_sort_events(all_events)
    
    # 写入输出文件

    f_path = Path(output_file)
    f_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_events, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_file}")
    print(f"总共 {len(sorted_events)} 个不同的日期")
    
    # 显示统计信息
    total_events = sum(len(date_item["events"]) for date_item in sorted_events)
    print(f"总共 {total_events} 个phone_id组")
    
    # 显示前几个日期的信息
    for i, date_item in enumerate(sorted_events[:3]):  # 只显示前3个
        print(f"\n日期 {date_item['date']}:")
        for j, event in enumerate(date_item["events"][:3]):  # 每个日期显示前3个event
            print(f"  tid {j+1}: {event['id']}")
            print(f"    conversation条目数: {len(event['conversation'])}")
            print(f"    sub_events条目数: {len(event['sub_events'])}")
        if len(date_item["events"]) > 3:
            print(f"  ... 还有 {len(date_item['events']) - 3} 个tid")

# ==========================
# 主程序
# ==========================
if __name__ == "__main__":
    # 输入目录
    input_dir = "filtered_data"
    
    # 输出文件
    output_file = "sorted_data/main_events.json"
    
    # 目标文件列表
    target_files = [
        "agent_chat.json",
        "calendar.json",
        "note.json",
        "photo.json",
        "push.json",
        "sms.json",
        "call.json"
        # 添加更多需要处理的文件
    ]
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 目录 {input_dir} 不存在")
        exit(1)
    
    process_target_files(input_dir, target_files, output_file)