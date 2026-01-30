import os
import json

def integrate_additional_data(existing_file, date_file, no_date_file, output_file):
    """
    整合额外数据到已有事件结构中
    
    :param existing_file: 已有事件整合文件路径
    :param date_file: 有date字段的文件路径
    :param no_date_file: 没有date字段的文件路径
    :param output_file: 输出文件路径
    """
    
    print("开始整合额外数据...")
    print(f"已有事件文件: {existing_file}")
    print(f"有date的文件: {date_file}")
    print(f"没有date的文件: {no_date_file}")
    print("=" * 50)
    
    # 1. 加载已有数据
    try:
        with open(existing_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        print(f"加载已有数据: {len(existing_data)} 个项目")
    except FileNotFoundError:
        print(f"警告: 已有事件文件 {existing_file} 不存在，创建空列表")
        existing_data = []
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析已有事件文件 {existing_file} - {e}")
        return
    
    # 2. 加载有date的数据
    try:
        with open(date_file, 'r', encoding='utf-8') as f:
            date_data = json.load(f)
        print(f"加载有date数据: {len(date_data)} 个项目")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {date_file}")
        return
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析文件 {date_file} - {e}")
        return
    
    # 3. 加载没有date的数据
    try:
        with open(no_date_file, 'r', encoding='utf-8') as f:
            no_date_data = json.load(f)
        print(f"加载没有date的数据: {len(no_date_data)} 个项目")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {no_date_file}")
        return
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析文件 {no_date_file} - {e}")
        return
    
    # 4. 处理有date的数据 - 按日期分组
    date_to_info = {}
    for item in date_data:
        if isinstance(item, dict) and "date" in item and "summarized_info" in item:
            date = item["date"]
            info = item["summarized_info"]
            
            if date not in date_to_info:
                date_to_info[date] = []
            date_to_info[date].append(info)
    
    print(f"\n处理有date的数据:")
    print(f"  提取了 {sum(len(v) for v in date_to_info.values())} 条信息")
    print(f"  分布在 {len(date_to_info)} 个日期")
    
    # 5. 处理没有date的数据 - 直接收集所有字典
    background_items = []
    for item in no_date_data:
        if isinstance(item, dict) and "date" not in item:
            background_items.append(item)
    
    print(f"\n处理没有date的数据:")
    print(f"  提取了 {len(background_items)} 个字典")
    
    # 6. 将额外数据整合到已有数据中
    # 6.1 添加extra_info到对应日期
    for date, info_list in date_to_info.items():
        # 查找是否已有该日期的项目
        found = False
        for item in existing_data:
            if isinstance(item, dict) and item.get("date") == date:
                if "extra_info" not in item:
                    item["extra_info"] = []
                item["extra_info"].extend(info_list)
                found = True
                break
        
        # 如果没有找到，创建新的日期项目
        if not found:
            new_item = {
                "date": date,
                "events": [],
                "extra_info": info_list
            }
            existing_data.append(new_item)
    
    # 6.2 添加background
    if background_items:
        # 查找是否已有background项目
        found = False
        for item in existing_data:
            if isinstance(item, dict) and "background" in item and "date" not in item:
                item["background"].extend(background_items)
                found = True
                break
        
        # 如果没有找到，创建新的background项目
        if not found:
            background_item = {
                "background": background_items
            }
            existing_data.append(background_item)
    
    # 7. 按日期排序
    dated_items = [item for item in existing_data if isinstance(item, dict) and "date" in item]
    non_dated_items = [item for item in existing_data if not (isinstance(item, dict) and "date" in item)]
    
    dated_items.sort(key=lambda x: x["date"])
    final_data = dated_items + non_dated_items
    
    # 8. 保存到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 50)
    print("整合完成!")
    print(f"最终数据包含 {len(final_data)} 个项目")
    print(f"  有日期的项目: {len(dated_items)}")
    print(f"  没有日期的项目: {len(non_dated_items)}")
    
    # 显示统计信息
    extra_info_count = sum(len(item.get("extra_info", [])) for item in final_data if isinstance(item, dict))
    background_count = sum(len(item.get("background", [])) for item in final_data if isinstance(item, dict))
    
    print(f"\n统计信息:")
    print(f"  extra_info总数: {extra_info_count}")
    print(f"  background字典数: {background_count}")
    print(f"结果已保存到: {output_file}")

# ==========================
# 主程序
# ==========================
if __name__ == "__main__":
    # 已有事件文件（上一步生成）
    existing_file = "sorted_data/main_events.json"
    
    # 有date字段的文件
    date_file = "filtered_data/fitness_health.json"
    
    # 没有date字段的文件
    no_date_file = "filtered_data/contact.json"
    
    # 输出文件
    output_file = "sorted_data/final_integrated_data.json"
    
    # 执行整合
    integrate_additional_data(existing_file, date_file, no_date_file, output_file)