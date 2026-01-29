import json

def restructure_data_and_add_qa(existing_file, qa_file, output_file):
    """
    重组数据并添加QA部分
    
    :param existing_file: 已有整合数据文件路径
    :param qa_file: QA数据文件路径
    :param output_file: 输出文件路径
    """
    
    print("开始重组数据并添加QA...")
    print(f"已有数据文件: {existing_file}")
    print(f"QA数据文件: {qa_file}")
    print("=" * 50)
    
    # 1. 加载已有数据
    try:
        with open(existing_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        print(f"加载已有数据: {len(existing_data)} 个项目")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {existing_file}")
        return
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析文件 {existing_file} - {e}")
        return
    
    # 2. 加载QA数据
    try:
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        print(f"加载QA数据: 找到 {len(qa_data.get('questions', []))} 个问题")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {qa_file}")
        return
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析文件 {qa_file} - {e}")
        return
    
    # 3. 分离有date的数据和background数据
    dates_data = []
    background_data = []
    
    for item in existing_data:
        if isinstance(item, dict):
            # 有date的数据
            if "date" in item:
                dates_data.append(item)
            # background数据
            elif "background" in item:
                background_data.extend(item["background"])
    
    print(f"\n数据分离完成:")
    print(f"  有date的数据: {len(dates_data)} 条")
    print(f"  background数据: {len(background_data)} 条")
    
    # 4. 提取QA数据（将questions改为qa）
    if "questions" in qa_data:
        qa_content = qa_data["questions"]
        print(f"  从QA文件中提取了 {len(qa_content)} 个问题")
    else:
        print(f"  警告: QA文件中没有questions键，使用整个文件")
        qa_content = qa_data
    
    # 5. 创建新的数据结构
    new_structure = {
        "dates": dates_data,
        "background": background_data,
        "qa": qa_content
    }
    
    # 6. 保存到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_structure, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 50)
    print("重组完成!")
    print(f"最终结构:")
    print(f"  dates: {len(dates_data)} 个日期")
    print(f"  background: {len(background_data)} 个背景信息")
    print(f"  qa: {len(qa_content)} 个问答对")
    print(f"结果已保存到: {output_file}")
    
    # 显示示例
    print(f"\n示例结构:")
    print(f"  第一个日期: {dates_data[0].get('date', 'N/A') if dates_data else '无'}")
    print(f"  第一个背景信息: {background_data[0] if background_data else '无'}")
    print(f"  第一个QA: {qa_content[0] if qa_content and len(qa_content) > 0 else '无'}")

def main():
    """
    主函数：重组数据并添加QA
    """
    # 已有整合数据文件
    existing_file = "sorted_data/final_integrated_data.json"
    
    # QA数据文件
    qa_file = "filtered_data/QA.json"
    
    # 输出文件
    output_file = "sorted_data/final_data.json"
    
    # 执行重组
    restructure_data_and_add_qa(existing_file, qa_file, output_file)

if __name__ == "__main__":
    main()