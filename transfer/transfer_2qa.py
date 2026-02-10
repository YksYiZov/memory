import json


def options_to_string(options):
    return " ".join(
        f"{item['option']}: {item['content']}."
        for item in options
        if 'option' in item and 'content' in item
    )

def process_json(input_file, output_file):
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_questions = []
    
    # 处理每个问题
    for question in data:
        # 创建新的问题字典，只保留需要的键
        new_question = {}
        
        # 保留question, answer, score_points
        for key in ['question', 'answer', 'score_points']:
            if key in question:
                new_question[key] = question[key]
        

        if "options" in question:
            new_question["question"] += options_to_string(question["options"])

        # 处理evidence
        if 'evidence' in question:
            # 提取所有phone_id的值到列表中
            phone_ids = []
            for evidence_item in question['evidence']:
                if isinstance(evidence_item, dict) and 'id' in evidence_item:
                    phone_ids.append(str(evidence_item['type']) + str(evidence_item['id']))
            new_question['evidence'] = phone_ids
        
        processed_questions.append(new_question)

        category_dict = {"Information Extraction": "0",
                         "Multi-hop reasoning": "1",
                         "Nondeclarative": "2",
                         "Temporal and Knowledge Updating": "3",
                         "Unanswerable": "4"}

        if 'question_type' in question:
            new_question["category"] = category_dict[question["question_type"]]
        
    # 创建新的数据结构
    result = {'questions': processed_questions}
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，结果已保存到 {output_file}")

# 使用示例
if __name__ == "__main__":
    input_file = "date_normalized_data/QA.json"  # 输入文件名
    output_file = "filtered_data/QA.json"  # 输出文件名
    
    # 调用处理函数
    process_json(input_file, output_file)