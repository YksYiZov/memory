import json
import re

def process_conversation_json(input_file, output_file):
    """
    处理对话JSON文件：
    1. 外层只保留date, phone_id和conversation
    2. 将conversation转换为列表，每个项是一个字典，包含"user"或"assistant"键和对应的content值
    3. 删除turn X层级，按顺序展开对话
    """
    
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理每个事件
    processed_events = []
    
    for event in data:
        # 创建新的事件字典，只保留需要的键
        new_event = {}
        
        # 保留date和phone_id
        for key in ['date', 'phone_id', 'type']:
            if key in event:
                new_event[key] = event[key]
        
        # 处理conversation
        if 'conversation' in event:
            conversation = event['conversation']
            processed_conversation = []
            
            # 提取所有turn键并按数字顺序排序
            turn_keys = [key for key in conversation.keys() if key.startswith('turn ')]
            
            # 按turn数字排序
            turn_keys.sort(key=lambda x: int(x.split()[1]) if x.split()[1].isdigit() else 0)
            
            # 处理每个turn
            for turn_key in turn_keys:
                turn_data = conversation[turn_key]
                
                # 处理user部分
                if 'user' in turn_data:
                    user_content = turn_data['user'].get('content', '')
                    if user_content:  # 只添加非空内容
                        processed_conversation.append({"user": user_content})
                
                # 处理assistant部分
                if 'assistant' in turn_data:
                    assistant_content = turn_data['assistant'].get('content', '')
                    if assistant_content:  # 只添加非空内容
                        processed_conversation.append({"assistant": assistant_content})
            
            new_event['conversation'] = processed_conversation
        
        processed_events.append(new_event)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_events, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，结果已保存到 {output_file}")
    print(f"处理了 {len(processed_events)} 个事件")

def process_conversation_json_with_details(input_file, output_file):
    """
    增强版本：包含更多细节和错误处理
    """
    
    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("错误：JSON根元素应该是一个列表")
            return
        
        processed_events = []
        
        for i, event in enumerate(data):
            # 跳过非字典元素
            if not isinstance(event, dict):
                print(f"警告：跳过第{i}个元素，因为它不是字典")
                continue
            
            # 创建新的事件字典
            new_event = {}
            
            # 提取date和phone_id
            for key in ['date', 'phone_id']:
                if key in event:
                    new_event[key] = event[key]
                else:
                    print(f"警告：事件{i}缺少键'{key}'")
            
            # 处理conversation
            if 'conversation' in event:
                conversation = event['conversation']
                
                if not isinstance(conversation, dict):
                    print(f"警告：事件{i}的conversation不是字典，跳过处理")
                    new_event['conversation'] = []
                else:
                    processed_conversation = []
                    
                    # 提取并排序turn键
                    turn_keys = []
                    for key in conversation.keys():
                        # 使用正则表达式匹配"turn X"格式
                        match = re.match(r'turn\s+(\d+)', key, re.IGNORECASE)
                        if match:
                            turn_keys.append((int(match.group(1)), key))
                    
                    # 按数字排序
                    turn_keys.sort(key=lambda x: x[0])
                    
                    # 处理每个turn
                    for turn_num, turn_key in turn_keys:
                        turn_data = conversation[turn_key]
                        
                        if not isinstance(turn_data, dict):
                            print(f"警告：事件{i}的{turn_key}不是字典，跳过")
                            continue
                        
                        # 处理user
                        if 'user' in turn_data:
                            user_data = turn_data['user']
                            if isinstance(user_data, dict) and 'content' in user_data:
                                user_content = user_data['content']
                                if user_content and str(user_content).strip():
                                    processed_conversation.append({"user": str(user_content).strip()})
                        
                        # 处理assistant
                        if 'assistant' in turn_data:
                            assistant_data = turn_data['assistant']
                            if isinstance(assistant_data, dict) and 'content' in assistant_data:
                                assistant_content = assistant_data['content']
                                if assistant_content and str(assistant_content).strip():
                                    processed_conversation.append({"assistant": str(assistant_content).strip()})
                    
                    new_event['conversation'] = processed_conversation
            else:
                new_event['conversation'] = []
            
            processed_events.append(new_event)
        
        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_events, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成，结果已保存到 {output_file}")
        print(f"处理了 {len(processed_events)} 个事件")
        
        # 统计信息
        total_messages = sum(len(event.get('conversation', [])) for event in processed_events)
        print(f"总共提取了 {total_messages} 条消息")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except json.JSONDecodeError as e:
        print(f"错误：JSON解析失败 - {e}")

# 使用示例
if __name__ == "__main__":
    # 简单版本
    input_file = "date_normalized_data/agent_chat.json"
    output_file = "filtered_data/agent_chat.json"
    process_conversation_json(input_file, output_file)
    