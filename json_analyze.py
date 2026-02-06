import json

def print_json_structure(data, indent=0, name="root"):
    prefix = "│   " * indent
    connector = "├── "

    if isinstance(data, dict):
        print(f"{prefix}{connector}{name} (dict)")
        for key, value in data.items():
            print_json_structure(value, indent + 1, key)

    elif isinstance(data, list):
        print(f"{prefix}{connector}{name} (list)")
        if data:  # 只展示第一个元素的结构
            print_json_structure(data[0], indent + 1, "[0]")
        else:
            print(f"{prefix}│   └── (empty list)")

    else:
        print(f"{prefix}{connector}{name} ({type(data).__name__})")


def show_json_structure(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print_json_structure(data)


if __name__ == "__main__":
    # json_file = "./filtered_data/single_hop_qa.json"  # 改成你的 json 文件路径
    # json_file = "./raw_data/phone_data/agent_chat.json"  # 改成你的 json 文件路径
    # json_file = "./dataset/locomo10.json"  # 改成你的 json 文件路径
    # json_file = "./filtered_data/agent_chat.json"  # 改成你的 json 文件路径
    # json_file = "./filtered_data/calendar.json"  # 改成你的 json 文件路径
    # json_file = "./filtered_data/sms.json"  # 改成你的 json 文件路径
    # json_file = "./sorted_data/main_events.json"  # 改成你的 json 文件路径
    # json_file = "./filtered_data/contact.json"  # 改成你的 json 文件路径
    # json_file = "./filtered_data/health.json"  # 改成你的 json 文件路径
    # json_file = "./sorted_data/final_data.json"  # 改成你的 json 文件路径
    json_file = "./雷明轩 memu.json"  # 改成你的 json 文件路径
    show_json_structure(json_file)
