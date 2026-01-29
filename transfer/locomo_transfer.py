import json
import argparse
from pathlib import Path


def convert_to_locomo(src: dict, sample_id: str) -> list:
    locomo = {}

    # ========== QA ==========
    locomo["qa"] = []
    for qa in src.get("qa", []):
        locomo["qa"].append({
            "question": qa["question"],
            "answer": qa["answer"],
            "evidence": qa.get("evidence", []),
            "category": qa["category"]
        })

    # ========== CONVERSATION ==========
    conversation = {
        "speaker_a": PROTAGONIST,
        "speaker_b": OTHER_SPEAKER
    }

    event_summary = {}
    observation = {}
    session_summary = {}

    for idx, date_block in enumerate(src.get("dates", []), start=1):
        session_key = f"session_{idx}"
        date_key = f"{session_key}_date_time"

        conversation[date_key] = date_block["date"]
        conversation[session_key] = []

        # ----- conversation -----
        for ev in date_block.get("events", []):
            for t_idx, turn in enumerate(ev.get("conversation", [])):
                if not isinstance(turn, dict) or len(turn) != 1:
                    continue  # 防御式：跳过异常结构

                speaker, text = next(iter(turn.items()))

                conversation[session_key].append({
                    "speaker": PROTAGONIST if speaker == "user" else OTHER_SPEAKER,  # "user" or "assistant"
                    "dia_id": f"{date_block['date']}_{ev['id']}_{t_idx}",
                    "text": text
                })

        for ev in date_block.get("events", []):
            sub_events = ev.get("sub_events", [])
            if sub_events:  # 只处理非空的 sub_events
                for se in sub_events:
                    # ----- 用户可自定义处理 -----
                    # 这里你可以根据 ev['id'] 判断要不要加工 se，比如加上第一人称、描述性修饰等
                    if "calendar" in ev['id']:
                        processed_info = se
                    elif "call" in ev['id']:
                        processed_info = f"我用手机打电话说："
                    elif "note" in ev["id"]:
                        processed_info = f"我用手机笔记："
                    elif "photo" in ev["id"]:
                        processed_info = f"我用手机相机："
                    elif "push" in ev['id']:
                        processed_info = f"我收到推送通知："
                    elif "sms" in ev['id']:
                        try:
                            processed_info = f"我用手机短信{ev['message_type']}了：" + se
                        except KeyError:
                            processed_info = f"我用手机短信功能："

                    conversation[session_key].append({
                        "speaker": PROTAGONIST,  # "user" or "assistant"
                        "dia_id": f"{date_block['date']}_{ev['id']}",
                        "text": processed_info
                    })
        # ----- event summary -----
        event_summary[f"events_{session_key}"] = {
        PROTAGONIST: [info for info in date_block.get("extra_info", [])],
        "date": date_block["date"]
        }

        # ----- observation -----
        observation[f"{session_key}_observation"] = {
            PROTAGONIST: [],
        }

        # ----- session summary -----
        session_summary[f"{session_key}_summary"] = ""

    locomo["conversation"] = conversation
    locomo["event_summary"] = event_summary
    locomo["observation"] = observation
    locomo["session_summary"] = session_summary
    locomo["sample_id"] = sample_id

    return [locomo]


def main(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        src = json.load(f)

    sample_id = PROTAGONIST
    locomo_data = convert_to_locomo(src, sample_id)  # 这是一个 list，通常长度为 1

    output_file = Path(output_path)

    # ===== 如果 our.json 不存在，直接保存 =====
    if not output_file.exists():
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(locomo_data, f, ensure_ascii=False, indent=2)
        print(f"已创建新文件并保存 sample_id={sample_id}至{output_file}")
        return

    # ===== 如果 our.json 已存在，读取并检查 =====
    with open(output_file, "r", encoding="utf-8") as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []

    # 防御：确保是 list
    if not isinstance(existing_data, list):
        existing_data = []

    # 检查是否已有相同 sample_id
    existing_ids = {
        item.get("sample_id")
        for item in existing_data
        if isinstance(item, dict)
    }

    if sample_id in existing_ids:
        print(f"该用户已存在（sample_id={sample_id}），未进行保存")
        return

    # ===== 合并并保存 =====
    existing_data.extend(locomo_data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    print(f"已成功追加 sample_id={sample_id}至{output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="LoCoMo 数据转换与去重保存工具"
    )

    parser.add_argument(
        "--input",
        required=False,
        help="输入的原始 JSON 文件路径",
        default="sorted_data/final_data.json"
    )

    parser.add_argument(
        "--output",
        required=False,
        help="输出的 our.json 路径",
        default="./dataset/our.json"
    )

    parser.add_argument(
        "--sample_id", "-s",
        required=True,
        help="样本 ID（默认使用 PROTAGONIST）"
    )

    args = parser.parse_args()

    PROTAGONIST = args.sample_id
    OTHER_SPEAKER = PROTAGONIST + "的Assistant"

    main(args.input, args.output)
