import json
from pathlib import Path


PROTAGONIST = "徐静"
OTHER_SPEAKER = PROTAGONIST + "的Assistant"


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
                        processed_info = f"我用手机打电话说：" + se.replace(PROTAGONIST, "").replace("我", "")
                    elif "note" in ev["id"]:
                        processed_info = f"我用手机笔记" + se.replace(PROTAGONIST, "").replace("我", "")
                    elif "photo" in ev["id"]:
                        processed_info = f"我用手机相机" + se.replace(PROTAGONIST, "").replace("我", "")
                    elif "push" in ev['id']:
                        processed_info = f"我收到推送通知" + se.replace(PROTAGONIST, "").replace("我", "")
                    elif "sms" in ev['id']:
                        try:
                            processed_info = f"我用手机短信{ev["message_type"]}了：" + se
                        except KeyError:
                            processed_info = f"我用手机短信功能："

                    conversation[session_key].append({
                        "speaker": PROTAGONIST,  # "user" or "assistant"
                        "dia_id": f"{date_block['date']}_{ev['id']}_{t_idx}",
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

    sample_id = Path(input_path).stem
    locomo_data = convert_to_locomo(src, sample_id)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(locomo_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main("sorted_data/final_data.json", "our.json")
