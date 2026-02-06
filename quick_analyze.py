import json
from typing import Dict

category_dict = {"single_hop": "0", "mutihop": "1", "reasoning": "2", "temporal": "3", "user_modeling": "4", "unanswerable": "5"}
def load_json(path: str) -> Dict:
    """读取 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_results_method_a(data: Dict) -> Dict:
    """
    假设 method A 的结果已经在 json 中统计完成
    你只需要在这里指定字段路径
    """
    return {
        "overall_accuracy": {"total": data["total_questions"], 
                             "acc": data["accuracy"]},
        "per_category_accuracy": {k: {"total": data["metadata"]["category_accuracies"][v]["total"], "acc": data["metadata"]["category_accuracies"][v]["mean"]} for k, v in category_dict.items() if v in data["metadata"]["category_accuracies"]}
    }


def extract_results_method_b(data: Dict) -> Dict:
    """
    如果字段名不同，在这里做一次映射
    """
    category_accuracy = {}
    for user in data["item_results"]:
        for k, v in category_dict.items():
            if v in user["metrics"]["category_stats"]:
                if k not in category_accuracy.keys():
                    category_accuracy[k] = {"total": user["metrics"]["category_stats"][v]["total"],
                                              "acc": user["metrics"]["category_stats"][v]["correct"]}
                else:
                    category_accuracy[k]["total"] += user["metrics"]["category_stats"][v]["total"]
                    category_accuracy[k]["acc"] +=user["metrics"]["category_stats"][v]["correct"]
    
    for k, v in category_accuracy.items():
        v["acc"] /= v["total"]

    return {
        "overall_accuracy": {"total": data["total_questions"], 
                             "acc": data["overall_accuracy"]},
        "per_category_accuracy": category_accuracy
    }


def extract_results_method_c(data: Dict) -> Dict:
    """
    method C 的结果结构
    """
    return {
        "overall_accuracy": {"total": data["summary"]["total_questions"], 
                             "acc": data["summary"]["total_correct"]},
        "per_category_accuracy": {k: {"total": data["summary"]["category_stats"][v]["total"], "acc": data["summary"]["category_stats"][v]["correct"]/data["summary"]["category_stats"][v]["total"]} for k, v in category_dict.items() if v in data["summary"]["category_stats"]}
    }


# =========================
# 汇总逻辑（只拼 JSON）
# =========================

def main(json_a, json_b, json_c, output_path):
    data_a = load_json(json_a)
    data_b = load_json(json_b)
    data_c = load_json(json_c)

    final_results = {
        "method_a": extract_results_method_a(data_a),
        "method_b": extract_results_method_b(data_b),
        "method_c": extract_results_method_c(data_c),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"结果汇总完成，已保存至 {output_path}")


if __name__ == "__main__":
    json_a = "MemOS/evaluation/results/our-memos/eval_results.json"
    json_b = "hindsight/benchmarks/our/results/benchmark_results.json"
    json_c = "MemU/enhanced_memory_test_results.json"
    output = "./result.json"

    main(json_a, json_b, json_c, output)
