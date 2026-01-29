import os
import re
from pathlib import Path

# ====== 用户需要填写的配置 ======
MEMOS_LLM_API_KEY = "your_llm_api_key_here"
MEMOS_LLM_MODEL = "your_llm_model_here"
MEMOS_LLM_BASE_URL = "your_llm_base_url_here"
MEMOS_MEMOS_KEY = "your_memos_api_key_here"

# Hindsight
HINDSIGHT_LLM_API_KEY = "your_llm_api_key_here"
HINDSIGHT_LLM_MODEL = "your_llm_model_here"
HINDSIGHT_LLM_BASE_URL = "your_llm_base_url_here"
HINDSIGHT_EMBEDDINGS_API_KEY = "your_embeddings_api_key_here"
HINDSIGHT_EMBEDDINGS_MODEL = "your_embeddings_model_here"
HINDSIGHT_EMBEDDINGS_BASE_URL = "your_embeddings_base_url_here"

# MemU
MEMU_OPENAI_API_KEY = "your_openai_api_key_here"
MEMU_OPENAI_BASE_URL = "your_openai_base_url_here"
MEMU_EMBEDDING_MODEL = "your_embedding_model_here"

NOW_EMBEDDING_MODEL = "your_embedding_model_here" # 除初次使用都需要更新

# # MemOS
# MEMOS_LLM_API_KEY = "sk-cb7d4d3296e145ba9f333c73ce8c1056"
# MEMOS_LLM_MODEL = "qwen3-max"
# MEMOS_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# MEMOS_MEMOS_KEY = "mpg-pLXzikhk2MdSK76tq4yOUIHY8KwwGpqt/5B3SYyT"

# # Hindsight
# HINDSIGHT_LLM_API_KEY = "sk-cb7d4d3296e145ba9f333c73ce8c1056"
# HINDSIGHT_LLM_MODEL = "qwen3-max"
# HINDSIGHT_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# HINDSIGHT_EMBEDDINGS_API_KEY = "sk-cb7d4d3296e145ba9f333c73ce8c1056"
# HINDSIGHT_EMBEDDINGS_MODEL = "text-embedding-v4"
# HINDSIGHT_EMBEDDINGS_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# # MemU
# MEMU_OPENAI_API_KEY = "sk-cb7d4d3296e145ba9f333c73ce8c1056"
# MEMU_OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# MEMU_EMBEDDING_MODEL = "text-embedding-v4"
# ================================

def replace_env_vars(env_file: Path, replacements: dict):
    if not env_file.exists():
        print(f"[WARN] {env_file} 不存在，跳过")
        return
    with open(env_file, "r", encoding="utf-8") as f:
        content = f.read()
    for key, value in replacements.items():
        pattern = rf"^{key}=.*$"
        replacement_line = f"{key}={value}"
        content, count = re.subn(pattern, replacement_line, content, flags=re.MULTILINE)
        if count == 0:
            # key 不存在则追加
            content += f"\n{replacement_line}"
    with open(env_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[INFO] 更新 {env_file}")

def replace_in_file(file_path: Path, pattern: str, replacement: str):
    if not file_path.exists():
        print(f"[WARN] {file_path} 不存在，跳过")
        return
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    content_new, count = re.subn(pattern, replacement, content)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content_new)
    print(f"[INFO] 替换 {count} 次 in {file_path}")

def setup_memos():
    env_file = Path("EverMemOS-main/.env")
    replace_env_vars(env_file, {
        "LLM_API_KEY": MEMOS_LLM_API_KEY,
        "LLM_MODEL": MEMOS_LLM_MODEL,
        "LLM_BASE_URL": MEMOS_LLM_BASE_URL
    })

    memos_yaml = Path("EverMemOS-main/evaluation/config/systems/memos.yaml")
    replace_in_file(memos_yaml, r"api_key: .*", f"api_key: {MEMOS_MEMOS_KEY}")

def setup_hindsight():
    env_file = Path("hindsight/.env")
    replace_env_vars(env_file, {
        "HINDSIGHT_API_LLM_API_KEY": HINDSIGHT_LLM_API_KEY,
        "HINDSIGHT_API_LLM_BASE_URL": HINDSIGHT_LLM_BASE_URL,
        "HINDSIGHT_API_LLM_MODEL": HINDSIGHT_LLM_MODEL,
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY": HINDSIGHT_EMBEDDINGS_API_KEY,
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL": HINDSIGHT_EMBEDDINGS_BASE_URL,
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL": HINDSIGHT_EMBEDDINGS_MODEL
    })

def setup_memu():
    env_file = Path("memU-experiment-main/.env")
    replace_env_vars(env_file, {
        "OPENAI_API_KEY": MEMU_OPENAI_API_KEY,
        "OPENAI_BASE_URL": MEMU_OPENAI_BASE_URL
    })

    embeddings_file = Path("memU-experiment-main/memu/memory/embeddings.py")
    replace_in_file(embeddings_file, NOW_EMBEDDING_MODEL, MEMU_EMBEDDING_MODEL)

def main():
    setup_memos()
    setup_hindsight()
    setup_memu()
    print("🎉 所有配置已完成")

if __name__ == "__main__":
    main()