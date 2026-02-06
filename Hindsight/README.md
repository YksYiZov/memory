# Hindsight LoCoMo Benchmark (Local)

Minimal LoCoMo benchmark runner for Hindsight.

## Install

```bash
pip install hindsight-all -U
```

## Dataset

- `benchmarks/locomo/datasets/locomo1.json`

## Configure

Create `Benchmark_memory_system/hindsight/.env`:

```bash
HINDSIGHT_API_LLM_PROVIDER=openai
HINDSIGHT_API_LLM_API_KEY=sk-cb7d4d3296e145ba9f333c73ce8c1056
HINDSIGHT_API_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
HINDSIGHT_API_LLM_MODEL=qwen3-max

HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai
HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY=sk-cb7d4d3296e145ba9f333c73ce8c1056
HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL=text-embedding-v4
HINDSIGHT_API_EMBEDDINGS_BATCH_SIZE=10
```

## Run

```bash
./run-locomo.sh
```

Results are saved in:

- `benchmarks/locomo/results/`
