# Hindsight Benchmarks (LoCoMo only)

This benchmark set is trimmed to LoCoMo.

## Run

```bash
# From repo root
./scripts/benchmarks/run-locomo.sh
```

## Options

```bash
./scripts/benchmarks/run-locomo.sh --max-conversations 1
./scripts/benchmarks/run-locomo.sh --skip-ingestion
./scripts/benchmarks/run-locomo.sh --use-think
./scripts/benchmarks/run-locomo.sh --conversation conv-0
```

Results are saved in `benchmarks/locomo/results/`.
