# Benchmarks (V2)

Real-model benchmark tooling for Project Remora.

## Contents

- `parse_llama_trace.py` — trace analyzer: Top-1/Top-5 router prediction accuracy,
  expert-fetch stall stats (mean/p95), stall reduction between baseline and Remora
  runs. Also `--sim` mode to reproduce the V1-style simulation baseline.

## Workflow

1. Build the `remora-llama` fork (see `../Remora-V2-Experiment-Plan.md` §1).
2. Run baseline and Remora builds on the same prompts with `--trace-out` (JSONL).
3. `python parse_llama_trace.py traces/baseline.jsonl traces/remora.jsonl`
4. Results land in `../results/` as JSON + the experiment report.

## Trace format (JSONL, one line per token)

```json
{"tok": 0, "layer": 3, "topk": [7], "scores": [0.42],
 "h": [0.1, -0.2, ...], "expert_us": 312.0, "prefetched": false}
```
