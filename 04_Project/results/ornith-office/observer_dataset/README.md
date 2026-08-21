# Remora Observer Dataset — Ornith-1.5-35B-A3B

Training set for the learned expert-importance observer (Exp C).

## Source traces (valid, non-gated runs only)
- trace_ornith.jsonl        — office A2000, 76 tokens, 15.8 MB
- trace_ornith3.jsonl       — office A2000, 3x76 tokens, 49.4 MB
- l4/ornith_l4_base.jsonl   — HF L4 full-GPU, 3x74 tokens, 41.5 MB

## Format (records.jsonl, one JSON per line)
{"l": layer (0-39), "logits": [256 router logits of token t],
 "topk": [2 selected expert ids of token t],
 "next_topk": [2 selected expert ids of token t+1]}

## Stats (stats.json)
- 22,440 records: 3,000 (office) + 10,600 (office3) + 8,840 (L4)
- naive baseline (predict next = current topk):
  exact 0.03%, Jaccard 21.06%

## Goal
Train a per-layer model f(logits_t) -> topk_{t+1} that beats the naive
baseline (Jaccard 21.1%). Any improvement = prefetch hits that the
identity heuristic cannot get. records.jsonl is regenerable via
convert_traces_to_observer_dataset.py (not committed; 30+ MB).
