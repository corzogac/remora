#!/usr/bin/env python3
"""Follow-up: corrected stall stats + weights schema for the Ornith office trace."""
import json
import statistics
from collections import defaultdict

ops, weights = [], []
with open("trace_ornith.jsonl", "rb") as f:
    for line in f:
        line = line.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("t") == "op":
            ops.append(r)
        elif r.get("t") == "weights":
            weights.append(r)

print("sample weights record:", json.dumps(weights[0])[:220])

fam = defaultdict(list)
for r in ops:
    fam[r["op"].split("-")[0]].append(r["us"])
print("\n== op family totals (corrected ms) ==")
for f, vals in sorted(fam.items(), key=lambda kv: -sum(kv[1]))[:8]:
    srt = sorted(vals)
    print(f"  {f:24s} sum={sum(vals)/1e3:9.1f} ms  per-token={sum(vals)/1e3/76:7.1f} ms  "
          f"median={statistics.median(vals):7.1f}us p99={srt[int(0.99*len(vals))-1]:7.1f}us max={max(vals):7.1f}us")

big5 = sum(1 for r in ops if r["us"] > 5000)
big20 = sum(1 for r in ops if r["us"] > 20000)
print(f"\n== stall events == ops>5ms: {big5}  ops>20ms: {big20}  (of {len(ops)})")

for k in ("ws", "w", "weights", "vals", "value"):
    if k in weights[0] and isinstance(weights[0][k], list):
        wl = [r[k] for r in weights if r.get(k)]
        top1 = [max(x) for x in wl]
        print(f"\n== weights field '{k}': top-1 mean={statistics.mean(top1)*100:.1f}% "
              f"median={statistics.median(top1)*100:.1f}% n={len(top1)}")
        break
