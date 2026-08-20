#!/usr/bin/env python3
"""Analyze an Ornith-1.5-35B-A3B remora trace (JSONL from the office A2000 run).

Records per decoded token, per MoE layer:
  t=op      per-op wall time (us)   -> stall data
  t=router  full router logits      -> the expert "wave"
  t=topk    selected expert ids     -> the path taken
  t=weights expert weights          -> distribution mass

Usage: python3 analyze_ornith_trace.py trace.jsonl
"""
import json
import math
import sys
from collections import Counter, defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else "trace_ornith.jsonl"

recs = {"op": [], "router": [], "topk": [], "weights": []}
with open(PATH, "rb") as f:
    for line in f:
        line = line.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        recs.setdefault(r.get("t"), []).append(r)

print("== record counts ==")
for t, rs in sorted(recs.items()):
    print(f"  {t:8s} {len(rs)}")

routers = recs["router"]
topks = recs["topk"]
weights = recs["weights"]
ops = recs["op"]

if routers:
    layers = sorted({r["l"] for r in routers})
    print(f"== router == layers={len(layers)} [{min(layers)}..{max(layers)}] "
          f"experts_per_layer={routers[0].get('ne')} topk={routers[0].get('nt')} "
          f"records={len(routers)} -> ~{len(routers)/max(len(layers),1):.1f} tokens/layer")

    # expert selection frequency (importance) per layer from topk
    sel = defaultdict(Counter)
    per_layer_topk = defaultdict(list)  # l -> list of (token_idx, ids)
    for r in topks:
        sel[r["l"]].update(r.get("ids", []))
        per_layer_topk[r["l"]].append((r.get("seq", 0), r.get("ids", [])))
    print("== expert importance (top-10 selected experts, all layers combined) ==")
    allsel = Counter()
    for c in sel.values():
        allsel.update(c)
    total_sel = sum(allsel.values())
    for eid, cnt in allsel.most_common(10):
        print(f"  expert {eid:4d}: {cnt:5d} selections ({100*cnt/total_sel:.2f}%)")
    print(f"  ... {len(allsel)} distinct experts ever selected of {routers[0].get('ne')}")

    # consecutive-token persistence (the prefetch predictor baseline)
    exact_hits = jacc = n = 0
    for l, lst in per_layer_topk.items():
        lst.sort()
        for i in range(1, len(lst)):
            a, b = set(lst[i-1][1]), set(lst[i][1])
            n += 1
            if a == b:
                exact_hits += 1
            if a | b:
                jacc += len(a & b) / len(a | b)
    if n:
        print(f"== consecutive-token top-k persistence =="
              f" exact={100*exact_hits/n:.1f}%  jaccard={100*jacc/n:.1f}%  (n={n} pairs)")

    # router logits entropy per layer (peaky = predictable = prefetchable)
    print("== router logits softmax entropy (top-4 most/least predictable layers) ==")
    ents = {}
    for l in layers:
        ls = [r["logits"] for r in routers if r["l"] == l][:64]
        if not ls:
            continue
        h = 0.0
        for lg in ls:
            mx = max(lg)
            es = [math.exp(x - mx) for x in lg]
            s = sum(es)
            h -= sum((e / s) * math.log(e / s) for e in es if e)
        ents[l] = h / len(ls)
    for l, h in sorted(ents.items(), key=lambda kv: kv[1])[:4]:
        print(f"  layer {l:3d}: entropy {h:6.2f} nats (peaky)")
    for l, h in sorted(ents.items(), key=lambda kv: -kv[1])[:4]:
        print(f"  layer {l:3d}: entropy {h:6.2f} nats (flat)")

if ops:
    by_op = Counter(r["op"] for r in ops)
    by_op_us = defaultdict(int)
    by_op_n = Counter()
    for r in ops:
        by_op_us[r["op"]] += r.get("us", 0)
        by_op_n[r["op"]] += 1
    tot_us = sum(by_op_us.values())
    print("== op timing breakdown (stall data) ==")
    for op, us in sorted(by_op_us.items(), key=lambda kv: -kv[1]):
        print(f"  {op:28s} n={by_op_n[op]:5d}  total={us/1e3:9.1f} ms  {100*us/tot_us:5.1f}%")
    print(f"  TOTAL traced ops: {tot_us/1e3:.1f} ms  -> {tot_us/1e3/1000:.1f} s over the traced tokens")

print("done")
