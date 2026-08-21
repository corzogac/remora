#!/usr/bin/env python3
"""Build the Remora learned-observer training set from banked traces.

Per MoE layer, per consecutive-token pair:
  features: router logits of token t (256 floats) + topk ids of token t
  label:    topk ids of token t+1 (the experts to prefetch)

Output: observer_dataset/records.jsonl + stats.json
Records are compact: {"l": layer, "logits": [...], "topk": [...], "next_topk": [...]}

Usage: python3 convert_traces_to_observer_dataset.py TRACE1 [TRACE2 ...]
"""
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "observer_dataset")
os.makedirs(OUT, exist_ok=True)


def load(path):
    routers, topks = [], []
    with open(path, "rb") as f:
        for line in f:
            line = line.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            t = r.get("t")
            if t == "router":
                routers.append(r)
            elif t == "topk":
                topks.append(r)
    return routers, topks


def build(path, out_fh, stats):
    routers, topks = load(path)
    # group by layer, order by seq
    per_layer_r = defaultdict(list)
    per_layer_k = defaultdict(list)
    for r in routers:
        per_layer_r[r["l"]].append(r)
    for r in topks:
        per_layer_k[r["l"]].append(r)
    n_records = 0
    for l in sorted(per_layer_r):
        rs = sorted(per_layer_r[l], key=lambda x: x.get("seq", 0))
        ks = sorted(per_layer_k[l], key=lambda x: x.get("seq", 0))
        # router and topk records alternate; pair by position (same token)
        pairs = []
        i = j = 0
        while i < len(rs) and j < len(ks):
            if rs[i]["seq"] <= ks[j]["seq"]:
                pairs.append((rs[i], ks[j] if j < len(ks) else None))
                i += 1
            else:
                j += 1
        for idx in range(len(pairs) - 1):
            r_now, k_now = pairs[idx]
            _, k_next = pairs[idx + 1]
            if k_now is None or k_next is None:
                continue
            rec = {
                "l": l,
                "logits": r_now["logits"],
                "topk": k_now.get("ids", []),
                "next_topk": k_next.get("ids", []),
            }
            out_fh.write(json.dumps(rec) + "\n")
            n_records += 1
            stats["n_records"] += 1
    stats["traces"].append({"file": os.path.basename(path), "records": n_records})
    return n_records


def main():
    paths = sys.argv[1:]
    stats = {"n_records": 0, "traces": [], "naive_baseline": {}}
    with open(os.path.join(OUT, "records.jsonl"), "w") as fh:
        for p in paths:
            build(p, fh, stats)

    # naive baseline over the whole dataset: predict next = current topk
    exact = jacc = n = 0
    with open(os.path.join(OUT, "records.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            a, b = set(r["topk"]), set(r["next_topk"])
            n += 1
            if a == b:
                exact += 1
            if a | b:
                jacc += len(a & b) / len(a | b)
    stats["naive_baseline"] = {
        "exact_pct": round(100 * exact / n, 2),
        "jaccard_pct": round(100 * jacc / n, 2),
        "n_pairs": n,
    }
    with open(os.path.join(OUT, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"dataset: {os.path.join(OUT, 'records.jsonl')}")


if __name__ == "__main__":
    main()
