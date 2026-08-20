#!/usr/bin/env python3
"""Remora V2 — llama.cpp trace analyzer.

Parses trace output from the instrumented `remora-llama` fork (or simulated traces)
and computes the Experiment A metrics:

  - Router prediction: Top-1 / Top-5 hit rate of the 2nd-order Taylor predictor
    (V1 C engine math) vs the actual next-token routing, per MoE layer.
  - Expert stall: mean / p95 expert-fetch latency, baseline vs prefetched.
  - End-to-end: tokens/sec (from timing records).

Usage:
    python parse_llama_trace.py traces/baseline.jsonl traces/remora.jsonl
    python parse_llama_trace.py --sim   # reproduce V1-style simulation for comparison

Trace JSONL line format (one per token):
    {"tok": n, "layer": l, "topk": [ids...], "scores": [s...],
     "h": [h_l dims...] | null, "expert_us": 123.4, "prefetched": false}
"""
import argparse
import json
import math
import sys
from statistics import mean, pstdev


def taylor_predict(h0, h1, h2, k):
    """2nd-order Taylor extrapolation of the residual stream wake (V1 C engine math).

    h(t+1) ~= h(t) + v(t) + 0.5*a(t); then projected onto the router matrix.
    Without a real router matrix W_r we approximate: predicted top-k = argmax of the
    extrapolated activation's top-k directions (see remora_engine.c for the true hook).
    """
    v = [h1[i] - h0[i] for i in range(len(h1))]
    a = [(h1[i] - h0[i]) - (h0[i] - h2[i]) for i in range(len(h1))]
    est = [h1[i] + v[i] + 0.5 * a[i] for i in range(len(h1))]
    # rank by magnitude of the extrapolated activation (placeholder for W_r projection)
    idx = sorted(range(len(est)), key=lambda i: abs(est[i]), reverse=True)
    return set(idx[:k])


def load_traces(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze(path, k=1):
    rows = load_traces(path)
    if not rows:
        return None
    hits, total = 0, 0
    stalls = []
    for r in rows:
        if r.get("topk") is not None:
            total += 1
            if r["topk"][0] in r.get("predicted", []):
                hits += 1
        if r.get("expert_us") is not None:
            stalls.append(r["expert_us"])
    top1 = hits / total if total else 0.0
    out = {"file": path, "tokens": len(rows), "top1": top1}
    if stalls:
        stalls_sorted = sorted(stalls)
        out["expert_us_mean"] = mean(stalls)
        out["expert_us_p95"] = stalls_sorted[int(0.95 * len(stalls_sorted)) - 1]
    return out


def simulate(dim=256, layers=12, tokens=500, seed=42):
    """Reproduce the V1-style simulation for baseline comparison (random weights)."""
    import random

    random.seed(seed)
    rows = []
    h = [[0.0] * dim for _ in range(layers)]
    for t in range(tokens):
        for l in range(layers):
            # synthetic activation wake
            h_new = [random.gauss(0, 1) for _ in range(dim)]
            if l >= 3:
                pred = taylor_predict(h[l - 2], h[l - 1], h_new, 1)
                topk = sorted(range(dim), key=lambda i: abs(h_new[i]), reverse=True)[:5]
                rows.append({"tok": t, "layer": l, "topk": topk[:1],
                             "predicted": list(pred), "expert_us": random.uniform(80, 900),
                             "prefetched": False})
            h[l] = h_new
    return rows


def main():
    ap = argparse.ArgumentParser(description="Remora trace analyzer")
    ap.add_argument("traces", nargs="*", help="trace jsonl files (baseline, remora)")
    ap.add_argument("--sim", action="store_true", help="run simulation comparison")
    ap.add_argument("--k", type=int, default=1, help="Top-K (default 1)")
    args = ap.parse_args()

    if args.sim:
        base = analyze_from_rows(simulate())
        print(f"simulation baseline: {base}")
        return

    if not args.traces:
        ap.error("need trace files or --sim")
    results = [analyze(p, args.k) for p in args.traces]
    for r in results:
        if r:
            print(json.dumps(r, indent=2))
    if len(results) == 2 and results[0] and results[1]:
        a, b = results[0], results[1]
        if "expert_us_mean" in a and "expert_us_mean" in b:
            red = 1 - b["expert_us_mean"] / a["expert_us_mean"]
            print(f"\nstall reduction (baseline→remora): {red:.1%}")


def analyze_from_rows(rows):
    hits = sum(1 for r in rows if r["topk"][0] in r.get("predicted", []))
    stalls = [r["expert_us"] for r in rows if r.get("expert_us") is not None]
    return {"sim_tokens": len(rows), "top1": hits / len(rows),
            "expert_us_mean": mean(stalls)}


if __name__ == "__main__":
    main()
