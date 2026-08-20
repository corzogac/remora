#!/usr/bin/env python3
"""Remora — real-trace analyzer for the REMORA_TRACE_FILE format.

Input: JSONL lines from the instrumented llama-server (common/remora-trace.cpp):
    {"t":"op",      "op":"ffn_moe_logits-2", "l":2, "us":17034}
    {"t":"router",  "l":2, "ne":32, "nt":1, "logits":[...]}
    {"t":"topk",    "l":2, "nu":8, "nt":1, "ids":[...]}
    {"t":"weights", "l":2, "nu":8, "nt":1, "w":[...]}

Outputs (Experiment A/B measurement layer):
  - router statistics per layer: top-1 prob, entropy, top-1 margin
  - expert firing distribution (the "path of knowledge": hot vs cold experts)
  - weight mass concentration (how much of the wave is carried by top-1)
  - expert op cost per layer (the stall baseline: mean/p95)
  - prefetch simulation: predict next-token expert set from current token's
    top-k (naive persistence predictor) -> honest hit-rate per layer

Usage:
    python analyze_trace.py results/trace_lfm_office_20260820.jsonl
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from statistics import mean, pstdev


def softmax(vals):
    m = max(vals)
    e = [math.exp(x - m) for x in vals]
    s = sum(e)
    return [x / s for x in e]


def entropy(probs):
    return -sum(p * math.log(p) for p in probs if p > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="path to REMORA_TRACE_FILE jsonl")
    args = ap.parse_args()

    routers = defaultdict(list)   # layer -> list of (logits, probs, softmaxed)
    topks = defaultdict(list)     # layer -> list of id lists (per token)
    weights = defaultdict(list)   # layer -> list of weight lists
    ops = defaultdict(list)       # layer -> list of us (moe ops)
    seq = []

    with open(args.trace) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["t"] == "router":
                logits = r["logits"]
                if not logits:
                    continue
                routers[r["l"]].append((logits, softmax(logits)))
            elif r["t"] == "topk":
                topks[r["l"]].append(r["ids"])
            elif r["t"] == "weights":
                weights[r["l"]].append(r["w"])
            elif r["t"] == "op":
                ops[r["l"]].append(r["us"])

    n_layers = len(routers)
    n_expert = 0
    for ll in sorted(routers):
        n_expert = len(routers[ll][0][0])
        break

    print("=" * 60)
    print("REMORA TRACE ANALYSIS")
    print(f"layers: {n_layers}   experts: {n_expert}")
    n_tok = max((len(v) for v in routers.values()), default=0)
    print(f"tokens (per layer, from router records): {n_tok}")
    print("=" * 60)

    # 1) router statistics
    print("\n[ROUTER] per-layer confidence (softmax over expert logits)")
    print(f"{'layer':>5} {'top1_prob':>10} {'entropy':>9} {'top1_margin':>12} {'top2_overlap%':>14}")
    all_top1 = []
    for ll in sorted(routers):
        rows = routers[ll]
        top1s, ents, margs, ov = [], [], [], []
        for logits, probs in rows:
            order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            top1s.append(probs[order[0]])
            ents.append(entropy(probs))
            margs.append(probs[order[0]] - probs[order[1]] if len(order) > 1 else 0.0)
            # top-2 overlap between consecutive tokens (persistence signal)
        all_top1 += top1s
        print(f"{ll:>5} {mean(top1s):>10.4f} {mean(ents):>9.3f} {mean(margs):>12.4f}")

    print(f"\nmean top-1 prob across all tokens/layers: {mean(all_top1):.4f}")

    # 2) expert firing distribution
    print("\n[PATH] expert firing counts (top-k selections), all layers")
    fire = Counter()
    per_layer_fire = {}
    for ll in sorted(topks):
        c = Counter()
        for ids in topks[ll]:
            for e in ids:
                c[e] += 1
                fire[e] += 1
        per_layer_fire[ll] = c
    total_fire = sum(fire.values())
    print(f"total expert selections: {total_fire}  distinct experts used: {len(fire)}/{n_expert}")
    top_hot = fire.most_common(8)
    print("hottest experts:", ", ".join(f"e{e}={c}" for e, c in top_hot))
    # concentration: share of top-4 experts
    top4 = sum(c for _, c in fire.most_common(4))
    print(f"top-4 experts carry {100.0 * top4 / total_fire:.1f}% of all selections")

    # 3) weight mass concentration
    print("\n[MASS] top-1 expert weight share (of the routed mass)")
    w1s = []
    for ll in sorted(weights):
        for w in weights[ll]:
            if w:
                w1s.append(max(w) / sum(w))
    if w1s:
        print(f"mean top-1 weight share: {mean(w1s):.3f}  (1/{n_expert} = {1.0/n_expert:.3f} if uniform)")

    # 4) op cost per layer (stall baseline)
    print("\n[STALL] moe op cost per layer (us)")
    print(f"{'layer':>5} {'ops':>6} {'mean_us':>9} {'p95_us':>9}")
    all_us = []
    for ll in sorted(ops):
        us = sorted(ops[ll])
        all_us += us
        p95 = us[int(0.95 * len(us))] if us else 0
        print(f"{ll:>5} {len(us):>6} {mean(us):>9.0f} {p95:>9.0f}")
    if all_us:
        all_us_s = sorted(all_us)
        print(f"\noverall: mean {mean(all_us):.0f} us  p95 {all_us_s[int(0.95*len(all_us_s))]:.0f} us  "
              f"std {pstdev(all_us):.0f} us")

    # 5) prefetch simulation: naive persistence predictor
    print("\n[PREFETCH-SIM] naive predictor: next token uses SAME top-k set")
    print(f"{'layer':>5} {'hit_rate%':>10} {'Jaccard':>9}")
    rates = []
    for ll in sorted(topks):
        seq_ids = topks[ll]
        hits = sum(1 for a, b in zip(seq_ids[:-1], seq_ids[1:]) if set(a) == set(b))
        rate = 100.0 * hits / max(1, len(seq_ids) - 1)
        rates.append(rate)
        jac = 0.0
        if len(seq_ids) > 1:
            jac = mean(len(set(a) & set(b)) / len(set(a) | set(b))
                       for a, b in zip(seq_ids[:-1], seq_ids[1:]) if set(a) | set(b))
        print(f"{ll:>5} {rate:>10.1f} {jac:>9.3f}")
    if rates:
        print(f"\nmean exact-set hit rate: {mean(rates):.1f}%")

    print("\nDONE — see 04_Project/experiments.md for interpretation.")


if __name__ == "__main__":
    main()
