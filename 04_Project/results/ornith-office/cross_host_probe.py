#!/usr/bin/env python3
"""Cross-host generalization probe: train cosine-NN observer on OFFICE router
traces, test on L4 full-GPU traces. Same model + weights, different host.

If router logits are deterministic, an observer trained on the cheap office
box must transfer to the fast GPU box — the deployment story of remora.
"""
import json
import statistics
from collections import defaultdict

import numpy as np


def load_pairs(path):
    """Per-layer list of (logits_t, next_topk) from a raw trace."""
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
            if r.get("t") == "router":
                routers.append(r)
            elif r.get("t") == "topk":
                topks.append(r)
    rl, kl = defaultdict(list), defaultdict(list)
    for r in routers:
        rl[r["l"]].append(r)
    for r in topks:
        kl[r["l"]].append(r)
    out = defaultdict(list)
    for l in rl:
        rs = sorted(rl[l], key=lambda x: x.get("seq", 0))
        ks = sorted(kl[l], key=lambda x: x.get("seq", 0))
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
            if len(r_now["logits"]) != 256:
                continue
            out[l].append((r_now["logits"], set(k_next.get("ids", []))))
    return out


TRAIN = ["trace_ornith.jsonl", "trace_ornith3.jsonl"]
TEST = ["l4/ornith_l4_base.jsonl"]
HERE = "/Users/gerald/Dropbox/04-Work/Projects/Remora/04_Project/results/ornith-office"

train = defaultdict(list)
for t in TRAIN:
    for l, pairs in load_pairs(f"{HERE}/{t}").items():
        train[l].extend(pairs)
test = defaultdict(list)
for t in TEST:
    for l, pairs in load_pairs(f"{HERE}/{t}").items():
        test[l].extend(pairs)

print(f"{'layer':>5} {'train':>6} {'test':>6} {'exact%':>8} {'jacc%':>8}")
exact_tot = jacc_tot = n_tot = 0
per_layer_jacc = []
for l in sorted(set(train) | set(test)):
    tr, te = train.get(l, []), test.get(l, [])
    if len(tr) < 10 or not te:
        continue
    Xtr = np.array([x[0] for x in tr], dtype=np.float32)
    Xte = np.array([x[0] for x in te], dtype=np.float32)
    Xtr /= (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-9)
    Xte /= (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9)
    ytr = [x[1] for x in tr]
    yte = [x[1] for x in te]
    sims = Xte @ Xtr.T
    nn = sims.argmax(axis=1)
    ex = jc = 0
    for i, (a, idx) in enumerate(zip(yte, nn)):
        b = ytr[idx]
        if a == b:
            ex += 1
        if a | b:
            jc += len(a & b) / len(a | b)
    n = len(te)
    print(f"{l:>5} {len(tr):>6} {n:>6} {100*ex/n:>8.1f} {100*jc/n:>8.1f}")
    exact_tot += ex
    jacc_tot += jc
    n_tot += n
    per_layer_jacc.append(100 * jc / n)

print(f"\nOFFICE->L4 transfer: exact {100*exact_tot/n_tot:.1f}%  Jaccard {100*jacc_tot/n_tot:.1f}%  (n={n_tot})")
print(f"Jaccard spread across layers: {min(per_layer_jacc):.1f}% .. {max(per_layer_jacc):.1f}%")
