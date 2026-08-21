#!/usr/bin/env python3
"""Learnability probe for the observer dataset (pure numpy).

Predict next-token top-k experts per layer using:
  A) cosine-NN on router logits (nearest training logits -> its next_topk)
  B) argmax persistence (this token's top-1 logit expert, repeated)
  C) random training record (chance)

Metrics: exact set match % and Jaccard %, per layer and overall.
"""
import json
import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = os.path.join(HERE, "observer_dataset", "records.jsonl")

per_layer = {}
with open(RECORDS) as fh:
    for line in fh:
        r = json.loads(line)
        per_layer.setdefault(r["l"], []).append(r)

rng = random.Random(42)
layers = sorted(per_layer)
print(f"{'layer':>5} {'n':>6} {'cos-exact%':>10} {'cos-jacc%':>10} {'argmax-jacc%':>13} {'chance-jacc%':>12}")
tot = {"n": 0, "cos_e": 0, "cos_j": 0.0, "argmax_j": 0.0, "rnd_j": 0.0}
for l in layers:
    recs = [r for r in per_layer[l] if len(r["logits"]) == 256]  # main MoE router only (exclude nextn-head records)
    rng.shuffle(recs)
    n_train = int(len(recs) * 0.7)
    train, test = recs[:n_train], recs[n_train:]
    if not train or not test:
        continue
    Xtr = np.array([r["logits"] for r in train], dtype=np.float32)
    Xte = np.array([r["logits"] for r in test], dtype=np.float32)
    Xtr /= (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-9)
    Xte /= (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9)
    ytr = [set(r["next_topk"]) for r in train]
    yte = [set(r["next_topk"]) for r in test]
    sims = Xte @ Xtr.T  # (m, n_train)
    nn = sims.argmax(axis=1)
    cos_e = cos_j = argmax_j = rnd_j = 0
    n = len(test)
    for i, (a, idx) in enumerate(zip(yte, nn)):
        b = ytr[idx]
        if a == b:
            cos_e += 1
        if a | b:
            cos_j += len(a & b) / len(a | b)
        # argmax persistence: this token's top logit expert, repeated twice
        top1 = int(np.argmax(Xte[i]))
        pred = {top1}
        if a | pred:
            argmax_j += len(a & pred) / len(a | pred)
        rnd = ytr[rng.randrange(len(train))]
        if a | rnd:
            rnd_j += len(a & rnd) / len(a | rnd)
    cos_e_p = 100 * cos_e / n
    cos_j_p = 100 * cos_j / n
    argmax_j_p = 100 * argmax_j / n
    rnd_j_p = 100 * rnd_j / n
    print(f"{l:>5} {n:>6} {cos_e_p:>10.2f} {cos_j_p:>10.2f} {argmax_j_p:>13.2f} {rnd_j_p:>12.2f}")
    tot["n"] += n
    tot["cos_e"] += cos_e
    tot["cos_j"] += cos_j
    tot["argmax_j"] += argmax_j
    tot["rnd_j"] += rnd_j

n = tot["n"]
print(f"{'ALL':>5} {n:>6} {100*tot['cos_e']/n:>10.2f} {100*tot['cos_j']/n:>10.2f} "
      f"{100*tot['argmax_j']/n:>13.2f} {100*tot['rnd_j']/n:>12.2f}")
print("\nreference: naive same-topk baseline exact 0.03% / Jaccard 21.1% (from stats.json)")
