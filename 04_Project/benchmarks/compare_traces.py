#!/usr/bin/env python3
"""Compare per-op MoE costs between two traces (e.g. K=4 vs K=2 on same box)."""
import json
import sys
from statistics import mean


def op_costs(path):
    ops = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("t") == "op":
                ops.setdefault(r["l"], []).append(r["us"])
    return {l: (mean(v), len(v)) for l, v in sorted(ops.items())}


def main():
    a, b = sys.argv[1], sys.argv[2]
    ka = op_costs(a)
    kb = op_costs(b)
    print(f"{'layer':>5} {'A mean_us':>10} {'B mean_us':>10} {'ratio':>7}")
    ta = tb = 0
    for l in ka:
        ma, na = ka[l]
        mb, nb = kb.get(l, (0, 0))
        ta += ma * na
        tb += mb * nb
        print(f"{l:>5} {ma:>10.0f} {mb:>10.0f} {mb / ma:>7.2f}")
    print(f"\ntotal op time A: {ta / 1e6:.2f}s  B: {tb / 1e6:.2f}s  ratio {tb / ta:.2f}")


if __name__ == "__main__":
    main()
