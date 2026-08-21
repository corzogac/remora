import json
from collections import Counter

c = Counter()
lens = Counter()
with open("observer_dataset/records.jsonl") as fh:
    for i, line in enumerate(fh):
        r = json.loads(line)
        c[r["l"]] += 1
        lens[len(r["logits"])] += 1
        if i > 40000:
            break
print("layers:", sorted(c.items())[:8])
print("logits lengths:", sorted(lens.items()))
