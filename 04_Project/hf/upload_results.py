#!/usr/bin/env python3
"""Upload Remora benchmark results to a private HF dataset."""
import glob
import os
from huggingface_hub import HfApi, login

DS = "gcorzo/remora-bench-results"
api = HfApi(token=os.environ.get("HF_TOKEN"))
try:
    api.create_repo(DS, repo_type="dataset", private=True, exist_ok=True)
except Exception as e:
    print("repo create warn:", e)

for f in glob.glob("/bench/results/*.json") + ["/bench/llama_bench_std.txt"]:
    if os.path.exists(f):
        api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f),
                        repo_id=DS, repo_type="dataset")
        print("uploaded", os.path.basename(f))
