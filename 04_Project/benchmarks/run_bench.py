#!/usr/bin/env python3
"""Remora V2 — benchmark runner (baseline vs Remora, before/after).

Runs the standard llama.cpp benchmark (llama-bench) plus Remora trace metrics
on the same prompts for TWO builds:

    --baseline  stock llama.cpp llama-server (or llama-bench)
    --remora    remora-llama build with prefetch hooks

and writes a comparison report (JSON + Markdown) to results/.

Designed to run locally, via GitHub Actions (self-hosted runner on the office
server), or inside an HF Space container — same command everywhere.

Usage:
    python run_bench.py --model model.gguf --baseline-dir C:/llama-cpp/stock --remora-dir C:/llama-cpp/remora
    python run_bench.py --api http://127.0.0.1:11435/v1 --api-key KEY   # live-server timing mode
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def run(cmd, timeout=1800):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def api_benchmark(base_url, api_key, model, prompt, max_tokens=200, runs=3):
    """Time generation via the OpenAI-compatible endpoint (works on any llama-server)."""
    import urllib.request

    results = []
    for i in range(runs):
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            base_url + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + api_key})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as resp:
            d = json.loads(resp.read())
        wall = time.time() - t0
        tim = d.get("timings", {})
        results.append({
            "run": i + 1,
            "wall_s": round(wall, 2),
            "gen_tok_s": round(tim.get("predicted_per_second", 0), 2),
            "prompt_tok_s": round(tim.get("prompt_per_second", 0), 2),
            "n_pred": tim.get("predicted_n", 0),
            "n_prompt": tim.get("prompt_n", 0),
        })
    return results


def llama_bench(binary_dir, model, threads=8):
    """Standard llama.cpp benchmark (llama-bench): the 'standard test' before/after."""
    bench = os.path.join(binary_dir, "llama-bench" + (".exe" if sys.platform == "win32" else ""))
    if not os.path.exists(bench):
        return {"error": f"llama-bench not found in {binary_dir}"}
    code, out, err = run([bench, "-m", model, "-t", str(threads), "-r", "2"])
    return {"rc": code, "output": (out or err)[-4000:]}


def main():
    ap = argparse.ArgumentParser(description="Remora benchmark runner")
    ap.add_argument("--model", required=True, help="path to the GGUF")
    ap.add_argument("--baseline-dir", help="stock llama.cpp bin dir")
    ap.add_argument("--remora-dir", help="remora-llama bin dir")
    ap.add_argument("--api", help="live server base URL (timing mode)")
    ap.add_argument("--api-key", default="", help="API key for live mode")
    ap.add_argument("--tag", default="", help="optional run tag")
    args = ap.parse_args()

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": os.path.basename(args.model),
        "tag": args.tag,
        "api_mode": bool(args.api),
    }

    if args.api:
        report["api_timing"] = api_benchmark(args.api, args.api_key,
                                             "LFM2.5-8B-A1B", 
                                             "Write a paragraph on hydrological modelling.", 200, 3)

    if args.baseline_dir:
        report["baseline_llama_bench"] = llama_bench(args.baseline_dir, args.model)
    if args.remora_dir:
        report["remora_llama_bench"] = llama_bench(args.remora_dir, args.model)

    if args.baseline_dir and args.remora_dir:
        # before/after diff (llama-bench summary lines)
        b = report["baseline_llama_bench"].get("output", "")
        r = report["remora_llama_bench"].get("output", "")
        report["diff_note"] = "compare baseline vs remora llama-bench tables above"

    os.makedirs(RESULTS, exist_ok=True)
    name = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.tag:
        name += f"_{args.tag}"
    jpath = os.path.join(RESULTS, name + ".json")
    with open(jpath, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {jpath}")


if __name__ == "__main__":
    main()
