# Running Remora Benchmarks on Hugging Face & CI

Options for "set them up and let them run", including the standard llama.cpp
benchmarks (llama-bench) **before and after** the Remora changes.

## Option A — GitHub Actions + self-hosted runner (recommended, free)

`../.github/workflows/remora-bench.yml` runs `run_bench.py` automatically:

- on push touching Remora code / benchmarks, on manual dispatch, and weekly (cron)
- runs on a self-hosted runner registered on the office server (RTX A2000) —
  install `actions-runner` (Windows) on the office box and label it `office`
- each run executes llama-bench on the **baseline build** (stock llama.cpp) and the
  **remora build** with identical flags/models/prompts → true before/after diff
- results are committed back to `04_Project/results/`

## Option B — HF Jobs (works on free accounts with credits; validated 2026-08-20)

HF Spaces hosting requires PRO (402 on free accounts), but **`hf jobs run`** works
with credits (your €10 ≈ 25 h of t4-small at $0.40/hr):

```
hf jobs run ghcr.io/ggml-org/llama.cpp:server-cuda --flavor t4-small --timeout 90m \
  --label experiment=remora-baseline -- bash -c '<benchmark script>'
```

KEY FACTS (learned the hard way, 2026-08-20):
- **`bash -lc` BREAKS: the CLI's own `-l` (--label) eats `-lc`** and bash receives
  the script as a filename. Always use **`bash -c`** (verified via argv probe
  job: `python -c "import sys;print(sys.argv)"` → `['-c']`).
- The prebuilt image's binary is **`/app/llama-server`** (NOT on PATH) — call it
  by full path. No `llama-bench` binary ships. Image is newer than b10509
  (fingerprint b10524).
- Download the GGUF in the FOREGROUND before starting the server (backgrounding
  `curl && llama-server &` makes the readiness poll race the download).
- Server ready-check must be conditional (`if curl ...; then ... else cat log; fi`).
- Pass `--reasoning off --reasoning-budget 0` for direct-answer mode.
- The job CLI blocks and streams stdout; jobs also appear at
  `https://huggingface.co/jobs/gcorzo/...` and are inspectable via `hf jobs ps/logs`.
- Cost per 4-minute T4 job ≈ $0.03.

VALIDATED recipe (full run, 2026-08-20): model download (5.15 GB, byte-verified),
server boot, chat completion at **138.3 tok/s** (eval 477 ms/67 tok), 66 graphs
reused (prefix caching live). Weekly schedule (Mon 03:00 UTC, id 6a871c26...):
`hf jobs scheduled run "0 3 * * 1" --flavor t4-small --timeout 15m <image> bash -c '<same script>'`

Result (2026-08-20): LFM2.5-8B-A1B Q4_K_M on T4 full GPU = **~138 tok/s gen**,
~250 tok/s prompt — 7.8× faster than the office server's partial-offload A2000.
Saved to `results/hf_t4_baseline_lfm.json`. The remora-llama fork must run the
IDENTICAL job/flags to produce the before/after delta.

## Option C — HF Spaces (GPU) mirror

For public, reproducible results with a paper link:

1. Create a **private Space** (Docker template) with a `Dockerfile` that builds
   `remora-llama` and runs `run_bench.py` on a T4/L4 GPU (`hf.co` paid hardware,
   ~$0.4–1/hr; use `sleep infinity` + manual triggers, or a lightweight in-Space
   scheduler to keep cost bounded).
2. The Space writes results to a **HF dataset** (via `huggingface_hub`, private
   first) — this becomes the paper's data repository and the before/after leaderboard.

## Option C — HF Datasets for traces

Whatever runs where, upload trace files (router logits, expert timings, activation
taps) to a private HF dataset (`remora-traces`). Datasets are the right artifact
for reproducibility + the arXiv data-availability statement.

## Current status (2026-08-20)

- Baseline runtime exists: llama.cpp b10509 on the office server, LFM2.5-8B-A1B
  Q4_K_M, live at 127.0.0.1:11435 (API timing mode of `run_bench.py` works NOW:
  `python run_bench.py --model ... --api http://127.0.0.1:11435/v1 --api-key ...`).
- NOT yet running: the instrumented `remora-llama` fork (router logits, expert
  timings, prefetch hook) — this is the critical missing piece. Baseline llama-bench
  numbers can be captured today; Remora numbers need the fork.
