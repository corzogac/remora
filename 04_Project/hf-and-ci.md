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

## Option B — HF Space (GPU) mirror

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
