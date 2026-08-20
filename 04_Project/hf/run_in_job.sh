#!/bin/bash
# Remora benchmark job — runs inside the HF Jobs container (T4).
# Downloads the model, starts llama-server (full GPU: -ngl 99), runs the
# baseline benchmark suite (API timing + llama-bench standard), uploads results.
set -e
export PATH=/usr/local/bin:$PATH
MODEL_URL="https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/main/LFM2.5-8B-A1B-Q4_K_M.gguf"
MODEL=/tmp/LFM2.5-8B-A1B-Q4_K_M.gguf

echo "== [1/5] downloading model =="
curl -sL -o "$MODEL" "$MODEL_URL"
ls -la "$MODEL"

echo "== [2/5] starting llama-server (full GPU) =="
llama-server -m "$MODEL" --host 127.0.0.1 --port 8080 -c 8192 -ngl 99 \
  --parallel 1 --reasoning off --reasoning-budget 0 --api-key localtest \
  > /bench/server.log 2>&1 &
SERVER_PID=$!
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8080/v1/models >/dev/null 2>&1 && break
  sleep 2
done
curl -sf http://127.0.0.1:8080/v1/models >/dev/null 2>&1 || { echo "server failed"; cat /bench/server.log; exit 1; }
echo "server ready"

echo "== [3/5] API timing benchmarks =="
python3 /bench/run_bench.py --model LFM2.5-8B-A1B \
  --api http://127.0.0.1:8080/v1 --api-key localtest --tag hf_job || true

echo "== [4/5] llama-bench standard =="
llama-bench -m "$MODEL" -ngl 99 -r 2 -t 4 > /bench/llama_bench_std.txt 2>&1 || true
cat /bench/llama_bench_std.txt | head -30

echo "== [5/5] uploading results =="
python3 /bench/upload_results.py || echo "upload failed (non-fatal)"

kill $SERVER_PID 2>/dev/null || true
echo "JOB COMPLETE"
