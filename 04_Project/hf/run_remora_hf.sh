#!/bin/bash
# Remora — generic traced run job (A/B/C).
# Env: MODEL_URL (GGUF), LABEL (result name), GATE_K (optional, experiment C),
#      HF_TOKEN (injected via --secrets).
# Downloads the remora binary + model, runs a traced greedy benchmark,
# uploads the trace + response to gcorzo/remora-traces.
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq > /dev/null 2>&1 || true
apt-get install -y -qq curl ca-certificates python3-pip > /dev/null 2>&1 || true
export HF_TOKEN="${HF_TOKEN}"
cd /tmp
curl -sL -H "Authorization: Bearer ${HF_TOKEN}" -o remora-bin.tar.gz "https://huggingface.co/datasets/gcorzo/remora-bin/resolve/main/remora-bin-linux-sm75.tar.gz"
mkdir -p /opt/remora && tar xzf remora-bin.tar.gz -C /opt/remora
curl -sSL -o /tmp/model.gguf "${MODEL_URL}"
ls -la /tmp/model.gguf
export REMORA_TRACE_FILE=/tmp/trace.jsonl
if [ -n "${GATE_K}" ]; then
    export REMORA_GATE_K="${GATE_K}"
    echo "GATE_K=${GATE_K}"
fi
/opt/remora/llama-server -m /tmp/model.gguf -ngl 99 -c 8192 --port 8080 --reasoning off --reasoning-budget 0 > /tmp/srv.log 2>&1 &
SRV=$!
for i in $(seq 1 240); do
    if curl -sf http://localhost:8080/health >/dev/null 2>&1; then echo HEALTH_OK; break; fi
    sleep 2
done
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"x","messages":[{"role":"user","content":"Write a haiku about water."}],"max_tokens":256,"temperature":0,"stream":false}' > /tmp/resp.json
echo "===RESPONSE==="
cat /tmp/resp.json
kill $SRV 2>/dev/null || true
echo "===TRACE_STATS==="
wc -l /tmp/trace.jsonl
python3 - "$HF_TOKEN" <<'EOF'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=sys.argv[1])
label = os.environ.get("LABEL", "run")
api.upload_file(path_or_fileobj="/tmp/trace.jsonl", path_in_repo=f"{label}.jsonl", repo_id="gcorzo/remora-traces", repo_type="dataset")
api.upload_file(path_or_fileobj="/tmp/resp.json", path_in_repo=f"{label}_resp.json", repo_id="gcorzo/remora-traces", repo_type="dataset")
print("UPLOAD_OK")
EOF
echo RUN_JOB_DONE
