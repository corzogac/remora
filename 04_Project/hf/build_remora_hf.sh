#!/bin/bash
# Remora — one-time Linux CUDA build job (runs on HF Jobs infra).
# Compiles the remora fork (trace + gate) for T4 (sm_75) and uploads the
# binary to the private dataset gcorzo/remora-bin.
set -e
export DEBIAN_FRONTEND=noninteractive
echo BUILD_START
apt-get update -qq > /dev/null
apt-get install -y -qq build-essential cmake ninja-build git curl python3-pip > /dev/null
echo DEPS_OK
cd /tmp
curl -sL -H "Authorization: Bearer ${HF_TOKEN}" -o remora.bundle "https://huggingface.co/datasets/gcorzo/remora-bin/resolve/main/remora-trace.bundle"
ls -la remora.bundle
git clone /tmp/remora.bundle /tmp/llama-cpp-remora 2>&1 | tail -1
cd /tmp/llama-cpp-remora
git checkout remora-trace 2>&1 | tail -1
git log -1 --oneline
echo CONFIGURE
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=ON
echo BUILDING
cmake --build build --target llama-server -j $(nproc)
cd build/bin
ls -la llama-server
tar czf /tmp/remora-bin.tar.gz llama-server
cd /tmp
python3 - "$HF_TOKEN" <<'EOF'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=sys.argv[1])
api.upload_file(path_or_fileobj="/tmp/remora-bin.tar.gz", path_in_repo="remora-bin-linux-sm75.tar.gz", repo_id="gcorzo/remora-bin", repo_type="dataset")
print("UPLOAD_OK")
EOF
echo BUILD_JOB_DONE
