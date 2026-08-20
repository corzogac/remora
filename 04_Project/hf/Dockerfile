# Remora benchmark container — llama.cpp (CUDA) + LFM2.5-8B-A1B + run_bench.py
# Runs on HF Jobs (t4-small) or locally. Build: docker build -t remora-bench .
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential && rm -rf /var/lib/apt/lists/*
# Pin to b10509 — the release validated on the office server
RUN git clone --depth 1 --branch b10509 https://github.com/ggml-org/llama.cpp /llama.cpp \
    && cmake -S /llama.cpp -B /llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /llama.cpp/build --target llama-server llama-bench -j$(nproc)

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates python3 python3-pip && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir huggingface_hub
COPY --from=build /llama.cpp/build/bin/llama-server /llama.cpp/build/bin/llama-bench /usr/local/bin/
COPY bench/ /bench/
WORKDIR /bench
ENTRYPOINT ["bash", "/bench/run_in_job.sh"]
