#!/bin/bash
set -eu
cd /root/sk/pipeline_20261103
mkdir -p runtime/{home,tmp,cache,torch,miopen,comgr,triton,hylog}
docker exec -w /workspace/pipeline_20261103 \
 -e HOME=/workspace/pipeline_20261103/runtime/home \
 -e TMPDIR=/workspace/pipeline_20261103/runtime/tmp \
 -e XDG_CACHE_HOME=/workspace/pipeline_20261103/runtime/cache \
 -e TORCH_HOME=/workspace/pipeline_20261103/runtime/torch \
 -e TORCH_EXTENSIONS_DIR=/workspace/pipeline_20261103/runtime/torch/extensions \
 -e TORCHINDUCTOR_CACHE_DIR=/workspace/pipeline_20261103/runtime/torch/inductor \
 -e TRITON_CACHE_DIR=/workspace/pipeline_20261103/runtime/triton \
 -e MIOPEN_USER_DB_PATH=/workspace/pipeline_20261103/runtime/miopen \
 -e MIOPEN_CUSTOM_CACHE_DIR=/workspace/pipeline_20261103/runtime/miopen \
 -e AMD_COMGR_CACHE_DIR=/workspace/pipeline_20261103/runtime/comgr \
 -e CUDA_CACHE_PATH=/workspace/pipeline_20261103/runtime/cache \
 -e HIP_LOG_OUTPUT_PATH=/workspace/pipeline_20261103/runtime/hylog/ \
 -e HYLOG_DISABLE_ERROR_LOG=1 -e HIP_LOG_ENABLE_FILE=0 -e GLOG_logtostderr=1 \
 -e PYTHONDONTWRITEBYTECODE=1 \
 sunkai_compute python3 -B driver.py
