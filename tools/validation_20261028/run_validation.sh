#!/bin/bash
set -eu
cd /root/sk/compute_validation_20261028
mkdir -p runtime/home runtime/tmp runtime/cache runtime/torch runtime/miopen runtime/comgr runtime/triton runtime/hylog traces

run_in_container() {
  docker exec -w /workspace/compute_validation_20261028 \
    -e HOME=/workspace/compute_validation_20261028/runtime/home \
    -e TMPDIR=/workspace/compute_validation_20261028/runtime/tmp \
    -e XDG_CACHE_HOME=/workspace/compute_validation_20261028/runtime/cache \
    -e TORCH_HOME=/workspace/compute_validation_20261028/runtime/torch \
    -e TORCH_EXTENSIONS_DIR=/workspace/compute_validation_20261028/runtime/torch/extensions \
    -e TORCHINDUCTOR_CACHE_DIR=/workspace/compute_validation_20261028/runtime/torch/inductor \
    -e TRITON_CACHE_DIR=/workspace/compute_validation_20261028/runtime/triton \
    -e MIOPEN_USER_DB_PATH=/workspace/compute_validation_20261028/runtime/miopen \
    -e MIOPEN_CUSTOM_CACHE_DIR=/workspace/compute_validation_20261028/runtime/miopen \
    -e AMD_COMGR_CACHE_DIR=/workspace/compute_validation_20261028/runtime/comgr \
    -e CUDA_CACHE_PATH=/workspace/compute_validation_20261028/runtime/cache \
    -e HIP_LOG_OUTPUT_PATH=/workspace/compute_validation_20261028/runtime/hylog/ \
    -e HIP_LOG_ENABLE_FILE=0 -e HYLOG_DISABLE_ERROR_LOG=1 -e GLOG_logtostderr=1 \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/workspace \
    sunkai_compute "$@"
}

run_in_container python3 -B validate_gpu.py
