#!/bin/bash
set -eu
cd /root/sk/compute_mmac_20261028
mkdir -p runtime/{home,tmp,cache,torch,miopen,comgr,triton,hylog}
docker exec -w /workspace/compute_mmac_20261028 \
 -e HOME=/workspace/compute_mmac_20261028/runtime/home \
 -e TMPDIR=/workspace/compute_mmac_20261028/runtime/tmp \
 -e XDG_CACHE_HOME=/workspace/compute_mmac_20261028/runtime/cache \
 -e TORCH_HOME=/workspace/compute_mmac_20261028/runtime/torch \
 -e TORCH_EXTENSIONS_DIR=/workspace/compute_mmac_20261028/runtime/torch/extensions \
 -e TORCHINDUCTOR_CACHE_DIR=/workspace/compute_mmac_20261028/runtime/torch/inductor \
 -e TRITON_CACHE_DIR=/workspace/compute_mmac_20261028/runtime/triton \
 -e MIOPEN_USER_DB_PATH=/workspace/compute_mmac_20261028/runtime/miopen \
 -e MIOPEN_CUSTOM_CACHE_DIR=/workspace/compute_mmac_20261028/runtime/miopen \
 -e AMD_COMGR_CACHE_DIR=/workspace/compute_mmac_20261028/runtime/comgr \
 -e CUDA_CACHE_PATH=/workspace/compute_mmac_20261028/runtime/cache \
 -e HIP_LOG_OUTPUT_PATH=/workspace/compute_mmac_20261028/runtime/hylog/ \
 -e HYLOG_DISABLE_ERROR_LOG=1 -e HIP_LOG_ENABLE_FILE=0 -e GLOG_logtostderr=1 \
 -e PYTHONDONTWRITEBYTECODE=1 \
 sunkai_compute "$@"
