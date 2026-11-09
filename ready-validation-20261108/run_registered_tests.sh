#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${TASK_ROOT:-/scratch/ziangli/sglang-flashinfer-swap-ab/ready-20261108}
cd "$TASK_ROOT/sglang"
export PYTHONPATH="$TASK_ROOT/sglang/python"
export CUDA_VISIBLE_DEVICES=${VALIDATION_GPU:-0}
export FLASHINFER_CUDA_ARCH_LIST=10.3a
export FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY=1
export FLASHINFER_DIST_AWARE_AUTOTUNE=0
export CUTE_DSL_CACHE_DIR="$TASK_ROOT/cache/cute-gpu$CUDA_VISIBLE_DEVICES"
mkdir -p "$TASK_ROOT/logs" "$CUTE_DSL_CACHE_DIR"

python -m pytest -q --disable-warnings \
  test/registered/unit/layers/quantization/test_modelopt_nvfp4_moe_scales.py \
  test/registered/unit/layers/quantization/test_nvfp4_online_weight_loader.py \
  test/registered/unit/model_loader/test_presharded_loader.py \
  2>&1 | tee "$TASK_ROOT/logs/host-contracts.log"

python -m pytest -q --disable-warnings \
  test/registered/unit/layers/quantization/test_nvfp4_moe_backends.py \
  -k flashinfer_cutedsl \
  2>&1 | tee "$TASK_ROOT/logs/production-w4a4.log"

python -m pytest -q --disable-warnings \
  test/registered/unit/layers/quantization/test_nvfp4_cutedsl_w4a16_situ_moe.py \
  2>&1 | tee "$TASK_ROOT/logs/production-w4a16-situ.log"
