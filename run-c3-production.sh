#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/scratch/ziangli/sglang-flashinfer-swap-ab
cd "$TASK_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$TASK_ROOT/sglang/python:$TASK_ROOT/flashinfer"
export CUTE_DSL_CACHE_DIR="$TASK_ROOT/cache/cute-gpu1"
export FLASHINFER_CUDA_ARCH_LIST=10.3a
printf 'FlashInfer source: '
git -C flashinfer rev-parse HEAD
sha256sum validate_production_swap_ab.py \
    sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py \
    sglang/python/sglang/srt/layers/quantization/modelopt_quant.py
exec /opt/sglang/bin/python validate_production_swap_ab.py "$@"
