#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="${TASK_ROOT:-/scratch/ziangli/sglang-flashinfer-swap-ab/ready-20261108}"
validation_python="${VALIDATION_PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${VALIDATION_GPU:-1}"
export PYTHONPATH="$TASK_ROOT/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
export FLASHINFER_CUDA_ARCH_LIST=10.3a
export CUTE_DSL_CACHE_DIR="$TASK_ROOT/cache/cute-graph-reload-gpu$CUDA_VISIBLE_DEVICES"
mkdir -p "$TASK_ROOT/logs" "$CUTE_DSL_CACHE_DIR"
cd "$TASK_ROOT"

"$validation_python" - <<'PY' | tee "$TASK_ROOT/logs/graph-reload-environment.log"
from pathlib import Path
import flashinfer
import sglang
import torch

assert flashinfer.__version__ == "0.7.4rc1", flashinfer.__version__
assert Path(sglang.__file__).resolve().is_relative_to(Path("sglang/python").resolve())
assert torch.cuda.get_device_capability() in ((10, 0), (10, 3))
print("FlashInfer:", flashinfer.__version__, flashinfer.__file__, flush=True)
print("SGLang:", sglang.__file__, flush=True)
print("GPU:", torch.cuda.get_device_name(), flush=True)
PY

# Match the original 14-arm correctness matrix, without recording timings.
export FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY=1
export FLASHINFER_DIST_AWARE_AUTOTUNE=0
"$validation_python" validate_production_swap_ab.py \
  --hidden 1024 --intermediate 1024 --experts 8 --topk 4 \
  --tokens 32 --port 29673 --output correctness-matrix \
  2>&1 | tee "$TASK_ROOT/logs/graph-reload.log"

# Exercise the intended autotuned W4A4 configuration. This only validates
# numerics, graph replay, reload, and frozen tactic selection; no timing run.
unset FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY FLASHINFER_DIST_AWARE_AUTOTUNE
"$validation_python" validate_production_swap_ab.py \
  --hidden 7168 --intermediate 2048 --experts 32 --topk 8 \
  --tokens 16 --cases w4a4_scalar:silu,w4a4_per_token:silu \
  --autotune --tactic-cache-dir tuned-correctness-cache \
  --port 29674 --output autotuned-correctness \
  2>&1 | tee "$TASK_ROOT/logs/autotuned-correctness.log"
