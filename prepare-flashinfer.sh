#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/scratch/ziangli/sglang-flashinfer-swap-ab
mkdir -p "$TASK_ROOT" "$TASK_ROOT/logs" "$TASK_ROOT/cache/cute"
cd "$TASK_ROOT"
if [[ ! -d flashinfer/.git ]]; then
  git init flashinfer
  git -C flashinfer remote add origin https://github.com/flashinfer-ai/flashinfer.git
fi
git -C flashinfer fetch --depth=1 origin 338924c2541673cedb6098185690eb54f26a5cde
git -C flashinfer checkout --detach 338924c2541673cedb6098185690eb54f26a5cde
git -C flashinfer submodule update --init --depth=1 --jobs=3 3rdparty/cutlass 3rdparty/spdlog 3rdparty/cccl
cd flashinfer
BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1 python -m pip install --no-deps --no-build-isolation -e .
python - <<'PY'
import flashinfer, inspect
from flashinfer.fused_moe import CuteDslMoEWrapper
from flashinfer.fused_moe import interleave_cute_dsl_moe_weights
print('flashinfer', flashinfer.__version__, flashinfer.__file__)
print('CuteDslMoEWrapper', inspect.signature(CuteDslMoEWrapper))
print('interleave_cute_dsl_moe_weights', inspect.signature(interleave_cute_dsl_moe_weights))
PY
