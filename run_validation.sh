#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT=${1:-/scratch/ziangli/sglang-megamoe-autotune-api-cleanup/ready-20261111}
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT/sglang/python"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NVSHMEM_REMOTE_TRANSPORT=none
export NVSHMEM_IB_ENABLE_IBGDA=0
export NVSHMEM_DISABLE_LOCAL_ONLY_PROXY=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MAX_JOBS=8
export SGLANG_CACHE_DIR="$TASK_ROOT/cache/sglang"
export FLASHINFER_WORKSPACE_BASE="$TASK_ROOT/cache/flashinfer"
export CUTE_DSL_CACHE_DIR="$TASK_ROOT/cache/cute-dsl"
export TORCH_EXTENSIONS_DIR="$TASK_ROOT/cache/torch-extensions"
unset FLASHINFER_DISABLE_VERSION_CHECK
mkdir -p "$TASK_ROOT/logs/native" "$TASK_ROOT/cache"

export SGLANG_VALIDATION_COMMIT
SGLANG_VALIDATION_COMMIT=$(python - <<'PY'
import hashlib, json
from pathlib import Path
proof = json.loads(Path('validation-source-proof.json').read_text())
assert Path('source-tested-commit.txt').read_text().strip() == proof['commit']
for name, expected in proof['files'].items():
    assert hashlib.sha256((Path('sglang') / name).read_bytes()).hexdigest() == expected, name
assert hashlib.sha256(Path('smoke_autotune_gpu.py').read_bytes()).hexdigest() == proof['harness_sha256']
print(proof['commit'])
PY
)

python - <<'PY' > "$TASK_ROOT/logs/native-environment.json"
import importlib.metadata as metadata
import inspect, json, os, platform
from pathlib import Path
import flashinfer
import torch
from flashinfer.moe_ep import MoEEpMegaLayer
from sglang.srt.layers import dp_attention
from sglang.srt.layers.moe import flashinfer_megamoe
from sglang.srt.layers.moe import flashinfer_megamoe_autotune
assert flashinfer.__version__ == '0.7.4'
assert 'knobs' in inspect.signature(MoEEpMegaLayer.create_workspace).parameters
assert hasattr(MoEEpMegaLayer, 'get_tuned_knobs')
assert flashinfer_megamoe_autotune._CACHE_NAMESPACE == 'sglang_flashinfer_megamoe_knobs'
assert os.environ.get('FLASHINFER_DISABLE_VERSION_CHECK', '0') == '0'
source = Path('sglang/python').resolve()
paths = {name: str(Path(module.__file__).resolve()) for name, module in (
    ('adapter', flashinfer_megamoe), ('autotune', flashinfer_megamoe_autotune))}
assert all(Path(path).is_relative_to(source) for path in paths.values())
packages = {}
for name in ['flashinfer-python','flashinfer-cubin','flashinfer-jit-cache','torch','nvidia-cutlass-dsl','nvshmem4py-cu13','apache-tvm-ffi','pytest']:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({'source_commit': os.environ['SGLANG_VALIDATION_COMMIT'],
    'packages': packages, 'python': platform.python_version(), 'cuda': torch.version.cuda,
    'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    'capabilities': [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())],
    'module_sources': paths,
    'environment': {key: os.environ.get(key) for key in ['CUDA_VISIBLE_DEVICES','PYTHONPATH',
        'FLASHINFER_DISABLE_VERSION_CHECK','NVSHMEM_REMOTE_TRANSPORT','NVSHMEM_IB_ENABLE_IBGDA',
        'NVSHMEM_DISABLE_LOCAL_ONLY_PROXY','SGLANG_CACHE_DIR','FLASHINFER_WORKSPACE_BASE',
        'CUTE_DSL_CACHE_DIR','TORCH_EXTENSIONS_DIR','MAX_JOBS']}}, indent=2))
PY

python -m pytest -q \
  "$TASK_ROOT/sglang/test/registered/unit/layers/moe/test_flashinfer_megamoe.py" \
  2>&1 | tee "$TASK_ROOT/logs/registered-unit-tests.log"

python -m pytest -q \
  "$TASK_ROOT/sglang/test/registered/unit/model_executor/runner/test_flashinfer_autotune_sync.py::TestMegaMoEAutotuneStartup" \
  2>&1 | tee "$TASK_ROOT/logs/startup-bridge-unit-tests.log"

for precision in w4a16 w4a4 mxfp8; do
  for reuse in 0 1; do
    label="${precision}-silu-reuse${reuse}"
    cp validation-source-proof.json "$TASK_ROOT/logs/native/$label.source-proof.json"
    date -u +%FT%TZ > "$TASK_ROOT/logs/native/$label.started-utc"
    printf 'Starting %s\n' "$label"
    if timeout 1800s torchrun --standalone --nnodes=1 --nproc-per-node=4 \
      "$TASK_ROOT/smoke_autotune_gpu.py" \
      --cache "$TASK_ROOT/cache/adapter-profiles" \
      --precision "$precision" --activation silu --reuse-cache "$reuse" \
      > "$TASK_ROOT/logs/native/$label.log" 2>&1; then
      printf '0\n' > "$TASK_ROOT/logs/native/$label.exitcode"
    else
      result=$?
      printf '%s\n' "$result" > "$TASK_ROOT/logs/native/$label.exitcode"
      date -u +%FT%TZ > "$TASK_ROOT/logs/native/$label.finished-utc"
      tail -100 "$TASK_ROOT/logs/native/$label.log"
      exit "$result"
    fi
    date -u +%FT%TZ > "$TASK_ROOT/logs/native/$label.finished-utc"
    printf 'Passed %s\n' "$label"
    python - "$TASK_ROOT/logs/native/$label.log" <<'PY'
import json, sys
from pathlib import Path
for chunk in Path(sys.argv[1]).read_text().split('MEGAMOE_AUTOTUNE_SMOKE=')[1:]:
    record, _ = json.JSONDecoder().raw_decode(chunk)
    print('MEGAMOE_AUTOTUNE_SMOKE=' + json.dumps(record), flush=True)
PY
  done
done
