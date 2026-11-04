# CuTe DSL MoE swap-AB integration validation

## Sources and runtime

- SGLang candidate: `354f91325a07fe327b10a29f4f2239a94cd5fa2e`.
- SGLang base: `58484246d26523eaf1a7f16ab524deee288d2120`.
- FlashInfer: exact #7290 merge `338924c2541673cedb6098185690eb54f26a5cde`.
- C3: 8 NVIDIA B300 SXM6 AC GPUs, 275040 MiB each, SM103, driver 595.91.07.
- Tests run on GPU0; layer/graph/reload comparisons run on GPU1. Each is a
  single-GPU workload, not a distributed throughput measurement.
- Image: `lmsysorg/sglang:nightly-dev-cu13-20261103-797af717`.
- AMD64 digest: `sha256:ccf61cf0262e29f002d42410e39583e702a23c9bfc92616b7f8195a7b4a6da29`.
- Python 3.12.16, PyTorch 2.14.1+cu130, CUDA 13.0, NVCC 13.0.88,
  CuTe DSL 4.8.0, pytest 9.2.0, sglang-kernel 0.4.10, Triton 3.8.0.
- FlashInfer's source metadata reports 0.7.3. The image's 0.7.2 cubin/JIT
  packages are ignored by the version check. No version-check override is used.

## Reproduction inside the image

```bash
git clone https://github.com/zianglih/sglang.git
git -C sglang checkout 354f91325a07fe327b10a29f4f2239a94cd5fa2e
git clone https://github.com/flashinfer-ai/flashinfer.git
git -C flashinfer checkout 338924c2541673cedb6098185690eb54f26a5cde
git -C flashinfer submodule update --init --depth=1 --jobs=3 \
  3rdparty/cutlass 3rdparty/spdlog 3rdparty/cccl
BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1 \
  python -m pip install --no-deps --no-build-isolation -e ./flashinfer
export PYTHONPATH="$PWD/sglang/python:$PWD/flashinfer"
export CUDA_VISIBLE_DEVICES=0 FLASHINFER_CUDA_ARCH_LIST=10.3a
export FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY=1 FLASHINFER_DIST_AWARE_AUTOTUNE=0
export CUTE_DSL_CACHE_DIR="$PWD/cute-cache"
cd sglang
python -m pytest -q --disable-warnings \
  test/registered/unit/layers/quantization/test_modelopt_nvfp4_moe_scales.py \
  test/registered/unit/layers/quantization/test_nvfp4_online_weight_loader.py \
  test/registered/unit/model_loader/test_presharded_loader.py
python -m pytest -q --disable-warnings \
  test/registered/unit/layers/quantization/test_nvfp4_moe_backends.py \
  -k flashinfer_cutedsl
python -m pytest -q --disable-warnings \
  test/registered/unit/layers/quantization/test_nvfp4_cutedsl_w4a16_situ_moe.py
```

`validate_production_swap_ab.py` additionally exercises the real FusedMoE
checkpoint loader, preparation, forward, CUDA graph replay, and reload of a
second checkpoint while checking captured weight/scale pointers. Both layouts
use identical quantized checkpoints, routes, and inputs with the same FlashInfer
merge. The legacy arm forces only SGLang's shared swap-AB selector to false.

The default matrix uses E8, H1024, I1024, top4, M32, TP1/EP1: W4A4 scalar,
W4A4 per-token, and W4A16 each with SwiGLU and ReLU2, plus W4A16 SiTU.
W4A16 is also checked against independent FP32 matmuls with dequantized weights
and BF16 FC1 activation output. W4A4 uses cross-layout comparison in this
harness; the separate existing production test supplies its numerical reference.

```bash
unset FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY FLASHINFER_DIST_AWARE_AUTOTUNE
python /path/to/validate_production_swap_ab.py --output /path/to/correctness
python /path/to/validate_production_swap_ab.py \
  --hidden 7168 --intermediate 2048 --experts 32 --topk 8 \
  --tokens 16,128,4096 \
  --cases w4a4_scalar:silu,w4a4_per_token:silu,w4a16:silu \
  --timing --warmup 10 --iters 100 --output /path/to/layer-timing
```

Optional timings are whole production-forward CUDA graphs with precomputed
routes and initial activation quantization included, using CUDA events and warm
L2. Preparation, reload, JIT, and graph capture are outside timing. Each mode
uses its default tactics without autotuning. These are scoped layer checks,
not frozen-tactic kernel gains or full-model throughput measurements.

## Results

Focused registered tests passed (zero failures/skips):

```text
70 passed, 20 warnings, 68 subtests passed in 23.25s
1 passed, 3 deselected, 46 warnings in 408.80s (0:06:48)
1 passed, 126 warnings in 22.73s
```

The 408.80 s includes first-time native JIT compilation. The graph/reload matrix passed all 14 arms / 7 matched cases. Every initial
and reloaded legacy/swap pair was bitwise identical. Every captured graph
matched eager output exactly before and after reload; captured weight,
MMA-scale, and resolved-scale pointers remained fixed. All W4A16 independent
reference checks passed, with minimum cosine similarity 0.9999971986.
The full untuned timing table is retained below. The targeted autotuned M16 follow-up below passed and removes the regression for these two cases; it does not replace the untuned table.

## Validation boundary

No full-model serving/accuracy run, distributed all-to-all run, sanitizer run,
or native SM100/SM107/SM120 qualification. The architecture selector and cache
keys are tested with mocked capability values; GPU execution is on SM103.
The SGLang dependency bump remains a separate upstream prerequisite.

## Untuned timing results

Single C3 B300, local TP1/EP1 with 32 experts, H7168/I2048, top-k8, SwiGLU. Identical seeded serialized weights, inputs and routes. Both arms use FlashInfer merge 338924c2: legacy forces swap_ab=False; candidate uses swap_ab=True. This is not distributed EP8 or a model-throughput measurement.

Two fresh-process sweeps, legacy→swap then swap→legacy. Each phase uses 10 warmups and 100 untrimmed CUDA-event samples around a whole-forward CUDA graph replay, warm L2. Initial activation quantization is included; precomputed route selection, weight preparation, reload and graph capture are excluded. Default mode-specific tactics, no autotuning. No clock locking or CUPTI timing. Pooled medians use all 200 samples per arm/case. Speedup = legacy pooled latency / swap pooled latency - 1. Spread is max(two phase medians)/min(two phase medians)-1; all negative rows remain.

| Mode | Tokens | Legacy first µs | Swap second µs | Swap first µs | Legacy second µs | Legacy pooled µs | Swap pooled µs | Speedup | Legacy spread | Swap spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w4a4_scalar | 16 | 146.800 | 251.088 | 250.496 | 146.624 | 146.736 | 250.880 | -41.51% | 0.12% | 0.24% |
| w4a4_scalar | 128 | 152.848 | 143.312 | 143.840 | 153.328 | 153.120 | 143.504 | +6.70% | 0.31% | 0.37% |
| w4a4_scalar | 4096 | 990.592 | 1017.392 | 1003.168 | 974.832 | 988.576 | 1004.000 | -1.54% | 1.62% | 1.42% |
| w4a4_per_token | 16 | 159.712 | 258.528 | 258.560 | 160.496 | 160.160 | 258.528 | -38.05% | 0.49% | 0.01% |
| w4a4_per_token | 128 | 167.088 | 158.912 | 158.880 | 167.280 | 167.168 | 158.912 | +5.20% | 0.11% | 0.02% |
| w4a4_per_token | 4096 | 1057.040 | 1079.952 | 1079.680 | 1056.448 | 1056.576 | 1079.952 | -2.16% | 0.06% | 0.03% |
| w4a16 | 16 | 171.600 | 171.952 | 171.024 | 171.376 | 171.488 | 171.312 | +0.10% | 0.13% | 0.54% |
| w4a16 | 128 | 180.896 | 180.912 | 178.208 | 177.456 | 179.696 | 179.968 | -0.15% | 1.94% | 1.52% |
| w4a16 | 4096 | 2321.936 | 2275.024 | 2304.928 | 2291.664 | 2313.088 | 2290.576 | +0.98% | 1.32% | 1.31% |

## Targeted autotuned M16 follow-up

Single C3 B300, local TP1/EP1 with 32 experts, H7168/I2048, top-k8, SwiGLU. Identical seeded serialized weights, inputs and routes. Both arms use FlashInfer merge 338924c2: legacy forces swap_ab=False; candidate uses swap_ab=True. This is not distributed EP8 or a model-throughput measurement.

Two fresh-process sweeps, legacy→swap then swap→legacy. Each phase uses 10 warmups and 100 untrimmed CUDA-event samples around a whole-forward CUDA graph replay, warm L2. Initial activation quantization is included; precomputed route selection, weight preparation, reload and graph capture are excluded. Each mode/layout tunes exactly bucket 16 before timing; the reverse-order process reuses the identical cache SHA256 and resolved tactic. Independent per-layout selections, not a common-tactic kernel comparison. No clock locking or CUPTI timing. Pooled medians use all 200 samples per arm/case. Spread is max(two phase medians)/min(two phase medians)-1; all negative rows remain.

| Mode | Tokens | Legacy first µs | Swap second µs | Swap first µs | Legacy second µs | Legacy pooled µs | Swap pooled µs | Speedup | Legacy spread | Swap spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w4a4_scalar | 16 | 146.912 | 132.224 | 131.328 | 147.296 | 147.136 | 131.712 | +11.71% | 0.26% | 0.68% |
| w4a4_per_token | 16 | 160.384 | 138.896 | 138.192 | 160.352 | 160.352 | 138.368 | +15.89% | 0.02% | 0.51% |

Selected tactics and immutable cache hashes:

```json
[
  {
    "mode": "w4a4_scalar",
    "tokens": 16,
    "swap_ab": false,
    "cache_sha256": "42a10dd4225e215a8d0eba25cc89e35b20610b0081902013893f46825eb7e883",
    "selected": {
      "op": "CuteDslMoEWrapper::run::w4a4::Swiglu",
      "runner": "CuteDslFusedMoERunner",
      "tactic": [
        128,
        [
          [
            128,
            128
          ],
          [
            1,
            1
          ],
          false
        ],
        [
          [
            128,
            128
          ],
          [
            1,
            1
          ],
          false
        ]
      ],
      "tuning_mode": false
    }
  },
  {
    "mode": "w4a4_scalar",
    "tokens": 16,
    "swap_ab": true,
    "cache_sha256": "1e284d2d7d7ff6eb0a4d6b2e56b8377c85b37a2519b5c9b1dc64ba1e0133f3b6",
    "selected": {
      "op": "CuteDslMoEWrapper::run::w4a4::Swiglu::SwapAB",
      "runner": "CuteDslFusedMoERunner",
      "tactic": [
        16,
        [
          [
            256,
            16
          ],
          [
            2,
            1
          ],
          false
        ],
        [
          [
            256,
            16
          ],
          [
            2,
            1
          ],
          false
        ],
        [
          8,
          8
        ],
        true
      ],
      "tuning_mode": false
    }
  },
  {
    "mode": "w4a4_per_token",
    "tokens": 16,
    "swap_ab": false,
    "cache_sha256": "3336bfb357a710db09db5313bb7ecfd7b161298faa0e99b9b7a0c22180215489",
    "selected": {
      "op": "CuteDslMoEWrapper::run::w4a4::Swiglu",
      "runner": "CuteDslFusedMoERunner",
      "tactic": [
        128,
        [
          [
            128,
            128
          ],
          [
            1,
            1
          ],
          false
        ],
        [
          [
            128,
            128
          ],
          [
            1,
            1
          ],
          false
        ]
      ],
      "tuning_mode": false
    }
  },
  {
    "mode": "w4a4_per_token",
    "tokens": 16,
    "swap_ab": true,
    "cache_sha256": "a4d26f53c4eabf1c8a28dd64bac3dd7ab52e5ab8162c5b3298d0f0da6860c580",
    "selected": {
      "op": "CuteDslMoEWrapper::run::w4a4::Swiglu::SwapAB",
      "runner": "CuteDslFusedMoERunner",
      "tactic": [
        16,
        [
          [
            128,
            16
          ],
          [
            1,
            1
          ],
          false
        ],
        [
          [
            128,
            16
          ],
          [
            1,
            1
          ],
          false
        ],
        [
          8,
          4
        ],
        true
      ],
      "tuning_mode": false
    }
  }
]
```

## Reproduce the balanced sweeps

From the parent of the source checkouts, obtain the harness from this evidence
branch (the PR links an immutable evidence commit):

```bash
git -C sglang fetch origin evidence/flashinfer-cutedsl-swap-ab-20261104
git -C sglang show FETCH_HEAD:validate_production_swap_ab.py > validate_production_swap_ab.py
unset FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY FLASHINFER_DIST_AWARE_AUTOTUNE
for order in legacy-first swap-first; do
  python validate_production_swap_ab.py \
    --hidden 7168 --intermediate 2048 --experts 32 --topk 8 \
    --tokens 16,128,4096 \
    --cases w4a4_scalar:silu,w4a4_per_token:silu,w4a16:silu \
    --timing --warmup 10 --iters 100 --arm-order "$order" \
    --output "timing-$order"
done
for order in legacy-first swap-first; do
  python validate_production_swap_ab.py \
    --hidden 7168 --intermediate 2048 --experts 32 --topk 8 \
    --tokens 16 --cases w4a4_scalar:silu,w4a4_per_token:silu \
    --timing --warmup 10 --iters 100 --arm-order "$order" \
    --autotune --tactic-cache-dir tuned-cache --output "tuned-$order"
done
```

Use an initially empty `tuned-cache` directory for a fresh selection. The second
process reuses and verifies the first process's immutable selections. Archived
cache JSON and actual selected-tactic traces are included beside the raw data.

All 11,880 transferred SGLang source files were verified unchanged after the
runs; the FlashInfer checkout remained clean at the exact merge commit.
