# Targeted M16 production MoE autotune follow-up

Single C3 B300, local TP1/EP1 with 32 experts, H7168/I2048, top-k8, SwiGLU. Identical seeded serialized weights, inputs and routes. Both arms use FlashInfer merge 338924c2: legacy forces swap_ab=False; candidate uses swap_ab=True. This is not distributed EP8 or a model-throughput measurement.

Two fresh-process sweeps, legacy→swap then swap→legacy. Each phase uses 10 warmups and 100 untrimmed CUDA-event samples around a whole-forward CUDA graph replay, warm L2. Initial activation quantization is included; precomputed route selection, weight preparation, reload and graph capture are excluded. Each mode/layout tunes exactly bucket16 before timing; the reverse-order process reuses the identical cache SHA256 and resolved tactic. Independent per-layout selections, not a common-tactic kernel comparison. No clock locking or CUPTI timing. Pooled medians use all200 samples per arm/case. Spread is max(two phase medians)/min(two phase medians)-1; all negative rows remain.

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
