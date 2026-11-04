#!/usr/bin/env python3
"""Aggregate a pair of balanced production-MoE timing runs."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--prefix", default="timing")
parser.add_argument("--expected-cases", type=int, default=9)
args = parser.parse_args()
paths = [
    ROOT / "results" / f"{args.prefix}-{name}"
    for name in ("legacy-first", "swap-first")
]
runs = [json.loads((path / "results.json").read_text()) for path in paths]
assert all(run.get("complete") for run in runs), "Both timing runs must finish"


def key(arm):
    return arm["mode"], arm["activation"], arm["tokens"], arm["swap_ab"]


indexed = [{key(arm): arm for arm in run["arms"]} for run in runs]
assert indexed[0].keys() == indexed[1].keys()
assert len(indexed[0]) == args.expected_cases * 2, len(indexed[0])
rows = []
modes = [
    mode
    for mode in ("w4a4_scalar", "w4a4_per_token", "w4a16")
    if any(item[0] == mode for item in indexed[0])
]
selections = []
for mode in modes:
    for tokens in sorted({item[2] for item in indexed[0] if item[0] == mode}):
        row = {"mode": mode, "activation": "silu", "tokens": tokens}
        for swap_ab, name in ((False, "legacy"), (True, "swap_ab")):
            arms = [index[(mode, "silu", tokens, swap_ab)] for index in indexed]
            if runs[0]["arguments"].get("autotune"):
                assert (
                    arms[0]["autotune"]["cache_sha256"]
                    == arms[1]["autotune"]["cache_sha256"]
                )
                assert not arms[0]["autotune"]["loaded_existing_cache"]
                assert arms[1]["autotune"]["loaded_existing_cache"]
                assert (
                    arms[0]["selected_tactics_outside_tuning"][0]
                    == arms[1]["selected_tactics_outside_tuning"][0]
                )
                selections.append(
                    {
                        "mode": mode,
                        "tokens": tokens,
                        "swap_ab": swap_ab,
                        "cache_sha256": arms[0]["autotune"]["cache_sha256"],
                        "selected": arms[0]["selected_tactics_outside_tuning"][0],
                    }
                )
            assert all(len(arm["latency_us_samples"]) == 100 for arm in arms)
            samples = [sample for arm in arms for sample in arm["latency_us_samples"]]
            medians = [statistics.median(arm["latency_us_samples"]) for arm in arms]
            row[f"{name}_first_process_us"] = medians[0]
            row[f"{name}_second_process_us"] = medians[1]
            row[f"{name}_pooled_median_us"] = statistics.median(samples)
            row[f"{name}_repeat_spread_percent"] = 100 * (
                max(medians) / min(medians) - 1
            )
        row["legacy_over_swap_ab"] = (
            row["legacy_pooled_median_us"] / row["swap_ab_pooled_median_us"]
        )
        row["speedup_percent"] = 100 * (row["legacy_over_swap_ab"] - 1)
        rows.append(row)

dest = ROOT / "results" / f"{args.prefix}-summary"
dest.mkdir(parents=True, exist_ok=True)
with (dest / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "rows": rows,
    "total_samples": len(rows) * 400,
    "per_arm_per_case_samples": 200,
    "negative_rows": sum(row["speedup_percent"] < 0 for row in rows),
    "mode_geomean_ratios": {
        mode: math.exp(
            statistics.mean(
                math.log(row["legacy_over_swap_ab"])
                for row in rows
                if row["mode"] == mode
            )
        )
        for mode in modes
    },
    "maximum_repeat_spread_percent": max(
        row[f"{name}_repeat_spread_percent"]
        for row in rows
        for name in ("legacy", "swap_ab")
    ),
    "method": runs[0]["method"],
    "environment": runs[0]["environment"],
    "selections": selections,
}
(dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
lines = [
    "# Representative production MoE timing smoke",
    "",
    "Single C3 B300, local TP1/EP1 with 32 experts, H7168/I2048, top-k8, "
    "SwiGLU. Identical seeded serialized weights, inputs and routes. Both arms "
    "use FlashInfer merge 338924c2: legacy forces swap_ab=False; candidate uses "
    "swap_ab=True. This is not distributed EP8 or a model-throughput measurement.",
    "",
    "Two fresh-process sweeps, legacy→swap then swap→legacy. Each phase uses "
    "10 warmups and 100 untrimmed CUDA-event samples around a whole-forward "
    "CUDA graph replay, warm L2. Initial activation quantization is included; "
    "precomputed route selection, weight preparation, reload and graph capture "
    "are excluded. Default mode-specific tactics, no autotuning. No clock locking "
    "or CUPTI timing. Pooled medians use all200 samples per arm/case. Spread is "
    "max(two phase medians)/min(two phase medians)-1; all negative rows remain.",
    "",
    "| Mode | Tokens | Legacy first µs | Swap second µs | Swap first µs | Legacy second µs | Legacy pooled µs | Swap pooled µs | Speedup | Legacy spread | Swap spread |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['mode']} | {row['tokens']} | "
        f"{row['legacy_first_process_us']:.3f} | {row['swap_ab_first_process_us']:.3f} | "
        f"{row['swap_ab_second_process_us']:.3f} | {row['legacy_second_process_us']:.3f} | "
        f"{row['legacy_pooled_median_us']:.3f} | {row['swap_ab_pooled_median_us']:.3f} | "
        f"{row['speedup_percent']:+.2f}% | {row['legacy_repeat_spread_percent']:.2f}% | "
        f"{row['swap_ab_repeat_spread_percent']:.2f}% |"
    )
if selections:
    lines[0] = "# Targeted M16 production MoE autotune follow-up"
    lines[4] = lines[4].replace(
        "Default mode-specific tactics, no autotuning.",
        "Each mode/layout tunes exactly bucket16 before timing; the reverse-order "
        "process reuses the identical cache SHA256 and resolved tactic. Independent "
        "per-layout selections, not a common-tactic kernel comparison.",
    )
    lines += [
        "",
        "Selected tactics and immutable cache hashes:",
        "",
        "```json",
        json.dumps(selections, indent=2),
        "```",
    ]
(dest / "README.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
