"""Verify and summarize existing EP4 validation outputs without changing them."""

import hashlib
import json
import re
from pathlib import Path


root = Path(__file__).resolve().parent
proof = json.loads((root / "validation-source-proof.json").read_text())
unit = (root / "logs/registered-unit-tests.log").read_text()
assert not re.search(r"\b(?:FAILED|ERROR)\b", unit), unit[-2000:]
passed = re.search(r"(\d+) passed", unit)
assert passed, unit[-2000:]
bridge = (root / "logs/startup-bridge-unit-tests.log").read_text()
assert not re.search(r"\b(?:FAILED|ERROR)\b", bridge), bridge[-2000:]
bridge_passed = re.search(r"(\d+) passed", bridge)
assert bridge_passed, bridge[-2000:]
cases = []
for precision in ("w4a16", "w4a4", "mxfp8"):
    for reuse in (0, 1):
        label = f"{precision}-silu-reuse{reuse}"
        directory = root / "logs/native"
        assert (directory / f"{label}.exitcode").read_text().strip() == "0", label
        assert json.loads((directory / f"{label}.source-proof.json").read_text()) == proof
        path = directory / f"{label}.log"
        raw = path.read_bytes()
        log = raw.decode()
        records = [
            json.JSONDecoder().raw_decode(chunk)[0]
            for chunk in log.split("MEGAMOE_AUTOTUNE_SMOKE=")[1:]
        ]
        assert len(records) == 4 and {r["rank"] for r in records} == set(range(4)), label
        for record in records:
            assert record["passed"] and record["precision"] == precision
            assert record["reuse_cache"] == bool(reuse)
            assert record["flashinfer_version"] == "0.7.4"
            assert record["source_commit"] == proof["commit"]
            assert record["cache_namespace"] == "sglang_flashinfer_megamoe_knobs"
            assert record["auto_calls"] == (0 if reuse else 3)
            assert record["bit_exact"] and record["graph_replays"] == 9
            assert record["pool_refcounts"] == [2, 1, 0]
            assert not record["decode_workspace_created"]
            assert record["configured_decode_workspace_capacity"] == (
                0 if precision == "mxfp8" else 2
            )
            for name, file in (
                ("adapter", "python/sglang/srt/layers/moe/flashinfer_megamoe.py"),
                ("autotune", "python/sglang/srt/layers/moe/flashinfer_megamoe_autotune.py"),
            ):
                module = record["module_sources"][name]
                assert module["path"].endswith("/sglang/" + file)
                assert module["sha256"] == proof["files"][file]
        cases.append(
            {
                "label": label,
                "exit_code": 0,
                "log_sha256": hashlib.sha256(raw).hexdigest(),
                "records": records,
            }
        )

summary = {
    "source_commit": proof["commit"],
    "flashinfer_version": "0.7.4",
    "unit_tests_passed": int(passed.group(1)),
    "startup_bridge_unit_tests_passed": int(bridge_passed.group(1)),
    "native_cases_passed": len(cases),
    "native_rank_records": sum(len(c["records"]) for c in cases),
    "graph_replays": sum(r["graph_replays"] for c in cases for r in c["records"]),
    "all_outputs_bit_exact_to_fixed_control": True,
    "all_fresh_process_hits_use_zero_auto_calls": True,
    "all_pool_refcounts": [2, 1, 0],
    "version_check_bypassed": False,
    "cache_namespace": "sglang_flashinfer_megamoe_knobs",
    "harness_sha256": proof["harness_sha256"],
    "cases": cases,
}
(root / "validation-results.json").write_text(json.dumps(summary, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in summary.items() if k != "cases"},
        indent=2,
    )
)
