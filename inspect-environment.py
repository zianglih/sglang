import importlib.metadata as metadata
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
from packaging.requirements import Requirement

root = Path(__file__).resolve().parent
requirements = json.loads((root / "flashinfer-python-release.json").read_text())["info"]["requires_dist"]
requirement_status = []
for raw in requirements:
    requirement = Requirement(raw)
    if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
        continue
    try:
        version = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        version = None
    requirement_status.append({
        "requirement": raw, "installed": version,
        "satisfied": version is not None and requirement.specifier.contains(version, prereleases=True),
    })

versions = {}
for name in (
    "torch", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache",
    "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-cu13", "triton", "pytest",
    "sglang-kernel", "apache-tvm-ffi", "nvidia-nvshmem-cu13", "nvshmem4py-cu13",
):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None

nvshmem_files = {}
for name in ("nvidia-nvshmem-cu13", "nvshmem4py-cu13"):
    try:
        distribution = metadata.distribution(name)
        nvshmem_files[name] = [str(distribution.locate_file(p)) for p in distribution.files
                               if ".so" in str(p) or str(p).endswith("nvshmem.h")]
    except metadata.PackageNotFoundError:
        nvshmem_files[name] = []

info = {
    "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
    "versions": versions, "cuda_runtime": torch.version.cuda,
    "flashinfer_nonoptional_requirements": requirement_status,
    "missing_or_incompatible_requirements": [r for r in requirement_status if not r["satisfied"]],
    "nvshmem_files": nvshmem_files,
    "version_check_bypass": os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK"),
    "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
              "capability": torch.cuda.get_device_capability(i),
              "total_memory": torch.cuda.get_device_properties(i).total_memory}
             for i in range(torch.cuda.device_count())],
    "p2p": [[i == j or torch.cuda.can_device_access_peer(i, j)
             for j in range(torch.cuda.device_count())]
            for i in range(torch.cuda.device_count())],
    "nvidia_smi": subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv"
    ], text=True),
    "topology": subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
    "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
}
if "--check-api" in sys.argv:
    import flashinfer
    from flashinfer.autotuner import AutoTuner
    from flashinfer.moe_ep import MoEEpMegaLayer
    from sglang.srt.layers.moe import flashinfer_megamoe, flashinfer_megamoe_autotune

    assert flashinfer.__version__ == "0.7.4"
    assert hasattr(MoEEpMegaLayer, "get_tuned_knobs")
    assert "knobs" in inspect.signature(MoEEpMegaLayer.create_workspace).parameters
    assert hasattr(AutoTuner, "get_namespaced_records")
    assert hasattr(AutoTuner, "publish_namespaced_records")
    assert not info["version_check_bypass"], "Version check bypass must stay unset"
    info["api"] = {
        "get_tuned_knobs": str(inspect.signature(MoEEpMegaLayer.get_tuned_knobs)),
        "create_workspace": str(inspect.signature(MoEEpMegaLayer.create_workspace)),
        "get_namespaced_records": str(inspect.signature(AutoTuner.get_namespaced_records)),
        "publish_namespaced_records": str(inspect.signature(AutoTuner.publish_namespaced_records)),
    }
    info["source_paths"] = {
        "flashinfer": flashinfer.__file__,
        "MoEEpMegaLayer": inspect.getsourcefile(MoEEpMegaLayer),
        "sglang_megamoe": flashinfer_megamoe.__file__,
        "sglang_autotune": flashinfer_megamoe_autotune.__file__,
    }

print(json.dumps(info, indent=2))
if info["missing_or_incompatible_requirements"]:
    raise SystemExit(1)
