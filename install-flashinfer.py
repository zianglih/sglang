import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
lock = json.loads((root / "wheel-lock.json").read_text())
wheels = root / "wheels"
wheels.mkdir(exist_ok=True)


def packages():
    output = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--format=json"], text=True
    )
    return {p["name"].lower().replace("_", "-"): p["version"] for p in json.loads(output)}


before = packages()
(root / "packages-before.json").write_text(json.dumps(before, indent=2))
paths = []
for wheel in lock:
    start = time.monotonic()
    path = wheels / wheel["filename"]
    subprocess.check_call([
        "curl", "--fail", "--location", "--retry", "3", "--silent", "--show-error",
        wheel["url"], "-o", str(path),
    ])
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    assert digest.hexdigest() == wheel["sha256"], wheel["filename"]
    print(f"Verified {wheel['filename']}: {path.stat().st_size} bytes, "
          f"sha256={digest.hexdigest()}, {time.monotonic() - start:.2f}s", flush=True)
    paths.append(str(path))

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", *paths,
])
after = packages()
(root / "packages-after.json").write_text(json.dumps(after, indent=2))
changes = {
    name: {"before": before.get(name), "after": after.get(name)}
    for name in sorted(set(before) | set(after))
    if before.get(name) != after.get(name)
}
(root / "package-changes.json").write_text(json.dumps(changes, indent=2))
assert set(changes) <= {wheel["name"] for wheel in lock}, changes
for wheel in lock:
    assert after[wheel["name"]] == wheel["version"], (wheel, after)
print(json.dumps(changes, indent=2), flush=True)
