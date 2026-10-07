"""Container entrypoint accepting a bounded, traversal-free project snapshot."""

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as archive:
    seen = set()
    total_bytes = 0
    for member in archive:
        path = PurePosixPath(member.name)
        total_bytes += member.size
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in member.name
            or ":" in member.name
            or not member.isfile()
            or member.size > 2_000_000
            or member.name in seen
            or len(seen) >= 1003
            or total_bytes > 34_000_000
        ):
            raise ValueError("invalid worker input")
        seen.add(member.name)
        Path(member.name).parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(member) as source:
            Path(member.name).write_bytes(source.read())

spec = json.loads(Path("/work/pcg_worker_spec.json").read_text())
env = {
    "PATH": os.environ["PATH"],
    "PYTHONPATH": os.pathsep.join(
        ["/work", *[str(Path("/work") / p) for p in spec["import_roots"]]]
    ),
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PCG_RESULT_PATH": "/work/results.json",
}
with open("/work/pytest.log", "wb") as output:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *spec["tests"],
            "-q",
            "-p",
            "pcg_worker_plugin",
            "-p",
            "no:cacheprovider",
        ],
        env=env,
        stdout=output,
        stderr=output,
        check=False,
    )
sys.stdout.write(Path("/work/results.json").read_text())
