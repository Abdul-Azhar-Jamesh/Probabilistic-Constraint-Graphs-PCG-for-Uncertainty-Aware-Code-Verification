"""Container entrypoint. Only allow the three expected input files."""

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
            or not member.name.endswith(".py")
            or not member.isfile()
            or member.size > 2_000_000
            or member.name in seen
            or len(seen) >= 203
            or total_bytes > 20_000_000
        ):
            raise ValueError("invalid worker input")
        seen.add(member.name)
        Path(member.name).parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(member) as source:
            Path(member.name).write_bytes(source.read())

env = {
    "PATH": os.environ["PATH"],
    "PYTHONPATH": "/work",
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
            "test_candidate.py",
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
