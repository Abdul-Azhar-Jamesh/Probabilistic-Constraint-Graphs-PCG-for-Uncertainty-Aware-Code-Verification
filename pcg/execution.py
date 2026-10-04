"""One execution contract for analysis, mutations and repair validation.

Local mode is for trusted code only. Docker mode fails closed and runs without
network, host mounts, application credentials, or a writable root filesystem.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExecutionConfig:
    backend: str = "local"
    timeout: int = 30
    image: str = "pcg-worker:latest"
    max_output_bytes: int = 2_000_000
    files: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.backend not in {"local", "docker", "disabled"}:
            raise ValueError("execution backend must be local, docker, or disabled")
        if self.timeout <= 0 or self.max_output_bytes <= 0:
            raise ValueError("execution limits must be positive")
        from pathlib import PurePosixPath

        if (
            len(self.files) > 200
            or sum(len(value.encode()) for value in self.files.values()) > 16_000_000
        ):
            raise ValueError("support files exceed worker limits")
        for name, content in self.files.items():
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or ":" in name
                or not name.endswith(".py")
                or name in {"candidate.py", "test_candidate.py", "pcg_worker_plugin.py"}
                or len(content.encode()) > 2_000_000
            ):
                raise ValueError(
                    "support files must be bounded relative Python source paths"
                )


@dataclass
class TestOutcome:
    __test__ = False
    nodeid: str
    outcome: str
    phases: dict[str, str] = field(default_factory=dict)
    duration: float = 0.0
    frames: list[int] = field(default_factory=list)
    exception: str | None = None
    lines: list[int] = field(default_factory=list)
    arcs: list[list[int]] = field(default_factory=list)


@dataclass
class ExecutionResult:
    tests: list[TestOutcome] = field(default_factory=list)
    status: str = "complete"
    errors: list[str] = field(default_factory=list)
    exit_code: int | None = None
    backend: str = "local"

    @property
    def passed_ids(self) -> set[str]:
        return {t.nodeid for t in self.tests if t.outcome == "passed"}

    @property
    def failed_ids(self) -> set[str]:
        return {t.nodeid for t in self.tests if t.outcome == "failed"}

    @property
    def valid(self) -> bool:
        return (
            self.status == "complete" and not self.errors and self.exit_code in (0, 1)
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        import signal

        try:
            getattr(os, "killpg")(proc.pid, getattr(signal, "SIGKILL"))
        except ProcessLookupError:
            pass
    proc.wait()


def run_tests(
    source: str, tests: str, config: ExecutionConfig | None = None
) -> ExecutionResult:
    """Run a single-file candidate with pytest; return observations, never regex counts."""
    config = config or ExecutionConfig()
    if len(source.encode()) > 2_000_000 or len(tests.encode()) > 2_000_000:
        raise ValueError("candidate source or tests exceed the worker input limit")
    if config.backend == "disabled":
        return ExecutionResult(status="disabled", backend=config.backend)
    if not tests.strip():
        return ExecutionResult(status="no_tests", backend=config.backend)
    container = "pcg-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="pcg_worker_") as wd:
        root = Path(wd)
        (root / "candidate.py").write_text(source, encoding="utf-8")
        (root / "test_candidate.py").write_text(tests, encoding="utf-8")
        shutil.copyfile(
            Path(__file__).with_name("pytest_plugin.py"), root / "pcg_worker_plugin.py"
        )
        for name, content in config.files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        result_path = root / "results.json"
        if config.backend == "docker":
            # Feed files over stdin; the container receives no host filesystem mounts.
            import io
            import tarfile

            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as archive:
                for name in (
                    "candidate.py",
                    "test_candidate.py",
                    "pcg_worker_plugin.py",
                    *config.files,
                ):
                    archive.add(root / name, arcname=name)
            command = [
                "docker",
                "run",
                "--rm",
                "-i",
                "--name",
                container,
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit=64",
                "--memory=512m",
                "--cpus=1",
                "--user=65534:65534",
                "--tmpfs=/work:rw,noexec,nosuid,size=64m,mode=1777",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=32m,mode=1777",
                config.image,
            ]
            stdin = stream.getvalue()
            env = None  # Docker receives no -e secrets from the host.
        else:
            command = [
                sys.executable,
                "-m",
                "pytest",
                "test_candidate.py",
                "-q",
                "--tb=short",
                "-p",
                "pcg_worker_plugin",
                "-p",
                "no:cacheprovider",
            ]
            stdin = None
            # Drop application secrets and prevent ambient plugins/options affecting results.
            allowed = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "LANG"}
            env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
            env.update(
                {
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                    "PYTHONPATH": wd,
                    "PCG_RESULT_PATH": str(result_path),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
        try:
            # Spool output to disk: a noisy candidate cannot exhaust parent process RAM.
            with (
                (root / "stdout.log").open("wb") as output,
                (root / "stderr.log").open("wb") as errors,
            ):
                proc = subprocess.Popen(
                    command,
                    cwd=wd,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=output,
                    stderr=errors,
                    start_new_session=os.name != "nt",
                )
                try:
                    proc.communicate(input=stdin, timeout=config.timeout)
                except subprocess.TimeoutExpired:
                    if config.backend == "docker":
                        subprocess.run(
                            ["docker", "rm", "-f", container],
                            capture_output=True,
                            check=False,
                        )
                    _kill_tree(proc)
                    return ExecutionResult(
                        status="timeout",
                        errors=["execution deadline exceeded"],
                        backend=config.backend,
                    )
            if config.backend == "docker":
                raw_path = root / "stdout.log"
                if raw_path.stat().st_size > config.max_output_bytes:
                    return ExecutionResult(
                        status="harness_error",
                        errors=["worker output limit exceeded"],
                        backend=config.backend,
                    )
                raw = raw_path.read_text(encoding="utf-8", errors="replace")
                payload = json.loads(raw)
            else:
                if (
                    not result_path.exists()
                    or result_path.stat().st_size > config.max_output_bytes
                ):
                    raise ValueError(
                        "worker did not produce a bounded structured result"
                    )
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                raise ValueError("worker returned an incompatible result schema")
            exit_code = int(payload["exit_code"])
            records = [TestOutcome(**row) for row in payload["tests"]]
            if len({record.nodeid for record in records}) != len(records) or any(
                record.outcome
                not in {"passed", "failed", "error", "skipped", "xfailed", "xpassed"}
                or any(type(line) is not int or line < 1 for line in record.lines)
                for record in records
            ):
                raise ValueError(
                    "worker returned invalid test identities, outcomes or coverage"
                )
            issues = list(payload.get("collection_errors", []))
            issues.extend(
                t.exception or t.nodeid for t in records if t.outcome == "error"
            )
            status = (
                "no_tests"
                if exit_code == 5
                else "complete"
                if exit_code in (0, 1)
                else "harness_error"
            )
            return ExecutionResult(records, status, issues, exit_code, config.backend)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return ExecutionResult(
                status="harness_error", errors=[str(exc)], backend=config.backend
            )


def preserves_tests(baseline: ExecutionResult, candidate: ExecutionResult) -> bool:
    """Require identical test identities and no regressions in previously passing tests."""
    completed_ids = {
        test.nodeid for test in candidate.tests if test.outcome in {"passed", "failed"}
    }
    return (
        baseline.valid
        and candidate.valid
        and {t.nodeid for t in baseline.tests} == {t.nodeid for t in candidate.tests}
        and baseline.passed_ids <= candidate.passed_ids
        and baseline.failed_ids <= completed_ids
        and candidate.failed_ids < baseline.failed_ids
    )
