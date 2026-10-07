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
    python_executable: str | None = None

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
    oracle: str = "user"
    initialization_lines: list[int] = field(default_factory=list)
    oracle_scope: str = "user"
    failure_details: str | None = None
    file_lines: dict[str, list[int]] = field(default_factory=dict)
    file_arcs: dict[str, list[list[int]]] = field(default_factory=dict)
    file_frames: list[dict] = field(default_factory=list)
    traces: list[dict] = field(default_factory=list)


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
    return run_project_tests(
        {"candidate.py": source, "test_candidate.py": tests, **config.files},
        ["candidate.py"],
        ["test_candidate.py"],
        config,
    )


def run_project_tests(
    files: dict[str, str],
    source_files: list[str],
    test_paths: list[str],
    config: ExecutionConfig | None = None,
    *,
    import_roots: list[str] | None = None,
) -> ExecutionResult:
    """Execute a bounded snapshot preserving its original package/import layout."""
    from pathlib import PurePosixPath

    config = config or ExecutionConfig()
    reserved = {"pcg_worker_plugin.py", "pcg_worker_trace.py", "pcg_worker_spec.json"}
    if len(files) > 1000 or sum(len(v.encode()) for v in files.values()) > 32_000_000:
        raise ValueError("project snapshot exceeds worker limits")
    for name, content in files.items():
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or ":" in name
            or name in reserved
            or len(content.encode()) > 2_000_000
        ):
            raise ValueError("invalid project snapshot path or file size")
    if not set(source_files) <= files.keys() or any(
        p.startswith("-") or p.split("::", 1)[0] not in files for p in test_paths
    ):
        raise ValueError("worker source and test paths must belong to snapshot")
    roots = import_roots or ["."]
    if any(
        PurePosixPath(p).is_absolute()
        or ".." in PurePosixPath(p).parts
        or ":" in p
        or "\\" in p
        for p in roots
    ):
        raise ValueError("invalid import root")
    if config.backend == "disabled":
        return ExecutionResult(status="disabled", backend=config.backend)
    if not test_paths or not any(
        files[p.split("::", 1)[0]].strip() for p in test_paths
    ):
        return ExecutionResult(status="no_tests", backend=config.backend)
    container = "pcg-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="pcg_worker_") as wd:
        root = Path(wd)
        for name, content in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        shutil.copyfile(
            Path(__file__).with_name("pytest_plugin.py"), root / "pcg_worker_plugin.py"
        )
        shutil.copyfile(
            Path(__file__).with_name("runtime_trace.py"), root / "pcg_worker_trace.py"
        )
        (root / "pcg_worker_spec.json").write_text(
            json.dumps(
                {"sources": source_files, "tests": test_paths, "import_roots": roots}
            ),
            encoding="utf-8",
        )
        result_path = root / "results.json"
        if config.backend == "docker":
            # Feed files over stdin; the container receives no host filesystem mounts.
            import io
            import tarfile

            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as archive:
                for name in (
                    *files,
                    "pcg_worker_plugin.py",
                    "pcg_worker_trace.py",
                    "pcg_worker_spec.json",
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
                config.python_executable or sys.executable,
                "-m",
                "pytest",
                *test_paths,
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
                    "PYTHONPATH": os.pathsep.join(
                        [str(root), *[str(root / p) for p in roots]]
                    ),
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
                except KeyboardInterrupt:
                    if config.backend == "docker":
                        subprocess.run(
                            ["docker", "rm", "-f", container],
                            capture_output=True,
                            check=False,
                        )
                    _kill_tree(proc)
                    raise
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
                or record.oracle not in {"user", "contract", "annotation", "probe"}
                or record.oracle_scope
                not in {
                    "user",
                    "example",
                    "contract",
                    "annotation",
                    "probe",
                    "property",
                    "metamorphic",
                }
                or any(type(line) is not int or line < 1 for line in record.lines)
                or any(
                    type(line) is not int or line < 1
                    for line in record.initialization_lines
                )
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
    baseline_checked = {t.nodeid: t for t in baseline.tests if t.oracle != "probe"}
    candidate_checked = {t.nodeid: t for t in candidate.tests if t.oracle != "probe"}
    baseline_passed = {n for n, t in baseline_checked.items() if t.outcome == "passed"}
    baseline_failed = {n for n, t in baseline_checked.items() if t.outcome == "failed"}
    candidate_passed = {
        n for n, t in candidate_checked.items() if t.outcome == "passed"
    }
    candidate_failed = {
        n for n, t in candidate_checked.items() if t.outcome == "failed"
    }
    completed_ids = {
        n for n, t in candidate_checked.items() if t.outcome in {"passed", "failed"}
    }
    return (
        baseline.valid
        and candidate.valid
        and {t.nodeid for t in baseline.tests} == {t.nodeid for t in candidate.tests}
        and {t.nodeid: (t.oracle, t.oracle_scope) for t in baseline.tests}
        == {t.nodeid: (t.oracle, t.oracle_scope) for t in candidate.tests}
        and baseline_passed <= candidate_passed
        and baseline_failed <= completed_ids
        and candidate_failed < baseline_failed
    )
