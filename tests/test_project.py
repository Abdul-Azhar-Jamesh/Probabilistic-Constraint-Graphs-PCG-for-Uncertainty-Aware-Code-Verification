"""Project boundaries and executed dependency diagnosis regression tests."""

from pathlib import Path
import os

import pytest

from pcg.execution import ExecutionConfig, run_project_tests
from pcg.project import analyze_project, discover, unpack_project
from pcg.project_graph import build_project_graph


def write(root: Path, files: dict[str, str]) -> None:
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def test_cross_file_failure_tracks_earlier_assignment(tmp_path):
    write(
        tmp_path,
        {
            "src/pkg/__init__.py": "",
            "src/pkg/base.py": "def denominator(n: int) -> int:\n    offset = n - 7\n    unrelated = n + 100\n    return offset\n",
            "src/pkg/calc.py": "from .base import denominator\ndef compute(n: int) -> float:\n    divisor = denominator(n)\n    return 1 / divisor\n",
            "tests/test_calc.py": "from pkg.calc import compute\ndef test_result():\n    assert compute(7) == 1\n",
        },
    )
    report = analyze_project(
        tmp_path, execution=ExecutionConfig("local", 30), max_generated_cases=12
    )
    assert report["status"] == "oracle-failure", report["execution"]
    diagnosis = next(d for d in report["diagnoses"] if "test_result" in d["test"])
    locations = {(c["file"], c["line"]) for c in diagnosis["candidate_causes"]}
    assert ("src/pkg/base.py", 2) in locations
    assert ("src/pkg/base.py", 3) not in locations
    assert ("src/pkg/calc.py", 4) in locations
    assert any(
        "cross-module-return" in e.get("relations", [])
        for e in report["graph"]["edges"]
    )


def test_new_code_automatically_finds_annotation_violation(tmp_path):
    write(
        tmp_path,
        {
            "logic.py": "def choose(n: int) -> int:\n    if n == 937:\n        return 'broken'\n    return n + 1\n"
        },
    )
    report = analyze_project(
        tmp_path, execution=ExecutionConfig("local", 30), max_generated_cases=16
    )
    assert report["summary"]["existing_test_files"] == 0
    assert report["summary"]["oracle_failures"] > 0, report["execution"]
    assert any(d["inputs"].get("n") == 937 for d in report["diagnoses"])
    assert report["testing_feedback"]
    assert report["summary"]["generated_checks"] <= 16
    assert report["adaptive_budget"]["reserved_cases"] > 0
    assert report["failure_validation"]["failures"]


def test_state_write_in_previous_call_is_retained(tmp_path):
    write(
        tmp_path,
        {
            "state.py": "class State:\n    def set(self):\n        self.divisor = 0\n    def get(self):\n        return 3 / self.divisor\n",
            "test_state.py": "from state import State\ndef test_state():\n    obj = State()\n    obj.set()\n    obj.get()\n",
        },
    )
    report = analyze_project(
        tmp_path, execution=ExecutionConfig("local", 30), max_generated_cases=6
    )
    diagnosis = next(
        d for d in report["diagnoses"] if "test_state.py::test_state" == d["test"]
    )
    assert {c["line"] for c in diagnosis["candidate_causes"]} >= {3, 5}


def test_same_named_functions_have_distinct_prior_families():
    bundle = build_project_graph(
        {"a.py": "def f():\n    return 1\n", "b.py": "def f():\n    return 2\n"}
    )
    assert len({b.qualname for b in bundle.blocks}) == len(bundle.blocks)


def test_worker_rejects_escaping_paths_and_reports_missing_dependency():
    with pytest.raises(ValueError):
        run_project_tests({"../bad.py": ""}, [], [], ExecutionConfig("local"))
    result = run_project_tests(
        {
            "pkg.py": "import pcg_nonexistent_dependency_xyz",
            "test_pkg.py": "import pkg",
        },
        ["pkg.py"],
        ["test_pkg.py"],
        ExecutionConfig("local", 15),
    )
    assert not result.valid
    assert result.status == "harness_error"


def test_discovery_and_cache_invalidate_on_source_change(tmp_path):
    write(
        tmp_path,
        {
            "maths.py": "def inc(n: int) -> int:\n    return n + 1\n",
            "out/stale.py": "raise RuntimeError()",
        },
    )
    assert "out/stale.py" not in discover(tmp_path).files
    config = ExecutionConfig("local", 30)
    cache = str(tmp_path / "out/cache")
    first = analyze_project(
        tmp_path, execution=config, cache_dir=cache, max_generated_cases=4
    )
    second = analyze_project(
        tmp_path, execution=config, cache_dir=cache, max_generated_cases=4
    )
    assert not first["summary"]["cache_hit"]
    assert second["summary"]["cache_hit"]
    write(tmp_path, {"maths.py": "def inc(n: int) -> int:\n    return 'bug'\n"})
    changed = analyze_project(
        tmp_path, execution=config, cache_dir=cache, max_generated_cases=4
    )
    assert not changed["summary"]["cache_hit"]
    assert changed["summary"]["oracle_failures"] > 0


def test_project_zip_is_validated_before_extracting(tmp_path):
    import io
    import zipfile

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("valid.py", "x=1")
        archive.writestr("../escape.py", "x=2")
    with pytest.raises(ValueError):
        unpack_project(payload.getvalue(), tmp_path)
    assert not (tmp_path / "valid.py").exists()


@pytest.mark.skipif(
    os.environ.get("PCG_TEST_DOCKER") != "1",
    reason="requires running Docker and built worker image",
)
def test_docker_preserves_src_package_and_records_trace():
    result = run_project_tests(
        {
            "src/pkg/__init__.py": "",
            "src/pkg/value.py": "def value(n):\n    return n + 1\n",
            "tests/test_value.py": "from pkg.value import value\ndef test_value():\n    assert value(3) == 4\n",
            "data/settings.json": '{"enabled": true}',
        },
        ["src/pkg/__init__.py", "src/pkg/value.py"],
        ["tests/test_value.py"],
        ExecutionConfig("docker", 30),
        import_roots=[".", "src"],
    )
    assert result.valid, result.errors
    assert result.tests[0].outcome == "passed"
    assert result.tests[0].file_lines["src/pkg/value.py"]
    assert result.tests[0].traces[0]["inputs"] == {"n": 3}
