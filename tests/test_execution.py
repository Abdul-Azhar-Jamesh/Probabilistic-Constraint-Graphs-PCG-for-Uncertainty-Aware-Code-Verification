"""Integration checks for the worker's observation and isolation contracts."""

import os

import pytest

from pcg.blocks import extract_blocks
from pcg.evidence import execution_evidence
from pcg.execution import (
    ExecutionConfig,
    ExecutionResult,
    TestOutcome,
    preserves_tests,
    run_tests,
)


SOURCE = """
def helper(x):
    return x + 1

def caller(x):
    return helper(x) * 2

def unused():
    return 999
"""


def test_coverage_and_structured_outcomes():
    tests = """
import pytest
from candidate import caller, helper
def test_pass():
    assert caller(1) == 4
def test_failure():
    assert helper(1) == 99
@pytest.mark.skip(reason='intentional')
def test_skip():
    pass
@pytest.mark.parametrize('x', [1, 2])
def test_parameters(x):
    assert helper(x) == x + 1
"""
    result = run_tests(SOURCE, tests)
    assert result.valid
    assert len(result.passed_ids) == 3
    assert len(result.failed_ids) == 1
    blocks = extract_blocks(SOURCE)
    ids = {b.name: b.bid for b in blocks}
    evidence = execution_evidence(result, blocks)
    assert not any(e.bid == ids["unused"] for e in evidence)
    assert any(e.bid == ids["helper"] and e.kind == "test_fail" for e in evidence)
    assert any(e.bid == ids["caller"] and e.kind == "test_pass" for e in evidence)
    assert not any(e.bid == ids["caller"] and e.kind == "test_fail" for e in evidence)
    assert any(e.meta["branches"] for e in evidence)


def test_setup_failure_is_not_candidate_defect():
    result = run_tests(SOURCE, "def test_missing(missing_fixture):\n    pass\n")
    assert not result.valid
    evidence = execution_evidence(result, extract_blocks(SOURCE))
    assert evidence and all(e.polarity == "neutral" and e.weight == 0 for e in evidence)


def test_collection_error_and_no_tests_are_distinct():
    bad = run_tests(SOURCE, "import nonexistent_pcg_dependency\n")
    empty = run_tests(SOURCE, "# deliberately no tests\n")
    assert bad.status == "harness_error" and bad.errors
    assert empty.status == "no_tests" and not empty.valid


def test_secrets_are_not_inherited(monkeypatch):
    monkeypatch.setenv("PCG_TEST_SECRET", "never expose this")
    result = run_tests(
        SOURCE,
        "import os\ndef test_environment():\n    assert 'PCG_TEST_SECRET' not in os.environ\n",
    )
    assert result.valid and result.passed_ids


def test_deadline_returns_neutral_evidence():
    result = run_tests(
        "def hang():\n    while True:\n        pass\n",
        "from candidate import hang\ndef test_hang():\n    hang()\n",
        ExecutionConfig(timeout=2),
    )
    assert result.status == "timeout"
    assert not result.valid


def test_disabled_execution_does_not_launch(monkeypatch):
    monkeypatch.setattr(
        "subprocess.Popen", lambda *a, **kw: pytest.fail("unexpected process launch")
    )
    assert (
        run_tests(SOURCE, "raise Exception()", ExecutionConfig("disabled")).status
        == "disabled"
    )


def test_support_modules_and_path_validation():
    result = run_tests(
        "from helpers.maths import add\ndef f(x):\n    return add(x, 1)\n",
        "from candidate import f\ndef test_f():\n    assert f(1) == 2\n",
        ExecutionConfig(
            files={
                "helpers/__init__.py": "",
                "helpers/maths.py": "def add(x, y):\n    return x + y\n",
            }
        ),
    )
    assert result.valid and result.passed_ids
    with pytest.raises(ValueError, match="relative"):
        ExecutionConfig(files={"../outside.py": "pass"})


def test_docker_unavailable_never_falls_back(monkeypatch):
    commands = []

    def missing(command, **kwargs):
        commands.append(command)
        raise FileNotFoundError("docker unavailable")

    monkeypatch.setattr("subprocess.Popen", missing)
    result = run_tests(SOURCE, "def test_ok():\n    pass\n", ExecutionConfig("docker"))
    assert result.status == "harness_error"
    assert len(commands) == 1 and commands[0][0] == "docker"
    command = commands[0]
    assert "--network=none" in command and "--read-only" in command
    assert not any(arg.startswith("--volume") or arg == "-v" for arg in command)


def outcome(passed, failed):
    return ExecutionResult(
        [
            *[TestOutcome(n, "passed") for n in passed],
            *[TestOutcome(n, "failed") for n in failed],
        ],
        exit_code=1 if failed else 0,
    )


def test_repair_cannot_trade_or_remove_tests():
    baseline = outcome({"a", "b"}, {"c", "d", "e"})
    assert not preserves_tests(baseline, outcome({"b", "c", "d", "e"}, {"a"}))
    assert not preserves_tests(baseline, outcome({"a", "b", "c"}, {"d"}))
    assert preserves_tests(baseline, outcome({"a", "b", "c"}, {"d", "e"}))
    skipped = outcome({"a", "b", "c"}, {"d"})
    skipped.tests.append(TestOutcome("e", "skipped"))
    assert not preserves_tests(baseline, skipped)


@pytest.mark.skipif(
    os.environ.get("PCG_TEST_DOCKER") != "1",
    reason="requires running Docker and built worker image",
)
def test_real_docker_isolation():
    result = run_tests(
        SOURCE,
        """
import os, socket
from candidate import helper
def test_ok():
    assert helper(1) == 2
    assert 'ANTHROPIC_API_KEY' not in os.environ
    assert not os.path.exists('/host')
def test_network_is_disabled():
    try:
        socket.create_connection(('1.1.1.1', 443), timeout=1)
    except OSError:
        return
    assert False, 'unexpected network access'
""",
        ExecutionConfig("docker"),
    )
    assert result.valid and len(result.passed_ids) == 2
