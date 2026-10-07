"""Failure reduction must preserve the oracle and candidate symptom."""

from pcg.execution import (
    ExecutionConfig,
    ExecutionResult,
    TestOutcome,
    run_project_tests,
)
from pcg.failure_validation import validate_failures


def test_reducer_shrinks_input_and_retains_the_same_symptom():
    files = {
        "candidate.py": "def bad(xs):\n    if xs and xs[0] > 0:\n        return 'bad'\n    return 0\n",
        "test_candidate.py": "from candidate import bad\ndef test_bad():\n    unused = 999999\n    xs = [22, 33, 44]\n    assert isinstance(bad(xs), int)\n",
    }
    config = ExecutionConfig("local", 15)
    initial = run_project_tests(files, ["candidate.py"], ["test_candidate.py"], config)
    report = validate_failures(
        files, ["candidate.py"], initial, config, minimize_trials=12
    )
    assert report["failures"][0]["status"] == "stable-failure"
    reduced = report["reproducers"][0]
    assert reduced["status"] == "reduced"
    assert "unused" not in reduced["source"]
    assert "assert isinstance(bad(xs), int)" in reduced["source"]
    assert reduced["reduced_bytes"] < reduced["original_bytes"]
    import ast

    assignment = next(
        n
        for n in ast.walk(ast.parse(reduced["source"]))
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "xs" for t in n.targets)
    )
    assert len(ast.literal_eval(assignment.value)) < 3
    assert files["test_candidate.py"].find("unused") >= 0


def test_reducer_removes_unnecessary_method_actions():
    files = {
        "state.py": "class State:\n    def noise(self):\n        self.extra = 10\n    def prime(self):\n        self.n = 0\n    def fail(self):\n        return 2 / self.n\n",
        "test_state.py": "from state import State\ndef test_actions():\n    obj = State()\n    obj.noise()\n    obj.prime()\n    obj.fail()\n",
    }
    config = ExecutionConfig("local", 15)
    initial = run_project_tests(files, ["state.py"], ["test_state.py"], config)
    report = validate_failures(files, ["state.py"], initial, config, minimize_trials=6)
    reduced = report["reproducers"][0]["source"]
    assert ".noise()" not in reduced
    assert ".prime()" in reduced and ".fail()" in reduced
    locations = {
        (row["file"], row["line"])
        for row in report["reproducers"][0]["preserved_dependency_locations"]
    }
    assert ("state.py", 5) in locations and ("state.py", 7) in locations
    assert ("state.py", 3) not in locations


def test_flaky_and_invalid_replays_are_not_stable_evidence(monkeypatch):
    import pcg.failure_validation as module

    original = TestOutcome(
        "test.py::test_f",
        "failed",
        exception="ZeroDivisionError: division by zero",
        file_frames=[{"file": "a.py", "line": 2}],
    )
    passed = TestOutcome(original.nodeid, "passed")
    runs = iter(
        [ExecutionResult([passed], exit_code=0), ExecutionResult(status="timeout")]
    )
    monkeypatch.setattr(module, "run_project_tests", lambda *a, **k: next(runs))
    report = validate_failures(
        {"test.py": "def test_f(): pass", "a.py": ""},
        ["a.py"],
        ExecutionResult([original], exit_code=1),
        ExecutionConfig(),
        minimize_trials=0,
    )
    assert report["failures"][0]["status"] == "flaky"
    assert report["excluded_from_inference"] == [original.nodeid]
    assert report["reproducers"] == []
    assert report["failures"][0]["beta_failure_rate"]["mean"] == 0.5


def test_invalid_replay_is_inconclusive(monkeypatch):
    import pcg.failure_validation as module

    original = TestOutcome("test.py::test_f", "failed", exception="AssertionError")
    monkeypatch.setattr(
        module,
        "run_project_tests",
        lambda *a, **k: ExecutionResult(status="harness_error"),
    )
    report = validate_failures(
        {},
        [],
        ExecutionResult([original], exit_code=1),
        ExecutionConfig(),
        minimize_trials=0,
    )
    assert report["failures"][0]["status"] == "inconclusive"
    assert report["excluded_from_inference"] == [original.nodeid]


def test_flaky_failure_cannot_create_a_bayesian_test_factor(monkeypatch):
    import pcg.failure_validation as module
    from pcg.pipeline import analyze

    monkeypatch.setattr(
        module,
        "validate_failures",
        lambda *a, **k: {
            "excluded_from_inference": ["test_candidate.py::test_bad"],
            "failures": [{"status": "flaky"}],
        },
    )
    analysis = analyze(
        "def f():\n    return 1\n",
        "from candidate import f\ndef test_bad():\n    assert f() == 2\n",
        failure_replays=2,
    )
    assert analysis.execution.valid
    assert analysis.execution.tests[0].outcome == "failed"
    assert analysis.graph.graph["inference"]["observed_tests"] == 0
    assert (
        analysis.to_dict()["automatic_validation"]["status"] == "no-correctness-oracle"
    )
    assert any(
        e.kind == "unstable_execution" and e.polarity == "neutral"
        for e in analysis.evidence
    )


def test_reduction_does_not_change_expected_value_assignments():
    from pcg.failure_validation import reductions
    import ast

    source = "from candidate import f\ndef test_value():\n    expected = 999\n    inputs = [55, 66]\n    result = f(inputs)\n    assert result == expected\n"
    candidates = reductions(source, "test.py::test_value")
    assert candidates
    for candidate in candidates:
        assignment = next(
            n
            for n in ast.walk(ast.parse(candidate))
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "expected" for t in n.targets)
        )
        assert ast.literal_eval(assignment.value) == 999
