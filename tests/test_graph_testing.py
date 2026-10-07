"""CFG paths must change generated inputs and actual execution, not just reports."""

import ast

from pcg.graph_testing import plan_graph_targets, add_uncovered_checks
from pcg.slicing import program_dependence
from pcg.pipeline import analyze
from pcg.execution import ExecutionConfig, ExecutionResult, TestOutcome
from pcg.testgen import TestPlan as GeneratedPlan


COUPLED = "def combine(x: int, y: int) -> int:\n    gate = x * 3 + 2\n    unused = str(x)\n    if gate == 2141:\n        if y - x == 278:\n            return 'bad type'\n    return 0\n"


def test_dependency_sliced_cfg_solves_interacting_inputs_and_ignores_unrelated_call():
    targets = plan_graph_targets(COUPLED, program_dependence(COUPLED))
    target = next(t for t in targets.targets if t["args"] == [713, 991])
    assert target["decisions"] == [
        {"line": 4, "outcome": True},
        {"line": 5, "outcome": True},
    ]
    assert 2 in target["dependency_slice"] and 3 not in target["dependency_slice"]
    assert targets.graph.has_edge(4, 5) and targets.graph.has_edge(5, 6)


def test_dependence_graph_is_required_for_assignments_not_just_metadata():
    dependence = program_dependence(COUPLED)
    dependence.graph.remove_edge(2, 4)
    targets = plan_graph_targets(COUPLED, dependence)
    assert not targets.targets
    assert any("Name" in r["reason"] for r in targets.unresolved)


def test_unsatisfiable_branch_is_reported_and_not_fabricated():
    src = "def f(x: int) -> int:\n    if x < 0 and x > 5:\n        return 'bad'\n    return 0\n"
    targets = plan_graph_targets(src, program_dependence(src))
    assert any(
        r["reason"] == "infeasible within input bounds" for r in targets.unresolved
    )
    assert not any(t["decisions"][0]["outcome"] for t in targets.targets)


def test_pure_helper_summary_generates_callers_input():
    src = "def helper(x: int) -> int:\n    return x * 7 + 1\ndef caller(x: int) -> int:\n    value = helper(x)\n    if value == 4992:\n        return 'bad'\n    return x\n"
    targets = plan_graph_targets(src, program_dependence(src))
    assert any(
        t["function"] == "caller" and t["args"] == [713] for t in targets.targets
    )


def test_string_and_length_constraints_are_real_graph_seeds():
    src = "def f(token: str, values: list[int]) -> int:\n    if token.startswith('pcg:') and len(values) == 7:\n        return 'bad'\n    return 0\n"
    targets = plan_graph_targets(src, program_dependence(src))
    seeded = [t for t in targets.targets if t["decisions"][0]["outcome"]]
    assert (
        seeded
        and seeded[0]["args"][0].startswith("pcg:")
        and len(seeded[0]["args"][1]) == 7
    )


def test_unsupported_loop_does_not_claim_symbolic_coverage():
    src = "def f(x: int) -> int:\n    for i in range(x):\n        x += 1\n    if x == 10:\n        return 'bad'\n    return 0\n"
    targets = plan_graph_targets(src, program_dependence(src))
    assert not targets.targets
    assert any("compound flow" in r["reason"] for r in targets.unresolved)


def test_reassignment_is_used_instead_of_stale_definition():
    src = "def f(x: int) -> int:\n    gate = x * 3\n    gate = x + 1\n    if gate == 714:\n        return 'bad'\n    return 0\n"
    targets = plan_graph_targets(src, program_dependence(src))
    target = next(t for t in targets.targets if t["decisions"][0]["outcome"])
    assert target["args"] == [713] and 2 not in target["dependency_slice"]


def test_graph_ablation_changes_detected_failure_on_new_code():
    baseline = analyze(
        COUPLED,
        autonomous_testing=True,
        graph_guided=False,
        execution=ExecutionConfig("local"),
    )
    directed = analyze(
        COUPLED,
        autonomous_testing=True,
        graph_guided=True,
        execution=ExecutionConfig("local"),
    )
    assert baseline.execution.valid and directed.execution.valid
    assert not any(
        t.oracle != "probe" and t.outcome == "failed" for t in baseline.execution.tests
    )
    assert any(
        t.oracle == "annotation" and t.outcome == "failed"
        for t in directed.execution.tests
    )
    assert directed.to_dict()["graph_testing"]["distinct_branch_edges_seen"] >= 3
    assert directed.to_dict()["graph_testing"]["feedback"]["rounds"] == 2
    assert (
        directed.graph.graph["inference"]["observed_tests"]
        > directed.graph.graph["inference"]["observation_families"]
    )
    direct_ids = {
        "test_candidate.py::" + c["name"]
        for c in directed.test_plan["cases"]
        if c.get("method") == "coverage-directed-cfg-replay"
    }
    assert any(
        d["test"] in direct_ids
        and 2 in d["candidate_cause_lines"]
        and 3 not in d["candidate_cause_lines"]
        and 7 not in d["candidate_cause_lines"]
        for d in directed.failure_diagnoses
    )


def test_module_literal_threshold_is_supported():
    src = "LIMIT = 713\ndef f(x: int) -> int:\n    if x == LIMIT:\n        return 'bad'\n    return 0\n"
    targets = plan_graph_targets(src, program_dependence(src))
    assert any(t["args"] == [713] for t in targets.targets)


def test_graph_planner_does_not_execute_candidate_module():
    src = "raise RuntimeError('host must not execute')\ndef f(x: int) -> int:\n    if x == 713:\n        return 'bad'\n    return 0\n"
    targets = plan_graph_targets(src, program_dependence(src))
    assert targets.targets


def test_feedback_adds_real_tests_only_for_uncovered_edges_and_respects_budget():
    targets = plan_graph_targets(COUPLED, program_dependence(COUPLED))
    execution = ExecutionResult(
        [TestOutcome("test_initial", "passed", arcs=[[2, 3], [3, 4], [4, 7]])],
        exit_code=0,
    )
    plan = GeneratedPlan()
    report = add_uncovered_checks(plan, targets, execution, COUPLED, max_cases=2)
    assert report["added_checks"] == 2 and len(plan.cases) == 2
    assert all(c["method"] == "coverage-directed-cfg-replay" for c in plan.cases)
    ast.parse(plan.source)
