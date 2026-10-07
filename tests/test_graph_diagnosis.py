"""Behavioral regression cases: earlier causes, branch joins and generated oracles."""

import json

import pytest
import networkx as nx

from pcg.blocks import extract_blocks, line_to_block
from pcg.execution import (
    ExecutionConfig,
    ExecutionResult,
    TestOutcome,
    preserves_tests,
    run_tests,
)
from pcg.evidence import _ast_smells
from pcg.graph import build_graph
from pcg.inference import local_posteriors, prior_correctness
from pcg.pipeline import analyze
from pcg.slicing import failure_slices, program_dependence
from pcg.testgen import plan_tests
from pcg.repair import repair, propose_patches
from pcg.test_quality import mutation_audit


def test_reassignment_kills_old_definition_but_retains_true_origin():
    src = "def f(x):\n    old = x + 1\n    old = 0\n    unused = x * 100\n    denominator = old\n    return 10 / denominator\n"
    pdg = program_dependence(src)
    assert set(pdg.backward({6})) == {3, 5, 6}
    assert 2 not in pdg.backward({6})
    assert 4 not in pdg.backward({6})


def test_unexecuted_branch_cannot_enter_coverage_filtered_slice():
    src = "def f(flag):\n    x = 1\n    if flag:\n        x = 0\n    return 10 / x\n"
    pdg = program_dependence(src)
    assert {2, 4} <= set(pdg.backward({5}))
    sliced = pdg.backward({5}, {1, 2, 3, 5})
    assert 2 in sliced and 4 not in sliced


def test_mutation_and_augmented_assignment_have_data_dependencies():
    src = "def f():\n    xs = []\n    xs.append(3)\n    xs[0] = 0\n    x = xs[0]\n    x += 1\n    return x\n"
    assert {2, 3, 4, 5, 6, 7} <= set(program_dependence(src).backward({7}))


def test_delayed_runtime_failure_points_to_assignment_not_only_traceback():
    src = "def f(x: int) -> float:\n    divisor = x - x\n    unrelated = 123\n    numerator = 10\n    return numerator / divisor\n"
    blocks = extract_blocks(src)
    result = run_tests(
        src,
        "from candidate import f\ndef test_valid_input():\n    assert f(3) == 1\n",
        ExecutionConfig(timeout=15),
    )
    assert result.valid and result.failed_ids
    diagnosis = failure_slices(program_dependence(src), result, blocks)[0]
    assert diagnosis["symptom_lines"] == [5]
    assert 2 in diagnosis["candidate_cause_lines"]
    assert 3 not in diagnosis["candidate_cause_lines"]
    owner = line_to_block(blocks)
    graph = build_graph(blocks, source=src)
    assert graph.has_edge(owner[2], owner[5])


def test_interprocedural_wrong_return_reaches_the_calling_expression():
    src = "def denominator(x):\n    return x - x\ndef divide(x):\n    d = denominator(x)\n    return 10 / d\n"
    assert {2, 4, 5} <= set(program_dependence(src).backward({5}))


def test_local_function_alias_and_instance_method_keep_their_origins():
    src = "def helper(x):\n    return x - x\ndef caller(x):\n    fn = helper\n    result = fn(x)\n    return result\n"
    assert 2 in program_dependence(src).backward({6})
    src = "class A:\n    def value(self):\n        return 0\ndef f():\n    obj = A()\n    result = obj.value()\n    return 1 / result\n"
    assert 3 in program_dependence(src).backward({7})


def test_callback_parameter_does_not_resolve_to_same_named_global():
    src = "def helper(x):\n    return x - x\ndef caller(helper, x):\n    return helper(x)\n"
    pdg = program_dependence(src)
    assert 2 not in pdg.backward({4})
    assert {"line": 4, "call": "helper"} in pdg.graph.graph["unresolved_calls"]


def test_loop_carried_value_has_an_earlier_definition():
    src = "def f(xs):\n    old = 0\n    total = 0\n    for x in xs:\n        total += old\n        old = x\n    return total\n"
    assert 6 in program_dependence(src).backward({7})


def test_import_time_origin_is_kept_separate_from_test_execution():
    src = "OFFSET = 0\ndef divide(x):\n    return x / OFFSET\n"
    blocks = extract_blocks(src)
    result = run_tests(
        src,
        "from candidate import divide\ndef test_value():\n    assert divide(2) == 2\n",
        ExecutionConfig(timeout=15),
    )
    assert 1 not in result.tests[0].lines
    assert 1 in result.tests[0].initialization_lines
    diagnosis = failure_slices(program_dependence(src), result, blocks)[0]
    assert 1 in diagnosis["initialization_origin_lines"]


def test_side_effect_assertion_can_slice_without_candidate_traceback():
    src = "def change(xs):\n    value = 0\n    unrelated = 99\n    xs[0] = value\n"
    blocks = extract_blocks(src)
    result = run_tests(
        src,
        "from candidate import change\ndef test_value():\n    xs = [1]\n    change(xs)\n    assert xs == [2]\n",
        ExecutionConfig(timeout=15),
    )
    assert not result.tests[0].frames
    diagnosis = failure_slices(program_dependence(src), result, blocks)[0]
    assert {2, 4} <= set(diagnosis["candidate_cause_lines"])
    assert 3 not in diagnosis["candidate_cause_lines"]


def test_module_lines_do_not_absorb_function_bodies():
    src = "x = 1\ndef f():\n    return x\ny = 2\n"
    blocks = extract_blocks(src)
    owner = line_to_block(blocks)
    assert next(b for b in blocks if b.bid == owner[3]).qualname == "f"
    module = next(b for b in blocks if b.kind == "module")
    assert module.owned_lines == {1, 4}


def test_splitting_does_not_create_extra_prior_defect_budget():
    src = "def f(x):\n    a = x + 1\n    b = a + 1\n    c = b + 1\n    return c\n"
    blocks = extract_blocks(src)
    parent = next(b for b in blocks if b.kind == "function")
    post = local_posteriors(blocks, [])
    product = 1.0
    for b in blocks:
        product *= post[b.bid].prior
    assert abs(product - prior_correctness(parent)) < 0.001
    assert nx.is_directed_acyclic_graph(build_graph(blocks, source=src))


@pytest.mark.parametrize(
    "src",
    [
        "def f(x):\n    if x:\n        return 1 / x\n",
        "def f(x):\n    if x == 0:\n        raise ValueError('zero')\n    return 1 / x\n",
    ],
)
def test_guarded_division_does_not_generate_a_false_alarm(src):
    assert not any(
        e.kind == "unguarded_division" for e in _ast_smells(src, extract_blocks(src))
    )


def test_reassignment_invalidates_the_nonzero_guard():
    src = "def f(x):\n    if x == 0:\n        return 0\n    x = 0\n    return 1 / x\n"
    assert any(
        e.kind == "unguarded_division" for e in _ast_smells(src, extract_blocks(src))
    )


def test_planner_does_not_import_candidate_or_make_unknown_output_assertions(tmp_path):
    marker = tmp_path / "should-not-exist"
    src = f"open({str(marker)!r}, 'w').write('bad')\ndef f(x: int):\n    return x + 1\n"
    blocks = extract_blocks(src)
    plan = plan_tests(src, blocks, build_graph(blocks, source=src))
    assert not marker.exists()
    assert plan.cases and all(c["oracle"] == "probe" for c in plan.cases)
    assert "assert _pcg_fn" not in plan.source


def test_boundary_probes_use_constants_and_budget_across_functions():
    src = "def f(x: int):\n    return x > 100\ndef g(s: str):\n    return len(s)\n"
    blocks = extract_blocks(src)
    plan = plan_tests(src, blocks, build_graph(blocks), max_cases=20)
    assert {"f", "g"} == {c["function"] for c in plan.cases}
    assert {"[99]", "[100]", "[101]"} <= {
        c["args"] for c in plan.cases if c["function"] == "f"
    }
    assert len(plan.cases) <= 20


def test_fractional_branch_boundaries_do_not_generate_invalid_typed_inputs():
    src = "def f(x: int) -> int:\n    if x < 0.5:\n        return x\n    return x\n"
    blocks = extract_blocks(src)
    plan = plan_tests(src, blocks, build_graph(blocks), annotation_contracts=True)
    import ast

    assert all(type(ast.literal_eval(case["args"])[0]) is int for case in plan.cases)


def test_probe_exception_is_not_fabricated_defect_evidence():
    src = "def reciprocal(x: int) -> float:\n    return 1 / x\n"
    analysis = analyze(src, auto_tests=True, execution=ExecutionConfig(timeout=15))
    assert analysis.execution.valid
    assert analysis.execution.failed_ids
    assert all(not e.meta.get("test_observation") for e in analysis.evidence)
    assert analysis.graph.graph["inference"]["test_factors"] == 0
    assert analysis.failure_diagnoses
    assert any(
        e.kind == "probe_exception" and e.polarity == "neutral"
        for e in analysis.evidence
    )


def test_importing_module_does_not_credit_an_uncalled_function():
    src = "def unused(x):\n    return x - 100\ndef used():\n    return 1\n"
    analysis = analyze(
        src,
        contracts=[{"function": "used", "expected": 1}],
        max_generated_cases=1,
        execution=ExecutionConfig(timeout=15),
    )
    unused = next(b for b in analysis.blocks if b.qualname == "unused")
    assert analysis.posteriors[unused.bid].execution_status == "untested"
    assert not any(
        e.bid == unused.bid and e.meta.get("test_observation")
        for e in analysis.evidence
    )


def test_declared_return_contract_detects_bug_without_handwritten_pytest():
    src = "def f(x: int) -> int:\n    return str(x)\n"
    analysis = analyze(
        src,
        auto_tests=True,
        annotation_contracts=True,
        execution=ExecutionConfig(timeout=15),
    )
    assert any(
        t.oracle == "annotation" and t.outcome == "failed"
        for t in analysis.execution.tests
    )
    assert analysis.graph.graph["inference"]["test_factors"] > 0
    assert analysis.posteriors[analysis.blocks[0].bid].culpability > 0.5
    json.dumps(analysis.to_dict(), allow_nan=False)


def test_passing_type_checks_do_not_exonerate_a_wrong_expected_value():
    src = "def f(x: int) -> int:\n    return x + 2\n"
    analysis = analyze(
        src,
        contracts=[{"function": "f", "args": [1], "expected": 2}],
        annotation_contracts=True,
        execution=ExecutionConfig(timeout=15),
    )
    factors = analysis.graph.graph["defect_posterior"].factors
    assert any(f.failed and f.sensitivity == 0.95 for f in factors)
    assert any(not f.failed and f.sensitivity < 0.3 for f in factors)
    assert analysis.posteriors[analysis.blocks[0].bid].culpability > 0.5


def test_contract_and_metamorphic_checks_detect_logic_defect():
    src = "def scale(x):\n    return -x\n"
    contracts = [
        {"function": "scale", "args": [2], "expected": 2},
        {
            "function": "scale",
            "args": [1],
            "other_args": [2],
            "relation": "nondecreasing",
        },
    ]
    blocks = extract_blocks(src)
    plan = plan_tests(
        src, blocks, build_graph(blocks), contracts=contracts, max_cases=2
    )
    result = run_tests(src, plan.source, ExecutionConfig(timeout=15))
    assert result.valid and len(result.failed_ids) == 2
    assert all(t.oracle == "contract" for t in result.tests)


def test_annotation_exception_does_not_invent_input_contract():
    src = "def reciprocal(x: int) -> float:\n    return 1 / x\n"
    blocks = extract_blocks(src)
    plan = plan_tests(src, blocks, build_graph(blocks), annotation_contracts=True)
    result = run_tests(src, plan.source, ExecutionConfig(timeout=15))
    assert any(
        t.oracle == "annotation" and t.outcome == "skipped" for t in result.tests
    )
    assert not any(
        t.oracle == "annotation" and t.outcome == "failed" for t in result.tests
    )


def test_unknown_interface_is_reported_and_can_take_explicit_cases():
    src = "def process(customer):\n    return customer['total']\n"
    blocks = extract_blocks(src)
    plan = plan_tests(src, blocks, build_graph(blocks))
    assert not plan.cases and plan.skipped
    contract = [{"function": "process", "args": [{"total": 5}], "expected": 5}]
    plan = plan_tests(src, blocks, build_graph(blocks), contracts=contract)
    assert plan.cases[0]["oracle"] == "contract"


def test_reusable_property_tests_many_inputs_without_expected_values():
    src = "def normalize(xs: list[int]) -> list[int]:\n    return xs[:-1]\n"
    analysis = analyze(
        src,
        contracts=[{"function": "normalize", "property": "permutation"}],
        execution=ExecutionConfig(timeout=15),
    )
    assert any(
        t.oracle == "contract" and t.outcome == "failed"
        for t in analysis.execution.tests
    )
    assert (
        len(
            [
                c
                for c in analysis.test_plan["cases"]
                if c.get("property") == "permutation"
            ]
        )
        > 1
    )


def test_disabled_execution_still_provides_information_gain_recommendations():
    src = "def f(x: int) -> int:\n    return x + 1\n"
    analysis = analyze(
        src, annotation_contracts=True, execution=ExecutionConfig("disabled")
    )
    rows = analysis.test_plan["recommendations"]
    assert rows and all(
        r["information_gain_bits"] >= 0 and r["oracle"] == "annotation" for r in rows
    )
    assert all(r["coverage_basis"].startswith("static") for r in rows)
    assert analysis.graph.graph["inference"]["test_factors"] == 0


def test_completed_checks_are_not_recommended_for_redundant_reruns():
    src = "def f(x: int) -> int:\n    return x + 1\n"
    analysis = analyze(
        src, annotation_contracts=True, execution=ExecutionConfig(timeout=15)
    )
    assert analysis.test_plan["recommendations"] == []


def test_probe_improvement_cannot_validate_a_repair():
    before = ExecutionResult(
        [TestOutcome("probe", "failed", oracle="probe")], exit_code=1
    )
    after = ExecutionResult(
        [TestOutcome("probe", "passed", oracle="probe")], exit_code=0
    )
    assert not preserves_tests(before, after)


def test_mutation_audit_ignores_oracle_free_exceptions(monkeypatch):
    result = ExecutionResult(
        [
            TestOutcome("check", "passed", oracle="contract"),
            TestOutcome("probe", "failed", oracle="probe"),
        ],
        exit_code=1,
    )
    monkeypatch.setattr("pcg.test_quality.run_tests", lambda *args: result)
    audit = mutation_audit("def f(x):\n    return x + 1\n", "test", max_mutants=1)
    assert audit["generated"] == 1 and audit["mutants"][0]["status"] == "survived"
    assert audit["oracle_checks"] == 1


def test_oracle_free_suite_cannot_claim_a_mutation_score(monkeypatch):
    result = ExecutionResult(
        [TestOutcome("probe", "passed", oracle="probe")], exit_code=0
    )
    monkeypatch.setattr("pcg.test_quality.run_tests", lambda *args: result)
    with pytest.raises(ValueError, match="passing baseline"):
        mutation_audit("def f(x):\n    return x + 1\n", "test", max_mutants=1)


def test_segment_repairs_are_not_discarded_before_validation(monkeypatch):
    src = "def f(x):\n    a = x + 1\n    b = a * 2\n    c = b - 1\n    return c\n"
    blocks = extract_blocks(src)
    segment = next(b for b in blocks if b.kind == "segment" and "a =" in b.source)
    from pcg.inference import BlockPosterior

    post = {
        segment.bid: BlockPosterior(segment.bid, 0.9, 0.1, 0.1, 0.5, culpability=0.9)
    }
    before = ExecutionResult([TestOutcome("case", "failed")], exit_code=1)
    after = ExecutionResult([TestOutcome("case", "passed")], exit_code=0)
    monkeypatch.setattr(
        "pcg.repair.run_tests",
        lambda candidate, tests, config: before if candidate == src else after,
    )
    patches = propose_patches(src, segment, max_patches=1)
    assert patches and patches[0].lineno == segment.lineno
    results = repair(src, "tests", post, blocks, max_targets=1, max_patches=1)
    assert results and results[0].accepted and results[0].target_bid == segment.bid


@pytest.mark.parametrize(
    "case",
    [
        {"function": "f", "args": [], "expected": float("nan")},
        {"function": "f", "args": [], "raises": "eval('bad')"},
        {"function": "missing", "expected": 1},
    ],
)
def test_invalid_contracts_fail_before_execution(case):
    src = "def f():\n    return 1\n"
    blocks = extract_blocks(src)
    with pytest.raises(ValueError):
        plan_tests(src, blocks, build_graph(blocks), contracts=[case])
