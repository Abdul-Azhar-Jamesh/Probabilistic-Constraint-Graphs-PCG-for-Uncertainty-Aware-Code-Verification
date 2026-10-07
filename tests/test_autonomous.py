"""Real workers exercise autonomous checks on previously unseen interfaces."""

import ast

import pytest

from pcg.autotest import branch_inputs, literal_examples
from pcg.blocks import extract_blocks
from pcg.execution import ExecutionConfig, run_tests
from pcg.graph import build_graph
from pcg.testgen import plan_tests


def run(source, **options):
    blocks = extract_blocks(source)
    plan = plan_tests(
        source, blocks, build_graph(blocks, source=source), autonomous=True, **options
    )
    return plan, run_tests(source, plan.source, ExecutionConfig("local", timeout=30))


def test_affine_branch_inputs_follow_earlier_assignment():
    fn = ast.parse(
        "def f(x: int):\n    y = x * 3 + 2\n    if y == 2141:\n        return 0\n"
    ).body[0]
    assert (713,) in branch_inputs(fn, ["x"])


def test_literal_doctest_never_evaluates_expressions():
    fn = ast.parse(
        'def f(x):\n    """>>> f(2)\n    4\n    >>> f(__import__("os").system("echo unsafe"))\n    0\n    """\n    return x\n'
    ).body[0]
    assert literal_examples(fn) == [
        {"function": "f", "args": [2], "kwargs": {}, "expected": 4}
    ]


def test_unknown_typed_code_finds_rare_branch_without_supplied_tests():
    plan, result = run(
        "def transform(x: int) -> int:\n    intermediate = x * 3 + 2\n    if intermediate == 2141:\n        return 'incorrect type'\n    return x\n"
    )
    assert result.valid
    failures = [
        t for t in result.tests if t.oracle == "annotation" and t.outcome == "failed"
    ]
    assert failures
    assert "713" in failures[0].failure_details
    assert any(c.get("branch_examples") for c in plan.cases)


def test_fuzzing_shrinks_collection_failure_beyond_old_boundaries():
    _, result = run(
        "def process(values: list[int]) -> int:\n    if len(values) >= 5:\n        return 'bad'\n    return len(values)\n",
        graph_guided=False,
    )
    failed = [
        t for t in result.tests if t.oracle == "annotation" and t.outcome == "failed"
    ]
    assert result.valid and failed
    assert (
        "Failing test case" in failed[0].failure_details
        or "Falsifying example" in failed[0].failure_details
    )
    assert "[0, 0, 0, 0, 0]" in failed[0].failure_details


def test_existing_doctest_finds_semantic_bug_without_new_oracle():
    _, result = run(
        'def square(x: int) -> int:\n    """>>> square(4)\n    16\n    """\n    return x + x\n'
    )
    assert result.valid
    assert any(t.oracle == "contract" and t.outcome == "failed" for t in result.tests)


def test_rejected_inputs_do_not_create_false_passing_type_evidence():
    _, result = run("def reject(x: int) -> int:\n    raise ValueError('unsupported')\n")
    assert result.valid
    assert all(t.outcome == "skipped" for t in result.tests if t.oracle == "annotation")
    assert any(t.outcome == "failed" and t.oracle == "probe" for t in result.tests)


def test_stateful_sequence_discovers_delayed_failure():
    plan, result = run(
        "class Counter:\n    def __init__(self):\n        self.n = 0\n    def step(self):\n        self.n += 1\n        if self.n >= 4:\n            raise RuntimeError('delayed failure')\n"
    )
    assert result.valid
    assert any(c.get("method") == "stateful-sequence-probe" for c in plan.cases)
    assert any(
        t.oracle == "probe" and t.outcome == "failed" and "steps=" in t.failure_details
        for t in result.tests
    )


@pytest.mark.parametrize(
    "source",
    [
        "def total(values: list[int]) -> int:\n    return sum(values)\n",
        "def stringify(x: int) -> str:\n    return str(x)\n",
        "def fraction(x: float, y: float) -> float:\n    if y == 0:\n        raise ValueError('zero')\n    return x / y\n",
    ],
)
def test_valid_unseen_programs_do_not_fail_automatic_oracles(source):
    _, result = run(source)
    assert result.valid
    assert not any(t.oracle != "probe" and t.outcome == "failed" for t in result.tests)


def test_declared_property_receives_fuzz_inputs_and_shrinking():
    plan, result = run(
        "def reorder(values: list[int]) -> list[int]:\n    if len(values) >= 5:\n        return []\n    return sorted(values)\n",
        contracts=[{"function": "reorder", "property": "permutation"}],
        graph_guided=False,
    )
    assert any(
        c.get("method") == "property-fuzzing" and c.get("property") == "permutation"
        for c in plan.cases
    )
    assert result.valid
    assert any(
        t.oracle_scope == "property"
        and t.outcome == "failed"
        and (
            "Failing test case" in t.failure_details
            or "Falsifying example" in t.failure_details
        )
        for t in result.tests
    )


def test_fuzz_budget_is_shared_across_functions():
    source = (
        "def a(x: int) -> int:\n    return x\ndef b(x: int) -> int:\n    return x\n"
    )
    blocks = extract_blocks(source)
    plan = plan_tests(
        source,
        blocks,
        build_graph(blocks, source=source),
        autonomous=True,
        max_cases=12,
    )
    fuzz = [c for c in plan.cases if c.get("method") == "property-fuzzing"]
    assert {c["function"] for c in fuzz} == {"a", "b"}
    assert len(plan.cases) <= 12
    assert len({c["name"] for c in plan.cases}) == len(plan.cases)
