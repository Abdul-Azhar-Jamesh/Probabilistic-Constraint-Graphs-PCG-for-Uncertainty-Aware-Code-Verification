from hypothesis import given, strategies as st

from pcg.corpus import REFERENCE_PROGRAMS
from pcg.execution import ExecutionResult, TestOutcome
from pcg.test_quality import mutation_audit


def test_mutation_audit_distinguishes_survivors_and_harness_errors(monkeypatch):
    outcomes = iter(
        [
            ExecutionResult([TestOutcome("test", "passed")], exit_code=0),
            ExecutionResult([TestOutcome("test", "failed")], exit_code=1),
            ExecutionResult([TestOutcome("test", "passed")], exit_code=0),
            ExecutionResult(status="harness_error", errors=["missing dependency"]),
        ]
    )
    monkeypatch.setattr("pcg.test_quality.run_tests", lambda *a, **kw: next(outcomes))
    result = mutation_audit(
        "def f(x):\n    if x > 0:\n        return x + 1\n    return x - 1\n",
        "test",
        max_mutants=3,
    )
    assert result["generated"] == 3
    assert result["scorable"] == 2
    assert result["mutation_score"] == 0.5
    assert [m["status"] for m in result["mutants"]] == [
        "killed",
        "survived",
        "inconclusive",
    ]


# Trusted contracts, independent of any candidate being analysed. These show
# how a real user can supply stronger tests for the bundled algorithms.
@given(st.lists(st.integers(min_value=-1000, max_value=1000), max_size=30))
def test_reference_sort_preserves_elements_and_order(xs):
    source = next(
        src for name, (src, _) in REFERENCE_PROGRAMS.items() if "def merge_sort(" in src
    )
    namespace = {}
    exec(source, namespace)
    original = list(xs)
    assert namespace["merge_sort"](xs) == sorted(original)
    assert xs == original


@given(
    st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=50)
)
def test_reference_normalization_is_idempotent(text):
    source = next(
        src for _, (src, _) in REFERENCE_PROGRAMS.items() if "def normalize(" in src
    )
    namespace = {}
    exec(source, namespace)
    normalize = namespace["normalize"]
    assert normalize(normalize(text)) == normalize(text)
