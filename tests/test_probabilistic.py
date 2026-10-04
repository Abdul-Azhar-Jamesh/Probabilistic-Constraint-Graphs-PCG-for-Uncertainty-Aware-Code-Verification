"""Analytical and property-based checks of the deployed graphical model."""

import math
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import pytest
from hypothesis import given, settings, strategies as st

from pcg.blocks import extract_blocks
from pcg.evidence import Evidence, collect_llm_confidence
from pcg.graph import build_graph
from pcg.inference import infer
from pcg.probabilistic import (
    TestFactor,
    condition_defects,
    dependency_impacts,
    rank_next_tests,
)
from pcg.sensors import SensorModel
from pcg.validation import program_group, validate_dataset


def test_exact_posterior_matches_bayes_rule():
    factor = TestFactor("failure", ("a",), True, sensitivity=0.9, leak=0.01)
    posterior = condition_defects({"a": 0.1}, [factor], method="exact")
    p_failure_defect = 1 - (1 - 0.01) * (1 - 0.9)
    expected = 0.1 * p_failure_defect / (0.1 * p_failure_defect + 0.9 * 0.01)
    assert posterior.marginals["a"] == pytest.approx(expected)


def test_shared_test_explaining_away():
    factor = TestFactor("failure", ("a", "b"), True)
    unknown = condition_defects({"a": 0.1, "b": 0.1}, [factor])
    known_cause = condition_defects({"a": 0.99, "b": 0.1}, [factor])
    assert known_cause.marginals["b"] < unknown.marginals["b"]


def test_duplicates_do_not_manufacture_confidence():
    factor = TestFactor("one", ("a",), False)
    single = condition_defects({"a": 0.3}, [factor])
    duplicate = condition_defects(
        {"a": 0.3}, [factor, TestFactor("copy", ("a",), False)]
    )
    assert duplicate.marginals == single.marginals
    blocks = extract_blocks("def f(x):\n    return x\n")
    item = Evidence(blocks[0].bid, "static", "warning", "negative", 0.8)
    first = infer(blocks, build_graph(blocks), [item])
    copies = infer(blocks, build_graph(blocks), [item, item, item])
    assert first[blocks[0].bid].posterior == copies[blocks[0].bid].posterior


def test_gibbs_matches_exact_and_reports_mixing():
    priors = {"a": 0.2, "b": 0.3}
    factors = [TestFactor("shared", ("a", "b"), True)]
    exact = condition_defects(priors, factors, method="exact")
    sampled = condition_defects(
        priors, factors, method="gibbs", draws=1000, burn_in=200, seed=17
    )
    for node in priors:
        assert sampled.marginals[node] == pytest.approx(exact.marginals[node], abs=0.04)
    assert sampled.diagnostics["max_rhat"] < 1.1
    trust, se = sampled.trust({"a": 1.0, "b": 0.8})
    assert trust == pytest.approx(exact.trust({"a": 1.0, "b": 0.8})[0], abs=0.04)
    assert 0 < se < 0.1


def test_joint_trust_does_not_multiply_correlated_marginals():
    posterior = condition_defects(
        {"a": 0.2, "b": 0.2}, [TestFactor("shared", ("a", "b"), True)]
    )
    naive = math.prod(1 - p for p in posterior.marginals.values())
    joint = posterior.trust({"a": 1.0, "b": 1.0})[0]
    assert joint != pytest.approx(naive, abs=0.01)


def test_cycles_and_diamonds_do_not_repeat_ancestor_defects():
    graph = nx.DiGraph()
    graph.add_edge("a", "b", strength=0.8)
    graph.add_edge("a", "c", strength=0.8)
    graph.add_edge("b", "d", strength=0.8)
    graph.add_edge("c", "d", strength=0.8)
    graph.add_edge("d", "a", strength=0.8)
    impacts = dependency_impacts(graph, "d")
    assert impacts["d"] == 1
    assert impacts["a"] == pytest.approx(0.64)
    assert len(impacts) == 4


def test_recursive_defects_remain_repair_candidates():
    from pcg.inference import BlockPosterior, find_root_repairs

    graph = nx.DiGraph([("a", "b"), ("b", "a"), ("b", "c")])
    post = {
        node: BlockPosterior(node, 0.8, 0.1, 0.1, 0.4, culpability=0.9)
        for node in graph
    }
    assert set(find_root_repairs(post, graph)) == {"a", "b"}


def test_class_method_resolution_does_not_cross_classes():
    source = """
class A:
    def helper(self):
        return 1
    def caller(self):
        return self.helper()
class B:
    def helper(self):
        return 2
"""
    blocks = extract_blocks(source)
    ids = {b.qualname: b.bid for b in blocks}
    graph = build_graph(blocks)
    assert graph.has_edge(ids["A.helper"], ids["A.caller"])
    assert not graph.has_edge(ids["B.helper"], ids["A.caller"])


def test_callable_parameter_does_not_create_false_call_edge():
    blocks = extract_blocks(
        "def helper():\n    pass\ndef f(helper):\n    return helper()\n"
    )
    assert not build_graph(blocks).number_of_edges()


def test_request_settings_do_not_leak_between_threads():
    blocks = extract_blocks("def f(x):\n    return x\n")
    evidence = [Evidence(blocks[0].bid, "static", "warning", "negative", 0.9)]

    def calculate(reliability):
        return infer(
            blocks, build_graph(blocks), evidence, reliabilities={"static": reliability}
        )[blocks[0].bid].posterior

    expected = [calculate(0), calculate(4)]
    with ThreadPoolExecutor(2) as pool:
        actual = list(pool.map(calculate, [0, 4]))
    assert actual == expected and actual[0] > actual[1]


def test_missing_llm_confidence_is_missing_evidence():
    assert collect_llm_confidence(extract_blocks("def f():\n    pass\n")) == []


def test_information_gain_prefers_uncertain_function():
    posterior = condition_defects({"uncertain": 0.5, "known": 0.001}, [])
    ranked = rank_next_tests(
        posterior,
        [{"name": "a", "blocks": ["uncertain"]}, {"name": "b", "blocks": ["known"]}],
    )
    assert ranked[0]["name"] == "a"
    assert 0 <= ranked[0]["information_gain_bits"] <= 1


def test_program_group_includes_reference_and_all_operators():
    assert (
        program_group("ref:stats")
        == program_group("stats:comparison:1")
        == program_group("stats:arithmetic:2")
    )


def test_dataset_rejects_repository_and_content_leakage():
    rows = [
        {
            "name": split,
            "repository": split,
            "source": split,
            "tests": "",
            "buggy": [],
            "unreliable": [],
            "split": split,
        }
        for split in ("train", "validation", "test")
    ]
    validate_dataset(rows)
    rows[1]["repository"] = "train"
    with pytest.raises(ValueError, match="multiple splits"):
        validate_dataset(rows)


def test_invalid_or_legacy_models_rejected(tmp_path):
    path = tmp_path / "model.json"
    path.write_text('{"source_reliability_fitted": {"exec": 0}}')
    with pytest.raises(ValueError, match="compatible"):
        SensorModel.load(path)
    path.write_text("[]")
    with pytest.raises(ValueError, match="compatible"):
        SensorModel.load(path)
    from pcg.sensors import DEFAULT_SENSORS

    invalid = {
        source: dict(probabilities) for source, probabilities in DEFAULT_SENSORS.items()
    }
    invalid["exec"]["flag_given_defect"] = float("nan")
    with pytest.raises(ValueError, match="probabilities"):
        SensorModel(likelihoods=invalid)


@given(
    st.floats(min_value=0.001, max_value=0.999, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=40)
def test_passing_test_reduces_defect_belief(prior):
    posterior = condition_defects({"a": prior}, [TestFactor("pass", ("a",), False)])
    assert 0 <= posterior.marginals["a"] < prior
    assert posterior.trust({"a": 1.0})[0] == pytest.approx(1 - posterior.marginals["a"])


@given(
    st.floats(min_value=0.001, max_value=0.999, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=40)
def test_failing_test_increases_defect_belief(prior):
    assert (
        prior
        < condition_defects({"a": prior}, [TestFactor("fail", ("a",), True)]).marginals[
            "a"
        ]
        <= 1
    )
