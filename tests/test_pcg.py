"""Tests for the PCG framework itself."""

import pytest

from pcg.blocks import extract_blocks, line_to_block
from pcg.evidence import (
    _blame_from_traceback,
    _parse_pytest_counts,
    collect_compile,
    ochiai_localization,
)
from pcg.graph import build_graph, structural_importance
from pcg.inference import (
    expected_calibration_error,
    infer,
    prior_correctness,
    repair_targets,
    sample_posteriors,
)
from pcg.pipeline import analyze

SIMPLE = '''
def helper(x):
    return x * 2


def caller(y):
    return helper(y) + 1
'''


class TestBlocks:
    def test_extracts_functions(self):
        blocks = extract_blocks(SIMPLE)
        names = {b.qualname for b in blocks}
        assert {"helper", "caller"} <= names

    def test_methods_get_own_block(self):
        src = "class A:\n    def m(self):\n        return 1\n"
        blocks = extract_blocks(src)
        assert any(b.qualname == "A.m" and b.kind == "method" for b in blocks)

    def test_call_and_read_detection(self):
        blocks = {b.qualname: b for b in extract_blocks(SIMPLE)}
        assert "helper" in blocks["caller"].calls

    def test_params_are_not_reads(self):
        blocks = {b.qualname: b for b in extract_blocks(SIMPLE)}
        assert "x" not in blocks["helper"].reads

    def test_complexity_counts_branches(self):
        src = "def f(x):\n    if x:\n        return 1\n    return 0\n"
        b = extract_blocks(src)[0]
        assert b.cyclomatic >= 2

    def test_line_ownership_is_total(self):
        blocks = extract_blocks(SIMPLE)
        owner = line_to_block(blocks)
        for b in blocks:
            for ln in range(b.lineno, b.end_lineno + 1):
                assert ln in owner

    def test_method_lines_beat_class_lines(self):
        src = "class A:\n    X = 1\n    def m(self):\n        return 2\n"
        blocks = extract_blocks(src)
        owner = line_to_block(blocks)
        method = next(b for b in blocks if b.kind == "method")
        assert owner[method.lineno] == method.bid

    def test_syntax_error_propagates(self):
        with pytest.raises(SyntaxError):
            extract_blocks("def f(:\n    pass\n")


BIG_FUNCTION = '''
def big(xs):
    total = 0
    for x in xs:
        total += x

    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1

    mode = None
    best = 0
    for k, v in counts.items():
        if v > best:
            best = v
            mode = k

    mean_v = total / len(xs)
    var = sum((x - mean_v) ** 2 for x in xs) / len(xs)

    spread = max(xs) - min(xs)
    normed = [(x - mean_v) / (spread or 1) for x in xs]

    buckets = {}
    for x in xs:
        b = int((x - min(xs)) / (spread or 1) * 4)
        buckets[b] = buckets.get(b, 0) + 1

    outliers = [x for x in xs if abs(x - mean_v) > 2 * var ** 0.5]

    ranks = {x: sorted(xs).index(x) for x in set(xs)}

    summary = {
        "total": total,
        "mode": mode,
        "variance": var,
        "normed": normed,
        "buckets": buckets,
        "outliers": outliers,
        "ranks": ranks,
    }
    return summary
'''


class TestSegments:
    def test_large_function_is_split(self):
        blocks = extract_blocks(BIG_FUNCTION)
        segs = [b for b in blocks if b.kind == "segment"]
        assert len(segs) >= 2
        assert all(b.qualname.startswith("big#seg") for b in segs)

    def test_small_function_not_split(self):
        blocks = extract_blocks(SIMPLE)
        assert not any(b.kind == "segment" for b in blocks)

    def test_segments_own_their_lines(self):
        blocks = extract_blocks(BIG_FUNCTION)
        owner = line_to_block(blocks)
        segs = [b for b in blocks if b.kind == "segment"]
        # Every segment line maps to that segment, not the parent function.
        for s in segs:
            assert owner[s.lineno] == s.bid

    def test_segment_containment_edge(self):
        blocks = extract_blocks(BIG_FUNCTION)
        g = build_graph(blocks)
        fn = next(b for b in blocks if b.name == "big" and b.kind == "function")
        segs = [b for b in blocks if b.kind == "segment"]
        # Function depends on each of its segments.
        for s in segs:
            assert g.has_edge(s.bid, fn.bid)

    def test_blame_lands_on_segment(self):
        blocks = extract_blocks(BIG_FUNCTION)
        tb = "xs = [3, 1]\nbig.py:12: ValueError\n"
        blame = _blame_from_traceback(tb, blocks, module_name="big")
        segs = {b.bid for b in blocks if b.kind == "segment"}
        assert set(blame) & segs, f"blame missed segments: {blame}"


class TestGraph:
    def test_edge_direction_is_dependency_to_dependent(self):
        blocks = extract_blocks(SIMPLE)
        g = build_graph(blocks)
        ids = {b.qualname: b.bid for b in blocks}
        # helper supports caller, so the edge runs helper -> caller.
        assert g.has_edge(ids["helper"], ids["caller"])
        assert not g.has_edge(ids["caller"], ids["helper"])

    def test_no_self_edges(self):
        src = "def f(n):\n    return f(n - 1) if n else 0\n"
        g = build_graph(extract_blocks(src))
        assert not any(u == v for u, v in g.edges)

    def test_importance_is_bounded(self):
        blocks = extract_blocks(SIMPLE)
        imp = structural_importance(build_graph(blocks), blocks)
        assert all(0.0 <= v <= 1.0 for v in imp.values())

    def test_handles_mutual_recursion(self):
        src = (
            "def a(n):\n    return b(n - 1)\n\n\n"
            "def b(n):\n    return a(n - 1) if n else 0\n"
        )
        blocks = extract_blocks(src)
        g = build_graph(blocks)
        # Cycle must not break inference.
        post = infer(blocks, g, collect_compile(src, blocks))
        assert all(0.0 < bp.posterior < 1.0 for bp in post.values())


class TestInference:
    def test_prior_decreases_with_complexity(self):
        simple = extract_blocks("def f(x):\n    return x\n")[0]
        complex_src = (
            "def g(a, b, c, d):\n"
            "    for i in range(a):\n"
            "        if i > b:\n"
            "            while c:\n"
            "                if d:\n"
            "                    c -= 1\n"
            "    return c\n"
        )
        cplx = extract_blocks(complex_src)[0]
        assert prior_correctness(simple) > prior_correctness(cplx)

    def test_priors_in_range(self):
        for src in (SIMPLE, "def f():\n    pass\n"):
            for b in extract_blocks(src):
                assert 0.0 < prior_correctness(b) < 1.0

    def test_posteriors_are_probabilities(self):
        a = analyze(SIMPLE)
        assert all(0.0 <= bp.posterior <= 1.0 for bp in a.posteriors.values())

    def test_syntax_error_tanks_confidence(self):
        blocks = extract_blocks(SIMPLE)
        g = build_graph(blocks)
        bad = collect_compile("def f(:\n    pass\n", blocks)
        post = infer(blocks, g, bad)
        assert all(bp.posterior < 0.5 for bp in post.values())

    def test_propagation_pulls_down_dependents(self):
        """A caller of broken code must lose confidence."""
        from pcg.evidence import Evidence

        blocks = extract_blocks(SIMPLE)
        g = build_graph(blocks)
        ids = {b.qualname: b.bid for b in blocks}
        ev = [
            Evidence(ids["helper"], "exec", "test_fail", "negative", 0.95, "broken")
        ]
        post = infer(blocks, g, ev)
        assert post[ids["caller"]].posterior < post[ids["caller"]].local

    def test_culpability_separates_own_from_inherited(self):
        from pcg.evidence import Evidence

        blocks = extract_blocks(SIMPLE)
        g = build_graph(blocks)
        ids = {b.qualname: b.bid for b in blocks}
        ev = [
            Evidence(ids["helper"], "exec", "test_fail", "negative", 0.95, "broken")
        ]
        post = infer(blocks, g, ev, structural_importance(g, blocks))
        assert post[ids["helper"]].culpability > post[ids["caller"]].culpability
        assert post[ids["caller"]].inherited > post[ids["helper"]].inherited

    def test_repair_targets_respect_threshold(self):
        from pcg.evidence import Evidence
        from pcg.inference import find_root_repairs

        blocks = extract_blocks(SIMPLE)
        g = build_graph(blocks)
        ids = {b.qualname: b.bid for b in blocks}
        ev = [
            Evidence(ids["helper"], "exec", "test_fail", "negative", 0.95, "broken")
        ]
        post = infer(blocks, g, ev)
        repairs = repair_targets(post, 0.5)
        for bid in repairs:
            assert post[bid].culpability >= 0.5

        roots = find_root_repairs(post, g, 0.5)
        assert ids["helper"] in roots
        assert ids["caller"] not in roots



class TestEvidenceParsing:
    def test_parses_quiet_summary(self):
        assert _parse_pytest_counts("6 failed, 3 passed in 0.25s") == (3, 6)

    def test_parses_banner_summary(self):
        out = "=== 2 failed, 5 passed in 1.2s ==="
        assert _parse_pytest_counts(out) == (5, 2)

    def test_parses_all_passing(self):
        assert _parse_pytest_counts("9 passed in 0.1s") == (9, 0)

    def test_no_counts(self):
        assert _parse_pytest_counts("collected 0 items") == (0, 0)

    def test_blame_maps_line_to_block(self):
        blocks = extract_blocks(SIMPLE)
        ids = {b.qualname: b.bid for b in blocks}
        helper = next(b for b in blocks if b.qualname == "helper")
        out = f"candidate.py:{helper.lineno}: ZeroDivisionError"
        blame = _blame_from_traceback(out, blocks)
        assert blame.get(ids["helper"], 0) > 0


class TestCalibration:
    def test_perfect_confidence_scores_zero(self):
        assert expected_calibration_error([1.0, 1.0, 0.0], [1, 1, 0]) == 0.0

    def test_confidently_wrong_scores_high(self):
        ece = expected_calibration_error([0.95, 0.95, 0.95], [0, 0, 0])
        assert ece > 0.9

    def test_zero_reliability_values_are_not_rejected(self, tmp_path, monkeypatch):
        from pcg import inference

        model = {
            "source_reliability_fitted": {
                "compile": 0.0,
                "static": 0.5521,
                "exec": 0.9378,
                "llm": 0.0,
                "critic": 0.25,
            },
            "prior_coefficients_fitted": {
                "log_cyclomatic": 0.0,
                "log_loc": 0.0,
                "depth": 0.0,
                "n_params": 0.0,
                "intercept": 0.0,
            },
        }
        path = tmp_path / "fitted_weights.json"
        path.write_text(__import__("json").dumps(model), encoding="utf-8")
        monkeypatch.setattr(inference, "FITTED_WEIGHTS_PATH", str(path))
        loaded = inference._load_fitted_weights()
        assert loaded is not None
        assert loaded["source_reliability_fitted"]["critic"] == 0.25

    def test_training_labels_keep_downstream_separate_from_bug_label(self):
        from pcg.build_training_set import _extract_block_features

        features = _extract_block_features(
            case_id="case:downstream",
            source=SIMPLE,
            tests="",
            buggy_qualnames=set(),
            downstream_qualnames={"caller"},
            critic_backend="mock",
        )
        caller = next(f for f in features if f.block_qualname == "caller")
        assert caller.label == 1
        assert caller.is_downstream == 1

    def test_training_pipeline_generates_critic_evidence(self):
        from pcg.build_training_set import _extract_block_features

        features = _extract_block_features(
            case_id="case:critic",
            source=SIMPLE,
            tests="",
            buggy_qualnames=set(),
            downstream_qualnames=set(),
            critic_backend="mock",
        )
        assert any(f.critic_neg_count or f.critic_pos_count for f in features)

    def test_interaction_features_are_defined_from_strengths(self):
        from pcg.build_training_set import _extract_block_features

        features = _extract_block_features(
            case_id="case:interactions",
            source=SIMPLE,
            tests="",
            buggy_qualnames=set(),
            downstream_qualnames=set(),
            critic_backend="mock",
        )
        assert any(hasattr(f, "static_exec_interaction") for f in features)
        assert any(hasattr(f, "static_critic_interaction") for f in features)
        assert any(hasattr(f, "exec_critic_interaction") for f in features)
        assert any(hasattr(f, "llm_critic_interaction") for f in features)

    def test_threshold_selection_loads_saved_validation_threshold(self, tmp_path, monkeypatch):
        from pcg import inference

        payload = {"selected_threshold": 0.63, "metric": "f1"}
        path = tmp_path / "selected_threshold.json"
        path.write_text(__import__("json").dumps(payload), encoding="utf-8")
        monkeypatch.setattr(inference, "SELECTED_THRESHOLD_PATH", str(path))
        assert inference.load_selected_threshold() == 0.63

    def test_abstention_band_uses_validation_margin(self):
        from pcg.inference import classify_abstention

        assert classify_abstention(0.20, threshold=0.63, calibration_ece=0.08) == "LIKELY_DEFECTIVE"
        assert classify_abstention(0.70, threshold=0.63, calibration_ece=0.08) == "UNCERTAIN"
        assert classify_abstention(0.90, threshold=0.63, calibration_ece=0.08) == "LIKELY_CORRECT"

    def test_root_defect_metrics_compute_from_ground_truth(self):
        from pcg.build_training_set import root_defect_metrics

        metrics = root_defect_metrics(
            truth_roots={"A", "B"},
            predicted_roots={"B", "C"},
        )
        assert metrics["precision"] == pytest.approx(0.5)
        assert metrics["recall"] == pytest.approx(0.5)
        assert metrics["f1"] == pytest.approx(0.5)


class TestEndToEnd:
    def test_detects_planted_bug(self):
        src = (
            "def half(xs):\n"
            "    return sum(xs) / len(xs)\n\n\n"
            "def use(xs):\n"
            "    return half(xs) + 1\n"
        )
        tests = (
            "from candidate import half\n\n"
            "def test_half():\n"
            "    assert half([1, 2, 3]) == 99\n"
        )
        a = analyze(src, tests)
        ids = {b.qualname: b.bid for b in a.blocks}
        assert a.posteriors[ids["half"]].posterior < 0.6

    def test_clean_code_stays_confident(self):
        src = "def add(a, b):\n    return a + b\n"
        tests = (
            "from candidate import add\n\n"
            "def test_add():\n    assert add(1, 2) == 3\n"
        )
        a = analyze(src, tests)
        ids = {b.qualname: b.bid for b in a.blocks}
        assert a.posteriors[ids["add"]].posterior > 0.75

    def test_empty_source_rejected(self):
        with pytest.raises(ValueError):
            analyze("")


class TestProbabilisticUpgrades:
    def test_ochiai_ranks_concentrated_failure_highest(self):
        # Block A executed only by the failing test; block B by failing AND
        # two passing tests. Ochiai must rank A above B.
        spectra = {
            "test_f": {"A", "B"},
            "test_p1": {"B"},
            "test_p2": {"B"},
        }
        scores = ochiai_localization(spectra, {"test_f"})
        assert scores["A"] > scores["B"]
        assert abs(scores["A"] - 1.0) < 1e-6  # sqrt(1*1)=1, nf/ne=1

    def test_ochiai_empty_when_no_failures(self):
        assert ochiai_localization({"t1": {"A"}}, set()) == {}

    def test_bidirectional_exoneration_lifts_leaf(self):
        # Leaf utility L is called by caller C. C has strong positive
        # evidence (passing tests) but L has none. Downward vouching should
        # leave L's posterior at or above its local belief, unlike pure
        # noisy-AND which can only drag it down.
        src = (
            "def leaf(x):\n"
            "    return x + 1\n\n\n"
            "def caller(y):\n"
            "    return leaf(y) * 2\n"
        )
        blocks = extract_blocks(src)
        g = build_graph(blocks)
        from pcg.evidence import Evidence

        ev = [
            Evidence("B02", "exec", "test_pass", "positive", 0.9, "tests passed"),
        ]
        post = infer(blocks, g, ev)
        leaf = post["B01"]
        assert leaf.posterior >= leaf.local - 0.05

    def test_mc_uncertainty_present_and_bounded(self):
        blocks = extract_blocks(SIMPLE)
        g = build_graph(blocks)
        post = infer(blocks, g, [])
        for bp in post.values():
            assert 0.0 <= bp.uncertainty < 0.5

    def test_mc_uncertainty_wider_with_conflicting_evidence(self):
        from pcg.evidence import Evidence

        blocks = extract_blocks(SIMPLE)
        calm = [Evidence("B01", "exec", "test_pass", "positive", 0.9, "ok")]
        noisy = [
            Evidence("B01", "exec", "test_pass", "positive", 0.9, "ok"),
            Evidence("B01", "static", "smell", "negative", 0.9, "bad"),
        ]
        u_calm = sample_posteriors(calm, blocks)["B01"][1]
        u_noisy = sample_posteriors(noisy, blocks)["B01"][1]
        assert u_noisy >= u_calm


class TestCritic:
    def test_mock_critic_judges_code(self):
        from pcg.critic import MockCritic

        critic = MockCritic(seed=42)
        verdict = critic.judge("def f(x):\n    return x + 1\n")
        assert isinstance(verdict, dict)
        assert "is_buggy" in verdict
        assert "confidence" in verdict
        assert 0.0 <= verdict["confidence"] <= 1.0

    def test_collect_llm_critic_generates_evidence(self):
        from pcg.evidence import collect_llm_critic

        blocks = extract_blocks(SIMPLE)
        ev = collect_llm_critic(SIMPLE, blocks, critic_backend="mock")
        assert len(ev) > 0
        assert all(e.source == "critic" for e in ev)
        assert all(e.kind == "critic_verdict" for e in ev)

    def test_pipeline_with_critic_backend(self):
        a = analyze(SIMPLE, critic_backend="mock")
        critic_ev = [e for e in a.evidence if e.source == "critic"]
        assert len(critic_ev) > 0
        assert len(a.posteriors) == len(a.blocks)

