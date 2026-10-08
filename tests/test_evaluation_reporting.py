"""Guard evaluation honesty, grouping and measurement semantics."""

import pytest

from pcg.benchmark_metrics import summarize_benchmark
from pcg.evaluation_cases import curated_dataset
from pcg.execution import ExecutionResult, TestOutcome
from pcg.findings import summarize_findings
from pcg.validation import (
    bootstrap_metrics,
    metrics,
    reliability_bins,
    validate_dataset,
)


def test_report_never_promotes_probe_or_incomplete_failure():
    tests = [
        TestOutcome("probe", "failed", oracle="probe"),
        TestOutcome("contract", "failed", oracle="annotation"),
        TestOutcome("flaky", "failed"),
    ]
    result = ExecutionResult(tests, exit_code=1)
    data = summarize_findings(result, [], {"excluded_from_inference": ["flaky"]})
    assert data["counts"]["oracle_failures"] == 1
    assert data["counts"]["probe_exceptions"] == 1
    assert data["counts"]["unstable_failures"] == 1
    assert data["categories"]["oracle_failures"][0]["repeatability"] == "not-replayed"
    result.status = "timeout"
    data = summarize_findings(result, [], {})
    assert data["counts"]["oracle_failures"] == 0
    assert data["counts"]["incomplete_checks"] == 4


def test_curated_families_disjoint_and_labels_valid():
    from pcg.blocks import extract_blocks

    rows = curated_dataset()
    validate_dataset(rows)
    assert len(rows) == 24
    for row in rows:
        assert set(row["unreliable"]) <= {
            b.qualname for b in extract_blocks(row["source"])
        }
        compile(row["tests"], "tests", "exec")


def prediction(case, repo, name, correct, defect):
    return {
        "case": case,
        "repository": repo,
        "block": name,
        "correct": correct,
        "reliable": correct,
        "defect": defect,
        "trust": 1 - defect,
        "risk": defect,
    }


def test_localization_is_per_program_and_uses_intrinsic_defects():
    rows = [
        prediction("a", "a", "clean", 1, 0.9),
        prediction("a", "a", "bug", 0, 0.8),
        prediction("b", "b", "bug", 0, 0.1),
    ]
    score = metrics(rows)
    assert score["localization"]["top_1"] == 0.5
    assert score["localization"]["mean_reciprocal_rank"] == 0.75
    assert score["defect"]["false_positive_rate"] == 1


def test_calibration_bins_include_endpoints_and_empty_bins():
    bins = reliability_bins([0.0, 0.95, 1.0], [0, 0, 1])
    assert sum(b["count"] for b in bins) == 3
    assert bins[-1]["observed_frequency"] == 0.5
    assert bins[4]["mean_probability"] is None


def test_cluster_bootstrap_reproducible_and_single_group_abstains():
    rows = [prediction("a", "a", "f", 1, 0.1), prediction("b", "b", "g", 0, 0.9)]
    first = bootstrap_metrics(rows, 0.5, repeats=40)
    assert first == bootstrap_metrics(rows, 0.5, repeats=40)
    assert first["percentile_intervals"]["defect.recall"] == pytest.approx([0, 1])
    assert bootstrap_metrics(rows[:1], 0.5)["status"] == "insufficient-repositories"


def test_benchmark_denominators_keep_unsupported_separate():
    rows = [
        {
            "case": "bug",
            "mode": "autonomous",
            "status": "no-correctness-oracle",
            "version": "buggy",
            "oracle_failures": 0,
            "probe_failures": 1,
            "seconds": 2,
        },
        {
            "case": "fixed",
            "mode": "autonomous",
            "status": "incomplete",
            "version": "fixed",
            "oracle_failures": 1,
            "seconds": 3,
        },
    ]
    score = summarize_benchmark(rows)["autonomous"]
    assert score["detection_rate"] == 0
    assert score["probe_exception_cases"] == 1
    assert score["fixed_controls"] == 0
    assert score["false_alarm_rate"] is None
    assert score["incomplete_or_unsupported"] == 1


def test_extracted_tqdm_fixture_preserves_python3_unicode_alias():
    from scripts.real_bug_benchmark import prepare

    source = (
        "def _text_width(s):\n    return len(_unicode(s))\n"
        "def disp_len(data):\n    return _text_width(RE_ANSI.sub('', data))\n"
        "def disp_trim(data, length):\n    return data if disp_len(data) <= length else data[:length]\n"
    )
    _, candidate, _, _, _ = prepare("tqdm-2", source, "")
    scope = {}
    exec(compile(candidate, "fixture", "exec"), scope)
    text = "\x1b[31mabc\x1b[0m"
    assert scope["disp_trim"](text, 3) == text


def test_static_warning_count_deduplicates_attribution_not_locations():
    from pcg.evidence import Evidence

    evidence = [
        Evidence(
            "a.py::f",
            "static",
            "F821",
            "negative",
            0.6,
            "undefined name",
            {"file": "a.py", "line": 2},
        )
    ]
    static = {
        "status": "complete",
        "findings": [
            {
                "filename": "a.py",
                "location": {"row": line},
                "code": "F821",
                "message": "undefined name",
            }
            for line in (2, 5)
        ],
    }
    result = summarize_findings(
        ExecutionResult(exit_code=0), evidence, {}, static=static
    )
    assert result["counts"]["static_warnings"] == 2
