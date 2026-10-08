"""Separate observed facts from model estimates in every report."""

from __future__ import annotations

from .evidence import Evidence
from .execution import ExecutionResult


def summarize_findings(
    execution: ExecutionResult | None,
    evidence: list[Evidence],
    validation: dict,
    *,
    static: dict | None = None,
    calibrated: bool = False,
) -> dict:
    excluded = set(validation.get("excluded_from_inference", []))
    replay = {
        r["test"]: r["status"] for r in validation.get("failures", []) if "test" in r
    }
    categories: dict[str, list[dict]] = {
        name: []
        for name in (
            "syntax_errors",
            "oracle_failures",
            "probe_exceptions",
            "unstable_failures",
            "static_warnings",
            "incomplete_checks",
        )
    }
    seen: set[tuple] = set()
    for item in evidence:
        if item.kind == "static_unavailable":
            unavailable_key = ("incomplete_checks", item.kind, item.detail, None)
            if unavailable_key not in seen:
                seen.add(unavailable_key)
                categories["incomplete_checks"].append(
                    {"sensor": "static", "status": "unavailable", "detail": item.detail}
                )
        if item.polarity != "negative" or item.source not in {"compile", "static"}:
            continue
        category = "syntax_errors" if item.source == "compile" else "static_warnings"
        file = item.meta.get(
            "file", item.bid.split("::", 1)[0] if "::" in item.bid else "candidate.py"
        )
        line = item.meta.get("line")
        key = (category, item.kind, item.detail, file, line)
        if key not in seen:
            seen.add(key)
            categories[category].append(
                {"kind": item.kind, "detail": item.detail, "file": file, "line": line}
            )
    if static:
        for item in static.get("findings", []):
            key = (
                "static_warnings",
                item["code"],
                item["message"],
                item["filename"],
                item["location"]["row"],
            )
            if key in seen:
                continue
            seen.add(key)
            categories["static_warnings"].append(
                {
                    "file": item["filename"],
                    "line": item["location"]["row"],
                    "kind": item["code"],
                    "detail": item["message"],
                }
            )
        if static.get("status") != "complete":
            categories["incomplete_checks"].append(
                {"sensor": "static", "status": static.get("status")}
            )
    passed = 0
    for test in execution.tests if execution else []:
        if (
            test.outcome == "passed"
            and test.oracle != "probe"
            and execution
            and execution.valid
        ):
            passed += 1
        if test.outcome != "failed":
            continue
        category = (
            "incomplete_checks"
            if not execution or not execution.valid
            else "unstable_failures"
            if test.nodeid in excluded
            else "probe_exceptions"
            if test.oracle == "probe"
            else "oracle_failures"
        )
        categories[category].append(
            {
                "test": test.nodeid,
                "oracle": test.oracle,
                "scope": test.oracle_scope,
                "exception": test.exception,
                "repeatability": replay.get(test.nodeid, "not-replayed"),
            }
        )
    if not execution or not execution.valid:
        categories["incomplete_checks"].append(
            {
                "sensor": "execution",
                "status": execution.status if execution else "not-run",
                "errors": execution.errors if execution else [],
            }
        )
    return {
        "counts": {key: len(value) for key, value in categories.items()},
        "categories": categories,
        "passed_oracle_checks": passed,
        "probability_status": "independently-evaluated-model"
        if calibrated
        else "uncalibrated-model-estimates",
        "interpretations": {
            "oracle_failures": "Observed violations of the named test or contract; root cause remains a hypothesis. Not-replayed failures lack a stability check.",
            "probe_exceptions": "Exceptions on generated inputs without a correctness oracle; accepted input behavior is unknown.",
            "static_warnings": "Potential issues requiring investigation, not confirmed defects.",
            "unstable_failures": "Flaky or inconclusive replays, excluded from correctness inference.",
            "incomplete_checks": "Environment, collection, disabled execution or deadline limitations; not evidence of correctness or proof of an infinite loop.",
            "passed_oracle_checks": "Bounded successful checks, not proof of intended behavior or absence of defects.",
            "probabilities": "Review estimates conditioned on model assumptions and evidence; confidence bands do not establish correctness.",
        },
    }
