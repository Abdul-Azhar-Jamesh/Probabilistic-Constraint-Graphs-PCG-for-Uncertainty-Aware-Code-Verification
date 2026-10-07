"""Bounded mutation audit of test strength, without assuming survivor equivalence."""

from __future__ import annotations

from .execution import ExecutionConfig, run_tests
from .mutate import generate_mutants


def mutation_audit(
    source: str,
    tests: str,
    *,
    max_mutants: int = 20,
    execution: ExecutionConfig | None = None,
    strengthen_generated: bool = False,
) -> dict:
    if not 1 <= max_mutants <= 200:
        raise ValueError("max_mutants must be between 1 and 200")
    execution = execution or ExecutionConfig("docker")
    baseline = run_tests(source, tests, execution)
    baseline_passed = {
        t.nodeid
        for t in baseline.tests
        if t.oracle != "probe" and t.outcome == "passed"
    }
    baseline_failed = {
        t.nodeid
        for t in baseline.tests
        if t.oracle != "probe" and t.outcome == "failed"
    }
    if not baseline.valid or baseline_failed or not baseline_passed:
        raise ValueError("mutation audit requires a complete passing baseline suite")
    mutants = generate_mutants(
        "candidate", source, tests, max_per_operator=max_mutants
    )[:max_mutants]
    rows = []
    for mutant in mutants:
        result = run_tests(mutant.source, tests, execution)
        valid_run = (
            result.valid
            and {test.nodeid for test in baseline.tests}
            == {test.nodeid for test in result.tests}
            and {t.nodeid: (t.oracle, t.oracle_scope) for t in baseline.tests}
            == {t.nodeid: (t.oracle, t.oracle_scope) for t in result.tests}
            and baseline_passed
            <= {
                test.nodeid
                for test in result.tests
                if test.outcome in {"passed", "failed"} and test.oracle != "probe"
            }
        )
        status = (
            "killed"
            if valid_run
            and any(t.outcome == "failed" and t.oracle != "probe" for t in result.tests)
            else "survived"
            if valid_run
            else "inconclusive"
        )
        rows.append(
            {
                "id": mutant.case_id,
                "block": mutant.buggy_block,
                "description": mutant.description,
                "line": mutant.lineno,
                "status": status,
            }
        )
    # Surviving mutations trigger a broader search against the same declared
    # oracles. Candidate outputs never become assertions. Validate the stronger
    # baseline before interpreting additional kills.
    feedback: dict = {
        "attempted": False,
        "reason": "no eligible surviving generated checks",
    }
    if (
        strengthen_generated
        and any(r["status"] == "survived" for r in rows)
        and "max_examples=80" in tests
    ):
        stronger = tests.replace("max_examples=80", "max_examples=160")
        checked = run_tests(source, stronger, execution)
        oracle_passed = {
            t.nodeid
            for t in checked.tests
            if t.oracle != "probe" and t.outcome == "passed"
        }
        safe_baseline = (
            checked.valid
            and baseline_passed <= oracle_passed
            and not any(
                t.oracle != "probe" and t.outcome == "failed" for t in checked.tests
            )
        )
        feedback = {
            "attempted": True,
            "baseline_valid": safe_baseline,
            "examples_per_check": 160,
            "additional_kills": 0,
        }
        if safe_baseline:
            for row, mutant in zip(rows, mutants):
                if row["status"] != "survived":
                    continue
                result = run_tests(mutant.source, stronger, execution)
                expanded_valid = result.valid and oracle_passed <= {
                    t.nodeid
                    for t in result.tests
                    if t.oracle != "probe" and t.outcome in {"passed", "failed"}
                }
                if expanded_valid and any(
                    t.nodeid in oracle_passed and t.outcome == "failed"
                    for t in result.tests
                ):
                    row["status"] = "killed"
                    row["feedback"] = "additional generated inputs killed mutation"
                    feedback["additional_kills"] += 1
                elif not expanded_valid:
                    row["feedback"] = (
                        "expanded run inconclusive; original status retained"
                    )
    killed = sum(row["status"] == "killed" for row in rows)
    valid = sum(row["status"] != "inconclusive" for row in rows)
    return {
        "generated": len(rows),
        "oracle_checks": len(baseline_passed),
        "killed": killed,
        "scorable": valid,
        "mutation_score": killed / valid if valid else None,
        "limitation": "Survivors may be equivalent or expose missing assertions; score is not correctness probability.",
        "mutants": rows,
        "feedback": feedback,
        "weak_spots": [
            {
                "block": r["block"],
                "line": r["line"],
                "mutation": r["description"],
                "interpretation": "undetected change; equivalence or missing semantic oracle remains possible",
            }
            for r in rows
            if r["status"] == "survived"
        ],
    }
