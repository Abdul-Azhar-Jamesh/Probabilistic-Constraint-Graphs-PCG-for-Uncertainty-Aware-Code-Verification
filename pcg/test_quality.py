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
) -> dict:
    if not 1 <= max_mutants <= 200:
        raise ValueError("max_mutants must be between 1 and 200")
    execution = execution or ExecutionConfig("docker")
    baseline = run_tests(source, tests, execution)
    if not baseline.valid or baseline.failed_ids or not baseline.passed_ids:
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
            and baseline.passed_ids
            <= {
                test.nodeid
                for test in result.tests
                if test.outcome in {"passed", "failed"}
            }
        )
        status = (
            "killed"
            if valid_run and result.failed_ids
            else "survived"
            if valid_run
            else "inconclusive"
        )
        rows.append(
            {
                "id": mutant.case_id,
                "block": mutant.buggy_block,
                "description": mutant.description,
                "status": status,
            }
        )
    killed = sum(row["status"] == "killed" for row in rows)
    valid = sum(row["status"] != "inconclusive" for row in rows)
    return {
        "generated": len(rows),
        "killed": killed,
        "scorable": valid,
        "mutation_score": killed / valid if valid else None,
        "limitation": "Survivors may be equivalent or expose missing assertions; score is not correctness probability.",
        "mutants": rows,
    }
