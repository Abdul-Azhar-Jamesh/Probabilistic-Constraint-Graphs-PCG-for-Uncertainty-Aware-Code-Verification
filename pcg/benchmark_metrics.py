"""Case-level evaluation keeps unavailable cases and probe crashes explicit."""

from __future__ import annotations

import numpy as np


def summarize_benchmark(rows: list[dict]) -> dict:
    result: dict = {}
    for mode in ("autonomous", "upstream-regression"):
        selected = [r for r in rows if r.get("mode") == mode]
        complete = [
            r
            for r in selected
            if r["status"] not in {"incomplete", "no-executable-checks", "unavailable"}
        ]
        buggy = [r for r in complete if r["version"] == "buggy"]
        fixed = [r for r in complete if r["version"] == "fixed"]
        detected = sum(r["oracle_failures"] > 0 for r in buggy)
        alarms = sum(r["oracle_failures"] > 0 for r in fixed)
        seconds = [r["seconds"] for r in selected]
        result[mode] = {
            "attempted": len(selected),
            "complete": len(complete),
            "incomplete_or_unsupported": len(selected) - len(complete),
            "buggy_cases": len(buggy),
            "fixed_controls": len(fixed),
            "detected_buggy_cases": detected,
            "false_alarms_on_fixed": alarms,
            "detection_rate": detected / len(buggy) if buggy else None,
            "false_alarm_rate": alarms / len(fixed) if fixed else None,
            "probe_exception_cases": sum(
                r.get("probe_failures", 0) > 0 for r in complete
            ),
            "median_seconds": float(np.median(seconds)) if seconds else None,
            "p95_seconds": float(np.quantile(seconds, 0.95)) if seconds else None,
        }
    result["unavailable_cases"] = [
        r["case"] for r in rows if r["status"] == "unavailable"
    ]
    result["interpretation"] = (
        "Detection counts oracle-backed failures only; probe exceptions and static warnings do not count as confirmed bugs. "
        "Rates are conditional on completed cases, with unsupported cases shown separately. Fixed controls establish the targeted regression behavior, not whole-project correctness."
    )
    return result
