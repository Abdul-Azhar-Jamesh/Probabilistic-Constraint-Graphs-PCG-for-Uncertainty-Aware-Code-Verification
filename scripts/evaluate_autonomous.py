"""Run curated bug/fix controls without supplying their behavior assertions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcg.benchmark_metrics import summarize_benchmark  # noqa: E402
from pcg.evaluation_cases import curated_dataset  # noqa: E402
from pcg.execution import ExecutionConfig  # noqa: E402
from pcg.pipeline import analyze  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out/autonomous-evaluation"))
    parser.add_argument("--execution", choices=["local", "docker"], default="docker")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in curated_dataset():
        started = time.monotonic()
        analysis = analyze(
            case["source"],
            execution=ExecutionConfig(args.execution, 25),
            autonomous_testing=True,
            max_generated_cases=8,
            failure_replays=2,
            minimize_trials=0,
        )
        report = analysis.to_dict()
        findings = report["findings"]["counts"]
        status = report["automatic_validation"]["status"]
        if analysis.execution and analysis.execution.status == "no_tests":
            status = "no-executable-checks"
        rows.append(
            {
                "case": case["name"],
                "repository": case["repository"],
                "version": "buggy" if case["buggy"] else "fixed",
                "mode": "autonomous",
                "status": status,
                "seconds": time.monotonic() - started,
                "oracle_failures": findings["oracle_failures"],
                "probe_failures": findings["probe_exceptions"],
                "static_warnings": findings["static_warnings"],
            }
        )
        (args.out / (case["name"].replace(":", "-") + ".json")).write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(case["name"], status, findings, flush=True)
    summary = {
        "scope": "Synthetic controls; assertions deliberately withheld; no production-accuracy claim.",
        "rows": rows,
        "metrics": summarize_benchmark(rows),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
