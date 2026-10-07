"""Compare graph-directed testing with the same autonomous fuzzing without it.

Curated held-out interface examples, not a production accuracy benchmark.
No candidate-specific pytest files or external behavior contracts are supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from pcg.execution import ExecutionConfig
from pcg.pipeline import analyze


CASES = {
    "coupled_inputs": "def combine(x: int, y: int) -> int:\n    gate = x * 3 + 2\n    unused = str(x)\n    if gate == 2141:\n        if y - x == 278:\n            return 'bad type'\n    return 0\n",
    "string_and_numeric_gate": "def quote(quantity: int, unit_price: int, coupon: str) -> int:\n    if quantity <= 0 or unit_price < 0:\n        raise ValueError('invalid order')\n    total = quantity * unit_price\n    if coupon == 'PCG-LIVE-7' and quantity == 13:\n        if unit_price == 941:\n            return 'broken discount'\n    return total\n",
    "helper_dependency": "def transform(x: int) -> int:\n    return x * 7 + 1\ndef process(x: int, y: int) -> int:\n    gate = transform(x)\n    if gate == 4992 and y == 991:\n        return 'wrong type'\n    return x + y\n",
    "string_collection_interaction": "def count(token: str, values: list[int]) -> int:\n    if token.startswith('pcg:live:') and len(values) == 7:\n        return 'wrong type'\n    return len(values)\n",
    "several_hidden_failures": "def dispatch(x: int, y: int) -> int:\n    gate = x * 3 + 2\n    if gate == 2141 and y == 991:\n        return 'bad one'\n    if gate == 2975 and y == 713:\n        return 1 // 0\n    if gate == 3704 and y == 9047:\n        return 'bad two'\n    if gate == 16505 and y == 8123:\n        return 1 // 0\n    return 0\n",
    "correct_guarded_quote": "def quote(quantity: int, unit_price: int, coupon: str) -> int:\n    if quantity <= 0 or unit_price < 0:\n        raise ValueError('invalid order')\n    total = quantity * unit_price\n    if coupon == 'PCG-LIVE-7' and quantity == 13:\n        if unit_price == 941:\n            return total - 100\n    return total\n",
    "infeasible_branch": "def f(x: int, y: int) -> int:\n    if x < 0 and x > 5 and y == 713:\n        return 'unreachable'\n    return x + y\n",
}


def run_case(source: str, guided: bool, backend: str) -> dict:
    start = perf_counter()
    analysis = analyze(
        source,
        autonomous_testing=True,
        graph_guided=guided,
        max_generated_cases=40,
        seed=7,
        execution=ExecutionConfig(backend, 30),
    )
    report = analysis.to_dict()
    tests = (
        analysis.execution.tests
        if analysis.execution and analysis.execution.valid
        else []
    )
    return {
        "seconds": round(perf_counter() - start, 3),
        "valid_execution": bool(analysis.execution and analysis.execution.valid),
        "oracle_failures": sum(
            t.oracle != "probe" and t.outcome == "failed" for t in tests
        ),
        "probe_failures": sum(
            t.oracle == "probe" and t.outcome == "failed" for t in tests
        ),
        "covered_lines": sorted({ln for t in tests for ln in t.lines}),
        "covered_arcs": sorted({tuple(a) for t in tests for a in t.arcs}),
        "report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", choices=["local", "docker"], default="local")
    parser.add_argument("--output", default="out/graph-testing-evaluation.json")
    args = parser.parse_args()
    rows = []
    for name, source in CASES.items():
        baseline = run_case(source, False, args.execution)
        graph = run_case(source, True, args.execution)
        rows.append(
            {"case": name, "source": source, "baseline": baseline, "graph": graph}
        )
        print(
            f"{name}: oracle failures {baseline['oracle_failures']} -> {graph['oracle_failures']}; executed lines {len(baseline['covered_lines'])} -> {len(graph['covered_lines'])}; feedback {graph['report'].get('graph_testing', {}).get('feedback', {})}",
            flush=True,
        )
    result = {
        "comparison": "Same source, seed=7, max_cases=40, 80 generated examples per fuzz check. Graph mode adds solved explicit examples and can spend six reserved cases in a second worker round; invocation counts and elapsed time are not identical.",
        "limitation": "Curated synthetic defect examples demonstrate mechanisms, not calibrated real-world accuracy. Probe failures do not establish a bug.",
        "cases": rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return (
        0
        if all(
            r[mode]["valid_execution"] for r in rows for mode in ("baseline", "graph")
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
