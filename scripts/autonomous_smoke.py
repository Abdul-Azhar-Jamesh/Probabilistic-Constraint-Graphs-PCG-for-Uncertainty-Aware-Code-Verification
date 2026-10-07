"""Reproducible unseen-input smoke evaluation without handwritten candidate tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcg.execution import ExecutionConfig
from pcg.pipeline import analyze


CASES = {
    "correct_sum": "def aggregate(values: list[int]) -> int:\n    return sum(values)\n",
    "correct_guard": "def fraction(x: float, y: float) -> float:\n    if y == 0:\n        raise ValueError('zero denominator')\n    return x / y\n",
    "rare_earlier_assignment": "def transform(x: int) -> int:\n    intermediate = x * 3 + 2\n    unrelated = 123\n    if intermediate == 2141:\n        return 'bad type'\n    return x\n",
    "longer_collection": "def process(values: list[int]) -> int:\n    if len(values) >= 5:\n        return 'bad type'\n    return len(values)\n",
    "documented_semantic_bug": 'def square(x: int) -> int:\n    """>>> square(4)\n    16\n    """\n    return x + x\n',
    "delayed_state_failure": "class Counter:\n    def __init__(self):\n        self.n = 0\n    def step(self):\n        self.n += 1\n        if self.n >= 4:\n            raise RuntimeError('delayed failure')\n",
    "unspecified_semantics": "def calculation(x):\n    return x + x\n",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="out/autonomous-smoke.json")
    parser.add_argument("--execution", choices=["local", "docker"], default="local")
    args = parser.parse_args()
    rows = []
    for name, source in CASES.items():
        result = analyze(
            source,
            autonomous_testing=True,
            execution=ExecutionConfig(args.execution, 30),
        )
        report = result.to_dict()
        rows.append({"case": name, "source": source, "report": report})
        print(name, report["automatic_validation"]["status"], flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "interpretation": "Curated unseen-interface smoke checks, not an accuracy benchmark; no handwritten candidate pytest or external contracts supplied.",
                "cases": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return (
        0
        if all(
            row["report"]["execution"]
            and row["report"]["execution"]["status"] == "complete"
            for row in rows
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
