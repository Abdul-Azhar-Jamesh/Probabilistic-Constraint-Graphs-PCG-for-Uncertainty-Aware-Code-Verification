import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path  # noqa: E402
from pcg.pipeline import analyze  # noqa: E402
from pcg.execution import ExecutionConfig  # noqa: E402

if __name__ == "__main__":
    root = Path(PROJECT_ROOT)
    analysis = analyze(
        (root / "demo/candidate.py").read_text(encoding="utf-8"),
        (root / "demo/test_candidate.py").read_text(encoding="utf-8"),
        execution=ExecutionConfig("local"),
    )
    print("Blocks:", len(analysis.blocks))
    for bid, bp in analysis.posteriors.items():
        print(bid, bp.posterior, bp.culpability, bp.inherited)
