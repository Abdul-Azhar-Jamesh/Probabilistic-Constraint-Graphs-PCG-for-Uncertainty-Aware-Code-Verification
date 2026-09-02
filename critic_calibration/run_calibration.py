"""
Runs the LLM critic against every example in the calibration dataset and
saves (ground_truth, predicted, confidence) triples for analysis.

Usage:
    python run_calibration.py --backend mock --dataset calibration_dataset.json
    python run_calibration.py --backend anthropic --dataset calibration_dataset.json
        (requires: export ANTHROPIC_API_KEY=sk-...)
"""
import argparse
import json
import sys
import time
from pathlib import Path

from llm_critic import get_critic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="calibration_dataset.json")
    ap.add_argument("--backend", choices=["anthropic", "mock"], default="mock")
    ap.add_argument("--out", default="calibration_results.json")
    ap.add_argument("--limit", type=int, default=None, help="cap number of examples (for quick/cheap test runs)")
    args = ap.parse_args()

    examples = json.loads(Path(args.dataset).read_text())
    if args.limit:
        examples = examples[: args.limit]

    critic = get_critic(args.backend)
    results = []

    for i, ex in enumerate(examples, 1):
        verdict = critic.judge(ex["code"], file=ex["file"])
        result = {**ex, **{f"critic_{k}": v for k, v in verdict.items()}}
        results.append(result)
        status = "OK" if verdict["is_buggy"] is not None else "PARSE_ERROR"
        print(f"[{i}/{len(examples)}] {ex['example_id']:35s} "
              f"truth={ex['ground_truth_is_buggy']!s:5s} "
              f"pred={verdict['is_buggy']!s:5s} conf={verdict['confidence']} [{status}]",
              file=sys.stderr)
        if args.backend == "anthropic":
            time.sleep(0.3)  # light rate-limit courtesy

    Path(args.out).write_text(json.dumps(results, indent=2))
    n_errors = sum(1 for r in results if r["critic_is_buggy"] is None)
    print(f"\nWrote {len(results)} results to {args.out} ({n_errors} parse/API errors)", file=sys.stderr)


if __name__ == "__main__":
    main()
