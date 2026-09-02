"""
Turns raw critic results into an answer to the actual question:
"When the LLM critic says it's 90% confident, is it right ~90% of the time?"

Reports:
  - Accuracy, precision, recall, F1 (is the critic even correct?)
  - Brier score (accuracy of the probability, not just the verdict)
  - Expected Calibration Error / ECE (the core calibration number)
  - A reliability diagram: confidence bucket vs actual accuracy in that bucket
  - A suggested weight to use when feeding this critic's confidence into a
    downstream Bayesian evidence combiner, based on how much its stated
    confidence should be trusted.

Usage:
    python analyze_calibration.py --results calibration_results.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_valid(results_path):
    results = json.loads(Path(results_path).read_text())
    valid = [r for r in results if r["critic_is_buggy"] is not None]
    dropped = len(results) - len(valid)
    return valid, dropped


def basic_metrics(valid):
    tp = fp = tn = fn = 0
    for r in valid:
        truth, pred = r["ground_truth_is_buggy"], r["critic_is_buggy"]
        if truth and pred:
            tp += 1
        elif not truth and pred:
            fp += 1
        elif not truth and not pred:
            tn += 1
        else:
            fn += 1

    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n if n else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def brier_score(valid):
    """Brier score on P(is_buggy=True). Lower is better; 0 is perfect,
    0.25 is what you get from always guessing 50/50."""
    errs = []
    for r in valid:
        p_true = r["critic_confidence"] if r["critic_is_buggy"] else 1 - r["critic_confidence"]
        y = 1.0 if r["ground_truth_is_buggy"] else 0.0
        errs.append((p_true - y) ** 2)
    return float(np.mean(errs))


def reliability_bins(valid, n_bins=5):
    """Bucket examples by the critic's *stated confidence in its verdict*
    (not P(buggy)) and check: within each bucket, how often was the
    verdict actually correct? A well-calibrated critic's accuracy in the
    "0.8-0.9 confidence" bucket should be close to 80-90%."""
    bins = [[] for _ in range(n_bins)]
    edges = np.linspace(0.5, 1.0, n_bins + 1)  # confidence is always >= 0.5 by construction

    for r in valid:
        conf = r["critic_confidence"]
        correct = r["critic_is_buggy"] == r["ground_truth_is_buggy"]
        idx = min(int((conf - 0.5) / 0.5 * n_bins), n_bins - 1)
        bins[idx].append((conf, correct))

    rows = []
    ece_terms = []
    total_n = len(valid)
    for i, b in enumerate(bins):
        lo, hi = edges[i], edges[i + 1]
        if not b:
            rows.append({"range": f"{lo:.2f}-{hi:.2f}", "n": 0, "avg_confidence": None, "accuracy": None})
            continue
        confs = [c for c, _ in b]
        corrects = [1 if c else 0 for _, c in b]
        avg_conf = float(np.mean(confs))
        acc = float(np.mean(corrects))
        rows.append({"range": f"{lo:.2f}-{hi:.2f}", "n": len(b), "avg_confidence": avg_conf, "accuracy": acc})
        ece_terms.append(len(b) / total_n * abs(avg_conf - acc))

    ece = float(sum(ece_terms))
    return rows, ece


def suggested_weight(ece, accuracy):
    """Heuristic mapping from calibration quality -> how much weight this
    critic's confidence should get relative to a well-calibrated source
    (weight 1.0) in a downstream Bayesian evidence combiner. This is a
    starting point for tuning, not a substitute for held-out validation
    on your own PCG posterior outputs."""
    if accuracy < 0.55:
        return 0.0  # not even directionally useful, worse than a coin flip
    penalty = min(1.0, ece / 0.3)  # ECE of 0.3+ treated as "fully untrustworthy"
    return round(max(0.05, 1.0 - penalty), 2)


def plot_reliability(rows, out_path):
    xs = [ (float(r["range"].split("-")[0]) + float(r["range"].split("-")[1])) / 2 for r in rows]
    accs = [r["accuracy"] if r["accuracy"] is not None else np.nan for r in rows]
    ns = [r["n"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(xs, accs, marker="o", color="crimson", label="LLM critic (actual)")
    for x, acc, n in zip(xs, accs, ns):
        if not np.isnan(acc):
            ax.annotate(f"n={n}", (x, acc), textcoords="offset points", xytext=(0, 8), fontsize=8)
    ax.set_xlabel("Stated confidence")
    ax.set_ylabel("Actual accuracy in that bucket")
    ax.set_title("Reliability diagram: LLM critic confidence vs. reality")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved reliability diagram to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="calibration_results.json")
    ap.add_argument("--plot-out", default="reliability_diagram.png")
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    valid, dropped = load_valid(args.results)
    if not valid:
        print("No valid results to analyze (all parse/API errors).")
        return

    metrics = basic_metrics(valid)
    brier = brier_score(valid)
    rows, ece = reliability_bins(valid, n_bins=args.bins)
    weight = suggested_weight(ece, metrics["accuracy"])

    print("=" * 60)
    print(f"N examples analyzed : {metrics['n']}  (dropped {dropped} parse/API errors)")
    print(f"Accuracy             : {metrics['accuracy']:.3f}")
    print(f"Precision            : {metrics['precision']:.3f}")
    print(f"Recall               : {metrics['recall']:.3f}")
    print(f"F1                   : {metrics['f1']:.3f}")
    print(f"Confusion (tp/fp/tn/fn): {metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}")
    print(f"Brier score           : {brier:.3f}  (0=perfect, 0.25=coin flip)")
    print(f"Expected Calib. Error : {ece:.3f}  (0=perfectly calibrated)")
    print("-" * 60)
    print("Reliability by confidence bucket:")
    print(f"{'range':>12} {'n':>5} {'avg_conf':>10} {'accuracy':>10}")
    for r in rows:
        ac = f"{r['avg_confidence']:.3f}" if r['avg_confidence'] is not None else "  n/a"
        acc = f"{r['accuracy']:.3f}" if r['accuracy'] is not None else "  n/a"
        print(f"{r['range']:>12} {r['n']:>5} {ac:>10} {acc:>10}")
    print("-" * 60)
    print(f"Suggested evidence weight for this critic: {weight}")
    print("  (1.0 = trust it as much as a calibrated source; 0.0 = ignore its confidence entirely)")
    print("=" * 60)

    plot_reliability(rows, args.plot_out)

    summary = {"metrics": metrics, "brier_score": brier, "ece": ece,
               "reliability_bins": rows, "suggested_weight": weight, "dropped": dropped}
    Path("calibration_summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote calibration_summary.json")


if __name__ == "__main__":
    main()
