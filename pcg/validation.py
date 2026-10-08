"""Repository-disjoint evaluation of the exact runtime model and its ablations.

Dataset rows contain name, repository, source, tests, buggy, unreliable and
split (train/validation/test). No held-out labels influence fitted parameters
or selected thresholds. Real repositories must supply explicit ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

from .blocks import Block, extract_blocks
from .evidence import collect_all
from .execution import ExecutionConfig, run_tests
from .graph import build_graph, structural_importance
from .inference import brier_score, expected_calibration_error, infer
from .sensors import DEFAULT_SENSORS, SensorModel


def program_group(case_id: str) -> str:
    """All operators and the clean reference of a program share one CV group."""
    parts = case_id.split(":")
    return parts[1] if parts[0] == "ref" and len(parts) > 1 else parts[0]


def validate_dataset(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("empty evaluation dataset")
    ownership: dict[str, str] = {}
    seen: set[str] = set()
    content_splits: dict[str, str] = {}
    for row in rows:
        required = {
            "name",
            "repository",
            "source",
            "tests",
            "buggy",
            "unreliable",
            "split",
        }
        if not required <= row.keys() or row["split"] not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(
                "dataset requires named repository-disjoint train/validation/test cases"
            )
        if not row["repository"] or row["name"] in seen:
            raise ValueError("case names must be unique and repositories nonempty")
        seen.add(row["name"])
        previous = ownership.setdefault(row["repository"], row["split"])
        if previous != row["split"]:
            raise ValueError("repository appears in multiple splits")
        digest = hashlib.sha256(row["source"].encode()).hexdigest()
        previous = content_splits.setdefault(digest, row["split"])
        if previous != row["split"]:
            raise ValueError("identical source appears across splits")
        if not set(row["buggy"]) <= set(row["unreliable"]):
            raise ValueError("intrinsically buggy blocks must also be unreliable")
    if set(ownership.values()) != {"train", "validation", "test"}:
        raise ValueError("all three disjoint splits are required")


def collect_case(row: dict, execution: ExecutionConfig) -> dict:
    try:
        blocks = extract_blocks(row["source"])
    except SyntaxError:
        blocks = [
            Block(
                "module:syntax-error",
                "module",
                "<syntax error>",
                "<syntax error>",
                1,
                max(1, row["source"].count("\n") + 1),
                row["source"],
            )
        ]
    if not blocks:
        raise ValueError("dataset contains empty source")
    names = {block.qualname for block in blocks}
    if not set(row["unreliable"]) <= names:
        raise ValueError(f"unknown label in {row['name']}")
    started = time.monotonic()
    result = run_tests(row["source"], row["tests"], execution)
    if not result.valid:
        raise ValueError(
            f"{row['name']}: incomplete test run ({result.status}, {result.errors})"
        )
    evidence = collect_all(row["source"], blocks, execution_result=result)
    return {
        "case": row,
        "blocks": blocks,
        "evidence": evidence,
        "execution": result,
        "seconds": time.monotonic() - started,
    }


def fit_sensor_model(cases: list[dict]) -> SensorModel:
    """Beta(1,1)-smoothed Bernoulli sensor likelihoods, learned on training only."""
    if not cases or any(case["case"]["split"] != "train" for case in cases):
        raise ValueError("sensor fitting accepts training cases only")
    counts = {source: {0: [0, 0], 1: [0, 0]} for source in DEFAULT_SENSORS}
    structural, labels = [], []
    for case in cases:
        for block in case["blocks"]:
            if block.kind in {"module", "segment"}:
                continue
            defective = int(block.qualname in case["case"]["buggy"])
            import math

            structural.append(
                [
                    math.log1p(block.cyclomatic - 1),
                    math.log1p(block.loc),
                    block.depth,
                    block.n_params,
                ]
            )
            labels.append(1 - defective)
            for source in counts:
                if source == "exec":
                    continue  # fit test outcome probabilities below, at test rather than block level
                observations = [
                    e
                    for e in case["evidence"]
                    if e.bid == block.bid
                    and e.source == source
                    and e.polarity != "neutral"
                ]
                if observations:
                    counts[source][defective][1] += 1
                    counts[source][defective][0] += int(
                        any(e.polarity == "negative" for e in observations)
                    )
        by_bid = {b.bid: b.qualname for b in case["blocks"]}
        test_observations = {
            e.meta["test_observation"]["name"]: e.meta["test_observation"]
            for e in case["evidence"]
            if "test_observation" in e.meta
        }
        for observation in test_observations.values():
            defective = int(
                any(
                    by_bid[bid] in case["case"]["buggy"]
                    for bid in observation["blocks"]
                )
            )
            counts["exec"][defective][1] += 1
            counts["exec"][defective][0] += int(observation["failed"])
    likelihoods = {source: dict(values) for source, values in DEFAULT_SENSORS.items()}
    for source, by_label in counts.items():
        # Do not pretend an unobserved class or disabled sensor was calibrated.
        if all(by_label[label][1] > 0 for label in (0, 1)):
            likelihoods[source] = {
                "flag_given_defect": (by_label[1][0] + 1) / (by_label[1][1] + 2),
                "flag_given_clean": (by_label[0][0] + 1) / (by_label[0][1] + 2),
            }
    prior = None
    if len(set(labels)) == 2:
        from sklearn.linear_model import LogisticRegression

        classifier = LogisticRegression(C=0.5, max_iter=1000).fit(structural, labels)
        prior = dict(
            zip(
                ["log_cyclomatic", "log_loc", "depth", "n_params"],
                classifier.coef_[0].tolist(),
            )
        )
        prior["intercept"] = float(classifier.intercept_[0])
    return SensorModel(
        likelihoods,
        prior,
        calibrated=False,
        provenance={
            "training_repositories": sorted({c["case"]["repository"] for c in cases}),
            "sensor_counts": counts,
            "smoothing": "Beta(1,1)",
            "limitation": "test noisy-OR sensitivity is pooled across covered defects",
        },
    )


def predict_cases(
    cases: list[dict], model: SensorModel, variant: str = "full"
) -> list[dict]:
    rows = []
    for case in cases:
        blocks, evidence = case["blocks"], case["evidence"]
        graph = build_graph(
            blocks,
            source=None
            if any(b.bid == "module:syntax-error" for b in blocks)
            else case["case"]["source"],
        )
        if variant == "without_graph":
            graph.remove_edges_from(list(graph.edges))
        elif variant in {"tests_only", "static_only"}:
            evidence = [
                e
                for e in evidence
                if e.source == ("exec" if variant == "tests_only" else "static")
            ]
            graph.remove_edges_from(list(graph.edges))
        elif variant != "full":
            raise ValueError("unknown ablation")
        post = infer(
            blocks, graph, evidence, structural_importance(graph, blocks), model=model
        )
        for block in blocks:
            if block.kind in {"module", "segment"}:
                continue
            posterior = post[block.bid]
            rows.append(
                {
                    "case": case["case"]["name"],
                    "repository": case["case"]["repository"],
                    "block": block.qualname,
                    "correct": int(block.qualname not in case["case"]["buggy"]),
                    "reliable": int(block.qualname not in case["case"]["unreliable"]),
                    "trust": posterior.posterior,
                    "defect": posterior.culpability,
                    "risk": posterior.risk,
                    "seconds": case.get("seconds", 0),
                }
            )
    return rows


def select_threshold(rows: list[dict]) -> float:
    """Select trust threshold only on validation rows of the final runtime model."""
    return max(
        np.linspace(0.1, 0.9, 33),
        key=lambda threshold: metrics(rows, float(threshold))["trust"]["f1"],
    )


def metrics(rows: list[dict], threshold: float = 0.5) -> dict:
    if not rows:
        raise ValueError("cannot score an empty split")

    def score(probabilities, labels, cut):
        tp = sum(p < cut and y == 0 for p, y in zip(probabilities, labels))
        fp = sum(p < cut and y == 1 for p, y in zip(probabilities, labels))
        fn = sum(p >= cut and y == 0 for p, y in zip(probabilities, labels))
        tn = sum(p >= cut and y == 1 for p, y in zip(probabilities, labels))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        return {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "brier": brier_score(probabilities, labels),
            "ece": expected_calibration_error(probabilities, labels),
            "false_trust_rate": fn / max(1, sum(y == 0 for y in labels)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "false_positive_rate": fp / max(1, fp + tn),
            "reliability_bins": reliability_bins(probabilities, labels),
        }

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["case"], []).append(row)
    # Localization ranks intrinsic defects inside each program, not unrelated
    # blocks across repositories; clean cases cannot have a root-cause rank.
    ranks = []
    precisions = []
    for group in groups.values():
        ranked = sorted(group, key=lambda row: (-row["defect"], row["block"]))
        if any(not row["correct"] for row in ranked):
            ranks.append(
                next(i for i, row in enumerate(ranked, 1) if not row["correct"])
            )
        precisions.append(
            sum(1 - row["correct"] for row in ranked[:3]) / min(3, len(ranked))
        )
    return {
        "blocks": len(rows),
        "repositories": len({r["repository"] for r in rows}),
        "trust": score(
            [r["trust"] for r in rows], [r["reliable"] for r in rows], threshold
        ),
        "defect": score(
            [1 - r["defect"] for r in rows], [r["correct"] for r in rows], 0.5
        ),
        "precision_at_3": float(np.mean(precisions)),
        "localization": {
            "buggy_cases": len(ranks),
            "top_1": sum(r == 1 for r in ranks) / len(ranks) if ranks else None,
            "top_3": sum(r <= 3 for r in ranks) / len(ranks) if ranks else None,
            "mean_reciprocal_rank": float(np.mean([1 / r for r in ranks]))
            if ranks
            else None,
            "interpretation": "Intrinsic-defect ranking per labelled buggy program; not exact-line causal proof.",
        },
    }


def reliability_bins(
    probabilities: list[float], labels: list[int], bins: int = 10
) -> list[dict]:
    """Include empty bins explicitly; endpoints 0 and 1 are both retained."""
    result = []
    for index in range(bins):
        selected = [
            (p, y)
            for p, y in zip(probabilities, labels)
            if min(int(p * bins), bins - 1) == index
        ]
        result.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": len(selected),
                "mean_probability": float(np.mean([p for p, _ in selected]))
                if selected
                else None,
                "observed_frequency": float(np.mean([y for _, y in selected]))
                if selected
                else None,
            }
        )
    return result


def bootstrap_metrics(
    rows: list[dict], threshold: float, repeats: int = 500, seed: int = 301
) -> dict:
    """Resample whole repositories, preserving correlated cases and blocks."""
    repositories = sorted({r["repository"] for r in rows})
    if len(repositories) < 2:
        return {
            "status": "insufficient-repositories",
            "repositories": len(repositories),
        }
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(repeats):
        selected = rng.choice(repositories, len(repositories), replace=True)
        sample = [row for repo in selected for row in rows if row["repository"] == repo]
        score = metrics(sample, threshold)
        for scope in ("trust", "defect"):
            for name in ("precision", "recall", "false_positive_rate", "brier", "ece"):
                samples.setdefault(f"{scope}.{name}", []).append(score[scope][name])
    return {
        "status": "measured",
        "repositories": len(repositories),
        "repeats": repeats,
        "seed": seed,
        "unit": "repository",
        "confidence": 0.95,
        "percentile_intervals": {
            key: np.quantile(value, [0.025, 0.975]).tolist()
            for key, value in samples.items()
        },
        "limitation": "Few repositories yield unstable intervals; these do not establish deployment calibration.",
    }


def demo_dataset() -> list[dict]:
    """Small synthetic smoke benchmark, explicitly not real-world validation."""
    from .corpus import REFERENCE_PROGRAMS
    from .mutate import generate_mutants

    rows = []
    for index, (name, (source, tests)) in enumerate(
        list(REFERENCE_PROGRAMS.items())[:6]
    ):
        split = ("train", "train", "validation", "validation", "test", "test")[index]
        rows.append(
            {
                "name": name + ":clean",
                "repository": name,
                "source": source,
                "tests": tests,
                "buggy": [],
                "unreliable": [],
                "split": split,
            }
        )
        # Keep all generated candidates for checking; surviving mutants are
        # not called equivalent. The CLI filters undetected synthetic mutants.
        for mutant in generate_mutants(name, source, tests, max_per_operator=1)[:2]:
            # Label construction stays independent of new statement edges.
            # These are synthetic call-graph labels, not real-world ground truth.
            graph = build_graph(extract_blocks(mutant.source))
            root = next(
                (
                    b.bid
                    for b in extract_blocks(mutant.source)
                    if b.qualname == mutant.buggy_block
                ),
                None,
            )
            impacted = (
                sorted(
                    graph.nodes[n]["block"].qualname
                    for n in nx.descendants(graph, root)
                )
                if root
                else []
            )
            rows.append(
                {
                    "name": mutant.case_id,
                    "repository": name,
                    "source": mutant.source,
                    "tests": tests,
                    "buggy": [mutant.buggy_block],
                    "unreliable": [mutant.buggy_block, *impacted],
                    "split": split,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--curated",
        action="store_true",
        help="independently specified synthetic bug/fix families",
    )
    parser.add_argument("--execution", choices=["local", "docker"], default="docker")
    parser.add_argument("--out", type=Path, default=Path("out/validation"))
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()
    if sum((bool(args.dataset), args.demo, args.curated)) != 1:
        parser.error("choose exactly one of --dataset, --demo or --curated")
    from .evaluation_cases import curated_dataset

    dataset = (
        demo_dataset()
        if args.demo
        else curated_dataset()
        if args.curated
        else json.loads(args.dataset.read_text(encoding="utf-8"))
    )
    validate_dataset(dataset)
    cases, excluded = [], []
    for row in dataset:
        print(f"Collecting {row['name']} ({row['split']})", flush=True)
        try:
            case = collect_case(row, ExecutionConfig(args.execution, args.timeout))
        except ValueError as exc:
            if not args.demo or "incomplete test run" not in str(exc):
                raise
            excluded.append({"case": row["name"], "reason": str(exc)})
            continue
        if args.demo and row["buggy"] and not case["execution"].failed_ids:
            excluded.append(
                {"case": row["name"], "reason": "survived tests; equivalence unknown"}
            )
            continue
        if args.demo and not row["buggy"] and case["execution"].failed_ids:
            raise ValueError("synthetic reference fails its own tests")
        cases.append(case)
    train = [c for c in cases if c["case"]["split"] == "train"]
    validation = [c for c in cases if c["case"]["split"] == "validation"]
    test = [c for c in cases if c["case"]["split"] == "test"]
    model = fit_sensor_model(train)
    validation_rows = predict_cases(validation, model)
    threshold = float(select_threshold(validation_rows))
    model = SensorModel(
        model.likelihoods, model.prior_coefficients, False, threshold, model.provenance
    )
    report: dict = {
        "scope": "synthetic mutation smoke benchmark"
        if args.demo
        else "independently specified synthetic families"
        if args.curated
        else "user-labelled held-out repositories",
        "dataset_sha256": hashlib.sha256(
            json.dumps(dataset, sort_keys=True).encode()
        ).hexdigest(),
        "splits": {
            s: sorted(
                {c["case"]["repository"] for c in cases if c["case"]["split"] == s}
            )
            for s in ("train", "validation", "test")
        },
        "excluded": excluded,
        "threshold": threshold,
        "validation": metrics(validation_rows, threshold),
        "test": {
            variant: metrics(predict_cases(test, model, variant), threshold)
            for variant in ("full", "without_graph", "tests_only", "static_only")
        },
        "default_model_test": metrics(predict_cases(test, SensorModel()), 0.5),
        "test_uncertainty": bootstrap_metrics(predict_cases(test, model), threshold),
        "test_predictions": predict_cases(test, model),
        "case_outcomes": [
            {
                "case": c["case"]["name"],
                "split": c["case"]["split"],
                "labelled_buggy": bool(c["case"]["buggy"]),
                "oracle_failure": bool(c["execution"].failed_ids),
                "seconds": c["seconds"],
            }
            for c in cases
        ],
        "calibration_claim": "Measured scores only; no automatic promotion to a calibrated deployment model.",
    }
    fitted_score = report["test"]["full"]
    default_score = report["default_model_test"]
    report["calibration_assessment"] = {
        "deployment_model_changed": False,
        "status": "insufficient-independent-real-world-data"
        if args.demo or args.curated
        else "held-out-measurements-only",
        "brier_change_fitted_minus_default": {
            name: fitted_score[name]["brier"] - default_score[name]["brier"]
            for name in ("trust", "defect")
        },
        "ece_change_fitted_minus_default": {
            name: fitted_score[name]["ece"] - default_score[name]["ece"]
            for name in ("trust", "defect")
        },
        "interpretation": "Negative changes are better on this split. ECE depends on binning and sample size; review Brier, reliability bins, false alarms and repository intervals together. Neither metric proves deployment calibration.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sensor_model.json").write_text(
        json.dumps(model.to_dict(), indent=2), encoding="utf-8"
    )
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Probability and localization evaluation",
        "",
        report["scope"],
        "",
        "| Model | Target | Precision | Recall | False-positive rate | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, score in [("default", default_score), *report["test"].items()]:
        for target in ("trust", "defect"):
            values = score[target]
            lines.append(
                f"| {name} | {target} | "
                + " | ".join(
                    f"{values[key]:.4f}"
                    for key in (
                        "precision",
                        "recall",
                        "false_positive_rate",
                        "brier",
                        "ece",
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Defect scores evaluate intrinsic defects; trust scores evaluate labelled downstream unreliability.",
            "",
            "Localization is measured within each program. Inspect report.json for ranks, reliability bins, predictions and repository bootstrap intervals.",
            "",
            "No deployment model was changed. Synthetic families are not independent real-world repositories. Small-sample scores do not establish deployment calibration.",
        ]
    )
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "scope": report["scope"],
                "threshold": threshold,
                "calibration_assessment": report["calibration_assessment"],
                "report": str(args.out / "report.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
