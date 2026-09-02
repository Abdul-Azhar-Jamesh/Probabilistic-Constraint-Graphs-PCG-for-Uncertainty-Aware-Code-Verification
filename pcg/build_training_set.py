"""Build a labelled training set from the mutation corpus.

For each detectable mutant (and each un-mutated reference program), runs the
evidence pipeline and extracts per-block feature vectors with ground-truth
correctness labels.

Run:  python -m pcg.build_training_set
Output: out/training_set.csv, out/training_set.json
"""

from __future__ import annotations

import csv
import json
import math
import os
import pickle
from dataclasses import asdict, dataclass

from .blocks import extract_blocks
from .corpus import REFERENCE_PROGRAMS
from .evidence import collect_all
from .mutate import Mutant, annotate_downstream, build_corpus


def _load_mutation_cases(
    use_cache: bool = True,
    max_mutants: int | None = None,
    verbose: bool = True,
) -> list[Mutant]:
    """Return the (possibly cached) detectable mutant corpus.

    Parameters
    ----------
    use_cache:
        Load from ``out/corpus_mutants.pkl`` when it exists.  Set to
        ``False`` (or delete the file) after adding new reference programs.
    max_mutants:
        Cap the number of mutants returned.  Useful during fast iteration
        cycles where running all 96+ cases is too slow.  E.g. pass 40 to
        use roughly the first 40 detectable mutants.  ``None`` means no cap.
    verbose:
        Print progress messages.
    """
    cache_path = os.path.join("out", "corpus_mutants.pkl")

    if use_cache and os.path.exists(cache_path):
        if verbose:
            print("Loading cached mutant corpus...")
        with open(cache_path, "rb") as fh:
            mutants: list[Mutant] = pickle.load(fh)
    else:
        if verbose:
            print("Building mutant corpus (this takes a few minutes)...")
        mutants = build_corpus(verbose=verbose)
        mutants = annotate_downstream(mutants)
        os.makedirs("out", exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(mutants, fh)

    if max_mutants is not None:
        mutants = mutants[:max_mutants]

    if verbose:
        print(f"Corpus: {len(mutants)} detectable mutants"
              f"{f' (capped at {max_mutants})' if max_mutants is not None else ''}")

    return mutants


@dataclass
class BlockFeatures:
    """Feature vector for one block in one program variant."""

    # Identifiers (not model features)
    case_id: str
    block_qualname: str
    label: int  # 1 = correct, 0 = locally buggy
    is_downstream: int = 0

    # Evidence counts per source
    compile_neg_count: int = 0
    compile_pos_count: int = 0
    static_neg_count: int = 0
    static_pos_count: int = 0
    exec_neg_count: int = 0
    exec_pos_count: int = 0
    llm_neg_count: int = 0
    llm_pos_count: int = 0
    critic_neg_count: int = 0
    critic_pos_count: int = 0

    # Evidence weight sums per source (total magnitude)
    compile_neg_weight_sum: float = 0.0
    compile_pos_weight_sum: float = 0.0
    static_neg_weight_sum: float = 0.0
    static_pos_weight_sum: float = 0.0
    exec_neg_weight_sum: float = 0.0
    exec_pos_weight_sum: float = 0.0
    llm_neg_weight_sum: float = 0.0
    llm_pos_weight_sum: float = 0.0
    critic_neg_weight_sum: float = 0.0
    critic_pos_weight_sum: float = 0.0

    # Structural features (for the prior)
    cyclomatic: int = 1
    loc: int = 1
    depth: int = 0
    n_params: int = 0

    # Derived structural features matching prior_correctness() transforms
    log_cyclomatic: float = 0.0
    log_loc: float = 0.0

    # Evidence-strength interactions: a mathematically-defined combination of
    # signed evidence strengths, not arbitrary heuristic boosts. The pairwise
    # interaction is the product of the net evidence signals, so it captures
    # whether two sources agree or disagree about the same block.
    static_exec_interaction: float = 0.0
    static_critic_interaction: float = 0.0
    exec_critic_interaction: float = 0.0
    llm_critic_interaction: float = 0.0


EVIDENCE_FEATURE_NAMES = [
    "compile_neg_count", "compile_pos_count",
    "static_neg_count", "static_pos_count",
    "exec_neg_count", "exec_pos_count",
    "llm_neg_count", "llm_pos_count",
    "critic_neg_count", "critic_pos_count",
    "compile_neg_weight_sum", "compile_pos_weight_sum",
    "static_neg_weight_sum", "static_pos_weight_sum",
    "exec_neg_weight_sum", "exec_pos_weight_sum",
    "llm_neg_weight_sum", "llm_pos_weight_sum",
    "critic_neg_weight_sum", "critic_pos_weight_sum",
    "static_exec_interaction", "static_critic_interaction",
    "exec_critic_interaction", "llm_critic_interaction",
]

STRUCTURAL_FEATURE_NAMES = [
    "log_cyclomatic", "log_loc", "depth", "n_params",
]

ALL_FEATURE_NAMES = EVIDENCE_FEATURE_NAMES + STRUCTURAL_FEATURE_NAMES


def _extract_block_features(
    case_id: str,
    source: str,
    tests: str,
    buggy_qualnames: set[str],
    downstream_qualnames: set[str],
    critic_backend: str | None = None,
) -> list[BlockFeatures]:
    """Run evidence pipeline and extract features for every block."""
    try:
        blocks = extract_blocks(source)
    except SyntaxError:
        return []

    if not blocks:
        return []

    ev = collect_all(source, blocks, tests, critic_backend=critic_backend)

    # Group evidence by block
    ev_by_block: dict[str, list] = {b.bid: [] for b in blocks}
    for e in ev:
        if e.bid in ev_by_block:
            ev_by_block[e.bid].append(e)

    results: list[BlockFeatures] = []
    for b in blocks:
        if b.kind == "module":
            continue  # skip module-level blocks

        # Determine label for intrinsic defect detection. Downstream blocks are
        # not automatically wrong; they may simply inherit doubt from a bad
        # dependency. Keep that information as a separate annotation.
        is_buggy = b.qualname in buggy_qualnames
        is_downstream = b.qualname in downstream_qualnames
        label = 0 if is_buggy else 1

        bf = BlockFeatures(
            case_id=case_id,
            block_qualname=b.qualname,
            label=label,
            is_downstream=1 if is_downstream else 0,
            cyclomatic=b.cyclomatic,
            loc=b.loc,
            depth=b.depth,
            n_params=b.n_params,
            log_cyclomatic=math.log1p(b.cyclomatic - 1),
            log_loc=math.log1p(b.loc),
        )

        # Accumulate evidence features
        for e in ev_by_block[b.bid]:
            src = e.source
            pol = e.polarity
            key_count = f"{src}_{pol[:3]}_count"
            key_weight = f"{src}_{pol[:3]}_weight_sum"
            if hasattr(bf, key_count):
                setattr(bf, key_count, getattr(bf, key_count) + 1)
            if hasattr(bf, key_weight):
                setattr(bf, key_weight, getattr(bf, key_weight) + e.weight)

        # Pairwise evidence interactions are derived from the existing signed
        # evidence strengths, not arbitrary tuning knobs.
        net = {
            src: getattr(bf, f"{src}_neg_weight_sum") - getattr(bf, f"{src}_pos_weight_sum")
            for src in ("static", "exec", "llm", "critic")
        }
        bf.static_exec_interaction = net["static"] * net["exec"]
        bf.static_critic_interaction = net["static"] * net["critic"]
        bf.exec_critic_interaction = net["exec"] * net["critic"]
        bf.llm_critic_interaction = net["llm"] * net["critic"]

        results.append(bf)

    return results


def root_defect_metrics(
    truth_roots: set[str] | list[str],
    predicted_roots: set[str] | list[str],
) -> dict[str, float]:
    """Precision, recall, and F1 for source-defect localisation."""
    truth = set(truth_roots)
    pred = set(predicted_roots)
    tp = len(truth & pred)
    fp = len(pred - truth)
    fn = len(truth - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def build_training_set(
    use_cache: bool = True,
    max_mutants: int | None = None,
    verbose: bool = True,
) -> list[BlockFeatures]:
    """Generate the full training set from mutations + clean references.

    Parameters
    ----------
    use_cache:
        Re-use ``out/corpus_mutants.pkl`` when it exists.
    max_mutants:
        Passed through to :func:`_load_mutation_cases`.  Cap the mutant
        count for fast iteration (e.g. ``max_mutants=40``).
    verbose:
        Print progress messages.
    """
    # -- Step 1: get or build the mutant corpus ----------------------------
    mutants = _load_mutation_cases(
        use_cache=use_cache,
        max_mutants=max_mutants,
        verbose=verbose,
    )
    # -- Step 2: extract features from mutants -----------------------------
    all_features: list[BlockFeatures] = []
    for i, m in enumerate(mutants):
        buggy = {m.buggy_block}
        downstream = m.downstream if m.downstream else set()

        feats = _extract_block_features(
            case_id=m.case_id,
            source=m.source,
            tests=m.tests,
            buggy_qualnames=buggy,
            downstream_qualnames=downstream,
            critic_backend="mock",
        )
        all_features.extend(feats)
        if verbose and (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{len(mutants)} mutants "
                  f"({len(all_features)} block samples so far)")

    # -- Step 3: add clean reference programs as all-correct examples ------
    if verbose:
        print("\nAdding clean reference programs...")
    for name, (src, tests) in REFERENCE_PROGRAMS.items():
        feats = _extract_block_features(
            case_id=f"ref:{name}",
            source=src,
            tests=tests,
            buggy_qualnames=set(),
            downstream_qualnames=set(),
            critic_backend="mock",
        )
        all_features.extend(feats)

    if verbose:
        n_correct = sum(1 for f in all_features if f.label == 1)
        n_buggy = sum(1 for f in all_features if f.label == 0)
        print(f"\nTraining set: {len(all_features)} block samples "
              f"({n_correct} correct, {n_buggy} buggy/downstream)")

    return all_features


def save_training_set(features: list[BlockFeatures], out_dir: str = "out") -> None:
    """Write the training set to CSV and JSON."""
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(out_dir, "training_set.csv")
    fieldnames = list(BlockFeatures.__dataclass_fields__.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for bf in features:
            writer.writerow(asdict(bf))

    # JSON
    json_path = os.path.join(out_dir, "training_set.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump([asdict(bf) for bf in features], fh, indent=2)

    print(f"Saved: {csv_path} ({len(features)} rows)")
    print(f"Saved: {json_path}")


def main() -> None:
    features = build_training_set()
    save_training_set(features)


if __name__ == "__main__":
    main()
