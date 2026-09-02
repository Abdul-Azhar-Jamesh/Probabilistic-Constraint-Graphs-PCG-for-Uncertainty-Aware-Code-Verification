"""Bayesian inference over the Probabilistic Constraint Graph.

Model
-----
Each block i carries a latent binary variable C_i ("this block is correct").
We want P(C_i = 1 | E), the posterior given all collected evidence.

Three stages:

1. **Prior** P(C_i) from structural complexity. Simple, short, low-branching
   blocks are a priori more likely correct.

2. **Local update** via naive-Bayes over conditionally-independent evidence
   items, applied in log-odds space. Each evidence item contributes a
   likelihood ratio derived from its source reliability and weight.

3. **Constraint propagation** over the graph. A block cannot be more correct
   than the things it depends on: we apply a noisy-AND over parents, then
   iterate to a fixed point so that evidence reaches transitive dependents.

Stage 3 is what makes this a *graph* method rather than per-block scoring, and
it is where a bug in a leaf utility correctly drags down every caller.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import networkx as nx

from .blocks import Block
from .evidence import Evidence

# Per-source reliability: how much a single observation from this source
# should move belief. These are the sensor model's tunable parameters and are
# the natural target for calibration against a labelled dataset.
SOURCE_RELIABILITY_HANDTUNED = {
    "compile": 3.2,
    "exec": 2.8,
    "static": 1.4,
    "llm": 0.45,
    "critic": 1.2,
}

# Prior coefficients used in prior_correctness() before calibration.
PRIOR_COEFFICIENTS_HANDTUNED = {
    "intercept": 0.0,
    "log_cyclomatic": 0.22,
    "log_loc": 0.030,
    "depth": 0.10,
    "n_params": 0.05,
}

# Validation-selected threshold persists in JSON so the final holdout run uses a
# fixed threshold chosen only on validation data.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTED_THRESHOLD_PATH = os.path.join(PROJECT_ROOT, "out", "selected_threshold.json")
FITTED_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "out", "fitted_weights.json")


def load_selected_threshold(default: float = 0.5) -> float:
    """Load the saved validation-only threshold, if present."""
    if not os.path.exists(SELECTED_THRESHOLD_PATH):
        return default
    try:
        with open(SELECTED_THRESHOLD_PATH, encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict) and "selected_threshold" in payload:
            value = payload["selected_threshold"]
            if isinstance(value, (int, float)):
                return float(value)
    except (json.JSONDecodeError, OSError):
        pass
    return default


def classify_abstention(p: float, threshold: float = 0.5, calibration_ece: float = 0.05) -> str:
    """Three-way posteriors: trusted, uncertain, or defective.

    The uncertain band is anchored on the validation-selected threshold and grows
    with calibration error, so the interval is empirical rather than arbitrary.
    """
    margin = max(0.05, 0.5 * calibration_ece + 0.05)
    if p >= threshold + margin:
        return "LIKELY_CORRECT"
    if p <= threshold - margin:
        return "LIKELY_DEFECTIVE"
    return "UNCERTAIN"


def _load_fitted_weights() -> dict | None:
    """Try to load fitted weights from calibration output.

    Resolved relative to the project root (not the CWD) so behaviour is
    identical no matter where the pipeline is invoked from. A fit is rejected
    only if the payload itself is malformed; zero-valued reliability entries are
    allowed because a source may legitimately contribute no signal in a given
    fit. The runtime still prefers a numerically valid model and will fall back
    to hand-tuned values only when no valid calibration file exists.
    """
    if not os.path.exists(FITTED_WEIGHTS_PATH):
        return None
    try:
        with open(FITTED_WEIGHTS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
            if not isinstance(data, dict):
                return None
            rel = data.get("source_reliability_fitted", {})
            if not isinstance(rel, dict):
                return None
            if any(not isinstance(v, (int, float)) for v in rel.values()):
                return None
            return data
    except (json.JSONDecodeError, OSError):
        return None


_FITTED = _load_fitted_weights()

SOURCE_RELIABILITY: dict[str, float] = (
    _FITTED["source_reliability_fitted"]
    if _FITTED and "source_reliability_fitted" in _FITTED
    else dict(SOURCE_RELIABILITY_HANDTUNED)
)

_PRIOR_COEFFICIENTS: dict[str, float] = (
    _FITTED["prior_coefficients_fitted"]
    if _FITTED and "prior_coefficients_fitted" in _FITTED
    else dict(PRIOR_COEFFICIENTS_HANDTUNED)
)

# Positive evidence is discounted relative to negative evidence of the same
# nominal weight. Passing tests and clean linters are weak proof of
# correctness (they only cover what they exercise), whereas a failure is
# near-proof of a defect. Without this asymmetry, an accumulation of cheap
# positives can outvote a single decisive failure.
POSITIVE_DISCOUNT = 0.62

MAX_LOG_ODDS = 8.0  # keeps probabilities inside ~[3e-4, 0.9997]


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    x = max(-MAX_LOG_ODDS, min(MAX_LOG_ODDS, x))
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class BlockPosterior:
    bid: str
    prior: float
    local: float  # after evidence, before propagation
    posterior: float  # after constraint propagation
    entropy: float
    importance: float = 0.0
    risk: float = 0.0
    culpability: float = 0.0  # doubt originating *here*, not inherited
    inherited: float = 0.0  # doubt arriving from dependencies
    uncertainty: float = 0.0  # std of posterior under sensor-model resampling
    direct_negative: float = 0.0
    direct_positive: float = 0.0
    contributions: list[tuple[str, float]] = field(default_factory=list)
    top_reasons: list[str] = field(default_factory=list)

    @property
    def repair_priority(self) -> float:
        """How urgently this block's own code needs modification."""
        return self.culpability

    @property
    def review_priority(self) -> float:
        """Overall human review urgency considering blast radius."""
        return self.risk

    @property
    def trust_score(self) -> float:
        """Can this block's output be trusted given current dependencies?"""
        return self.posterior


def prior_correctness(b: Block) -> float:
    """Structural prior on P(correct).

    Anchored at 0.86 for a trivial block, decaying with complexity. The
    coefficients (including the intercept) are loaded from fitted weights
    (``out/fitted_weights.json``) when available, otherwise falling back to
    the hand-tuned defaults.
    """
    c = _PRIOR_COEFFICIENTS
    penalty = (
        c["log_cyclomatic"] * math.log1p(b.cyclomatic - 1)
        + c["log_loc"] * math.log1p(b.loc)
        + c["depth"] * b.depth
        + c["n_params"] * b.n_params
    )
    if b.kind == "module":
        penalty *= 0.6  # module-level glue is usually simple assignments
    base = 0.86 * math.exp(-penalty)
    # Fitted models express their anchor as logit(intercept); apply it only
    # when a calibrated intercept is actually present so hand-tuned behaviour
    # is unchanged.
    if _FITTED and "prior_coefficients_fitted" in _FITTED:
        intercept = float(_PRIOR_COEFFICIENTS.get("intercept", 0.0))
        base = _sigmoid(_logit(base) + intercept)
    return min(0.93, max(0.25, base))


def evidence_log_lr(e: Evidence) -> float:
    """Log-likelihood ratio log[ P(e|correct) / P(e|incorrect) ].

    Positive evidence pushes toward correct, negative away. Magnitude scales
    with source reliability and the observation's own weight, with positives
    discounted (absence of evidence of a bug is weak evidence of absence).
    """
    rel = SOURCE_RELIABILITY.get(e.source, 1.0)
    mag = rel * e.weight
    return mag * POSITIVE_DISCOUNT if e.polarity == "positive" else -mag


def local_posteriors(
    blocks: list[Block], evidence: list[Evidence]
) -> dict[str, BlockPosterior]:
    """Stage 1 + 2: prior, then naive-Bayes evidence fusion."""
    by_block: dict[str, list[Evidence]] = {b.bid: [] for b in blocks}
    for e in evidence:
        if e.bid in by_block:
            by_block[e.bid].append(e)

    out: dict[str, BlockPosterior] = {}
    for b in blocks:
        pri = prior_correctness(b)
        lo = _logit(pri)
        contribs: list[tuple[str, float]] = []
        direct_negative = 0.0
        direct_positive = 0.0
        for e in by_block[b.bid]:
            d = evidence_log_lr(e)
            lo += d
            if e.polarity == "negative":
                direct_negative += abs(d)
            else:
                direct_positive += d
            contribs.append((f"{e.source}:{e.kind}", round(d, 3)))
        p = _sigmoid(lo)
        contribs.sort(key=lambda t: t[1])
        out[b.bid] = BlockPosterior(
            bid=b.bid,
            prior=round(pri, 4),
            local=round(p, 4),
            posterior=round(p, 4),
            entropy=0.0,
            direct_negative=round(direct_negative, 4),
            direct_positive=round(direct_positive, 4),
            contributions=contribs,
            top_reasons=[
                e.detail
                for e in sorted(by_block[b.bid], key=lambda x: -abs(evidence_log_lr(x)))
                if e.polarity == "negative"
            ][:3],
        )
    return out


def propagate(
    g: nx.DiGraph,
    post: dict[str, BlockPosterior],
    iterations: int = 12,
    damping: float = 0.55,
) -> dict[str, BlockPosterior]:
    """Stage 3: noisy-AND constraint propagation to a fixed point.

    For node i with dependency parents pa(i):

        P(C_i) <- P_local(i) * prod_{j in pa(i)} [ 1 - s_ij * (1 - P(C_j)) ]

    where s_ij is the coupling strength of the edge. If a dependency is
    certainly correct the factor is 1 (no effect); if it is certainly wrong the
    factor is (1 - s_ij), so a strong `calls` edge caps the dependent at 0.15.

    Damped fixed-point iteration handles cycles (mutual recursion) gracefully.
    """
    belief = {bid: bp.local for bid, bp in post.items()}

    for _ in range(iterations):
        delta = 0.0
        nxt = dict(belief)
        for bid in g.nodes:
            if bid not in post:
                continue
            factor = 1.0
            for parent in g.predecessors(bid):
                s = g.edges[parent, bid].get("strength", 0.5)
                factor *= 1.0 - s * (1.0 - belief.get(parent, 1.0))
            target = post[bid].local * factor
            new = damping * belief[bid] + (1 - damping) * target
            delta = max(delta, abs(new - belief[bid]))
            nxt[bid] = new
        belief = nxt
        if delta < 1e-5:
            break

    for bid, bp in post.items():
        p = min(max(belief.get(bid, bp.local), 1e-4), 1 - 1e-4)
        bp.posterior = round(p, 4)
        bp.entropy = round(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)), 4)
    return post


def propagate_bidirectional(
    g: nx.DiGraph,
    post: dict[str, BlockPosterior],
    iterations: int = 12,
    damping: float = 0.55,
) -> dict[str, BlockPosterior]:
    """Two-direction Bayesian message passing over the constraint graph.

    The classic noisy-AND only pushes doubt *upward* (callee -> caller). But
    evidence flows both ways in a Bayesian network:

    **Upward (doubt):** if my callee is broken, I am probably wrong — the
    noisy-AND factor, as in `propagate`.

    **Downward (exoneration):** if my caller executed and its tests PASSED,
    then I probably behaved correctly for that input — a passing caller is
    likelihood evidence FOR my correctness. Formally, P(C_j | C_i=1, test
    through i passed) > P(C_j). We implement this as a soft upward lift:

        support(j) += sum_i s_ij * (P(C_i) - local_i)   for callers i

    where a caller whose posterior exceeds its own local belief has
    "vouched" for its dependencies. This lets clean top-level tests partially
    exonerate leaf utilities that no test directly covers — something pure
    noisy-AND can never express.

    Both messages are damped and iterated to a fixed point.
    """
    belief = {bid: bp.local for bid, bp in post.items()}
    local = {bid: bp.local for bid, bp in post.items()}

    for _ in range(iterations):
        delta = 0.0
        nxt = dict(belief)
        for bid in g.nodes:
            if bid not in post:
                continue
            # Upward: noisy-AND over dependencies (doubt from below).
            up_factor = 1.0
            for parent in g.predecessors(bid):
                s = g.edges[parent, bid].get("strength", 0.5)
                up_factor *= 1.0 - s * (1.0 - belief.get(parent, 1.0))
            # Downward: exoneration from passing callers (support from above).
            down_lift = 0.0
            for child in g.successors(bid):
                if child not in post:
                    continue
                s = g.edges[bid, child].get("strength", 0.5)
                vouch = max(0.0, belief.get(child, 0.0) - local[child])
                down_lift += s * vouch
            target = min(1.0, local[bid] * up_factor + 0.30 * down_lift)
            new = damping * belief[bid] + (1 - damping) * target
            delta = max(delta, abs(new - belief[bid]))
            nxt[bid] = new
        belief = nxt
        if delta < 1e-5:
            break

    for bid, bp in post.items():
        p = min(max(belief.get(bid, bp.local), 1e-4), 1 - 1e-4)
        bp.posterior = round(p, 4)
        bp.entropy = round(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)), 4)
    return post


def sample_posteriors(
    evidence: list[Evidence],
    blocks: list[Block],
    n_samples: int = 200,
    seed: int = 7,
) -> dict[str, tuple[float, float]]:
    """Monte-Carlo uncertainty over the sensor model.

    The naive-Bayes fusion treats evidence weights as exact. They are not:
    they are estimates of sensor behaviour. We model each source's reliability
    as a Beta distribution centred on its nominal value and resample the
    fusion `n_samples` times. Returns per block (mean, std) of the local
    posterior — the std is an honest "how sure is the model about itself"
    number that downstream UIs can surface.

    Blocks whose evidence comes from one unreliable source get wide intervals;
    blocks with converging multi-source evidence stay tight.
    """
    import random

    rng = random.Random(seed)
    by_block: dict[str, list[Evidence]] = {b.bid: [] for b in blocks}
    for e in evidence:
        if e.bid in by_block:
            by_block[e.bid].append(e)

    sources = sorted({e.source for e in evidence})
    samples: dict[str, list[float]] = {b.bid: [] for b in blocks}
    for _ in range(n_samples):
        # Reliabilities are unbounded positive weights, not probabilities, so
        # we resample multiplicatively around the nominal value (log-normal
        # style jitter) rather than a Beta. This keeps draws strictly
        # positive and centred on the calibrated weight.
        rel_draw = {
            s: SOURCE_RELIABILITY.get(s, 1.0) * rng.lognormvariate(0.0, 0.25)
            for s in sources
        }
        for b in blocks:
            lo = _logit(prior_correctness(b))
            for e in by_block[b.bid]:
                mag = rel_draw.get(e.source, 1.0) * e.weight
                # Jitter individual observation weights too (sensor noise).
                mag *= rng.uniform(0.85, 1.15)
                lo += mag * POSITIVE_DISCOUNT if e.polarity == "positive" else -mag
            samples[b.bid].append(_sigmoid(lo))

    out: dict[str, tuple[float, float]] = {}
    for bid, vals in samples.items():
        if not vals:
            continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        out[bid] = (round(m, 4), round(math.sqrt(var), 4))
    return out


def infer(
    blocks: list[Block],
    g: nx.DiGraph,
    evidence: list[Evidence],
    importance: dict[str, float] | None = None,
    damping: float = 0.55,
) -> dict[str, BlockPosterior]:
    """Full pipeline: prior -> evidence fusion -> constraint propagation."""
    post = local_posteriors(blocks, evidence)
    post = propagate_bidirectional(g, post, damping=damping)
    if importance:
        for bid, bp in post.items():
            bp.importance = round(importance.get(bid, 0.0), 4)

    # Monte-Carlo sensor-model uncertainty: how stable is each posterior if
    # the evidence weights themselves are uncertain?
    try:
        mc = sample_posteriors(evidence, blocks)
        for bid, (_, std) in mc.items():
            if bid in post:
                post[bid].uncertainty = std
    except Exception:
        pass  # uncertainty is supplementary; never fail the analysis for it

    # Decompose total doubt into what originates here vs. what was inherited
    # from dependencies. Only direct evidence determines culpability and repair.
    for bid, bp in post.items():
        bp.culpability = round(
            bp.direct_negative
            / (1.0 + bp.direct_negative + max(bp.direct_positive, 0.0)),
            4,
        )
        bp.inherited = round(max(0.0, bp.local - bp.posterior), 4)

    for bid, bp in post.items():
        weight = 0.55 + 0.45 * bp.importance if importance else 1.0
        # Explicit risk formulation balancing own defect, inherited doubt, and entropy
        bp.risk = round(
            (0.70 * bp.culpability + 0.20 * bp.inherited + 0.10 * bp.entropy)
            * weight,
            4,
        )
    return post


def review_ranking(
    post: dict[str, BlockPosterior], blocks: list[Block]
) -> list[tuple[Block, BlockPosterior]]:
    """Blocks ordered by how urgently a human should look at them."""
    bmap = {b.bid: b for b in blocks}
    pairs = [(bmap[bid], bp) for bid, bp in post.items() if bid in bmap]
    pairs.sort(key=lambda t: (-t[1].risk, t[1].posterior))
    return pairs


def repair_targets(
    post: dict[str, BlockPosterior],
    threshold: float = 0.5,
) -> list[str]:
    """Return blocks whose own implementation is suspicious.

    `posterior` answers:
        Can I trust this block's output in the current program?

    `culpability` answers:
        Is this block itself likely defective?

    Repair should use culpability so that a correct caller is not
    regenerated merely because a dependency is broken.
    """
    return [
        bid
        for bid, bp in sorted(
            post.items(),
            key=lambda item: (-item[1].culpability, item[1].posterior),
        )
        if bp.culpability >= threshold
    ]


def find_root_repairs(
    post: dict[str, BlockPosterior],
    graph: nx.DiGraph,
    threshold: float = 0.5,
) -> list[str]:
    """Find root cause defective blocks using the constraint graph.

    Filters candidates whose culpability is >= threshold to those that do NOT
    have an upstream dependency that is also a candidate, isolating the
    originating defect rather than cascade effects.
    """
    candidates = {
        bid for bid, bp in post.items() if bp.culpability >= threshold
    }

    roots = []
    for bid in candidates:
        upstream_bad = any(
            parent in candidates for parent in graph.predecessors(bid)
        )
        if not upstream_bad:
            roots.append(bid)

    return sorted(roots, key=lambda b: -post[b].culpability)


def selected_threshold_from_validation(
    validation_scores: list[tuple[float, int]],
    metric: str = "f1",
    default: float = 0.5,
) -> float:
    """Choose a threshold from validation data only.

    This is deliberately simple: we sweep a grid in [0.1, 0.9], score on the held-out
    validation set, and keep the threshold that maximises the selected metric. The
    value is saved for later final holdout use so the test split cannot be tuned.
    """
    if not validation_scores:
        return default
    best_threshold = default
    best_value = float("-inf")
    for threshold in [x / 100 for x in range(10, 91)]:
        preds = [1 if p >= threshold else 0 for p, _ in validation_scores]
        labels = [int(y) for _, y in validation_scores]
        tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
        fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
        fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        score = f1 if metric == "f1" else precision if metric == "precision" else recall
        if score > best_value:
            best_value = score
            best_threshold = threshold
    return round(best_threshold, 2)


def save_selected_threshold(threshold: float, out_dir: str = "out") -> None:
    """Persist the validation-selected threshold for the final holdout run."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "selected_threshold.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"selected_threshold": float(threshold), "metric": "f1"}, fh, indent=2)


def expected_calibration_error(
    probs: list[float], labels: list[int], bins: int = 10
) -> float:
    """ECE — the standard metric for whether stated confidence is honest.

    This is the number that justifies the whole framework: a system that says
    "0.7" should be right about 70% of the time.
    """
    if not probs:
        return 0.0
    n = len(probs)
    ece = 0.0
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo < p <= hi or (k == 0 and p == 0))]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(acc - conf)
    return round(ece, 4)


def brier_score(probs: list[float], labels: list[int]) -> float:
    if not probs:
        return 0.0
    return round(sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs), 4)
