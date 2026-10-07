"""Graph-based latent-defect inference and output reliability queries.

Local observation likelihoods update structural priors. Shared, coverage-backed
noisy-OR test factors condition the joint defect model. Exact enumeration or
Gibbs sampling supplies posterior marginals. Dependency paths then determine
output trust while preserving correlations and counting each ancestor once.

Observation models are supplied explicitly; obsolete fitted artifacts are not loaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx

from .blocks import Block
from .evidence import Evidence
from .probabilistic import TestFactor, condition_defects, dependency_impacts
from .sensors import SensorModel, test_sensitivity, ORACLE_SENSITIVITY_SCALE

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


def classify_abstention(
    p: float, threshold: float = 0.5, calibration_ece: float = 0.05
) -> str:
    """Three-way scores: likely correct, uncertain, or defective.

    The review band is a policy margin. Supply independently measured calibration
    error explicitly; the default is a conservative display assumption.
    """
    margin = max(0.05, 0.5 * calibration_ece + 0.05)
    if p >= threshold + margin:
        return "LIKELY_CORRECT"
    if p <= threshold - margin:
        return "LIKELY_DEFECTIVE"
    return "UNCERTAIN"


SOURCE_RELIABILITY: dict[str, float] = dict(SOURCE_RELIABILITY_HANDTUNED)
_PRIOR_COEFFICIENTS: dict[str, float] = dict(PRIOR_COEFFICIENTS_HANDTUNED)


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
    uncertainty: float = 0.0  # Monte Carlo standard error of the trust query
    execution_status: str = "untested"
    inference_method: str = "exact"
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


def prior_correctness(b: Block, model: SensorModel | None = None) -> float:
    """Structural prior; fitted logistic coefficients retain their original link."""
    if model and model.prior_coefficients is not None:
        c = model.prior_coefficients
        value = (
            c["intercept"]
            + c["log_cyclomatic"] * math.log1p(b.cyclomatic - 1)
            + c["log_loc"] * math.log1p(b.loc)
            + c["depth"] * b.depth
            + c["n_params"] * b.n_params
        )
        return _sigmoid(value)
    c = _PRIOR_COEFFICIENTS
    penalty = (
        c["log_cyclomatic"] * math.log1p(b.cyclomatic - 1)
        + c["log_loc"] * math.log1p(b.loc)
        + c["depth"] * b.depth
        + c["n_params"] * b.n_params
    )
    if b.kind == "module":
        penalty *= 0.6
    return min(0.93, max(0.25, 0.86 * math.exp(-penalty)))


def evidence_log_lr(
    e: Evidence,
    model: SensorModel | None = None,
    reliabilities: dict[str, float] | None = None,
) -> float:
    """Log P(observation|clean)/P(observation|defective).

    Each source's aggregated finding is a Bernoulli observation. Tempering by
    severity is an explicit power likelihood, not a measured probability.
    Structured test outcomes are handled jointly by noisy-OR factors instead.
    """
    if (
        e.polarity == "neutral"
        or e.meta.get("test_observation")
        or e.kind == "confidence_proxy"
    ):
        return 0.0
    model = model or SensorModel()
    sensor = model.likelihoods.get(e.source)
    if sensor is None:
        return 0.0
    defect, clean = sensor["flag_given_defect"], sensor["flag_given_clean"]
    lr = (
        math.log(clean / defect)
        if e.polarity == "negative"
        else math.log((1 - clean) / (1 - defect))
    )
    rel = (reliabilities or SOURCE_RELIABILITY).get(e.source, 1.0)
    default = SOURCE_RELIABILITY_HANDTUNED.get(e.source, 1.0)
    return lr * min(1.0, max(0.0, e.weight)) * rel / default


def local_posteriors(
    blocks: list[Block],
    evidence: list[Evidence],
    model: SensorModel | None = None,
    reliabilities: dict[str, float] | None = None,
) -> dict[str, BlockPosterior]:
    """Stage 1 + 2: prior, then naive-Bayes evidence fusion."""
    by_block: dict[str, list[Evidence]] = {b.bid: [] for b in blocks}
    for e in evidence:
        if e.bid in by_block:
            by_block[e.bid].append(e)

    out: dict[str, BlockPosterior] = {}
    priors = {b.bid: prior_correctness(b, model) for b in blocks}
    by_qualname = {b.qualname: b for b in blocks}
    families: dict[str, list[Block]] = {}
    for block in blocks:
        if block.kind == "segment":
            families.setdefault(block.qualname.rsplit("#seg", 1)[0], []).append(block)
    for name, segments in families.items():
        parent = by_qualname.get(name)
        if parent is None:
            continue
        baseline = priors[parent.bid]
        # Partition a function's prior risk instead of adding a fresh independent
        # defect budget for every fragment. Product of family priors = baseline.
        header_weight = max(1, parent.body_lineno - parent.lineno)
        total_weight = header_weight + sum(segment.loc for segment in segments)
        priors[parent.bid] = baseline ** (header_weight / total_weight)
        for segment in segments:
            priors[segment.bid] = baseline ** (segment.loc / total_weight)
    for b in blocks:
        pri = priors[b.bid]
        lo = _logit(pri)
        contribs: list[tuple[str, float]] = []
        direct_negative = 0.0
        direct_positive = 0.0
        # A linter emits correlated messages. Aggregate by source, retaining
        # the strongest finding rather than multiplying duplicate likelihoods.
        aggregated: dict[str, Evidence] = {}
        for e in by_block[b.bid]:
            if e.meta.get("test_observation") or e.polarity == "neutral":
                continue
            previous = aggregated.get(e.source)
            if previous is None or (e.polarity == "negative", e.weight) > (
                previous.polarity == "negative",
                previous.weight,
            ):
                aggregated[e.source] = e
        for e in aggregated.values():
            d = evidence_log_lr(e, model, reliabilities)
            lo += d
            if e.polarity == "negative":
                direct_negative += abs(d)
            else:
                direct_positive += d
            contribs.append((f"{e.source}:{e.kind}", round(d, 3)))
        p = _sigmoid(lo)
        if any(e.kind in {"syntax_error", "compile_error"} for e in by_block[b.bid]):
            p = 0.0001  # observed inability to compile is a hard trust failure
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
    """Compatibility wrapper for cycle-safe graph reliability queries.

    Reverse inference comes from conditioning shared test factors in infer().
    This wrapper evaluates dependency impact without inventing reverse support.
    """
    posterior = condition_defects({bid: 1 - bp.local for bid, bp in post.items()}, [])
    for bid, bp in post.items():
        bp.posterior = round(posterior.trust(dependency_impacts(g, bid))[0], 4)
    return post


def sample_posteriors(
    evidence: list[Evidence],
    blocks: list[Block],
    n_samples: int = 200,
    seed: int = 7,
) -> dict[str, tuple[float, float]]:
    """Monte-Carlo uncertainty over the sensor model.

    The naive-Bayes fusion treats evidence weights as exact. They are not:
    they are estimates of sensor behaviour. We jitter each source reliability multiplicatively
    with log-normal draws and resample the
    fusion `n_samples` times. Returns per block (mean, std) of the local
    posterior — the std is an honest "how sure is the model about itself"
    sensitivity diagnostic. Runtime infer() reports sampling error separately.

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
    *,
    model: SensorModel | None = None,
    reliabilities: dict[str, float] | None = None,
    method: str = "auto",
    seed: int = 7,
) -> dict[str, BlockPosterior]:
    """Condition latent defects on covered tests, then query graph output trust."""
    if not 0 <= damping <= 1:
        raise ValueError("damping must be in [0, 1]")
    model = model or SensorModel()
    if reliabilities is not None and any(
        type(value) not in {float, int} or not math.isfinite(value) or value < 0
        for value in reliabilities.values()
    ):
        raise ValueError("source reliabilities must be finite and nonnegative")
    post = local_posteriors(blocks, evidence, model, reliabilities)
    observed = {}
    for item in evidence:
        observation = item.meta.get("test_observation")
        if observation:
            observed[observation["name"]] = observation
    families: dict[str, list[dict]] = {}
    for row in observed.values():
        families.setdefault(row.get("oracle_family", row["name"]), []).append(row)
    grouped = []
    for name, rows in families.items():
        # Generated repetitions of one return-type contract are a single
        # observation family. If it failed, passing paths cannot exonerate it.
        selected = [r for r in rows if r["failed"]] or rows
        grouped.append(
            {
                **selected[0],
                "name": name,
                "blocks": sorted({b for r in selected for b in r["blocks"]}),
            }
        )
    factors = [
        TestFactor(
            row["name"],
            tuple(row["blocks"]),
            row["failed"],
            test_sensitivity(model, row.get("oracle_scope", "user")),
            model.likelihoods["exec"]["flag_given_clean"],
        )
        for row in grouped
    ]
    posterior = condition_defects(
        {bid: max(1e-6, min(1 - 1e-6, 1 - bp.local)) for bid, bp in post.items()},
        factors,
        method=method,
        seed=seed,
    )
    g.graph["inference"] = {
        "observed_tests": len(observed),
        "observation_families": len(grouped),
        **posterior.diagnostics,
        "calibrated": model.calibrated,
        "probability_scope": "specified behavior and observed test evidence",
        "uncertainty_kind": "Monte Carlo standard error; zero for exact inference",
        "partial_oracle_sensitivity_scales": dict(ORACLE_SENSITIVITY_SCALE),
    }
    # Retain the joint model for optional next-test decisions; reports export
    # only the serializable diagnostics, not its arrays.
    g.graph["defect_posterior"] = posterior
    marginal = posterior.marginals
    for bid, bp in post.items():
        bp.local = round(1 - marginal[bid], 4)
        trust, se = posterior.trust(dependency_impacts(g, bid))
        bp.posterior = round(trust, 4)
        bp.culpability = round(marginal[bid], 4)
        bp.inherited = round(max(0.0, bp.local - trust), 4)
        bp.uncertainty = round(se, 6)
        p = min(1 - 1e-12, max(1e-12, trust))
        bp.entropy = round(-p * math.log2(p) - (1 - p) * math.log2(1 - p), 4)
        bp.importance = round((importance or {}).get(bid, 0.0), 4)
        own = [e for e in evidence if e.bid == bid]
        statuses = [
            e.meta["execution_status"] for e in own if "execution_status" in e.meta
        ]
        bp.execution_status = (
            "tested"
            if any(e.meta.get("test_observation") for e in own)
            else statuses[0]
            if statuses
            else "untested"
        )
        bp.inference_method = next(
            c.method for c in posterior.components if bid in c.nodes
        )
        weight = 0.55 + 0.45 * bp.importance if importance else 1.0
        bp.risk = round(
            (0.70 * bp.culpability + 0.20 * bp.inherited + 0.10 * bp.entropy) * weight,
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
    candidates = {bid for bid, bp in post.items() if bp.culpability >= threshold}

    # Collapse recursion before selecting roots. Otherwise a cycle of suspects
    # suppresses every member and yields an empty repair recommendation.
    condensed = nx.condensation(graph)
    mapping = condensed.graph["mapping"]
    suspect_components = {mapping[bid] for bid in candidates if bid in mapping}
    root_components = {
        component
        for component in suspect_components
        if not (nx.ancestors(condensed, component) & suspect_components)
    }
    roots = [bid for bid in candidates if mapping.get(bid) in root_components]

    return sorted(roots, key=lambda b: -post[b].culpability)


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
