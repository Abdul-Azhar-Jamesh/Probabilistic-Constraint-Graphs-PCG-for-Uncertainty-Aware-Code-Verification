"""Explicit latent-defect model with noisy-OR test likelihoods.

The code graph may contain cycles. The Bayesian model does not: independent
defect roots generate observed test outcomes. Conditioning on a shared test
couples roots (explaining away). Graph reachability defines a separate output
reliability query, evaluated over the *joint* defect posterior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class TestFactor:
    __test__ = False
    name: str
    blocks: tuple[str, ...]
    failed: bool
    sensitivity: float = 0.95
    leak: float = 0.01

    def __post_init__(self):
        if not 0 < self.sensitivity < 1 or not 0 < self.leak < 1:
            raise ValueError(
                "test sensitivity and nuisance failure probability must be in (0, 1)"
            )
        if not self.blocks or len(set(self.blocks)) != len(self.blocks):
            raise ValueError("test factors require distinct covered blocks")


@dataclass
class ComponentPosterior:
    nodes: tuple[str, ...]
    states: np.ndarray
    weights: np.ndarray
    method: str
    rhat: float = 1.0

    def expectation(self, values: np.ndarray) -> float:
        return float(self.weights @ values)


@dataclass
class DefectPosterior:
    components: list[ComponentPosterior]
    factors: tuple[TestFactor, ...]
    diagnostics: dict = field(default_factory=dict)

    @property
    def marginals(self) -> dict[str, float]:
        return {
            node: component.expectation(component.states[:, i])
            for component in self.components
            for i, node in enumerate(component.nodes)
        }

    def trust(self, impacts: dict[str, float]) -> tuple[float, float]:
        """E[prod_i (1-impact_i*D_i)]; never multiply correlated marginals."""
        value = 1.0
        relative_variance = 0.0
        for component in self.components:
            strengths = np.array([impacts.get(node, 0.0) for node in component.nodes])
            values = np.prod(1.0 - component.states * strengths, axis=1)
            mean = component.expectation(values)
            if component.method == "gibbs" and len(values) >= 40:
                # Batch means account for within-chain autocorrelation, approximately.
                batches = np.array_split(values, min(40, len(values) // 10))
                means = np.array([batch.mean() for batch in batches])
                se = float(means.std(ddof=1) / math.sqrt(len(means)))
                relative_variance += (se / max(mean, 1e-12)) ** 2
            value *= mean
        return value, value * math.sqrt(relative_variance)

    def information_gain(
        self, blocks: tuple[str, ...], sensitivity: float = 0.95, leak: float = 0.01
    ) -> float:
        """Mutual information I(D; next test), in bits, from the joint posterior."""
        TestFactor("candidate", blocks, False, sensitivity, leak)
        known = set(self.marginals)
        if not set(blocks) <= known:
            raise ValueError("candidate test refers to unknown blocks")
        # Conditional pass likelihood = (1-leak) prod_i (1-sensitivity*D_i).
        # Independent posterior components combine by an outer product; cap
        # work with seeded posterior draws when the exact Cartesian product is large.
        rng = np.random.default_rng(7)
        q = np.full(4096, 1.0 - leak)
        for component in self.components:
            positions = [i for i, n in enumerate(component.nodes) if n in blocks]
            if positions:
                indices = rng.choice(
                    len(component.weights), size=len(q), p=component.weights
                )
                q *= np.prod(
                    1.0 - sensitivity * component.states[indices][:, positions], axis=1
                )

        def entropy(p):
            return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

        q = np.clip(q, 1e-12, 1 - 1e-12)
        return max(0.0, float(entropy(q.mean()) - entropy(q).mean()))


def _log_joint(
    states: np.ndarray, priors: np.ndarray, factors: list[tuple[list[int], TestFactor]]
) -> np.ndarray:
    logp = states @ np.log(priors) + (1 - states) @ np.log1p(-priors)
    for positions, factor in factors:
        p_pass = (1 - factor.leak) * np.prod(
            1 - factor.sensitivity * states[:, positions], axis=1
        )
        logp += np.log1p(-p_pass) if factor.failed else np.log(p_pass)
    return logp


def condition_defects(
    priors: dict[str, float],
    factors: list[TestFactor],
    *,
    method: str = "auto",
    exact_limit: int = 14,
    draws: int = 1000,
    burn_in: int = 300,
    chains: int = 4,
    seed: int = 7,
) -> DefectPosterior:
    """Exact enumeration for small components; seeded multi-chain Gibbs otherwise."""
    if method not in {"auto", "exact", "gibbs"}:
        raise ValueError("inference method must be auto, exact, or gibbs")
    if not 1 <= exact_limit <= 20 or draws < 20 or burn_in < 0 or chains < 2:
        raise ValueError("invalid inference limits")
    if any(not math.isfinite(p) or not 0 < p < 1 for p in priors.values()):
        raise ValueError("defect priors must be finite and in (0, 1)")
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(priors)
    # Collapse repeated identical observations to avoid counting duplicates as independent tests.
    unique: dict[tuple, TestFactor] = {}
    for factor in factors:
        if not set(factor.blocks) <= priors.keys():
            raise ValueError("test factor refers to unknown blocks")
        unique.setdefault(
            (
                tuple(sorted(factor.blocks)),
                factor.failed,
                factor.sensitivity,
                factor.leak,
            ),
            factor,
        )
    factors = list(unique.values())
    for factor in factors:
        graph.add_edges_from((factor.blocks[0], node) for node in factor.blocks[1:])
    components = []
    rng = np.random.default_rng(seed)
    for members in nx.connected_components(graph):
        nodes = tuple(sorted(members))
        probabilities = np.array([priors[node] for node in nodes])
        positions = {node: i for i, node in enumerate(nodes)}
        relevant = [
            ([positions[node] for node in factor.blocks], factor)
            for factor in factors
            if factor.blocks[0] in members
        ]
        use_exact = method != "gibbs" and len(nodes) <= exact_limit
        if method == "exact" and not use_exact:
            raise ValueError("component exceeds exact enumeration limit")
        if use_exact:
            ids = np.arange(1 << len(nodes), dtype=np.uint64)
            states = (
                (ids[:, None] >> np.arange(len(nodes), dtype=np.uint64)) & 1
            ).astype(float)
            logp = _log_joint(states, probabilities, relevant)
            weights = np.exp(logp - logp.max())
            weights /= weights.sum()
            components.append(ComponentPosterior(nodes, states, weights, "exact"))
        else:
            samples = []
            for _ in range(chains):
                state = rng.integers(0, 2, len(nodes)).astype(float)
                chain_samples = []
                for sweep in range(burn_in + draws):
                    for i in rng.permutation(len(nodes)):
                        pair = np.tile(state, (2, 1))
                        pair[:, i] = [0, 1]
                        adjacent = [
                            (idx, factor) for idx, factor in relevant if i in idx
                        ]
                        logp = _log_joint(pair, probabilities, adjacent)
                        p_one = 1.0 / (
                            1.0 + math.exp(float(np.clip(logp[0] - logp[1], -700, 700)))
                        )
                        state[i] = rng.random() < p_one
                    if sweep >= burn_in:
                        chain_samples.append(state.copy())
                samples.append(np.array(chain_samples))
            chain_array = np.array(samples)
            within = chain_array.var(axis=1, ddof=1).mean(axis=0)
            between = draws * chain_array.mean(axis=1).var(axis=0, ddof=1)
            rhat = np.sqrt(
                ((draws - 1) / draws * within + between / draws)
                / np.maximum(within, 1e-12)
            )
            # Constant chains with differing means indicate failure to mix.
            rhat = np.where(
                (within == 0) & (between > 0), np.inf, np.maximum(1.0, rhat)
            )
            states = chain_array.reshape(-1, len(nodes))
            components.append(
                ComponentPosterior(
                    nodes,
                    states,
                    np.full(len(states), 1 / len(states)),
                    "gibbs",
                    float(rhat.max()),
                )
            )
    max_rhat = max((c.rhat for c in components), default=1.0)
    diagnostics = {
        "model": "latent-defect-noisy-or-v1",
        "components": len(components),
        "methods": sorted({c.method for c in components}),
        "test_factors": len(factors),
        "max_rhat": max_rhat if math.isfinite(max_rhat) else None,
        "converged": all(c.rhat <= 1.1 for c in components),
        "seed": seed,
        "calibrated": False,
    }
    return DefectPosterior(components, tuple(factors), diagnostics)


def dependency_impacts(graph: nx.DiGraph, target: str) -> dict[str, float]:
    """Strongest dependency-path activation; shared ancestors counted once.

    Conditional trust gates have these probabilities. Positive edge lengths
    ensure mutual recursion cannot amplify impact by repeatedly traversing a cycle.
    """
    reverse = graph.reverse(copy=True)
    for _, _, data in reverse.edges(data=True):
        strength = float(data.get("strength", 0.5))
        if not 0 <= strength <= 1:
            raise ValueError("dependency strengths must be in [0, 1]")
        data["distance"] = -math.log(max(strength, 1e-300))
    distances = nx.single_source_dijkstra_path_length(
        reverse, target, weight="distance"
    )
    return {node: math.exp(-distance) for node, distance in distances.items()}


def rank_next_tests(posterior: DefectPosterior, candidates: list[dict]) -> list[dict]:
    """Rank named, covered candidate tests by expected information per second."""
    ranked = []
    for candidate in candidates:
        cost = float(candidate.get("cost_seconds", 1.0))
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError("test cost must be finite and positive")
        gain = posterior.information_gain(tuple(candidate["blocks"]))
        ranked.append(
            {
                "name": candidate["name"],
                "information_gain_bits": gain,
                "cost_seconds": cost,
                "gain_per_second": gain / cost,
            }
        )
    return sorted(ranked, key=lambda row: -row["gain_per_second"])
