"""Construction of the Probabilistic Constraint Graph (PCG).

Nodes are code blocks. Edges are *constraints*: typed, directed dependencies
that carry correctness pressure between blocks. If block A calls block B, then
A being correct is contingent on B being correct, so evidence against B must
propagate to A.

Edge kinds
----------
calls     A invokes B                     (strong coupling)
dataflow  A reads a name B defines        (strong coupling)
sequence  A executes before B at module   (weak coupling)
"""

from __future__ import annotations

import networkx as nx

from .blocks import Block

# Default coupling strengths. These are intentionally kept as defaults only;
# the learnable calibration layer can overwrite them without changing the graph
# structure itself.
EDGE_STRENGTH = {
    "calls": 0.85,
    "dataflow": 0.70,
    "sequence": 0.25,
}

# Global edge strengths used in propagation. They can be replaced by fitted
# values while preserving the same graph topology and dependency semantics.
LEARNED_EDGE_STRENGTH = dict(EDGE_STRENGTH)


def build_graph(blocks: list[Block]) -> nx.DiGraph[str]:
    """Build the constraint graph.

    Edge direction convention: an edge ``B -> A`` means "B supports A", i.e.
    evidence flows from the depended-upon block toward its dependents. This
    orientation makes the graph a Bayesian network where each node's parents
    are the things it relies on.
    """
    # Nodes are block ids, so the graph is keyed by str.
    g: nx.DiGraph[str] = nx.DiGraph()
    by_name: dict[str, str] = {}
    for b in blocks:
        g.add_node(b.bid, block=b)
        for d in b.defines:
            by_name[d] = b.bid
        # Methods are also reachable by bare name from call sites.
        if b.kind == "method":
            by_name.setdefault(b.name, b.bid)

    for b in blocks:
        seen: set[tuple[str, str]] = set()
        for callee in b.calls:
            tgt = by_name.get(callee)
            if tgt and tgt != b.bid and (tgt, "calls") not in seen:
                seen.add((tgt, "calls"))
                g.add_edge(
                    tgt,
                    b.bid,
                    kind="calls",
                    detail=f"{b.qualname} calls {callee}",
                    strength=LEARNED_EDGE_STRENGTH["calls"],
                )
        for name in b.reads:
            tgt = by_name.get(name)
            if (
                tgt
                and tgt != b.bid
                and (tgt, "calls") not in seen
                and (tgt, "dataflow") not in seen
            ):
                seen.add((tgt, "dataflow"))
                g.add_edge(
                    tgt,
                    b.bid,
                    kind="dataflow",
                    detail=f"{b.qualname} reads {name}",
                    strength=LEARNED_EDGE_STRENGTH["dataflow"],
                )

    # Segment containment: a function depends on each of its segments, so a
    # defect localised to one segment drags down the enclosing function (and
    # from there, its callers) through normal propagation.
    by_qual = {b.qualname: b for b in blocks}
    for seg in blocks:
        if seg.kind != "segment" or "#seg" not in seg.qualname:
            continue
        parent_qual = seg.qualname.rsplit("#seg", 1)[0]
        parent = by_qual.get(parent_qual)
        if parent and parent.bid != seg.bid:
            g.add_edge(
                seg.bid,
                parent.bid,
                kind="calls",
                detail=f"{parent.qualname} contains {seg.qualname}",
                strength=LEARNED_EDGE_STRENGTH["calls"],
            )
    return g


def structural_importance(g: nx.DiGraph, blocks: list[Block]) -> dict[str, float]:
    """How much does the rest of the program lean on each block?

    Combines reachability mass (how many blocks transitively depend on it),
    PageRank on the dependency graph, and intrinsic complexity. Returned
    values are normalised to [0, 1].
    """
    bmap = {b.bid: b for b in blocks}
    n = max(len(blocks), 1)

    # Transitive dependents: descendants in the "supports" orientation.
    reach: dict[str, float] = {}
    for bid in g.nodes:
        try:
            reach[bid] = len(nx.descendants(g, bid)) / n
        except Exception:
            reach[bid] = 0.0

    try:
        pr = nx.pagerank(g, alpha=0.85) if g.number_of_edges() else {}
    except Exception:
        pr = {}
    if pr:
        mx = max(pr.values()) or 1.0
        pr = {k: v / mx for k, v in pr.items()}

    cx_raw = {b.bid: b.cyclomatic + 0.25 * b.loc for b in blocks}
    cx_mx = max(cx_raw.values()) if cx_raw else 1.0
    cx = {k: v / (cx_mx or 1.0) for k, v in cx_raw.items()}

    out: dict[str, float] = {}
    for bid in bmap:
        out[bid] = round(
            0.45 * reach.get(bid, 0.0)
            + 0.30 * pr.get(bid, 0.0)
            + 0.25 * cx.get(bid, 0.0),
            4,
        )
    return out
