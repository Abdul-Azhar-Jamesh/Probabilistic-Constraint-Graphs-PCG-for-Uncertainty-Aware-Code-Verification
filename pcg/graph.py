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
    "control": 0.55,
    "contains": 1.0,
}

# Global edge strengths used in propagation. They can be replaced by fitted
# values while preserving the same graph topology and dependency semantics.
LEARNED_EDGE_STRENGTH = dict(EDGE_STRENGTH)


def build_graph(
    blocks: list[Block],
    strengths: dict[str, float] | None = None,
    *,
    source: str | None = None,
) -> nx.DiGraph[str]:
    """Build the constraint graph.

    Edge direction convention: an edge ``B -> A`` means "B supports A", i.e.
    evidence flows from the depended-upon block toward its dependents. This
    This is a code dependency graph, possibly cyclic. The separate latent-
    defect Bayesian model conditions on test outcomes and queries this graph.
    """
    # Nodes are block ids, so the graph is keyed by str.
    g: nx.DiGraph[str] = nx.DiGraph()
    edge_strengths = strengths or LEARNED_EDGE_STRENGTH
    by_name: dict[str, str] = {}
    qualified_ids = {b.qualname: b.bid for b in blocks}
    for b in blocks:
        g.add_node(b.bid, block=b)
        if b.kind != "method":
            for d in b.defines:
                by_name[d] = b.bid

    def resolve(name: str, block: Block) -> str | None:
        if name.startswith(("self.", "cls.")) and "." in block.qualname:
            parent = block.qualname.split("#seg", 1)[0].rsplit(".", 1)[0]
            return qualified_ids.get(parent + "." + name.split(".", 1)[1])
        if "." in name:
            return qualified_ids.get(name)
        return by_name.get(name)

    for b in blocks:
        seen: set[tuple[str, str]] = set()
        for callee in b.calls:
            tgt = resolve(callee, b)
            if tgt and tgt != b.bid and (tgt, "calls") not in seen:
                seen.add((tgt, "calls"))
                g.add_edge(
                    tgt,
                    b.bid,
                    kind="calls",
                    detail=f"{b.qualname} calls {callee}",
                    strength=edge_strengths["calls"],
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
                    strength=edge_strengths["dataflow"],
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
                kind="contains",
                detail=f"{parent.qualname} contains {seg.qualname}",
                strength=1.0,
            )
    # Reaching definitions replace arbitrary adjacent-segment sequence links.
    # These edges can identify an earlier assignment as the origin of bad data.
    if source is not None:
        from .blocks import line_to_block
        from .slicing import program_dependence

        dependence = program_dependence(source)
        g.graph["dependence"] = dependence
        owner = line_to_block(blocks)
        block_by_id = {block.bid: block for block in blocks}
        for before, after, data in dependence.graph.edges(data=True):
            dependency, dependent = owner.get(before), owner.get(after)
            if not dependency or not dependent or dependency == dependent:
                continue
            kinds = data["kinds"]
            # Argument-flow edges belong to the diagnostic statement graph, not
            # global intrinsic correctness of a callee across all call contexts.
            if kinds == ["call_argument"]:
                continue
            source_block, target_block = block_by_id[dependency], block_by_id[dependent]
            if (
                source_block.kind in {"function", "method"}
                and target_block.kind == "segment"
                and target_block.qualname.rsplit("#seg", 1)[0] == source_block.qualname
            ):
                # A header supplies parameters; the umbrella also aggregates
                # segment trust. Do not create an artificial containment cycle.
                continue
            kind = (
                "calls"
                if any(k.startswith("call_") for k in kinds)
                else "control"
                if kinds == ["control"]
                else "dataflow"
            )
            strength = edge_strengths.get(kind, EDGE_STRENGTH[kind])
            existing = g.get_edge_data(dependency, dependent, {})
            g.add_edge(
                dependency,
                dependent,
                kind=existing.get("kind", kind),
                strength=max(strength, existing.get("strength", 0)),
                detail=existing.get("detail", f"line {before} supports line {after}"),
                relations=sorted(set(existing.get("relations", [])) | set(kinds)),
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
        pr = nx.pagerank(g.reverse(), alpha=0.85) if g.number_of_edges() else {}
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
