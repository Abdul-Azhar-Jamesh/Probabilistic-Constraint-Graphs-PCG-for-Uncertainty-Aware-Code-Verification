"""Bounded state-dependency graph seeds for existing class sequence probes."""

from __future__ import annotations

import ast
from typing import Any

import networkx as nx


def state_graph(methods: dict[str, ast.FunctionDef]) -> nx.DiGraph:
    """Connect state writers to readers; include local helper summaries.

    Edges describe possible state dependencies, not valid API transitions.
    """
    graph: nx.DiGraph = nx.DiGraph()
    calls = {}
    mutators = {
        "append",
        "extend",
        "update",
        "pop",
        "clear",
        "add",
        "remove",
        "sort",
        "reverse",
        "insert",
        "discard",
    }

    def field(node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr not in methods
        ):
            return node.attr
        if isinstance(node, (ast.Subscript, ast.Attribute)):
            return field(node.value)
        return None

    for name, method in methods.items():
        reads: set[str] = set()
        writes: set[str] = set()
        invoked: set[str] = set()
        # Nested functions/classes are separate scopes, not executed method bodies.
        stack: list[ast.AST] = list(method.body)
        while stack:
            node = stack.pop()
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ):
                continue
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr not in methods
            ):
                (writes if isinstance(node.ctx, (ast.Store, ast.Del)) else reads).add(
                    node.attr
                )
            if isinstance(node, ast.AugAssign):
                target = field(node.target)
                if target:
                    reads.add(target)
                    writes.add(target)
            if isinstance(node, ast.Subscript) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                target = field(node.value)
                if target:
                    writes.add(target)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and node.func.attr in methods
                ):
                    invoked.add(node.func.attr)
                if node.func.attr in mutators:
                    target = field(node.func.value)
                    if target:
                        writes.add(target)
            stack.extend(ast.iter_child_nodes(node))
        graph.add_node(name, reads=reads, writes=writes, line=method.lineno)
        calls[name] = invoked
    for _ in range(min(16, len(methods))):
        changed = False
        for name, invoked in calls.items():
            for helper in invoked:
                for key in ("reads", "writes"):
                    before = len(graph.nodes[name][key])
                    graph.nodes[name][key].update(graph.nodes[helper][key])
                    changed |= before != len(graph.nodes[name][key])
        if not changed:
            break
    for writer in graph:
        for reader in graph:
            shared = graph.nodes[writer]["writes"] & graph.nodes[reader]["reads"]
            if shared:
                graph.add_edge(writer, reader, fields=sorted(shared))
    for name in graph:
        graph.nodes[name]["reads"] = sorted(graph.nodes[name]["reads"])
        graph.nodes[name]["writes"] = sorted(graph.nodes[name]["writes"])
    return graph


def sequence_seeds(
    methods: dict[str, ast.FunctionDef], supported: list[str]
) -> tuple[list[list], dict]:
    """Prioritize long producer-to-consumer chains, with bounded repetition."""
    from .testgen import _type_spec, _values

    full_graph = state_graph(methods)
    graph = full_graph.subgraph(supported)
    paths: list[tuple[str, ...]] = []
    stack: list[tuple[str, ...]] = [(name,) for name in reversed(sorted(graph))]
    explored = 0
    while stack and explored < 256:
        path = stack.pop()
        explored += 1
        if len(path) >= 2:
            paths.append(path)
        if len(path) >= 6:
            continue
        for nxt in reversed(sorted(graph.successors(path[-1]))):
            if nxt not in path or nxt == path[-1] and path.count(nxt) < 4:
                stack.append((*path, nxt))
    variants: dict[str, list[tuple]] = {}
    for name in supported:
        params = methods[name].args.args[1:]
        domains = [_values(_type_spec(p.annotation)) for p in params]
        if any(not d for d in domains):
            continue
        base = tuple(d[0] for d in domains)
        targeted = list(base)
        for i, param in enumerate(params):
            for node in ast.walk(methods[name]):
                if (
                    isinstance(node, ast.Compare)
                    and isinstance(node.left, ast.Name)
                    and node.left.id == param.arg
                ):
                    for other in node.comparators:
                        if (
                            isinstance(other, ast.Constant)
                            and type(other.value) is type(base[i])
                            and isinstance(other.value, (int, str, bool))
                        ):
                            if (
                                isinstance(other.value, str)
                                and len(other.value) > 40
                                or type(other.value) is int
                                and abs(other.value) > 10000
                            ):
                                continue
                            targeted[i] = other.value
        variants[name] = [base, tuple(targeted)]
    seeds = []
    seen = set()
    for path in sorted(set(paths), key=lambda p: (-len(p), p)):
        if any(name not in variants for name in path):
            continue
        for variant in range(2):
            steps = [(name, variants[name][variant]) for name in path]
            key = repr(steps)
            if key not in seen:
                seen.add(key)
                seeds.append(steps)
            if len(seeds) >= 16:
                break
        if len(seeds) >= 16:
            break
    metadata: dict[str, Any] = {
        "nodes": [{"method": n, **d} for n, d in full_graph.nodes(data=True)],
        "edges": [
            {"writer": a, "reader": b, **d} for a, b, d in full_graph.edges(data=True)
        ],
        "seed_sequences": seeds,
        "explored_states": explored,
        "limits": {"walk_states": 256, "sequences": 16, "steps": 6},
        "interpretation": "possible field dependencies; sequences are probes, not inferred valid API contracts",
    }
    return seeds, metadata
