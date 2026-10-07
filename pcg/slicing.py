"""Conservative program dependence and coverage-filtered backward slices.

This is a may-depend analysis, not a symbolic proof or a dynamic value trace.
Branch joins retain both definitions; straight-line reassignment kills old ones.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field

import networkx as nx

from .blocks import Block, line_to_block
from .execution import ExecutionResult


@dataclass
class Dependence:
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    returns: dict[str, set[int]] = field(default_factory=dict)
    scopes: dict[int, str] = field(default_factory=dict)

    def backward(self, seeds: set[int], executed: set[int] | None = None) -> list[int]:
        # Filter by execution before traversal, never bridge an unexecuted branch.
        graph = self.graph
        if executed is not None:
            allowed = {
                n for n, d in graph.nodes(data=True) if set(d["lines"]) & executed
            }
            graph = graph.subgraph(allowed)
        reached = set()
        for seed in seeds:
            if seed in graph:
                reached.add(seed)
                reached.update(nx.ancestors(graph, seed))
        return sorted(reached)


def program_dependence(source: str) -> Dependence:
    tree = ast.parse(source)
    result = Dependence()
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    def discover(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}.{node.name}" if prefix else node.name
                functions[name] = node
                discover(node.body, name)
            elif isinstance(node, ast.ClassDef):
                discover(node.body, f"{prefix}.{node.name}" if prefix else node.name)

    discover(tree.body)
    pending: list[tuple[int, str, str]] = []
    unresolved: set[tuple[int, str]] = set()

    def possible_targets(
        name: str, env: dict[str, set[int]], seen: set[str] | None = None
    ) -> set[str]:
        seen = set() if seen is None else seen
        if name in seen:
            return set()
        seen = seen | {name}
        receiver, dot, method = name.partition(".")
        if receiver not in env:
            return {name}
        targets: set[str] = set()
        for definition in env[receiver]:
            values = (
                result.graph.nodes[definition].get("bindings", {}).get(receiver, [])
            )
            for value in values:
                for target in possible_targets(value, env, seen):
                    targets.add(target + (dot + method if dot else ""))
        return targets

    def edge(a: int, b: int, kind: str) -> None:
        if a != b:
            kinds = set(result.graph.get_edge_data(a, b, {}).get("kinds", []))
            result.graph.add_edge(a, b, kinds=sorted(kinds | {kind}))

    def process(
        body: list[ast.stmt],
        env: dict[str, set[int]],
        scope: str,
        controls: tuple[int, ...] = (),
        loop_revisit: bool = False,
    ) -> dict[str, set[int]]:
        env = {k: set(v) for k, v in env.items()}
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            line = node.lineno
            # Inspect a compound statement's header only; children have their own nodes.
            children = [v for _, v in ast.iter_fields(node) if isinstance(v, ast.AST)]
            if isinstance(node, (ast.If, ast.While)):
                children = [node.test]
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                children = [node.target, node.iter]
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                children = list(node.items)
            elif isinstance(node, ast.Try):
                children = []
            atoms = [a for child in children for a in ast.walk(child)]
            # Assignment targets/values are often lists, rather than direct AST fields.
            if not isinstance(
                node,
                (
                    ast.If,
                    ast.While,
                    ast.For,
                    ast.AsyncFor,
                    ast.With,
                    ast.AsyncWith,
                    ast.Try,
                ),
            ):
                atoms = list(ast.walk(node))
            reads = {
                a.id
                for a in atoms
                if isinstance(a, ast.Name) and isinstance(a.ctx, ast.Load)
            }
            writes = {
                a.id
                for a in atoms
                if isinstance(a, ast.Name) and isinstance(a.ctx, (ast.Store, ast.Del))
            }
            for a in atoms:
                if isinstance(a, ast.AugAssign):
                    reads.update(
                        n.id for n in ast.walk(a.target) if isinstance(n, ast.Name)
                    )
                if isinstance(a, (ast.Subscript, ast.Attribute)) and isinstance(
                    a.ctx, ast.Store
                ):
                    writes.update(
                        n.id for n in ast.walk(a.value) if isinstance(n, ast.Name)
                    )
                if (
                    isinstance(a, ast.Call)
                    and isinstance(a.func, ast.Attribute)
                    and a.func.attr
                    in {
                        "append",
                        "extend",
                        "update",
                        "pop",
                        "clear",
                        "add",
                        "remove",
                        "sort",
                        "reverse",
                    }
                ):
                    writes.update(
                        n.id for n in ast.walk(a.func.value) if isinstance(n, ast.Name)
                    )
                if isinstance(a, ast.Call):
                    call_name = ast.unparse(a.func)
                    targets = (
                        {call_name}
                        if call_name.startswith(("self.", "cls."))
                        else possible_targets(call_name, env)
                    )
                    if not targets:
                        unresolved.add((line, call_name))
                    pending.extend((line, target, scope) for target in targets)
            end = node.end_lineno or line
            header_end = min(
                (
                    s.lineno - 1
                    for s in getattr(node, "body", [])
                    if isinstance(s, ast.stmt)
                ),
                default=end,
            )
            bindings: dict[str, list[str]] = {}
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                if isinstance(value, ast.Name):
                    targets = possible_targets(value.id, env)
                    bindings = {name: sorted(targets) for name in writes}
                elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                    bindings = {name: [value.func.id] for name in writes}
            result.graph.add_node(
                line,
                lines=list(range(line, max(line, header_end) + 1)),
                reads=sorted(reads),
                writes=sorted(writes),
                scope=scope,
                bindings=bindings,
                side_effect=any(
                    isinstance(a, (ast.Subscript, ast.Attribute))
                    and isinstance(a.ctx, ast.Store)
                    or isinstance(a, ast.Call)
                    and isinstance(a.func, ast.Attribute)
                    and a.func.attr
                    in {
                        "append",
                        "extend",
                        "update",
                        "pop",
                        "clear",
                        "add",
                        "remove",
                        "sort",
                        "reverse",
                    }
                    for a in atoms
                ),
            )
            result.scopes[line] = scope
            for name in reads:
                for definition in env.get(name, set()):
                    edge(definition, line, "dataflow")
            for control in controls:
                edge(control, line, "control")
            if isinstance(node, ast.Return):
                result.returns.setdefault(scope, set()).add(line)
            for name in writes:
                env[name] = {line}
            branches: list[list[ast.stmt]] = []
            if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While)):
                branches = [node.body, node.orelse]
            elif isinstance(node, ast.Try):
                branches = [node.body, *[h.body for h in node.handlers], node.orelse]
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                env = process(node.body, env, scope, controls + (line,), loop_revisit)
            if branches:
                joined: dict[str, set[int]] = {}
                for branch in branches:
                    for name, definitions in process(
                        branch, env, scope, controls + (line,), loop_revisit
                    ).items():
                        joined.setdefault(name, set()).update(definitions)
                if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                    # Backedges account for values carried from prior iterations.
                    for name in reads:
                        for definition in joined.get(name, set()):
                            edge(definition, line, "loop_carried")
                    for name, definitions in env.items():
                        joined.setdefault(name, set()).update(definitions)
                    # A second conservative pass connects reads to definitions
                    # carried from an earlier loop iteration (not just the header).
                    if not loop_revisit:
                        for name, definitions in process(
                            node.body, joined, scope, controls + (line,), True
                        ).items():
                            joined.setdefault(name, set()).update(definitions)
                env = joined
                if isinstance(node, ast.Try):
                    env = process(node.finalbody, env, scope, controls, loop_revisit)
        return env

    globals_env = process(tree.body, {}, "<module>")
    for name, fn in functions.items():
        params = [
            p.arg for p in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)
        ]
        if fn.args.vararg:
            params.append(fn.args.vararg.arg)
        if fn.args.kwarg:
            params.append(fn.args.kwarg.arg)
        result.graph.add_node(
            fn.lineno,
            lines=[fn.lineno],
            reads=[],
            writes=params,
            scope=name,
            parameter_header=fn.body[0].lineno > fn.lineno,
        )
        result.scopes[fn.lineno] = name
        local_names = {
            n.id
            for s in fn.body
            for n in ast.walk(s)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        env = {k: v for k, v in globals_env.items() if k not in local_names}
        env.update({p: {fn.lineno} for p in params})
        process(fn.body, env, name)
    for line, callee, scope in pending:
        if callee.startswith(("self.", "cls.")):
            callee = scope.rsplit(".", 1)[0] + "." + callee.split(".", 1)[1]
        if callee not in functions:
            lexical = scope + "." + callee
            if lexical in functions:
                callee = lexical
        if callee in functions:
            for returned in result.returns.get(callee, set()):
                edge(returned, line, "call_return")
            # Caller expressions are possible origins of bad arguments.
            edge(line, functions[callee].lineno, "call_argument")
        else:
            unresolved.add((line, callee))
    result.graph.graph["unresolved_calls"] = [
        {"line": line, "call": call} for line, call in sorted(unresolved)
    ]
    return result


def failure_slices(
    dependence: Dependence, execution: ExecutionResult | None, blocks: list[Block]
) -> list[dict]:
    if execution is None or not execution.valid:
        return []
    owner = line_to_block(blocks)
    rows = []
    for test in execution.tests:
        if test.outcome != "failed":
            continue
        executed = set(test.lines) | set(test.initialization_lines)
        # Runtime exceptions seed the innermost recorded frame. Assertions outside
        # candidate.py have no such frame: consider executed return statements.
        frame = test.frames[-1:] if test.frames else []
        seeds = {
            n
            for n, d in dependence.graph.nodes(data=True)
            if any(ln in d["lines"] for ln in frame)
        }
        if not seeds:
            seeds = {
                n
                for values in dependence.returns.values()
                for n in values
                if n in executed
            }
            seeds.update(
                n
                for n, values in dependence.graph.nodes(data=True)
                if values.get("side_effect") and set(values["lines"]) & executed
            )
        if not seeds:
            seeds = {
                n
                for n, values in dependence.graph.nodes(data=True)
                if values["writes"]
                and values["scope"] != "<module>"
                and set(values["lines"]) & executed
            }
        sliced = dependence.backward(seeds, executed)
        relevant = [
            t
            for t in execution.tests
            if (t.oracle == "probe") == (test.oracle == "probe")
        ]
        failed_tests = [t for t in relevant if t.outcome == "failed"]
        passed_tests = [t for t in relevant if t.outcome == "passed"]
        ranking = []
        for line in sliced:
            lines = set(dependence.graph.nodes[line]["lines"])
            failed_hits = sum(
                bool(lines & (set(t.lines) | set(t.initialization_lines)))
                for t in failed_tests
            )
            passed_hits = sum(
                bool(lines & (set(t.lines) | set(t.initialization_lines)))
                for t in passed_tests
            )
            divisor = math.sqrt(len(failed_tests) * (failed_hits + passed_hits))
            ranking.append(
                {
                    "line": line,
                    "block": owner.get(line),
                    "ochiai": round(failed_hits / divisor, 4) if divisor else 0.0,
                    "role": "input boundary"
                    if dependence.graph.nodes[line].get("parameter_header")
                    else "symptom"
                    if line in seeds
                    else "possible earlier origin",
                    "writes": dependence.graph.nodes[line]["writes"],
                }
            )
        ranking.sort(
            key=lambda row: (
                -row["ochiai"],
                {"possible earlier origin": 0, "symptom": 1, "input boundary": 2}[
                    row["role"]
                ],
                row["line"],
            )
        )
        rows.append(
            {
                "test": test.nodeid,
                "oracle": test.oracle,
                "symptom_lines": sorted(seeds),
                "candidate_cause_lines": sliced,
                "candidate_blocks": sorted({owner[n] for n in sliced if n in owner}),
                "cause_ranking": ranking,
                "initialization_origin_lines": sorted(
                    set(sliced) & set(test.initialization_lines)
                ),
                "interpretation": "executed static may-depend slice; candidate causes, not proven causes",
            }
        )
    return rows
