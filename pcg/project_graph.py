"""Project-level source, import/call, dependency and control-flow graphs."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

import networkx as nx

from .blocks import Block, extract_blocks, line_to_block
from .graph import build_graph


@dataclass
class ProjectGraph:
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    control_flow: nx.DiGraph = field(default_factory=nx.DiGraph)
    blocks: list[Block] = field(default_factory=list)
    modules: dict[str, str] = field(default_factory=dict)
    by_file: dict[str, list[Block]] = field(default_factory=dict)
    owners: dict[tuple[str, int], str] = field(default_factory=dict)
    unresolved: list[dict] = field(default_factory=list)


def module_name(path: str) -> str:
    name = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
    return (
        name.removesuffix(".__init__")
        if name.endswith(".__init__")
        else ("" if name == "__init__" else name)
    )


def control_flow(source: str, file: str) -> nx.DiGraph:
    """Explicit loops, breaks, continues and conservative exception/finally edges."""
    graph: nx.DiGraph = nx.DiGraph()
    tree = ast.parse(source)

    def function(fn: ast.FunctionDef | ast.AsyncFunctionDef, scope: str) -> None:
        end = f"{file}:{scope}:exit"
        graph.add_node(end, file=file, line=fn.lineno, scope=scope, kind="exit")

        def edge(a: str, b: str, kind: str) -> None:
            previous = graph.get_edge_data(a, b, {}).get("kinds", [])
            graph.add_edge(a, b, kinds=sorted(set(previous) | {kind}))

        def wire(
            body: list[ast.stmt],
            following: str,
            brk: str | None = None,
            cont: str | None = None,
            handlers: tuple[str, ...] = (),
            final: str | None = None,
        ) -> str:
            cursor = following
            for stmt in reversed(body):
                key = f"{file}:{scope}:{stmt.lineno}"
                graph.add_node(
                    key,
                    file=file,
                    line=stmt.lineno,
                    scope=scope,
                    kind=type(stmt).__name__,
                )
                if isinstance(stmt, ast.If):
                    edge(
                        key, wire(stmt.body, cursor, brk, cont, handlers, final), "true"
                    )
                    edge(
                        key,
                        wire(stmt.orelse, cursor, brk, cont, handlers, final),
                        "false",
                    )
                elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                    exhausted = wire(stmt.orelse, cursor, brk, cont, handlers, final)
                    edge(
                        key,
                        wire(stmt.body, key, cursor, key, handlers, final),
                        "iterate",
                    )
                    edge(key, exhausted, "exhausted")
                elif isinstance(stmt, ast.Break):
                    edge(key, final or brk or end, "break")
                elif isinstance(stmt, ast.Continue):
                    edge(key, final or cont or end, "continue")
                elif isinstance(stmt, (ast.Return, ast.Raise)):
                    destinations = (
                        handlers
                        if isinstance(stmt, ast.Raise) and handlers
                        else (final or end,)
                    )
                    for destination in destinations:
                        edge(
                            key,
                            destination,
                            "raise" if isinstance(stmt, ast.Raise) else "return",
                        )
                elif isinstance(stmt, ast.Try):
                    final_entry = (
                        wire(stmt.finalbody, cursor, brk, cont, handlers, final)
                        if stmt.finalbody
                        else final
                    )
                    normal = final_entry or cursor
                    handler_keys = []
                    for handler in stmt.handlers:
                        entry = f"{file}:{scope}:except:{handler.lineno}"
                        graph.add_node(
                            entry,
                            file=file,
                            line=handler.lineno,
                            scope=scope,
                            kind="exception-handler",
                        )
                        edge(
                            entry,
                            wire(
                                handler.body, normal, brk, cont, handlers, final_entry
                            ),
                            "handle",
                        )
                        handler_keys.append(entry)
                    after_try = wire(
                        stmt.orelse, normal, brk, cont, handlers, final_entry
                    )
                    edge(
                        key,
                        wire(
                            stmt.body,
                            after_try,
                            brk,
                            cont,
                            tuple(handler_keys) or handlers,
                            final_entry,
                        ),
                        "try",
                    )
                    if stmt.finalbody:
                        # A shared finally body can resume several continuations.
                        # This is a conservative may-flow graph, not a path proof.
                        for final_stmt in stmt.finalbody[-1:]:
                            tail = f"{file}:{scope}:{final_stmt.lineno}"
                            edge(tail, end, "possible-finally-return")
                elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                    edge(
                        key,
                        wire(stmt.body, cursor, brk, cont, handlers, final),
                        "context",
                    )
                elif isinstance(stmt, ast.Match):
                    for case in stmt.cases:
                        edge(
                            key,
                            wire(case.body, cursor, brk, cont, handlers, final),
                            "case",
                        )
                    edge(key, cursor, "no-match")
                else:
                    edge(key, cursor, "next")
                for handler_key in handlers:
                    edge(key, handler_key, "possible-exception")
                cursor = key
            return cursor

        start = f"{file}:{scope}:entry"
        graph.add_node(start, file=file, line=fn.lineno, scope=scope, kind="entry")
        edge(start, wire(fn.body, end), "enter")

    def discover(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}.{node.name}" if prefix else node.name
                function(node, name)
                discover(node.body, name)
            elif isinstance(node, ast.ClassDef):
                discover(node.body, f"{prefix}.{node.name}" if prefix else node.name)

    discover(tree.body)
    return graph


def build_project_graph(sources: dict[str, str]) -> ProjectGraph:
    result = ProjectGraph()
    functions: dict[str, str] = {}
    trees = {}
    for file, source in sources.items():
        module = module_name(file)
        result.modules[file] = module
        try:
            blocks = extract_blocks(source)
            trees[file] = ast.parse(source)
            local = build_graph(blocks, source=source)
            result.control_flow.update(control_flow(source, file))
        except SyntaxError as exc:
            blocks = [
                Block(
                    "syntax-error",
                    "module",
                    "<syntax error>",
                    "<syntax error>",
                    1,
                    source.count("\n") + 1,
                    source,
                )
            ]
            local = build_graph(blocks)
            result.unresolved.append(
                {
                    "file": file,
                    "line": exc.lineno,
                    "reason": "syntax error",
                    "message": exc.msg,
                }
            )
        mapping = {b.bid: f"{file}::{b.bid}" for b in blocks}
        result.graph.update(nx.relabel_nodes(local, mapping))
        for block in blocks:
            block.bid = mapping[block.bid]
            if block.kind in {"function", "method"}:
                functions[f"{module}.{block.qualname}"] = block.bid
            block.qualname = f"{file}::{block.qualname}"
        result.blocks.extend(blocks)
        result.by_file[file] = blocks
        result.owners.update(
            {(file, line): bid for line, bid in line_to_block(blocks).items()}
        )
    for file, tree in trees.items():
        module = result.modules[file]
        package = module if file.endswith("__init__.py") else module.rpartition(".")[0]
        aliases = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for name in node.names:
                    aliases[name.asname or name.name.split(".")[0]] = (
                        name.name if name.asname else name.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom):
                parts = package.split(".") if package else []
                prefix = (
                    ".".join(parts[: len(parts) - node.level + 1]) if node.level else ""
                )
                imported = ".".join(p for p in [prefix, node.module or ""] if p)
                for name in node.names:
                    if name.name == "*":
                        result.unresolved.append(
                            {
                                "file": file,
                                "line": node.lineno,
                                "reason": "wildcard import",
                            }
                        )
                    else:
                        aliases[name.asname or name.name] = f"{imported}.{name.name}"
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            call_name = ast.unparse(call.func)
            first, dot, rest = call_name.partition(".")
            resolved = aliases.get(first)
            if not resolved:
                continue
            target = functions.get(resolved + (dot + rest if dot else ""))
            owner = result.owners.get((file, call.lineno))
            if target and owner:
                result.graph.add_edge(
                    target,
                    owner,
                    kind="calls",
                    strength=0.85,
                    relations=["cross-module-return"],
                    kinds=["calls"],
                )
            else:
                result.unresolved.append(
                    {
                        "file": file,
                        "line": call.lineno,
                        "call": call_name,
                        "reason": "external or unresolved import/call",
                    }
                )
    return result
