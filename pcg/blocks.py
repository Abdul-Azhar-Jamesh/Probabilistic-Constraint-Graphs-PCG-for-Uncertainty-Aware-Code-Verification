"""AST-based decomposition of a Python module into verifiable code blocks.

A *block* is the unit at which we estimate correctness. We use function and
method bodies as the primary granularity, plus module-level statement groups,
because that is the granularity at which both evidence (test outcomes, linter
messages) and repair (regenerate one function) naturally operate.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field


@dataclass
class Block:
    """One verifiable region of source code."""

    bid: str
    kind: str  # "function" | "method" | "class" | "module"
    name: str
    qualname: str
    lineno: int
    end_lineno: int
    source: str
    # Names this block reads that are defined elsewhere in the module.
    reads: set[str] = field(default_factory=set)
    # Names this block defines and exports to the module namespace.
    defines: set[str] = field(default_factory=set)
    # Functions this block calls.
    calls: set[str] = field(default_factory=set)
    # Structural features used for importance and prior estimation.
    n_stmts: int = 0
    depth: int = 0
    n_branches: int = 0
    n_loops: int = 0
    has_return: bool = False
    n_params: int = 0

    @property
    def loc(self) -> int:
        return self.end_lineno - self.lineno + 1

    @property
    def cyclomatic(self) -> int:
        """McCabe complexity: decision points + 1."""
        return self.n_branches + self.n_loops + 1


def _end_lineno(node: ast.stmt) -> int:
    """Last source line of `node`, falling back to its first line.

    The parser always populates `end_lineno`, but it stays optional on nodes
    built by hand — which :mod:`pcg.mutate` and :mod:`pcg.repair` both do when
    they rewrite a tree. Reading it directly would raise `TypeError` on the
    arithmetic below, so every read goes through here.
    """
    return node.lineno if node.end_lineno is None else node.end_lineno


class _BodyAnalyzer(ast.NodeVisitor):
    """Collects reads/writes/calls and structural stats within one block body.

    Nested function definitions are *not* descended into as separate blocks
    here; they are folded into the parent so every line belongs to exactly one
    block. Their free variables still count as reads of the parent.
    """

    def __init__(self, param_names: set[str]) -> None:
        self.reads: set[str] = set()
        self.calls: set[str] = set()
        self.local: set[str] = set(param_names)
        self.n_stmts = 0
        self.n_branches = 0
        self.n_loops = 0
        self.has_return = False
        self.max_depth = 0
        self._depth = 0

    # -- helpers ---------------------------------------------------------
    def _descend(self, node: ast.AST) -> None:
        self._depth += 1
        self.max_depth = max(self.max_depth, self._depth)
        self.generic_visit(node)
        self._depth -= 1

    # -- visitors --------------------------------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.local.add(node.id)
        elif isinstance(node.ctx, (ast.Load, ast.Del)):
            if node.id not in self.local:
                self.reads.add(node.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        if isinstance(fn, ast.Name):
            self.calls.add(fn.id)
            if fn.id not in self.local:
                self.reads.add(fn.id)
        elif isinstance(fn, ast.Attribute):
            self.calls.add(fn.attr)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.n_branches += 1
        self._descend(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.n_branches += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.n_loops += 1
        self._descend(node)

    def visit_While(self, node: ast.While) -> None:
        self.n_loops += 1
        self._descend(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.n_branches += len(node.handlers)
        self._descend(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # Short-circuit operators are decision points too.
        self.n_branches += len(node.values) - 1
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.has_return = True
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.n_loops += 1
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.stmt):
            self.n_stmts += 1
        super().generic_visit(node)


def _analyze(body: list[ast.stmt], params: set[str]) -> _BodyAnalyzer:
    an = _BodyAnalyzer(params)
    for stmt in body:
        an.visit(stmt)
    return an


def _params_of(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    a = node.args
    names = [p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return names


def extract_blocks(source: str, filename: str = "<module>") -> list[Block]:
    """Decompose `source` into blocks. Raises SyntaxError if it does not parse."""
    tree = ast.parse(source, filename=filename)
    lines = source.splitlines()
    blocks: list[Block] = []
    counter = 0

    def new_id() -> str:
        nonlocal counter
        counter += 1
        return f"B{counter:02d}"

    def src_of(node: ast.stmt) -> str:
        return "\n".join(lines[node.lineno - 1 : _end_lineno(node)])

    def add_function(
        node: ast.FunctionDef | ast.AsyncFunctionDef, prefix: str, kind: str
    ) -> None:
        params = _params_of(node)
        an = _analyze(node.body, set(params))
        qual = f"{prefix}.{node.name}" if prefix else node.name
        # Decorators are part of the block's surface area.
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        end = _end_lineno(node)
        blocks.append(
            Block(
                bid=new_id(),
                kind=kind,
                name=node.name,
                qualname=qual,
                lineno=start,
                end_lineno=end,
                source="\n".join(lines[start - 1 : end]),
                reads=an.reads,
                defines={node.name},
                calls=an.calls,
                n_stmts=an.n_stmts,
                depth=an.max_depth,
                n_branches=an.n_branches,
                n_loops=an.n_loops,
                has_return=an.has_return,
                n_params=len(params),
            )
        )
        # Intra-procedural splitting: a large function additionally gets
        # segment blocks so blame can localise inside it. The parent block is
        # kept as the umbrella that owns `defines`/graph edges; segments own
        # their lines (they win in line_to_block because they are shorter).
        for i, (s0, s1) in enumerate(_split_function_segments(node, lines), 1):
            seg_src = textwrap.dedent("\n".join(lines[s0 - 1 : s1]))
            seg_an = _analyze(ast.parse(seg_src).body, set(params))
            blocks.append(
                Block(
                    bid=new_id(),
                    kind="segment",
                    name=f"{node.name}#seg{i}",
                    qualname=f"{qual}#seg{i}",
                    lineno=s0,
                    end_lineno=s1,
                    source=seg_src,
                    reads=set(seg_an.reads) - set(params),
                    calls=seg_an.calls,
                    n_stmts=seg_an.n_stmts,
                    depth=seg_an.max_depth,
                    n_branches=seg_an.n_branches,
                    n_loops=seg_an.n_loops,
                    has_return=seg_an.has_return,
                    n_params=len(params),
                )
            )

    module_level: list[ast.stmt] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_function(node, "", "function")
        elif isinstance(node, ast.ClassDef):
            # The class becomes a block for its own body statements; each
            # method becomes its own block.
            methods = [
                n
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            non_methods = [n for n in node.body if n not in methods]
            if non_methods:
                an = _analyze(non_methods, set())
                blocks.append(
                    Block(
                        bid=new_id(),
                        kind="class",
                        name=node.name,
                        qualname=node.name,
                        lineno=node.lineno,
                        end_lineno=max(_end_lineno(n) for n in non_methods),
                        source=src_of(node),
                        reads=an.reads,
                        defines={node.name},
                        calls=an.calls,
                        n_stmts=an.n_stmts,
                        depth=an.max_depth,
                        n_branches=an.n_branches,
                        n_loops=an.n_loops,
                    )
                )
            for m in methods:
                add_function(m, node.name, "method")
        else:
            module_level.append(node)

    if module_level:
        an = _analyze(module_level, set())
        blocks.append(
            Block(
                bid=new_id(),
                kind="module",
                name="<module>",
                qualname="<module>",
                lineno=min(n.lineno for n in module_level),
                end_lineno=max(_end_lineno(n) for n in module_level),
                source="\n".join(
                    "\n".join(lines[n.lineno - 1 : _end_lineno(n)])
                    for n in module_level
                ),
                reads=an.reads,
                defines=an.local,
                calls=an.calls,
                n_stmts=an.n_stmts,
                depth=an.max_depth,
                n_branches=an.n_branches,
                n_loops=an.n_loops,
            )
        )

    blocks.sort(key=lambda b: b.lineno)
    return blocks


# A function at or above this many lines is split into segments so that
# evidence (traceback blame, linter messages) can localise *within* it.
SEGMENT_THRESHOLD_LOC = 25

# Target segment size in lines. Segments are cut on top-level statement
# boundaries closest to this size, never splitting a statement.
SEGMENT_TARGET_LOC = 10


def _split_function_segments(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
) -> list[tuple[int, int]]:
    """Cut a large function body into contiguous line ranges (segments).

    Segments follow top-level statement boundaries of the function body,
    aiming for ~SEGMENT_TARGET_LOC lines each. Compound statements (if/for/
    while/try and their bodies) are kept whole — splitting inside a branch
    would produce fragments with no independent meaning for repair.
    """
    stmts = [s for s in node.body if s.lineno <= _end_lineno(s)]
    if not stmts:
        return []
    total = _end_lineno(node) - node.lineno + 1
    if total < SEGMENT_THRESHOLD_LOC or len(stmts) < 2:
        return []

    segments: list[tuple[int, int]] = []
    seg_start = stmts[0].lineno
    seg_end = _end_lineno(stmts[0])
    for s in stmts[1:]:
        if _end_lineno(s) - seg_start + 1 >= SEGMENT_TARGET_LOC:
            segments.append((seg_start, seg_end))
            seg_start = s.lineno
        seg_end = _end_lineno(s)
    segments.append((seg_start, seg_end))
    # A "split" yielding one segment is no split at all.
    return segments if len(segments) > 1 else []


def line_to_block(blocks: list[Block]) -> dict[int, str]:
    """Map each source line number to the id of the block owning it.

    The most specific owner wins: segments beat functions, functions beat
    classes. Implemented as "shortest spanning block wins": blocks are
    applied longest-first so shorter (more specific) blocks overwrite them.
    """
    owner: dict[int, str] = {}
    for b in sorted(blocks, key=lambda x: x.loc, reverse=True):
        for ln in range(b.lineno, b.end_lineno + 1):
            owner[ln] = b.bid
    return owner
