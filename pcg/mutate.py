"""Mutation-based dataset generation for evaluating fault localisation.

Hand-labelling buggy code does not scale, and self-authored bugs carry the
author's bias about what a bug looks like. Instead we take *correct* reference
programs and apply mutation operators -- systematic single-point edits that
mimic real defect classes. Ground truth is then exact and free: we know which
block was mutated, because we did it.

This is standard practice in the fault-localisation and mutation-testing
literature (Jia & Harman 2011). The operators below target the defect classes
LLMs actually produce: off-by-one boundaries, flipped comparisons, wrong
initialisers, dropped guards, and swapped operands.

A mutant is only kept if it is *detectable* -- it must still parse, and it must
actually change behaviour on the reference test suite. A mutant that passes all
tests is "equivalent" and is discarded, since no evidence source could
distinguish it and it would be an unfair negative.
"""

from __future__ import annotations

import ast
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field


@dataclass
class Mutant:
    """One mutated program with exact ground truth."""

    case_id: str
    program: str  # reference program name
    operator: str  # which mutation was applied
    source: str  # mutated source
    tests: str
    buggy_block: str  # qualname of the mutated function -- ground truth
    description: str
    lineno: int = 0
    # Blocks that transitively consume the mutated one; filled in later.
    downstream: set[str] = field(default_factory=set)


# --------------------------------------------------------------------------
# Mutation operators
# --------------------------------------------------------------------------
class _MutationVisitor(ast.NodeTransformer):
    """Applies exactly one mutation, at the `target_index`-th opportunity."""

    def __init__(self, operator: str, target_index: int) -> None:
        self.operator = operator
        self.target_index = target_index
        self.seen = 0
        self.applied = False
        self.lineno = 0
        self.detail = ""

    def _should_apply(self) -> bool:
        if self.applied:
            return False
        hit = self.seen == self.target_index
        self.seen += 1
        return hit

    # -- comparison operators ------------------------------------------
    _CMP_FLIP = {
        ast.Lt: (ast.LtE, "< -> <="),
        ast.LtE: (ast.Lt, "<= -> <"),
        ast.Gt: (ast.GtE, "> -> >="),
        ast.GtE: (ast.Gt, ">= -> >"),
        ast.Eq: (ast.NotEq, "== -> !="),
        ast.NotEq: (ast.Eq, "!= -> =="),
    }

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if self.operator != "boundary" or len(node.ops) != 1:
            return node
        op_t = type(node.ops[0])
        if op_t not in self._CMP_FLIP:
            return node
        if not self._should_apply():
            return node
        new_t, detail = self._CMP_FLIP[op_t]
        node.ops = [new_t()]
        self.applied = True
        self.lineno = node.lineno
        self.detail = f"comparison {detail}"
        return node

    # -- arithmetic operators ------------------------------------------
    _BIN_SWAP = {
        ast.Add: (ast.Sub, "+ -> -"),
        ast.Sub: (ast.Add, "- -> +"),
        ast.Mult: (ast.Div, "* -> /"),
        ast.FloorDiv: (ast.Div, "// -> /"),
    }

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if self.operator != "arithmetic":
            return node
        op_t = type(node.op)
        if op_t not in self._BIN_SWAP:
            return node
        if not self._should_apply():
            return node
        new_t, detail = self._BIN_SWAP[op_t]
        node.op = new_t()
        self.applied = True
        self.lineno = node.lineno
        self.detail = f"arithmetic {detail}"
        return node

    # -- off-by-one on integer constants -------------------------------
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.operator != "offbyone":
            return node
        if not isinstance(node.value, int) or isinstance(node.value, bool):
            return node
        if not self._should_apply():
            return node
        old = node.value
        node.value = old + 1
        self.applied = True
        self.lineno = node.lineno
        self.detail = f"constant {old} -> {node.value}"
        return node

    # -- drop a guard clause -------------------------------------------
    def visit_If(self, node: ast.If) -> ast.AST | None:
        self.generic_visit(node)
        if self.operator != "dropguard":
            return node
        # Only drop guards that early-return; those are the safety checks.
        if not (len(node.body) == 1 and isinstance(node.body[0], ast.Return)):
            return node
        if node.orelse:
            return node
        if not self._should_apply():
            return node
        self.applied = True
        self.lineno = node.lineno
        self.detail = "removed early-return guard"
        return None  # delete the statement

    # -- swap operands of a non-commutative call -----------------------
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if self.operator != "swapargs" or len(node.args) < 2:
            return node
        if not self._should_apply():
            return node
        node.args[0], node.args[1] = node.args[1], node.args[0]
        self.applied = True
        self.lineno = node.lineno
        self.detail = "swapped first two arguments"
        return node


OPERATORS = ["boundary", "arithmetic", "offbyone", "dropguard", "swapargs"]


def _enclosing_function(tree: ast.AST, lineno: int) -> str | None:
    """Qualname of the function containing `lineno`."""
    best: tuple[int, str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= lineno <= (node.end_lineno or node.lineno):
                span = (node.end_lineno or node.lineno) - node.lineno
                if best is None or span < best[0]:
                    best = (span, node.name)
    return best[1] if best else None


def generate_mutants(
    program: str, source: str, tests: str, max_per_operator: int = 4
) -> list[Mutant]:
    """Produce all viable single-point mutants of `source`."""
    out: list[Mutant] = []
    for op in OPERATORS:
        for idx in range(max_per_operator):
            try:
                tree = ast.parse(source)
            except SyntaxError:
                return out
            v = _MutationVisitor(op, idx)
            new_tree = v.visit(tree)
            if not v.applied:
                break  # no more opportunities for this operator
            ast.fix_missing_locations(new_tree)
            try:
                mutated = ast.unparse(new_tree)
            except Exception:
                continue
            if mutated.strip() == source.strip():
                continue
            fn = _enclosing_function(ast.parse(source), v.lineno)
            if not fn:
                continue
            out.append(
                Mutant(
                    case_id=f"{program}:{op}:{idx}",
                    program=program,
                    operator=op,
                    source=mutated,
                    tests=tests,
                    buggy_block=fn,
                    description=v.detail,
                    lineno=v.lineno,
                )
            )
    return out


# --------------------------------------------------------------------------
# Viability filtering
# --------------------------------------------------------------------------
def is_detectable(source: str, tests: str, timeout: int = 25) -> bool:
    """Does this mutant actually fail the reference suite?

    Mutants that still pass every test are *equivalent mutants*: no evidence
    source could possibly detect them, so scoring the framework against them
    would measure nothing. Standard practice is to exclude them.
    """
    wd = tempfile.mkdtemp(prefix="mut_")
    try:
        with open(os.path.join(wd, "candidate.py"), "w", encoding="utf-8") as fh:
            fh.write(source)
        with open(os.path.join(wd, "test_candidate.py"), "w", encoding="utf-8") as fh:
            fh.write(tests)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "test_candidate.py", "-q",
             "--no-header", "-p", "no:cacheprovider"],
            capture_output=True, text=True, cwd=wd, timeout=timeout,
        )
        # Non-zero exit == at least one failure == detectable.
        return proc.returncode != 0
    except subprocess.TimeoutExpired:
        return True  # a hang is very much a detectable defect
    except Exception:
        return False
    finally:
        # pytest leaves __pycache__ behind, so rmdir is not enough.
        shutil.rmtree(wd, ignore_errors=True)


def build_corpus(
    programs: dict[str, tuple[str, str]] | None = None,
    max_per_operator: int = 4,
    seed: int = 0,
    verbose: bool = True,
) -> list[Mutant]:
    """Generate and filter a mutant corpus from reference programs."""
    from .corpus import REFERENCE_PROGRAMS

    programs = programs or REFERENCE_PROGRAMS
    random.seed(seed)
    kept: list[Mutant] = []
    n_generated = 0

    for name, (src, tests) in programs.items():
        # Sanity: the reference itself must pass, or ground truth is meaningless.
        if is_detectable(src, tests):
            if verbose:
                print(f"  [skip] reference program '{name}' fails its own tests")
            continue
        muts = generate_mutants(name, src, tests, max_per_operator)
        n_generated += len(muts)
        for m in muts:
            if is_detectable(m.source, m.tests):
                kept.append(m)
        if verbose:
            n_kept = sum(1 for m in kept if m.program == name)
            print(f"  {name:12} {len(muts):3} generated -> {n_kept:3} detectable")

    if verbose:
        print(
            f"\n  corpus: {len(kept)} detectable mutants "
            f"({n_generated - len(kept)} equivalent mutants discarded)"
        )
    return kept


def annotate_downstream(mutants: list[Mutant]) -> list[Mutant]:
    """Fill in which blocks transitively consume each mutated block."""
    import networkx as nx

    from .blocks import extract_blocks
    from .graph import build_graph

    for m in mutants:
        try:
            blocks = extract_blocks(m.source)
        except SyntaxError:
            continue
        g = build_graph(blocks)
        by_name = {b.qualname: b.bid for b in blocks}
        bid = by_name.get(m.buggy_block)
        if not bid:
            continue
        names = {b.bid: b.qualname for b in blocks}
        try:
            m.downstream = {names[d] for d in nx.descendants(g, bid)}
        except Exception:
            m.downstream = set()
    return mutants
