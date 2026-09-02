"""Targeted repair: from "which block is wrong" to "here is the fix".

The framework previously stopped at localisation -- it ranked blocks by
culpability and left the fix to the human. This module closes the loop.

Why not just ask an LLM to rewrite the block?
--------------------------------------------
Two reasons. First, practical: generation is autoregressive, so on the CPU-only
hardware this project targets, rewriting one function costs minutes where
*scoring* it costs under a second. Second, and more important: an LLM rewrite
discards everything the constraint graph just computed. We know which block is
culpable, which evidence fired, and -- via per-token surprisal -- which *line*
inside the block the model itself found least expected. Throwing that away to
resample the whole function is strictly less informed.

So repair here is a guided search. Three things narrow it:

1. **Which block** -- culpability ranking, not posterior. A block that is merely
   downstream of a defect must not be patched; patching it would mask the real
   fault. This is the own-vs-inherited distinction earning its keep.
2. **Which line** -- peak token surprisal from :mod:`pcg.llm`, when available.
   A flipped comparison shows up as one very-low-probability token.
3. **Which edit** -- the inverse of the mutation operators. Defects that LLMs
   actually emit are a small, enumerable set: off-by-one bounds, flipped
   comparisons, wrong initialisers, missing guards, swapped arguments.

Every candidate patch is then *validated* by running the test suite, so a
proposed fix is only reported if it demonstrably works. Nothing is accepted on
plausibility alone.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .blocks import Block, extract_blocks
from .inference import BlockPosterior


@dataclass
class Patch:
    """One candidate repair."""

    bid: str
    qualname: str
    operator: str  # which repair template produced it
    description: str
    lineno: int
    source: str  # full patched file
    before: str = ""
    after: str = ""
    # Filled in by validation:
    validated: bool = False
    verdict: str = ""  # "full" | "partial"
    tests_passed: int = 0
    tests_failed: int = 0
    posterior_before: float = 0.0
    posterior_after: float = 0.0

    @property
    def delta(self) -> float:
        return round(self.posterior_after - self.posterior_before, 4)


# --------------------------------------------------------------------------
# Repair templates -- inverses of the mutation operators
# --------------------------------------------------------------------------
_CMP_ALTERNATIVES: dict[type[ast.cmpop], list[type[ast.cmpop]]] = {
    ast.Lt: [ast.LtE, ast.Gt],
    ast.LtE: [ast.Lt, ast.GtE],
    ast.Gt: [ast.GtE, ast.Lt],
    ast.GtE: [ast.Gt, ast.LtE],
    ast.Eq: [ast.NotEq],
    ast.NotEq: [ast.Eq],
}
_CMP_SYM: dict[type[ast.cmpop], str] = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.Eq: "==", ast.NotEq: "!=",
}

_BIN_ALTERNATIVES: dict[type[ast.operator], list[type[ast.operator]]] = {
    ast.Add: [ast.Sub],
    ast.Sub: [ast.Add],
    ast.Mult: [ast.FloorDiv, ast.Div],
    ast.Div: [ast.FloorDiv, ast.Mult],
    ast.FloorDiv: [ast.Div],
}
_BIN_SYM: dict[type[ast.operator], str] = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
}


class _Rewriter(ast.NodeTransformer):
    """Applies one specific edit, identified by (kind, occurrence index)."""

    def __init__(self, kind: str, index: int, variant: int, lo: int, hi: int) -> None:
        self.kind, self.index, self.variant = kind, index, variant
        self.lo, self.hi = lo, hi  # line range to restrict edits to
        self.seen = 0
        self.applied = False
        self.lineno = 0
        self.detail = ""
        self.before = ""
        self.after = ""

    def _in_range(self, node: ast.AST) -> bool:
        ln = getattr(node, "lineno", None)
        return ln is not None and self.lo <= ln <= self.hi

    def _hit(self, node: ast.AST) -> bool:
        if self.applied or not self._in_range(node):
            return False
        hit = self.seen == self.index
        self.seen += 1
        return hit

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if self.kind != "compare" or len(node.ops) != 1:
            return node
        op_t = type(node.ops[0])
        alts = _CMP_ALTERNATIVES.get(op_t, [])
        if not alts or not self._hit(node):
            return node
        new_t = alts[self.variant % len(alts)]
        self.before, self.after = _CMP_SYM[op_t], _CMP_SYM[new_t]
        node.ops = [new_t()]
        self.applied = True
        self.lineno = node.lineno
        self.detail = f"comparison {self.before} -> {self.after}"
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if self.kind != "binop":
            return node
        op_t = type(node.op)
        alts = _BIN_ALTERNATIVES.get(op_t, [])
        if not alts or not self._hit(node):
            return node
        new_t = alts[self.variant % len(alts)]
        self.before, self.after = _BIN_SYM[op_t], _BIN_SYM[new_t]
        node.op = new_t()
        self.applied = True
        self.lineno = node.lineno
        self.detail = f"operator {self.before} -> {self.after}"
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.kind != "offbyone":
            return node
        if not isinstance(node.value, int) or isinstance(node.value, bool):
            return node
        if not self._hit(node):
            return node
        delta = (1, -1)[self.variant % 2]
        self.before, self.after = str(node.value), str(node.value + delta)
        node.value = node.value + delta
        self.applied = True
        self.lineno = node.lineno
        self.detail = f"constant {self.before} -> {self.after}"
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """Adjust a slice bound -- the classic off-by-one in list handling."""
        self.generic_visit(node)
        if self.kind != "slice" or not isinstance(node.slice, ast.Slice):
            return node
        sl = node.slice
        target = sl.upper if self.variant % 2 == 0 else sl.lower
        if target is None or not self._hit(node):
            return node
        delta = 1 if self.variant % 4 < 2 else -1
        op = ast.Add() if delta > 0 else ast.Sub()
        new = ast.BinOp(left=target, op=op, right=ast.Constant(value=1))
        if self.variant % 2 == 0:
            sl.upper = new
        else:
            sl.lower = new
        self.applied = True
        self.lineno = node.lineno
        self.before = "slice bound"
        self.after = f"{'+' if delta > 0 else '-'}1"
        self.detail = f"slice bound {self.after}"
        return node

    def _shift(self, expr: ast.expr) -> ast.expr:
        delta = 1 if self.variant % 2 == 0 else -1
        self.before = _unparse(expr)
        self.after = f"({self.before} {'+' if delta > 0 else '-'} 1)"
        return ast.BinOp(
            left=expr,
            op=ast.Add() if delta > 0 else ast.Sub(),
            right=ast.Constant(value=1),
        )


class _ShiftRewriter(_Rewriter):
    """Shift a whole subexpression by +/-1.

    The single most common LLM arithmetic defect is not a wrong *constant* but
    a wrong *expression*: dividing by `len(xs)` where the sample formula needs
    `len(xs) - 1`, or indexing `s[k]` where `k` can reach `len(s)`. Neither is
    reachable by editing a literal, because there is no literal to edit -- so
    the constant-tweaking template that handles textbook off-by-ones misses
    exactly the cases that matter most.

    We target the two positions where such an error is both common and cheaply
    testable: a division denominator, and a subscript index.
    """

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if self.kind != "denominator":
            return node
        if not isinstance(node.op, (ast.Div, ast.FloorDiv)):
            return node
        if isinstance(node.right, ast.Constant):
            return node  # a literal denominator is handled by `offbyone`
        if not self._hit(node):
            return node
        node.right = self._shift(node.right)
        self.applied = True
        self.lineno = node.lineno
        self.detail = f"denominator {self.before} -> {self.after}"
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if self.kind != "index" or isinstance(node.slice, ast.Slice):
            return node
        if isinstance(node.slice, ast.Constant) and not isinstance(
            node.slice.value, int
        ):
            return node
        if not self._hit(node):
            return node
        node.slice = self._shift(node.slice)
        self.applied = True
        self.lineno = node.lineno
        self.detail = f"index {self.before} -> {self.after}"
        return node


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "expr"


class _GuardInserter(ast.NodeTransformer):
    """Insert a defensive early-return guard at the top of a function.

    Missing guards are the single most common LLM defect class -- division by
    zero, indexing an empty sequence -- and they are the one class that cannot
    be fixed by rewriting an existing node, because the fix is an *addition*.
    """

    def __init__(self, qualname: str, variant: int) -> None:
        self.qualname, self.variant = qualname, variant
        self.applied = False
        self.lineno = 0
        self.detail = ""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        if node.name != self.qualname.split(".")[-1] or self.applied:
            return node
        if not node.args.args:
            return node
        first = node.args.args[0].arg
        # Don't stack a guard on top of an existing one.
        if node.body and isinstance(node.body[0], ast.If):
            body0 = node.body[0]
            if len(body0.body) == 1 and isinstance(body0.body[0], ast.Return):
                return node

        templates = [
            (f"if not {first}:\n    return []", "guard empty -> []"),
            (f"if not {first}:\n    return 0.0", "guard empty -> 0.0"),
            (f"if not {first}:\n    return {first}", "guard empty -> passthrough"),
            (f"if not {first}:\n    return None", "guard empty -> None"),
            (f"if not {first}:\n    return ''", "guard empty -> ''"),
        ]
        src, detail = templates[self.variant % len(templates)]
        try:
            guard = ast.parse(src).body[0]
        except SyntaxError:
            return node
        node.body.insert(0, guard)
        self.applied = True
        self.lineno = node.lineno
        self.detail = detail
        return node


_EDIT_KINDS = ["compare", "binop", "offbyone", "slice"]
_SHIFT_KINDS = ["denominator", "index"]
_MAX_OCCURRENCE = 6
_MAX_VARIANT = 4


def propose_patches(
    source: str,
    block: Block,
    surprisal_lines: list[int] | None = None,
    max_patches: int = 60,
) -> list[Patch]:
    """Enumerate candidate repairs for one block, most-promising first.

    `surprisal_lines` (from :mod:`pcg.llm`) reorders the search so edits on the
    lines the language model found least expected are tried first. This does
    not change which patches are reachable, only how quickly a correct one is
    found -- but on a validated search, ordering is the whole cost.
    """
    out: list[Patch] = []
    seen: set[str] = set()
    lo, hi = block.lineno, block.end_lineno

    def add(p: Patch) -> None:
        # Several (occurrence, variant) pairs collapse onto the same edit when
        # an operator has fewer alternatives than variants. Validating those
        # duplicates would multiply the cost of the search for nothing.
        key = p.source
        if key in seen:
            return
        seen.add(key)
        out.append(p)

    for kind in _EDIT_KINDS:
        for idx in range(_MAX_OCCURRENCE):
            exhausted = True
            for variant in range(_MAX_VARIANT):
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    return out
                rw = _Rewriter(kind, idx, variant, lo, hi)
                new_tree = rw.visit(tree)
                if not rw.applied:
                    continue
                exhausted = False
                ast.fix_missing_locations(new_tree)
                try:
                    patched = ast.unparse(new_tree)
                    ast.parse(patched)
                except Exception:
                    continue
                if patched.strip() == source.strip():
                    continue
                add(
                    Patch(
                        bid=block.bid, qualname=block.qualname, operator=kind,
                        description=rw.detail, lineno=rw.lineno, source=patched,
                        before=rw.before, after=rw.after,
                    )
                )
            if exhausted:
                break

    for kind in _SHIFT_KINDS:
        for idx in range(_MAX_OCCURRENCE):
            exhausted = True
            for variant in range(2):
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    return out
                rw = _ShiftRewriter(kind, idx, variant, lo, hi)
                new_tree = rw.visit(tree)
                if not rw.applied:
                    continue
                exhausted = False
                ast.fix_missing_locations(new_tree)
                try:
                    patched = ast.unparse(new_tree)
                    ast.parse(patched)
                except Exception:
                    continue
                add(
                    Patch(
                        bid=block.bid, qualname=block.qualname, operator=kind,
                        description=rw.detail, lineno=rw.lineno, source=patched,
                        before=rw.before, after=rw.after,
                    )
                )
            if exhausted:
                break

    # Guard insertion, for defects whose fix is an addition rather than an edit.
    if block.kind in ("function", "method"):
        for variant in range(5):
            try:
                tree = ast.parse(source)
            except SyntaxError:
                break
            gi = _GuardInserter(block.qualname, variant)
            new_tree = gi.visit(tree)
            if not gi.applied:
                break
            ast.fix_missing_locations(new_tree)
            try:
                patched = ast.unparse(new_tree)
                ast.parse(patched)
            except Exception:
                continue
            add(
                Patch(
                    bid=block.bid, qualname=block.qualname, operator="guard",
                    description=gi.detail, lineno=gi.lineno, source=patched,
                    before="(none)", after=gi.detail,
                )
            )

    if surprisal_lines:
        rank = {ln: i for i, ln in enumerate(surprisal_lines)}
        big = len(rank) + 1
        out.sort(key=lambda p: rank.get(p.lineno, big))

    return out[:max_patches]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _run_tests(source: str, tests: str, timeout: int = 25) -> tuple[int, int]:
    """Return (passed, failed). A hang counts as a failure, not a pass."""
    import re

    wd = tempfile.mkdtemp(prefix="repair_")
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
        out = proc.stdout + proc.stderr
        passed = failed = 0
        for m in re.finditer(r"(\d+)\s+(passed|failed|error|errors)\b", out):
            n, what = int(m.group(1)), m.group(2)
            if what == "passed":
                passed += n
            else:
                failed += n
        return passed, failed
    except subprocess.TimeoutExpired:
        return 0, 1
    except Exception:
        return 0, 1
    finally:
        shutil.rmtree(wd, ignore_errors=True)


@dataclass
class RepairResult:
    target_bid: str
    target_qualname: str
    culpability: float
    attempted: int = 0
    accepted: list[Patch] = field(default_factory=list)
    baseline_passed: int = 0
    baseline_failed: int = 0

    @property
    def repaired(self) -> bool:
        return bool(self.accepted)

    @property
    def best(self) -> Patch | None:
        if not self.accepted:
            return None
        return max(self.accepted, key=lambda p: (p.tests_passed, -p.tests_failed))


def _verdict(
    passed: int, failed: int, base_passed: int, base_failed: int
) -> str | None:
    """Classify a patch outcome. Returns None if the patch is not an improvement.

    A real program usually contains more than one defect, so demanding that a
    single patch turn the suite fully green would reject every correct fix in a
    multi-bug file -- which is exactly what happened on the demo. But merely
    counting failures is too lax: a patch that fixes one test while breaking
    another leaves the count unchanged and would sneak through.

    So we require improvement on *both* axes: strictly more tests passing and
    strictly fewer failing. That admits a genuine partial repair and rejects a
    trade.
    """
    if failed == 0 and passed > 0:
        return "full"
    if passed > base_passed and failed < base_failed:
        return "partial"
    return None


def repair(
    source: str,
    tests: str,
    posteriors: dict[str, BlockPosterior],
    blocks: list[Block] | None = None,
    max_targets: int = 2,
    max_patches: int = 60,
    surprisal: dict[str, list[int]] | None = None,
    verbose: bool = False,
) -> list[RepairResult]:
    """Attempt validated repairs on the most culpable blocks.

    Targets are chosen by **culpability**, not posterior: a block that only
    looks bad because its dependencies are bad has nothing to fix, and
    attempting to patch it would paper over the real defect while making the
    tests pass. That distinction is the reason the two-number decomposition
    exists.
    """
    blocks = blocks if blocks is not None else extract_blocks(source)
    by_bid = {b.bid: b for b in blocks}

    base_pass, base_fail = _run_tests(source, tests)
    if base_fail == 0:
        return []

    ranked = sorted(
        (bp for bp in posteriors.values() if bp.culpability > 0.10),
        key=lambda bp: bp.culpability,
        reverse=True,
    )[:max_targets]

    results: list[RepairResult] = []
    for bp in ranked:
        blk = by_bid.get(bp.bid)
        if blk is None or blk.kind not in ("function", "method"):
            continue
        res = RepairResult(
            target_bid=bp.bid, target_qualname=blk.qualname,
            culpability=bp.culpability,
            baseline_passed=base_pass, baseline_failed=base_fail,
        )
        cands = propose_patches(
            source, blk, (surprisal or {}).get(bp.bid), max_patches
        )
        # Validation is subprocess-bound, so threads give real parallelism here
        # despite the GIL. We evaluate in chunks and stop at the first complete
        # fix, so a patch found early still avoids the remaining test runs.
        chunk = max(1, min(8, (os.cpu_count() or 4)))
        for start in range(0, len(cands), chunk):
            batch = cands[start : start + chunk]
            with ThreadPoolExecutor(max_workers=chunk) as pool:
                outcomes = list(pool.map(lambda p: _run_tests(p.source, tests), batch))
            res.attempted += len(batch)
            found = False
            for p, (passed, failed) in zip(batch, outcomes):
                p.tests_passed, p.tests_failed = passed, failed
                verdict = _verdict(passed, failed, base_pass, base_fail)
                if verdict is None:
                    continue
                p.validated = True
                p.verdict = verdict
                res.accepted.append(p)
                if verbose:
                    print(f"  [{verdict}] {blk.qualname}: {p.description} "
                          f"({base_pass}P/{base_fail}F -> {passed}P/{failed}F)")
                # A full fix ends the search; a partial one is worth reporting
                # but we keep looking in case a later patch does better.
                if verdict == "full":
                    found = True
                    break
            if found:
                break
        results.append(res)
    return results


def rescore_patch(patch: Patch, tests: str, posteriors: dict[str, BlockPosterior]) -> Patch:
    """Re-run PCG inference on the patched source to confirm doubt actually fell."""
    from .pipeline import analyze

    before = posteriors.get(patch.bid)
    patch.posterior_before = before.posterior if before else 0.0
    try:
        a = analyze(patch.source, tests)
    except Exception:
        return patch
    by_name = {b.qualname: b.bid for b in a.blocks}
    bid = by_name.get(patch.qualname)
    if bid and bid in a.posteriors:
        patch.posterior_after = a.posteriors[bid].posterior
    return patch
