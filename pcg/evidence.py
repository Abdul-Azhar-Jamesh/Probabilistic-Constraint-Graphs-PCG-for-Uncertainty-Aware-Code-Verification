"""Evidence collection: compiler, static analysis, execution, and LLM signals.

Each collector returns a list of `Evidence`, already attributed to a block id.
Evidence is deliberately kept as raw observations; the mapping from observation
to likelihood lives in `inference.py` so the sensor model can be tuned or
learned without touching collection.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Any

from .blocks import Block, line_to_block
from .execution import ExecutionConfig, ExecutionResult, run_tests


@dataclass
class Evidence:
    """One observation bearing on the correctness of a block."""

    bid: str
    source: str  # "compile" | "static" | "exec" | "llm"
    kind: str  # e.g. "syntax_error", "undefined_name", "test_fail"
    polarity: str  # "positive" | "negative"
    weight: float  # observation strength in [0, 1]
    detail: str = ""
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# 1. Compiler / parser diagnostics
# --------------------------------------------------------------------------
def collect_compile(source: str, blocks: list[Block]) -> list[Evidence]:
    """Syntax and compile-time diagnostics.

    A syntax error is a global fact: it invalidates the whole module, but we
    attribute it most strongly to the block containing the offending line.
    """
    ev: list[Evidence] = []
    owner = line_to_block(blocks)
    try:
        compile(source, "<pcg>", "exec")
    except SyntaxError as e:
        ln = e.lineno or 1
        bid = owner.get(ln)
        # If the error line falls outside every block (blank line, stray token
        # before the first definition), we cannot localise it -- so no block
        # gets exonerated. An unparseable module means nothing in it is
        # trustworthy.
        unlocalised = bid is None
        for b in blocks:
            is_site = b.bid == bid
            ev.append(
                Evidence(
                    bid=b.bid,
                    source="compile",
                    kind="syntax_error",
                    polarity="negative",
                    weight=0.98 if (is_site or unlocalised) else 0.55,
                    detail=f"{e.msg} (line {ln})",
                    meta={"line": ln, "at_site": is_site, "localised": not unlocalised},
                )
            )
        return ev
    except ValueError as e:
        for b in blocks:
            ev.append(
                Evidence(
                    bid=b.bid,
                    source="compile",
                    kind="compile_error",
                    polarity="negative",
                    weight=0.60,
                    detail=str(e),
                )
            )
        return ev

    for b in blocks:
        ev.append(
            Evidence(
                bid=b.bid,
                source="compile",
                kind="parses",
                polarity="positive",
                weight=0.10,
                detail="module compiles",
            )
        )
    return ev


# --------------------------------------------------------------------------
# 2. Static analysis
# --------------------------------------------------------------------------
# Pylint message ids that indicate genuine correctness risk, with severity.
_PYLINT_SEVERITY = {
    "E0602": 0.92,  # undefined-variable
    "E1101": 0.75,  # no-member
    "E1120": 0.90,  # no-value-for-parameter
    "E1121": 0.90,  # too-many-function-args
    "E0601": 0.92,  # used-before-assignment
    "E0102": 0.85,  # function-redefined
    "E1136": 0.70,  # unsubscriptable-object
    "W0631": 0.65,  # undefined-loop-variable
    "W0612": 0.25,  # unused-variable
    "W0613": 0.15,  # unused-argument
    "W0104": 0.40,  # pointless-statement
    "W0702": 0.30,  # bare-except
    "R1710": 0.55,  # inconsistent-return-statements
}


def _warn_static_degraded(reason: str) -> None:
    """Make silent degradation of static analysis visible.

    A missing/broken pylint must not look like 'clean static evidence' —
    emit a warning so the operator knows the sensor is offline.
    """
    warnings.warn(
        f"[pcg] static analysis degraded: {reason}; continuing with AST checks only",
        RuntimeWarning,
        stacklevel=3,
    )


def collect_static(source: str, blocks: list[Block]) -> list[Evidence]:
    """Run pylint, plus a few hand-rolled AST checks it tends to miss."""
    ev: list[Evidence] = []
    owner = line_to_block(blocks)

    # -- pylint ---------------------------------------------------------
    tmp = None
    pylint_available = False
    try:
        fd, tmp = tempfile.mkstemp(suffix=".py", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(source)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pylint",
                "--output-format=json",
                "--score=n",
                "--disable=C,R0801",
                tmp,
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        raw = proc.stdout.strip()
        if not raw.startswith("["):
            raise ValueError("pylint did not return JSON diagnostics")
        msgs = json.loads(raw)
        pylint_available = True
    except subprocess.TimeoutExpired:
        msgs = []
        _warn_static_degraded("pylint timed out after 90s")
    except Exception as exc:  # pylint not installed, non-JSON output, ...
        msgs = []
        _warn_static_degraded(f"pylint unavailable ({exc.__class__.__name__}: {exc})")
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass

    flagged: set[str] = set()
    for m in msgs:
        mid = m.get("message-id", "")
        sev = _PYLINT_SEVERITY.get(mid)
        if sev is None:
            continue
        bid = owner.get(m.get("line", 0))
        if not bid:
            continue
        flagged.add(bid)
        ev.append(
            Evidence(
                bid=bid,
                source="static",
                kind=m.get("symbol", mid),
                polarity="negative",
                weight=sev,
                detail=f"{mid} {m.get('message', '')} (line {m.get('line')})",
                meta={"line": m.get("line")},
            )
        )

    # -- targeted AST checks --------------------------------------------
    smells = _ast_smells(source, blocks)
    ev.extend(smells)
    flagged.update(e.bid for e in smells)

    # A clean bill of health is weak positive evidence, and only for blocks
    # that nothing flagged -- otherwise a block would be simultaneously
    # rewarded for being clean and penalised for its findings.
    for b in blocks:
        if b.bid not in flagged and pylint_available:
            ev.append(
                Evidence(
                    bid=b.bid,
                    source="static",
                    kind="static_clean",
                    polarity="positive",
                    weight=0.22,
                    detail="no static findings",
                )
            )
        elif not pylint_available:
            ev.append(
                Evidence(
                    b.bid,
                    "static",
                    "static_unavailable",
                    "neutral",
                    0.0,
                    "pylint unavailable; only targeted AST checks ran",
                )
            )
    return ev


def _ast_smells(source: str, blocks: list[Block]) -> list[Evidence]:
    """Cheap structural checks for correctness-relevant patterns."""
    out: list[Evidence] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out
    owner = line_to_block(blocks)

    for node in ast.walk(tree):
        # Mutable default argument.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.args.defaults:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    bid = owner.get(node.lineno)
                    if bid:
                        out.append(
                            Evidence(
                                bid,
                                "static",
                                "mutable_default",
                                "negative",
                                0.55,
                                f"mutable default in {node.name}()",
                            )
                        )
        # Comparison to None/True with == instead of is.
        if isinstance(node, ast.Compare):
            for op, cmp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(
                    cmp, ast.Constant
                ):
                    if cmp.value is None or isinstance(cmp.value, bool):
                        bid = owner.get(node.lineno)
                        if bid:
                            out.append(
                                Evidence(
                                    bid,
                                    "static",
                                    "identity_compare",
                                    "negative",
                                    0.35,
                                    f"== used with {cmp.value!r} (line {node.lineno})",
                                )
                            )
        # Division without a guarded denominator is a classic off-by-zero.
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv)):
            if not isinstance(node.right, ast.Constant):
                bid = owner.get(node.lineno)
                if bid:
                    out.append(
                        Evidence(
                            bid,
                            "static",
                            "unguarded_division",
                            "negative",
                            0.30,
                            f"division by non-constant (line {node.lineno})",
                        )
                    )
    return out


# --------------------------------------------------------------------------
# 3. Execution evidence
# --------------------------------------------------------------------------
def execution_evidence(result: ExecutionResult, blocks: list[Block]) -> list[Evidence]:
    """Attach observations only to covered blocks; keep harness errors neutral."""
    owner = line_to_block(blocks)
    spectra = {
        test.nodeid: {owner[line] for line in test.lines if line in owner}
        for test in result.tests
        if test.outcome in {"passed", "failed"}
    }
    failed = result.failed_ids
    scores = ochiai_localization(spectra, failed)
    evidence = []
    if result.status != "complete" or result.errors:
        for block in blocks:
            evidence.append(
                Evidence(
                    block.bid,
                    "exec",
                    result.status if result.status != "complete" else "harness_error",
                    "neutral",
                    0.0,
                    "; ".join(result.errors) or result.status,
                    {"execution_status": result.status, "backend": result.backend},
                )
            )
        return evidence
    # For segmented functions, line ownership names only the segment. Do not
    # add the umbrella function as a second possible defect on the same lines.
    for test in result.tests:
        if test.outcome not in {"passed", "failed"}:
            continue
        covered = spectra[test.nodeid]
        if not covered:
            continue
        observation = {
            "name": test.nodeid,
            "blocks": sorted(covered),
            "failed": test.outcome == "failed",
        }
        frame_blocks = {owner[ln] for ln in test.frames if ln in owner}
        for bid in sorted(covered):
            evidence.append(
                Evidence(
                    bid,
                    "exec",
                    "test_fail" if observation["failed"] else "test_pass",
                    "negative" if observation["failed"] else "positive",
                    1.0,
                    f"{test.nodeid}: {test.outcome}",
                    {
                        "test_observation": observation,
                        "ochiai": scores.get(bid, 0.0),
                        "in_traceback": bid in frame_blocks,
                        "lines": test.lines,
                        "branches": test.arcs,
                        "duration": test.duration,
                    },
                )
            )
    return evidence


def collect_exec(
    source: str,
    blocks: list[Block],
    tests: str | None,
    timeout: int = 30,
    execution: ExecutionConfig | None = None,
) -> list[Evidence]:
    if not tests:
        return []
    return execution_evidence(
        run_tests(source, tests, execution or ExecutionConfig(timeout=timeout)), blocks
    )


def _count_while(source: str) -> int:
    """Number of `while` statements in a block's source."""
    try:
        return sum(
            1 for n in ast.walk(ast.parse(source.strip())) if isinstance(n, ast.While)
        )
    except SyntaxError:
        return len(re.findall(r"^\s*while\b", source, re.MULTILINE))


def _parse_pytest_counts(out: str) -> tuple[int, int]:
    """Extract pass/fail counts from pytest's summary line.

    Matches counts anywhere in the output rather than anchoring on the `=`
    banner, because `-q --no-header` emits a bare `6 failed, 3 passed in 0.2s`.
    """
    passed = failed = 0
    for m in re.finditer(r"(\d+)\s+(passed|failed|error|errors)\b", out):
        n = int(m.group(1))
        if m.group(2) == "passed":
            passed = max(passed, n)
        else:
            failed = max(failed, n)
    return passed, failed


def _blame_from_traceback(
    out: str, blocks: list[Block], module_name: str = "candidate"
) -> dict[str, float]:
    """Spectrum-based fault localisation from pytest failure output.

    Blame is computed *per failing test* and then accumulated. Doing it
    globally is wrong: one failure that localises cleanly to a source line
    would suppress the weaker heuristics needed by a second failure that only
    manifests as a side effect, silently dropping the second bug.
    """
    sections = _split_failures(out)
    if not sections:
        sections = [out]

    totals: dict[str, float] = {}
    for sec in sections:
        for bid, w in _blame_one_failure(sec, blocks, module_name).items():
            totals[bid] = totals.get(bid, 0.0) + w

    if not totals:
        return {}
    mx = max(totals.values()) or 1.0
    return {k: min(1.0, v / mx) for k, v in totals.items()}


def _split_failures(out: str) -> list[str]:
    """Split pytest output into one chunk per failing test."""
    m = re.search(r"^=+ FAILURES =+$", out, re.MULTILINE)
    if not m:
        return []
    body = out[m.end() :]
    body = re.split(r"^=+ (?:short test summary|warnings summary)", body, flags=re.M)[0]
    parts = re.split(r"^_{3,}\s+\S.*?\s+_{3,}$", body, flags=re.MULTILINE)
    return [p for p in parts if p.strip()]


def _blame_one_failure(
    sec: str, blocks: list[Block], module_name: str
) -> dict[str, float]:
    """Attribute a single test failure to the blocks most likely responsible."""
    owner = line_to_block(blocks)
    scores: dict[str, float] = {}

    def bump(bid: str, w: float) -> None:
        scores[bid] = max(scores.get(bid, 0.0), w)

    fname = re.escape(f"{module_name}.py")

    # Innermost frame: "candidate.py:3: ValueError" -- the frame that raised.
    for m in re.finditer(rf"{fname}:(\d+):\s*(\w*Error|\w*Exception)\b", sec):
        bid = owner.get(int(m.group(1)))
        if bid:
            bump(bid, 1.0)

    # Enclosing frames: "candidate.py:52: in build_config". Callers on the path
    # to the fault, not the fault itself, so they earn markedly less blame.
    for m in re.finditer(rf"{fname}:(\d+):\s+in\s+\w+", sec):
        bid = owner.get(int(m.group(1)))
        if bid:
            bump(bid, 0.35)

    # Any other reference to a line in the module under test.
    for m in re.finditer(rf"{fname}:(\d+)", sec):
        bid = owner.get(int(m.group(1)))
        if bid:
            bump(bid, 0.30)

    names = {b.name: b.bid for b in blocks if b.name != "<module>"}

    # Source context pytest prints for the failing frame: "    def parse_kv(...)".
    for m in re.finditer(r"^\s*def\s+(\w+)\s*\(", sec, re.MULTILINE):
        bid = names.get(m.group(1))
        if bid:
            bump(bid, 0.75)

    if scores:
        return scores

    # Assertion-only failure: the module never appears in the traceback.
    # Recover blame from the asserted expression.
    for m in re.finditer(r"^E\s+(?:assert|\+\s+where)\s+(.*)$", sec, re.MULTILINE):
        for call in re.finditer(r"\b(\w+)\s*\(", m.group(1)):
            bid = names.get(call.group(1))
            if bid:
                bump(bid, 0.9)

    if scores:
        return scores

    # Side-effect failure (argument mutated, expected write missing): the
    # assertion names no function. Blame every module function called anywhere
    # in the failing test body.
    for call in re.finditer(r"\b(\w+)\s*\(", sec):
        bid = names.get(call.group(1))
        if bid:
            bump(bid, 0.8)

    return scores


def _failure_sections_with_names(out: str) -> list[tuple[str, str]]:
    """Pair each failing-test output section with the test's name.

    The test name is what lets us build a per-test coverage spectrum: which
    tests exercise which blocks. Falls back to ("", section) when pytest's
    `_____ test_name _____` banner is absent.
    """
    sections = _split_failures(out)
    named: list[tuple[str, str]] = []
    banners = re.findall(r"^_{3,}\s+(\S.*?)\s+_{3,}$", out, re.MULTILINE)
    for i, sec in enumerate(sections):
        name = banners[i] if i < len(banners) else ""
        named.append((name.strip(), sec))
    return named


def _blocks_in_section(sec: str, blocks: list[Block], module_name: str) -> set[str]:
    """Blocks whose lines appear anywhere in one test's failure output."""
    owner = line_to_block(blocks)
    fname = re.escape(f"{module_name}.py")
    hit: set[str] = set()
    for m in re.finditer(rf"{fname}:(\d+)", sec):
        bid = owner.get(int(m.group(1)))
        if bid:
            hit.add(bid)
    names = {b.name: b.bid for b in blocks if b.name != "<module>"}
    for m in re.finditer(r"\b(\w+)\s*\(", sec):
        bid = names.get(m.group(1))
        if bid:
            hit.add(bid)
    return hit


def ochiai_localization(
    spectra: dict[str, set[str]],
    failed_tests: set[str],
) -> dict[str, float]:
    """Tarantula/Ochiai spectrum-based fault localisation.

    For each block we know which tests exercised it (its *spectrum*). The
    Ochiai coefficient ranks suspects by how concentrated failures are on
    them relative to their total exposure:

        ochiai(b) = failed_exposing(b) / sqrt(total_failed * total_exposing(b))

    This is strictly stronger than traceback blame alone because it uses
    *co-occurrence across all tests*: a block executed by every failing test
    but also by many passing ones is a weaker suspect than one executed only
    by failing tests — exactly the intuition traceback regexes cannot capture.

    `spectra` maps test name -> set of block ids that test exercised.
    `failed_tests` is the subset of test names that failed.
    """
    n_failed = len(failed_tests)
    if n_failed == 0 or not spectra:
        return {}
    exposed_by: dict[str, int] = {}
    failed_by: dict[str, int] = {}
    for tname, bids in spectra.items():
        is_fail = tname in failed_tests
        for bid in bids:
            exposed_by[bid] = exposed_by.get(bid, 0) + 1
            if is_fail:
                failed_by[bid] = failed_by.get(bid, 0) + 1
    scores: dict[str, float] = {}
    for bid, nf in failed_by.items():
        ne = exposed_by.get(bid, 0)
        denom = math.sqrt(n_failed * ne)
        scores[bid] = round(nf / denom, 4) if denom else 0.0
    return scores


# --------------------------------------------------------------------------
# 4. LLM self-confidence
# --------------------------------------------------------------------------
def collect_llm_confidence(
    blocks: list[Block],
    token_logprobs: dict[str, list[float]] | None = None,
) -> list[Evidence]:
    """Turn generator confidence into evidence.

    If real token log-probabilities are supplied (mapping block id -> list of
    per-token logprobs), we combine three statistics — mean token probability,
    the 10th-percentile token probability, and the fraction of low-confidence
    tokens — into a single confidence score. Mean alone is a poor signal: it is
    dominated by boilerplate tokens and hides localised uncertainty. The
    percentile and low-fraction terms surface exactly those spikes.
    Missing log-probabilities produce no evidence: complexity already informs
    the prior and must not be counted again as a simulated LLM observation.
    """
    ev: list[Evidence] = []
    for b in blocks:
        if not token_logprobs or not token_logprobs.get(b.bid):
            continue
        if token_logprobs and b.bid in token_logprobs and token_logprobs[b.bid]:
            lps = sorted(token_logprobs[b.bid])
            n = len(lps)
            mean_prob = math.exp(sum(lps) / n)
            # 10th-percentile logprob: the weakest ~10% of tokens.
            p10_prob = math.exp(lps[max(0, int(0.10 * n) - 1)])
            frac_low = sum(1 for lp in lps if lp < math.log(0.5)) / n
            conf = 0.5 * mean_prob + 0.3 * p10_prob + 0.2 * (1.0 - frac_low)
            detail = (
                f"mean token prob {mean_prob:.3f}, "
                f"p10 {p10_prob:.3f}, frac_low {frac_low:.2f}"
            )
            kind = "logprob"
            stats_meta = {
                "confidence": round(conf, 4),
                "mean_prob": round(mean_prob, 4),
                "p10_prob": round(p10_prob, 4),
                "frac_low": round(frac_low, 4),
            }

        conf = min(max(conf, 0.02), 0.98)
        ev.append(
            Evidence(
                bid=b.bid,
                source="llm",
                kind=kind,
                polarity="positive" if conf >= 0.5 else "negative",
                weight=abs(conf - 0.5) * 2.0,
                detail=detail,
                meta=stats_meta,
            )
        )
    return ev


def collect_llm_critic(
    source: str,
    blocks: list[Block],
    critic: Any = None,
    critic_backend: str | None = None,
) -> list[Evidence]:
    """Gather semantic review evidence from an LLM critic.

    Queries the critic (e.g. Anthropic Claude or Mock heuristic) for each
    non-module block. If the critic judges the block to be buggy, a negative
    evidence item is recorded with weight scaled by its confidence; if clean,
    a positive evidence item is recorded.
    """
    from .critic import get_critic

    if critic is None and critic_backend:
        critic = get_critic(critic_backend)

    if critic is None:
        return []

    ev: list[Evidence] = []
    source_lines = source.splitlines()
    for b in blocks:
        if b.kind == "module":
            continue
        code = b.source or (
            "\n".join(source_lines[b.lineno - 1 : b.end_lineno])
            if b.lineno and b.end_lineno
            else ""
        )
        if not code.strip():
            continue

        verdict = critic.judge(code, file=f"{b.qualname}.py")
        if not isinstance(verdict, dict) or verdict.get("is_buggy") is None:
            continue

        is_buggy = verdict["is_buggy"]
        conf = float(verdict.get("confidence", 0.7) or 0.7)
        conf = min(max(conf, 0.5), 1.0)
        reason = verdict.get("reason", "")

        weight = round(min(1.0, max(0.1, (conf - 0.5) * 2.0)), 3)
        polarity = "negative" if is_buggy else "positive"
        detail = (
            f"LLM critic flagged defect ({conf:.0%} conf): {reason}"
            if is_buggy
            else f"LLM critic approved ({conf:.0%} conf): {reason}"
        )

        ev.append(
            Evidence(
                bid=b.bid,
                source="critic",
                kind="critic_verdict",
                polarity=polarity,
                weight=weight,
                detail=detail,
                meta={
                    "is_buggy": is_buggy,
                    "confidence": conf,
                    "reason": reason,
                    "raw": verdict.get("raw"),
                },
            )
        )
    return ev


def collect_all(
    source: str,
    blocks: list[Block],
    tests: str | None = None,
    token_logprobs: dict[str, list[float]] | None = None,
    critic: Any = None,
    critic_backend: str | None = None,
    execution: ExecutionConfig | None = None,
    execution_result: ExecutionResult | None = None,
) -> list[Evidence]:
    ev = collect_compile(source, blocks)
    fatal = any(e.kind == "syntax_error" for e in ev)
    ev += collect_llm_confidence(blocks, token_logprobs)
    if critic or critic_backend:
        ev += collect_llm_critic(
            source, blocks, critic=critic, critic_backend=critic_backend
        )
    if not fatal:
        ev += collect_static(source, blocks)
        ev += (
            execution_evidence(execution_result, blocks)
            if execution_result is not None
            else collect_exec(source, blocks, tests, execution=execution)
        )
    return ev
