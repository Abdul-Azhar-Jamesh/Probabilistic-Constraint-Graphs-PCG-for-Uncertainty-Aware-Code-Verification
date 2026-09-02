"""Evaluation harness: does the framework's confidence mean anything?

The central claim of an uncertainty-aware system is *calibration* — when it
says 0.7, it should be right about 70% of the time. Ranking quality matters
too: a reviewer with limited time should find the bugs by reading from the top.

Run: python -m pcg.evaluate
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .inference import PROJECT_ROOT, brier_score, expected_calibration_error
from .pipeline import analyze


@dataclass
class Case:
    name: str
    source: str
    tests: str
    buggy: set[str]  # qualnames whose OWN code is defective
    # buggy + everything that consumes them. Left empty it defaults to `buggy`,
    # since a defective block is always also untrustworthy.
    unreliable: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.unreliable:
            self.unreliable = set(self.buggy)


# Two questions, two ground truths
# --------------------------------
# The framework emits two numbers per block and they answer different
# questions. Scoring both against a single label is what made the earlier
# "false positives" look like errors when they were correct answers to the
# other question:
#
#   posterior   -> "can I trust this block's output?"  Label: `unreliable`.
#                  `summarize` calls a broken `median`, so its output really
#                  is wrong. Flagging it is CORRECT.
#   culpability -> "is this block's own code defective?"  Label: `buggy`.
#                  `summarize` itself is fine, so NOT flagging it is CORRECT.
#
# Only the second drives repair. Reporting them separately is the honest
# framing, and it is the part of the design worth defending in the report.


def _load_builtin_cases() -> list[Case]:
    """Cases with hand-labelled ground truth.

    Each case pairs a module containing known defects with a test suite. The
    `buggy` set lists only blocks whose *own* code is wrong -- callers of buggy
    code are not labelled buggy, which is exactly the distinction the
    culpability decomposition is meant to capture.
    """
    stats_src = '''
def mean(xs):
    return sum(xs) / len(xs)


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n % 2 == 0:
        return s[n // 2]
    return s[n // 2]


def variance(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def stdev(xs):
    return variance(xs) ** 0.5


def summarize(xs):
    return {"mean": mean(xs), "median": median(xs), "stdev": stdev(xs)}
'''
    stats_tests = '''
import pytest
from candidate import mean, median, variance, stdev, summarize

def test_mean():
    assert mean([1, 2, 3, 4]) == 2.5

def test_median_even():
    assert median([1, 2, 3, 4]) == 2.5

def test_median_odd():
    assert median([3, 1, 2]) == 2

def test_variance():
    assert variance([1, 2, 3, 4]) == pytest.approx(1.6666666, rel=1e-4)

def test_stdev():
    assert stdev([1, 2, 3, 4]) == pytest.approx(1.2909944, rel=1e-4)

def test_summarize():
    assert summarize([1, 2, 3, 4])["median"] == 2.5
'''

    search_src = '''
def binary_search(xs, target):
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        if xs[mid] < target:
            lo = mid
        else:
            hi = mid
    return -1


def contains(xs, target):
    return binary_search(xs, target) >= 0


def count_range(xs, lo, hi):
    return sum(1 for x in xs if lo <= x <= hi)
'''
    search_tests = '''
from candidate import binary_search, contains, count_range

def test_found():
    assert binary_search([1, 3, 5, 7], 5) == 2

def test_missing():
    assert binary_search([1, 3, 5, 7], 4) == -1

def test_contains():
    assert contains([1, 3, 5, 7], 7) is True or contains([1, 3, 5, 7], 7) == True

def test_count_range():
    assert count_range([1, 2, 3, 4, 5], 2, 4) == 3
'''

    text_src = '''
def word_count(text):
    return len(text.split())


def capitalize_all(words):
    return [w.capitalize() for w in words]


def initials(name):
    parts = name.split(" ")
    return "".join(p[0] for p in parts)


def truncate(text, n):
    if len(text) < n:
        return text
    return text[:n] + "..."
'''
    text_tests = '''
from candidate import word_count, capitalize_all, initials, truncate

def test_word_count():
    assert word_count("a b c") == 3

def test_capitalize():
    assert capitalize_all(["ab", "cd"]) == ["Ab", "Cd"]

def test_initials():
    assert initials("ada lovelace") == "al"

def test_initials_double_space():
    assert initials("ada  lovelace") == "al"

def test_truncate_exact():
    assert truncate("abcde", 5) == "abcde"
'''

    return [
        Case(
            "stats",
            stats_src,
            stats_tests,
            buggy={"median", "variance"},
            # stdev calls variance; summarize calls median and stdev.
            unreliable={"median", "variance", "stdev", "summarize"},
        ),
        Case(
            "search",
            search_src,
            search_tests,
            buggy={"binary_search"},
            unreliable={"binary_search", "contains"},
        ),
        Case(
            "text",
            text_src,
            text_tests,
            buggy={"initials", "truncate"},
            unreliable={"initials", "truncate"},
        ),
    ]


def _prf(flagged_true: int, flagged_false: int, missed: int) -> dict:
    p = flagged_true / (flagged_true + flagged_false) if (flagged_true + flagged_false) else 0.0
    r = flagged_true / (flagged_true + missed) if (flagged_true + missed) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "tp": flagged_true,
        "fp": flagged_false,
        "fn": missed,
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f, 3),
    }


def evaluate(cases: list[Case] | None = None, threshold: float = 0.5) -> dict:
    cases = cases or _load_builtin_cases()
    rows: list[dict] = []

    for case in cases:
        a = analyze(case.source, case.tests)
        bmap = {b.bid: b for b in a.blocks}
        for bid, bp in a.posteriors.items():
            b = bmap[bid]
            if b.kind == "module":
                continue
            rows.append(
                {
                    "case": case.name,
                    "block": b.qualname,
                    "is_buggy": b.qualname in case.buggy,
                    "is_unreliable": b.qualname in case.unreliable,
                    "posterior": bp.posterior,
                    "culpability": bp.culpability,
                    "inherited": bp.inherited,
                    "risk": bp.risk,
                }
            )

    # -- Task A: trustworthiness, scored on the posterior ----------------
    probs = [r["posterior"] for r in rows]
    labels_trust = [0 if r["is_unreliable"] else 1 for r in rows]
    trust = _prf(
        sum(1 for r in rows if r["posterior"] < threshold and r["is_unreliable"]),
        sum(1 for r in rows if r["posterior"] < threshold and not r["is_unreliable"]),
        sum(1 for r in rows if r["posterior"] >= threshold and r["is_unreliable"]),
    )
    trust["calibration"] = {
        "ece": expected_calibration_error(probs, labels_trust),
        "brier": brier_score(probs, labels_trust),
    }

    # -- Task B: localisation, scored on culpability ---------------------
    cul_cut = 0.5
    blame = _prf(
        sum(1 for r in rows if r["culpability"] >= cul_cut and r["is_buggy"]),
        sum(1 for r in rows if r["culpability"] >= cul_cut and not r["is_buggy"]),
        sum(1 for r in rows if r["culpability"] < cul_cut and r["is_buggy"]),
    )
    cul_probs = [1 - r["culpability"] for r in rows]
    labels_blame = [0 if r["is_buggy"] else 1 for r in rows]
    blame["calibration"] = {
        "ece": expected_calibration_error(cul_probs, labels_blame),
        "brier": brier_score(cul_probs, labels_blame),
    }
    blame["threshold"] = cul_cut

    # -- Ranking quality: how far must a reviewer read? ------------------
    ranked = sorted(rows, key=lambda r: -r["risk"])
    n_bugs = sum(1 for r in rows if r["is_buggy"])
    found = 0
    depth = len(ranked)
    for i, r in enumerate(ranked, 1):
        if r["is_buggy"]:
            found += 1
            if found == n_bugs:
                depth = i
                break
    p_at_k = {
        k: round(
            sum(1 for r in ranked[:k] if r["is_buggy"]) / min(k, len(ranked)), 3
        )
        for k in (1, 3, 5)
    }

    # Random-order baseline: expected depth to find all bugs by chance.
    n = len(ranked)
    baseline_depth = round(n_bugs * (n + 1) / (n_bugs + 1), 1) if n_bugs else 0.0

    return {
        "n_blocks": len(rows),
        "n_buggy": n_bugs,
        "n_unreliable": sum(1 for r in rows if r["is_unreliable"]),
        "trust_task": trust,
        "blame_task": blame,
        "threshold": threshold,
        "ranking": {
            "precision_at_k": p_at_k,
            "blocks_read_to_find_all_bugs": depth,
            "random_baseline_depth": baseline_depth,
            "total_blocks": n,
            "review_effort_saved": round(1 - depth / max(n, 1), 3),
        },
        "rows": rows,
    }


def main() -> None:
    from rich.console import Console
    from rich.table import Table

    from .report import _console

    console: Console = _console()
    res = evaluate()

    console.rule("[bold]PCG Evaluation")
    console.print(
        f"\n[bold]{res['n_blocks']}[/bold] blocks across 3 cases · "
        f"[bold]{res['n_buggy']}[/bold] defective · "
        f"[bold]{res['n_unreliable']}[/bold] untrustworthy "
        f"[dim](defective + downstream consumers)[/dim]\n"
    )

    t = res["trust_task"]
    b = res["blame_task"]

    console.print(
        "[bold]Task A — Trustworthiness[/bold] "
        f"[dim](posterior < {res['threshold']}: 'can I rely on this output?')[/dim]"
    )
    console.print(
        f"  precision {t['precision']:.3f}   recall {t['recall']:.3f}   "
        f"F1 {t['f1']:.3f}   [dim]TP={t['tp']} FP={t['fp']} FN={t['fn']}[/dim]"
    )
    console.print(
        f"  ECE {t['calibration']['ece']:.4f}   "
        f"Brier {t['calibration']['brier']:.4f}\n"
    )

    console.print(
        "[bold]Task B — Fault localisation[/bold] "
        f"[dim](culpability >= {b['threshold']}: 'is this block itself broken?')[/dim]"
    )
    console.print(
        f"  precision {b['precision']:.3f}   recall {b['recall']:.3f}   "
        f"F1 {b['f1']:.3f}   [dim]TP={b['tp']} FP={b['fp']} FN={b['fn']}[/dim]"
    )
    console.print(
        f"  ECE {b['calibration']['ece']:.4f}   "
        f"Brier {b['calibration']['brier']:.4f}\n"
    )

    r = res["ranking"]
    console.print("[bold]Review ranking[/bold]")
    for k, v in r["precision_at_k"].items():
        console.print(f"  P@{k}  {v:.3f}")
    console.print(
        f"  all {res['n_buggy']} bugs found within top "
        f"[bold]{r['blocks_read_to_find_all_bugs']}[/bold] of "
        f"{r['total_blocks']} blocks "
        f"[dim](random order would need ~{r['random_baseline_depth']})[/dim]"
    )
    console.print(
        f"  [green]{100*r['review_effort_saved']:.0f}% of manual review "
        f"skipped[/green]\n"
    )

    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Case", width=7)
    tbl.add_column("Block", width=15)
    tbl.add_column("Truth", width=10)
    tbl.add_column("P(ok)", justify="right", width=6)
    tbl.add_column("Own", justify="right", width=5)
    tbl.add_column("Inh", justify="right", width=5)
    tbl.add_column("Trust", width=9)
    tbl.add_column("Blame", width=9)

    for row in sorted(res["rows"], key=lambda x: -x["risk"]):
        flagged_t = row["posterior"] < res["threshold"]
        flagged_b = row["culpability"] >= b["threshold"]

        def verdict(flagged: bool, truth: bool) -> str:
            if flagged and truth:
                return "[green]hit[/green]"
            if not flagged and not truth:
                return "[dim]ok[/dim]"
            if flagged:
                return "[yellow]false pos[/yellow]"
            return "[bold red]MISS[/bold red]"

        truth_label = (
            "BUG"
            if row["is_buggy"]
            else "downstream" if row["is_unreliable"] else "ok"
        )
        tbl.add_row(
            row["case"],
            row["block"],
            truth_label,
            f"{row['posterior']:.3f}",
            f"{row['culpability']:.2f}",
            f"{row['inherited']:.2f}",
            verdict(flagged_t, row["is_unreliable"]),
            verdict(flagged_b, row["is_buggy"]),
        )
    console.print(tbl)

    out_path = os.path.join(PROJECT_ROOT, "out", "evaluation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    console.print(f"\n[dim]Full results -> {out_path}[/dim]")


if __name__ == "__main__":
    main()
