"""Rendering: the Correctness Probability Map and ranked review report."""

from __future__ import annotations

import html
import json
from dataclasses import asdict

from .blocks import Block
from .evidence import Evidence
from .inference import BlockPosterior, find_root_repairs, repair_targets, review_ranking

# Risk bands drive both the console colours and the heat map.
BANDS = [
    (0.85, "SAFE", "green"),
    (0.65, "LIKELY OK", "cyan"),
    (0.45, "UNCERTAIN", "yellow"),
    (0.25, "SUSPECT", "orange3"),
    (0.00, "CRITICAL", "red"),
]


def band_of(p: float) -> tuple[str, str]:
    for thresh, label, colour in BANDS:
        if p >= thresh:
            return label, colour
    return "CRITICAL", "red"


def _console():
    """A Console that survives legacy Windows code pages.

    cmd.exe defaults to cp1252, which cannot encode the arrows and box glyphs
    rich emits; without this the whole report dies mid-render on Windows.
    """
    import sys

    from rich.console import Console

    stream = sys.stdout
    reconf = getattr(stream, "reconfigure", None)
    if reconf is not None:
        try:
            reconf(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
    return Console(file=stream, legacy_windows=False, soft_wrap=False)


def render_console(
    blocks: list[Block],
    post: dict[str, BlockPosterior],
    evidence: list[Evidence],
    graph,
    threshold: float = 0.5,
) -> None:
    """Ranked correctness table + per-line heat map, printed to the terminal."""
    from rich.table import Table
    from rich.text import Text

    console = _console()
    ranked = review_ranking(post, blocks)

    console.rule("[bold]Correctness Probability Map")

    table = Table(show_header=True, header_style="bold", expand=False, pad_edge=False)
    table.add_column("#", justify="right", width=2)
    table.add_column("Block", style="bold", width=12, no_wrap=True)
    table.add_column("Lines", justify="center", width=7, no_wrap=True)
    table.add_column("Pri", justify="right", width=4)
    table.add_column("P(ok)", justify="right", width=5)
    table.add_column("Own", justify="right", width=4)
    table.add_column("Inh", justify="right", width=4)
    table.add_column("Imp", justify="right", width=4)
    table.add_column("Risk", justify="right", width=5)
    table.add_column("Verdict", width=9, no_wrap=True)

    for i, (b, bp) in enumerate(ranked, 1):
        label, colour = band_of(bp.posterior)
        table.add_row(
            str(i),
            b.qualname,
            f"{b.lineno}-{b.end_lineno}",
            f"{bp.prior:.2f}",
            Text(f"{bp.posterior:.3f}", style=colour),
            f"{bp.culpability:.2f}",
            f"{bp.inherited:.2f}",
            f"{bp.importance:.2f}",
            Text(f"{bp.risk:.3f}", style="bold " + colour),
            Text(label, style=colour),
        )
    console.print(table)
    console.print(
        "[dim]Pri=structural prior · P(ok)=posterior after propagation · "
        "Own=doubt originating here · Inh=doubt inherited from dependencies · "
        "Imp=structural impact · Risk=review priority[/dim]"
    )

    # -- why: the top negative drivers for the riskiest blocks --------
    console.print()
    console.rule("[bold]Why these blocks are suspect")
    bmap = {b.bid: b for b in blocks}
    for b, bp in ranked[:5]:
        if bp.posterior >= threshold and not bp.top_reasons:
            continue
        label, colour = band_of(bp.posterior)
        console.print(
            f"\n[bold]{b.qualname}[/bold] "
            f"[{colour}]P(correct)={bp.posterior:.3f} · {label}[/{colour}]"
        )
        for r in bp.top_reasons:
            console.print(f"    [red]![/red] {r}")
        drivers = [c for c in bp.contributions if c[1] < 0][:4]
        if drivers:
            detail = "  ".join(f"{k} ({v:+.2f})" for k, v in drivers)
            console.print(f"    [dim]log-odds: {detail}[/dim]")
        if bp.inherited > 0.05:
            weak = sorted(
                (
                    (bmap[p].qualname, post[p].posterior)
                    for p in graph.predecessors(b.bid)
                    if p in post
                ),
                key=lambda t: t[1],
            )[:3]
            names = ", ".join(f"{n} (P={p:.2f})" for n, p in weak)
            console.print(
                f"    [yellow]~[/yellow] inherits {bp.inherited:.2f} doubt "
                f"from dependencies: {names}"
            )
            if bp.culpability < 0.10:
                console.print(
                    "    [dim]This block's own evidence is clean; it is "
                    "suspect only via its dependencies. Fix those first.[/dim]"
                )

    # -- targeted repair set & root analysis ---------------------------
    root_bids = set(find_root_repairs(post, graph, threshold=threshold))
    repair_bids = set(repair_targets(post, threshold=threshold))
    flagged = [(b, bp) for b, bp in ranked if bp.posterior < threshold or bp.culpability >= threshold]
    console.print()
    console.rule("[bold]Targeted Repair & Root Defect Analysis")
    if not flagged and not repair_bids:
        console.print(
            f"[green]No block falls below threshold (all P(correct) >= {threshold:.2f} "
            "and culpability < {threshold:.2f}). Full regeneration unnecessary.[/green]"
        )
    else:
        total_loc = sum(b.loc for b in blocks)
        flagged_loc = sum(b.loc for b, _ in flagged)
        root_blocks = [(b, bp) for b, bp in flagged if b.bid in root_bids]
        cascade_repairs = [(b, bp) for b, bp in flagged if b.bid in repair_bids and b.bid not in root_bids]
        collateral = [(b, bp) for b, bp in flagged if b.bid not in repair_bids and bp.posterior < threshold]

        console.print(
            f"[yellow]{len(flagged)} of {len(blocks)} blocks suspect "
            f"({flagged_loc}/{total_loc} lines = "
            f"{100*flagged_loc/max(total_loc,1):.0f}% of the file).[/yellow]\n"
        )
        if root_blocks:
            console.print("[bold red]ROOT DEFECTS (Fix these first — originating bugs):[/bold red]")
            for b, bp in sorted(root_blocks, key=lambda t: -t[1].culpability):
                console.print(
                    f"    [bold red]>[/bold red] [bold]{b.qualname}[/bold] (lines {b.lineno}-{b.end_lineno})\n"
                    f"       Intrinsic Correctness: {100*(1.0-bp.culpability):.1f}%  |  "
                    f"Effective Trust: {100*bp.posterior:.1f}%  |  "
                    f"Repair Priority: [bold red]HIGH[/bold red] (own doubt: {bp.culpability:.2f})"
                )
        if cascade_repairs:
            console.print("\n[bold orange3]SECONDARY REPAIRS (Subordinate defective blocks):[/bold orange3]")
            for b, bp in sorted(cascade_repairs, key=lambda t: -t[1].culpability):
                console.print(
                    f"    [orange3]>[/orange3] {b.qualname} (lines {b.lineno}-{b.end_lineno})\n"
                    f"       Intrinsic Correctness: {100*(1.0-bp.culpability):.1f}%  |  "
                    f"Effective Trust: {100*bp.posterior:.1f}%  |  "
                    f"Repair Priority: [orange3]MEDIUM[/orange3] (own doubt: {bp.culpability:.2f})"
                )
        if collateral:
            console.print(
                "\n[bold yellow]AFFECTED CALLERS (Collateral only — re-verify after roots are fixed):[/bold yellow]"
            )
            for b, bp in collateral:
                console.print(
                    f"    [yellow]~[/yellow] {b.qualname} (lines {b.lineno}-{b.end_lineno})\n"
                    f"       Intrinsic Correctness: {100*(1.0-bp.culpability):.1f}%  |  "
                    f"Effective Trust: {100*bp.posterior:.1f}%  |  "
                    f"Repair Priority: [green]LOW[/green] (inherited doubt: {bp.inherited:.2f})"
                )

        root_loc = sum(b.loc for b, _ in root_blocks)
        console.print(
            f"\n[dim]Root repairs touch only {root_loc}/{total_loc} lines "
            f"({100*root_loc/max(total_loc,1):.0f}%), leaving "
            f"{100 - 100*root_loc/max(total_loc,1):.0f}% of the program "
            f"untouched.[/dim]"
        )


def render_heatmap_console(
    source: str, blocks: list[Block], post: dict[str, BlockPosterior]
) -> None:
    """Print the source with each line tinted by its block's posterior."""
    from rich.text import Text

    from .blocks import line_to_block

    console = _console()
    console.print()
    console.rule("[bold]Source heat map")
    owner = line_to_block(blocks)
    for i, line in enumerate(source.splitlines(), 1):
        bid = owner.get(i)
        if bid and bid in post:
            p = post[bid].posterior
            _, colour = band_of(p)
            marker = f"{p:.2f}"
        else:
            colour, marker = "dim", "    "
        console.print(
            Text(f"{i:>3} ", style="dim")
            + Text(f"{marker:>5} ", style=colour)
            + Text(line, style=colour if colour != "dim" else "")
        )


def to_dict(
    blocks: list[Block],
    post: dict[str, BlockPosterior],
    evidence: list[Evidence],
    threshold: float = 0.5,
) -> dict:
    """Machine-readable report."""
    ev_by_block: dict[str, list[dict]] = {}
    for e in evidence:
        ev_by_block.setdefault(e.bid, []).append(asdict(e))

    ranked = review_ranking(post, blocks)
    return {
        "summary": {
            "n_blocks": len(blocks),
            "mean_posterior": round(
                sum(bp.posterior for bp in post.values()) / max(len(post), 1), 4
            ),
            "min_posterior": round(
                min((bp.posterior for bp in post.values()), default=1.0), 4
            ),
            "n_below_threshold": sum(
                1 for bp in post.values() if bp.posterior < threshold
            ),
            "threshold": threshold,
        },
        "ranking": [
            {
                "rank": i,
                "bid": b.bid,
                "qualname": b.qualname,
                "kind": b.kind,
                "lines": [b.lineno, b.end_lineno],
                "loc": b.loc,
                "cyclomatic": b.cyclomatic,
                "prior": bp.prior,
                "local": bp.local,
                "posterior": bp.posterior,
                "entropy": bp.entropy,
                "importance": bp.importance,
                "culpability": bp.culpability,
                "inherited": bp.inherited,
                "risk": bp.risk,
                "verdict": band_of(bp.posterior)[0],
                "reasons": bp.top_reasons,
                "contributions": bp.contributions,
                "evidence": ev_by_block.get(b.bid, []),
            }
            for i, (b, bp) in enumerate(ranked, 1)
        ],
    }


def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def write_html(path: str, source: str, blocks: list[Block], post, data: dict) -> None:
    """Standalone HTML heat map — the visual deliverable for the report."""
    from .blocks import line_to_block

    owner = line_to_block(blocks)
    css_for = {
        "SAFE": "#1a7f37",
        "LIKELY OK": "#2da44e",
        "UNCERTAIN": "#bf8700",
        "SUSPECT": "#d1662b",
        "CRITICAL": "#cf222e",
    }

    rows = []
    for i, line in enumerate(source.splitlines(), 1):
        bid = owner.get(i)
        if bid and bid in post:
            p = post[bid].posterior
            label, _ = band_of(p)
            colour = css_for[label]
            alpha = 0.06 + 0.34 * (1 - p)
            rows.append(
                f'<tr style="background:rgba({_rgb(colour)},{alpha:.2f})">'
                f'<td class="ln">{i}</td>'
                f'<td class="pv" style="color:{colour}">{p:.2f}</td>'
                f"<td><pre>{html.escape(line) or '&nbsp;'}</pre></td></tr>"
            )
        else:
            rows.append(
                f'<tr><td class="ln">{i}</td><td class="pv"></td>'
                f"<td><pre>{html.escape(line) or '&nbsp;'}</pre></td></tr>"
            )

    ranked_rows = "".join(
        f"<tr><td>{r['rank']}</td><td><code>{html.escape(r['qualname'])}</code></td>"
        f"<td>{r['lines'][0]}&ndash;{r['lines'][1]}</td>"
        f"<td>{r['prior']:.2f}</td>"
        f"<td style=\"color:{css_for[r['verdict']]};font-weight:600\">"
        f"{r['posterior']:.3f}</td>"
        f"<td>{r['culpability']:.2f}</td><td>{r['inherited']:.2f}</td>"
        f"<td>{r['importance']:.2f}</td>"
        f"<td><b>{r['risk']:.3f}</b></td>"
        f"<td style=\"color:{css_for[r['verdict']]}\">{r['verdict']}</td>"
        f"<td class=\"why\">{html.escape('; '.join(r['reasons'][:2]))}</td></tr>"
        for r in data["ranking"]
    )

    s = data["summary"]
    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Correctness Probability Map</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:2rem auto;
      max-width:1100px;color:#1f2328}}
 h1{{font-size:1.5rem;margin-bottom:.2rem}}
 .sub{{color:#656d76;margin-bottom:1.5rem}}
 table{{border-collapse:collapse;width:100%;margin-bottom:2rem}}
 th,td{{padding:.35rem .5rem;text-align:left;border-bottom:1px solid #d0d7de;
        font-size:13px;vertical-align:top}}
 th{{background:#f6f8fa;font-weight:600}}
 pre{{margin:0;font:12.5px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;
      white-space:pre-wrap}}
 .ln{{width:3rem;color:#8c959f;text-align:right;font-family:ui-monospace,monospace;
      user-select:none}}
 .pv{{width:3.2rem;font-family:ui-monospace,monospace;font-size:11.5px;
      text-align:right}}
 .why{{color:#656d76;font-size:12px;max-width:22rem}}
 .cards{{display:flex;gap:1rem;margin-bottom:1.5rem;flex-wrap:wrap}}
 .card{{border:1px solid #d0d7de;border-radius:8px;padding:.7rem 1rem;min-width:8rem}}
 .card b{{display:block;font-size:1.5rem;line-height:1.2}}
 .card span{{color:#656d76;font-size:12px}}
</style>
<h1>Correctness Probability Map</h1>
<div class="sub">Posterior P(correct) per code block, after Bayesian fusion of
compiler, static-analysis, execution and generator-confidence evidence,
propagated across the constraint graph.</div>
<div class="cards">
  <div class="card"><b>{s['n_blocks']}</b><span>blocks</span></div>
  <div class="card"><b>{s['mean_posterior']:.3f}</b><span>mean P(correct)</span></div>
  <div class="card"><b>{s['min_posterior']:.3f}</b><span>min P(correct)</span></div>
  <div class="card"><b>{s['n_below_threshold']}</b>
       <span>below {s['threshold']:.2f}</span></div>
</div>
<h2 style="font-size:1.1rem">Review priority</h2>
<table><thead><tr><th>#</th><th>Block</th><th>Lines</th><th>Prior</th>
<th>P(correct)</th><th>Own</th><th>Inh</th><th>Impact</th><th>Risk</th>
<th>Verdict</th><th>Top reason</th></tr></thead><tbody>{ranked_rows}</tbody></table>
<p class="sub" style="font-size:12px"><b>Own</b> = doubt originating from this
block's own evidence. <b>Inh</b> = doubt inherited from its dependencies via
constraint propagation. A block with high Inh but near-zero Own is not itself
broken &mdash; fix its dependencies instead.</p>
<h2 style="font-size:1.1rem">Source heat map</h2>
<table><tbody>{''.join(rows)}</tbody></table>
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)


def _rgb(hexcolour: str) -> str:
    h = hexcolour.lstrip("#")
    return ",".join(str(int(h[i : i + 2], 16)) for i in (0, 2, 4))
