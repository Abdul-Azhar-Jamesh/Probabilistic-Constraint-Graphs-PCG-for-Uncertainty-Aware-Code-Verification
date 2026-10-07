"""Interactive Correctness Probability Map (Streamlit).

Run with:  streamlit run app.py

The console report is a ranked list, which answers "what should I review
first?" but not "why?". The two questions the static report cannot answer are:

  * **Where did this number come from?** The posterior is a product of a
    structural prior, fused evidence, and inherited doubt from dependencies.
    Seeing only the final number makes the system a black box -- which is
    exactly the failure mode an uncertainty-aware tool exists to avoid.
  * **What happens if I disagree?** Source reliabilities and edge strengths are
    hand-set constants. A reader should be able to move them and watch the
    ranking respond, rather than take them on faith.

This app makes both interactive: click a block to see its evidence ledger and
dependency-attributed doubt, and drag the reliability sliders to see live how
sensitive the conclusions are to those choices.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pcg.inference import find_root_repairs, repair_targets, review_ranking  # noqa: E402
from pcg.report import band_of  # noqa: E402
from pcg.pipeline import analyze  # noqa: E402
from pcg.execution import ExecutionConfig  # noqa: E402
from pcg.sensors import SensorModel  # noqa: E402

st.set_page_config(page_title="PCG — Correctness Probability Map", layout="wide")

DEMO = Path(__file__).parent / "demo" / "candidate.py"
DEMO_TESTS = Path(__file__).parent / "demo" / "test_candidate.py"


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
st.title("Probabilistic Constraint Graphs")
st.caption(
    "Correctness Probability Map for AI-generated code — "
    "which blocks need manual review, and why."
)

with st.sidebar:
    st.header("Input")
    mode = st.radio(
        "Source",
        ["Demo (planted bugs)", "Paste your own", "Upload", "Upload project (.zip)"],
    )

    source = tests = ""
    if mode == "Demo (planted bugs)":
        source = DEMO.read_text(encoding="utf-8") if DEMO.exists() else ""
        tests = DEMO_TESTS.read_text(encoding="utf-8") if DEMO_TESTS.exists() else ""
    elif mode == "Paste your own":
        source = st.text_area("Python source", height=200)
        tests = st.text_area("pytest file (optional)", height=120)
    elif mode == "Upload project (.zip)":
        project_upload = st.file_uploader("Python project ZIP", type=["zip"])
    else:
        up = st.file_uploader("Python file", type=["py"])
        ut = st.file_uploader("Test file (optional)", type=["py"])
        if up:
            source = up.read().decode("utf-8")
        if ut:
            tests = ut.read().decode("utf-8")

    execution_mode = st.selectbox(
        "Test execution", ["Disabled", "Docker isolation", "Local trusted code"]
    )
    trusted = execution_mode != "Local trusted code" or st.checkbox(
        "I trust this source and its tests to run on this machine", value=False
    )
    run_tests = execution_mode != "Disabled" and trusted
    execution_backend = (
        "docker"
        if execution_mode == "Docker isolation"
        else "local"
        if run_tests
        else "disabled"
    )
    import os

    if os.environ.get("PCG_PUBLIC_DEPLOYMENT") == "1":
        if execution_backend == "local":
            st.error("Public deployments require Docker isolation for test execution.")
            run_tests = False
            execution_backend = "disabled"
    uploaded_model = st.file_uploader("Optional fitted sensor model", type=["json"])
    model_json = uploaded_model.getvalue().decode("utf-8") if uploaded_model else ""
    inference_method = st.selectbox("Inference", ["auto", "exact", "gibbs"])
    auto_tests = st.checkbox("Automatically explore and test this code", value=True)
    annotation_contracts = st.checkbox(
        "Use declared return types as test contracts", value=False
    )
    contracts_json = st.text_area(
        "Optional behavior contracts (JSON cases)", height=100
    )
    st.caption(
        "Probes explore inputs without asserting correctness. Expected outputs or metamorphic contracts provide stronger checks."
    )

    st.header("Model parameters")
    st.caption(
        "Sensitivity controls for assumed observation likelihoods. "
        "Changing them invalidates any fitted calibration claim."
    )
    rel_exec = 2.8  # joint test likelihoods use sensor probabilities directly
    test_sensitivity = st.slider(
        "P(test fails from an exercised defect)", 0.05, 0.99, 0.95, 0.01
    )
    test_leak = st.slider(
        "P(test fails without a candidate defect)", 0.001, 0.30, 0.01, 0.001
    )
    rel_compile = st.slider("Reliability: compile", 0.0, 5.0, 3.2, 0.1)
    rel_static = st.slider("Reliability: static analysis", 0.0, 5.0, 1.4, 0.1)
    rel_llm = st.slider("Reliability: LLM confidence", 0.0, 5.0, 0.45, 0.05)
    threshold = st.slider("Abstention threshold", 0.0, 1.0, 0.5, 0.05)

    use_llm = st.checkbox(
        "Use real LLM token log-probs (slow first run: downloads model)", False
    )

    critic_choice = st.selectbox(
        "LLM Code Review Critic",
        ["None", "Mock (Heuristic)", "Anthropic Claude"],
    )
    critic_backend_map = {
        "None": None,
        "Mock (Heuristic)": "mock",
        "Anthropic Claude": "anthropic",
    }
    critic_backend = critic_backend_map[critic_choice]

if mode == "Upload project (.zip)":
    import json
    import tempfile
    from pcg.project import analyze_project, unpack_project

    st.caption(
        "Analyse packages, existing tests and automatic graph-guided checks. Supply dependencies in the worker environment."
    )
    if project_upload and st.button("Analyse project"):
        try:
            with tempfile.TemporaryDirectory(prefix="pcg_upload_") as folder:
                project_root = unpack_project(project_upload.getvalue(), Path(folder))
                with st.spinner("Analysing project..."):
                    project_report = analyze_project(
                        project_root, execution=ExecutionConfig(execution_backend)
                    )
            st.session_state["project_report"] = project_report
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    if "project_report" in st.session_state:
        project_report = st.session_state["project_report"]
        st.write("Status:", project_report["status"])
        st.json(project_report["summary"])
        st.dataframe(project_report["ranking"], hide_index=True)
        st.json(
            {
                "diagnoses": project_report["diagnoses"],
                "static": project_report["static"],
                "failure_validation": project_report["failure_validation"],
                "adaptive_budget": project_report["adaptive_budget"],
                "limitations": project_report["limitations"],
            },
            expanded=False,
        )
        st.download_button(
            "Download project report",
            json.dumps(project_report, indent=2),
            "project-report.json",
            "application/json",
        )
    st.stop()

if not source.strip():
    st.info("Provide some Python source in the sidebar to begin.")
    st.stop()


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Scoring with language model…")
def _llm_logprobs(src: str):
    from pcg.blocks import extract_blocks as _eb
    from pcg.llm import score_blocks

    blks = _eb(src)
    return {bid: s.logprobs for bid, s in score_blocks(src, blks).items()}


@st.cache_data(show_spinner="Analysing...")
def _analyse(
    src: str,
    tst: str,
    rels: tuple,
    llm: bool,
    critic_b: str | None,
    backend: str,
    method: str,
    model_json: str,
    sensitivity: float,
    leak: float,
    auto_tests: bool,
    annotation_contracts: bool,
    contracts_json: str,
):
    import json

    model = None
    if model_json:
        payload = json.loads(model_json)
        if (
            payload.get("schema_version") != 1
            or payload.get("model") != "latent-defect-noisy-or-v1"
        ):
            raise ValueError("Unsupported model format")
        model = SensorModel(
            likelihoods=payload["likelihoods"],
            prior_coefficients=payload.get("prior_coefficients"),
            calibrated=False,
            posterior_threshold=payload.get("posterior_threshold", 0.5),
        )
    base = model or SensorModel()
    likelihoods = {source: dict(values) for source, values in base.likelihoods.items()}
    if not model_json:
        likelihoods["exec"] = {
            "flag_given_defect": sensitivity,
            "flag_given_clean": leak,
        }
    model = SensorModel(
        likelihoods, base.prior_coefficients, False, base.posterior_threshold
    )
    reliabilities = dict(
        zip(["exec", "compile", "static", "llm", "critic"], (*rels, 1.2))
    )
    return analyze(
        src,
        tst or None,
        _llm_logprobs(src) if llm else None,
        critic_backend=critic_b,
        execution=ExecutionConfig(backend),
        model=model,
        reliabilities=reliabilities,
        inference_method=method,
        auto_tests=auto_tests or annotation_contracts,
        autonomous_testing=auto_tests,
        annotation_contracts=annotation_contracts,
        contracts=json.loads(contracts_json) if contracts_json.strip() else None,
        failure_replays=2,
    )


try:
    analysis = _analyse(
        source,
        tests if run_tests else "",
        (rel_exec, rel_compile, rel_static, rel_llm),
        use_llm,
        critic_backend,
        execution_backend,
        inference_method,
        model_json,
        test_sensitivity,
        test_leak,
        auto_tests,
        annotation_contracts,
        contracts_json,
    )
except (ValueError, KeyError) as exc:
    st.error(str(exc))
    st.stop()
blocks, g, evidence, post = (
    analysis.blocks,
    analysis.graph,
    analysis.evidence,
    analysis.posteriors,
)
st.caption(
    "Model estimates concern specified behavior; they are not a proof of correctness. "
    "Default likelihoods are assumptions until independently validated."
)
st.json(g.graph["inference"], expanded=False)
if analysis.failure_validation.get("failures"):
    st.json({"failure_validation": analysis.failure_validation}, expanded=False)
if analysis.execution and any(t.outcome == "failed" for t in analysis.execution.tests):
    from pcg.project import trace_diagnoses

    st.json(
        {"executed_dependency_diagnoses": trace_diagnoses(analysis.execution)},
        expanded=False,
    )
if analysis.test_plan:
    with st.expander("Generated test plan and unsupported interfaces"):
        st.json(analysis.test_plan)
if g.graph.get("test_targets") is not None:
    with st.expander("Graph-directed checks and measured branch coverage"):
        st.json(analysis.to_dict()["graph_testing"])
if analysis.failure_diagnoses:
    with st.expander("Trace symptoms back to possible earlier causes", expanded=True):
        st.json(analysis.failure_diagnoses)
if analysis.execution and not analysis.execution.valid:
    st.warning("Test execution incomplete: " + analysis.execution.status)
    if analysis.execution.errors:
        st.code("\n".join(analysis.execution.errors)[:4000])

by_bid = {b.bid: b for b in blocks}
# review_ranking returns list[tuple(Block, BlockPosterior)]; convert to list of b.bids
ranked = [b.bid for b, _ in review_ranking(post, blocks)]


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
flagged = [bid for bid, bp in post.items() if bp.posterior < threshold]
own_fault = [bid for bid, bp in post.items() if bp.culpability > 0.5]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Blocks", len(blocks))
c2.metric("Below threshold", len(flagged))
c3.metric("Likely intrinsic defects", len(own_fault))
c4.metric(
    "Above trust threshold",
    f"{100 * (1 - len(flagged) / max(1, len(blocks))):.0f}%",
)


# --------------------------------------------------------------------------
# Probability map
# --------------------------------------------------------------------------
tab_map, tab_graph, tab_block, tab_evidence = st.tabs(
    ["Probability map", "Constraint graph", "Block detail", "Evidence ledger"]
)

with tab_map:
    import pandas as pd
    import plotly.express as px

    root_bids = set(find_root_repairs(post, g, threshold=0.5))
    repair_bids = set(repair_targets(post, threshold=0.5))

    rows = []
    for bid, bp in post.items():
        b = by_bid[bid]
        label, _ = band_of(bp.posterior)
        priority = (
            "HIGH (Root Defect)"
            if bid in root_bids
            else "MEDIUM (Secondary)"
            if bid in repair_bids
            else "LOW (Collateral)"
            if bp.posterior < threshold
            else "NONE (Clean)"
        )
        rows.append(
            {
                "block": b.qualname,
                "Intrinsic Correctness": f"{100 * (1.0 - bp.culpability):.1f}%",
                "Effective Trust": bp.posterior,
                "Repair Priority": priority,
                "own doubt": bp.culpability,
                "inherited doubt": bp.inherited,
                "importance": bp.importance,
                "risk": bp.risk,
                "band": label,
                "lines": f"{b.lineno}–{b.end_lineno}",
                "cc": b.cyclomatic,
            }
        )
    df = pd.DataFrame(rows).sort_values("risk", ascending=False)

    st.subheader("Targeted Repair & Root Defect Analysis")
    col_roots, col_affected = st.columns(2)
    with col_roots:
        st.markdown("**Root Defects (Fix these first):**")
        roots_found = [by_bid[bid].qualname for bid in root_bids]
        if roots_found:
            for r in roots_found:
                st.error(f"🔴 `{r}` — direct evidence indicates implementation bug")
        else:
            st.success("No root defects detected above threshold.")

    with col_affected:
        st.markdown("**Affected Callers (Re-verify after roots fixed):**")
        affected = [
            by_bid[bid].qualname
            for bid, bp in post.items()
            if bp.posterior < threshold and bid not in repair_bids
        ]
        if affected:
            for a in affected:
                st.warning(f"🟡 `{a}` — output unreliable due to upstream dependencies")
        else:
            st.info("No collateral caller blocks affected.")

    st.subheader("Where the doubt comes from")
    st.caption(
        "A block can look suspect for two very different reasons. **Own doubt** "
        "means its own evidence is bad — that is a repair target. **Inherited "
        "doubt** means it only looks bad because something it depends on is "
        "broken — patching it would mask the real fault. The console report "
        "and this chart keep them separate for that reason."
    )
    stacked = df.melt(
        id_vars=["block"],
        value_vars=["own doubt", "inherited doubt"],
        var_name="kind",
        value_name="doubt",
    )
    fig = px.bar(
        stacked,
        x="doubt",
        y="block",
        color="kind",
        orientation="h",
        color_discrete_map={"own doubt": "#d62728", "inherited doubt": "#ff9e4a"},
        height=max(300, 34 * len(df)),
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        legend_title="",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ranked review order (Three Core Metrics)")
    st.dataframe(
        df[
            [
                "block",
                "Intrinsic Correctness",
                "Effective Trust",
                "Repair Priority",
                "own doubt",
                "inherited doubt",
                "risk",
                "band",
                "lines",
            ]
        ]
        .style.background_gradient(
            subset=["Effective Trust"], cmap="RdYlGn", vmin=0, vmax=1
        )
        .background_gradient(subset=["own doubt", "risk"], cmap="Reds", vmin=0, vmax=1)
        .format(
            {
                "Effective Trust": "{:.3f}",
                "own doubt": "{:.3f}",
                "inherited doubt": "{:.3f}",
                "risk": "{:.3f}",
            }
        ),
        use_container_width=True,
        height=min(600, 40 + 35 * len(df)),
    )

    st.subheader("Importance vs correctness")
    st.caption(
        "The top-left quadrant is what matters: structurally important code "
        "that the framework doubts. Bubble size is own doubt."
    )
    scat = px.scatter(
        df,
        x="Effective Trust",
        y="importance",
        size=[max(0.02, v) for v in df["own doubt"]],
        color="band",
        hover_name="block",
        color_discrete_map={
            "LIKELY OK": "#17becf",
            "UNCERTAIN": "#bcbd22",
            "SUSPECT": "#ff7f0e",
            "CRITICAL": "#d62728",
        },
    )
    scat.add_vline(x=threshold, line_dash="dash", line_color="grey")
    st.plotly_chart(scat, use_container_width=True)


with tab_graph:
    import networkx as nx
    import plotly.graph_objects as go

    st.caption(
        "An edge B → A means A depends on B, so doubt flows along the arrow. "
        "Node colour is estimated output trust; size is structural importance."
    )
    try:
        pos = nx.spring_layout(g, seed=7, k=1.4)
    except Exception:
        pos = {n: (i, i % 3) for i, n in enumerate(g.nodes)}

    edge_x, edge_y = [], []
    for u, v in g.edges():
        if u not in pos or v not in pos:
            continue
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    node_x = [pos[n][0] for n in g.nodes if n in pos]
    node_y = [pos[n][1] for n in g.nodes if n in pos]
    node_c = [post[n].posterior if n in post else 0.5 for n in g.nodes if n in pos]
    node_t = [
        f"{by_bid[n].qualname}<br>P={post[n].posterior:.3f}"
        f"<br>own={post[n].culpability:.3f}"
        if n in post and n in by_bid
        else str(n)
        for n in g.nodes
        if n in pos
    ]
    node_s = [
        14 + 26 * (post[n].importance if n in post else 0.3)
        for n in g.nodes
        if n in pos
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="#bbb"),
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[by_bid[n].name if n in by_bid else n for n in g.nodes if n in pos],
            textposition="top center",
            hovertext=node_t,
            hoverinfo="text",
            marker=dict(
                size=node_s,
                color=node_c,
                colorscale="RdYlGn",
                cmin=0,
                cmax=1,
                line=dict(width=1, color="#333"),
                colorbar=dict(title="P(correct)"),
            ),
        )
    )
    fig.update_layout(
        showlegend=False,
        height=620,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


with tab_block:
    names = [by_bid[bid].qualname for bid in ranked]
    pick = st.selectbox("Block", names)
    bid = next(b.bid for b in blocks if b.qualname == pick)
    bp, blk = post[bid], by_bid[bid]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P(correct)", f"{bp.posterior:.3f}")
    m2.metric("Own doubt", f"{bp.culpability:.3f}")
    m3.metric("Inherited doubt", f"{bp.inherited:.3f}")
    m4.metric("Importance", f"{bp.importance:.3f}")

    st.markdown(
        f"**How this number was reached** — structural prior "
        f"`{bp.prior:.3f}` → after own evidence `{bp.local:.3f}` → after "
        f"constraint propagation `{bp.posterior:.3f}`."
    )
    if bp.inherited > 0.02:
        deps = [by_bid[p].qualname for p in g.predecessors(bid) if p in by_bid]
        st.warning(
            f"Most of this block's doubt is **inherited** from "
            f"{', '.join(f'`{d}`' for d in deps) or 'its dependencies'}. "
            "Fix those first, then re-verify — do not patch this block."
        )
    elif bp.culpability > 0.5:
        st.error("This block's **own** evidence is bad. It is a repair target.")

    if bp.top_reasons:
        st.markdown("**Why:**")
        for r in bp.top_reasons:
            st.markdown(f"- {r}")

    st.code(blk.source, language="python")


with tab_evidence:
    import pandas as pd

    st.caption(
        "Sensor observations and covered test outcomes. Structured test outcomes "
        "update a joint noisy-OR likelihood, so their individual log-odds entry is zero."
    )
    from pcg.inference import evidence_log_lr

    erows = [
        {
            "block": by_bid[e.bid].qualname if e.bid in by_bid else e.bid,
            "source": e.source,
            "kind": e.kind,
            "polarity": e.polarity,
            "weight": e.weight,
            "log-odds": round(evidence_log_lr(e), 3),
            "detail": e.detail,
        }
        for e in evidence
    ]
    edf = pd.DataFrame(erows)
    src_filter = st.multiselect(
        "Sources", sorted(edf["source"].unique()), sorted(edf["source"].unique())
    )
    edf = edf[edf["source"].isin(src_filter)]
    st.dataframe(
        edf.sort_values("log-odds").style.background_gradient(
            subset=["log-odds"], cmap="RdYlGn"
        ),
        use_container_width=True,
        height=min(700, 40 + 32 * len(edf)),
    )
