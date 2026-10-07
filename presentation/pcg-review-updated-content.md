# PCG — revised Canva content

Prepared 6 October 2026 against `temp.pdf` (14-slide review template), current implementation and `out/validation/report.json`. This is the prepared content; the remote Canva presentation has not been updated because Canva authentication is unavailable.

## 1. Title of the Project
**Probabilistic Constraint Graphs**
Uncertainty-Aware Code Verification

22AIE301 · Probabilistic Reasoning · Semester 5 · Mid-Sem Review

Abdul Azhar — CB.SC.U4AIE24301  
Deepak Prabhu — CB.SC.U4AIE24312  
Pranesh M — CB.SC.U4AIE24345  
Sandhiya D — CB.SC.U4AIE24353

Visual: dependency graph with one suspected defect and highlighted dependent nodes. Preserve existing team details; no group number was available in the source presentation.

## 2. Outline of the Presentation
- Introduction
- Literature review and research gaps
- Objectives
- Methodology: model, inference, learning and decisions
- Progress and evaluation
- Completion timeline
- References

## 3. Introduction
**Passing tests provide evidence—not a guarantee of correctness.**

- Hidden logic defects can affect several dependent functions.
- PCG combines observed evidence with an explicit probabilistic model.
- Reviewers see likely defect locations, inherited risk and useful next tests.

Practical use: assist review of a candidate Python module before accepting a change.

Speaker note: Current localization is within one candidate module. The system supports human review; its probabilities are conditional on the model and available tests.

## 4. Literature Review
| Work | Contribution | Relation to PCG |
|---|---|---|
| Sharma & David, 2025 [1] | Estimates correctness-related uncertainty from generated-code samples; supports abstention | Motivates uncertainty-aware review; PCG targets block-level diagnosis with execution evidence |
| Mündler et al., 2025 [2] | Uses type-constrained decoding to improve well-typed code generation | Complementary generation-time checks; PCG studies remaining defects after generation |

Speaker note: These are complementary approaches, not experimentally compared baselines. Type correctness alone does not establish satisfaction of a program's specification.

## 5. Takeaways / Research Gaps from Literature
**Research question:** Can test evidence and code dependencies improve uncertainty-aware defect localization?

- Move from whole-answer uncertainty to actionable block-level review.
- Separate a block's own defect probability from risk inherited through dependencies.
- Account for missing coverage, shared tests and uncertain failure causes.
- Evaluate both detection quality and probability quality on held-out programs.

Speaker note: These motivate this project's scope; no claim of being the first system to address these problems is made.

## 6. Objectives
1. Extract a code dependency graph from Python source.
2. Collect traceable compiler, static-analysis and coverage-linked test evidence.
3. Infer intrinsic defects and dependency-aware trust using probabilistic reasoning.
4. Evaluate precision, recall, F1, Brier score, ECE and false-trust rate; compare ablations.

Visual: four numbered cards; connect each objective to its measurable output.

## 7. Methodology — Overall
**Source → AST blocks → dependency graph → evidence → posterior → review / test / repair**

Two connected representations:
- **Code graph:** calls, dataflow and sequence dependencies; recursion may create cycles.
- **Bayesian observation model:** latent block defects generate sensor and test observations.

Coverage links each test to the blocks it actually executed. Joint posterior queries carry defect risk through the code graph.

Speaker note: The cyclic code graph is not itself a Bayesian network. Keep the code graph visible alongside the separate defect-to-test observation graph. Shared ancestors are counted once; cycles cannot repeatedly amplify impact.

## 8. Methodology — PR Inference
Let Dᵢ = 1 indicate an intrinsic defect; Aⱼ is the coverage set of test j.

**P(Tⱼ = pass | D) = (1 − leak) ∏ᵢ∈Aⱼ (1 − sensitivity × Dᵢ)**

- Bayesian conditioning combines priors, sensor likelihoods and test outcomes.
- Exact enumeration: components with at most 14 defect variables.
- Larger components: seeded multi-chain Gibbs sampling; report R-hat and sampling error.
- Shared failed tests permit explaining away between alternative defect causes.

Speaker note: Leak models unexplained failures; sensitivity models imperfect defect detection. Both are assumptions until supported by representative data. Defect components are formed by shared tests. Numerical convergence diagnostics do not validate model assumptions. Effective trust is E[∏ᵢ(1 − aᵢᵦDᵢ) | E], where aᵢᵦ is the strongest dependency-path coupling to block b, including aᵦᵦ = 1. This uses joint states rather than multiplying marginal probabilities.

## 9. Methodology — PR Learning
**Learn parameters; keep the extracted code structure explicit.**

- AST rules determine graph structure; dependency couplings remain modelling assumptions.
- Estimate sensor probabilities with Beta(1,1) smoothing.
- Optionally learn structural defect priors using logistic regression.
- Split by repository/program family: train parameters → select threshold on validation → evaluate once on test.

Speaker note: Versioned sensor models are opt-in. Duplicate source content across splits is rejected. The small synthetic fit did not outperform defaults and is not automatically promoted. The pooled test sensor fit is approximate when tests cover multiple defects.

## 10. Methodology — Decision, Intervention or What-If Scenarios
**Choose the next action from uncertainty.**

- Rank review targets using defect probability and dependency-aware trust.
- Rank candidate tests by expected information gain per unit runtime.
- Distinguish intrinsic suspicion from inherited dependency risk.
- Accept a proposed repair only after rerunning preserved tests with regression checks.

Speaker note: Test selection uses I(D; T_next | evidence) / estimated cost and requires predicted coverage. Changes to model assumptions are sensitivity analysis, not identified causal interventions. Repair validation requires execution; it is not a proof that every behavior is correct.

## 11. Methodology — Combining with Modern AI
**AI proposes code; probabilistic reasoning organizes the evidence.**

- Candidate input may be human-written or AI-generated Python.
- Optional genuine token log-probability or critic findings enter as sensor evidence.
- Missing AI evidence contributes nothing; complexity is not relabelled as model confidence.
- The same verification pipeline evaluates proposed repairs.

Speaker note: Do not claim an always-running language model, a fixed one-second inference time, or calibrated AI confidence. Core graph and Bayesian inference work without an AI sensor. Explain clearly which optional sensor is enabled during the live demonstration.

## 12. Progress in terms of Project Objectives
**Implemented:** Python project discovery, cross-module dependency/CFG graphs, per-call executed dependency tracing, graph-seeded Hypothesis tests with shrinking, coverage-feedback replay, stateful probes, static bug rules, exact/Gibbs Bayesian inference, execution caching, ZIP dashboard input, stable-failure reduction, replay-based flaky checks and adaptive Bayesian test budgets.

Latest checks: **153 passed; 2 Docker checks skipped**. Ruff and mypy pass. Docker daemon is unavailable locally; container execution remains unverified here.

| Evaluation | Observed result | Scope |
|---|---|---|
| Seven-program graph comparison | Five planted return-contract defects detected; two controls without oracle failures | Curated synthetic mechanism check; graph mode spends extra executions |
| BugsInPy three-repository compatibility replay | Upstream regression fails buggy revision and passes fixed revision | tqdm, HTTPie and Luigi; extracted compatibility cases, not original environments |
| Automatic probes for tqdm-1 | No confirmed correctness finding without regression assertions | Honest missing-oracle limitation |
| Project regressions | Cross-file earlier assignment, previous field write, rare branch and cache invalidation checked | Correctness regression suite |

Speaker note: Bayesian variables and shared noisy-OR observations remain the probabilistic core. Default likelihoods are assumptions. These results do not establish production accuracy or calibration. Static slices are conservative; executed traces provide more specific possible causes. Existing tests remain alongside generated checks. Legacy calibration artifacts and disconnected research utilities were removed.

## 13. Weekly Timeline for Project Completion
**Proposed next four weeks — relative to the next project checkpoint**

| Week | Deliverable | Acceptance evidence |
|---|---|---|
| 1 | Independently labelled real-repository cases | Confirmed defects/fixes; repository-separated splits |
| 2 | Broader evaluation and calibration study | Ablations, false-trust rates and repository-level uncertainty intervals |
| 3 | Deployment validation | Docker integration run, resource limits and realistic test environments |
| 4 | Review demonstration and final report | Reproducible graph → evidence → inference → repair walkthrough |

Speaker note: These are proposed milestones, not completed work or committed calendar dates. Execution is disabled by default; public mode rejects local execution. Deployment should follow validation in the intended environment.

## 14. References
[1] Sharma, A. & David, C. (2025). *Assessing Correctness in LLM-Based Code Generation via Uncertainty Estimation.* https://arxiv.org/abs/2502.11620

[2] Mündler, N. et al. (2025). *Type-Constrained Code Generation with Language Models.* https://arxiv.org/abs/2504.09246

[3] Amrita Vishwa Vidyapeetham. *B.Tech CSE (AI) curriculum: 22AIE301 Probabilistic Reasoning.* https://intranet.av.amrita.edu/download/DeanEngg/Curriculum_Syllabus/Undergraduate_Programs/B_Tech_02/241_B_Tech_CSE_AI_Curr_and_Syllabus.pdf

Project evidence: `pcg/project.py`, `pcg/project_graph.py`, `pcg/runtime_trace.py`, `pcg/probabilistic.py`, `pcg/validation.py`, `out/graph-testing-evaluation.json`, `out/real-bug-benchmark/summary.json`.

## Canva layout specification
- Retain the template's 14-slide order and academic section names. Use the landscape slide area shown inside each printed PDF page; do not reproduce the surrounding printer margins.
- Preserve the existing deck's brand palette where readable. Use one dark text color, one accent and a light background with strong contrast.
- Use consistent title placement, generous margins and short visible text. Put mathematical caveats and evaluation details into presenter notes where supported.
- Use editable shapes for graphs and the pipeline, with explicit arrow direction and labels. Avoid decorative stock imagery.
- Give inference and results the clearest visual hierarchy. Format equations separately from explanatory bullets; align numeric table columns.
- Keep source numbering on the literature slide and full references at the end.
- Inspect all 14 slide previews for clipping, cramped text, unsupported symbols and contrast before saving.
