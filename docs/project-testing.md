# Project analysis

```powershell
python -m pcg.project path/to/project --execution local --cache out/cache --json out/project.json --html out/project.html
python -m pcg.project path/to/project --execution docker --worker-image project-worker:latest
python -m pcg.project path/to/project --execution local --test tests/test_api.py::test_result --python path/to/venv/python.exe
```

Local execution is for trusted projects. Dependencies must already be installed in the selected Python environment or Docker image. Execution is disabled by default. The `pcg` CLI also routes a directory input to project analysis. The dashboard accepts a ZIP with the same project engine.

The file tree, package initializers, conftest, pytest configuration and UTF-8 resources are preserved. Source, generated checks and discovered existing tests run together. Skipped resources, unsupported signatures, unresolved calls, deadlines and missing dependencies remain visible. No missing dependency is treated as a passing test.

## Simple pipeline

**Project input → blocks and dependency/control-flow graphs → automatic inputs and existing tests → worker execution → per-call traces and coverage → target uncovered branches → trace possible earlier causes → Bayesian update → ranked report.**

Failure handling now adds **repeat in fresh workers → mark stable/flaky/inconclusive → reduce a stable reproducer while preserving its candidate symptom** before final inference. CLI and dashboard use two replays by default. Project CLI budgets are `--failure-replays 2 --minimize-trials 6 --validation-time-budget 60`; zero replays explicitly disables validation. The single-source library API keeps replay opt-in to avoid repeatedly validating internal mutation/repair experiments. Replays never add duplicate Bayesian factors; flaky/inconclusive selected failures contribute neutral diagnostics instead of correctness evidence. Order-dependent failures can also appear unstable when replayed without unrelated tests.

The reducer removes unnecessary setup/actions and shrinks literals while retaining assertions and oracle provenance. For supported primitive return contracts, captured trace inputs can become a concrete replay test. A trial must still fail at the same candidate symptom with the same exception category. The report contains the accepted test source and replay target. Original project files and tests stay intact; reduction is bounded, not a claim of globally smallest input or unique causal proof.

Adaptive project allocation reserves part of the generated-case budget, then ranks modules by expected information gain under the joint posterior, unseen branch edges and observed execution cost. Oracle-free probes get zero correctness information gain. Coverage forecasts and costs are estimates; allocations and unused budget are reported. Beta(1,1) replay summaries estimate failure frequency under an exchangeable-run assumption, not defect probability or guaranteed reliability.

Graph-derived inputs seed Hypothesis fuzzing and shrinking. Reserved case slots replay unseen branches after coverage feedback. Stateful probes seed producer/consumer method sequences. Existing regression assertions, literal examples and declared return types provide distinct correctness oracles; unlabelled exceptions remain neutral probes. Neither the graph nor random input generation can infer every intended business rule.

Statement tracing records versions of executed variable definitions, object fields, enclosing controls and called function results. It can follow an earlier assignment, another module's return, or a previous method's field write into the symptom. It excludes unrelated executed assignments in the tested cases. Trace truncation and conservative alias handling are disclosed; a dependency slice identifies possible causes, not a proven unique defect.

The model retains binary latent defects, structural priors, sensor likelihoods, shared noisy-OR test factors, Bayes conditioning and exact/Gibbs inference. Repeated return-type checks share an oracle family. Default probabilities remain assumptions; repository-separated evaluation is needed before calibration claims. These parts connect directly to 22AIE301 probability, conditional independence, graphical models and approximate inference.

Cache keys include source/resources, tests, execution settings, worker implementation and dependency or image fingerprints. Invalid executions are never cached. Reports include static findings, test provenance, captured bounded inputs, executed slices, graph edges, inference diagnostics and limitations. Whole test suites continue to run; the graph methods supplement them.

Limits: Python source only, 1000 snapshot files, 32 MB snapshot, 2 MB per file, 2048 blocks and 500 generated cases maximum. Binary resources are skipped. Symbolic solving supports a narrower subset than the conservative project CFG: loops/try flow is graphed but not generally solved. Reflection, arbitrary dynamic dispatch, services, native extensions and concurrent behavior need further environment-specific support.

Reduction also requires the original captured dependency locations to remain in the trial's backward event-graph slice. Expected-value assignments used by assertions are protected. When a trace is unavailable, the report explicitly labels reduction as symptom-only validation. Captured slices can be conservative or truncated; preserving them still does not establish a unique cause. Failure validation defaults to at most eight selected failures and a 60-second total deadline; budget exhaustion is recorded.
