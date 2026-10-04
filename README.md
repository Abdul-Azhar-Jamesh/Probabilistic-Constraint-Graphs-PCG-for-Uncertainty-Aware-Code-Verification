# Probabilistic Constraint Graphs

PCG is a Python code-review research tool for **22AIE301 Probabilistic Reasoning**. It combines a code dependency graph with a latent-defect Bayesian model, coverage-backed test evidence, and separate estimates of intrinsic defects and downstream output trust.

Scores concern specified behavior and available evidence. They are not proofs of correctness. Default observation probabilities are documented assumptions; fitted models require independent evaluation before deployment claims.

## Install and run

```sh
python -m pip install -e ".[dev,app]"
python -m pcg.pipeline demo/candidate.py --json out/review.json
python -m pcg.pipeline demo/candidate.py -t demo/test_candidate.py --execution local --json out/review.json
streamlit run app.py
```

`local` executes **trusted source and tests only**. CLI and dashboard execution are disabled by default. To analyse untrusted code, build the isolated worker and select Docker:

```sh
docker build -f deploy/Dockerfile.worker -t pcg-worker:latest deploy
python -m pcg.pipeline demo/candidate.py -t demo/test_candidate.py --execution docker
```

The worker has no network, host mounts, application secrets or writable root filesystem, and has CPU, memory, process and time limits. Docker availability is mandatory; failures never fall back to local execution. Set `PCG_PUBLIC_DEPLOYMENT=1` on a hosted dashboard to prohibit local execution. See [deployment guidance](docs/deployment.md) before hosting.

## Probabilistic model

Each code block has a binary latent defect variable. Structural priors and aggregated compiler/static/optional LLM observations supply local likelihoods. Covered test outcomes supply shared noisy-OR factors. Exact enumeration handles small connected components; larger components use seeded multi-chain Gibbs sampling with mixing diagnostics.

The original code graph remains central: dependency paths define downstream trust queries over the **joint** defect posterior. Shared ancestors are counted once, and recursive call graphs do not create invalid Bayesian-network cycles.

Reports expose defect probability, output trust, inherited doubt, execution status, inference method and Monte Carlo standard error. Optional next-test candidates are ranked by estimated information gain per second.

## Validation and test strength

```sh
python -m pytest -q
python -m ruff check pcg tests scripts deploy app.py
python -m pcg.validation --demo --execution local
python -m pcg.validation --dataset data/cases.json --execution docker
python -m pcg.pipeline demo/candidate.py --model out/validation/sensor_model.json
```

The demo benchmark is synthetic. User datasets must explicitly separate whole repositories into training, validation and test splits. Runtime evaluation compares the full graph model, fusion without dependency edges, tests alone and static analysis alone. Thresholds are selected on validation data only. Model files are explicit and versioned; legacy fitted weights are never loaded automatically.

For a candidate with a passing baseline suite, `--mutation-audit 20` measures test strength. Surviving mutations are not automatically declared equivalent. Repairs must preserve every previously passing test identity, retain the entire suite, and resolve at least one failure.

## Project layout

- `pcg/`: extraction, graph, observation model, inference, workers, reports and evaluation.
- `app.py`: the dashboard, using the same pipeline as the CLI.
- `tests/`: analytical, property-based, execution, repair-contract and UI checks.
- `deploy/`: isolated execution image and entrypoint.
- `demo/`: intentionally defective example and tests.
- `critic_calibration/`: separate optional LLM-critic experiment and its datasets.
- `experiments/archive/`: historical results from the previous model; inactive.
- `presentation/`: preserved original presentation artifact.
- `out/`: generated results, excluded from version control.

See [model mathematics and course mapping](docs/methodology.md), [evaluation](docs/evaluation.md), and [implementation architecture](docs/architecture.md).

## Current scope

A candidate module is analysed at function/method/segment granularity (maximum 512 blocks). Additional Python support files can be supplied through `ExecutionConfig.files`, but dependency extraction and defect localization cover the candidate module. Dynamic dispatch, cross-module dependency inference and arbitrary project environments are outside the current graph model. Install required dependencies into a dedicated worker image.

The automated Docker integration check requires a running daemon and built image. No successful Docker execution is implied by passing local tests.
