# Evaluation and acceptance checks

## Automated checks

```sh
python -m pip install -e ".[dev,app]"
python -m pytest -q
python -m ruff check pcg tests scripts deploy app.py
```

Tests cover analytical Bayes calculations, explaining away, exact/Gibbs agreement, posterior correlations, duplicate evidence, cyclic and diamond dependencies, class-method resolution, request isolation, actual test coverage, harness failures, skipped and parametrized tests, deadlines, support files and repair regression contracts. Property-based checks verify inference monotonicity and independent contracts for bundled reference programs. Streamlit tests check startup and public-mode execution policy.

The Docker test is opt-in locally (`PCG_TEST_DOCKER=1`) and enabled in its CI job after building the image. Local tests do not establish Docker execution success.

## Runtime benchmark

```sh
python -m pcg.validation --demo --execution local
python -m pcg.validation --dataset data/cases.json --execution docker
```

The bundled demo uses six small synthetic program families. It demonstrates the evaluation machinery; it is not evidence of production generalization. Surviving and timed-out mutations are recorded as excluded/inconclusive, not mislabelled as equivalent or assigned artificial confidence.

User dataset JSON is a list of cases:

```json
[
  {
    "name": "unique-case-name",
    "repository": "repository-or-program-family",
    "split": "train",
    "source": "def add(a, b):\n    return a + b\n",
    "tests": "from candidate import add\ndef test_add():\n    assert add(1, 2) == 3\n",
    "buggy": [],
    "unreliable": []
  }
]
```

Provide all three splits and multiple representative repositories. All versions of a repository must share a split. Duplicate source content across splits is rejected. Labels refer to candidate-module qualified names, and intrinsic defects must also be labelled unreliable. Ground truth for real defects must come from specifications and confirmed fixes, not PCG's own predictions. The example above illustrates one row, not a complete benchmark.

Training estimates sensor probabilities with Beta(1,1) smoothing and optionally fits logistic structural priors. Validation selects the final runtime trust threshold. The test split is evaluated only after those choices are fixed. Models remain marked uncalibrated: measured ECE on a small dataset does not establish deployment calibration.

Reports compare the full graph model, fusion without dependency edges, tests-only and static-only variants. Each variant uses the same held-out cases; the trust threshold is the full model's fixed validation threshold. Report defect/trust precision, recall, F1, Brier, ECE, false-trust rate and top-three review precision. A separate default-model comparison helps identify regressions from fitting.

Before real-world release, extend this benchmark with independently labelled repositories, bootstrap uncertainty over repositories, realistic test-environment images, flaky-test measurements and operational latency/resource measurements. Do not promote a model because a synthetic result improved.

## Historical results

`experiments/archive/` preserves the previous fitted model and evaluation output for traceability. They evaluated a different prediction procedure and are never automatically loaded by runtime inference. Optional `pcg.fit_weights` and `pcg.build_training_set` remain feature-model research utilities; their outputs are separate from current calibration.
