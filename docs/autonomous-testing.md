# Automatic testing of new Python inputs

The CLI automatically plans tests for a new source file. No handwritten test file
or JSON contract is required. The dashboard enables automatic planning by default.
Execution still uses the chosen local, Docker or disabled worker.

```powershell
python -m pcg.pipeline path/to/program.py --execution local --json out/program.json --html out/program.html
```

Local execution is for trusted programs. Uploaded code should use `--execution docker`.
`--no-autonomous` restores explicit test planning. `--mutation-audit 0` disables the
automatic three-mutant CLI audit. Larger explicit limits remain available.

## How the automatic checks work

The [graph-directed testing stage](graph-directed-testing.md) now adds actual CFG
path solving with dependency slices and worker coverage feedback. This supersedes
the limited affine-only connection described below; affine seeds remain a fallback.

- Existing AST data/control/call dependencies prioritize functions and expose
  earlier assignments influencing branch conditions. Bounded affine inversion
  seeds numeric branches through simple local arithmetic; it is not a general
  SMT solver and does not solve arbitrary paths.
- Hypothesis explores supported primitive/container annotations, or uncertain
  input hypotheses when annotations are absent. Each function check explores up
  to 80 examples, subject to the worker timeout, and shrinks discovered failures.
- Supported return annotations become partial output contracts automatically.
  Existing literal doctest call/result pairs supply explicit expected values.
  Candidate outputs are never copied into expected answers.
- Supplied reusable properties also receive fuzzing and shrinking. They are
  optional; unknown business properties are not invented from function names.
- Supported classes with default construction receive sequences of up to twelve
  public, synchronous method calls, with up to sixty examples and sequence
  shrinking. This explores state transitions but has no invented state invariant.
- Exceptions from uncertain input domains remain neutral probes. An annotation
  check with no accepted inputs is skipped, rather than credited as passing.
- A passing oracle-backed CLI baseline receives a bounded mutation audit. Surviving
  mutations trigger up to 160 examples per generated check and a revalidated
  original baseline. The report lists undetected changes as weak spots; survivors
  are not automatically called bugs or equivalent programs.

## Probabilistic reasoning

The existing joint Bayesian latent-defect/noisy-OR model remains responsible for
posterior inference. Fuzz examples are grouped into one test observation per
function and check, rather than eighty independent confirmations. Partial type
and property checks retain their weaker assumed sensitivity. Probe failures,
shrinking and mutation scores are diagnostic/test-quality information, not
independent correctness factors. Related distinct tests can still be correlated;
these likelihood assumptions require real dataset calibration.

The graph supplies dependency structure and backward failure slices. Traceback
and coverage localize candidate earlier causes. Fuzz-test coverage is the union
across explored inputs, not a precise trace of only the shrunk failing input;
reported slices are conservative and can contain unrelated explored branches.

## Inspecting results

JSON `test_plan.cases` records input hypotheses, explicit numeric seeds, branch
dependency slices and check provenance. `execution.tests[].failure_details`
includes Hypothesis's minimized example. `failure_diagnoses` reports possible
earlier causes. `automatic_validation` distinguishes detected oracle failures,
bounded oracle checks, and absence of a correctness oracle.

This is bounded automatic fault discovery for one Python module. Async interfaces,
generators, custom constructor inputs, arbitrary dependencies, persistent services
and full application integrations are not automatically supported. No algorithm
can derive an unspecified business requirement from arbitrary source alone.
The workflow needs no new manual tests to run, but cannot guarantee semantic
correctness for a program lacking a usable specification. Run-level timeouts
remain inconclusive and do not prove a particular function is defective.

## Established methods consulted

- [Hypothesis](https://hypothesis.readthedocs.io/en/latest/): property-based
  generation and failure shrinking, used directly inside the worker.
- [Pynguin](https://pynguin.readthedocs.io/latest/dev/overview.html): automated
  generation and mutation-based assertion assessment. Its coverage-driven search
  motivates the graph prioritization; its engine is not bundled into PCG.
- [CrossHair](https://crosshair.readthedocs.io/en/latest/how_does_it_work.html):
  symbolic execution with Z3 and explicit contracts. Its engine is not integrated.
  PCG's new static CFG solver uses Z3 directly and supports a bounded expression
  subset; it is not a general symbolic execution engine.

Regression fixtures cover previously unseen arithmetic, collections, literal
documentation contracts, guarded division and object lifecycles. These are smoke
and behavior regressions, not a real-world accuracy benchmark.
