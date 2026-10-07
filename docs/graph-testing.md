# Graph-guided testing and earlier-cause diagnosis

PCG can now produce executable checks for a new Python module without importing it in the analysis process. The planner reads its AST, supported annotations, literal defaults and numeric branch boundaries. It explores empty/singleton containers, numeric zero and nearby values, strings and small argument interactions. A fixed case budget and round-robin allocation prevent one function consuming the entire plan. Code-graph reachability prioritizes functions with downstream dependents.

## Run the supplied example

```powershell
python -m pcg.pipeline demo/graph_candidate.py --auto-tests --annotation-contracts --contracts demo/contracts.json --execution local --json out/graph-review.json --html out/graph-review.html --write-generated-tests out/graph-generated-tests.py
```

Local execution is for trusted code. Use `--execution docker` with a validated worker for uploaded code. CLI and public dashboard execution remain disabled by default. Test generation alone does not enable execution. Without execution, the report still contains a test plan and joint-posterior information-gain recommendations for checks with an oracle; coverage forecasts and unit runtime costs are explicitly assumptions.

The dashboard exposes boundary generation, optional return-type contracts and the same JSON contracts. Failure diagnoses are visible there and in the HTML/JSON report.

## What counts as evidence

| Check | Oracle | Effect on inference |
|---|---|---|
| Generated boundary probe | None | Neutral; records execution and exceptions for investigation |
| Declared return-type check | Supported annotation, explicitly enabled | Valid returned values produce coverage-linked observations; input exceptions are skipped because accepted input domains are unspecified |
| Example case | Supplied expected result or expected exception | Coverage-linked noisy-OR test factor |
| Metamorphic example | Supplied relation between two outputs | Coverage-linked noisy-OR test factor |
| Reusable property | Supplied invariant applied across generated inputs | Coverage-linked factor for valid returned outputs; unspecified input exceptions are skipped |

PCG never takes the candidate's own output and turns it into the expected output. Passing a probe does not increase trust. Importing a function definition does not count as exercising its body. Probe failures and oracle failures are separated in the report and localization statistics.

Partial oracles use a lower assumed sensitivity for the broad latent-defect variable: return-type checks use 0.25 times the base test sensitivity, and properties/metamorphic relations use 0.5 times it. Their passing outcomes therefore provide weaker positive evidence than specified-output tests. These discounts are visible in inference diagnostics and are modelling assumptions, not learned probabilities. A regression case verifies that passing type checks cannot exonerate code that fails a supplied expected-value test.

Repair validation and mutation auditing also require improvement or detection by an oracle-backed check. An exception disappearing from a probe cannot validate a patch or kill a mutant. Segment targets are now eligible for bounded AST edits in their own line ranges.

Reusable properties are selected deliberately; PCG cannot infer the intended behavior of arbitrary software from its implementation. A sorting routine can declare `permutation`, `sorted` and `idempotent` once, and PCG generates input cases. These properties are not valid for every function.

Supported property names: `nonnegative`, `length_preserving`, `sorted`, `permutation`, `input_unchanged`, `idempotent`. Length/permutation checks compare against the first input. Idempotence assumes a function accepts its own output as a single input. Permutation currently uses sorted comparison and requires mutually orderable elements.

Example explicit cases:

```json
[
  {"function": "score", "args": [2], "expected": 4},
  {"function": "score", "args": [-1], "raises": "ValueError"},
  {"function": "score", "args": [1], "other_args": [2], "relation": "nondecreasing"},
  {"function": "normalize", "property": "idempotent"}
]
```

Explicit metamorphic relations support `equal`, `nondecreasing` and `idempotent`. Keyword inputs use `kwargs`/`other_kwargs`. Generated cases live in the reserved `test_pcg_` namespace.

## Why an earlier line can be found

The additional statement dependence graph records reaching definitions, control predicates, loop-carried dependencies, call arguments/returns, simple function aliases and constructor-to-method bindings. Straight-line reassignment kills the previous definition; branches retain possible definitions from both paths. Subscript writes and common container mutations connect changed state to later reads.

For each failed test, PCG seeds a backward slice at the recorded exception frame, or at executed returns/observable writes for an assertion outside the candidate. It filters traversal to actually executed lines and separately recorded module initialization. A denominator created on line 12 can therefore appear as a possible origin of a division error on line 18. Unrelated assignments are excluded when the dependence analysis can establish that separation. Ochiai ranks the remaining lines using passing/failing spectra; it is a localization score, not a probability of causation.

Initialization during collection is reported separately instead of crediting every test with executing it. This retains earlier module-global origins for diagnosis. Such origins are not automatically added as extra test factors.

The block graph projects these statement dependencies into segment/function relationships and keeps the original dependency-to-dependent direction. Short multi-stage functions can now be segmented on complete top-level statements; long compound statements remain intact for repair. A function's prior defect budget is partitioned across its header and segments so simply slicing it more finely does not create extra independent prior defect risk. Segment containment has strength 1; other couplings remain modelling assumptions.

Parameter-flow edges remain in the diagnostic statement graph. They are not projected into global callee correctness, and a function header is not linked back to its own segments. This avoids inventing caller/callee or containment cycles in otherwise straight-line code.

## Course connection and limits

The newer [autonomous testing workflow](autonomous-testing.md) adds shrinking
property fuzzing, existing literal doctest discovery and supported object sequence
probes. The CLI plans these by default, without a separate test file.

22AIE301 concepts remain explicit: binary latent defects, observation likelihoods, exact enumeration/Gibbs inference, explaining away, sensor learning and decisions using expected information gain. Program slicing supplies structure and evidence attribution; it is not a substitute for probabilistic inference. Previously fitted models need evaluation with the revised segmentation and graph.

The slice is an executed **static may-depend** analysis, not a value-level dynamic trace or proof of the unique root cause. Shared-call contexts, reflection, aliasing through arbitrary objects, external modules, complex exceptions and dynamic dispatch can leave ambiguous or missing dependencies. Unresolved calls are listed. Automated input generation targets top-level synchronous functions with supported signatures; methods, async interfaces, decorators, variadic APIs and external-service behavior require explicit integration tests or an adapter. Unsupported domains are reported rather than silently tested with made-up business assumptions.

A generated suite is not sufficient to certify arbitrary real-world code. Use independently specified contracts, existing repository tests, mutation auditing and held-out real defects. The new regression suite checks earlier assignments, reassignment, branch exclusion, loop-carried values, alias dispatch, import-time origins, side effects, oracle provenance and generated contract failures.
