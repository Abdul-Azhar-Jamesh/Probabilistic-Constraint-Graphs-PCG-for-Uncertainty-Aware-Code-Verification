# Graph-directed testing

PCG now uses graph structure to construct actual test inputs, not only to rank
functions or explain a failure. Automatic testing enables it by default.

```powershell
python -m pcg.pipeline path/to/code.py --execution local --json out/result.json --html out/result.html
python -m scripts.graph_testing_evaluation
```

Local mode is for trusted code. Use the configured Docker worker for uploads.
`--no-graph-guided` keeps ordinary autonomous fuzzing for an ablation comparison.

## Three concrete methods

1. **Dependency-sliced control-flow path testing.** Build a directed CFG for
   synchronous functions, with true/false edges on supported `if` statements.
   Traverse bounded entry-to-exit paths. Use the existing program dependence
   graph's backward slice to select assignments influencing path conditions.
   This excludes unrelated operations that would otherwise block symbolic
   translation, and preserves reaching definitions across reassignment.
2. **Constraint-generated inputs.** Translate supported path expressions to Z3
   terms and solve their conjunction. This can solve interacting parameters,
   nested guards, integer arithmetic, boolean combinations, string equality and
   prefixes, and integer-list length conditions. Small pure straight-line helper
   functions have bounded summaries. Feed the solved inputs into the same
   Hypothesis checks, alongside ordinary random generation. No candidate is
   imported or executed by this planner.
3. **Coverage feedback and isolated path replay.** Measure actual worker arcs.
   Use up to six remaining case slots to target unseen predicted branch edges,
   or replay paths of functions whose fuzz checks failed. Execute these as
   separate single-input checks, so backward diagnosis has more precise coverage
   than an aggregate fuzz run. Rerun the combined suite once; only final-run
   observations enter inference. The second round does not double-count first
   round evidence.

An example is `gate = x * 3 + 2`, then `gate == 2141`, then `y - x == 278`.
The graph-guided solver generates `x=713, y=991`; changing just one argument with
the other fixed at zero cannot reach that path. Worker coverage determines whether
the predicted path actually executes.

## Evidence and course fit

The existing stateful sequence probe now also uses a method-level state dependency
graph. Writers of `self.field` connect to readers of that field; common collection
mutations and local helper summaries contribute edges. Bounded graph walks seed
producer-to-consumer call sequences, alongside random Hypothesis sequences.
For example, `open -> authorize -> load -> send` can expose a delayed failure
after earlier setup. Repeated self-dependent methods can receive four-call seeds.
The plan exports nodes, shared fields and seed sequences as
`state_dependency_graph`. Up to sixteen sequences of six steps are seeded from
256 graph-walk states. Aliases, reflection and arbitrary external mutations are
not resolved. These are possible dependencies, not inferred API-valid transitions;
state-sequence failures remain neutral clues.

Path reachability and coverage are not correctness oracles. Declared return types,
existing doctest outputs and optional behavior contracts supply checks. Exceptions
on guessed inputs remain neutral probes. An input model is labelled predicted
until executed. The existing joint Bayesian latent-defect/noisy-OR inference
remains central to the 22AIE301 probabilistic reasoning component.

Repeated generated return-type checks for one function are grouped into a single
observation family. When that family fails, passing replays cannot exonerate it.
This reduces artificial certainty from correlated generated examples; it is not
a calibrated general model of all test correlations. Other likelihood parameters
and partial-oracle sensitivities remain assumptions requiring held-out validation.

JSON includes `control_flow_graph`, `graph_testing.targets`, `graph_testing.feedback`
and unresolved paths. The dashboard and HTML report expose directed-testing
coverage. Single-input replay diagnoses can exclude branches executed only during
other fuzz inputs. Fuzz-test coverage itself remains an aggregate across examples.

## Research basis

- [DAFL, USENIX Security 2023](https://www.usenix.org/conference/usenixsecurity23/presentation/kim-tae-eun)
  uses target-relevant data dependencies to guide directed grey-box fuzzing. PCG
  adapts the principle through dependency slices; it does not reproduce DAFL's
  native-code instrumentation or claim its reported performance.
- [Constraint-guided directed greybox fuzzing, USENIX Security 2021](https://www.usenix.org/conference/usenixsecurity21/presentation/lee-gwangmu)
  considers ordered target sites and data conditions. PCG combines CFG path order
  with data constraints in a bounded Python implementation.
- [SAGE / Automated Whitebox Fuzz Testing](https://www.microsoft.com/en-us/research/publication/automated-whitebox-fuzz-testing/)
  generates inputs through symbolic path constraints and coverage-guided search.
  PCG uses static bounded path solving, not SAGE's full dynamic symbolic tracing.
- [Z3](https://github.com/Z3Prover/z3) supplies the SMT solver; Hypothesis continues
  supplying ordinary property fuzzing and shrinking. Graph-generated explicit
  examples are not themselves Hypothesis-shrunk failing examples.

## Bounds and unsupported behavior

At most 32 completed paths per function and 128 per module are solved, with
150 ms per solver query. Traversal is bounded to 512 states per function and 2048
per module; paths are at most 64 nodes. Integer inputs are bounded to +/-10000,
strings to length 40, and generated integer lists to length 20 with zero elements.
List-content constraints are unsupported. These limits describe search coverage,
not proof of global infeasibility.

Floats, arbitrary heap mutations, general recursion, external functions and
complex loop/exception flow are not symbolically solved. Such compound CFG nodes
are opaque and explicitly unresolved; Hypothesis/probes remain available. Unsupported
or infeasible-within-bounds paths are reported. A satisfying SMT model can still
miss its predicted target because of unmodelled initialization or runtime behavior.
Actual coverage is authoritative. Solver use is serialized across dashboard
threads because the planner uses Z3's default context.

The evaluation script compares identical sources, seeds and nominal case budgets.
Graph mode adds solved explicit examples and may run a second worker round, so
invocation counts and elapsed time differ. Its seven curated examples are
mechanism checks, not evidence of real-world accuracy or complete verification.
