# Probabilistic model and 22AIE301 alignment

## Variables and observations

For each extracted code block i, D_i is binary: 1 denotes an intrinsic defect relative to the specified behavior. A structural model supplies P(D_i). Fitted priors use the original logistic link; default complexity coefficients are assumptions.

Compiler, static-analysis, real token-log-probability and critic findings act as sensor observations. Each enabled sensor has Bernoulli likelihoods P(flag|D_i=1) and P(flag|D_i=0). The strongest observation from each source is retained to limit double counting. Severity tempers the likelihood (a power likelihood); this is an explicit modelling assumption. A missing sensor contributes no evidence. There is no synthetic LLM confidence derived from complexity.

Compilation failure is an observed hard failure of the candidate module and caps trust independently of learned sensor weights.

## Test likelihood and nuisance failures

Test j has an observed outcome T_j and a coverage set A_j. The worker records actual executed lines and branches, including test identities and setup/call/teardown outcomes.

For a valid test outcome, the noisy-OR likelihood is:

    P(T_j=pass | D) = (1 - leak) * product_{i in A_j}(1 - sensitivity * D_i)
    P(T_j=fail | D) = 1 - P(T_j=pass | D)

`leak` marginalizes a nuisance failure cause, representing mistaken assertions and unexplained failures. `sensitivity` represents the probability an exercised defect manifests on that test. Both are assumptions until learned from representative data. The pooled sensor fit is an approximation when tests exercise several defects.

The joint conditional model is proportional to the product of local defect priors and observed test likelihoods. Distinct tests are assumed conditionally independent given defects and nuisance probabilities; identical coverage/outcome likelihoods are conservatively collapsed. This can underweight distinct assertions on the same path, and remains a limitation.

Tests attach only to blocks actually executed. Segment line ownership prevents counting both an umbrella function and its segment as independent candidate defects for the same execution. Missing dependencies, setup errors, timeouts and no-test runs are reported separately and do not become fabricated defect evidence.

## Inference

`condition_defects` builds connected components induced by shared tests. Independent components factorize.

- Exact enumeration sums normalized joint state probabilities for components of at most 14 variables by default.
- Larger components use seeded multi-chain Gibbs updates from the joint conditional distribution.
- Reports include the inference method and a basic between/within-chain R-hat diagnostic. This diagnostic alone is not a guarantee of convergence.
- Trust-query sampling error uses approximate batch means. Zero uncertainty for exact inference means zero numerical sampling error, not certainty about the assumptions or program.

A failed shared test couples possible causes and permits explaining away: evidence of a defect in one function can reduce suspicion of another. This is actual backward probabilistic inference, replacing the old ineffective reverse-support update.

## Code graph and trust query

The graph now includes projected statement data/control dependencies, simple call aliases, and loop-carried definitions. Complete-statement segments can also localize short multi-stage functions. A function's baseline prior is partitioned across its header and segments rather than multiplied as a new defect budget per fragment. Containment is an identity relation with coupling 1. The extra graph structure and allocation are explicit modelling assumptions requiring re-evaluation of old fitted parameters.

The directed code graph remains dependency-to-dependent. It may contain recursive cycles; it is not itself labelled a Bayesian network.

For target block b, let a_ib be the product of coupling strengths along the strongest dependency path from i to b, with a_bb=1. Each reachable intrinsic defect is counted once. Cycles cannot repeatedly amplify impact.

    trust(b) = E[product_i(1 - a_ib * D_i) | observed evidence]

The expectation is evaluated over joint posterior states, preserving correlations from shared tests. It is not the product of marginal correctness scores. Couplings and strongest-path activation are explicit assumptions; they do not represent proven semantic execution probabilities.

`culpability` is P(D_b=1|E). `local` is intrinsic correctness after all evidence. `posterior` is effective output trust. `inherited` is the nonnegative difference between intrinsic correctness and effective trust.

## Decision support

See [graph-guided testing](graph-testing.md) for generated boundary probes, opt-in annotation checks, reusable properties, example contracts and failure slices. Oracle-free probes remain neutral; executed static slices identify possible earlier origins without manufacturing new independent evidence. The generated-check recommendations use joint-posterior information gain, labelled static coverage forecasts and an assumed unit cost.

A candidate next test has a predicted coverage set and an estimated runtime cost. Expected information gain is I(D;T_next|E), estimated using posterior draws and the conditional test-outcome entropy. Candidates are ranked by information gain divided by cost. Predictions depend on the specified coverage and likelihood assumptions.

## Course connections

The published 22AIE301 syllabus includes uncertain knowledge representation, Bayesian/Markov networks, exact and approximate inference, message passing, MCMC and decision networks.

| Course topic | Implemented demonstration |
|---|---|
| Uncertain knowledge | Explicit binary defect variables and observation likelihoods |
| Bayesian networks | Defect roots generating test observations |
| Exact inference | Normalized enumeration, checked against analytical Bayes calculations |
| Approximate inference / MCMC | Multi-chain Gibbs sampling compared with exact inference |
| Learning | Smoothed sensor probabilities and correctly linked logistic priors |
| Decision making | Expected information gain and test cost |

Gibbs updates are not described as loopy belief propagation. The project does not claim to implement HMMs or every syllabus topic.

Reference: https://intranet.av.amrita.edu/download/DeanEngg/Curriculum_Syllabus/Undergraduate_Programs/B_Tech_02/241_B_Tech_CSE_AI_Curr_and_Syllabus.pdf
