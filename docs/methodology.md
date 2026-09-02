# Methodology

PCG treats code verification as a probabilistic inference problem over code blocks rather than a single binary verdict for the whole file.

## Evidence sources

- compiler diagnostics
- static analysis
- execution failures
- critic verdicts
- LLM confidence proxies

## Inference model

- estimate a prior based on structural complexity
- fuse local evidence in log-odds space
- propagate uncertainty through dependency edges
- isolate local culpability from inherited uncertainty
