# Historical PCG results

These files preserve results from the previous heuristic pipeline. They are research artifacts, not active runtime settings. In particular, the old fitted weights disabled execution evidence and performed poorly on their saved holdout comparison.

Current evaluation lives in `pcg.validation` and writes versioned observation models and reports under `out/validation/`. Runtime model loading is explicit through `--model` or `SensorModel.load`.
