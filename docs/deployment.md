# Execution and deployment

## Trusted local analysis

Use `--execution local` only for source and tests you trust to run on your machine. Local mode strips application environment variables, disables ambient pytest plugins, runs in a disposable working directory and enforces a deadline. It does not prevent filesystem or network access by Python code.

## Isolated uploaded execution

```sh
docker build -f deploy/Dockerfile.worker -t pcg-worker:latest deploy
python -m pcg.pipeline demo/candidate.py -t demo/test_candidate.py --execution docker
```

Input sources are sent over stdin as a bounded archive. Workers receive no host mounts or API credentials. Container execution uses a non-root user, disabled networking, read-only root filesystem, dropped capabilities, no-new-privileges, memory/CPU/process limits, bounded temporary filesystems and an execution deadline. Results are structured JSON with a size limit. Container failures never fall back to local execution.

For a hosted dashboard, set `PCG_PUBLIC_DEPLOYMENT=1`. Public mode prohibits local execution. Keep the dashboard and Docker execution service on infrastructure appropriate for hostile workloads; do not mount a Docker daemon socket into an unrestricted public app container. Container restrictions are one isolation layer, not proof against kernel vulnerabilities.

The worker image supports pytest, coverage and Hypothesis. Add application dependencies to a dedicated, pinned worker image and set `ExecutionConfig.image` explicitly. Project mode preserves packages, import paths, existing tests and UTF-8 resources, and localizes resolvable cross-module dependencies. Use `python -m pcg.project PROJECT --execution docker --worker-image IMAGE`. It does not install uploaded requirements. CLI local project mode can select a preprovisioned interpreter with `--python`.

## Release checks

- Run the normal suite and Docker integration test on the deployment host.
- Verify repository-disjoint performance of the actual selected model; inspect false-trust rate.
- Add authentication, bounded request queues, rate limits and operational monitoring suitable for the hosting environment.
- Keep source-code retention explicit and obtain user authorization before enabling an external critic service.

No site is deployed by this repository update. The Docker daemon must be available to execute the container integration check.
