# ParetoPilot

ParetoPilot turns "it runs on Arm" into reproducible evidence that it runs well on Arm.

The project benchmarks candidate inference configurations on the same Arm64 target, applies
declared quality and resource constraints, computes a Pareto frontier, and recommends a
deployment configuration without hiding the underlying measurements.

## Current status

This repository contains the benchmark-result contract, recommendation engine, upstream
`llama-bench` validation, and a reproducible native Arm64 benchmark workflow. The included
example data is explicitly synthetic and exists only to exercise the local workflow. It must
never be presented as measured Arm performance.

The first measured experiment runs on GitHub's public `ubuntu-24.04-arm` runner at no compute
cost. It builds one pinned `llama.cpp` commit twice, changes only KleidiAI enablement, and
benchmarks both binaries against the same pinned Qwen Q4_0 model in balanced A-B-B-A order.

## MVP workflow

1. Run controlled baseline and optimized configurations on one pinned Arm64 target.
2. Record raw measurements and quality results in the canonical JSON contract.
3. Apply quality, memory, and latency constraints.
4. Compute the non-dominated configurations.
5. Select the best eligible configuration for the declared objective.
6. Export the recommendation and all deltas from the baseline.

## Try the local recommendation engine

Use Python 3.12 or newer:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\paretopilot doctor
.\.venv\Scripts\paretopilot validate-llama-bench `
  tests\fixtures\llama_bench.synthetic.jsonl
.\.venv\Scripts\paretopilot validate examples\synthetic-results.json
.\.venv\Scripts\paretopilot recommend `
  examples\synthetic-results.json `
  --constraints configs\constraints.example.json
```

On an Arm64 Linux benchmark host, use `paretopilot doctor --require-evidence-host` as an evidence
gate.
The command exits nonzero on this Windows x64 development machine and labels it smoke-test-only.
The checked-in `llama-bench` fixture is also synthetic and tests parsing only.

Run the dependency-free test suite:

```powershell
py -3.12 -m unittest discover -s tests -v
```

## Run the free native Arm64 benchmark

The repository must remain public for the standard Arm64 runner to be free. From the GitHub
Actions page, select **Native Arm64 benchmark**, choose **Run workflow**, and keep the canonical
defaults for publishable evidence. The workflow uploads a compact evidence bundle containing
raw samples, command arrays, environment details, build/model hashes, summaries, comparisons,
and a completion status. Models and build trees are never uploaded.

See [the GitHub Actions benchmark guide](docs/github-actions-benchmark.md) for exact pins,
validity boundaries, and artifact review steps.

## Evidence policy

- Never mix measurements from different runner instances or CPU identities in one comparison.
- Pin the model checksum, runtime commit, compiler, build flags, and workload.
- Warm up the runtime and publish every repetition, not only the best run.
- Report quality changes alongside speed, memory, and model-size changes.
- Label simulated or synthetic data prominently.
- Keep raw evidence immutable after a release is tagged.

Reviewed compact evidence belongs under `results/published/`; unreviewed runs, models, and large
profiler captures remain ignored by git.

See [the benchmark methodology](docs/benchmark-methodology.md) for the initial protocol.
The exact upstream JSONL assumptions live in
[the llama-bench contract](docs/llama-bench-contract.md).

## Scope guardrails

The hackathon MVP targets one model family, one `llama.cpp` runtime, and one controlled paired
experiment per Arm64 runner instance. Multi-cloud orchestration, multiple inference runtimes,
model training, authentication, and an autonomous code-rewriting agent are deliberately out of
scope.

## License

Apache License 2.0. See [LICENSE](LICENSE).
