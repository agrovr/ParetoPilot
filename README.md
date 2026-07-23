# ParetoPilot

ParetoPilot is an evidence-first deployment advisor for AI inference on Arm64. It verifies
benchmark provenance, applies quality and resource guardrails, computes the Pareto frontier, and
exports a deployment recommendation without assuming that an "optimized" build must win.

This is a **Cloud AI** project for the Arm AI Optimization Challenge. Arm Performix can add
optional profiler context, but it is not required by the product or its evidence pipeline.

[Open the live decision report](https://agrovr.github.io/ParetoPilot/) |
[Review the canonical evidence](results/published/29940067201/README.md) |
[Reproduce the result](docs/reproducibility.md)

## Why ParetoPilot

Inference optimization is a multi-objective decision. A faster configuration may use more
memory, reduce answer quality, or show an apparent gain that disappears when the run order
changes. ParetoPilot makes those tradeoffs inspectable:

1. collect candidates on one controlled Arm64 host;
2. validate source revisions, model hashes, commands, settings, and raw samples;
3. reject candidates that miss declared quality or resource constraints;
4. find the non-dominated configurations;
5. select the best eligible candidate for a declared objective; and
6. produce a deterministic, self-contained HTML report and machine-readable recommendation.

The baseline is allowed to win. That is a feature: an honest no-change decision is more useful
than deploying a noise-scale optimization.

## Evidence available today

### Canonical two-candidate Arm64 study

The reviewed canonical run compared generic and KleidiAI-enabled `llama.cpp` builds on the same
GitHub-hosted Ubuntu 24.04 Arm64 runner. Both used the same pinned Qwen2.5 1.5B Instruct Q4_0
model, source revision, compiler policy, CPU-only settings, and balanced A-B-B-A run order.

| Workload | Generic median | KleidiAI median | Change | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 512-token prompt processing | 113.6605 tok/s | 114.2580 tok/s | +0.5257% | Small, consistent gain |
| 128-token generation | 35.1134 tok/s | 35.1307 tok/s | +0.0493% | Inconclusive |

Generation changed by +2.1583% in the first pair and -1.2145% in the second. Because the paired
directions disagreed and the pooled change was below the predeclared 1% practical-effect
threshold, ParetoPilot retained the generic baseline. The result is measured, non-synthetic
evidence from [run 29940067201](https://github.com/agrovr/ParetoPilot/actions/runs/29940067201).

This first study measures throughput only. It does **not** claim measured TTFT, request latency,
peak memory, energy, cost, or direct task-quality improvement. The model file was identical
between candidates, so the recorded quality basis is model identity rather than an independent
task evaluation. See the [evidence boundaries](results/published/29940067201/README.md#evidence-boundaries).

### Expanded four-candidate study

The repository also contains the
[`Native Arm64 candidate study`](.github/workflows/candidate-study-arm64.yml), which is designed
to compare four stages on one native Arm64 runner:

| Candidate | Purpose |
| --- | --- |
| Q8 generic | Higher-precision reference |
| Q4 generic | Quantization-only change |
| Q4 with KleidiAI | Arm-kernel change |
| Q4 with KleidiAI and a 512-token micro-batch | Runtime-tuning change |

That workflow pins source and model revisions, uses balanced A-B-C-D-D-C-B-A throughput and
server passes, runs a fixed five-case exact-match quality smoke suite, measures streamed TTFT and
end-to-end latency over fixed 64-token generations, records peak RSS, and assembles a checksummed
report. A predeclared 1% objective tolerance favors the simpler candidate when a latency lead is
too small to justify added complexity. **Its outputs are not final evidence until a
successful canonical run on the default branch has been reviewed and published.** No result from
this four-candidate workflow is claimed in this README yet.

## Quick start

ParetoPilot requires Python 3.12 or newer and has no runtime package dependencies.

```bash
python -m venv .venv
```

Activate it with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell or
`source .venv/bin/activate` on Linux and macOS. Then install and verify:

```bash
python -m pip install -e ".[dev]"
python -m paretopilot --version
python -m unittest discover -s tests -v
```

Run the synthetic recommendation example to exercise the engine without presenting it as Arm
evidence:

```bash
python -m paretopilot validate examples/synthetic-results.json
python -m paretopilot recommend examples/synthetic-results.json --constraints configs/constraints.example.json
```

Verify the checked-in measured bundle and rebuild its report into a fresh output directory:

```bash
python -m paretopilot verify-study results/published/29940067201
python -m paretopilot assemble-study results/published/29940067201 --benchmarks-output output/reproduction/benchmark-set.json --constraints-output output/reproduction/constraints.json --assembly-output output/reproduction/assembly.json
python -m paretopilot report output/reproduction/benchmark-set.json --constraints output/reproduction/constraints.json --output output/reproduction/report.html --recommendation-output output/reproduction/recommendation.json
```

ParetoPilot refuses to overwrite output files. Use a new output directory for each reproduction
attempt. On an Arm64 Linux evidence host, this command is the architecture gate:

```bash
python -m paretopilot doctor --require-evidence-host
```

It intentionally exits nonzero on Windows x64 and other smoke-test-only hosts.

## Run the native Arm64 study

GitHub Actions provides the project's current native Arm64 path. From a public fork or repository,
open **Actions**, select **Native Arm64 candidate study**, and dispatch it with the default ten
repetitions. A default-branch run with the canonical input is eligible for canonical
classification; branch runs are marked exploratory.

The workflow downloads pinned model artifacts during the run and does not upload models or build
trees. It uploads the compact measurements, environment records, commands, hashes, recommendation,
and offline report. Follow the [reproducibility guide](docs/reproducibility.md) before describing
any run as submission evidence.

## Trust model

- Measured and synthetic inputs are explicitly separated.
- Strict JSON readers reject duplicate keys and non-finite values; evidence-specific parsers also
  enforce closed schemas and bounded records or artifacts.
- Multi-candidate artifacts are content-addressed and resolved below the experiment directory.
- Candidate summaries reconcile runtime-reported settings with declared settings.
- The report includes source fingerprints, constraint outcomes, rejection reasons, baseline
  deltas, and frontier membership. It exports authoritative launch arguments when evidence
  supplies them and refuses to guess when it does not.
- Raw evidence remains available for audit; the recommendation is never the only output.

The initial protocol is documented in [benchmark methodology](docs/benchmark-methodology.md), and
the upstream parser contract is documented in [llama-bench evidence contract](docs/llama-bench-contract.md).

## Project map

```text
src/paretopilot/                 validation, assembly, selection, and reporting
evals/qwen-smoke-v1.json         fixed quality and latency smoke suite
configs/                         declared decision constraints
results/published/29940067201/   reviewed two-candidate Arm64 evidence
.github/workflows/               cross-platform CI and native Arm64 studies
docs/                            architecture, methods, and reproduction guides
```

## License

ParetoPilot is available under the [Apache License 2.0](LICENSE). Third-party software and model
artifacts retain their own licenses and are not redistributed by this repository. See
[third-party notices](THIRD_PARTY_NOTICES.md) for the exact pinned components and upstream terms.
