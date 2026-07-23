# ParetoPilot

ParetoPilot is an evidence-first deployment advisor for AI inference on Arm64. It verifies
benchmark provenance, applies quality and resource guardrails, computes the Pareto frontier, and
recommends a configuration without assuming that an "optimized" build must win.

I designed and built ParetoPilot as a solo Cloud AI entry for the Arm AI Optimization Challenge.
Arm Performix is an optional profiling enhancement; it is not required by the product or its
evidence pipeline.

[Live decision report](https://agrovr.github.io/ParetoPilot/) |
[Canonical result](results/published/29973188507/README.md) |
[Complete evidence release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.0.0) |
[Reproduction guide](docs/reproducibility.md)

## What it does

Inference optimization is a multi-objective decision. A faster setup may use more memory, lose
quality, or appear better only because of run order. ParetoPilot turns controlled measurements
into an inspectable deployment decision:

1. validate candidate identity, source revisions, model hashes, commands, settings, and samples;
2. reject configurations that miss declared quality or resource limits;
3. compute the non-dominated candidates across latency, throughput, memory, size, and quality;
4. select against a declared objective and practical-effect tolerance; and
5. export a deterministic JSON recommendation and self-contained HTML report.

The baseline is allowed to win. An honest no-change result is more useful than extra deployment
complexity for a gain below the decision tolerance.

## Canonical Arm64 result

[Run `29973188507`](https://github.com/agrovr/ParetoPilot/actions/runs/29973188507)
compared four configurations in balanced order on one GitHub-hosted Arm Neoverse-N2 runner.

| Candidate | E2E p95 | TTFT p95 | Prompt tok/s | Generation tok/s | Peak RSS | Model size | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q8 generic reference | 2335.917 ms | 553.415 ms | 101.7475 | **36.5399** | 3437.613 MiB | 1806.767 MiB | 5/5 |
| Q4 generic | **2330.914 ms** | 483.327 ms | 113.5460 | 34.7130 | **1966.461 MiB** | **1016.834 MiB** | 5/5 |
| Q4 + KleidiAI | 2393.537 ms | **472.764 ms** | 114.1015 | 34.5675 | 1966.477 MiB | **1016.834 MiB** | 5/5 |
| Q4 + KleidiAI tuned | 2337.799 ms | 472.799 ms | **131.2565** | 34.6329 | 1966.477 MiB | **1016.834 MiB** | 5/5 |

ParetoPilot retained **Q8 generic reference**. Q4 generic had the lowest measured p95 end-to-end
latency, but its 0.214% lead was inside the predeclared 1% objective tolerance. The deterministic
simpler-first preference therefore kept Q8 instead of treating that small difference as a
deployment win.

The other results remain useful. Compared with Q8, the tuned Q4 + KleidiAI candidate used a 43.7%
smaller model and 42.8% less peak RSS, raised prompt throughput by 29.0%, and reduced p95 TTFT by
14.6%. It also had 0.08% higher p95 end-to-end latency and 5.2% lower generation throughput. No
candidate won every metric, and all four stayed on the Pareto frontier.

## How the evidence is built

- One native `ubuntu-24.04-arm` job builds pinned generic and KleidiAI-enabled `llama.cpp`
  binaries and verifies two pinned Qwen2.5 1.5B Instruct model files.
- Four stages isolate Q8, Q4 quantization, KleidiAI kernels, and one micro-batch change.
- Throughput and server passes use the balanced order `A-B-C-D-D-C-B-A` on the same runner.
- `llama-bench` supplies twenty prompt and generation samples per candidate and workload.
- `llama-server` supplies a fixed five-case exact-answer smoke gate and twenty streamed 64-token
  latency samples per candidate; GNU `time -v` supplies peak RSS.
- Runtime logs must prove KleidiAI dispatch only for the intended candidates.
- A closed manifest binds the decision evidence before selection; a bundle-level SHA-256 file then
  locks the final archive.

The permanent release contains every raw sample, command, environment record, build log,
dispatch log, manifest, recommendation, and offline report. For the canonical run, the Git
repository keeps only a short reviewed summary and an exact archive lock.

## Quick start

ParetoPilot requires Python 3.12 or newer and has no runtime package dependencies.

```bash
git clone https://github.com/agrovr/ParetoPilot.git
cd ParetoPilot
python -m venv .venv
```

Activate it with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell or
`source .venv/bin/activate` on Linux and macOS, then run:

```bash
python -m pip install -e ".[dev]"
python -m paretopilot --version
python -m unittest discover -s tests
python -m paretopilot validate examples/synthetic-results.json
python -m paretopilot recommend examples/synthetic-results.json --constraints configs/constraints.example.json
```

The example is explicitly synthetic and exists only to exercise the engine. It is not presented
as Arm64 benchmark evidence.

## Reproduce the canonical decision

Download
[`paretopilot-v1.0.0-arm64-evidence-29973188507.zip`](https://github.com/agrovr/ParetoPilot/releases/download/v1.0.0/paretopilot-v1.0.0-arm64-evidence-29973188507.zip)
and verify this SHA-256 before extraction:

```text
fb4f4c86a729a5eb42e23dbd3c6346fd4ab31ce14423dbb8c7672b11b6a6fd00
```

After extracting it to `evidence/`, rebuild the measured benchmark set and report into fresh
paths:

```bash
python -m paretopilot assemble-experiment evidence/experiment/manifest.json --output output/reproduction/benchmark-set.json
python -m paretopilot report output/reproduction/benchmark-set.json --constraints evidence/experiment/constraints.json --output output/reproduction/report.html --recommendation-output output/reproduction/recommendation.json
```

ParetoPilot refuses to overwrite existing outputs. See the [reproduction guide](docs/reproducibility.md)
for checksum, byte-comparison, and fresh Arm64 measurement instructions.

## Run a new native Arm64 study

Open **Actions → Native Arm64 candidate study → Run workflow** on the default branch and retain
the canonical input of ten repetitions. The workflow labels branch runs or changed inputs as
exploratory, even when their measurements complete successfully.

The workflow downloads sources and models during the job but never uploads model files, build
trees, or credentials. Only compact evidence is retained.

## Trust and limits

- Measured and synthetic inputs are explicitly separated.
- Closed ParetoPilot schemas reject duplicate keys, non-finite numbers, unknown fields, and
  malformed evidence; the upstream `llama-bench` reader validates a required subset while
  tolerating additional upstream fields. Experiment paths are bounded below their evidence root.
- Candidate summaries reconcile reported runtime settings with declared settings.
- Missing fingerprints, mismatched commands, invalid checksums, incomplete runs, or absent
  dispatch proof fail closed.
- The five-case exact-answer evaluation is a smoke gate, not a broad language-model quality
  benchmark.
- The canonical result is one controlled hosted-runner comparison. It does not claim the same
  ranking on every Arm processor, model, workload, or concurrency level.
- Energy and cost were not measured.

## Project map

```text
src/paretopilot/                 validation, assembly, selection, and reporting
evals/qwen-smoke-v1.json         fixed quality and latency smoke suite
configs/                         declared decision constraints
results/published/29973188507/   concise canonical summary and release lock
.github/workflows/               cross-platform CI, Arm64 study, and verified Pages deploy
docs/                            architecture, methodology, contracts, and reproduction
tests/                           deterministic behavior and failure-path coverage
```

## License

ParetoPilot is available under the [Apache License 2.0](LICENSE). Third-party software and model
artifacts retain their own licenses and are not redistributed by this repository. See
[third-party notices](THIRD_PARTY_NOTICES.md) for pinned components and upstream terms.
