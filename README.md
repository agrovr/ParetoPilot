# ParetoPilot

ParetoPilot is an evidence-first deployment advisor for AI inference on Arm64. It validates
benchmark provenance, applies quality and resource guardrails, computes the Pareto frontier, and
recommends a configuration without assuming that an "optimized" build must win.

I designed and built ParetoPilot as a solo Cloud AI entry for the Arm AI Optimization Challenge.
Arm Performix is an optional profiling enhancement; it is not required by the product or its
evidence pipeline.

[Live decision report](https://agrovr.github.io/ParetoPilot/) |
[Canonical v1.1 result](results/published/30055662526/README.md) |
[Complete v1.1 evidence](https://github.com/agrovr/ParetoPilot/releases/tag/v1.1.0) |
[Reproduction guide](docs/reproducibility.md)

## What it does

Inference optimization is a multi-objective decision. A faster setup may use more memory, lose
quality, or appear better only because of run order. ParetoPilot turns controlled measurements
into an inspectable deployment decision:

1. validate candidate identity, source revisions, model hashes, commands, settings, and samples;
2. reject configurations that miss declared quality or resource limits;
3. compute the non-dominated candidates across latency, throughput, memory, size, and quality;
4. select against a declared objective and practical-effect tolerance; and
5. export deterministic JSON recommendations and a self-contained HTML report.

The baseline is allowed to win. An honest no-change result is more useful than extra deployment
complexity when the measured alternatives do not improve the declared objective.

## Canonical Arm64 result

[Run `30055662526`](https://github.com/agrovr/ParetoPilot/actions/runs/30055662526)
completed on one GitHub-hosted Ubuntu 24.04 Arm64 runner with a 4-vCPU Arm Neoverse-N2 CPU. It
compared four configurations in balanced order from commit
[`8a9ddce`](https://github.com/agrovr/ParetoPilot/commit/8a9ddce0afa2272c4a4097fe87ef6f06cb7689a9).

| Candidate | E2E p95 | TTFT p95 | Prompt tok/s | Generation tok/s | Peak RSS | Model size | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q8 generic reference** | **2231.933 ms** | 545.374 ms | 102.6185 | **38.7265** | 3437.598 MiB | 1806.767 MiB | **21/24** |
| Q4 generic | 2311.125 ms | 483.113 ms | 113.8210 | 35.0124 | **1966.473 MiB** | **1016.834 MiB** | 20/24 |
| Q4 + KleidiAI | 2299.454 ms | 470.402 ms | 114.4480 | 35.3764 | 1966.484 MiB | **1016.834 MiB** | 20/24 |
| Q4 + KleidiAI tuned | 2307.715 ms | **469.968 ms** | **131.4565** | 35.0959 | 1966.480 MiB | **1016.834 MiB** | 20/24 |

ParetoPilot retained **Q8 generic reference** because it was also the numeric p95 end-to-end
latency winner. The predeclared 1% objective cutoff was 2254.2522 ms, so none of the three Q4
candidates entered the canonical shortlist. The result is a useful no-change decision: the
measured optimization alternatives had real resource advantages, but they did not beat the
declared latency objective.

The tuned Q4 + KleidiAI configuration is the strongest measured resource alternative. Compared
with Q8, it used a 43.72% smaller model and 42.79% less peak RSS, reduced p95 TTFT by 13.83%, and
raised prompt throughput by 28.10%. The tradeoff was 3.40% slower p95 end-to-end latency, 9.37%
lower generation throughput, and one fewer passing behavior case.

## What v1.1 adds

The canonical v1.1 release makes four supplementary views inspectable without changing the core
selection rule:

- **Behavior gate:** a checksummed 24-case suite uses declared `trimmed-exact` and strict
  `json-exact` matching. Q8 passed 21/24 cases; every Q4 candidate passed 20/24. All four cleared
  the 0.80 absolute floor and 95% baseline-retention rule.
- **Policy sensitivity:** five profiles recompute the decision from the same benchmark. Canonical
  latency and decode-first select Q8; memory-first selects Q4 generic; first-token-first and
  prompt-ingest-first select tuned Q4 + KleidiAI.
- **Bounded load:** every candidate ran eight measured requests at concurrency 1, 2, and 4. All
  had 100% completion; concurrency 1 was the highest level that met both the 2000 ms p95 TTFT and
  6500 ms p95 end-to-end SLOs.
- **Repeat stability:** two independently reconstructed balanced passes produced 24 comparison
  rows. All six reported metrics were directionally consistent for the three Q4 candidates. The
  largest relative spread for those candidates was 1.6695%; this is an observed consistency
  result, not a statistical significance claim.

The release contains 150 checksummed payloads. A separate offline replay verified the complete
bundle, reproduced the selected candidate, matched all nine core and report comparisons, and
returned no differences or warnings.

## How the evidence was built

- One native `ubuntu-24.04-arm` job built pinned generic and KleidiAI-enabled `llama.cpp`
  binaries and verified two pinned Qwen2.5 1.5B Instruct model files.
- Four stages isolated Q8, Q4 quantization, KleidiAI kernels, and one micro-batch change.
- Throughput and server passes used the balanced order `A-B-C-D-D-C-B-A` on the same runner.
- `llama-bench` supplied prompt-processing and generation-throughput samples.
- `llama-server` supplied the deterministic behavior gate, streamed 64-token TTFT and
  end-to-end samples, and the bounded 1/2/4-client load sweep; GNU `time -v` supplied peak RSS.
- Runtime logs had to prove KleidiAI dispatch only for the intended candidates.
- A closed manifest bound the decision evidence before selection, and bundle-level SHA-256
  checksums locked the release archive.

The measurement pins include `llama.cpp`
`67b9b0e7f6ce45d929a4411907d3c48ec719e81c`, KleidiAI `1.24.0`, Qwen2.5 1.5B Instruct
revision `91cad51170dc346986eccefdc2dd33a9da36ead9`, and evaluation-suite SHA-256
`e49c16fba32fd65c947264aef4141026ab68b1fd415ef09eeea6e8ade9a545c7`.

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
[`paretopilot-v1.1.0-arm64-evidence-30055662526.zip`](https://github.com/agrovr/ParetoPilot/releases/download/v1.1.0/paretopilot-v1.1.0-arm64-evidence-30055662526.zip)
and verify this outer SHA-256 before extraction:

```text
b5586878ccd214667911390f417db0417111ac2c31d163a2f5f55c4469aefeb2
```

After extracting it to `evidence/`, replay the complete archived contract into a fresh directory:

```bash
python -m paretopilot replay evidence --output-dir output/reproduction
```

The canonical replay reports `replay_contract: "1.1"`, `valid: true`,
`decision_reproduced: true`, `fully_reproduced: true`, and `selected_id: "q8-generic"`, with
empty differences and warnings. See the [reproduction guide](docs/reproducibility.md) for the
complete checksum and comparison procedure.

## Historical v1.0 evidence

The earlier [run `29973188507`](results/published/29973188507/README.md) and
[`v1.0.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.0.0) remain preserved as
historical evidence. That study used a five-case smoke gate and produced a different measured
latency ordering on a separate ephemeral runner. It is independently reproducible, but it is not
pooled with or substituted for the current v1.1 canonical run.

## Run a new native Arm64 study

Open **Actions → Native Arm64 candidate study → Run workflow** on the default branch and retain
the canonical input of ten repetitions. The workflow labels branch runs or changed inputs as
exploratory, even when their measurements complete successfully.

The workflow downloads sources and models during the job but never uploads model files, build
trees, or credentials. Only compact evidence is retained.

## Trust and limits

- Measured and synthetic inputs are explicitly separated.
- Closed schemas reject duplicate keys, non-finite numbers, unknown fields, malformed evidence,
  mismatched settings, missing fingerprints, path escapes, and invalid checksums.
- The 24-case behavior suite is a deterministic deployment gate, not a broad language-model
  quality benchmark.
- The 1/2/4-client sweep is bounded evidence, not a general capacity study.
- Two balanced passes support an observed consistency description, not a significance claim.
- The canonical result is one model and workload on one controlled hosted Arm64 runner. It does
  not claim the same ranking on every Arm processor or deployment.
- Energy and cost were not measured.
- Arm Performix remains optional and does not substitute profiler output for benchmark evidence.

## Project map

```text
src/paretopilot/                 validation, assembly, selection, replay, and reporting
evals/                           versioned behavior and latency suites
configs/                         decision, policy-profile, and bounded-load declarations
results/published/30055662526/   current v1.1 canonical summary and release lock
results/published/29973188507/   preserved v1.0 historical summary and release lock
.github/workflows/               cross-platform CI, Arm64 study, and verified Pages deploy
docs/                            architecture, methodology, contracts, and reproduction
tests/                           deterministic behavior and failure-path coverage
```

## License

ParetoPilot is available under the [Apache License 2.0](LICENSE). Third-party software and model
artifacts retain their own licenses and are not redistributed by this repository. See
[third-party notices](THIRD_PARTY_NOTICES.md) for pinned components and upstream terms.
