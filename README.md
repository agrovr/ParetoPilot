# ParetoPilot

ParetoPilot compares Arm64 inference configurations using benchmark data. It checks where the
measurements came from, applies quality and resource limits, compares the tradeoffs, and
recommends a configuration for the selected goal.

[Live results](https://agrovr.github.io/ParetoPilot/) |
[Published results](#published-arm64-results) |
[Verify the archives](#verify-published-results) |
[GitHub Action](docs/github-action.md) |
[Reproduction guide](docs/reproducibility.md)

## Results at a glance

The published v1.1 model and latency study compared four Qwen2.5 1.5B configurations on one
native Arm64 runner. Q8 remained the best choice for the p95 end-to-end latency goal. Tuned Q4 used a
43.72% smaller model and 42.79% less peak memory, but its end-to-end latency was 3.40% slower and
its generation throughput was 9.37% lower.

A separate capacity study asked a different question: how should Q8 and tuned Q4 be configured
for concurrent serving? At four server slots and four simultaneous clients (P4/C4), tuned Q4
measured 6.74% higher generation throughput and 41.09% lower peak memory. That serving result
does not replace the published Q8 model choice.

| Study | Question | Result | Published archive |
| --- | --- | --- | --- |
| Model and latency v1.1 | Which model configuration best meets the latency goal? | Q8 generic | [v1.1.0](https://github.com/agrovr/ParetoPilot/releases/tag/v1.1.0) |
| Capacity v1.4 | Which serving point works best for each selected candidate? | 4 slots / 4 clients for Q8 and tuned Q4 | [v1.4.0](https://github.com/agrovr/ParetoPilot/releases/tag/v1.4.0) |

The source package is currently version 1.4.1. The measured archives remain versioned separately
so the published results can be reproduced exactly.

## Install

ParetoPilot requires Python 3.12 or newer and has no runtime package dependencies.

```bash
git clone https://github.com/agrovr/ParetoPilot.git
cd ParetoPilot
python -m venv .venv
```

Activate the environment:

```bash
# Linux or macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Then install the package:

```bash
python -m pip install -e .
python -m paretopilot doctor
```

## Verify published results

This command downloads the two pinned release archives, checks their size and SHA-256 digest,
replays their decisions, and writes a Markdown and JSON summary:

```bash
paretopilot verify-published --output-dir ../paretopilot-published-proof
```

A successful run prints:

```text
PASS: pinned canonical v1.1 and capacity v1.4 archives verified and replayed.
```

Here, `canonical v1.1` is the verifier's internal label for the published model and latency
study. The capacity archive contains the separate serving study.

Open `../paretopilot-published-proof/published-proof.md` for the readable summary. The command
verifies archived measurements; it does not rerun inference or measure the current computer.

For offline verification, download both release assets and pass their local paths:

```bash
paretopilot verify-published \
  --canonical-archive paretopilot-v1.1.0-arm64-evidence-30055662526.zip \
  --capacity-archive paretopilot-v1.4.0-arm64-capacity-30144901854.zip \
  --output-dir ../paretopilot-published-proof
```

See the [reproduction guide](docs/reproducibility.md) for manual archive checks and fresh Arm64
measurements.

## Run an example

The `init` command creates a small synthetic project with benchmark inputs, constraints, and a
GitHub Actions workflow:

```bash
python -m paretopilot init ../paretopilot-example
cd ../paretopilot-example
paretopilot ci-gate benchmarks/benchmark-set.json \
  --constraints constraints/deployment.json \
  --output-dir paretopilot-output \
  --allow-synthetic \
  --expect-selected-id q4-kleidiai
```

Open `paretopilot-output/optimization-receipt.md` for the decision summary or
`paretopilot-output/report.html` for the full report. The generated values are clearly marked as
synthetic and are not Arm64 benchmark evidence.

## How it works

ParetoPilot:

1. checks each candidate and where its measurements came from;
2. removes configurations that miss the chosen quality or resource limits;
3. finds the best tradeoff configurations, called the Pareto frontier;
4. selects a candidate for the chosen goal while ignoring changes too small to matter; and
5. writes `recommendation.json`, machine-readable decision details in
   `decision-passport.json`, a Markdown decision summary in `optimization-receipt.md`, and a
   self-contained `report.html`.

If an alternative does not improve the chosen goal, ParetoPilot retains the baseline.

## Published Arm64 results

### Latency study

[Run `30055662526`](https://github.com/agrovr/ParetoPilot/actions/runs/30055662526)
used one GitHub-hosted Ubuntu 24.04 Arm64 runner with a 4-vCPU Arm Neoverse-N2 CPU. It compared the
four configurations in the mirrored order `A-B-C-D-D-C-B-A`.

| Candidate | E2E p95 | TTFT p95 | Prompt tok/s | Generation tok/s | Peak RSS | Model size | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q8 generic reference** | **2231.933 ms** | 545.374 ms | 102.6185 | **38.7264** | 3437.598 MiB | 1806.767 MiB | **21/24** |
| Q4 generic | 2311.125 ms | 483.113 ms | 113.8210 | 35.0124 | **1966.473 MiB** | **1016.834 MiB** | 20/24 |
| Q4 + KleidiAI | 2299.454 ms | 470.402 ms | 114.4480 | 35.3764 | 1966.484 MiB | **1016.834 MiB** | 20/24 |
| Q4 + KleidiAI tuned | 2307.715 ms | **469.968 ms** | **131.4565** | 35.0959 | 1966.480 MiB | **1016.834 MiB** | 20/24 |

The 1% cutoff, defined before the run, was 2254.2522 ms. None of the Q4 candidates entered that
shortlist, so ParetoPilot selected Q8 for the latency objective.

The same release also includes:

- a 24-case behavior check;
- five deployment priorities calculated from the same measurements;
- a bounded concurrency 1/2/4 load test; and
- two reconstructed mirrored passes for an observed stability check.

Read the [model and latency result](results/published/30055662526/README.md) or open the
[archived v1.1 HTML report](https://agrovr.github.io/ParetoPilot/evidence/report-v1.1.html).

### Capacity study

[Run `30144901854`](https://github.com/agrovr/ParetoPilot/actions/runs/30144901854)
tested two candidates across server-slot levels 1, 2, and 4 and simultaneous-client levels 1, 2,
and 4. Both candidates performed best at four server slots and four clients (P4/C4). Across all
18 tested combinations and two mirrored passes, 288 measured requests completed with no recorded
failures.

At the selected points, Q8 measured 84.94 generated tokens/s and 3448.1 MiB peak RSS. Tuned Q4
measured 90.67 generated tokens/s and 2031.3 MiB peak RSS, while retaining 20/24 behavior cases
versus Q8's 21/24.

Read the [capacity result](results/published/30144901854/README.md) and
[capacity-study method](docs/capacity-study.md).

## GitHub Action

The composite Action can validate measured inputs, publish the decision artifacts, and fail when
the selected candidate changes unexpectedly:

```yaml
- uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0
  with:
    python-version: "3.12"
- id: decision
  uses: agrovr/ParetoPilot@db9ccaf37e3c7e807832652e237de813675ed807 # v1.4.0
  with:
    benchmarks: benchmarks/benchmark-set.json
    constraints: constraints/deployment.json
    expected-selected-id: q4-kleidiai
    require-arm64-provenance: "true"
```

Measured evidence is required by default. Synthetic inputs must be explicitly enabled for tests
or examples. See the [GitHub Action guide](docs/github-action.md) for all inputs, outputs, and an
Arm64 workflow example.

## Run new measurements

The repository includes manual workflows for a new native Arm64 candidate study and a separate
capacity study. New runs are labeled independently and do not change the published v1.1 or v1.4
results.

- [Model and latency measurement and replay instructions](docs/reproducibility.md)
- [Benchmark methodology](docs/benchmark-methodology.md)
- [Capacity-study method](docs/capacity-study.md)

## Limitations

- The published result covers one model family and workload on one hosted Arm64 runner.
- The 24-case behavior suite is a deployment check, not a general language-model quality
  benchmark.
- The capacity study is bounded and closed-loop; it is not an open-loop production, cost, energy,
  or MLPerf result.
- Two mirrored passes and declared spread limits support an observed consistency description, not
  a statistical significance claim.
- Energy and cost were not measured.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
python -m unittest discover -s tests -v
```

## Documentation

- [Architecture](docs/architecture.md)
- [Benchmark methodology](docs/benchmark-methodology.md)
- [Reproduction guide](docs/reproducibility.md)
- [GitHub Action](docs/github-action.md)
- [Capacity study](docs/capacity-study.md)
- [llama-bench input format](docs/llama-bench-contract.md)

## License

ParetoPilot is available under the [Apache License 2.0](LICENSE). Third-party software and model
artifacts retain their own licenses and are not redistributed by this repository. See
[third-party notices](THIRD_PARTY_NOTICES.md).
