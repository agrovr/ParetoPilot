# Canonical Arm64 result: run 30055662526

This directory is the compact repository lock for ParetoPilot's current canonical v1.1 result.
The complete raw and generated evidence is preserved in the
[`v1.1.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.1.0).

## Release lock

| Field | Value |
| --- | --- |
| GitHub Actions run | [`30055662526`](https://github.com/agrovr/ParetoPilot/actions/runs/30055662526) |
| Classification | `canonical` |
| Source commit | [`8a9ddce0afa2272c4a4097fe87ef6f06cb7689a9`](https://github.com/agrovr/ParetoPilot/commit/8a9ddce0afa2272c4a4097fe87ef6f06cb7689a9) |
| Runner | Ubuntu 24.04 Arm64, Arm Neoverse-N2, 4 CPUs |
| Release | [`v1.1.0`](https://github.com/agrovr/ParetoPilot/releases/tag/v1.1.0) |
| Asset | [`paretopilot-v1.1.0-arm64-evidence-30055662526.zip`](https://github.com/agrovr/ParetoPilot/releases/download/v1.1.0/paretopilot-v1.1.0-arm64-evidence-30055662526.zip) |
| Asset size | 402,899 bytes |
| Archive SHA-256 | `b5586878ccd214667911390f417db0417111ac2c31d163a2f5f55c4469aefeb2` |
| Checksummed payloads | 150 |

The machine-readable [`evidence.json`](evidence.json) binds the run, release asset, outer archive
digest, source and runner identity, important artifact digests, and replay review.

## Measured candidates

| Candidate | E2E p95 | TTFT p95 | Prompt tok/s | Generation tok/s | Peak RSS | Model size | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q8 generic reference** | **2231.932869 ms** | 545.373894 ms | 102.6185 | **38.72645** | 3437.597656 MiB | 1806.766632 MiB | **21/24** |
| Q4 generic | 2311.125148 ms | 483.113115 ms | 113.8210 | 35.01235 | **1966.472656 MiB** | **1016.833527 MiB** | 20/24 |
| Q4 + KleidiAI | 2299.454336 ms | 470.402254 ms | 114.4480 | 35.37635 | 1966.484375 MiB | **1016.833527 MiB** | 20/24 |
| Q4 + KleidiAI tuned | 2307.715263 ms | **469.968079 ms** | **131.4565** | 35.09590 | 1966.480469 MiB | **1016.833527 MiB** | 20/24 |

## Decision

ParetoPilot selected **`q8-generic`**. It had the lowest measured p95 end-to-end latency, so the
baseline was both the numeric winner and the canonical recommendation. The declared 1% cutoff was
2254.2522 ms; only Q8 entered the shortlist.

The result does not erase the resource tradeoff. Compared with Q8, tuned Q4 + KleidiAI had:

- 43.72% smaller model size;
- 42.79% lower peak RSS;
- 13.83% lower p95 TTFT;
- 28.10% higher prompt throughput;
- 3.40% slower p95 end-to-end latency;
- 9.37% lower generation throughput; and
- one fewer passing behavior case.

That makes the tuned candidate a measured resource alternative, not the canonical latency winner.

## Supplementary v1.1 evidence

- **Policy profiles:** `canonical-latency` and `decode-first` select Q8; `memory-first` selects Q4
  generic; `first-token-first` and `prompt-ingest-first` select tuned Q4 + KleidiAI.
- **Bounded load:** each candidate completed eight measured requests at concurrency 1, 2, and 4.
  Completion was 100%; concurrency 1 was the highest level to meet both the 2000 ms p95 TTFT and
  6500 ms p95 end-to-end SLOs for every candidate.
- **Repeat stability:** the two reconstructed passes produced 24 comparison rows. All six metrics
  were directionally consistent for each Q4 candidate. The largest relative spread was 1.6695%
  for Q4 generic, 1.3919% for Q4 + KleidiAI, and 0.8029% for tuned Q4 + KleidiAI. These are
  observed spreads, not statistical-significance claims.

## Pinned identity

- `llama.cpp`: `67b9b0e7f6ce45d929a4411907d3c48ec719e81c`
- KleidiAI: `1.24.0`
- Qwen2.5 1.5B Instruct revision: `91cad51170dc346986eccefdc2dd33a9da36ead9`
- Evaluation-suite SHA-256:
  `e49c16fba32fd65c947264aef4141026ab68b1fd415ef09eeea6e8ade9a545c7`

## Verify and replay

After downloading the release asset, verify the outer archive:

```bash
printf '%s  %s\n' \
  b5586878ccd214667911390f417db0417111ac2c31d163a2f5f55c4469aefeb2 \
  paretopilot-v1.1.0-arm64-evidence-30055662526.zip | sha256sum --check
```

Extract it into `evidence/`, verify the payload manifest, and replay into a fresh directory:

```bash
(cd evidence && sha256sum --check SHA256SUMS)
python -m paretopilot replay evidence --output-dir output/replay-v1.1
```

The independent release review returned `replay_contract: "1.1"`, `valid: true`,
`decision_reproduced: true`, `fully_reproduced: true`, and `selected_id: "q8-generic"`. All nine
comparisons matched, and both `differences` and `warnings` were empty.

## Boundaries

This is one model and workload on one ephemeral hosted runner. The bounded load sweep tested only
concurrency 1, 2, and 4. The deterministic behavior suite is a deployment gate, not a broad model
evaluation. Two balanced passes do not establish statistical significance. Energy and cost were
not measured, and Arm Performix was not required.

The earlier [v1.0 result](../29973188507/README.md) remains preserved as a separate historical
experiment and is not pooled with this run.
