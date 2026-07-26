# What ParetoPilot v1.1 adds

ParetoPilot v1.1 adds a 24-case behavior check, deployment priorities, a bounded multi-client
test, a comparison of both measured passes, and a combined report. The v1.0 benchmark and
recommendation remain available for replay.

## Published v1.1 result

Published [run `30055662526`](https://github.com/agrovr/ParetoPilot/actions/runs/30055662526)
completed the v1.1 study on Ubuntu 24.04 Arm64 with a 4-vCPU Arm Neoverse-N2 CPU. Its archive is
release
[`v1.1.0`](https://github.com/agrovr/ParetoPilot/releases/tag/v1.1.0), produced from commit
[`8a9ddce0afa2272c4a4097fe87ef6f06cb7689a9`](https://github.com/agrovr/ParetoPilot/commit/8a9ddce0afa2272c4a4097fe87ef6f06cb7689a9).

The earlier [`v1.0.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.0.0) remains
a reproducible historical result. Its measurements are not combined with v1.1.

The core decision artifacts remain:

- `experiment/manifest.json`
- `experiment/benchmark-set.json`
- `experiment/constraints.json`
- `experiment/recommendation.json`
- `experiment/report.html`
- the raw candidate artifacts bound by the manifest

V1.1 adds files beside the core v1.0 set. Replay requires the complete v1.1 file set whenever the
checksum manifest lists an extension file or `report-v1.1.html`.

## Additional files

### Behavior check

`experiment/evaluation-suite.json` is a checksummed input to experiment assembly, and
`extensions/evaluation-suite.json` is its identical archived extension copy. The
`paretopilot-qwen-behavior-v2` suite declares 24 cases:

- 20 `trimmed-exact` cases compare text after removing leading and trailing whitespace only; and
- 4 `json-exact` cases require valid standard JSON and compare a canonical structural form.

The JSON parser rejects duplicate keys and non-standard constants. Assembly verifies each case
id, prompt, accepted answer, match mode, generation length, and recorded result against the
archived suite. Candidate constraints require a 0.80 absolute quality floor and at least 95%
retention of the measured Q8 score.

The published run measured 21/24 for Q8 and 20/24 for every Q4 candidate. All four passed the
declared gate. This is a deterministic deployment check, not a broad language-model quality
benchmark, and a one-case net difference is not a behavioral-equivalence claim.

The threshold was set before the published run. Incomplete diagnostic
[run `30050573298`](https://github.com/agrovr/ParetoPilot/actions/runs/30050573298) was used only
to set that threshold.

### Policy profiles

`extensions/policy-profiles.json` contains five recommendations calculated from the same
validated `BenchmarkSet`:

| Profile | Classification | Selected candidate |
| --- | --- | --- |
| `canonical-latency` | Canonical | `q8-generic` |
| `memory-first` | Derived non-canonical | `q4-generic` |
| `first-token-first` | Derived non-canonical | `q4-kleidiai-tuned` |
| `prompt-ingest-first` | Derived non-canonical | `q4-kleidiai-tuned` |
| `decode-first` | Derived non-canonical | `q8-generic` |

The artifact binds the benchmark, constraints, and `extensions/policy-config.json` by SHA-256.
Report generation recalculates every recommendation from those inputs.
The four derived profiles are sensitivity views over the same measurements, not additional
benchmark runs.

### Bounded load evaluation

`extensions/load-evaluation.json` combines the per-candidate files below
`extensions/load/<candidate>/load-evaluation.json`. The declared plan uses:

- concurrency 1, 2, and 4;
- three fixed prompts;
- 64 output tokens;
- four warmup requests per level;
- eight measured requests per level;
- 100% required completion;
- p95 TTFT at or below 2,000 ms; and
- p95 end-to-end latency at or below 6,500 ms.

Each row retains request-level evidence and recomputable aggregates for completion, errors,
request and token throughput, TTFT, end-to-end latency, and measured peak RSS.

The evidence records the SHA-256 of `extensions/load-plan.json`, exact request origin, and both
the load-server and canonical deployment command for every candidate. It retains command digests
and full argument arrays, verifies the request host and explicit port against the launched server,
and permits only declared host and port binding differences. Model, runtime, parallelism, thread,
batch, micro-batch, context, and CPU settings must remain materially equivalent.

In the published run, every candidate completed every request. Concurrency 1 was the highest
SLO-passing level for all four candidates. The test covers only the listed concurrency levels.

### Raw repeat-pass reconstruction

`extensions/benchmark-set-pass-1.json` and `extensions/benchmark-set-pass-2.json` are rebuilt from
checksummed raw files under `experiment/candidates/<candidate>/`. `assemble-repeat-pass` does not
split the final aggregate in half. For each candidate and pass, it:

1. follows artifact references from the published benchmark;
2. verifies throughput settings, raw `llama-bench` JSONL, `llama-server` evaluation, and GNU
   `time -v` files against recorded SHA-256 values;
3. recomputes prompt and generation medians from the raw pass;
4. validates behavior cases and latency samples against the archived suite;
5. parses peak RSS from the raw process measurement; and
6. carries forward only candidate identity, parameters, and immutable model size from the
   published benchmark.

Each resulting benchmark set is labeled `supplementary-repeat-pass` and records its source
benchmark and source-artifact fingerprints.

`extensions/repeat-stability.json` compares those validated pass sets. It binds both pass files
and candidate configurations and reports pass values, relative spread, and observed direction
versus the baseline. The published artifact has 24 rows. All six metrics were directionally
consistent for each Q4 candidate; their maximum relative spreads were 1.6695% for Q4 generic,
1.3919% for Q4 + KleidiAI, and 0.8029% for tuned Q4 + KleidiAI.

Two passes do not support a statistical-significance or confidence-interval claim, and
ParetoPilot does not make one.

### V1.1 report

`report-v1.1.html` is a self-contained view of the recommendation and its policy, load, and
stability results. The renderer checks file hashes, candidates, commands, and recalculates the
recommendations before building the report.

Q8 had the lowest p95 end-to-end latency and was the only candidate inside the declared 1%
cutoff. Tuned Q4 + KleidiAI is also shown because it used less memory and had a lower TTFT.

### Published report site

The Pages site presents the v1.1 results in a more readable layout. Before building the page, the
workflow verifies the release, rebuilds all nine outputs, and checks `report-v1.1.html`.

The original report remains available at `evidence/report-v1.1.html`.

## What replay checks

`paretopilot replay` verifies `SHA256SUMS`, canonical completion status, safe relative paths, and
the full required artifact set before writing anything. For a v1.1 archive it regenerates and
compares:

1. the canonical benchmark set;
2. the canonical recommendation;
3. policy profiles;
4. the combined load evaluation;
5. pass 1 benchmark set;
6. pass 2 benchmark set;
7. the repeat-stability summary;
8. the core report; and
9. the v1.1 report.

A missing or different decision artifact makes `decision_reproduced: false` and invalidates the
replay. The published v1.1 release matched all nine outputs.

Release verification returned:

```json
{
  "replay_contract": "1.1",
  "valid": true,
  "decision_reproduced": true,
  "fully_reproduced": true,
  "report_matches_archive": true,
  "selected_id": "q8-generic",
  "differences": [],
  "warnings": []
}
```

Replay uses the archived measurements. A fresh hosted-runner workflow creates a separate
measurement.

## Bundle integrity

The release asset is
[`paretopilot-v1.1.0-arm64-evidence-30055662526.zip`](https://github.com/agrovr/ParetoPilot/releases/download/v1.1.0/paretopilot-v1.1.0-arm64-evidence-30055662526.zip).

| Field | Value |
| --- | --- |
| Size | 402,899 bytes |
| Outer SHA-256 | `b5586878ccd214667911390f417db0417111ac2c31d163a2f5f55c4469aefeb2` |
| Checksummed payloads | 150 |

The archive pins `llama.cpp`
`67b9b0e7f6ce45d929a4411907d3c48ec719e81c`, KleidiAI `1.24.0`, Qwen2.5 1.5B Instruct
revision `91cad51170dc346986eccefdc2dd33a9da36ead9`, and evaluation-suite SHA-256
`e49c16fba32fd65c947264aef4141026ab68b1fd415ef09eeea6e8ade9a545c7`.

## Validation rules

1. JSON files with duplicate keys, unknown fields, non-finite values, identity mismatches,
   incomplete records, or inconsistent aggregates are rejected.
2. Measured recommendations, policy profiles, load evidence, and stability evidence require
   SHA-256 input bindings.
3. `canonical` and `derived-non-canonical` priorities are labeled separately in JSON and HTML.
4. Missing measurements are labeled as not measured.
5. A bundle-level `SHA256SUMS` covers core, raw, and extension artifacts.
6. Report generation and replay remain offline and dependency-free after extraction.
7. Arm Performix is optional and is not used for the benchmark, quality, load, or checksum checks.

## V1.1 bundle layout

```text
status.json
SHA256SUMS
experiment/
  manifest.json
  evaluation-suite.json
  benchmark-set.json
  constraints.json
  recommendation.json
  report.html
  candidates/
    <candidate>/
      raw/
      server-command.json
extensions/
  evaluation-suite.json
  load-plan.json
  policy-config.json
  policy-profiles.json
  benchmark-set-pass-1.json
  benchmark-set-pass-2.json
  repeat-stability.json
  load-evaluation.json
  load/
    <candidate>/
      load-evaluation.json
      server-command.json
report-v1.1.html
```
