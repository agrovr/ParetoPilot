# ParetoPilot v1.1 evidence extensions

ParetoPilot v1.1 adds expanded deterministic behavior evaluation, deployment-policy scenarios,
bounded multi-client measurements, pass-level reconstruction, repeat-stability summaries, and a
combined evidence report. The extension is additive: it preserves the strict core benchmark and
recommendation boundary introduced in v1.0.

## Published status and compatibility boundary

Canonical [run `30055662526`](https://github.com/agrovr/ParetoPilot/actions/runs/30055662526)
completed the full v1.1 contract on Ubuntu 24.04 Arm64 with a 4-vCPU Arm Neoverse-N2 CPU. Its
permanent archive is release
[`v1.1.0`](https://github.com/agrovr/ParetoPilot/releases/tag/v1.1.0), produced from commit
[`8a9ddce0afa2272c4a4097fe87ef6f06cb7689a9`](https://github.com/agrovr/ParetoPilot/commit/8a9ddce0afa2272c4a4097fe87ef6f06cb7689a9).

The earlier [`v1.0.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.0.0) remains
reproducible historical evidence. V1.1 does not rewrite or pool those measurements.

The core decision artifacts remain:

- `experiment/manifest.json`
- `experiment/benchmark-set.json`
- `experiment/constraints.json`
- `experiment/recommendation.json`
- `experiment/report.html`
- the raw candidate artifacts bound by the manifest

V1.1 artifacts live beside that core. They are not inserted into a v1.0 object under undeclared
fields. An archive is recognized as v1.1 when its checksum manifest lists an extension artifact
or `report-v1.1.html`; once recognized, the complete v1.1 artifact contract is required rather
than silently accepting a partial extension set.

## Extension artifacts

### Checksummed behavior suite

`experiment/evaluation-suite.json` is a checksummed input to experiment assembly, and
`extensions/evaluation-suite.json` is its identical archived extension copy. The
`paretopilot-qwen-behavior-v2` suite declares 24 cases:

- 20 `trimmed-exact` cases compare text after removing leading and trailing whitespace only; and
- 4 `json-exact` cases require valid standard JSON and compare a canonical structural form.

The JSON parser rejects duplicate keys and non-standard constants. Assembly verifies each case
id, prompt, accepted answer, match mode, generation length, and recorded result against the
archived suite. Candidate constraints require a 0.80 absolute quality floor and at least 95%
retention of the measured Q8 score.

The canonical run measured 21/24 for Q8 and 20/24 for every Q4 candidate. All four passed the
declared gate. This is a deterministic deployment check, not a broad language-model quality
benchmark, and a one-case net difference is not a behavioral-equivalence claim.

The threshold was declared before the canonical run. Incomplete diagnostic
[run `30050573298`](https://github.com/agrovr/ParetoPilot/actions/runs/30050573298) was used only
for pre-canonical calibration and correctly records invalid evidence.

### Policy profiles

`extensions/policy-profiles.json` contains five precomputed recommendations derived from the same
validated `BenchmarkSet`:

| Profile | Classification | Selected candidate |
| --- | --- | --- |
| `canonical-latency` | Canonical | `q8-generic` |
| `memory-first` | Derived non-canonical | `q4-generic` |
| `first-token-first` | Derived non-canonical | `q4-kleidiai-tuned` |
| `prompt-ingest-first` | Derived non-canonical | `q4-kleidiai-tuned` |
| `decode-first` | Derived non-canonical | `q8-generic` |

The artifact binds the benchmark, constraints, and `extensions/policy-config.json` by SHA-256.
Report generation recomputes every recommendation rather than trusting a supplied selected id.
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

In the canonical run, every candidate completed every request. Concurrency 1 was the highest
SLO-passing level for all four candidates. These rows are bounded operational evidence; they are
not converted into cost, energy, general capacity, or sustainability claims.

### Raw repeat-pass reconstruction

`extensions/benchmark-set-pass-1.json` and `extensions/benchmark-set-pass-2.json` are rebuilt from
checksummed raw files under `experiment/candidates/<candidate>/`. `assemble-repeat-pass` does not
split the final aggregate in half. For each candidate and pass, it:

1. follows bounded artifact references from the canonical benchmark;
2. verifies throughput settings, raw `llama-bench` JSONL, `llama-server` evaluation, and GNU
   `time -v` files against recorded SHA-256 values;
3. recomputes prompt and generation medians from the raw pass;
4. validates behavior cases and latency samples against the archived suite;
5. parses peak RSS from the raw process measurement; and
6. carries forward only candidate identity, parameters, and immutable model size from the
   canonical benchmark.

Each resulting benchmark set is labeled `supplementary-repeat-pass` and records its source
benchmark and source-artifact fingerprints.

`extensions/repeat-stability.json` compares those validated pass sets. It binds both pass files
and candidate configurations and reports pass values, relative spread, and observed direction
versus the baseline. The canonical artifact has 24 rows. All six metrics were directionally
consistent for each Q4 candidate; their maximum relative spreads were 1.6695% for Q4 generic,
1.3919% for Q4 + KleidiAI, and 0.8029% for tuned Q4 + KleidiAI.

Two passes do not support a statistical-significance or confidence-interval claim, and
ParetoPilot does not make one.

### V1.1 report

`report-v1.1.html` is a deterministic, self-contained view of the canonical recommendation and
the additive policy, load, and stability evidence. The renderer validates input fingerprints,
candidate coverage, command bindings, and internally recomputed recommendations before drawing
the report. It does not run inference or change the selection decision.

The report keeps the primary finding honest: Q8 was the numeric p95 end-to-end winner and the
only candidate within the declared 1% cutoff. Tuned Q4 + KleidiAI is shown as a resource
alternative with lower model size, RSS, and TTFT rather than being mislabeled as the canonical
winner.

## Replay semantics

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
replay. Presentation-only report drift can be surfaced separately from measurement drift, but the
published v1.1 release matched all nine comparisons exactly.

The independent release replay returned:

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

Replay never reruns inference. A fresh hosted-runner workflow is new measurement evidence, not a
replay of the archived hardware environment.

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

## Trust rules

1. Every JSON artifact has a closed schema: duplicate keys, unknown fields, non-finite values,
   identity mismatches, incomplete records, and inconsistent aggregates fail closed.
2. Measured recommendations, policy profiles, load evidence, and stability evidence require
   SHA-256 input bindings.
3. Canonical and derived decisions are visibly distinguished in JSON and HTML.
4. Missing extension data is shown as not measured; it is never inferred.
5. A bundle-level `SHA256SUMS` covers core, raw, and extension artifacts.
6. Report generation and replay remain offline and dependency-free after extraction.
7. Arm Performix is optional and cannot replace benchmark, quality, load, or checksum evidence.

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
