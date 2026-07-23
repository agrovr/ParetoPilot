# ParetoPilot v1.1 evidence extensions

ParetoPilot v1.1 adds stricter quality evaluation, deployment-policy scenarios, bounded
multi-client measurements, pass-level reconstruction, and repeat-stability summaries. The
extension is additive: it does not change the schema or byte-level rebuild path of the published
v1.0 evidence.

## Evidence status and compatibility boundary

The published v1.0 release remains the authoritative measured result until a fresh native Arm64
v1.1 run completes, passes review, and is locked in a permanent release. The v1.1 source and
workflow describe an implemented evidence contract; configuration files and tests are not
benchmark results.

The v1.0 decision core remains valid:

- `experiment/manifest.json`
- `experiment/benchmark-set.json`
- `experiment/constraints.json`
- `experiment/recommendation.json`
- the raw candidate artifacts bound by the manifest

V1.1 artifacts live beside that core. They are never inserted into a v1.0 object under an
undeclared field, and their absence cannot change a v1.0 recommendation. A v1.1 archive is
recognized when its checksum manifest lists any extension artifact or `report-v1.1.html`; once
recognized, the complete v1.1 contract is required rather than silently accepting a partial
extension set.

## Extension artifacts

### Checksummed behavior suite

`experiment/evaluation-suite.json` is a checksummed input to experiment assembly, and
`extensions/evaluation-suite.json` is the identical archived copy used by extension replay. The
current `paretopilot-qwen-behavior-v2` suite declares 24 cases:

- 20 `trimmed-exact` cases compare text after removing leading and trailing whitespace only; and
- 4 `json-exact` cases require valid standard JSON and compare a canonical structural form.

The JSON parser rejects duplicate keys and non-standard constants. Assembly verifies every
recorded case id, prompt, accepted answer, match mode, generation length, and suite digest against
the archived suite. The candidate constraints add a 0.80 absolute quality floor and require at
least 95% retention of the measured baseline score.

This remains a deterministic behavior gate, not a broad language-model quality benchmark.
The thresholds were declared after
[non-canonical exploratory run `30050573298`](https://github.com/agrovr/ParetoPilot/actions/runs/30050573298)
and before any canonical v1.1 run. The measured Q8 reference scored 21/24 and each Q4 candidate
scored 20/24 in both passes, so the combined gate requires at least 20/24 for that reference and
rejects 19/24. Case-level failures remain checksummed; the one-case net difference is not a claim
of behavioral equivalence or general quality. The
[benchmark methodology](benchmark-methodology.md#stricter-behavior-gate) records the disclosed
failure types and calibration rationale.

### Policy profiles

`extensions/policy-profiles.json` contains five precomputed recommendations derived from the same
validated `BenchmarkSet`:

- canonical p95 end-to-end latency;
- memory first;
- time to first token first;
- prompt ingest first; and
- token decode first.

Only `canonical-latency` is canonical. The other entries are labeled
`derived-non-canonical`; they explain how different operational objectives read the same
measurements and are not additional benchmark runs. The artifact binds the benchmark,
constraints, and `extensions/policy-config.json` by SHA-256. Report generation also recomputes
each recommendation instead of trusting a supplied selected candidate.

### Bounded load evaluation

`extensions/load-evaluation.json` combines the per-candidate files under
`extensions/load/<candidate>/load-evaluation.json`. The declared load plan uses concurrency
levels 1, 2, and 4 with fixed prompts, output length, warmup count, measured request count, and
SLO thresholds. Each row retains request-level evidence and recomputable aggregates for:

- completion and error counts;
- request throughput and generated-token throughput;
- time-to-first-token and end-to-end latency distributions; and
- peak RSS when measured.

The evidence binding records the SHA-256 of `extensions/load-plan.json`, the exact request origin,
and both the load-server and canonical deployment command for every candidate. It retains command
digests and full argument arrays, verifies the request host and explicit port against the launched
server, and permits only the declared host and port binding differences. Model, runtime,
parallelism, thread, batch, micro-batch, context, and CPU settings must remain materially
equivalent.

Load results are not converted into cost, energy, capacity, or sustainability claims unless those
quantities are separately measured.

### Raw repeat-pass reconstruction

`extensions/benchmark-set-pass-1.json` and `extensions/benchmark-set-pass-2.json` are rebuilt from
the checksummed raw files in `experiment/candidates/<candidate>/`. `assemble-repeat-pass` does not
split the final aggregate in half. For each candidate and pass, it:

1. follows the canonical benchmark's bounded artifact references;
2. verifies the throughput settings, raw `llama-bench` JSONL, `llama-server` evaluation, and GNU
   `time -v` file against their recorded SHA-256 values;
3. recomputes prompt and generation medians from the raw pass;
4. validates quality cases and latency samples against the archived behavior suite;
5. parses peak RSS from the raw process measurement; and
6. carries forward only the candidate identity, parameters, and immutable model size from the
   canonical benchmark.

Each resulting benchmark set is labeled `supplementary-repeat-pass` and records its source
benchmark and source-artifact fingerprints.

`extensions/repeat-stability.json` then compares those validated pass benchmark sets. It binds
both pass files and candidate configurations, reports pass values and relative spread, and labels
the observed direction versus the baseline as `consistent`, `mixed`, or `no change`. Two passes
do not support a statistical-significance or confidence-interval claim, and ParetoPilot does not
make one.

### V1.1 report

`report-v1.1.html` is a deterministic, self-contained view of the canonical recommendation and
the additive policy, load, and stability evidence. The renderer validates input fingerprints,
candidate coverage, command bindings, and internally recomputed recommendations before drawing
the report. It does not run inference or change the selection decision.

## Replay semantics

`paretopilot replay` verifies `SHA256SUMS`, canonical completion status, safe relative paths, and
the full required artifact set before writing anything. For a v1.1 archive it regenerates and
compares:

- the canonical benchmark set and recommendation;
- policy profiles;
- the combined load evaluation from per-candidate load files;
- both pass benchmark sets from raw experiment artifacts; and
- the repeat-stability summary.

These are core comparisons: a missing or different core artifact makes
`decision_reproduced: false` and the replay invalid. The v1.0 and v1.1 HTML reports are
presentation outputs. If verified evidence and all core comparisons match but generated HTML
differs, replay remains valid, reports `fully_reproduced: false`, and emits a presentation
warning. This separation permits accessible layout improvements without describing them as new
measurements.

Replay never reruns the inference workload. A fresh hosted-runner workflow is new measurement
evidence, not a replay of the archived hardware environment.

## Trust rules

1. Every JSON artifact has a closed schema: duplicate keys, unknown fields, non-finite values,
   identity mismatches, incomplete records, and inconsistent aggregates fail closed.
2. Measured recommendations, policy profiles, load evidence, and stability evidence require
   SHA-256 input bindings.
3. Canonical and derived decisions are visibly distinguished in JSON and HTML.
4. Missing extension data is shown as not measured; it is never inferred.
5. A bundle-level `SHA256SUMS` covers core, raw, and extension artifacts.
6. Report generation and replay remain offline and dependency-free after extraction.

## Intended v1.1 bundle layout

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
