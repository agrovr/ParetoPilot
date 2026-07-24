# Benchmark methodology

This document defines the controlled four-candidate protocol used for ParetoPilot's current
canonical v1.1 Arm64 study. The decision policy and all source, model, evaluation, load, and
runtime pins were declared before the canonical measurement.

Run [`30055662526`](https://github.com/agrovr/ParetoPilot/actions/runs/30055662526) is preserved
in release [`v1.1.0`](https://github.com/agrovr/ParetoPilot/releases/tag/v1.1.0). The earlier
[v1.0 study](../results/published/29973188507/README.md) remains historical evidence from a
separate runner and is not pooled with v1.1.

## Decision question

Which measured configuration should be deployed when the primary objective is minimum p95
end-to-end latency, subject to:

- a 0.80 absolute behavior-score floor;
- at least 95% retention of the measured Q8 reference score;
- a 15,000 ms p95 end-to-end latency ceiling; and
- a 4,096 MiB peak-RSS ceiling?

The reference is allowed to win. A predeclared 1% objective tolerance keeps the simpler candidate
when a smaller measured latency difference is not large enough to justify added complexity.

## Candidates

Each stage changes one declared variable relative to the preceding stage.

| Stage | Candidate | Deliberate change |
| --- | --- | --- |
| Reference | `q8-generic` | Q8_0 model on the generic CPU build |
| Quantization | `q4-generic` | Q8_0 to Q4_0 with the same generic build |
| Arm kernels | `q4-kleidiai` | Generic to KleidiAI-enabled build with the same Q4_0 model |
| Runtime tuning | `q4-kleidiai-tuned` | Micro-batch size 128 to 512 |

## Controlled inputs

All four candidates ran in one native Arm64 GitHub Actions job and shared:

- Ubuntu 24.04 on one 4-vCPU Arm Neoverse-N2 runner;
- four CPU threads and CPU-only execution;
- `llama.cpp` commit `67b9b0e7f6ce45d929a4411907d3c48ec719e81c`;
- KleidiAI `1.24.0`;
- Qwen2.5 1.5B Instruct revision `91cad51170dc346986eccefdc2dd33a9da36ead9`;
- evaluation-suite SHA-256
  `e49c16fba32fd65c947264aef4141026ab68b1fd415ef09eeea6e8ade9a545c7`;
- a 512-token batch, one server slot, and fixed prompt and generation shapes; and
- the same checksummed behavior, latency, and load inputs.

The Q8 and Q4 files intentionally differ because quantization is one measured stage. Model
family, upstream revision, prompts, and every non-quantization setting stay controlled.

## Balanced execution order

Throughput and server measurements use `A-B-C-D-D-C-B-A`, where A is the Q8 reference and D is
the tuned KleidiAI candidate. This gives each candidate one early and one late pass on the same
ephemeral host. Ten seconds separate server candidates.

The order reduces time-dependent hosted-runner bias, but it cannot turn an ephemeral runner into a
fixed hardware SKU. Results from separate runs are not pooled as though they were one controlled
experiment.

## Measurements

- `llama-bench` records ten repetitions per pass for prompt processing and generation. The two
  passes provide twenty samples per candidate and workload.
- `llama-server` runs the 24-case deterministic behavior suite and the fixed 64-token streamed
  latency workload. Candidate summaries report p50 and p95 TTFT and end-to-end latency.
- GNU `time -v` records process maximum resident set size. The candidate value is the larger
  measured pass value.
- Model size comes from the pinned, hash-verified GGUF file.
- A separate bounded load stage runs the same declared 1/2/4-client plan for every candidate.

Throughput, behavior, latency, memory, and model size each come from their declared producer; one
metric is never inferred from another.

## Expanded behavior gate

The checksummed `paretopilot-qwen-behavior-v2` suite contains 24 deterministic cases:

- instruction following, extraction, classification, arithmetic, and factual cases use
  `trimmed-exact`, which removes only surrounding whitespace; and
- four structured-output cases use `json-exact`, which strictly parses JSON, rejects duplicate
  keys and non-standard constants, and compares a canonical structural representation.

Every candidate must satisfy the declared 0.80 absolute quality floor and retain at least 95% of
the measured baseline score. These are narrow reproducibility gates, not estimates of general
model quality.

The thresholds were declared after incomplete diagnostic
[run `30050573298`](https://github.com/agrovr/ParetoPilot/actions/runs/30050573298) and before the
canonical v1.1 measurement. That diagnostic correctly recorded invalid evidence and was used only
to calibrate the pre-canonical rule. A 24-case binary suite has 1/24, or 4.17 percentage-point,
resolution, so the declared gate permits one net case below a 21/24 reference and rejects 19/24.

The canonical run then measured:

| Candidate | Passing cases | Score | Gate |
| --- | ---: | ---: | --- |
| `q8-generic` | 21/24 | 0.8750 | Pass |
| `q4-generic` | 20/24 | 0.8333 | Pass |
| `q4-kleidiai` | 20/24 | 0.8333 | Pass |
| `q4-kleidiai-tuned` | 20/24 | 0.8333 | Pass |

Case-level prompts, answers, accepted values, match modes, and outcomes remain in the checksummed
archive. A one-case net difference is not a claim of behavioral equivalence or general model
quality.

## Bounded load plan

Each candidate runs a fixed load plan at concurrency 1, 2, and 4 against one `llama-server`
process. The plan declares:

- three fixed prompts and 64 output tokens;
- four warmup requests per level;
- eight measured requests per level;
- 100% required completion;
- a 2,000 ms p95 TTFT ceiling; and
- a 6,500 ms p95 end-to-end ceiling.

The evaluator retains success and error samples and recomputes request throughput,
generated-token throughput, p50/p95 TTFT, p50/p95 end-to-end latency, completion rate, and SLO
status. The load artifact binds:

- the exact load-plan digest;
- the explicit request origin used by the evaluator;
- the exact load-server command and digest; and
- the candidate's canonical deployment command and digest.

Only host and port may differ between load and canonical commands. Runtime, model, parallelism,
thread counts, batch, micro-batch, context, and CPU settings stay equivalent.

All four candidates completed 100% of the canonical requests. Concurrency 1 was the highest level
that met both latency SLOs for every candidate. This bounded sweep is supplementary operational
evidence; it does not change the canonical single-client recommendation or establish capacity
beyond the tested levels.

## Repeat-pass reconstruction and stability

The two pass benchmark sets are rebuilt independently from the checksummed raw files referenced
by the canonical benchmark. For every candidate, each pass verifies and parses its throughput
settings, `llama-bench` JSONL, server evaluation, and GNU `time -v` output. It recomputes pass
medians, validates behavior cases and raw latency samples against the archived suite, and parses
peak RSS.

The stability summary compares prompt throughput, generation throughput, behavior score, TTFT,
end-to-end latency, and peak RSS. The canonical summary has 24 rows: six metrics for each of four
candidates. All six metrics were directionally consistent across both passes for each of the
three Q4 candidates. The largest relative spread was:

| Candidate | Maximum relative spread |
| --- | ---: |
| `q4-generic` | 1.6695% |
| `q4-kleidiai` | 1.3919% |
| `q4-kleidiai-tuned` | 0.8029% |

Labels such as `consistent`, `mixed`, and `no change` describe the two observed passes only. No
p-value, confidence interval, or statistical-significance claim is made.

## Policy scenarios

The canonical latency policy reuses the declared constraints and preference order and must match
the core recommendation. Four `derived-non-canonical` policies change only the objective. They
are sensitivity views over the same measurements, not extra benchmark trials.

| Profile | Objective | Selected candidate |
| --- | --- | --- |
| `canonical-latency` | Minimum p95 end-to-end latency with 1% tolerance | `q8-generic` |
| `memory-first` | Minimum peak RSS | `q4-generic` |
| `first-token-first` | Minimum p95 TTFT | `q4-kleidiai-tuned` |
| `prompt-ingest-first` | Maximum prompt throughput | `q4-kleidiai-tuned` |
| `decode-first` | Maximum generation throughput | `q8-generic` |

## Canonical selection

ParetoPilot first rejects candidates that fail the quality, latency, or memory constraints. It
then computes the non-dominated frontier across latency, generation throughput, model size, peak
RSS, quality, and TTFT.

All four candidates were eligible and stayed on the Pareto frontier. Q8 had the lowest measured
p95 end-to-end latency at 2231.932869 ms. Its 1% cutoff was 2254.2522 ms, and no Q4 candidate
entered that shortlist. ParetoPilot therefore selected `q8-generic` as both the numeric winner and
the canonical recommendation.

The tuned Q4 + KleidiAI candidate remains a useful measured resource alternative: versus Q8, it
used a 43.72% smaller model and 42.79% less peak RSS, reduced p95 TTFT by 13.83%, and increased
prompt throughput by 28.10%. It was also 3.40% slower on the canonical p95 end-to-end objective,
9.37% lower on generation throughput, and one behavior case lower.

## Evidence safeguards

- Source revisions, model hashes, evaluation-suite hash, build flags, executable hashes, exact
  command arrays, and runtime settings are recorded.
- Generic server logs must not contain the `CPU_KLEIDIAI model buffer` dispatch marker. Both
  KleidiAI candidates must contain it.
- Strict parsers reject malformed, duplicate-key, non-finite, mismatched, oversized,
  path-escaping, or synthetic source data.
- The experiment manifest binds critical artifacts by SHA-256. A bundle-level `SHA256SUMS` binds
  150 released payloads.
- V1.1 binds the behavior suite, load plan, load and canonical command files, request endpoints,
  policy configuration, reconstructed pass inputs, and candidate-configuration fingerprints.
- A run is canonical only when it uses the default branch, exactly ten repetitions, and passes
  every environment, measurement, dispatch, integrity, selection, and reporting gate.
- Independent replay must reproduce the selected candidate and the archived decision artifacts
  without rerunning inference.

## Limitations

The study is one controlled comparison for one model and workload on one GitHub-hosted Arm
Neoverse runner. It does not claim the same ranking for every Arm processor, model, prompt
distribution, concurrency level, or deployment environment. The behavior suite is deterministic
and bounded. Two balanced passes do not establish statistical significance. Energy and cost were
not measured.

Arm Performix is an optional follow-up for hotspot analysis when a compatible target exposes the
required counters. It is not required by ParetoPilot and does not replace benchmark evidence.
