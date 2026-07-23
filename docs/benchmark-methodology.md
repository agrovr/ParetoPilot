# Benchmark methodology

This document defines the controlled four-candidate protocol used for ParetoPilot's published
v1.0 Arm64 study, followed by the additive v1.1 measurement contract in the current source. The
decision policy and all source, model, evaluation, and runtime pins are declared before
measurement.

Run `29973188507` and release `v1.0.0` remain the authoritative measured evidence. The v1.1
contract below describes implemented workflow gates and does not claim that new measurements
already exist.

## Published v1.0 decision question

Which measured configuration should be deployed when the primary objective is minimum p95
end-to-end latency, subject to full retention of the reference smoke-test score, a 15,000 ms p95
latency ceiling, and a 4,096 MiB peak-RSS ceiling?

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

All four candidates run in one native Arm64 GitHub Actions job and share:

- the same runner, CPU identity, operating-system image, and four CPU threads;
- a pinned `llama.cpp` commit and pinned KleidiAI source;
- pinned Qwen2.5 1.5B Instruct model revisions and file hashes;
- a 512-token batch, CPU-only execution, and one server slot;
- the same 512-token prompt-processing and 128-token generation shapes; and
- the same fixed five-case exact-answer evaluation suite.

The Q8 and Q4 files intentionally differ because quantization is one of the measured stages. Model
family, upstream revision, prompts, and every non-quantization setting remain fixed.

## Balanced execution order

Throughput and server measurements both use `A-B-C-D-D-C-B-A`, where A is the Q8 reference and D
is the tuned KleidiAI candidate. This gives every candidate one early and one late pass on the
same ephemeral host. Ten seconds separate server candidates.

The order reduces time-dependent hosted-runner bias, but it cannot turn an ephemeral runner into a
fixed hardware SKU. Results from separate runs are not pooled as though they came from one
controlled experiment.

## Measurements

- `llama-bench` records ten repetitions per pass for prompt processing and generation. The two
  passes provide twenty samples per candidate and workload.
- `llama-server` runs the five-case exact-answer suite in each pass and requires the outcomes to
  agree. A separate fixed 64-token performance prompt runs once as warmup and ten times for
  measurement per pass, giving twenty pooled latency samples per candidate.
- The server evaluator records exact-answer smoke score, time to first token, and end-to-end
  latency. Candidate summaries report p50 and p95 latency.
- GNU `time -v` records process maximum resident set size. The published candidate value is the
  larger of its two passes.
- Model file size is taken from the pinned, hash-verified GGUF file.

The five-case suite is a deterministic smoke gate, not a broad model-quality benchmark. Throughput,
quality, latency, and memory each come from their declared producer; one metric is never inferred
from another.

## Additive v1.1 measurement contract

V1.1 preserves the same candidates, balanced order, single-runner boundary, throughput shapes,
fixed 64-token streamed latency workload, and canonical selection objective. It adds three
supplementary evidence layers plus precomputed policy scenarios.

### Stricter behavior gate

The checksummed `paretopilot-qwen-behavior-v2` suite contains 24 deterministic cases:

- instruction following, extraction, classification, arithmetic, and factual cases use
  `trimmed-exact`, which removes only surrounding whitespace; and
- four structured-output cases use `json-exact`, which strictly parses JSON, rejects duplicate
  keys and non-standard constants, and compares a canonical structural representation.

Every candidate must satisfy the declared 0.80 absolute quality floor and retain at least 95% of
the measured baseline score. These are narrow reproducibility gates, not estimates of general
model quality.

These v1.1 limits were declared after
[non-canonical exploratory run `30050573298`](https://github.com/agrovr/ParetoPilot/actions/runs/30050573298)
and before any canonical v1.1 measurement. The 24 binary cases have a 1/24, or 4.17
percentage-point, resolution. In both passes, the Q8 reference scored 21/24 and every Q4
candidate scored 20/24. The combined rule therefore requires at least 20/24 for this measured
reference and rejects 19/24.

The calibration did not remove or recategorize failures. Q8 missed one single-word casing case
and returned two otherwise-valid JSON objects inside code fences. Each Q4 candidate returned a
draft date instead of the requested final date and fenced three otherwise-valid JSON responses.
Those responses and outcomes remain visible and checksummed. The one-case net difference does
not mean the candidates failed the same cases, are behaviorally equivalent, or have the same
general model quality.

### Bounded load plan

Each candidate runs a fixed load plan at concurrency 1, 2, and 4 against one `llama-server`
process. The plan declares three prompts, 64 output tokens, four warmup requests, eight measured
requests at each level, a 100% completion requirement, a 2,000 ms p95 TTFT ceiling, and a 6,500 ms
p95 end-to-end ceiling.

The evaluator retains success and error samples and recomputes request throughput,
generated-token throughput, p50/p95 TTFT, p50/p95 end-to-end latency, completion rate, and SLO
status. The load artifact binds:

- the exact load-plan digest;
- the explicit request origin used by the evaluator;
- the exact load-server command and digest; and
- the candidate's canonical deployment command and digest.

Only host and port may differ between the load and canonical commands. Runtime, model,
parallelism, thread counts, batch, micro-batch, context, and CPU settings remain equivalent. Load
rows are supplementary operational evidence; they do not alter the canonical single-client
recommendation.

### Repeat-pass reconstruction and stability

The two pass benchmark sets are rebuilt independently from the checksummed raw files already
referenced by the canonical benchmark. For every candidate, each pass verifies and parses its
throughput settings, `llama-bench` JSONL, server evaluation, and GNU `time -v` output. It
recomputes pass medians, validates behavior cases and raw latency samples against the archived
suite, and parses peak RSS.

The stability summary compares the two reconstructed pass sets for prompt throughput, generation
throughput, quality, TTFT, end-to-end latency, and peak RSS. It records the direction versus the
baseline and relative spread. Labels such as `consistent`, `mixed`, and `no change` describe the
two observed passes only. No p-value, confidence interval, or statistical-significance claim is
made.

### Policy scenarios

The canonical latency policy reuses the declared constraints and preference order and must match
the core recommendation. Four `derived-non-canonical` policies change only the objective to
memory, TTFT, prompt throughput, or generation throughput. They are sensitivity views over the
same measurements, not extra benchmark trials and not competing canonical winners.

## Selection rules

ParetoPilot first rejects candidates that fail the quality, latency, or memory constraints. It then
computes the non-dominated frontier across latency, generation throughput, model size, peak RSS,
quality, and TTFT.

Among eligible candidates, the engine minimizes p95 end-to-end latency. Candidates within 1% of
the numeric best enter a deterministic shortlist ordered from the reference through quantization,
Arm kernels, and runtime tuning. The resulting recommendation explains the numeric best, the
shortlist, any preference-based change, rejected candidates, and frontier membership.

## Evidence safeguards

- Source revisions, model hashes, evaluation-suite hash, build flags, executable hashes, exact
  command arrays, and runtime settings are recorded.
- Generic server logs must not contain the `CPU_KLEIDIAI model buffer` dispatch marker. Both
  KleidiAI candidates must contain it in both passes.
- Strict parsers reject malformed, duplicate-key, non-finite, mismatched, oversized, path-escaping,
  or synthetic source data.
- The experiment manifest binds critical artifacts by SHA-256. A bundle-level `SHA256SUMS` binds
  every released evidence file.
- V1.1 additionally binds the behavior suite itself, load plan, load and canonical command files,
  request endpoints, policy configuration, reconstructed pass inputs, and candidate-configuration
  fingerprints.
- A run is canonical only when it uses the default branch, exactly ten repetitions, and passes
  every environment, measurement, dispatch, integrity, selection, and reporting gate.

## Limitations

The study is one controlled comparison on one GitHub-hosted Arm Neoverse runner. It does not claim
that the same ranking applies to every Arm processor, model, prompt distribution, concurrency
level, or deployment environment. Two balanced passes do not establish statistical significance.
Energy and cost were not measured.

Arm Performix is an optional follow-up for hotspot analysis when a compatible target exposes the
required counters. It is not required by ParetoPilot and does not replace benchmark evidence.
