# Benchmark methodology

This document defines the controlled four-candidate protocol used for ParetoPilot's canonical
Arm64 study. The decision policy and all source, model, evaluation, and runtime pins are declared
before measurement.

## Decision question

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
- A run is canonical only when it uses the default branch, exactly ten repetitions, and passes
  every environment, measurement, dispatch, integrity, selection, and reporting gate.

## Limitations

The study is one controlled comparison on one GitHub-hosted Arm Neoverse runner. It does not claim
that the same ranking applies to every Arm processor, model, prompt distribution, concurrency
level, or deployment environment. Energy and cost were not measured.

Arm Performix is an optional follow-up for hotspot analysis when a compatible target exposes the
required counters. It is not required by ParetoPilot and does not replace benchmark evidence.
