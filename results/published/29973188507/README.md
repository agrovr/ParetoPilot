# Canonical Arm64 study 29973188507

[GitHub Actions run `29973188507`](https://github.com/agrovr/ParetoPilot/actions/runs/29973188507)
completed on July 23, 2026 UTC. The workflow classified the study as canonical with
`measurement_valid: true` and `valid_evidence: true`.

## Decision

ParetoPilot retained **Q8 generic reference** for the declared p95 end-to-end latency objective.
Q4 generic was the numeric best at 2330.914 ms, but its 0.214% lead over Q8 at 2335.917 ms was
inside the predeclared 1% objective tolerance. The deterministic simpler-first preference
therefore kept the reference instead of treating the below-tolerance difference as a deployment
win.

All four candidates passed the quality and resource gates, and all four remained on the Pareto
frontier. The decision shortlist contained Q8 generic, Q4 generic, and Q4 + KleidiAI tuned.

## Results

| Candidate | E2E p95 | TTFT p95 | Prompt tok/s | Generation tok/s | Peak RSS | Model size | Smoke gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q8 generic reference | 2335.917 ms | 553.415 ms | 101.7475 | 36.5399 | 3437.613 MiB | 1806.767 MiB | 5/5 |
| Q4 generic quantization | **2330.914 ms** | 483.327 ms | 113.5460 | 34.7130 | 1966.461 MiB | 1016.834 MiB | 5/5 |
| Q4 + KleidiAI | 2393.537 ms | **472.764 ms** | 114.1015 | 34.5675 | 1966.477 MiB | 1016.834 MiB | 5/5 |
| Q4 + KleidiAI tuned | 2337.799 ms | 472.799 ms | **131.2565** | 34.6329 | 1966.477 MiB | 1016.834 MiB | 5/5 |

No candidate won every metric. Compared with Q8, the tuned Q4 + KleidiAI candidate used a 43.7%
smaller model and 42.8% less peak RSS, improved prompt throughput by 29.0%, and reduced p95 TTFT
by 14.6%. Its p95 end-to-end latency was 0.08% higher and generation throughput was 5.2% lower.
Those visible tradeoffs are the reason ParetoPilot uses a declared decision policy instead of one
headline speedup.

## Experiment identity

- One GitHub-hosted Ubuntu 24.04 Arm64 runner: Arm Neoverse-N2, 4 vCPUs.
- ParetoPilot `1.0.0` at commit
  [`0c144b1`](https://github.com/agrovr/ParetoPilot/commit/0c144b1aae1b26a8cffa7356eec346457188edc6).
- Pinned `llama.cpp` commit `67b9b0e7f6ce45d929a4411907d3c48ec719e81c`.
- Pinned KleidiAI `v1.24.0` with a recorded source-archive hash.
- Pinned Qwen2.5 1.5B Instruct Q8_0 and Q4_0 files with recorded SHA-256 hashes.
- Balanced `A-B-C-D-D-C-B-A` throughput and server order on the same runner.
- Twenty throughput samples per candidate and workload; twenty fixed 64-token latency samples
  per candidate after two warmups.
- Five exact-answer smoke cases repeated in both passes with matching outcomes.

## Verification

The original 234,873-byte Actions artifact is preserved unchanged in the
[`v1.0.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.0.0). Its SHA-256 is:

```text
fb4f4c86a729a5eb42e23dbd3c6346fd4ab31ce14423dbb8c7672b11b6a6fd00
```

A separate verification pass checked all 124 `SHA256SUMS` entries and exact file coverage. Generic
server passes contained zero KleidiAI dispatch markers; each KleidiAI server pass contained exactly one.
Using the tagged `v1.0.0` source, rebuilding the benchmark set and regenerating the recommendation
and archived report produced exact byte-for-byte matches. The machine-readable archive lock is in
[`evidence.json`](evidence.json). The live report may receive presentation-only improvements while
continuing to rebuild from the same verified benchmark set and exact recommendation.

## Limits

This is one controlled comparison on one ephemeral hosted Arm64 runner. The five-case evaluation
is a smoke gate, not a broad language-model quality benchmark. The study did not measure energy,
cost, concurrency, or performance on every Arm processor, model, or prompt distribution.
