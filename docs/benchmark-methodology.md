# Initial benchmark methodology

This document defines the evidence standard for the ParetoPilot hackathon MVP. Canonical run
[`29940067201`](../results/published/29940067201/README.md) is the first measurement reviewed
under this protocol; the protocol itself does not imply that every listed metric has been
collected.

## Comparison contract

All candidates in a comparison must share:

- the same Arm64 runner instance, CPU identity, and operating-system image;
- model family, source revision, tokenizer, and prompt/evaluation set;
- runtime source commit and compiler version;
- power mode, process placement policy, and background-service policy;
- input/output lengths and end-to-end request behavior.

Each experiment changes one declared variable at a time. The first GitHub Actions experiment is:

1. pinned Qwen Q4_0 model with a generic `llama.cpp` CPU build;
2. the identical model and build flags with KleidiAI enabled;
3. a second KleidiAI pass;
4. a second generic pass.

This A-B-B-A order reduces, but cannot eliminate, time-dependent hosted-VM noise. Compare paired
ratios inside one job. Do not pool absolute throughput from separate workflow runs unless the CPU
identity and runner image are identical and the aggregation is explicitly justified.

## Required measurements

- model file size in MiB;
- peak resident set size in MiB;
- model load time;
- time to first token;
- prompt-processing and generation tokens per second;
- end-to-end p50 and p95 latency;
- requests per second at declared concurrency levels;
- quality score on a versioned evaluation set;
- Optional Arm Performix hotspot evidence for selected baseline and optimized runs when a target
  exposes the required hardware counters.

The first free-runner milestone measures prompt-processing and token-generation throughput.
Server-level latency, memory, quality-suite, and concurrency remain later experiments. Performix
is a later optional experiment; `llama-bench` output must not be presented as proof of those
metrics.

## Repetition and reporting

- Pin every input and tool revision before measuring.
- Warm up before recorded repetitions.
- Record at least ten repetitions for final evidence.
- Publish raw samples plus aggregation code.
- Report central tendency and variation; never report only the fastest sample.
- When optional profiling is enabled, keep the benchmark active long enough for meaningful
  profiler sampling.
- Treat cloud burst credits, noisy neighbors, and throttling as validity risks.
- Randomize or alternate candidate order to reduce thermal and noisy-neighbor bias.
- Use one authoritative producer per metric; do not infer TTFT from `llama-bench`.
- Treat a GitHub-hosted runner as an ephemeral paired test environment, not a guaranteed fixed
  cloud SKU.

## Quality gate

The default software fixture uses 95% retention of the baseline quality score. The final quality
metric and threshold will be declared before running the optimization search, not chosen after
seeing the results.

## Synthetic data

Files under `examples/` may contain synthetic numbers for testing the recommendation engine.
Every synthetic file must set `synthetic` to `true` and include a visible warning. Synthetic data
must not appear in the final submission as measured evidence.
