# Initial benchmark methodology

This document defines the evidence standard for the ParetoPilot hackathon MVP. It is a protocol,
not a claim that measurements have already been collected.

## Comparison contract

All candidates in a comparison must share:

- the exact Arm64 target SKU and operating-system image;
- model family, source revision, tokenizer, and prompt/evaluation set;
- runtime source commit and compiler version;
- power mode, process placement policy, and background-service policy;
- input/output lengths and end-to-end request behavior.

Each experiment changes one declared variable at a time. The initial sequence is:

1. generic FP16 or Q8 baseline;
2. quantized Q4 model with the generic build;
3. identical Q4 model with KleidiAI enabled;
4. identical KleidiAI build with tuned runtime parameters.

## Required measurements

- model file size in MB;
- peak resident set size in MB;
- model load time;
- time to first token;
- prompt-processing and generation tokens per second;
- end-to-end p50 and p95 latency;
- requests per second at declared concurrency levels;
- quality score on a versioned evaluation set;
- Arm Performix hotspot evidence for selected baseline and optimized runs.

## Repetition and reporting

- Pin every input and tool revision before measuring.
- Warm up before recorded repetitions.
- Record at least ten repetitions for final evidence.
- Publish raw samples plus aggregation code.
- Report central tendency and variation; never report only the fastest sample.
- Keep the benchmark active long enough for meaningful profiler sampling.
- Treat cloud burst credits, noisy neighbors, and throttling as validity risks.
- Randomize or alternate candidate order to reduce thermal and noisy-neighbor bias.
- Use one authoritative producer per metric; do not infer TTFT from `llama-bench`.

## Quality gate

The default software fixture uses 95% retention of the baseline quality score. The final quality
metric and threshold will be declared before running the optimization search, not chosen after
seeing the results.

## Synthetic data

Files under `examples/` may contain synthetic numbers for testing the recommendation engine.
Every synthetic file must set `synthetic` to `true` and include a visible warning. Synthetic data
must not appear in the final submission as measured evidence.
