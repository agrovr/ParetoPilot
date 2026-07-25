# Supplementary Arm64 capacity study

ParetoPilot v1.4 adds a separate contract for measuring an operating envelope. It does not edit
the canonical v1.1 archive, recommendation, replay path, or report. This page describes the
predeclared method; measured results are added only after a native Arm64 run completes strict
assembly and recomputation, with every passing and failing per-cell outcome preserved.

The frozen v1.1 load sweep varied simultaneous clients while keeping llama-server at
`--parallel 1`. That correctly measured queue pressure for the canonical deployment command, but
it could not answer a second deployment question: how many server slots should be configured for
a bounded number of simultaneous clients?

## Predeclared question

The capacity workflow compares two already measured candidates:

| Candidate | Role |
| --- | --- |
| `q8-generic` | Frozen v1.1 canonical reference |
| `q4-kleidiai-tuned` | Frozen v1.1 measured resource alternative |

For each candidate, it measures the full 3×3 matrix:

- server slots (`--parallel`): 1, 2, and 4; and
- simultaneous clients: 1, 2, and 4.

This is a candidate-local operating-point decision. A capacity result cannot silently promote a
different candidate or rewrite the latency-first canonical choice.

## Why context scales with server slots

llama-server defines `--parallel` as the number of concurrent server slots, not the number of CPU
worker threads. llama.cpp's official server example scales total context when it increases
parallel slots because the configured context is shared across them. The capacity plan therefore
holds usable context at 2,048 tokens per slot:

| Server slots | Total `--ctx-size` | Context per slot |
| ---: | ---: | ---: |
| 1 | 2,048 | 2,048 |
| 2 | 4,096 | 2,048 |
| 4 | 8,192 | 2,048 |

Without this rule, increasing slots would also reduce each request's context budget and confound
the comparison.

Primary references:

- [Pinned llama-server parameters and endpoints](https://github.com/ggml-org/llama.cpp/blob/67b9b0e7f6ce45d929a4411907d3c48ec719e81c/tools/server/README.md)
- [Pinned llama-server shared-batch design](https://github.com/ggml-org/llama.cpp/blob/67b9b0e7f6ce45d929a4411907d3c48ec719e81c/tools/server/README-dev.md)
- [Pinned llama.cpp server example](https://github.com/ggml-org/llama.cpp/tree/67b9b0e7f6ce45d929a4411907d3c48ec719e81c#llama-server)
- [Deploy llama.cpp with KleidiAI on Arm servers](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama-cpu/)

The workflow pins llama.cpp to commit
`67b9b0e7f6ce45d929a4411907d3c48ec719e81c` and KleidiAI to `v1.24.0`; current upstream
documentation is methodology context, not a substitute for the pinned binary, command, build, and
runtime logs archived by the run.

## Controlled variables

The workflow downloads and fully replays the frozen v1.1 release before measuring anything. It
extracts the exact archived Q8 and tuned-Q4 server commands and compares each capacity command with
its candidate's archived command.

Only these values may differ:

1. `--parallel`, restricted to 1, 2, or 4;
2. `--ctx-size`, forced to `2,048 × --parallel`; and
3. `--port`, used only to isolate local server processes.

Runtime path, model path, CPU thread count, batch size, micro-batch size, GPU-offload setting,
verbosity, and host binding must remain identical. Any other argument drift fails the assembly.

## Counterbalanced order and measurement window

The predeclared order is counterbalanced by exact reversal:

1. forward: Q8 then tuned Q4; server slots 1, 2, then 4; clients 1, 2, then 4;
   and
2. reverse: tuned Q4 then Q8; server slots 4, 2, then 1; clients 4, 2, then 1.

Each performance server process starts fresh. The model loads before the health check. At every
client level, four warmup requests run immediately before eight measured requests. Cold startup is
outside the capacity window. GNU `time -v` wraps the load-only process and records peak RSS for the
complete 1/2/4-client sweep.

After both performance passes, six additional fresh server processes run the behavior guard once
for each candidate and server-slot level. Keeping quality runs separate gives both performance
passes the same workload history and binds every quality result to its exact command, port,
candidate, slot count, workflow run, and attempt.

Every load pass uses the existing checksummed plan:

- three fixed prompts;
- 64 output tokens;
- prompt caching disabled;
- four warmup requests per client level;
- eight measured requests per client level;
- 100% required completion;
- observed p95 TTFT at or below 2,000 ms; and
- observed p95 end-to-end latency at or below 6,500 ms.

Eight requests make this an observed p95, not a stable tail-distribution estimate. ParetoPilot does
not present p99 or confidence intervals from this sample.

## Quality, memory, and model-buffer-marker gates

The checksummed 24-case behavior suite runs sequentially for both candidates at every server-slot
level. It is a task-specific deployment guard, not a broad language-model quality benchmark or a
concurrent-correctness claim. Each candidate must:

- score at least 0.80;
- retain at least 95% of the Q8 reference score at the same slot level; and
- preserve its exact pass/fail outcomes across slot levels.

Peak server RSS must remain at or below 4,096 MiB. This is a predeclared example service budget,
not a natural Arm64 capacity boundary. A KleidiAI-enabled candidate must contain the
`CPU_KLEIDIAI model buffer` marker; the generic candidate must not contain it. This supports the
wording "KleidiAI-enabled build with an observed model-buffer marker." It does not prove that a
particular microkernel executed or attribute every performance difference to KleidiAI.

## Selection rule

For each candidate, a cell is eligible only when:

1. both counterbalanced passes meet the load SLO;
2. the quality gate passes for that server-slot level; and
3. the larger of the two process peak-RSS measurements is at or below 4,096 MiB;
4. forward/reverse generation-rate spread is at most 15%; and
5. forward/reverse observed-p95 end-to-end-latency spread is at most 20%.

Among eligible cells, ParetoPilot finds the highest median generated-token rate across the two
passes. Points within the predeclared 1% objective tolerance are treated as effectively tied; the
choice then favors lower median observed-p95 end-to-end latency, lower median observed-p95 TTFT,
lower maximum RSS, fewer server slots, then fewer clients. The P1/C1 reference is allowed to win.

Throughput is total completed generated tokens divided by each pass's common measured wall time.
ParetoPilot does not sum per-response server rates across overlapping requests.

## Run and retrieve it

Run **Supplementary Arm64 capacity study** manually from the repository's Actions page after the
workflow is on the default branch. The job uses GitHub's native Ubuntu 24.04 Arm64 runner, has a
300-minute timeout, and downloads about 3 GB of pinned Q8 and Q4 model files plus pinned source
archives. Models and build trees are not uploaded.

On success, download the `supplementary-arm64-capacity-<run>-<attempt>` artifact. Start with
`status.json`, verify `SHA256SUMS`, then read `capacity-receipt.md`; the full raw and embedded
evidence remains available for deeper review. A failed run uploads a seven-day `INCOMPLETE`
diagnostic bundle whose status records the failed stage.

## Evidence outputs

`paretopilot assemble-capacity` validates and binds:

- the capacity plan, load plan, source manifest, and frozen canonical evidence hashes;
- every measured load artifact and raw request sample;
- every GNU `time -v` file;
- every server log and model-buffer-marker result;
- every source-bound quality artifact and outcome vector;
- exact capacity and canonical command arrays; and
- native Arm64 runner, source, model, runtime, and build identity.

It embeds the validated load and quality source artifacts in `capacity-study.json`. ParetoPilot
recomputes every readable quality check, cell, gate, and selection whenever it validates that
artifact.

Exact input-file byte hashes are checked during assembly and again when the complete bundle is
reproduced. Standalone JSON validation recomputes the embedded canonical-content hashes; the
bundle-level `SHA256SUMS` binds the separate original plan, manifest, logs, and measurements.
`paretopilot capacity-receipt` renders a deterministic Markdown proof with selected points, the
complete two-candidate envelope, exact rejected gates, methodology, provenance, input hashes, and
a reproduction command.

Both outputs are classified `supplementary-capacity` and state
`canonical_outputs_modified: false`.

The successful GitHub Actions artifact is retained for 90 days. After the run is reviewed, the
compact final bundle is also published as an immutable release asset so judging does not depend on
temporary Actions retention.

## Boundary

This is a bounded fixed-concurrency study on one GitHub-hosted Ubuntu 24.04 Arm64 runner. It is not
an open-loop arrival-rate test, an MLPerf Server benchmark, a universal scaling claim, or evidence
about cost, energy, sustainability, or every Arm processor. A larger production study would need
controlled request arrivals, substantially more queries, longer steady-state windows, and stronger
tail-latency analysis.
