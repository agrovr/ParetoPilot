# Arm64 capacity study

ParetoPilot v1.4 adds a separate server-capacity study. It tests how server slots and simultaneous
clients affect the two candidates carried forward from the v1.1 comparison. It does not change the
published v1.1 latency result.

The v1.1 load test varied simultaneous clients while `llama-server` used `--parallel 1`. It
measured queue pressure, but not how performance changes when the server-slot count changes.

## Question

The capacity workflow compares two already measured candidates:

| Candidate | Role |
| --- | --- |
| `q8-generic` | Published v1.1 latency choice |
| `q4-kleidiai-tuned` | Published v1.1 lower-memory alternative |

For each candidate, it measures the full 3×3 matrix:

- server slots (`--parallel`): 1, 2, and 4; and
- simultaneous clients: 1, 2, and 4.

The workflow chooses an operating point for each candidate separately. It does not compare the
candidates or change the v1.1 model choice.

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

## Test order and timing

The predeclared order uses one forward pass and its exact reverse:

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

## Quality and memory checks

The checksummed 24-case behavior suite runs sequentially for both candidates at every server-slot
level. It is a task-specific deployment guard, not a broad language-model quality benchmark or a
concurrent-correctness claim. Each candidate must:

- score at least 0.80;
- retain at least 95% of the Q8 reference score at the same slot level; and
- preserve its exact pass/fail outcomes across slot levels.

Peak server RSS must remain at or below 4,096 MiB. This is the memory limit chosen for this study,
not a universal Arm64 limit. A KleidiAI-enabled candidate must contain the
`CPU_KLEIDIAI model buffer` marker; the generic candidate must not contain it. The marker confirms
which build `llama.cpp` reported. It is not a kernel trace and does not attribute every
performance difference to KleidiAI.

## Selection rule

For each candidate, a cell is eligible only when:

1. both mirrored passes meet the load SLO;
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

## Published result and replay

Published [run `30144901854`](../results/published/30144901854/README.md) completed this study on
one native Arm64 runner. Both candidates selected P4/C4 among their own passing cells, and the
published Q8 model choice did not change. The
[`v1.4.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.4.0) preserves the original
Actions ZIP unchanged.

Use the current `main` branch and the command below to rebuild the capacity outputs and replay the
embedded v1.1 result without rerunning inference:

```bash
python -m paretopilot replay-capacity evidence \
  --output-dir output/replay-capacity
```

`replay-capacity` checks every `SHA256SUMS` entry, reconstructs the capacity study and receipt from
the archived load, RSS, server-log, and quality sources, requires both outputs to match byte for
byte, and replays the embedded v1.1 result into a new output directory.

To produce new results, run **Arm64 capacity study** manually from the repository's
Actions page after the workflow is on the default branch. A new hosted-runner execution is a new
measurement, not an exact hardware replay.

The job uses GitHub's native Ubuntu 24.04 Arm64 runner, has a 300-minute timeout, and downloads
about 3 GB of pinned Q8 and Q4 model files plus pinned source archives. Models and build trees are
not uploaded.

On success, download the `supplementary-arm64-capacity-<run>-<attempt>` artifact. Start with
`status.json`, verify `SHA256SUMS`, then read `capacity-receipt.md`. The artifact also includes the
raw measurements and embedded v1.1 archive. A failed run uploads a seven-day `INCOMPLETE`
diagnostic bundle whose status records the failed stage.

## Output files

`paretopilot assemble-capacity` checks:

- the capacity plan, load plan, source manifest, and published v1.1 result hashes;
- every measured load artifact and raw request sample;
- every GNU `time -v` file;
- every server log and model-buffer-marker result;
- every source-bound quality artifact and outcome vector;
- exact capacity and published v1.1 command arrays; and
- native Arm64 runner, source, model, runtime, and build identity.

`capacity-study.json` contains the validated load and quality inputs. ParetoPilot recalculates each
readable quality check, cell result, and selection whenever it validates the file.

Input-file hashes are checked during assembly and again during replay. `SHA256SUMS` covers the
original plan, manifest, logs, and measurements. `paretopilot capacity-receipt` writes a Markdown
summary with the selected points, every cell and rejection reason, the method, source details,
input hashes, and a replay command.

Both outputs are classified `supplementary-capacity` and state
`canonical_outputs_modified: false`.

The successful GitHub Actions artifact is retained for 90 days. The `v1.4.0` release keeps the
original archive beyond temporary Actions retention and records its SHA-256. GitHub guarantees
release immutability only when that setting is enabled for the release.

## Study limits

This is a bounded fixed-concurrency study on one GitHub-hosted Ubuntu 24.04 Arm64 runner. It is not
an open-loop arrival-rate test, an MLPerf Server benchmark, a universal scaling claim, or evidence
about cost, energy, sustainability, or every Arm processor. A larger production study would need
controlled request arrivals, substantially more queries, longer steady-state windows, and stronger
tail-latency analysis.
