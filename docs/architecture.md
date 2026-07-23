# ParetoPilot architecture

![ParetoPilot evidence-to-deployment architecture](assets/architecture.svg)

ParetoPilot is an evidence-to-deployment decision pipeline. It does not sit in the inference
request path. Instead, it runs controlled candidates on one native Arm64 runner, validates the
resulting artifacts, applies declared quality and resource constraints, and exports a reproducible
deployment recommendation with an offline report.

## End-to-end flow

1. **Pin the experiment.** The candidate-study workflow fixes the model revisions and hashes,
   `llama.cpp` commit, KleidiAI release, evaluation suite, benchmark shape, and decision
   constraints before measurement begins.
2. **Build on native Arm64.** One `ubuntu-24.04-arm` job builds CPU-only generic and
   KleidiAI-enabled `llama.cpp` binaries and records the runner, operating system, compiler,
   build options, executable hashes, and exact launch arguments.
3. **Run attributable candidates.** Four candidates separate the Q8 reference, Q4 quantization,
   Arm kernel dispatch, and one runtime micro-batch change. Throughput and server measurements use
   the balanced order A-B-C-D-D-C-B-A on the same hosted runner.
4. **Measure two evidence lanes.** `llama-bench` produces prompt and generation throughput.
   `llama-server` runs the fixed evaluation suite and records exact-match quality, streamed TTFT,
   end-to-end latency for fixed 64-token generations, and GNU `time -v` peak RSS. Two server
   passes are pooled per candidate.
5. **Assemble strictly.** `paretopilot assemble-experiment` verifies the closed manifest schema,
   artifact SHA-256 digests, candidate identities, model and runtime pins, evaluation-suite
   identity, exact throughput and deployment arguments, balanced aggregate recomputation, and
   captured KleidiAI dispatch logs before producing a `BenchmarkSet`.
6. **Decide under declared constraints.** The recommendation engine rejects candidates that fail
   quality or resource gates, computes the Pareto frontier, and minimizes the declared objective.
   A predeclared 1% objective tolerance and preference order retain the simpler candidate when the
   numeric latency lead is too small to justify extra complexity.
7. **Export reviewable outputs.** ParetoPilot writes `recommendation.json`, a self-contained
   `report.html`, candidate and environment evidence, `status.json`, and a bundle-level
   `SHA256SUMS` file.

## Candidate attribution

| Candidate | Deliberate change | Attribution stage |
| --- | --- | --- |
| `q8-generic` | Q8_0 model on the generic CPU build | Reference baseline |
| `q4-generic` | Q4_0 model on the generic CPU build | Quantization |
| `q4-kleidiai` | Same Q4_0 model with the KleidiAI build | Arm kernel |
| `q4-kleidiai-tuned` | Same KleidiAI candidate with micro-batch size 512 | Runtime tuning |

The workflow hashes and re-verifies runtime logs: generic candidates must not report the
`CPU_KLEIDIAI model buffer`, while both KleidiAI candidates must report it. That check proves the
intended dispatch distinction without treating a build flag alone as runtime evidence.

## Evidence and decision boundaries

- All candidate comparisons belong to one ephemeral Arm64 job. Results from different processor
  identities or runner images are not pooled as if they were one controlled experiment.
- Missing, malformed, mismatched, non-finite, or digest-invalid source data fails assembly; the
  pipeline does not estimate absent measurements.
- Quality, latency, throughput, and peak RSS have separate authoritative producers. For example,
  TTFT is not inferred from `llama-bench` output.
- The candidate workflow marks a run canonical only when it uses the default branch, the declared
  ten repetitions, and every measurement, integrity, selection, and reporting gate passes.
  Other successful runs remain explicitly exploratory.
- Arm Performix is an optional follow-up for hotspot analysis when the target exposes the required
  profiling capabilities. It is outside the required path and never blocks measurement,
  selection, or report generation.

## Implementation map

| Component | Responsibility |
| --- | --- |
| [`.github/workflows/candidate-study-arm64.yml`](../.github/workflows/candidate-study-arm64.yml) | Native Arm64 build, measurement, provenance capture, integrity checks, and artifact upload |
| [`evals/qwen-smoke-v1.json`](../evals/qwen-smoke-v1.json) | Versioned fixed quality evaluation inputs |
| [`src/paretopilot/llama_summary.py`](../src/paretopilot/llama_summary.py) | Validated multi-pass throughput aggregation |
| [`src/paretopilot/server_eval.py`](../src/paretopilot/server_eval.py) | Exact-match quality and streamed server-latency evaluation |
| [`src/paretopilot/experiment.py`](../src/paretopilot/experiment.py) | Strict multi-candidate manifest and artifact assembly |
| [`src/paretopilot/analysis.py`](../src/paretopilot/analysis.py) | Constraint filtering, Pareto frontier, and deterministic selection |
| [`configs/constraints.candidate-study.json`](../configs/constraints.candidate-study.json) | Declared quality, latency, memory, frontier, and objective policy |
| [`src/paretopilot/report.py`](../src/paretopilot/report.py) | Deterministic, dependency-free HTML decision report |

## Truthful interpretation

The diagram documents the executable candidate-study path; it is not itself benchmark evidence.
A run is publishable only after its `status.json`, raw measurements, provenance, dispatch logs,
and checksums pass review. ParetoPilot reports measured software performance and does not claim
energy savings or hardware-counter findings unless a separate source actually measures them.

Canonical run [`29973188507`](../results/published/29973188507/README.md) completed this path and
was rebuilt from its permanent release archive in a separate verification pass.
