# ParetoPilot architecture

![ParetoPilot evidence-to-deployment architecture](assets/architecture.svg)

ParetoPilot is an evidence-to-deployment decision pipeline. It does not sit in the inference
request path. Instead, it runs controlled candidates on one native Arm64 runner, validates the
resulting artifacts, applies declared quality and resource constraints, and exports a reproducible
deployment recommendation with an offline report.

The diagram shows the core candidate-study path completed by the published v1.0 run. The current
code adds a v1.1 evidence lane after the same strict core; no canonical v1.1 measurement is
published yet.

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

## Additive v1.1 evidence lane

The v1.1 workflow extends the core without changing its `BenchmarkSet` or recommendation schema:

1. **Bind the behavior contract.** A 24-case suite is copied into the experiment, identified in
   the closed manifest, and verified by SHA-256. Assembly checks every case, answer, match mode,
   generation length, and pooled server result against that exact file.
2. **Measure bounded concurrency.** Each candidate runs the same declared 1/2/4-client load plan.
   Per-candidate artifacts retain raw request samples, SLO results, the request origin, and both
   the exact load and canonical server commands. Only host and port binding differences are
   allowed.
3. **Reconstruct each balanced pass.** `assemble-repeat-pass` follows the source references already
   bound in the canonical benchmark, verifies the raw throughput, settings, server-evaluation,
   and process-memory files, and recomputes one supplementary `BenchmarkSet` per pass.
4. **Describe stability without overclaiming.** The two reconstructed passes are compared for
   observed direction and relative spread. They are not used to claim statistical significance.
5. **Precompute decision scenarios.** One canonical policy and four non-canonical profiles are
   evaluated from the same benchmark set. The canonical profile must reproduce the core
   recommendation.
6. **Render, then support offline replay.** The measurement workflow renders
   `report-v1.1.html` from the bound core and extension evidence. After a complete canonical
   bundle is downloaded, the separate offline replay command rebuilds every core decision
   artifact; report-only differences remain presentation warnings. Replay is not run inside the
   measurement job.

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
- A repeat-pass benchmark is reconstructed from raw pass files; it is not estimated by splitting
  a pooled median or copying the final aggregate.
- Load evidence must match its declared plan, request endpoint, candidate identity, and server
  commands. A successful HTTP response alone is not sufficient provenance.
- The candidate workflow marks a run canonical only when it uses the default branch, the declared
  ten repetitions, and every measurement, integrity, selection, and reporting gate passes.
  Other successful runs remain explicitly exploratory.
- Two balanced passes support an observed consistency description, not a significance or
  confidence-interval claim.
- Arm Performix is an optional follow-up for hotspot analysis when the target exposes the required
  profiling capabilities. It is outside the required path and never blocks measurement,
  selection, or report generation.

## Implementation map

| Component | Responsibility |
| --- | --- |
| [`.github/workflows/candidate-study-arm64.yml`](../.github/workflows/candidate-study-arm64.yml) | Native Arm64 build, measurement, provenance capture, integrity checks, and artifact upload |
| [`evals/qwen-smoke-v1.json`](../evals/qwen-smoke-v1.json) | Published v1.0 fixed quality inputs |
| [`evals/qwen-behavior-v2.json`](../evals/qwen-behavior-v2.json) | V1.1 checksummed 24-case behavior and latency contract |
| [`configs/load.arm64.json`](../configs/load.arm64.json) | Bounded load shape and SLO declaration |
| [`configs/policies.arm64.json`](../configs/policies.arm64.json) | Canonical and derived deployment-policy profiles |
| [`src/paretopilot/llama_summary.py`](../src/paretopilot/llama_summary.py) | Validated multi-pass throughput aggregation |
| [`src/paretopilot/server_eval.py`](../src/paretopilot/server_eval.py) | Exact-match quality and streamed server-latency evaluation |
| [`src/paretopilot/experiment.py`](../src/paretopilot/experiment.py) | Strict multi-candidate manifest and artifact assembly |
| [`src/paretopilot/analysis.py`](../src/paretopilot/analysis.py) | Constraint filtering, Pareto frontier, and deterministic selection |
| [`configs/constraints.candidate-study.json`](../configs/constraints.candidate-study.json) | Declared quality, latency, memory, frontier, and objective policy |
| [`src/paretopilot/report.py`](../src/paretopilot/report.py) | Deterministic, dependency-free HTML decision report |
| [`src/paretopilot/pass_eval.py`](../src/paretopilot/pass_eval.py) | Raw repeat-pass verification and reconstruction |
| [`src/paretopilot/load_eval.py`](../src/paretopilot/load_eval.py) | Bounded multi-client evaluation and command binding |
| [`src/paretopilot/profiles.py`](../src/paretopilot/profiles.py) | Precomputed canonical and derived policy decisions |
| [`src/paretopilot/stability.py`](../src/paretopilot/stability.py) | Pass direction and spread summary without significance claims |
| [`src/paretopilot/replay.py`](../src/paretopilot/replay.py) | Checksummed core regeneration and comparison |
| [`src/paretopilot/report_v11.py`](../src/paretopilot/report_v11.py) | Deterministic additive evidence report |

## Truthful interpretation

The diagram documents the executable candidate-study path; it is not itself benchmark evidence.
A run is publishable only after its `status.json`, raw measurements, provenance, dispatch logs,
and checksums pass review. ParetoPilot reports measured software performance and does not claim
energy savings or hardware-counter findings unless a separate source actually measures them.

Canonical run [`29973188507`](../results/published/29973188507/README.md) completed this path and
was rebuilt from its permanent release archive in a separate verification pass. It remains the
authoritative measured result until a fresh v1.1 archive completes the added lane and receives the
same review.
