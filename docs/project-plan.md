# ParetoPilot hackathon plan

Target submission deadline: **August 14, 2026 at 4:00 PM PDT / 6:00 PM CDT**.
The project is deliberately scoped to one model family, `llama.cpp`, and controlled paired runs
on GitHub's native Arm64 hosted runner.

## Product promise

Given controlled inference candidates and a declared quality floor, ParetoPilot rejects invalid
options, shows the latency/throughput/memory/quality Pareto frontier, and exports a transparent
deployment recommendation with raw evidence.

The baseline is allowed to win. ParetoPilot is a measurement tool, not a predetermined claim
that every optimization is better.

## MVP acceptance criteria

- The same command works locally for smoke testing and natively on an Arm64 Linux host.
- `doctor --require-evidence-host` requires native Arm64 Linux before collecting submission
  evidence.
- Generic and KleidiAI candidates use the same pinned `llama.cpp` commit, model, and workload.
- Raw `llama-bench` JSONL, manifests, hashes, and every repetition are retained.
- A fixed evaluation set enforces at least the predeclared quality-retention threshold.
- Missing, non-finite, mismatched, or synthetic evidence fails closed.
- The tool emits a deterministic recommendation and a self-contained human-readable report.
- A clean-machine reproduction run succeeds from the public Apache-2.0 repository.

## Build sequence

### July 21-23: foundation and target access

- Finish schemas, environment doctor, upstream JSONL parser, constraints, and Pareto selection.
- Establish the free public `ubuntu-24.04-arm` workflow and run the environment doctor.
- Build generic and KleidiAI-enabled binaries from the same pinned commit.
- Run one smoke benchmark before investing in presentation work.

### July 24-29: benchmark runner

- Add typed experiment manifests, command capture, checksums, and immutable result directories.
- Add warmups, repetitions, timeouts, partial-result handling, and deterministic candidate order.
- Aggregate prompt/decode throughput, variation, model size, and peak RSS.
- Add a small fixed quality evaluation for the selected model family.

### July 30-August 4: controlled optimization study

- Measure generic baseline, quantization-only, KleidiAI, and runtime-tuned candidates.
- Keep quantization and Arm-kernel gains separate so attribution is defensible.
- Capture Arm Performix evidence for the baseline and selected optimized candidate if the chosen
  target exposes the required hardware counters.
- Freeze the experiment contract before reviewing final results.

### August 5-9: report and product workflow

- Generate a self-contained HTML report with the frontier, constraints, rejection reasons, and
  baseline deltas.
- Export a reproducible recommended `llama-server` launch configuration.
- Test the main workflow on Windows and Arm64 Linux.

### August 10-13: submission hardening

- Re-run from a clean Arm64 host, preserve raw evidence, and tag the release.
- Finish the Devpost write-up, screenshots, architecture graphic, setup guide, and source links.
- Audit every performance statement back to a checked-in artifact.
- Submit at least one day early when possible.

### August 14: contingency only

- Use the final day for submission-form fixes, not core implementation or benchmark changes.

## Submission strategy

No video is planned because the current Devpost requirements mark it optional. The submission
will instead prioritize a concise written story, strong screenshots, a clear architecture image,
a public repository, one-command reproduction steps, and traceable benchmark artifacts.

The write-up should lead with one narrow before/after result, explain the quality constraint,
show why the selected point is non-dominated, and translate the system metric into deployment
value. Any incomplete integration or unverified measurement must be labeled plainly.
