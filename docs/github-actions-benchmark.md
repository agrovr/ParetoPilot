# Free native Arm64 benchmark on GitHub Actions

ParetoPilot's first measured experiment uses GitHub's standard `ubuntu-24.04-arm` hosted runner.
For public repositories this provides native Arm64 execution without provisioning or paying for
a separate cloud VM.

## Canonical experiment

The workflow lives at `.github/workflows/benchmark-arm64.yml` and pins every external input used
by the comparison:

- `llama.cpp` commit `67b9b0e7f6ce45d929a4411907d3c48ec719e81c`;
- KleidiAI `v1.24.0`, pre-downloaded and verified by SHA-256 before CMake sees it;
- `Qwen/Qwen2.5-1.5B-Instruct-GGUF` revision
  `91cad51170dc346986eccefdc2dd33a9da36ead9`;
- `qwen2.5-1.5b-instruct-q4_0.gguf`, verified by size and SHA-256;
- GCC/G++ 13, static Release builds, CPU-only backends, four threads, and fixed batch sizes;
- 512 prompt tokens, 128 generated tokens, and ten recorded repetitions per pass.

Q4_0 is intentional. At the pinned runtime commit it exercises KleidiAI-compatible kernels;
using a K-quant model would not prove the optimization path was used.

The two builds use the same source, compiler, flags, and static-link policy. The only experimental
build variable is `GGML_CPU_KLEIDIAI=OFF` versus `ON`. Runtime logs must prove both optimized
passes allocated the `CPU_KLEIDIAI` model buffer, while both generic logs must prove its absence.
The workflow enables `llama-bench --verbose` specifically because the tool suppresses runtime
logs by default; JSONL remains isolated on stdout and the dispatch proof is captured on stderr.

## Running it

1. Open the repository's **Actions** tab.
2. Select **Native Arm64 benchmark**.
3. Select **Run workflow** on the default branch.
4. Keep `512`, `128`, and `10` for canonical evidence.
5. Wait for the single paired job to finish and download its `arm64-benchmark-*` artifact.

Input overrides are useful for exploratory runs but must be labeled exploratory. The canonical
submission evidence uses the declared defaults.

Only a manually dispatched default-branch run with all three canonical defaults receives
`classification: canonical`, `valid_evidence: true`, and the normal `arm64-benchmark-*` artifact
name. Feature-branch, push-triggered, and input-override successes are preserved as
`EXPLORATORY-arm64-benchmark-*` with valid measurements but are not publishable evidence.

## Evidence bundle

A successful job uploads only compact evidence:

- raw JSONL samples and complete stderr logs for all four passes;
- strict validation reports, pooled summaries, and two within-run paired comparisons;
- exact benchmark argument arrays;
- model, source archive, and executable hashes;
- CMake caches, configure/build logs, and static-link reports;
- CPU, operating-system, runner-image, compiler, and PMU-access details;
- an immutable experiment manifest, completion status, and `SHA256SUMS`.

The approximately 1 GB model and temporary build directories are not uploaded or cached. A failed
job is uploaded separately as `INCOMPLETE-*`, marked invalid, and may be used only for diagnosis.

## Interpretation boundaries

GitHub does not promise one fixed processor SKU for every standard hosted run. Therefore:

- compare generic and KleidiAI results only within the same A-B-B-A job;
- report paired speedups and dispersion, not just the fastest sample;
- disclose CPU identity, flags, runner image, and workflow run URL;
- do not combine absolute throughput across dissimilar runner identities;
- do not claim TTFT, p95 request latency, memory reduction, energy savings, or quality results
  from this throughput-only experiment.

Both variants use the exact same model file, so the experiment isolates runtime-kernel throughput;
it does not independently measure model quality. A later versioned quality evaluation will be
required before ParetoPilot recommends changes to model or quantization.

Arm Performix remains an optional deeper profiling stage. The workflow records whether the hosted
runner exposes performance counters, but Performix is not a source of compute and is not required
for this benchmark to succeed.

## Preserving final evidence

GitHub Actions artifacts expire. After a successful run, review the status, hashes, dispatch logs,
and comparisons, then copy only the verified compact bundle into `results/published/<run-id>/` or
attach it to a tagged GitHub release. Never publish a partial bundle as measured evidence.
