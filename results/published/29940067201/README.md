# Canonical Arm64 benchmark 29940067201

This is the reviewed compact evidence bundle from
[GitHub Actions run 29940067201](https://github.com/agrovr/ParetoPilot/actions/runs/29940067201).
The workflow completed successfully on July 22, 2026 and classified the run as canonical with
`measurement_valid: true` and `valid_evidence: true`.

## Experiment identity

- Repository commit: `1c17c1f7e1edb331a2bf4eb8016927622dc11d45`
- Runner: GitHub-hosted Ubuntu 24.04 Arm64, 4-vCPU Arm Neoverse-N2
- Runner image: `ubuntu24-arm64` version `20260714.61.1`
- `llama.cpp`: `67b9b0e7f6ce45d929a4411907d3c48ec719e81c`
- KleidiAI: `v1.24.0`
- Model: Qwen2.5 1.5B Instruct Q4_0 at pinned revision and SHA-256
- Workloads: 512-token prompt processing and 128-token generation
- Runtime settings: four threads, batch 512, micro-batch 128, CPU-only
- Order: generic, KleidiAI, KleidiAI, generic
- Samples: ten repetitions per pass; twenty pooled samples per variant and workload

The two binaries used the same source, compiler, static-link policy, model, and benchmark settings.
The experimental build variable was `GGML_CPU_KLEIDIAI=OFF` versus `ON`.

## Results

| Workload | Generic median | KleidiAI median | Median change | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Prompt processing (`pp512`) | 113.6605 tok/s | 114.2580 tok/s | +0.5257% | Small, consistent gain |
| Token generation (`tg128`) | 35.1134 tok/s | 35.1307 tok/s | +0.0493% | Inconclusive |

The prompt-processing gain was positive in both paired comparisons: +0.5751% and +0.4201%.
Generation disagreed by order: +2.1583% in pair 1 and -1.2145% in pair 2. The pooled generation
delta is therefore noise-scale evidence, not a credible optimization claim.

This result is still useful to ParetoPilot: it shows that enabling an Arm optimization library is
not automatically a meaningful deployment win. The project should retain this candidate as
measured evidence while evaluating stronger runtime, batching, and model candidates.

## Verification performed

- All 61 entries in the original artifact `SHA256SUMS` matched after download.
- All four raw JSONL artifacts passed ParetoPilot's evidence validator independently.
- Every file reported ten repetitions, four threads, batch 512, micro-batch 128, zero GPU layers,
  device `none`, and no operation offload.
- Both KleidiAI stderr logs contain exactly one `CPU_KLEIDIAI model buffer` dispatch marker.
- Both generic stderr logs contain zero KleidiAI dispatch markers.
- Pooled and paired comparisons passed the workflow's compatibility gates.
- The compact repository bundle was re-hashed after review.

`SOURCE-SHA256SUMS` is the unchanged checksum list from the complete Actions artifact. It includes
diagnostic files that intentionally remain only in the expiring artifact. `SHA256SUMS` covers this
selected repository bundle, including its review metadata, except for `SHA256SUMS` itself.

To verify the compact bundle from this directory on Linux:

```bash
sha256sum --check SHA256SUMS
```

## Evidence boundaries

- This is one paired experiment on one ephemeral hosted runner, not a claim about every Arm CPU.
- Both variants used the same model, so model quality should be unchanged but was not independently
  evaluated.
- This throughput-only run does not measure TTFT, p95 request latency, memory reduction, energy,
  concurrency, or cost.
- Arm Performix is optional. Its absence does not affect this run's validity, and optional profiler
  capability probes remain only in the complete Actions artifact.
- Do not combine these measurements with exploratory run `29895431069`.
