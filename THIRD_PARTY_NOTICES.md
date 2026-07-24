# Third-party notices

ParetoPilot's own source code is licensed under the Apache License 2.0. See
[`LICENSE`](LICENSE). The reproducible Arm64 workflows fetch the following third-party projects
and model files at run time. This repository does not redistribute their source trees, compiled
binaries, or model weights.

## Runtime and optimization sources

| Component | Pinned version | How ParetoPilot uses it | License | Upstream source |
|---|---|---|---|---|
| llama.cpp | commit `67b9b0e7f6ce45d929a4411907d3c48ec719e81c` | Builds `llama-bench` and `llama-server` for the measured candidates | MIT | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/67b9b0e7f6ce45d929a4411907d3c48ec719e81c) and its [license](https://github.com/ggml-org/llama.cpp/blob/67b9b0e7f6ce45d929a4411907d3c48ec719e81c/LICENSE) |
| KleidiAI | `v1.24.0` | Supplies Arm CPU kernels for candidates built with `GGML_CPU_KLEIDIAI=ON` | Apache-2.0 and BSD-3-Clause files are included by upstream; individual files carry SPDX identifiers | [Arm KleidiAI v1.24.0](https://github.com/ARM-software/kleidiai/tree/v1.24.0) and its [LICENSES directory](https://github.com/ARM-software/kleidiai/tree/v1.24.0/LICENSES) |

The workflow records source revisions, archive checksums, compiler configuration, and resulting
binary checksums in each evidence bundle. Consult the SPDX header of an upstream file to determine
which KleidiAI license applies to that file.

## Model artifacts

| Component | Pinned revision | Files used | License | Upstream source |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct-GGUF | `91cad51170dc346986eccefdc2dd33a9da36ead9` | `qwen2.5-1.5b-instruct-q8_0.gguf` and `qwen2.5-1.5b-instruct-q4_0.gguf` | Apache-2.0 | [Qwen model repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/91cad51170dc346986eccefdc2dd33a9da36ead9) and its [license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/blob/91cad51170dc346986eccefdc2dd33a9da36ead9/LICENSE) |

The workflow downloads these files directly from the pinned upstream revision and verifies their
expected byte sizes and SHA-256 checksums before use. Model files are excluded from Git and from
uploaded evidence artifacts.

## Evaluation prompts

The five deterministic prompts in [`evals/qwen-smoke-v1.json`](evals/qwen-smoke-v1.json) and the
24 deterministic cases in
[`evals/qwen-behavior-v2.json`](evals/qwen-behavior-v2.json) were written for ParetoPilot and are
dedicated to the public domain under CC0-1.0. They are narrow exact-match behavior checks, not a
claim of broad model quality or safety.

## Build and automation services

The GitHub Actions workflows invoke commit-pinned releases of `actions/checkout`,
`actions/setup-python`, `actions/upload-artifact`, `actions/configure-pages`,
`actions/upload-pages-artifact`, and `actions/deploy-pages`. Those actions run only in CI and are
not bundled with ParetoPilot. Their license notices remain in their respective upstream
repositories.

Third-party names and marks belong to their respective owners. Inclusion here does not imply
endorsement of ParetoPilot by Arm, Qwen, Hugging Face, GitHub, or the llama.cpp maintainers.
