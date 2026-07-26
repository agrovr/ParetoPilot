# Arm64 capacity study: run 30144901854

This study asks how many server slots and simultaneous clients each candidate should use after the
v1.1 model comparison. On this native Arm64 runner, **four clients needed four server slots**.
P4/C4—four server slots and four simultaneous clients—was the best observed passing point for both
the Q8 reference and the Q4 resource alternative.

Download the complete logs, commands, raw samples, build records, environment capture, and
generated outputs from the
[`v1.4.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.4.0). This directory keeps
the summary and its machine-readable [`evidence.json`](evidence.json).

## Run and archive details

| Field | Value |
| --- | --- |
| GitHub Actions run | [`30144901854`](https://github.com/agrovr/ParetoPilot/actions/runs/30144901854), attempt `1` |
| Classification | `supplementary-capacity` |
| Source commit | [`db9ccaf37e3c7e807832652e237de813675ed807`](https://github.com/agrovr/ParetoPilot/commit/db9ccaf37e3c7e807832652e237de813675ed807) |
| Runner | Ubuntu 24.04 Arm64, Arm Neoverse-N2, 4 vCPUs |
| Release | [`v1.4.0`](https://github.com/agrovr/ParetoPilot/releases/tag/v1.4.0) |
| Asset | [`paretopilot-v1.4.0-arm64-capacity-30144901854.zip`](https://github.com/agrovr/ParetoPilot/releases/download/v1.4.0/paretopilot-v1.4.0-arm64-capacity-30144901854.zip) |
| Asset size | 794,681 bytes |
| Archive SHA-256 | `a73d801bc3997f1c0b0158e92c8305987da8638501b74c5ecd2af3aaca57aaa7` |
| Checksummed payloads | 121 |
| Published v1.1 result | Run [`30055662526`](../30055662526/README.md), release `v1.1.0` |

## Selected operating points

| Candidate | Role | Selected point | Median generation rate | Gain vs own P1/C1 | TTFT p95 | E2E p95 | Peak RSS | Quality |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q8 generic reference** | Published latency choice | **P4 / C4** | 84.94 tok/s | +142.0% | 756.6 ms | 3056.8 ms | 3448.1 MiB | 21/24 |
| **Q4 + KleidiAI, 512-token micro-batch** | Measured resource alternative | **P4 / C4** | 90.67 tok/s | +176.3% | 639.9 ms | 2852.6 ms | 2031.3 MiB | 20/24 |

Each candidate had six passing cells out of nine. The selection maximized median generated tokens
per second only among cells where both mirrored passes met the latency, quality, memory, and
consistency limits. Exactly one cell per candidate was within the 1% tolerance set before testing.

At the two selected P4/C4 points, Q4 measured 6.74% higher median generation throughput, 15.42%
lower observed-p95 TTFT, 6.68% lower observed-p95 E2E latency, and 41.09% lower peak RSS than Q8.
These capacity measurements do not change the published Q8 latency result.

## What was measured

- Two mirrored forward/reverse passes over two candidates, three server-slot levels (`1`, `2`,
  `4`), and three simultaneous-client levels (`1`, `2`, `4`).
- Four warmups and eight measured requests per client level per pass: 144 warmups and 288 measured
  requests in total, each requesting 64 output tokens.
- All 288 measured requests completed with zero recorded request failures.
- Six of nine cells passed for each candidate. Every rejected cell missed the TTFT limit; three
  also missed the E2E limit.
- The fixed 24-case quality guard passed at every slot level. Q8 scored 21/24. Q4 scored 20/24,
  which is **95.2% of the reference score**.

## Verification

The extracted archive passed all 121 payload checksums with exact file coverage. A fresh assembly
from the raw inputs reproduced `capacity-study.json` byte for byte, and regenerating the receipt
reproduced `capacity-receipt.md` byte for byte. All 18 cells were separately recomputed with zero
metric mismatches.

Replay also confirmed the published Q8 decision and its report, with no differences or warnings.

## Verify

After downloading the release asset, verify the outer ZIP:

```bash
printf '%s  %s\n' \
  a73d801bc3997f1c0b0158e92c8305987da8638501b74c5ecd2af3aaca57aaa7 \
  paretopilot-v1.4.0-arm64-capacity-30144901854.zip | sha256sum --check
```

Extract it into `evidence/`, verify every archived payload, then read the receipt for the exact
reassembly and byte-comparison commands:

```bash
(cd evidence && sha256sum --check SHA256SUMS)
less evidence/capacity-receipt.md
```

## Study limits

This fixed-concurrency study covers one model family and workload on one temporary native Arm64
runner. It does not measure open-loop traffic, production capacity, cost, or energy.

Each cell has eight measured requests in each pass. Within a pass, p95 is the maximum observed
request; the displayed cell value is the median of the two pass-level p95 values. Q4 P4/C2 passed,
but its 14.78% generation-rate spread was close to the 15% limit. The Q4 quality result was just
above the 95% retention floor. The KleidiAI log marker shows that `llama.cpp` reported a
`CPU_KLEIDIAI` model buffer; it is not a kernel-level trace.
