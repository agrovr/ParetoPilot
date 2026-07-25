# Published evidence

The repository keeps compact, human-reviewed pointers to released results. Complete raw samples,
commands, logs, build records, environment captures, and generated outputs live in versioned
release archives instead of cluttering the source tree.

| Run | Status | Host | Decision | Complete evidence |
| --- | --- | --- | --- | --- |
| [`30144901854`](30144901854/README.md) | **Supplementary capacity v1.4** | Ubuntu 24.04 Arm64, 4-vCPU Neoverse-N2 | P4/C4 was the best observed passing point for both candidate envelopes; canonical model decision unchanged | [`v1.4.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.4.0) |
| [`30055662526`](30055662526/README.md) | **Current canonical v1.1** | Ubuntu 24.04 Arm64, 4-vCPU Neoverse-N2 | Q8 reference was the numeric p95 E2E winner and was retained | [`v1.1.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.1.0) |
| [`29973188507`](29973188507/README.md) | Historical canonical v1.0 | 4-vCPU Arm Neoverse-N2 | Q8 reference retained under the declared 1% tolerance | [`v1.0.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.0.0) |

The canonical v1.0 and v1.1 rows are separate controlled experiments on ephemeral runners; their
samples are never pooled. The v1.0 release remains reproducible historical evidence, and v1.1 is
the current canonical result because it completed the expanded behavior, policy, load, stability,
integrity, and replay contract.

The v1.4 row is supplementary. It keeps the frozen v1.1 candidate decision intact and adds a
separate bounded serving-capacity study for the Q8 reference and Q4 resource alternative. Its raw
evidence stays in the versioned, SHA-256-locked release archive rather than in the source tree.
