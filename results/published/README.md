# Published evidence

The repository keeps compact, human-reviewed pointers to released results. Complete raw samples,
commands, logs, build records, environment captures, and generated outputs live in permanent
release archives instead of cluttering the source tree.

| Run | Status | Host | Decision | Complete evidence |
| --- | --- | --- | --- | --- |
| [`30055662526`](30055662526/README.md) | **Current canonical v1.1** | Ubuntu 24.04 Arm64, 4-vCPU Neoverse-N2 | Q8 reference was the numeric p95 E2E winner and was retained | [`v1.1.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.1.0) |
| [`29973188507`](29973188507/README.md) | Historical canonical v1.0 | 4-vCPU Arm Neoverse-N2 | Q8 reference retained under the declared 1% tolerance | [`v1.0.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.0.0) |

The two rows are separate controlled experiments on ephemeral runners. Their samples are never
pooled. The v1.0 release remains reproducible historical evidence; v1.1 is the current canonical
result because it completed the expanded behavior, policy, load, stability, integrity, and replay
contract.
