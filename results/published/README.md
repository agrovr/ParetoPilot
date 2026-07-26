# Published benchmark results

Each row is a separate experiment. Its release archive contains the raw samples, commands, logs,
build records, environment captures, generated outputs, and checksums.

| Run | Status | Host | Result | Release archive |
| --- | --- | --- | --- | --- |
| [`30144901854`](30144901854/README.md) | **Capacity v1.4** | Ubuntu 24.04 Arm64, 4-vCPU Neoverse-N2 | P4/C4 was the best observed passing point for both candidates | [`v1.4.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.4.0) |
| [`30055662526`](30055662526/README.md) | **Current v1.1 result** | Ubuntu 24.04 Arm64, 4-vCPU Neoverse-N2 | Q8 had the lowest p95 end-to-end latency | [`v1.1.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.1.0) |
| [`29973188507`](29973188507/README.md) | Historical v1.0 result | 4-vCPU Arm Neoverse-N2 | Q8 reference retained under the declared 1% tolerance | [`v1.0.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.0.0) |

The v1.0 and v1.1 samples are not pooled because they ran as separate experiments. The v1.4
capacity study uses the published v1.1 candidates but answers a different server-sizing question.
