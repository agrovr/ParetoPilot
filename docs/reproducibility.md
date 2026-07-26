# Reproducing ParetoPilot

ParetoPilot supports three levels of reproduction:

1. **Code verification** runs the software checks on any supported development host.
2. **Decision reproduction** audits the pinned canonical archive and regenerates its core,
   policy, load, stability, and report outputs.
3. **Measurement reproduction** dispatches a fresh native Arm64 experiment. A new hosted runner is
   new evidence, not an exact hardware replay.

The current canonical evidence is GitHub Actions
[run `30055662526`](https://github.com/agrovr/ParetoPilot/actions/runs/30055662526), preserved in
the [`v1.1.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.1.0). It was produced
from commit
[`8a9ddce0afa2272c4a4097fe87ef6f06cb7689a9`](https://github.com/agrovr/ParetoPilot/commit/8a9ddce0afa2272c4a4097fe87ef6f06cb7689a9).

The earlier [v1.0 result](../results/published/29973188507/README.md) remains reproducible
historical evidence. It came from a separate ephemeral runner and is not pooled with v1.1.

The supplementary capacity envelope is GitHub Actions
[run `30144901854`](https://github.com/agrovr/ParetoPilot/actions/runs/30144901854), preserved in
the [`v1.4.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.4.0). It is a
candidate-local serving study and does not replace or modify the canonical v1.1 decision.
Use ParetoPilot
[`0f196a4f`](https://github.com/agrovr/ParetoPilot/commit/0f196a4f3764526db48c0481b69f7f90eaffc9e4)
or newer to replay that unchanged archive. The version 1.4.1 release is intentionally not linked
until it is published.

## Requirements

Local verification needs Git and Python 3.12 or newer. A fresh measurement additionally needs a
public GitHub repository or fork with Actions enabled, access to `ubuntu-24.04-arm`, network
access for pinned source and model downloads, and sufficient Actions time and storage.

The Python package has no runtime dependencies. Development checks use the bounded tools in the
`dev` extra. GitHub runner availability and billing are external conditions and should be checked
before repeated dispatches.

## One-command published proof

After installing ParetoPilot, a reviewer can verify both public evidence releases without an
Arm64 machine:

```bash
paretopilot verify-published --output-dir output/published-proof
```

The command downloads the exact pinned v1.1 canonical and v1.4 capacity archives, enforces each
official GitHub URL, byte size, and SHA-256 digest, safely extracts them, and runs both existing
replay contracts. It refuses an existing output directory. A successful run prints:

```text
PASS: pinned canonical v1.1 and capacity v1.4 archives verified and replayed.
```

Inspect `output/published-proof/published-proof.md` for the human-readable proof or
`published-proof.json` for the deterministic contract. They record the two Actions run IDs,
release links, archive hashes, all nine canonical byte matches, the reproduced `q8-generic`
decision, both P4/C4 capacity selections, the two capacity artifact matches, the embedded
canonical replay, and explicit evidence limits.

To stay offline after downloading the official files once:

```bash
paretopilot verify-published \
  --canonical-archive paretopilot-v1.1.0-arm64-evidence-30055662526.zip \
  --capacity-archive paretopilot-v1.4.0-arm64-capacity-30144901854.zip \
  --output-dir output/published-proof
```

Local archives must still match the packaged official size and digest pins. This command verifies
the archived bytes and deterministic decisions; it does not rerun inference or prove current
hardware performance. The manual steps below expose every part of the same audit.

## 1. Verify the code

```bash
git clone https://github.com/agrovr/ParetoPilot.git
cd ParetoPilot
git switch main
python -m venv .venv
```

Activate `.venv`, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m paretopilot --version
ruff check .
ruff format --check .
python -m unittest discover -s tests
python -m build --wheel
```

Expect ParetoPilot `1.4.1` or newer; earlier commits do not include the one-command published
proof. The frozen measurements remain pinned to their v1.1.0 and v1.4.0 release archives even
when a newer verifier replays them.

CI repeats the test and lint checks on Ubuntu x64, Windows x64, and native Ubuntu Arm64. A
separate Ubuntu x64 job installs the built wheel into an isolated environment.

## 2. Download and verify the canonical archive

Download the pinned v1.1.0 release asset:

```bash
curl -L \
  https://github.com/agrovr/ParetoPilot/releases/download/v1.1.0/paretopilot-v1.1.0-arm64-evidence-30055662526.zip \
  -o paretopilot-v1.1.0-arm64-evidence-30055662526.zip
```

The file must be exactly 402,899 bytes. Verify its outer SHA-256:

```bash
printf '%s  %s\n' \
  b5586878ccd214667911390f417db0417111ac2c31d163a2f5f55c4469aefeb2 \
  paretopilot-v1.1.0-arm64-evidence-30055662526.zip | sha256sum --check
```

In Windows PowerShell:

```powershell
(Get-Item .\paretopilot-v1.1.0-arm64-evidence-30055662526.zip).Length
(Get-FileHash -Algorithm SHA256 .\paretopilot-v1.1.0-arm64-evidence-30055662526.zip).Hash
```

The second command must display
`B5586878CCD214667911390F417DB0417111AC2C31D163A2F5F55C4469AEFEB2`.

Extract into a new directory and verify the 150 payloads listed in `SHA256SUMS`:

```bash
mkdir evidence
unzip paretopilot-v1.1.0-arm64-evidence-30055662526.zip -d evidence
(cd evidence && sha256sum --check SHA256SUMS)
```

In Windows PowerShell:

```powershell
New-Item -ItemType Directory -Path .\evidence
Expand-Archive `
  -LiteralPath .\paretopilot-v1.1.0-arm64-evidence-30055662526.zip `
  -DestinationPath .\evidence
```

The cross-platform `paretopilot replay` command in the next step verifies every entry in
`SHA256SUMS` and rejects missing, extra, changed, or unsafe payload paths before rebuilding
anything. Windows users therefore do not need a separate `sha256sum` installation.

The bundle's `status.json` must report:

```json
{
  "classification": "canonical",
  "measurement_valid": true,
  "status": "complete",
  "valid_evidence": true
}
```

The committed
[`evidence.json`](../results/published/30055662526/evidence.json) records the Actions artifact,
release asset, outer digest, source and runner identity, important artifact hashes, and replay
review.

## 3. Replay the complete v1.1 contract

Use a new destination outside the extracted evidence directory. ParetoPilot refuses to overwrite
existing evidence or reports.

```bash
python -m paretopilot replay evidence \
  --output-dir output/replay-v1.1
```

The archive already contains the policy configuration, so no external `--policies` argument is
needed. Replay first validates safe relative paths, canonical completion status, the complete
`SHA256SUMS` coverage, and the required v1.1 artifact set. It then:

1. reassembles the canonical benchmark and recommendation;
2. recomputes all five policy profiles from the archived policy configuration;
3. validates the load plan and each per-candidate load file, including request endpoints and both
   server-command digests, then rebuilds the combined load evaluation;
4. reconstructs both pass benchmark sets from checksummed raw throughput, settings,
   server-evaluation, and GNU `time -v` files;
5. regenerates repeat stability from those pass sets; and
6. renders and compares the core and v1.1 reports.

Inspect `output/replay-v1.1/replay.json`. The canonical release must report:

| Field | Expected value |
| --- | --- |
| Replay contract | `"1.1"` |
| Valid | `true` |
| Decision reproduced | `true` |
| Fully reproduced | `true` |
| Report matches archive | `true` |
| Selected candidate | `"q8-generic"` |
| Differences | `[]` |
| Warnings | `[]` |
| Policy profile count | `5` |

All nine comparisons must be present and match: `benchmark-set`, `recommendation`,
`policy-profiles`, `load-evaluation`, both pass benchmark sets, `repeat-stability`, `report`, and
`report-v1.1`.

### Pages publication is downstream from replay

The public [evidence showcase](https://agrovr.github.io/ParetoPilot/) is not part of the nine-item
replay contract. The Pages workflow first completes the verification above and confirms the
reviewed `report-v1.1.html` digest. It then:

1. copies that exact replayed file to
   [`evidence/report-v1.1.html`](https://agrovr.github.io/ParetoPilot/evidence/report-v1.1.html);
2. supplies the reviewed evidence lock and exact report to `paretopilot showcase-v11`; and
3. writes the separate judge-facing presentation to `index.html`.

The showcase renderer refuses a canonical-report mismatch. It does not rerun inference, create
new evidence, or change the archived recommendation.

To inspect pass reconstruction independently, use fresh output paths:

```bash
python -m paretopilot assemble-repeat-pass \
  --experiment evidence/experiment \
  --pass-number 1 \
  --output output/pass-1.json
python -m paretopilot assemble-repeat-pass \
  --experiment evidence/experiment \
  --pass-number 2 \
  --output output/pass-2.json

cmp output/pass-1.json evidence/extensions/benchmark-set-pass-1.json
cmp output/pass-2.json evidence/extensions/benchmark-set-pass-2.json
```

The byte comparisons in Windows PowerShell are:

```powershell
$Pass1 = (Get-FileHash -Algorithm SHA256 .\output\pass-1.json).Hash
$ExpectedPass1 = (
  Get-FileHash -Algorithm SHA256 .\evidence\extensions\benchmark-set-pass-1.json
).Hash
$Pass2 = (Get-FileHash -Algorithm SHA256 .\output\pass-2.json).Hash
$ExpectedPass2 = (
  Get-FileHash -Algorithm SHA256 .\evidence\extensions\benchmark-set-pass-2.json
).Hash
if ($Pass1 -ne $ExpectedPass1 -or $Pass2 -ne $ExpectedPass2) {
  throw "Reconstructed pass evidence does not match the archive."
}
```

The command follows only artifact paths and SHA-256 digests bound by the canonical benchmark. It
does not derive a pass by splitting pooled metrics. Replay does not rerun inference.

## Replay the supplementary v1.4 capacity evidence

Download the reviewed capacity archive:

```bash
curl -L \
  https://github.com/agrovr/ParetoPilot/releases/download/v1.4.0/paretopilot-v1.4.0-arm64-capacity-30144901854.zip \
  -o paretopilot-v1.4.0-arm64-capacity-30144901854.zip
```

The file must be exactly 794,681 bytes. Verify its outer SHA-256, extract it into a fresh
directory, and verify all 121 archived payloads:

```bash
printf '%s  %s\n' \
  a73d801bc3997f1c0b0158e92c8305987da8638501b74c5ecd2af3aaca57aaa7 \
  paretopilot-v1.4.0-arm64-capacity-30144901854.zip | sha256sum --check
mkdir capacity-evidence
unzip paretopilot-v1.4.0-arm64-capacity-30144901854.zip -d capacity-evidence
(cd capacity-evidence && sha256sum --check SHA256SUMS)
```

In Windows PowerShell:

```powershell
(Get-Item .\paretopilot-v1.4.0-arm64-capacity-30144901854.zip).Length
(Get-FileHash `
  -Algorithm SHA256 `
  .\paretopilot-v1.4.0-arm64-capacity-30144901854.zip).Hash
New-Item -ItemType Directory -Path .\capacity-evidence
Expand-Archive `
  -LiteralPath .\paretopilot-v1.4.0-arm64-capacity-30144901854.zip `
  -DestinationPath .\capacity-evidence
```

The expected size is `794681`, and the hash is
`A73D801BC3997F1C0B0158E92C8305987DA8638501B74C5ECD2AF3AACA57AAA7`.
As with the canonical archive, `replay-capacity` verifies all 121 `SHA256SUMS` entries and exact
file coverage before reconstruction, so no separate Unix checksum tool is required on Windows.

Replay into a new directory outside the evidence bundle:

```bash
python -m paretopilot replay-capacity capacity-evidence \
  --output-dir output/replay-capacity
```

The command does not rerun inference. It reconstructs all 18 capacity cells from the archived
load, RSS, server-log, and quality inputs; requires byte-identical `capacity-study.json` and
`capacity-receipt.md` outputs; and safely replays the embedded frozen v1.1 archive. Inspect
`output/replay-capacity/capacity-replay.json`; it must report `valid: true`,
`capacity_study_reproduced: true`, `capacity_receipt_reproduced: true`, and a canonical
`selected_id` of `q8-generic`.

The compact [reviewed result](../results/published/30144901854/README.md) records the Actions
artifact identity, release digest, exact selected operating points, recomputation audit, and study
limits. This replay answers whether the archived decision is reproducible; dispatching the
capacity workflow again creates a new measurement on a new ephemeral runner.

## 4. Compare the expected decision

The canonical benchmark should reproduce:

| Candidate | E2E p95 | TTFT p95 | Prompt tok/s | Generation tok/s | Peak RSS | Model size | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `q8-generic` | 2231.932869 ms | 545.373894 ms | 102.6185 | 38.72645 | 3437.597656 MiB | 1806.766632 MiB | 21/24 |
| `q4-generic` | 2311.125148 ms | 483.113115 ms | 113.8210 | 35.01235 | 1966.472656 MiB | 1016.833527 MiB | 20/24 |
| `q4-kleidiai` | 2299.454336 ms | 470.402254 ms | 114.4480 | 35.37635 | 1966.484375 MiB | 1016.833527 MiB | 20/24 |
| `q4-kleidiai-tuned` | 2307.715263 ms | 469.968079 ms | 131.4565 | 35.09590 | 1966.480469 MiB | 1016.833527 MiB | 20/24 |

Expected core decision:

| Field | Expected value |
| --- | --- |
| Synthetic source | `false` |
| Baseline | `q8-generic` |
| Numeric objective best | `q8-generic` |
| Selected candidate | `q8-generic` |
| Preference changed winner | `false` |
| Objective tolerance | `1%` |
| Shortlist cutoff | `2254.2522 ms` |
| Shortlist | `q8-generic` only |
| Eligible candidates | all four |
| Pareto frontier | all four |
| Rejected candidates | none |

The five profile selections must be:

| Profile | Expected selection |
| --- | --- |
| `canonical-latency` | `q8-generic` |
| `memory-first` | `q4-generic` |
| `first-token-first` | `q4-kleidiai-tuned` |
| `prompt-ingest-first` | `q4-kleidiai-tuned` |
| `decode-first` | `q8-generic` |

The tuned Q4 + KleidiAI candidate is the measured resource alternative: relative to Q8 it has
43.72% smaller model size, 42.79% lower peak RSS, 13.83% lower p95 TTFT, and 28.10% higher prompt
throughput. It is also 3.40% slower on p95 end-to-end latency, 9.37% lower on generation
throughput, and one behavior case lower.

## 5. Audit provenance, load, and stability

The archive records:

- native runner, compiler, build, runtime, model, and evaluation-suite identity;
- exact `llama-bench` and `llama-server` command arrays;
- raw prompt and generation throughput samples;
- 24 deterministic behavior outcomes and streamed 64-token latency samples;
- peak RSS from GNU `time -v`;
- model size, quantization, batch, micro-batch, KleidiAI build flag, and observed model-buffer
  marker state;
- raw bounded-load requests and SLO aggregates for concurrency 1, 2, and 4;
- both pass-level reconstructed benchmark sets and the 24-row stability summary; and
- the closed manifest, recommendation, reports, status, and bundle checksums.

The source identities must include:

- `llama.cpp` `67b9b0e7f6ce45d929a4411907d3c48ec719e81c`;
- KleidiAI `1.24.0`;
- Qwen2.5 1.5B Instruct revision `91cad51170dc346986eccefdc2dd33a9da36ead9`; and
- evaluation-suite SHA-256
  `e49c16fba32fd65c947264aef4141026ab68b1fd415ef09eeea6e8ade9a545c7`.

Every candidate completed 100% of its measured load requests. Concurrency 1 was the highest level
to meet both the 2,000 ms p95 TTFT and 6,500 ms p95 end-to-end SLOs.

Across the two reconstructed passes, all six metrics were directionally consistent for the three
Q4 candidates. Their maximum relative spreads were 1.6695%, 1.3919%, and 0.8029% respectively.
These are observed repeat summaries, not significance claims.

## 6. Dispatch a fresh native Arm64 study

From GitHub Actions, select **Native Arm64 candidate study**, choose **Run workflow**, use the
default branch, and retain `10` repetitions. The equivalent GitHub CLI command is:

```bash
gh workflow run candidate-study-arm64.yml --ref main -f repetitions=10
```

A run is canonical only when it is manually dispatched from the default branch with exactly ten
repetitions and every environment, build, dispatch, benchmark, behavior, latency, memory, load,
integrity, selection, and report gate passes. Branch runs and changed inputs remain exploratory.
Failed workflows may upload diagnostic artifacts with `valid_evidence: false`; those must not
replace canonical evidence.

## Historical v1.0 reproduction

Run `29973188507` and release `v1.0.0` remain available for historical comparison:

- [reviewed v1.0 summary](../results/published/29973188507/README.md)
- [v1.0 release archive](https://github.com/agrovr/ParetoPilot/releases/tag/v1.0.0)

That archive uses replay contract `1.0` and the earlier five-case smoke gate. Follow its reviewed
summary for the matching archive digest and expected decision. Do not combine its samples with
v1.1; each release represents one separate controlled hosted-runner experiment.

## Evidence limits

- The hosted runner is ephemeral. Separate workflow runs are not pooled.
- The 24-case suite is a deterministic deployment gate, not a broad model-quality benchmark.
- The load sweep tested only concurrency 1, 2, and 4 with eight measured requests per level.
- Two reconstructed mirrored passes describe observed direction and spread; they do not establish
  statistical significance.
- Results may not generalize to every Arm CPU, model, prompt distribution, concurrency level, or
  deployment environment.
- Energy and cost were not measured and must not be inferred from throughput.
- Models and build trees are intentionally absent from the archive; their pinned identities and
  hashes remain present.

## Optional Performix profiling

Arm Performix is outside the required pipeline. A compatible host may add separate hotspot context
for a candidate, but profiler output does not replace measured throughput, behavior, latency,
memory, load, or checksums. Missing Performix output does not invalidate ParetoPilot evidence.

For the experimental design and decision rules, see
[`benchmark-methodology.md`](benchmark-methodology.md).
