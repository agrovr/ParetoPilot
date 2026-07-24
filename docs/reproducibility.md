# Reproducing ParetoPilot

ParetoPilot supports three levels of reproduction:

1. **Code verification** runs the software checks on any supported development host.
2. **Decision reproduction** audits the permanent canonical archive and regenerates its benchmark
   set, recommendation, and report.
3. **Measurement reproduction** dispatches a fresh native Arm64 experiment. A new hosted runner is
   new evidence, not an exact hardware replay.

The current canonical evidence is GitHub Actions run
[`29973188507`](https://github.com/agrovr/ParetoPilot/actions/runs/29973188507), preserved in the
[`v1.0.0` release](https://github.com/agrovr/ParetoPilot/releases/tag/v1.0.0). The repository
implements an additive v1.1 evidence contract, but no canonical v1.1 archive is published yet.
Configuration files and local test output must not be presented as measured v1.1 results.

## Requirements

Local verification needs Git and Python 3.12 or newer. A fresh measurement additionally needs a
public GitHub repository or fork with Actions enabled, access to `ubuntu-24.04-arm`, network access
for pinned source and model downloads, and sufficient Actions time and storage.

The Python package has no runtime dependencies. Development checks use the bounded tools in the
`dev` extra. GitHub runner availability and billing are external conditions and should be checked
before repeated dispatches.

## 1. Verify the code

```bash
git clone https://github.com/agrovr/ParetoPilot.git
cd ParetoPilot
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

CI repeats the test and lint checks on Ubuntu x64, Windows x64, and native Ubuntu Arm64. A separate
Ubuntu x64 job installs the built wheel into an isolated environment.

## 2. Download and verify the canonical archive

Download the permanent release asset:

```bash
curl -L \
  https://github.com/agrovr/ParetoPilot/releases/download/v1.0.0/paretopilot-v1.0.0-arm64-evidence-29973188507.zip \
  -o paretopilot-v1.0.0-arm64-evidence-29973188507.zip
```

Verify its outer SHA-256:

```bash
printf '%s  %s\n' \
  fb4f4c86a729a5eb42e23dbd3c6346fd4ab31ce14423dbb8c7672b11b6a6fd00 \
  paretopilot-v1.0.0-arm64-evidence-29973188507.zip | sha256sum --check
```

In Windows PowerShell, the equivalent check is:

```powershell
(Get-FileHash -Algorithm SHA256 .\paretopilot-v1.0.0-arm64-evidence-29973188507.zip).Hash
```

The displayed digest must match the lowercase value above. Extract the ZIP into a new directory,
then verify all 124 files listed in `SHA256SUMS`:

```bash
mkdir evidence
unzip paretopilot-v1.0.0-arm64-evidence-29973188507.zip -d evidence
(cd evidence && sha256sum --check SHA256SUMS)
```

The bundle's `status.json` must say `complete`, `canonical`, `measurement_valid: true`, and
`valid_evidence: true`. The committed
[`evidence.json`](../results/published/29973188507/evidence.json) records the Actions artifact,
release asset, outer digest, review checks, and derived-output fingerprints.

## 3. Replay and compare the decision

Use fresh output paths; ParetoPilot intentionally refuses to overwrite evidence or reports.
With the current source, use the replay command so a v1.0 archive is compared under its recorded
producer-version identity:

```bash
python -m paretopilot replay evidence \
  --output-dir output/reproduction \
  --policies configs/policies.arm64.json
```

Inspect `output/reproduction/replay.json`. For the published v1.0 archive,
`replay_contract` must be `1.0`, `valid` and `decision_reproduced` must be `true`, the
`benchmark-set` and `recommendation` authoritative comparisons must match, and `selected_id` must
be `q8-generic`.

The replay path matters across releases. A direct report command from newer source records the
current ParetoPilot package version in newly generated JSON. Replay instead restores the archived
producer version before comparing the authoritative recommendation, so version metadata does not
look like decision drift.

The HTML report is a presentation generated from verified inputs, so its layout may improve after
the evidence release. A report-only difference leaves `valid: true`, keeps
`decision_reproduced: true`, and appears as a presentation warning. To reproduce every v1.0 output
byte-for-byte, including the archived HTML, check out tag `v1.0.0` and run the reproduction
commands from that tag.

Expected decision:

| Field | Expected value |
| --- | --- |
| Synthetic source | `false` |
| Baseline | `q8-generic` |
| Numeric objective best | `q4-generic` |
| Selected candidate | `q8-generic` |
| Preference changed winner | `true` |
| Eligible candidates | all four |
| Pareto frontier | all four |
| Rejected candidates | none |

Q4 generic's p95 end-to-end latency was 0.214% below Q8, inside the declared 1% objective
tolerance. The preference order therefore retained Q8. Open `output/reproduction/report.html`
directly in a browser to inspect every tradeoff and source fingerprint.

## 4. Audit provenance and dispatch

The archive records:

- native runner, compiler, build, runtime, model, and evaluation-suite identity;
- exact `llama-bench` and `llama-server` command arrays;
- twenty prompt and generation throughput samples per candidate and workload;
- five fixed exact-answer smoke cases with matching outcomes across both server passes;
- twenty fixed 64-token TTFT and end-to-end latency samples per candidate after two warmups;
- peak RSS from the larger of two GNU `time -v` measurements;
- model size, quantization, batch, micro-batch, and KleidiAI dispatch state; and
- the closed experiment manifest, recommendation, report, status, and bundle checksums.

The two generic candidates must have zero `CPU_KLEIDIAI model buffer` markers in both server
passes. Both KleidiAI candidates must have exactly one marker per pass. The canonical archive has
counts `[0, 0]`, `[0, 0]`, `[1, 1]`, and `[1, 1]` respectively.

## 5. Replay a v1.1 archive

This section applies only after the current workflow produces a complete v1.1 artifact directory.
It does not apply to the published v1.0 archive, which intentionally lacks the additive files.

Use a new destination outside the extracted evidence directory:

```bash
python -m paretopilot replay evidence-v1.1 \
  --output-dir output/replay-v1.1
```

Replay verifies the complete `SHA256SUMS` list, safe paths, canonical completion status, and the
required v1.1 artifact set. It then:

1. reassembles the canonical benchmark and recommendation;
2. recomputes every policy profile from the archived policy configuration;
3. validates the declared load plan and each per-candidate load file, checks the request endpoint
   and both server-command digests, and rebuilds the combined load evaluation;
4. reconstructs both pass benchmark sets from their checksummed raw throughput, settings,
   server-evaluation, and GNU `time -v` files;
5. regenerates repeat stability from those pass sets; and
6. renders fresh v1.0 and v1.1 reports.

To inspect pass reconstruction independently, use fresh output paths:

```bash
python -m paretopilot assemble-repeat-pass \
  --experiment evidence-v1.1/experiment \
  --pass-number 1 \
  --output output/pass-1.json
python -m paretopilot assemble-repeat-pass \
  --experiment evidence-v1.1/experiment \
  --pass-number 2 \
  --output output/pass-2.json

cmp output/pass-1.json evidence-v1.1/extensions/benchmark-set-pass-1.json
cmp output/pass-2.json evidence-v1.1/extensions/benchmark-set-pass-2.json
```

The command follows only artifact paths and SHA-256 digests bound by the canonical benchmark; it
does not derive a pass by splitting pooled metrics.

`benchmark-set`, `recommendation`, `policy-profiles`, `load-evaluation`, both pass benchmark sets,
and `repeat-stability` are core replay comparisons. Any difference invalidates
`decision_reproduced`. HTML is presentation: if the verified core matches but a newly generated
report differs, replay remains valid, marks `fully_reproduced` false, and records a warning. Replay
does not rerun inference.

## 6. Dispatch a fresh native Arm64 study

From GitHub Actions, select **Native Arm64 candidate study**, choose **Run workflow**, use the
default branch, and retain `10` repetitions. The equivalent GitHub CLI command is:

```bash
gh workflow run candidate-study-arm64.yml --ref main -f repetitions=10
```

A run is canonical only when it is manually dispatched from the default branch with exactly ten
repetitions and every environment, build, dispatch, benchmark, quality, latency, memory,
integrity, selection, and report gate passes. Branch runs and changed inputs remain explicitly
exploratory. Failed workflows may upload diagnostic artifacts with `valid_evidence: false`; those
must not replace canonical evidence.

## Evidence limits

- The hosted runner is ephemeral. Do not pool separate workflow runs as one experiment.
- The published v1.0 five-case suite and v1.1 24-case suite are deterministic gates, not broad
  model-quality benchmarks.
- Two reconstructed balanced passes describe observed direction and spread; they do not establish
  statistical significance.
- Results may not generalize to every Arm CPU, model, prompt distribution, concurrency level, or
  deployment environment.
- Energy and cost were not measured and must not be inferred from throughput.
- Models and build trees are intentionally absent from the archive; their pinned identities and
  hashes remain present.

## Optional Performix profiling

Arm Performix is outside the required pipeline. A compatible host may add separate hotspot
context for the reference and selected candidate, but profiler output does not replace measured
throughput, quality, latency, memory, or checksums. Missing Performix output does not invalidate
ParetoPilot evidence.

For the experimental design and decision rules, see
[`benchmark-methodology.md`](benchmark-methodology.md).
