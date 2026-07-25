# ParetoPilot GitHub Action

The ParetoPilot decision gate turns a validated benchmark set and declared constraints into five
CI artifacts:

- `recommendation.json`: the machine-readable selection, frontier, rejections, deltas, and input
  fingerprints;
- `decision-passport.json`: supplementary source-declared attribution grade, optimization ladder,
  cutoff runway, and eligible measured resource alternative;
- `optimization-receipt.md`: a deterministic, human-readable decision receipt with the selected
  candidate, objective, stage attribution, tradeoffs, provenance, and source fingerprints;
- `report.html`: the self-contained decision report; and
- `gate.json`: a compact receipt with the selected candidate and artifact SHA-256 digests.

Measured evidence is required by default. An explicitly synthetic benchmark is accepted only when
`require-measured` is set to `false`, which keeps smoke tests from being mistaken for deployment
evidence. The passport describes the decision but never replaces or changes the recommendation.

## 60-second proof path

1. Review the measured result in the
   [optimization ladder](https://agrovr.github.io/ParetoPilot/#optimization-ladder).
2. Open the
   [latest green main-branch CI run](https://github.com/agrovr/ParetoPilot/actions/workflows/ci.yml?query=branch%3Amain)
   and inspect `action-smoke`, which exercises all five artifacts and verifies the fail-closed
   measured-evidence guard.
3. From an installed ParetoPilot checkout, run the bundled synthetic software smoke test:

```bash
python -m paretopilot ci-gate examples/synthetic-results.json --constraints configs/constraints.example.json --output-dir paretopilot-output --allow-synthetic --expect-selected-id q4-kleidiai
```

The bundled fixture demonstrates the gate contract only. It is explicitly synthetic and is not
Arm64 benchmark evidence.

## Launch Kit

Create a self-contained beginner project without copying paths or workflow YAML by hand:

```bash
python -m paretopilot init ../paretopilot-launch-kit-demo
```

The new folder contains:

- `.github/workflows/paretopilot.yml`;
- `benchmarks/benchmark-set.json`;
- `constraints/deployment.json`;
- `.gitignore`; and
- a step-by-step `README.md`.

The command refuses every existing destination and has no overwrite mode. If a filesystem write
fails after the destination is claimed, the command reports and preserves the incomplete folder;
it never performs a destructive rollback. The generated workflow pins the reviewed v1.4.0 Action
commit, grants read-only repository access, and labels its input as synthetic. The generated
README explains how to replace the example with measured Arm64 inputs before enabling the
measured-evidence and provenance gates.

## Workflow example

```yaml
name: Deployment decision

on:
  workflow_dispatch:
  push:
    paths:
      - "benchmarks/**"
      - "constraints/**"

permissions:
  contents: read

jobs:
  paretopilot:
    runs-on: ubuntu-24.04-arm
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          persist-credentials: false
      - uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0
        with:
          python-version: "3.12"
      - id: decision
        uses: agrovr/ParetoPilot@db9ccaf37e3c7e807832652e237de813675ed807 # v1.4.0
        with:
          benchmarks: benchmarks/benchmark-set.json
          constraints: constraints/deployment.json
          output-dir: paretopilot-output
          expected-selected-id: q4-kleidiai
          require-arm64-provenance: "true"
      - uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
        with:
          name: paretopilot-decision
          path: paretopilot-output
          if-no-files-found: error
```

The two input paths in this workflow are measured files supplied by the repository that consumes
the Action. Use the [Launch Kit](#launch-kit) to generate a clearly synthetic starter project.

Pin `agrovr/ParetoPilot` to a reviewed commit SHA in a production workflow. The example already
pins the public v1.4.0 commit.

`expected-selected-id` is optional. When supplied, the job fails if a benchmark change selects a
different candidate. This makes the Action useful as a deployment-regression gate without
assuming that an optimized candidate must beat the baseline.

`require-arm64-provenance` is also optional. Enable it for a native Arm64 deployment gate to
require complete source-declared runner, run, source, runtime, model, and evaluation-suite
identities. This is a metadata-completeness gate, not authentication of those claims or a
cryptographic binding to candidate artifacts. The canonical release supplies the stronger layer:
checksummed source artifacts, a frozen evidence lock, and exact replay. Leave the input off for
the explicitly synthetic software smoke path above.

## Inputs

| Input | Required | Default | Meaning |
| --- | --- | --- | --- |
| `benchmarks` | yes | — | Strict ParetoPilot benchmark-set JSON |
| `constraints` | yes | — | Strict ParetoPilot constraints JSON |
| `output-dir` | no | `paretopilot-output` | New or empty artifact directory |
| `require-measured` | no | `true` | Reject explicitly synthetic inputs |
| `expected-selected-id` | no | empty | Optional selected-candidate regression check |
| `require-arm64-provenance` | no | `false` | Require complete source-declared Arm64 attribution metadata |

## Outputs

| Output | Meaning |
| --- | --- |
| `selected-id` | Candidate selected by the declared policy |
| `synthetic-source` | `true` only for an explicitly allowed synthetic smoke test |
| `recommendation` | Path to `recommendation.json` |
| `report` | Path to `report.html` |
| `decision-passport` | Path to `decision-passport.json` |
| `optimization-receipt` | Path to `optimization-receipt.md` |
| `evidence-grade` | `synthetic`, `measured-unattributed`, or `arm64-attributed`; the last grade means source-declared metadata is complete, not independently authenticated |
| `receipt` | Path to `gate.json` |
| `recommendation-sha256` | Recommendation digest |
| `report-sha256` | Report digest |
| `decision-passport-sha256` | Decision-passport digest |
| `optimization-receipt-sha256` | Optimization-receipt digest |
| `receipt-sha256` | Receipt digest |

The Action also writes the human-readable Optimization Receipt to the GitHub Actions job summary.

## Local equivalent

The same gate is available without GitHub Actions. For measured evidence, replace the
`benchmarks/benchmark-set.json` and `constraints/deployment.json` paths below with files from your
consuming repository:

```bash
python -m paretopilot ci-gate benchmarks/benchmark-set.json --constraints constraints/deployment.json --output-dir paretopilot-output --require-arm64-provenance --expect-selected-id q4-kleidiai
```

For a deliberately synthetic source-checkout test, use the exact command in the
[60-second proof path](#60-second-proof-path) instead.

To export only the supplementary machine-readable context:

```bash
python -m paretopilot passport benchmarks/benchmark-set.json \
  --constraints constraints/deployment.json \
  --output decision-passport.json \
  --require-arm64-provenance
```

To export the same context as a deterministic Markdown receipt:

```bash
python -m paretopilot optimization-receipt benchmarks/benchmark-set.json \
  --constraints constraints/deployment.json \
  --output optimization-receipt.md \
  --require-arm64-provenance
```

Use `--allow-synthetic` only for a deliberately synthetic software smoke test. Invalid JSON,
unknown schema fields, non-finite metrics, failed constraints, unexpected selections, existing
outputs, and synthetic inputs presented as measured evidence all fail closed.
