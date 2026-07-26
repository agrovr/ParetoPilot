# ParetoPilot GitHub Action

The ParetoPilot decision gate turns a validated benchmark set and chosen limits into five CI
artifacts:

- `recommendation.json`: the selected configuration and the comparison behind it;
- `decision-passport.json`: machine-readable decision details, including source metadata,
  optimization stages, the shortlist cutoff, and a lower-resource alternative when one exists;
- `optimization-receipt.md`: a Markdown decision summary with the same selection, tradeoffs, and
  source details in a readable format;
- `report.html`: a self-contained HTML report; and
- `gate.json`: a compact CI result with the selected candidate and artifact SHA-256 digests.

Measured benchmark data is required by default. Set `require-measured` to `false` only for
synthetic tests or examples.

## Quick verification

1. Review the measured result in the
   [optimization ladder](https://agrovr.github.io/ParetoPilot/#optimization-ladder).
2. Open the
   [latest green main-branch CI run](https://github.com/agrovr/ParetoPilot/actions/workflows/ci.yml?query=branch%3Amain)
   and inspect `action-smoke`, which exercises all five artifacts and checks that synthetic data
   is rejected when measured input is required.
3. From an installed ParetoPilot checkout, run the bundled synthetic software smoke test:

```bash
python -m paretopilot ci-gate examples/synthetic-results.json --constraints configs/constraints.example.json --output-dir paretopilot-output --allow-synthetic --expect-selected-id q4-kleidiai
```

The bundled fixture demonstrates the gate contract only. It is explicitly synthetic and is not
Arm64 benchmark evidence.

## Starter project

Create a self-contained starter project without copying paths or workflow YAML by hand:

```bash
python -m paretopilot init ../paretopilot-example
```

The new folder contains:

- `.github/workflows/paretopilot.yml`;
- `benchmarks/benchmark-set.json`;
- `constraints/deployment.json`;
- `.gitignore`; and
- a step-by-step `README.md`.

The command creates a new directory and never overwrites an existing path. The generated workflow
uses read-only repository access and clearly labels the included data as synthetic. Its README
explains how to replace the example with measured Arm64 inputs.

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
the Action. Use the [starter project](#starter-project) to generate a clearly synthetic example.

Pin `agrovr/ParetoPilot` to a reviewed commit SHA in a production workflow. The example already
pins the public v1.4.0 commit.

`expected-selected-id` is optional. When supplied, the job fails if a benchmark change selects a
different candidate. This catches unexpected selection changes in CI without assuming that an
optimized candidate must beat the baseline.

`require-arm64-provenance` is also optional. Enable it to require Arm64 runner, run, source,
runtime, model, and evaluation-suite metadata. This checks that the fields are present; it does
not independently authenticate them. The published release adds checksummed source files and
exact replay. Leave this input off for the synthetic example above.

## Inputs

| Input | Required | Default | Meaning |
| --- | --- | --- | --- |
| `benchmarks` | yes | — | Path to a ParetoPilot benchmark-set JSON file |
| `constraints` | yes | — | Path to a ParetoPilot constraints JSON file |
| `output-dir` | no | `paretopilot-output` | New or empty artifact directory |
| `require-measured` | no | `true` | Reject explicitly synthetic inputs |
| `expected-selected-id` | no | empty | Optional selected-candidate regression check |
| `require-arm64-provenance` | no | `false` | Reject inputs missing required Arm64 runner and source metadata |

## Outputs

| Output | Meaning |
| --- | --- |
| `selected-id` | Candidate selected by the chosen policy |
| `synthetic-source` | `true` only for an explicitly allowed synthetic smoke test |
| `recommendation` | Path to `recommendation.json` |
| `report` | Path to `report.html` |
| `decision-passport` | Path to `decision-passport.json`, the machine-readable decision details |
| `optimization-receipt` | Path to `optimization-receipt.md`, the Markdown decision summary |
| `evidence-grade` | `synthetic`, `measured-unattributed`, or `arm64-attributed`; the last grade means source-declared metadata is complete, not independently authenticated |
| `receipt` | Path to `gate.json` |
| `recommendation-sha256` | Recommendation digest |
| `report-sha256` | Report digest |
| `decision-passport-sha256` | SHA-256 digest of `decision-passport.json` |
| `optimization-receipt-sha256` | SHA-256 digest of `optimization-receipt.md` |
| `receipt-sha256` | Receipt digest |

The Action also adds the Markdown decision summary to the GitHub Actions job summary.

## Local equivalent

The same gate is available without GitHub Actions. For measured evidence, replace the
`benchmarks/benchmark-set.json` and `constraints/deployment.json` paths below with files from your
consuming repository:

```bash
python -m paretopilot ci-gate benchmarks/benchmark-set.json --constraints constraints/deployment.json --output-dir paretopilot-output --require-arm64-provenance --expect-selected-id q4-kleidiai
```

For a deliberately synthetic source-checkout test, use the exact command in the
[quick verification](#quick-verification) instead.

To export only the machine-readable decision details:

```bash
python -m paretopilot passport benchmarks/benchmark-set.json \
  --constraints constraints/deployment.json \
  --output decision-passport.json \
  --require-arm64-provenance
```

To export a Markdown decision summary from the same inputs:

```bash
python -m paretopilot optimization-receipt benchmarks/benchmark-set.json \
  --constraints constraints/deployment.json \
  --output optimization-receipt.md \
  --require-arm64-provenance
```

Use `--allow-synthetic` only for a deliberately synthetic software smoke test. The command rejects
invalid input, failed constraints, unexpected selections, existing output directories, and
synthetic data unless it is explicitly allowed.
