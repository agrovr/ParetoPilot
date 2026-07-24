# ParetoPilot GitHub Action

The ParetoPilot decision gate turns a validated benchmark set and declared constraints into three
CI artifacts:

- `recommendation.json`: the machine-readable selection, frontier, rejections, deltas, and input
  fingerprints;
- `report.html`: the self-contained decision report; and
- `gate.json`: a compact receipt with the selected candidate and artifact SHA-256 digests.

Measured evidence is required by default. An explicitly synthetic benchmark is accepted only when
`require-measured` is set to `false`, which keeps smoke tests from being mistaken for deployment
evidence.

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
        uses: agrovr/ParetoPilot@main
        with:
          benchmarks: benchmarks/benchmark-set.json
          constraints: constraints/deployment.json
          output-dir: paretopilot-output
          expected-selected-id: q4-kleidiai
      - uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
        with:
          name: paretopilot-decision
          path: paretopilot-output
          if-no-files-found: error
```

Pin `agrovr/ParetoPilot` to a reviewed commit SHA in a production workflow. The `@main` reference
above keeps the introductory example easy to try.

`expected-selected-id` is optional. When supplied, the job fails if a benchmark change selects a
different candidate. This makes the Action useful as a deployment-regression gate without
assuming that an optimized candidate must beat the baseline.

## Inputs

| Input | Required | Default | Meaning |
| --- | --- | --- | --- |
| `benchmarks` | yes | — | Strict ParetoPilot benchmark-set JSON |
| `constraints` | yes | — | Strict ParetoPilot constraints JSON |
| `output-dir` | no | `paretopilot-output` | New or empty artifact directory |
| `require-measured` | no | `true` | Reject explicitly synthetic inputs |
| `expected-selected-id` | no | empty | Optional selected-candidate regression check |

## Outputs

| Output | Meaning |
| --- | --- |
| `selected-id` | Candidate selected by the declared policy |
| `synthetic-source` | `true` only for an explicitly allowed synthetic smoke test |
| `recommendation` | Path to `recommendation.json` |
| `report` | Path to `report.html` |
| `receipt` | Path to `gate.json` |
| `recommendation-sha256` | Recommendation digest |
| `report-sha256` | Report digest |
| `receipt-sha256` | Receipt digest |

The Action also writes a compact decision table to the GitHub Actions job summary.

## Local equivalent

The same gate is available without GitHub Actions:

```bash
python -m paretopilot ci-gate benchmarks/benchmark-set.json \
  --constraints constraints/deployment.json \
  --output-dir paretopilot-output \
  --expect-selected-id q4-kleidiai
```

Use `--allow-synthetic` only for a deliberately synthetic software smoke test. Invalid JSON,
unknown schema fields, non-finite metrics, failed constraints, unexpected selections, existing
outputs, and synthetic inputs presented as measured evidence all fail closed.
