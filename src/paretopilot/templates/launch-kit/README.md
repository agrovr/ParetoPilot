# ParetoPilot Launch Kit

> This folder is a synthetic software smoke test. Its candidate numbers are examples, not
> measured Arm64 results.

## Run the decision locally

ParetoPilot requires Python 3.12 or newer.

```bash
python -m pip install --no-deps "https://github.com/agrovr/ParetoPilot/archive/db9ccaf37e3c7e807832652e237de813675ed807.zip"
paretopilot ci-gate benchmarks/benchmark-set.json --constraints constraints/deployment.json --output-dir paretopilot-output --allow-synthetic --expect-selected-id q4-kleidiai
```

Expected: exit code `0`, selected ID `q4-kleidiai`, and these five files:

- `recommendation.json`
- `decision-passport.json`
- `optimization-receipt.md`
- `report.html`
- `gate.json`

Open `paretopilot-output/optimization-receipt.md` first.

## Run it in GitHub Actions

Push this folder to a GitHub repository and run **ParetoPilot decision**. The workflow uses the
same synthetic inputs and uploads the five decision artifacts.

## Replace the example with measured Arm64 evidence

1. Replace every example metric and candidate setting with values from one controlled measurement.
2. Keep `"synthetic": true` until that replacement is complete. Never relabel example numbers as
   measurements.
3. Add complete source, runner, runtime, model, and evaluation-suite metadata.
4. Change `require-measured` to `"true"`.
5. Enable `require-arm64-provenance: "true"` only after the metadata is complete.
6. Update or remove `expected-selected-id` after reviewing the first measured result.
7. Keep both ParetoPilot pins at a reviewed commit SHA; update them only after review.

The Launch Kit demonstrates the decision workflow. It does not collect measurements or replace
ParetoPilot's published Arm64 evidence.
