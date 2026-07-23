# Reproducing ParetoPilot

ParetoPilot has two reproducibility levels:

1. **Repository reproduction** verifies the software and regenerates the current report from the
   checked-in canonical evidence on any supported development host.
2. **Measurement reproduction** dispatches a fresh native Arm64 experiment. A fresh run is new
   evidence, not an exact hardware replay of an ephemeral hosted runner.

The current published evidence is the two-candidate throughput study from GitHub Actions run
[`29940067201`](https://github.com/agrovr/ParetoPilot/actions/runs/29940067201). The four-candidate
workflow is implemented but must not be described as final measured evidence until a successful
canonical artifact has been reviewed and published.

## Requirements

For local verification:

- Git;
- Python 3.12 or newer; and
- enough access to create a virtual environment and local output files.

For a fresh measurement run:

- a public GitHub repository or fork with Actions enabled;
- access to the standard `ubuntu-24.04-arm` hosted runner;
- network access during the job to fetch pinned sources and model files; and
- enough runner time and storage for two `llama.cpp` builds and the pinned Q8 and Q4 models.

The Python package has no runtime dependencies. Development checks use the bounded tools in the
`dev` extra. GitHub runner availability and billing policy are external conditions; verify them
for the account and repository before dispatching repeated experiments.

## 1. Clean local verification

Clone the public repository and create a Python 3.12 environment:

```bash
git clone https://github.com/agrovr/ParetoPilot.git
cd ParetoPilot
python -m venv .venv
```

Activate it with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell or
`source .venv/bin/activate` on Linux and macOS. Then install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m paretopilot --version
```

If the environment is activated, `paretopilot` can replace `python -m paretopilot` in the commands
below.

Run the full local verification set:

```bash
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
python -m build --wheel
```

CI defines the same quality checks for Ubuntu, Windows, and native Arm64. A workflow definition is
not proof of a passing run; verify the commit's GitHub checks before release.

## 2. Verify the published evidence

ParetoPilot's verifier checks the compact bundle, recomputes the paired comparisons, rebuilds the
selection inputs, and confirms the resulting recommendation without writing output files:

```bash
python -m paretopilot verify-study results/published/29940067201
```

A valid run reports two candidates and selects `generic-baseline`. It also reports that the
KleidiAI adoption gate is ineligible because paired generation directions are inconsistent and
the pooled change does not reach the 1% threshold.

On a system with GNU `sha256sum`, the repository bundle can also be checked directly:

```bash
cd results/published/29940067201
sha256sum --check SHA256SUMS
cd ../../..
```

`SOURCE-SHA256SUMS` is the checksum manifest retained from the complete Actions artifact.
`SHA256SUMS` covers the reviewed compact repository bundle. Read
[`results/published/29940067201/README.md`](../results/published/29940067201/README.md) before
interpreting either one.

## 3. Regenerate the current decision report

Use fresh output paths. Multi-file commands preflight their destinations, and ParetoPilot refuses
to overwrite existing evidence or reports by default.

```bash
python -m paretopilot assemble-study results/published/29940067201 --benchmarks-output output/reproduction/benchmark-set.json --constraints-output output/reproduction/constraints.json --assembly-output output/reproduction/assembly.json
python -m paretopilot report output/reproduction/benchmark-set.json --constraints output/reproduction/constraints.json --output output/reproduction/report.html --recommendation-output output/reproduction/recommendation.json
```

Expected decision:

| Field | Expected value |
| --- | --- |
| Synthetic source | `false` |
| Baseline | `generic-baseline` |
| Selected candidate | `generic-baseline` |
| Frontier | `generic-baseline` |
| Rejected candidate | `kleidiai-optimized` |

The HTML contains no external assets or generation timestamp. For identical benchmark and
constraint inputs, it is deterministic; the embedded SHA-256 fingerprints identify those inputs.
Open `output/reproduction/report.html` directly in a browser.

## 4. Understand the current evidence boundary

The published two-candidate study supports these statements:

- native Arm64 Linux execution on the recorded GitHub-hosted Neoverse-N2 runner;
- generic versus KleidiAI `llama.cpp` throughput with one identical Q4_0 model;
- ten repetitions in each of two balanced passes per variant;
- a +0.5257% pooled median prompt-processing throughput change; and
- an inconclusive +0.0493% pooled generation-throughput change.

It does not support claims about request TTFT, p95 latency, RSS, energy, cost, concurrency, every
Arm processor, or direct downstream-task quality. Matching model hashes establish model identity
between those two runtime candidates, not an independent quality score.

## 5. Dispatch the expanded native Arm64 study

The candidate workflow compares four attribution stages on one runner:

1. Q8 generic reference;
2. Q4 generic quantization;
3. Q4 with KleidiAI; and
4. Q4 with KleidiAI and a 512-token micro-batch.

From the GitHub Actions interface, select **Native Arm64 candidate study**, choose **Run workflow**,
select the default branch, and retain `10` repetitions. With GitHub CLI, the equivalent is:

```bash
gh workflow run candidate-study-arm64.yml --ref main -f repetitions=10
```

The workflow classifies a run as canonical only when all of the following are true:

- it was manually dispatched;
- it ran from the repository's default branch;
- the input was exactly ten repetitions; and
- every environment, build, dispatch, benchmark, quality, latency, memory, integrity, selection,
  and report gate passed.

A branch run or changed repetition count can still be useful for debugging, but its artifact is
labeled `EXPLORATORY` and must not replace canonical evidence.

## 6. What the expanded workflow records

All candidates share the declared model family and pinned `llama.cpp` revision. The workflow
records:

- native runner, compiler, build, runtime, model, and evaluation-suite identity;
- exact `llama-bench` and `llama-server` command arrays;
- balanced prompt-processing and generation-throughput samples;
- a fixed five-case exact-match quality score;
- streamed TTFT and end-to-end p50 and p95 latency over twenty fixed-length measured requests,
  pooled from two balanced passes after one warmup per pass;
- process peak RSS from GNU `time -v`;
- model size, quantization, batch, micro-batch, and KleidiAI dispatch state;
- per-artifact SHA-256 references, the assembled benchmark set, constraints, recommendation, and
  self-contained report; and
- a final status distinguishing complete canonical, complete exploratory, and incomplete runs.

The quality suite at [`evals/qwen-smoke-v1.json`](../evals/qwen-smoke-v1.json) is intentionally
small. It is a deterministic smoke gate for this experiment, not a claim of broad language-model
quality. The predeclared constraints require full retention of the reference score, bound p95
latency and peak RSS, and apply a 1% objective tolerance with a simpler-first preference before a
noise-scale latency lead can change the deployment recommendation.

## 7. Review and preserve a candidate artifact

After a successful workflow:

1. record the run URL, run ID, attempt, repository commit, runner identity, and artifact ID;
2. download the artifact before its retention window expires;
3. run `sha256sum --check SHA256SUMS` from the artifact root;
4. confirm `status.json` says `complete`, `measurement_valid: true`, and the expected canonical or
   exploratory classification;
5. confirm the generic logs have no KleidiAI dispatch marker and both KleidiAI variants do;
6. run `python -m paretopilot assemble-experiment` against the downloaded manifest and compare its
   digest with the included benchmark set;
7. open the offline report and reconcile the verdict with `recommendation.json`; and
8. publish a compact reviewed bundle before the original artifact expires.

Models and build directories are intentionally not uploaded. Do not add downloaded GGUF files,
temporary build trees, secrets, or credentials to the repository.

## 8. Optional Performix enrichment

Arm Performix is not part of either required workflow. If a compatible host exposes the necessary
hardware counters, a separate profile may add hotspot context for a baseline and selected
candidate. Keep that output visibly optional, identify its host and capture settings, and do not
use it to replace throughput, quality, latency, or memory measurements. A missing Performix trace
does not invalidate ParetoPilot evidence.

## Troubleshooting and failure semantics

- `doctor --require-evidence-host` exits nonzero unless the host is native Arm64 Linux.
- Existing output paths are rejected to prevent silent evidence replacement.
- Malformed, oversized, duplicate-key, non-finite, mismatched, or synthetic evidence fails closed.
- Incomplete workflows upload short-lived diagnostic artifacts with `valid_evidence: false`.
- Hosted-runner results can vary. Do not combine separate runs unless CPU identity and runner image
  match and the aggregation method is explicitly justified.
- A workflow that fails after collecting some samples is still incomplete evidence.

For the experimental design and interpretation rules, see
[`docs/benchmark-methodology.md`](benchmark-methodology.md). For release readiness, use
[`docs/submission-checklist.md`](submission-checklist.md).
