# ParetoPilot Devpost draft

This is claim-safe narrative copy for the currently published two-candidate evidence. It is not a
complete submission until the public demo, final evidence, account eligibility, and signed-in form
gates in the submission checklist are verified. If a final four-candidate canonical run is
published, update only the measured-results passages after completing the claim audit at the end
of this file.

## Form fields

**Project name**

ParetoPilot

**Tagline**

An evidence-first advisor that finds deployable AI inference configurations on Arm64.

**Track**

Cloud AI

**Source code**

https://github.com/agrovr/ParetoPilot

**Demo or test build**

Add the final public URL after verifying it in a signed-out browser. Until then, the checked-in
offline report is at `demo/canonical-study/report.html` in the public repository.

**Suggested technologies**

Arm64, Arm KleidiAI, Python, llama.cpp, Qwen2.5, GitHub Actions, HTML, JSON, SHA-256

**Video**

Omitted. Video is optional under the current challenge requirements.

## Submission narrative

### Inspiration

AI optimization is often reduced to a single speedup number. That can produce the wrong
deployment decision: a candidate can be faster but use more memory, miss a quality floor, or win
only because of run-order noise. We wanted a tool that treats optimization as an evidence and
decision problem, not a marketing claim.

ParetoPilot asks a practical question: given several inference configurations measured on the
same Arm64 target, which one should an engineering team actually deploy?

### What it does

ParetoPilot ingests controlled benchmark candidates, verifies their provenance, and evaluates
them against declared quality and resource guardrails. It rejects ineligible configurations,
computes the Pareto frontier across latency, throughput, memory, and quality, then selects the
best eligible candidate for the chosen objective.

The output is both machine-readable and human-readable. A deterministic offline HTML report
shows the verdict, baseline deltas, constraint failures, frontier membership, and source hashes.
When a study supplies validated `llama-server` arguments, the report exports them; otherwise it
refuses to guess a deployment command. The baseline is allowed to win when an optimization is not
convincing.

### How we built it

The core is a dependency-free Python 3.12 package with strict schemas for benchmark sets,
constraints, upstream `llama-bench` output, server evaluation, resource measurements, and
multi-candidate manifests. Inputs fail closed on malformed JSON, duplicate keys, non-finite
numbers, mismatched settings, missing fingerprints, or path escapes. Evidence-specific parsers
also bound records and artifacts, and synthetic data cannot be presented as measured evidence.

ParetoPilot was created during the challenge submission period; repository history preserves the
development timeline.

For Arm64 measurements, GitHub Actions builds a pinned `llama.cpp` revision in generic and Arm
KleidiAI-enabled variants. The first study used a balanced A-B-B-A sequence on one native Arm64
runner. The expanded workflow separates four attribution stages: a Q8 generic reference, Q4
quantization, Q4 with KleidiAI, and Q4 with KleidiAI plus runtime tuning. It also includes a fixed
quality smoke gate, two balanced fixed-length streamed-latency passes, peak RSS, raw samples,
exact commands, a predeclared 1% objective tolerance, and SHA-256
manifests.

Arm Performix is an optional profiling enhancement. ParetoPilot does not require it and does not
substitute profiler output for benchmark evidence.

### Current measured result

Our reviewed canonical run executed on a GitHub-hosted Ubuntu 24.04 Arm64 runner with an Arm
Neoverse-N2 CPU. It compared generic and KleidiAI-enabled CPU builds using the same pinned
Qwen2.5 1.5B Instruct Q4_0 model and ten repetitions per pass.

KleidiAI improved median 512-token prompt-processing throughput from 113.6605 to 114.2580 tokens
per second, a small but consistent 0.5257% gain. For 128-token generation, the pooled change was
only 0.0493%, and the paired results disagreed in direction (+2.1583% and -1.2145%). ParetoPilot
therefore rejected the optimized candidate under its predeclared consistency and 1% practical
effect gate and retained the generic baseline.

That outcome demonstrates the product's purpose: it did not turn an Arm optimization flag into
an inflated speedup claim. It preserved the positive prompt result, labeled generation
inconclusive, and recommended no deployment change. This first run is throughput-only; it does
not claim measured request latency, memory, energy, cost, or direct task-quality improvement.

### Challenges we ran into

The hardest part was making benchmark evidence harder to fool. Hosted runners are ephemeral,
upstream benchmark formats can drift, and small gains can disappear when run order changes. We
had to pin every meaningful input, validate runtime-reported settings against declarations,
preserve every repetition, verify that KleidiAI actually dispatched only in the intended builds,
and keep partial runs from being labeled valid.

We also had to represent an inconclusive result well. A recommendation engine that always chooses
the candidate named "optimized" is not useful. ParetoPilot instead encodes the adoption rule and
keeps rejection reasons visible in the report.

### Accomplishments that we are proud of

- Produced a reviewed, checksummed native Arm64 evidence bundle with raw samples, environment
  identity, exact commands, build and model fingerprints, paired comparisons, and explicit
  validity status.
- Built a strict multi-objective engine that can reject candidates before ranking them and can
  retain the baseline without hiding positive secondary metrics.
- Created a deterministic, responsive report that works offline and carries its source
  fingerprints with the decision.
- Kept synthetic fixtures, exploratory runs, and measured evidence visibly separate.
- Designed an expanded study that attributes quantization, Arm-kernel, and runtime-tuning changes
  instead of collapsing them into one opaque before-and-after comparison.

### Why this project should win

ParetoPilot turns Arm optimization work into a decision artifact another engineer can inspect,
reproduce, and deploy. Its sponsor integration is substantive: native Arm64 execution and verified
KleidiAI runtime dispatch are measured as separate attribution stages, not mentioned only as build
flags. Its strict verifier reassembles pooled metrics from balanced raw passes, binds commands and
inputs by SHA-256, and rejects rehashed tampering. The product is also useful when the honest answer
is "keep the baseline," which protects teams from adopting extra complexity for a noise-scale win.
That combination of Arm relevance, technical rigor, reusable developer experience, and transparent
tradeoff handling directly supports the challenge's impact, implementation, and optimization
goals.

### What we learned

Small performance deltas need experimental structure more than visual polish. Balanced ordering
made it clear that the generation result was not stable enough to deploy. We also learned that
"same model" can support a model-identity retention statement, but it is not a replacement for a
direct task-quality evaluation when quantization changes.

Most importantly, trustworthy optimization includes the option to stop. A transparent baseline
decision can save an engineering team from shipping extra complexity for no reproducible benefit.

### What's next

The next milestone is to complete and review the four-candidate canonical study, including direct
quality-smoke, TTFT, end-to-end p95 latency, peak RSS, model size, and prompt/generation throughput.
We will keep quantization and Arm-kernel effects separate, publish only checksummed evidence, and
update the report only after every gate passes.

Future work could add larger evaluation suites, concurrency curves, controlled dedicated Arm
instances, and optional Performix hotspot context. Energy and cost will remain out of scope until
they can be measured directly rather than estimated from throughput.

## Try it

With Python 3.12 or newer:

```bash
git clone https://github.com/agrovr/ParetoPilot.git
cd ParetoPilot
python -m venv .venv
# Activate .venv for your shell, then run:
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m paretopilot verify-study results/published/29940067201
```

Then follow [`docs/reproducibility.md`](reproducibility.md) to regenerate the checked-in decision
from measured evidence or dispatch the expanded native Arm64 workflow.

## Suggested gallery assets

Use these in this order:

1. Report hero with **Baseline retained** and the selected generic configuration.
2. Measured-impact table showing the small prompt gain and inconclusive generation result.
3. Candidate evidence and rejection reason for the KleidiAI configuration.
4. Architecture graphic showing evidence collection, validation, constraint filtering, Pareto
   selection, and report export.
5. Provenance section with source fingerprints and canonical run identity.

Do not use a synthetic report in the gallery. If the four-candidate run becomes the final demo,
regenerate all result screenshots from its reviewed canonical artifact.

## Update after a successful four-candidate run

Keep the current evidence section until the new artifact passes every checklist item. Then:

1. add the canonical run URL and published evidence path;
2. name the selected candidate and baseline exactly as recorded;
3. report only metrics present in the final `benchmark-set.json`;
4. specify p50 or p95, units, workload, repetition count, and comparison baseline;
5. describe the five-case exact-match evaluation as a smoke gate, not broad model quality;
6. explain every rejected candidate using recorded constraint reasons;
7. update screenshots and the demo URL; and
8. retain any negative result that materially explains the decision.

Never blend the two-candidate throughput numbers with four-candidate latency or memory numbers as
if they came from one run.
