# ParetoPilot submission checklist

This is the release gate for the **Cloud AI** track of the Arm AI Optimization Challenge. It
separates completed repository work from evidence and account actions that still require review.

## Dates

- Official submission deadline: **Friday, August 14, 2026 at 4:00 PM PDT / 6:00 PM CDT**.
- Internal target: **Thursday, August 13, 2026 at 6:00 PM CDT**.
- Keep the public project or test build available through at least **September 4, 2026**.

Use the [official challenge dates](https://arm-ai-optimization-challenge.devpost.com/details/dates)
as the final authority if Devpost posts a change.

## Account and eligibility gate

- [x] The participant confirmed that the eligibility statement applies to them.
- [x] The participant confirmed agreement to the
  [official rules](https://arm-ai-optimization-challenge.devpost.com/rules) and Devpost terms.
- [ ] Confirm enrollment in the
  [Arm Developer Program](https://developer.arm.com/arm-developer-program) while signed in.
- [ ] Confirm every listed team member independently satisfies the official eligibility rules.
- [ ] Confirm the correct team and owner are selected in the signed-in Devpost submission.

Do not infer Arm Developer Program enrollment from an Arm, GitHub, or Devpost account. Record the
confirmation before final submission.

## Product and evidence gate

- [x] Public Apache-2.0 source repository exists at
  [agrovr/ParetoPilot](https://github.com/agrovr/ParetoPilot).
- [x] Canonical two-candidate native Arm64 evidence is checked in under
  [`results/published/29940067201/`](../results/published/29940067201/README.md).
- [x] The current report truthfully retains the baseline because the measured generation result
  is inconsistent and below the predeclared practical-effect threshold.
- [x] Synthetic examples are visibly labeled and excluded from performance claims.
- [x] Arm Performix is optional and no claim depends on it.
- [x] A deterministic offline report generator and machine-readable recommendation are present.
- [ ] Complete the four-candidate workflow on the default branch with canonical inputs.
- [ ] Download its artifact before expiration and verify every entry in `SHA256SUMS`.
- [ ] Review logs for native Arm64 identity, pinned inputs, KleidiAI dispatch, and successful gates.
- [ ] Publish a compact final evidence bundle in the repository without model files or build trees.
- [ ] Replace or supplement the current demo report only after the final evidence review.
- [ ] Trace every number in the final Devpost text and screenshots to a published artifact.

The existing two-candidate result is valid evidence and can support an honest submission if the
expanded run does not pass. In that case, keep its throughput-only boundaries prominent and do
not claim measured latency, RSS, or direct task quality.

## Repository and release gate

- [x] Setup and reproduction instructions are documented in
  [`docs/reproducibility.md`](reproducibility.md).
- [x] A Devpost narrative draft exists in [`docs/devpost-draft.md`](devpost-draft.md).
- [ ] All unit tests pass from a clean Python 3.12 environment.
- [ ] Ruff check and format verification pass.
- [ ] Wheel build and isolated install smoke test pass.
- [ ] GitHub CI passes on Ubuntu, Windows, and native Arm64.
- [ ] Third-party notices identify the pinned runtime, Arm library, and model licenses.
- [ ] Architecture graphic is checked in and legible at Devpost display size.
- [ ] Create a final version tag and GitHub release after evidence is frozen.
- [ ] Verify every repository link from a signed-out browser.
- [ ] Preserve the final evidence and demo beyond September 4, 2026.

## Devpost deliverables

- [x] Project name: **ParetoPilot**.
- [x] Track: **Cloud AI**.
- [x] English tagline and project story drafted.
- [x] Public source URL identified.
- [x] Video intentionally omitted because it is optional under the current requirements.
- [ ] Add the final public demo or test-build URL and verify it without authentication.
- [ ] Add final screenshots: verdict, measured impact, candidate evidence, and provenance.
- [ ] Upload the architecture graphic.
- [ ] Update the draft if the reviewed four-candidate run produces stronger measured evidence.
- [ ] Confirm all technologies and third-party assets are accurately credited.
- [ ] Preview the complete submission on desktop and mobile.
- [ ] Save the draft, reopen it, and verify that formatting and links survived.
- [ ] Submit through the
  [signed-in Devpost form](https://devpost.com/submit-to/30218-arm-create-ai-optimization-challenge/manage/submissions)
  before the official deadline.
- [ ] Capture the submitted confirmation page and final public submission URL.

The user must perform or explicitly authorize the final signed-in submission. A saved draft is
not proof that Devpost accepted the entry.

## Final claim audit

Before submitting, read the narrative and every screenshot as a judge would:

- Is each performance claim tied to a run ID, candidate, workload, statistic, and unit?
- Are throughput-only results kept separate from server-level latency and memory results?
- Is the five-case exact-match suite described as a smoke gate rather than general model quality?
- Are exploratory branch outputs excluded from canonical claims?
- Are negative and inconclusive findings shown, not hidden?
- Does the recommendation match the checked-in `recommendation.json`?
- Is the selected launch command copied from authoritative recorded arguments rather than guessed?
- Are Performix, energy, cost, and broad Arm-CPU claims omitted unless separately measured?

## Stop-ship conditions

Do not submit a new performance claim if any of these remain true:

- the evidence bundle is incomplete, exploratory, expired, or fails checksums;
- runner architecture or CPU identity is missing;
- model, runtime, evaluation-suite, or command fingerprints do not match;
- a screenshot shows synthetic data without a visible synthetic warning;
- the public demo requires the submitter's credentials;
- required license notices are missing; or
- Arm Developer Program enrollment has not been confirmed.
