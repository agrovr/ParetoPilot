"""Tests for the published-results presentation layer."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from paretopilot.decision_passport import build_decision_passport
from paretopilot.domain import BenchmarkSet, ValidationError
from paretopilot.profiles import PolicySet, evaluate_policy_profiles
from paretopilot.showcase import (
    _capacity_failure_label,
    _capacity_result_from_study,
    _comparison_bar_widths,
    _metric_label,
    _metric_value,
    _optimization_ladder_markup,
    _plain_language_report_copy,
    _relative_measure_phrase,
    render_showcase_v11,
)

from test_capacity_eval import CapacityFixture
from test_report_v11 import (
    canonical_benchmarks,
    canonical_constraints,
    canonical_recommendation,
    derived_profiles,
    measured_load_sweep,
    measured_stability,
    rendered_v11,
)


def attributed_benchmarks() -> BenchmarkSet:
    benchmarks = canonical_benchmarks()
    attribution_stages = {
        "q8-generic": "reference",
        "q4-generic": "quantization",
        "q4-kleidiai": "arm-kernel",
        "q4-kleidiai-tuned": "runtime-tuning",
    }
    candidates = []
    for candidate in benchmarks.candidates:
        parameters = deepcopy(dict(candidate.parameters))
        configuration = deepcopy(dict(parameters["configuration"]))
        configuration["attribution_stage"] = attribution_stages[candidate.candidate_id]
        parameters["configuration"] = configuration
        candidates.append(
            {
                "id": candidate.candidate_id,
                "label": candidate.label,
                "parameters": parameters,
                "metrics": dict(candidate.metrics),
            }
        )
    return BenchmarkSet.from_mapping(
        {
            "schema_version": benchmarks.schema_version,
            "baseline_id": benchmarks.baseline_id,
            "synthetic": benchmarks.synthetic,
            "metadata": dict(benchmarks.metadata),
            "candidates": candidates,
        }
    )


def contrast_ratio(foreground: str, background: str) -> float:
    def luminance(value: str) -> float:
        channels = [int(value[index : index + 2], 16) / 255 for index in range(1, len(value), 2)]
        linear = [
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    foreground_luminance = luminance(foreground)
    background_luminance = luminance(background)
    lighter = max(foreground_luminance, background_luminance)
    darker = min(foreground_luminance, background_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def css_rule_body(document: str, selector: str) -> str:
    marker = f"{selector} {{"
    selector_start = document.find(marker)
    if selector_start == -1:
        raise AssertionError(f"CSS rule is missing: {selector}")

    opening_brace = selector_start + len(marker) - 1
    depth = 0
    for index in range(opening_brace, len(document)):
        if document[index] == "{":
            depth += 1
        elif document[index] == "}":
            depth -= 1
            if depth == 0:
                return document[opening_brace + 1 : index]
    raise AssertionError(f"CSS rule is not closed: {selector}")


def css_hex_tokens(document: str, selector: str) -> dict[str, str]:
    tokens = {
        name: value.lower()
        for name, value in re.findall(
            r"(--[\w-]+)\s*:\s*(#[0-9a-fA-F]{6})\s*;",
            css_rule_body(document, selector),
        )
    }
    if not tokens:
        raise AssertionError(f"CSS rule has no hexadecimal tokens: {selector}")
    return tokens


def evidence_lock(
    *,
    artifacts_sha256: dict[str, str] | None = None,
) -> dict[str, object]:
    if artifacts_sha256 is None:
        artifacts_sha256 = {
            "benchmark_set": "a" * 64,
            "recommendation": "b" * 64,
            "policy_profiles": "c" * 64,
            "load_evaluation": "d" * 64,
            "repeat_stability": "e" * 64,
            "report_v1_1": ("bda915aba2b748b10daf510c25e931795051878ce3b75734217be05532e67f1b"),
        }
    return {
        "schema_version": "1.1",
        "classification": "canonical",
        "source": {"run_id": "not supplied"},
        "archive": {
            "release_tag": "v1.1.0",
            "release_url": "https://github.com/agrovr/ParetoPilot/releases/tag/v1.1.0",
            "sha256": "f" * 64,
        },
        "review": {
            "checksum_entries": 150,
            "checksum_manifest_sha256": "a" * 64,
            "artifacts_sha256": artifacts_sha256,
            "all_checksums_verified": True,
            "exact_file_coverage": True,
            "status_complete": True,
            "measurement_valid": True,
            "valid_evidence": True,
            "replay": {
                "valid": True,
                "decision_reproduced": True,
                "fully_reproduced": True,
                "report_matches_archive": True,
                "differences": [],
                "warnings": [],
                "checksum_entry_count": 150,
                "checksum_manifest_sha256": "a" * 64,
                "selected_id": "q8-generic",
                "authoritative_comparisons": [
                    "benchmark-set",
                    "benchmark-set-pass-1",
                    "benchmark-set-pass-2",
                    "load-evaluation",
                    "policy-profiles",
                    "recommendation",
                    "repeat-stability",
                    "report",
                    "report-v1.1",
                ],
            },
        },
    }


def frozen_v11_recommendation(data: BenchmarkSet) -> dict[str, object]:
    recommendation = canonical_recommendation(data)
    recommendation["paretopilot_version"] = "1.1.0"
    return recommendation


def frozen_v11_profiles(data: BenchmarkSet) -> dict[str, object]:
    profiles = derived_profiles(data)
    for profile in profiles["profiles"]:
        profile["recommendation"]["paretopilot_version"] = "1.1.0"
    return profiles


def cockpit_profiles(data: BenchmarkSet) -> dict[str, object]:
    source_kind = "synthetic fixture" if data.synthetic else "measured evidence"
    policy_set = PolicySet.from_mapping(
        {
            "schema_version": "1.0",
            "canonical_profile_id": "canonical-latency",
            "profiles": [
                {
                    "id": "canonical-latency",
                    "label": "Canonical latency",
                    "description": "Use the predeclared p95 end-to-end latency objective.",
                    "classification": "canonical",
                    "objective": {"metric": "e2e_latency_ms_p95", "direction": "min"},
                    "objective_tolerance_percent": 1.0,
                    "preference_policy": "canonical",
                },
                {
                    "id": "memory-first",
                    "label": "Memory first",
                    "description": f"Prefer the lowest peak memory in the {source_kind}.",
                    "classification": "derived-non-canonical",
                    "objective": {"metric": "peak_rss_mib", "direction": "min"},
                    "objective_tolerance_percent": 0.0,
                    "preference_policy": "none",
                },
                {
                    "id": "first-token-first",
                    "label": "First token first",
                    "description": f"Prefer the lowest p95 first-token latency in the {source_kind}.",
                    "classification": "derived-non-canonical",
                    "objective": {"metric": "ttft_ms_p95", "direction": "min"},
                    "objective_tolerance_percent": 0.0,
                    "preference_policy": "none",
                },
            ],
        }
    )
    profiles = dict(
        evaluate_policy_profiles(
            data,
            canonical_constraints(),
            policy_set,
        )
    )
    profiles["input_fingerprints"] = {
        "benchmarks_sha256": "a" * 64,
        "constraints_sha256": "f" * 64,
        "policies_sha256": "9" * 64,
    }
    for profile in profiles["profiles"]:
        profile["recommendation"]["paretopilot_version"] = "1.1.0"
    return profiles


def rendered_cockpit_showcase(
    *,
    benchmarks: BenchmarkSet | None = None,
    profiles: dict[str, object] | None = None,
) -> str:
    data = canonical_benchmarks() if benchmarks is None else benchmarks
    policy_profiles = cockpit_profiles(data) if profiles is None else profiles
    return render_showcase_v11(
        data,
        frozen_v11_recommendation(data),
        policy_profiles=policy_profiles,
        canonical_report_href="evidence/report-v1.1.html",
        benchmarks_sha256="a" * 64,
        recommendation_sha256="b" * 64,
        profiles_sha256="c" * 64,
    )


def rendered_showcase(
    *,
    lock: bool = True,
    canonical_html: str | None = None,
    load_sweep: dict[str, object] | None = None,
    benchmarks: BenchmarkSet | None = None,
) -> str:
    data = canonical_benchmarks() if benchmarks is None else benchmarks
    recommendation = frozen_v11_recommendation(data)
    profiles = frozen_v11_profiles(data)
    load = measured_load_sweep() if load_sweep is None else load_sweep
    canonical = rendered_v11(data=data, load=load)
    supplied_canonical = (canonical if canonical_html is None else canonical_html) if lock else None
    lock_payload = evidence_lock() if lock else None
    if lock_payload is not None:
        review = lock_payload["review"]
        assert isinstance(review, dict)
        artifacts = review["artifacts_sha256"]
        assert isinstance(artifacts, dict)
        artifacts["report_v1_1"] = hashlib.sha256(canonical.encode()).hexdigest()
    return render_showcase_v11(
        data,
        recommendation,
        policy_profiles=profiles,
        load_sweep=load,
        stability_summary=measured_stability(data),
        evidence_lock=lock_payload,
        canonical_html=supplied_canonical,
        canonical_report_href="evidence/report-v1.1.html",
        benchmarks_sha256="a" * 64,
        recommendation_sha256="b" * 64,
        profiles_sha256="c" * 64,
        load_sha256="d" * 64,
        stability_sha256="e" * 64,
    )


def capacity_benchmarks() -> BenchmarkSet:
    data = canonical_benchmarks()
    metadata = deepcopy(dict(data.metadata))
    metadata["source"] = {
        "repository": "agrovr/ParetoPilot",
        "run_id": "30055662526",
        "runner": {
            "os": "Ubuntu 24.04",
            "architecture": "arm64",
            "cpu": "Neoverse-N2",
            "cpu_count": 4,
        },
    }
    return BenchmarkSet.from_mapping(
        {
            "schema_version": data.schema_version,
            "baseline_id": data.baseline_id,
            "synthetic": data.synthetic,
            "metadata": metadata,
            "candidates": [
                {
                    "id": candidate.candidate_id,
                    "label": candidate.label,
                    "parameters": dict(candidate.parameters),
                    "metrics": dict(candidate.metrics),
                }
                for candidate in data.candidates
            ],
        }
    )


def capacity_evidence_lock(
    study: dict[str, object],
    *,
    study_sha256: str = "7" * 64,
) -> dict[str, object]:
    provenance = study["provenance"]
    assert isinstance(provenance, dict)
    source = provenance["source"]
    runner = provenance["runner"]
    assert isinstance(source, dict)
    assert isinstance(runner, dict)
    result = _capacity_result_from_study(study)
    return {
        "schema_version": "1.4",
        "classification": "supplementary-capacity",
        "source": {
            "run_id": source["run_id"],
            "run_attempt": source["run_attempt"],
            "head_sha": source["revision"],
            "workflow": source["workflow"],
            "runner": deepcopy(runner),
        },
        "archive": {
            "actions_digest": f"sha256:{'8' * 64}",
            "release_tag": "v1.4.0",
            "release_asset_name": "paretopilot-v1.4.0-arm64-capacity.zip",
            "release_asset_url": (
                "https://github.com/agrovr/ParetoPilot/releases/download/v1.4.0/"
                "paretopilot-v1.4.0-arm64-capacity.zip"
            ),
            "size_bytes": 12345,
            "sha256": "8" * 64,
        },
        "canonical_evidence": {
            "run_id": "30055662526",
            "release_tag": "v1.1.0",
            "release_sha256": ("b5586878ccd214667911390f417db0417111ac2c31d163a2f5f55c4469aefeb2"),
            "lock_sha256": ("9a00187cb4619daec3596139c97de49127841ceb3c2c7edd85092df2474c578d"),
            "outputs_modified": False,
        },
        "review": {
            "checksum_entries": 121,
            "checksum_manifest_sha256": "6" * 64,
            "all_checksums_verified": True,
            "archive_digest_matches_actions_digest": True,
            "exact_file_coverage": True,
            "status_complete": True,
            "measurement_valid": True,
            "valid_evidence": True,
            "synthetic": False,
            "canonical_outputs_modified": False,
            "artifacts_sha256": {"capacity_study": study_sha256},
            "recomputation": {
                "raw_inputs_reassembled": True,
                "capacity_study_exact_match": True,
                "capacity_receipt_regenerated": True,
                "capacity_receipt_exact_match": True,
                "recomputed_cell_count": result["cell_count"],
                "mismatched_cell_count": 0,
                "measured_request_count": result["measured_request_count"],
                "completed_request_count": result["measured_request_count"],
                "failed_request_count": 0,
            },
            "replay": {
                "valid": True,
                "decision_reproduced": True,
                "fully_reproduced": True,
                "authoritative_outputs_match": True,
                "report_matches_archive": True,
                "differences": [],
                "warnings": [],
            },
        },
        "result": result,
    }


def rendered_capacity_showcase(
    study: dict[str, object],
    *,
    capacity_lock: dict[str, object] | None = None,
    capacity_study_href: str = "evidence/capacity-study.json",
    capacity_receipt_href: str = "evidence/capacity-receipt.md",
) -> str:
    data = capacity_benchmarks()
    recommendation = frozen_v11_recommendation(data)
    profiles = frozen_v11_profiles(data)
    load = measured_load_sweep()
    canonical = rendered_v11(data=data, load=load)
    canonical_lock = evidence_lock()
    canonical_source = canonical_lock["source"]
    canonical_archive = canonical_lock["archive"]
    canonical_review = canonical_lock["review"]
    assert isinstance(canonical_source, dict)
    assert isinstance(canonical_archive, dict)
    assert isinstance(canonical_review, dict)
    canonical_source["run_id"] = "30055662526"
    canonical_archive["sha256"] = "b5586878ccd214667911390f417db0417111ac2c31d163a2f5f55c4469aefeb2"
    artifacts = canonical_review["artifacts_sha256"]
    assert isinstance(artifacts, dict)
    artifacts["report_v1_1"] = hashlib.sha256(canonical.encode()).hexdigest()
    return render_showcase_v11(
        data,
        recommendation,
        policy_profiles=profiles,
        load_sweep=load,
        stability_summary=measured_stability(data),
        evidence_lock=canonical_lock,
        evidence_lock_sha256=("9a00187cb4619daec3596139c97de49127841ceb3c2c7edd85092df2474c578d"),
        canonical_html=canonical,
        canonical_report_href="evidence/report-v1.1.html",
        capacity_study=study,
        capacity_evidence_lock=capacity_lock or capacity_evidence_lock(study),
        capacity_study_sha256="7" * 64,
        capacity_study_href=capacity_study_href,
        capacity_receipt_href=capacity_receipt_href,
        benchmarks_sha256="a" * 64,
        recommendation_sha256="b" * 64,
        profiles_sha256="c" * 64,
        load_sha256="d" * 64,
        stability_sha256="e" * 64,
    )


class ShowcaseV11Tests(unittest.TestCase):
    def test_archived_renderer_fixture_digest_remains_frozen(self) -> None:
        canonical = rendered_v11()

        self.assertEqual(
            hashlib.sha256(canonical.encode()).hexdigest(),
            "bda915aba2b748b10daf510c25e931795051878ce3b75734217be05532e67f1b",
        )
        self.assertNotIn('class="showcase', canonical)
        self.assertNotIn("Measured Flight Log", canonical)

    def test_showcase_is_deterministic_and_keeps_the_evidence_story(self) -> None:
        first = rendered_showcase()
        second = rendered_showcase()

        self.assertEqual(first.encode(), second.encode())
        self.assertIn("<title>ParetoPilot | Arm64 inference results</title>", first)
        self.assertIn('<link rel="icon" href="data:image/svg+xml,', first)
        self.assertNotIn('<link rel="icon" href="data:,">', first)
        self.assertIn(
            '<meta property="og:title" content="ParetoPilot | Arm64 inference results">',
            first,
        )
        self.assertIn(
            '<meta name="twitter:description" '
            'content="Reproducible Arm64 inference results from ParetoPilot&#x27;s '
            'published benchmark archives.">',
            first,
        )
        self.assertIn('class="showcase is-verified"', first)
        self.assertIn(
            'Compare measured Arm64 <span class="hero-selection">inference configurations.</span>',
            first,
        )
        self.assertIn(
            "selected Q8 generic reference using the stated quality requirements",
            first,
        )
        self.assertIn("The published v1.1 archive was verified and reproduced.", first)
        self.assertIn("1.00% decision tolerance", first)
        self.assertIn("150 files verified", first)
        self.assertIn("9 outputs replayed and matched", first)
        self.assertIn("arm64 vCPUs", first)
        self.assertIn("View archived v1.1 report", first)
        self.assertIn("View v1.1.0 release", first)
        self.assertIn('href="evidence/report-v1.1.html"', first)
        self.assertIn("Published latency result", first)
        self.assertIn("Alternative priority", first)
        self.assertNotIn("submission decision", first.lower())
        self.assertNotIn("uses the submission", first.lower())
        self.assertIn("Verification and reproduction", first)
        self.assertIn("Benchmark SHA-256", first)
        self.assertIn("Published report SHA-256", first)
        self.assertIn("Evidence archive SHA-256", first)
        self.assertIn("Checksum manifest SHA-256", first)
        self.assertIn("Published run", first)
        self.assertIn("<dt>Report version</dt>", first)

    def test_policy_cockpit_uses_only_three_precomputed_profiles_above_the_fold(
        self,
    ) -> None:
        report = rendered_cockpit_showcase()
        cockpit_start = report.index('<section class="policy-cockpit"')
        cockpit_end = report.index("</section>\n", cockpit_start) + len("</section>\n")
        cockpit = report[cockpit_start:cockpit_end]
        hero_start = report.index('<div class="hero-layout">')
        hero_end = report.index('</div>\n<nav class="flight-log"', hero_start)

        self.assertLess(hero_start, cockpit_start)
        self.assertLess(cockpit_end, hero_end)
        self.assertEqual(cockpit.count('data-cockpit-target="'), 3)
        self.assertEqual(cockpit.count('data-cockpit-panel="'), 3)
        self.assertIn("<strong>Latency first</strong><span>Published</span>", cockpit)
        self.assertIn("<strong>Memory first</strong><span>Alternative</span>", cockpit)
        self.assertIn("<strong>First token first</strong><span>Alternative</span>", cockpit)
        self.assertIn("<h3>Q8 generic reference</h3>", cockpit)
        self.assertIn("<h3>Q4 generic</h3>", cockpit)
        self.assertIn("<dt>Objective</dt><dd>p95 end-to-end latency</dd>", cockpit)
        self.assertIn("<dt>Result</dt><dd>2,335.9 ms</dd>", cockpit)
        self.assertIn(
            "<strong>43.7% lower</strong><span>model size vs baseline</span>",
            cockpit,
        )
        self.assertIn(
            "<strong>5.0% lower</strong><span>generation throughput vs baseline</span>",
            cockpit,
        )
        self.assertNotIn('class="decision-rail"', report[hero_start:hero_end])
        self.assertIn(
            '<div class="profile-tabs" role="tablist" aria-label="Deployment priorities">',
            report,
        )

    def test_policy_cockpit_links_to_the_generated_optimization_receipt(self) -> None:
        report = rendered_cockpit_showcase()

        self.assertIn(
            '<a href="evidence/optimization-receipt.md">View calculation details</a>',
            report,
        )
        self.assertEqual(report.count('href="evidence/optimization-receipt.md"'), 1)
        self.assertIn(
            "Verified sources → Declared limits → Pareto frontier → Selected priority",
            report,
        )

    def test_policy_cockpit_has_one_tab_stop_focusable_panels_and_full_keyboard_contract(
        self,
    ) -> None:
        report = rendered_cockpit_showcase()
        cockpit_start = report.index('<section class="policy-cockpit"')
        cockpit_end = report.index("</section>\n", cockpit_start) + len("</section>\n")
        cockpit = report[cockpit_start:cockpit_end]

        self.assertIn(
            'role="tablist" aria-label="Choose a deployment priority"',
            cockpit,
        )
        self.assertEqual(cockpit.count('role="tab"'), 3)
        self.assertEqual(cockpit.count('role="tabpanel"'), 3)
        self.assertEqual(cockpit.count('tabindex="0"'), 4)
        self.assertEqual(cockpit.count('tabindex="-1"'), 2)
        self.assertEqual(cockpit.count('aria-selected="true"'), 1)
        self.assertEqual(cockpit.count('aria-selected="false"'), 2)
        self.assertEqual(cockpit.count('class="cockpit-panel" role="tabpanel"'), 3)
        self.assertEqual(cockpit.count('data-cockpit-panel="0" tabindex="0"'), 1)
        self.assertEqual(cockpit.count('data-cockpit-panel="1" tabindex="0" hidden'), 1)
        self.assertEqual(cockpit.count('data-cockpit-panel="2" tabindex="0" hidden'), 1)
        self.assertIn('aria-live="polite" aria-atomic="true"', cockpit)
        self.assertIn(
            "The latency-first summary is shown here; all policy results appear in the "
            "detailed section below.",
            cockpit,
        )
        for key in ("ArrowLeft", "ArrowRight", "Home", "End"):
            self.assertIn(f'"{key}"', report)
        self.assertIn(
            'panel.hidden = panel.getAttribute("data-cockpit-panel") !== target;',
            report,
        )
        self.assertIn('item.setAttribute("tabindex", active ? "0" : "-1");', report)
        self.assertIn(
            'tab.scrollIntoView({ block: "nearest", inline: "nearest" });',
            report,
        )
        self.assertIn(".showcase .cockpit-tabs { display: none; }", report)

    def test_policy_cockpit_uses_fixture_language_for_synthetic_inputs(self) -> None:
        measured = canonical_benchmarks()
        synthetic = BenchmarkSet.from_mapping(
            {
                "schema_version": measured.schema_version,
                "baseline_id": measured.baseline_id,
                "synthetic": True,
                "metadata": dict(measured.metadata),
                "candidates": [
                    {
                        "id": candidate.candidate_id,
                        "label": candidate.label,
                        "parameters": dict(candidate.parameters),
                        "metrics": dict(candidate.metrics),
                    }
                    for candidate in measured.candidates
                ],
            }
        )
        report = rendered_cockpit_showcase(benchmarks=synthetic)
        cockpit_start = report.index('<section class="policy-cockpit"')
        cockpit_end = report.index("</section>\n", cockpit_start) + len("</section>\n")
        cockpit = report[cockpit_start:cockpit_end]

        self.assertIn(
            "Each tab shows a decision already calculated from the same synthetic example.",
            cockpit,
        )
        self.assertIn("<span>Primary fixture</span>", cockpit)
        self.assertEqual(cockpit.count("<span>Derived fixture</span>"), 2)
        self.assertIn("<dt>Fixture improvement</dt>", cockpit)
        self.assertIn("<dt>Fixture tradeoff</dt>", cockpit)
        self.assertIn("synthetic example", cockpit)
        self.assertIn("This example keeps its starting configuration.", cockpit)
        self.assertNotIn("measured", cockpit.lower())

    def test_policy_cockpit_explains_a_retained_baseline_plainly(self) -> None:
        report = rendered_cockpit_showcase()
        cockpit_start = report.index('<section class="policy-cockpit"')
        cockpit_end = report.index("</section>\n", cockpit_start) + len("</section>\n")
        cockpit = report[cockpit_start:cockpit_end]
        canonical_start = cockpit.index('id="cockpit-panel-0"')
        canonical_end = cockpit.index("</section>", canonical_start)
        canonical_panel = cockpit[canonical_start:canonical_end]

        self.assertEqual(cockpit.count("<strong>None</strong>"), 2)
        self.assertIn(
            "The baseline already had the best result for this priority.",
            cockpit,
        )
        self.assertIn("There is no change from the baseline.", cockpit)
        self.assertIn(
            "<dt>Largest improvement</dt><dd><strong>None</strong>"
            "<span>The baseline already had the best result for this priority.</span>",
            canonical_panel,
        )
        self.assertIn(
            "<dt>Main tradeoff</dt><dd><strong>None</strong>"
            "<span>There is no change from the baseline.</span>",
            canonical_panel,
        )
        self.assertNotIn("does not improve", cockpit)
        self.assertNotIn("does not worsen", cockpit)

    def test_showcase_rephrases_frozen_report_copy_without_changing_it(self) -> None:
        source = (
            "ParetoPilot retained the measured baseline under the predeclared "
            "objective tolerance and preference policy. "
            "ParetoPilot retained the measured baseline because no alternative "
            "delivered a better eligible objective result on the declared frontier. "
            "Canonical measured evidence. Canonical policy. "
            "This is the canonical predeclared decision. "
            "Derived scenario only; it does not replace the canonical decision. "
            "ParetoPilot keeps the primary recommendation, derived policy scenarios, "
            "measurements, and evidence limits visibly separate. "
            "Preference order selected a candidate within the objective tolerance instead "
            "of the numeric objective winner. "
            "No preference order was supplied; selected the lexicographically earliest "
            "candidate id from the objective-tolerance shortlist. "
            "Canonical latency<span>Canonical</span> "
            "Published latency priority<span>Canonical</span> "
            "Memory first<span>Derived</span> "
            "Uses the submission&#x27;s end-to-end p95 latency objective, 1% practical "
            "tolerance, and declared preference order. "
            "The predeclared objective, tolerance, constraints, and preference order. "
            "The predeclared latency objective and simpler-first preference. "
            "Configuration binding: every measured load command was validated as materially "
            "identical to its canonical deployment command. Only the declared host or port "
            "binding may differ. Validated load-to-deployment command bindings. "
            "Source-supplied benchmark metadata. Reproduction contract. "
            "Rebuild the benchmark set and primary recommendation from the validated inputs. "
            "Render into a fresh path and compare the resulting artifact hash. "
            "“Better” and “Tradeoff” use only declared frontier directions. "
            "The predeclared cutoff was 100 ms. Predeclared objective tolerance. "
            "canonical deployment command. Canonical parallel. Canonical command SHA-256."
        )

        rendered = _plain_language_report_copy(source)

        self.assertIn(
            "After applying the stated tolerance and preference rules, the baseline "
            "remains the recommendation.",
            rendered,
        )
        self.assertIn(
            "No other configuration that passed every constraint performed better on "
            "the selected objective, so the baseline remains the recommendation.",
            rendered,
        )
        self.assertIn("Published Arm64 results", rendered)
        self.assertIn("Published latency priority", rendered)
        self.assertIn("This is the published latency result.", rendered)
        self.assertIn(
            "This is an alternative priority using the same measurements.",
            rendered,
        )
        self.assertIn(
            "The published recommendation, alternative priorities, measurements, "
            "and limitations are shown separately.",
            rendered,
        )
        self.assertIn(
            "The stated preference order selected a configuration within the tolerance range.",
            rendered,
        )
        self.assertIn(
            "No preference order was supplied. Candidate ID was used to resolve results "
            "within the tolerance range.",
            rendered,
        )
        self.assertIn("Latency first<span>Published</span>", rendered)
        self.assertIn("Published latency priority<span>Published</span>", rendered)
        self.assertIn("Memory first<span>Alternative</span>", rendered)
        self.assertIn(
            "Uses p95 end-to-end latency, a 1% practical tolerance, and the declared "
            "preference order.",
            rendered,
        )
        self.assertIn("The tolerance cutoff was 100 ms", rendered)
        self.assertIn("Tolerance defined before the run", rendered)
        self.assertIn("recorded deployment command", rendered)
        self.assertIn("Recorded parallel", rendered)
        self.assertIn("Recorded command SHA-256", rendered)
        self.assertIn(
            "Uses the main objective, tolerance, limits, and preference order.",
            rendered,
        )
        self.assertIn(
            "Uses p95 latency and prefers the simpler configuration when results are "
            "within tolerance.",
            rendered,
        )
        self.assertIn(
            "Each load test used the same settings as its recorded server command; only "
            "the host or port could differ.",
            rendered,
        )
        self.assertIn("Test command match:", rendered)
        self.assertIn("Recorded load and server commands", rendered)
        self.assertIn("Benchmark source details", rendered)
        self.assertIn("How to reproduce this result", rendered)
        self.assertIn(
            "Rebuild the benchmark data and published recommendation from the verified inputs.",
            rendered,
        )
        self.assertIn(
            "Generate a new report and compare its SHA-256 hash.",
            rendered,
        )
        self.assertIn(
            "Labels follow the stated better-or-worse direction for each metric.",
            rendered,
        )
        self.assertNotIn("<span>Canonical</span>", rendered)
        self.assertNotIn("materially identical", rendered)
        self.assertNotIn("predeclared", rendered)
        self.assertNotIn("eligible objective result", rendered)
        self.assertNotIn("Canonical measured evidence", rendered)
        self.assertNotIn("Derived scenario only", rendered)
        self.assertNotIn("derived policy scenarios", rendered)
        self.assertNotIn("submission", rendered.lower())
        self.assertNotIn("predeclared", rendered.lower())

    def test_policy_cockpit_falls_back_when_a_required_profile_is_missing(self) -> None:
        complete = cockpit_profiles(canonical_benchmarks())
        incomplete = deepcopy(complete)
        incomplete["profiles"] = [
            profile for profile in incomplete["profiles"] if profile["id"] != "first-token-first"
        ]
        report = rendered_cockpit_showcase(profiles=incomplete)
        hero_start = report.index('<div class="hero-layout">')
        hero_end = report.index('</div>\n<nav class="flight-log"', hero_start)

        self.assertNotIn('class="policy-cockpit"', report)
        self.assertIn('class="decision-rail"', report[hero_start:hero_end])
        self.assertIn("Selected p95 end-to-end latency", report[hero_start:hero_end])
        self.assertIn("Within tolerance", report[hero_start:hero_end])
        self.assertIn("Decision tolerance", report[hero_start:hero_end])
        self.assertIn("All deployment priorities", report)

    def test_optimization_ladder_is_passport_derived_accessible_and_first(self) -> None:
        benchmarks = attributed_benchmarks()
        report = rendered_showcase(benchmarks=benchmarks)
        passport = build_decision_passport(benchmarks, canonical_constraints())
        ladder = passport["ladder"]
        self.assertIsInstance(ladder, list)

        main_start = report.index('<main id="main-content" class="report-main" tabindex="-1">')
        ladder_start = report.index(
            '<section id="optimization-ladder" class="optimization-ladder" '
            'aria-labelledby="optimization-ladder-heading">'
        )
        canonical_start = report.index(
            '<section class="report-section" aria-labelledby="why-heading">'
        )
        self.assertLess(main_start, ladder_start)
        self.assertLess(ladder_start, canonical_start)
        self.assertIn(
            (
                '<h2 id="optimization-ladder-heading">'
                "Four configurations, compared step by step.</h2>"
            ),
            report,
        )
        self.assertIn(
            '<ol class="optimization-stages" style="--stage-count: 4" '
            'aria-label="Measured optimization stages">',
            report,
        )
        self.assertIn(
            'role="group" aria-label="Decision tolerance details"',
            report,
        )
        self.assertIn("Decision tolerance", report)
        self.assertIn("Tolerance cutoff", report)
        self.assertIn("Selected configuration", report)
        self.assertIn("Closest option outside the cutoff", report)
        self.assertIn("inside the current cutoff", report)
        self.assertIn("outside the current cutoff", report)
        self.assertIn(
            "These comparisons use the supplied measurements and do not change the "
            "selected configuration.",
            report,
        )
        self.assertIn("Results apply only to this runner, model, and workload.", report)
        self.assertIn(
            (
                "<span>Evidence status</span>"
                "<strong>Measured; Arm64 source not fully attributed</strong>"
            ),
            report,
        )

        stage_labels = ("Reference", "Quantization", "KleidiAI build", "Runtime tuning")
        stage_positions = [
            report.index(f'<p class="stage-role">{label}</p>') for label in stage_labels
        ]
        self.assertEqual(stage_positions, sorted(stage_positions))
        for stage in ladder:
            self.assertIn(f"<h3>{stage['label']}</h3>", report)
            self.assertIn(f'<code class="stage-id">{stage["candidate_id"]}</code>', report)
        self.assertEqual(report.count('class="stage-technical-change"'), 4)
        self.assertIn("Reference configuration", report)
        self.assertIn(
            "<span>Quantization</span> <code>Q8_0</code>",
            report,
        )
        self.assertIn(
            "<span>Quantization</span> <code>Q8_0</code> "
            '<span class="technical-change-arrow" aria-label="to">→</span> '
            "<code>Q4_0</code>",
            report,
        )
        self.assertIn(
            "<span>KleidiAI</span> <code>off</code> "
            '<span class="technical-change-arrow" aria-label="to">→</span> '
            "<code>on</code>",
            report,
        )
        self.assertIn(
            "<span>Micro-batch</span> <code>128</code> "
            '<span class="technical-change-arrow" aria-label="to">→</span> '
            "<code>512</code>",
            report,
        )

        arm_stage = next(stage for stage in ladder if stage["attribution_stage"] == "arm-kernel")
        objective_metric = str(passport["objective"]["metric"])
        objective_change = next(
            change
            for change in arm_stage["delta_from_previous"]["metrics"]
            if change["metric"] == objective_metric
        )
        expected_direction = "lower" if objective_change["percent"] < 0 else "higher"
        expected_change = f"{abs(objective_change['percent']):,.2f}% {expected_direction}"
        self.assertIn(expected_change, report)

        self.assertIn(
            '<a class="action-secondary" href="#optimization-ladder">Compare configurations</a>',
            report,
        )
        self.assertIn(
            '<a href="#optimization-ladder"><strong>00</strong>Configurations</a>',
            report,
        )
        scripts = "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", report, re.DOTALL))
        self.assertNotIn("optimization-ladder", scripts)

    def test_capacity_envelope_is_strict_semantic_and_follows_the_ladder(self) -> None:
        with TemporaryDirectory() as directory:
            study = CapacityFixture(Path(directory)).assemble()
            report = rendered_capacity_showcase(study)

        ladder_start = report.index('<section id="optimization-ladder"')
        capacity_start = report.index('<section id="capacity-envelope"')
        canonical_start = report.index(
            '<section class="report-section" aria-labelledby="why-heading">'
        )
        capacity_end = report.index("</section>\n", capacity_start)
        capacity = report[capacity_start:capacity_end]

        self.assertLess(ladder_start, capacity_start)
        self.assertLess(capacity_start, canonical_start)
        self.assertIn(
            '<p class="section-kicker">Capacity study</p>',
            capacity,
        )
        self.assertIn(
            "Measured Arm64 results · Report v1.1 · Capacity v1.4",
            report,
        )
        self.assertIn("Reference configuration", capacity)
        self.assertIn(
            '<a href="evidence/capacity-receipt.md">Open capacity details</a>',
            capacity,
        )
        selections = study["selections"]
        assert isinstance(selections, list)
        selected_coordinates = {
            (
                int(selection["selected_cell"]["server_parallel"]),
                int(selection["selected_cell"]["client_concurrency"]),
            )
            for selection in selections
            if isinstance(selection, dict) and isinstance(selection.get("selected_cell"), dict)
        }
        self.assertEqual(len(selected_coordinates), 1)
        parallel, concurrency = selected_coordinates.pop()
        self.assertIn(
            (
                f'<h2 id="capacity-heading">Selected capacity point: '
                f"{parallel} server slots / {concurrency} clients</h2>"
            ),
            capacity,
        )
        self.assertIn(
            f"{parallel} server slots and {concurrency} concurrent clients",
            capacity,
        )
        self.assertEqual(capacity.count('class="capacity-board '), 2)
        self.assertEqual(capacity.count('class="capacity-cell '), 18)
        self.assertEqual(capacity.count('data-capacity-state="selected"'), 2)
        cells = study["cells"]
        assert isinstance(cells, list)
        expected_blocked = sum(
            isinstance(cell, dict)
            and isinstance(cell.get("summary"), dict)
            and cell["summary"].get("capacity_gate_met") is not True
            for cell in cells
        )
        self.assertEqual(
            capacity.count('data-capacity-state="blocked"'),
            expected_blocked,
        )
        self.assertEqual(capacity.count("<table"), 2)
        self.assertEqual(capacity.count("<caption>"), 2)
        self.assertEqual(
            capacity.count(
                'class="capacity-table-wrap" role="region" tabindex="0" aria-label="Scrollable '
            ),
            2,
        )
        self.assertEqual(
            capacity.count(
                '<p class="capacity-scroll-hint">Swipe to inspect every client level →</p>'
            ),
            2,
        )
        self.assertEqual(capacity.count('<th scope="row">'), 6)
        self.assertIn("Measured requests</dt><dd>288</dd>", capacity)
        self.assertIn("Repeat passes</dt><dd>2</dd>", capacity)
        self.assertIn(
            f"Blocked points</dt><dd>{expected_blocked}</dd>",
            capacity,
        )
        for gate_label in (
            "TTFT p95",
            "E2E p95",
            "Peak RSS",
            "Completion",
            "Quality",
            "Quality outcomes",
            "Generation stability",
            "E2E stability",
        ):
            with self.subTest(gate_label=gate_label):
                self.assertIn(f"<dt>{gate_label}</dt>", capacity)
        self.assertIn("Same result at every server-slot setting", capacity)
        self.assertIn('aria-label="Capacity limits"', capacity)
        self.assertIn("View capacity data (JSON)", capacity)
        self.assertIn(
            "This study compares the measured server settings for each candidate.",
            capacity,
        )
        self.assertIn(
            "Each displayed p95 is the median of two pass-level p95 values",
            capacity,
        )
        self.assertIn(
            '<li><a href="#capacity-envelope"><strong>S1</strong>Capacity</a></li>',
            report,
        )
        self.assertNotIn("winner", capacity.lower())
        mobile_css = css_rule_body(report, "@media (max-width: 47.99rem)")
        self.assertIn(
            ".showcase .capacity-matrix { min-width: 27rem; }",
            mobile_css,
        )

    def test_capacity_failure_labels_explain_every_declared_gate(self) -> None:
        cases = (
            (
                ["forward:completion_rate_below_minimum"],
                "Completion",
                "Completion rate was below the predeclared minimum.",
            ),
            (
                ["forward:ttft_ms_p95_above_maximum"],
                "TTFT",
                "The observed p95 time to first token exceeded the declared limit.",
            ),
            (
                ["reverse:e2e_latency_ms_p95_above_maximum"],
                "E2E",
                "The observed p95 end-to-end latency exceeded the declared limit.",
            ),
            (
                ["server_peak_rss_above_maximum"],
                "Memory",
                "Peak server memory exceeded the predeclared limit.",
            ),
            (
                ["quality_gate_failed"],
                "Quality",
                "The task-specific quality guard failed.",
            ),
            (
                ["throughput_relative_spread_above_maximum"],
                "Throughput stability",
                "Generation throughput varied too much between the mirrored passes.",
            ),
            (
                ["e2e_relative_spread_above_maximum"],
                "E2E stability",
                "End-to-end latency varied too much between the mirrored passes.",
            ),
        )
        for reasons, expected_short, expected_explanation in cases:
            with self.subTest(reasons=reasons):
                short, explanation = _capacity_failure_label(reasons)
                self.assertEqual(short, expected_short)
                self.assertEqual(explanation, expected_explanation)

        short, explanation = _capacity_failure_label(
            [
                "one_or_more_passes_failed_load_slo",
                "forward:ttft_ms_p95_above_maximum",
                "reverse:e2e_latency_ms_p95_above_maximum",
            ]
        )
        self.assertEqual(short, "TTFT + E2E")
        self.assertIn("time to first token", explanation)
        self.assertIn("end-to-end latency", explanation)

    def test_capacity_flight_brief_puts_measured_decision_and_proof_first(self) -> None:
        with TemporaryDirectory() as directory:
            study = CapacityFixture(Path(directory)).assemble()
            result = _capacity_result_from_study(study)
            report = rendered_capacity_showcase(study)
        provenance = study["provenance"]
        assert isinstance(provenance, dict)
        capacity_source = provenance["source"]
        assert isinstance(capacity_source, dict)
        capacity_run_id = str(capacity_source["run_id"])

        selected_points = result["selected_operating_points"]
        assert isinstance(selected_points, dict)
        alternative = next(
            point
            for point in selected_points.values()
            if isinstance(point, dict) and point.get("role") == "resource-alternative"
        )
        comparisons = result["q4_vs_q8_at_selected_points_percent"]
        assert isinstance(comparisons, dict)
        parallel = int(alternative["server_parallel"])
        concurrency = int(alternative["client_concurrency"])
        quality = alternative["quality"]
        assert isinstance(quality, dict)
        throughput_phrase = _relative_measure_phrase(
            float(comparisons["generated_tokens_per_second_median"]),
            "generation throughput",
        )
        rss_phrase = _relative_measure_phrase(
            float(comparisons["server_peak_rss_mib_max"]),
            "peak RSS",
        )
        benchmarks = canonical_benchmarks()
        recommendation = canonical_recommendation(benchmarks)
        objective = recommendation["objective"]
        assert isinstance(objective, dict)
        objective_metric = str(objective["metric"])
        selected = benchmarks.by_id(str(recommendation["selected_id"]))
        objective_summary = (
            f"{_metric_value(objective_metric, selected.metrics[objective_metric])} · "
            f"{_metric_label(objective_metric)}"
        )

        hero_start = report.index('<div class="hero-layout">')
        brief_start = report.index('<section class="flight-brief"', hero_start)
        brief_end = report.index("</section>\n", brief_start) + len("</section>\n")
        actions_start = report.index('<nav class="hero-actions"', brief_end)
        capacity_start = report.index('<section id="capacity-envelope"', actions_start)
        brief = report[brief_start:brief_end]
        verdict_start = brief.index(
            '<div class="flight-brief-verdict" role="note" aria-label="Decision summary">'
        )
        instrument_start = brief.index('<figure class="flight-brief-instrument"')

        self.assertLess(hero_start, brief_start)
        self.assertLess(brief_start, brief_end)
        self.assertLess(brief_end, actions_start)
        self.assertLess(actions_start, capacity_start)
        self.assertIn("Measured results", brief)
        self.assertIn("Latency choice and serving capacity", brief)
        self.assertIn(
            "The first study chooses a configuration for single-client latency. "
            "The second compares server settings for two candidates.",
            brief,
        )
        self.assertEqual(brief.count('role="note" aria-label="Decision summary"'), 1)
        self.assertLess(verdict_start, instrument_start)
        self.assertIn("Latency recommendation", brief)
        self.assertIn(objective_summary, brief)
        self.assertIn("Capacity alternative", brief)
        self.assertIn(f"{parallel} server slots / {concurrency} clients", brief)
        self.assertIn(
            f"Measured tradeoff at {parallel} server slots / {concurrency} clients",
            brief,
        )
        self.assertIn(str(alternative["label"]), brief)
        self.assertIn(throughput_phrase, brief)
        self.assertIn(rss_phrase, brief)
        self.assertIn(
            f"{float(quality['retention_vs_reference']) * 100:.1f}% quality retention",
            brief[verdict_start:instrument_start],
        )
        self.assertIn(
            (
                f'<a href="https://github.com/agrovr/ParetoPilot/actions/runs/'
                f'{capacity_run_id}">Capacity run {capacity_run_id}</a> · v1.4.0 · '
                f"{result['measured_request_count']} requests · zero recorded failures"
            ),
            brief,
        )
        self.assertIn(
            f"<strong>{float(quality['retention_vs_reference']) * 100:.1f}%</strong>",
            brief,
        )
        self.assertIn(
            '<a class="flight-brief-primary" href="#optimization-ladder">'
            "Compare configurations</a>",
            brief,
        )
        self.assertIn(
            '<a class="flight-brief-secondary" href="#capacity-envelope">View capacity results</a>',
            brief,
        )
        self.assertIn(
            '<a class="flight-brief-secondary" '
            'href="https://github.com/agrovr/ParetoPilot#run-an-example">'
            "Run an example</a>",
            brief,
        )
        self.assertIn("Measured tradeoff at", brief)
        self.assertNotIn("flight-brief-boundary", brief)
        self.assertIn("Q8 reference → tuned Q4", brief)
        self.assertEqual(brief.count('class="flight-brief-metric"'), 3)
        self.assertEqual(brief.count('class="is-reference"'), 3)
        self.assertEqual(brief.count('class="is-alternative"'), 3)
        self.assertNotIn("<script", brief)

        verdict_css = css_rule_body(report, ".showcase .flight-brief-verdict")
        self.assertIn("min-width: 0;", verdict_css)
        self.assertIn("font-size: .875rem;", verdict_css)
        verdict_value_css = css_rule_body(report, ".showcase .flight-brief-verdict dd")
        self.assertIn("min-width: 0;", verdict_value_css)
        self.assertIn("overflow-wrap: anywhere;", verdict_value_css)
        mobile_css = css_rule_body(report, "@media (max-width: 47.99rem)")
        self.assertIn(
            ".showcase .flight-brief-verdict dl > div {\n"
            "    grid-template-columns: minmax(0, 1fr);",
            mobile_css,
        )

    def test_capacity_relative_claims_preserve_metric_direction(self) -> None:
        self.assertEqual(
            _relative_measure_phrase(6.75, "throughput"),
            "6.75% more throughput",
        )
        self.assertEqual(
            _relative_measure_phrase(-41.09, "peak RSS"),
            "41.09% less peak RSS",
        )
        self.assertEqual(
            _relative_measure_phrase(0.0, "throughput"),
            "no change in throughput",
        )

    def test_capacity_comparison_bars_preserve_reference_ratios(self) -> None:
        reference, alternative = _comparison_bar_widths(106.74)
        self.assertAlmostEqual(reference / alternative, 100.0 / 106.74)
        self.assertEqual(alternative, 100.0)

        reference, alternative = _comparison_bar_widths(58.91)
        self.assertEqual(reference, 100.0)
        self.assertAlmostEqual(alternative, 58.91)

        with self.assertRaisesRegex(
            ValidationError,
            "comparison percentage of reference cannot be negative",
        ):
            _comparison_bar_widths(-0.01)

    def test_capacity_envelope_fails_closed_on_unlocked_or_tampered_evidence(self) -> None:
        with TemporaryDirectory() as directory:
            study = CapacityFixture(Path(directory)).assemble()
        data = capacity_benchmarks()

        with self.assertRaisesRegex(
            ValidationError,
            "capacity_study and capacity_evidence_lock must be supplied together",
        ):
            render_showcase_v11(
                data,
                frozen_v11_recommendation(data),
                capacity_study=study,
            )

        tampered = deepcopy(study)
        cells = tampered["cells"]
        assert isinstance(cells, list)
        first_cell = cells[0]
        assert isinstance(first_cell, dict)
        summary = first_cell["summary"]
        assert isinstance(summary, dict)
        summary["generated_tokens_per_second_median"] = 999.0
        with self.assertRaises(ValidationError):
            rendered_capacity_showcase(tampered)

        mismatched_lock = capacity_evidence_lock(study)
        canonical = mismatched_lock["canonical_evidence"]
        assert isinstance(canonical, dict)
        canonical["release_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ValidationError,
            "canonical linkage does not match study",
        ):
            rendered_capacity_showcase(study, capacity_lock=mismatched_lock)

        invalid_archive = capacity_evidence_lock(study)
        archive = invalid_archive["archive"]
        assert isinstance(archive, dict)
        archive["release_asset_url"] = None
        with self.assertRaisesRegex(
            ValidationError,
            "capacity release asset URL must be a non-empty string",
        ):
            rendered_capacity_showcase(study, capacity_lock=invalid_archive)

        unrelated_archive = capacity_evidence_lock(study)
        archive = unrelated_archive["archive"]
        assert isinstance(archive, dict)
        archive["release_asset_url"] = (
            "https://example.com/releases/download/v1.4.0/paretopilot-v1.4.0-arm64-capacity.zip"
        )
        with self.assertRaisesRegex(
            ValidationError,
            "does not match its repository lock",
        ):
            rendered_capacity_showcase(study, capacity_lock=unrelated_archive)

        result_drift = capacity_evidence_lock(study)
        result = result_drift["result"]
        assert isinstance(result, dict)
        result["cell_count"] = 999
        with self.assertRaisesRegex(
            ValidationError,
            "locked result does not match",
        ):
            rendered_capacity_showcase(study, capacity_lock=result_drift)

        for href in (
            "evidence/%2e%2e/capacity-study.json",
            "evidence/capacity-study.json?download=1",
            "evidence/not-capacity-study.json",
        ):
            with (
                self.subTest(capacity_study_href=href),
                self.assertRaisesRegex(
                    ValidationError,
                    "capacity_study_href",
                ),
            ):
                rendered_capacity_showcase(
                    study,
                    capacity_study_href=href,
                )

    def test_capacity_matrix_stacks_without_forced_horizontal_overflow(self) -> None:
        with TemporaryDirectory() as directory:
            study = CapacityFixture(Path(directory)).assemble()
            report = rendered_capacity_showcase(study)

        boards_css = css_rule_body(report, ".showcase .capacity-boards")
        table_css = css_rule_body(report, ".showcase .capacity-matrix")
        cell_css = css_rule_body(report, ".showcase .capacity-cell")
        wide_css = report[
            report.index("@media (min-width: 68rem) {") : report.index(
                "@media (max-width: 47.99rem) {"
            )
        ]

        self.assertIn("display: grid;", boards_css)
        self.assertNotIn("grid-template-columns", boards_css)
        self.assertIn("width: 100%;", table_css)
        self.assertIn("table-layout: fixed;", table_css)
        self.assertNotIn("min-width", table_css)
        self.assertIn("min-width: 0;", cell_css)
        self.assertNotIn("overflow", cell_css)
        self.assertIn(
            ".showcase .capacity-boards {\n    grid-template-columns: repeat(2, minmax(0, 1fr));",
            wide_css,
        )

    def test_optimization_ladder_adapts_stage_count_and_synthetic_language(self) -> None:
        measured = attributed_benchmarks()
        synthetic = BenchmarkSet.from_mapping(
            {
                "schema_version": measured.schema_version,
                "baseline_id": measured.baseline_id,
                "synthetic": True,
                "metadata": dict(measured.metadata),
                "candidates": [
                    {
                        "id": candidate.candidate_id,
                        "label": candidate.label,
                        "parameters": dict(candidate.parameters),
                        "metrics": dict(candidate.metrics),
                    }
                    for candidate in measured.candidates
                ],
            }
        )
        passport = deepcopy(build_decision_passport(synthetic, canonical_constraints()))
        passport["ladder"] = passport["ladder"][:3]
        passport["ladder"][1]["objective_value"] = None

        markup = _optimization_ladder_markup(passport, synthetic)

        self.assertIn("00 · Synthetic example", markup)
        self.assertIn("Three configurations, compared step by step.", markup)
        self.assertIn('style="--stage-count: 3"', markup)
        self.assertIn('aria-label="Example optimization stages"', markup)
        self.assertIn("largest synthetic differences from the previous row", markup)
        self.assertIn(
            "example data and do not describe measured deployment performance",
            markup,
        )
        self.assertIn("<strong>Unavailable</strong>", markup)
        self.assertIn("Fixture improvement", markup)
        self.assertIn("Fixture tradeoff", markup)
        self.assertIn("Reference configuration", markup)
        self.assertNotIn("Measured configurations", markup)
        self.assertNotIn("Measured improvement", markup)
        self.assertNotIn("Measured tradeoff", markup)
        self.assertNotIn("Verified Arm64 measurements", markup)

        report = render_showcase_v11(
            synthetic,
            frozen_v11_recommendation(synthetic),
            canonical_report_href="evidence/report-v1.1.html",
            benchmarks_sha256="a" * 64,
            recommendation_sha256="b" * 64,
        )
        self.assertIn("<title>ParetoPilot | synthetic decision preview</title>", report)
        self.assertIn("4 synthetic fixture candidates", report)
        self.assertIn("ParetoPilot compared 4 example configurations", report)
        self.assertIn(
            "Explore a deployment decision from "
            '<span class="hero-selection">synthetic data.</span>',
            report,
        )
        self.assertIn('<h2 id="tradeoffs-heading">Measured tradeoffs</h2>', report)
        for measured_claim in (
            "measured candidates",
            "ParetoPilot measured",
            "Measured improvement",
            "Measured change",
            "Verified Arm64 measurements",
            "Measured stage",
            "one measured candidate",
            "measured objective values",
        ):
            self.assertNotIn(measured_claim, report)

    def test_optimization_ladder_reads_quantization_from_the_model_record(self) -> None:
        measured = attributed_benchmarks()
        candidates = []
        for candidate in measured.candidates:
            parameters = deepcopy(dict(candidate.parameters))
            configuration = deepcopy(dict(parameters["configuration"]))
            quantization = configuration.pop("quantization")
            parameters["configuration"] = configuration
            parameters["model"] = {"quantization": quantization}
            candidates.append(
                {
                    "id": candidate.candidate_id,
                    "label": candidate.label,
                    "parameters": parameters,
                    "metrics": dict(candidate.metrics),
                }
            )
        nested_model_benchmarks = BenchmarkSet.from_mapping(
            {
                "schema_version": measured.schema_version,
                "baseline_id": measured.baseline_id,
                "synthetic": measured.synthetic,
                "metadata": dict(measured.metadata),
                "candidates": candidates,
            }
        )
        passport = build_decision_passport(
            nested_model_benchmarks,
            canonical_constraints(),
        )

        markup = _optimization_ladder_markup(passport, nested_model_benchmarks)

        self.assertIn(
            "<span>Quantization</span> <code>Q8_0</code> "
            '<span class="technical-change-arrow" aria-label="to">→</span> '
            "<code>Q4_0</code>",
            markup,
        )

    def test_optimization_ladder_normalizes_setting_types_before_comparison(self) -> None:
        measured = attributed_benchmarks()
        candidates = []
        for candidate in measured.candidates:
            parameters = deepcopy(dict(candidate.parameters))
            if candidate.candidate_id == "q4-kleidiai":
                parameters["configuration"]["ubatch_size"] = 128
            elif candidate.candidate_id == "q4-kleidiai-tuned":
                parameters["configuration"]["ubatch_size"] = "128"
                argv = parameters["deployment_argv"]
                argv[argv.index("--ubatch-size") + 1] = "128"
            candidates.append(
                {
                    "id": candidate.candidate_id,
                    "label": candidate.label,
                    "parameters": parameters,
                    "metrics": dict(candidate.metrics),
                }
            )
        equivalent_settings = BenchmarkSet.from_mapping(
            {
                "schema_version": measured.schema_version,
                "baseline_id": measured.baseline_id,
                "synthetic": measured.synthetic,
                "metadata": dict(measured.metadata),
                "candidates": candidates,
            }
        )
        passport = build_decision_passport(
            equivalent_settings,
            canonical_constraints(),
        )

        markup = _optimization_ladder_markup(passport, equivalent_settings)

        self.assertNotIn(
            "<span>Micro-batch</span> <code>128</code> "
            '<span class="technical-change-arrow" aria-label="to">→</span> '
            "<code>128</code>",
            markup,
        )
        self.assertIn(
            "No configuration change was recorded for this stage.",
            markup,
        )

    def test_optimization_ladder_rejects_conflicting_setting_declarations(self) -> None:
        measured = attributed_benchmarks()
        candidates = []
        for candidate in measured.candidates:
            parameters = deepcopy(dict(candidate.parameters))
            if candidate.candidate_id == "q8-generic":
                parameters["model"] = {"quantization": "Q4_0"}
            candidates.append(
                {
                    "id": candidate.candidate_id,
                    "label": candidate.label,
                    "parameters": parameters,
                    "metrics": dict(candidate.metrics),
                }
            )
        conflicting_settings = BenchmarkSet.from_mapping(
            {
                "schema_version": measured.schema_version,
                "baseline_id": measured.baseline_id,
                "synthetic": measured.synthetic,
                "metadata": dict(measured.metadata),
                "candidates": candidates,
            }
        )
        passport = build_decision_passport(
            conflicting_settings,
            canonical_constraints(),
        )

        with self.assertRaisesRegex(
            ValidationError,
            "q8-generic.*conflicting Quantization",
        ):
            _optimization_ladder_markup(passport, conflicting_settings)

    def test_recognized_ladder_stage_does_not_borrow_an_unrelated_change(self) -> None:
        measured = attributed_benchmarks()
        candidates = []
        for candidate in measured.candidates:
            parameters = deepcopy(dict(candidate.parameters))
            if candidate.candidate_id == "q4-kleidiai":
                parameters["configuration"]["kleidiai"] = False
                parameters["configuration"]["ubatch_size"] = 256
                argv = parameters["deployment_argv"]
                argv[argv.index("--ubatch-size") + 1] = "256"
            candidates.append(
                {
                    "id": candidate.candidate_id,
                    "label": candidate.label,
                    "parameters": parameters,
                    "metrics": dict(candidate.metrics),
                }
            )
        unrelated_change = BenchmarkSet.from_mapping(
            {
                "schema_version": measured.schema_version,
                "baseline_id": measured.baseline_id,
                "synthetic": measured.synthetic,
                "metadata": dict(measured.metadata),
                "candidates": candidates,
            }
        )
        passport = build_decision_passport(
            unrelated_change,
            canonical_constraints(),
        )

        markup = _optimization_ladder_markup(passport, unrelated_change)
        arm_stage_start = markup.index('<code class="stage-id">q4-kleidiai</code>')
        next_stage_start = markup.index(
            '<code class="stage-id">q4-kleidiai-tuned</code>',
            arm_stage_start,
        )
        arm_stage_markup = markup[arm_stage_start:next_stage_start]

        self.assertIn(
            "No configuration change was recorded for this stage.",
            arm_stage_markup,
        )
        self.assertNotIn("Micro-batch", arm_stage_markup)

    def test_optimization_ladder_bounds_numeric_setting_text(self) -> None:
        measured = attributed_benchmarks()
        candidates = []
        for candidate in measured.candidates:
            parameters = deepcopy(dict(candidate.parameters))
            if candidate.candidate_id == "q8-generic":
                parameters["configuration"]["ubatch_size"] = "9" * 5000
            candidates.append(
                {
                    "id": candidate.candidate_id,
                    "label": candidate.label,
                    "parameters": parameters,
                    "metrics": dict(candidate.metrics),
                }
            )
        oversized_setting = BenchmarkSet.from_mapping(
            {
                "schema_version": measured.schema_version,
                "baseline_id": measured.baseline_id,
                "synthetic": measured.synthetic,
                "metadata": dict(measured.metadata),
                "candidates": candidates,
            }
        )
        passport = build_decision_passport(
            oversized_setting,
            canonical_constraints(),
        )

        with self.assertRaisesRegex(
            ValidationError,
            "q8-generic.*Micro-batch.*positive integer",
        ):
            _optimization_ladder_markup(passport, oversized_setting)

    def test_optimization_ladder_has_horizontal_mobile_and_print_compositions(self) -> None:
        report = rendered_showcase(benchmarks=attributed_benchmarks())

        self.assertIn(
            "@media (min-width: 48rem) {\n"
            "  .showcase .optimization-ladder-heading {\n"
            "    grid-template-columns: minmax(0, .8fr) minmax(0, 1.2fr);",
            report,
        )
        self.assertIn(
            "  .showcase .optimization-stages {\n"
            "    grid-template-columns: repeat(var(--stage-count, 4), minmax(0, 1fr));\n"
            "    gap: 1.25rem;\n"
            "  }",
            report,
        )
        self.assertIn(
            "@media (max-width: 47.99rem) {",
            report,
        )
        self.assertIn(
            "  .showcase .optimization-stages {\n    grid-template-columns: 1fr;\n  }",
            report,
        )
        print_css = report[report.index("@media print {") :]
        self.assertIn(
            ".showcase .report-header,\n"
            "  .showcase .optimization-ladder,\n"
            "  .showcase .trust-section,",
            print_css,
        )
        self.assertIn(
            ".showcase .optimization-stages {\n"
            "    grid-template-columns: repeat(2, minmax(0, 1fr));\n"
            "  }",
            print_css,
        )
        self.assertNotIn("gradient", report.lower())

    def test_desktop_grids_can_shrink_when_text_is_resized(self) -> None:
        report = rendered_showcase()

        self.assertIn(
            "grid-template-columns: minmax(0, 1.05fr) minmax(0, 2fr) minmax(0, .65fr);",
            css_rule_body(report, ".showcase .tolerance-row"),
        )
        self.assertIn(
            ".showcase .section-heading {\n"
            "    grid-template-columns: minmax(0, .65fr) minmax(0, 1.35fr);",
            report,
        )
        self.assertIn(
            ".showcase .why-layout {\n"
            "    grid-template-columns: minmax(0, .72fr) minmax(0, 1.28fr);",
            report,
        )
        self.assertIn(
            ".showcase .tradeoff-row {\n    grid-template-columns:\n      minmax(0, 1.1fr)",
            report,
        )
        self.assertIn("      minmax(0, 1fr);", report)
        self.assertIn(
            "grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);",
            css_rule_body(report, ".showcase .tradeoff-row"),
        )
        self.assertIn(
            "overflow-wrap: anywhere;",
            css_rule_body(report, ".showcase .tradeoff-row > *"),
        )
        self.assertIn(
            "overflow-wrap: anywhere;",
            css_rule_body(report, ".showcase .decision-rail dd"),
        )
        self.assertIn(
            "min-width: 0;",
            css_rule_body(report, ".showcase .decision-pair > div"),
        )
        self.assertIn(
            "contain: strict;",
            css_rule_body(report, ".showcase .sr-only"),
        )
        self.assertIn(
            "min-width: 0;",
            css_rule_body(report, ".showcase .why-layout > *"),
        )
        self.assertIn("html { overflow-x: clip; }", report)
        self.assertIn("overflow-x: clip;", css_rule_body(report, ".showcase"))
        self.assertIn(
            "position: absolute;",
            css_rule_body(report, ".showcase .tradeoff-board > .sr-only > span"),
        )
        self.assertIn(
            "min-width: 0;",
            css_rule_body(report, ".showcase .profile-metrics"),
        )
        self.assertIn(
            "overflow-wrap: anywhere;",
            css_rule_body(
                report,
                ".showcase .profile-metrics span,\n.showcase .profile-metrics strong",
            ),
        )
        self.assertIn(
            "min-width: 0;",
            css_rule_body(report, ".showcase .policy-cockpit"),
        )
        cockpit_tabs = css_rule_body(report, ".showcase .cockpit-tabs")
        self.assertIn("max-width: 100%;", cockpit_tabs)
        self.assertIn("overflow-x: auto;", cockpit_tabs)
        self.assertIn("scrollbar-width: none;", cockpit_tabs)

        self.assertIn(
            "min-width: 0;",
            css_rule_body(report, ".showcase .cockpit-tabs button"),
        )
        narrow_css = css_rule_body(report, "@media (max-width: 47.99rem)")
        self.assertIn(
            ".showcase .cockpit-heading,\n"
            "  .showcase .cockpit-decision {\n"
            "    grid-template-columns: minmax(0, 1fr);",
            narrow_css,
        )
        self.assertIn(
            ".showcase .cockpit-deltas { grid-template-columns: minmax(0, 1fr); }",
            narrow_css,
        )

    def test_mobile_headline_wraps_only_between_words(self) -> None:
        report = rendered_showcase()
        mobile_css = css_rule_body(report, "@media (max-width: 47.99rem)")
        heading_rule = (
            ".showcase h1 {\n"
            "    font-size: clamp(2.2rem, 11vw, 3.5rem);\n"
            "    overflow-wrap: normal;\n"
            "    word-break: normal;\n"
            "    hyphens: none;\n"
            "  }"
        )

        self.assertIn(heading_rule, mobile_css)
        self.assertNotIn(
            ".showcase h1 {\n"
            "    font-size: clamp(2.2rem, 11vw, 3.5rem);\n"
            "    overflow-wrap: anywhere;",
            mobile_css,
        )

    def test_desktop_hero_pairs_headline_with_proof_without_wrapping_full_width_rows(
        self,
    ) -> None:
        report = rendered_showcase()
        hero_start = report.index('<div class="hero-layout">')
        headline_start = report.index('<div class="hero-headline"><h1>', hero_start)
        proof_start = report.index('<div class="hero-proof"><p class="report-lede">', hero_start)
        actions_start = report.index('<nav class="hero-actions"', proof_start)
        hero_end = report.index('</div>\n<nav class="flight-log"', actions_start)
        flight_log_start = report.index('<nav class="flight-log"', hero_end)
        verdict_start = report.index('<section class="verdict-layout"', flight_log_start)

        self.assertLess(headline_start, proof_start)
        self.assertLess(proof_start, actions_start)
        self.assertLess(actions_start, hero_end)
        self.assertLess(hero_end, flight_log_start)
        self.assertLess(flight_log_start, verdict_start)
        self.assertLess(report.index('class="provenance-strip"'), hero_start)
        self.assertLess(report.index('class="brand-line"'), hero_start)

        desktop_css = css_rule_body(report, "@media (min-width: 64rem)")
        self.assertIn(
            ".showcase .hero-layout {\n"
            "    display: grid;\n"
            "    width: 100%;\n"
            "    max-width: 78rem;\n"
            "    margin-inline: auto;\n"
            "    grid-template-columns: minmax(0, 1.08fr) minmax(0, .92fr);",
            desktop_css,
        )
        self.assertIn(
            ".showcase .hero-proof .decision-rail {\n"
            "    grid-template-columns: repeat(2, minmax(0, 1fr));",
            desktop_css,
        )
        narrow_css = css_rule_body(report, "@media (max-width: 47.99rem)")
        self.assertNotIn(
            ".showcase .hero-layout {\n    display: grid;",
            narrow_css,
        )
        self.assertIn(
            ".showcase .hero-layout,\n  .showcase .hero-headline,\n  .showcase .hero-proof,",
            narrow_css,
        )

    def test_ladder_warning_accents_use_inverse_contrast(self) -> None:
        report = rendered_showcase(benchmarks=attributed_benchmarks())

        for selector in (
            ".showcase .optimization-stage.is-closest .stage-marker",
            ".showcase .is-closest .stage-decision-label",
            ".showcase .stage-changes .is-tradeoff em",
        ):
            with self.subTest(selector=selector):
                rule = css_rule_body(report, selector)
                self.assertIn("var(--flight-focus-inverse)", rule)
                self.assertNotIn("var(--flight-amber)", rule)

    def test_showcase_preserves_canonical_sections_and_evidence_tables(self) -> None:
        canonical = rendered_v11()
        showcase = rendered_showcase()
        canonical_section_ids = set(re.findall(r'<section[^>]+id="([^"]+)"', canonical))
        showcase_section_ids = set(re.findall(r'<section[^>]+id="([^"]+)"', showcase))
        canonical_measurements = set(
            re.findall(
                r">(-?\d[\d,.]*(?:%| ms| MiB| tok/s)?)<",
                canonical,
            )
        )
        canonical_captions = re.findall(r"<caption>.*?</caption>", canonical, re.DOTALL)
        canonical_tab_controls = set(re.findall(r'aria-controls="([^"]+)"', canonical))
        showcase_tab_controls = set(re.findall(r'aria-controls="([^"]+)"', showcase))
        showcase_ids = set(re.findall(r'\sid="([^"]+)"', showcase))

        self.assertGreaterEqual(len(canonical_section_ids), 3)
        self.assertLessEqual(canonical_section_ids, showcase_section_ids)
        self.assertEqual(canonical.count("<tbody>"), showcase.count("<tbody>"))
        self.assertEqual(canonical.count("<tr>"), showcase.count("<tr>"))
        self.assertEqual(canonical.count("<td"), showcase.count("<td"))
        self.assertGreaterEqual(len(canonical_measurements), 20)
        self.assertGreaterEqual(len(canonical_captions), 5)
        self.assertGreaterEqual(len(canonical_tab_controls), 3)
        for measurement in canonical_measurements:
            self.assertIn(f">{measurement}<", showcase)
        for caption in canonical_captions:
            self.assertIn(_plain_language_report_copy(caption), showcase)
        self.assertIn(
            '<main id="main-content" class="report-main" tabindex="-1">',
            showcase,
        )
        self.assertIn(
            'Compare measured Arm64 <span class="hero-selection">inference configurations.</span>',
            showcase,
        )
        self.assertIn("Use the GitHub Action", showcase)
        self.assertIn(
            "https://github.com/agrovr/ParetoPilot/blob/main/docs/github-action.md",
            showcase,
        )
        self.assertIn('aria-label="Project links"', showcase)
        self.assertIn('role="tablist"', showcase)
        self.assertLessEqual(canonical_tab_controls, showcase_tab_controls)
        self.assertLessEqual(canonical_tab_controls, showcase_ids)
        self.assertNotEqual(canonical.encode(), showcase.encode())

    def test_policy_tabs_do_not_reserve_a_desktop_scrollbar_gutter(self) -> None:
        report = rendered_showcase()
        tablist_css = css_rule_body(report, ".showcase .profile-tabs")

        self.assertIn(
            ".showcase .profile-tabs {\n"
            "  display: grid;\n"
            "  grid-auto-columns: minmax(10.5rem, 1fr);\n"
            "  grid-auto-flow: column;\n"
            "  gap: 0;\n"
            "  padding: 0;",
            report,
        )
        self.assertIn("overflow-x: clip;", tablist_css)
        self.assertIn("overflow-y: hidden;", tablist_css)
        self.assertIn("scrollbar-width: none;", tablist_css)
        self.assertIn(
            '.showcase .profile-tabs[data-overflow="scroll"] { overflow-x: auto; }',
            report,
        )
        self.assertIn(
            '<p class="policy-overflow-hint">Swipe to inspect every deployment priority →</p>',
            report,
        )
        scrollbar_css = css_rule_body(report, ".showcase .profile-tabs::-webkit-scrollbar")
        self.assertIn("width: 0;", scrollbar_css)
        self.assertIn("height: 0;", scrollbar_css)
        self.assertIn("display: none;", scrollbar_css)
        self.assertIn("border: 2px solid var(--flight-ink);", tablist_css)
        self.assertIn(
            "min-height: 4.15rem;", css_rule_body(report, ".showcase .profile-tabs button")
        )
        self.assertIn(
            "const overflows = tablist.scrollWidth > tablist.clientWidth + 1;",
            report,
        )
        self.assertIn(
            'tablist.dataset.overflow = overflows ? "scroll" : "fit";',
            report,
        )
        self.assertIn(
            "const overflowObserver = new window.ResizeObserver(syncTabOverflow);",
            report,
        )
        self.assertIn("overflowObserver.observe(tablist);", report)
        self.assertIn("for (const tab of tabs) overflowObserver.observe(tab);", report)
        self.assertIn(
            'tab.scrollIntoView({ block: "nearest", inline: "nearest" });',
            report,
        )
        narrow_css = css_rule_body(report, "@media (max-width: 47.99rem)")
        self.assertNotIn(".profile-tabs", narrow_css)
        self.assertIn(
            ".showcase .policy-overflow-hint { display: block; }",
            narrow_css,
        )
        self.assertNotIn("@media (min-width: 56rem)", report)

    def test_charts_use_stable_series_tags_and_responsive_html_legends(self) -> None:
        report = rendered_showcase()
        expected_shapes = {
            "Q8 generic reference": "circle",
            "Q4 generic": "circle",
            "Q4 + KleidiAI": "square",
            "Q4 + KleidiAI tuned": "triangle",
        }

        self.assertGreaterEqual(report.count('data-series-style="0"'), 6)
        self.assertIn("Chart legend", report)
        self.assertIn('<span class="series-name">Q4 + KleidiAI tuned</span>', report)
        self.assertIn(".showcase .chart-figure svg", report)
        self.assertIn("min-width: 0;", report)
        self.assertIn(".showcase .direct-label,", report)
        self.assertIn("display: none;", report)
        self.assertIn(".showcase [data-series-style] .chart-line", report)
        self.assertIn("fill: none;", report)
        self.assertIn('class="series-swatch"', report)
        self.assertIn('class="legend-marker"', report)
        self.assertIn('class="legend-marker legend-marker-selected"', report)
        self.assertIn("stroke-dasharray: var(--series-dash);", report)
        self.assertIn(".showcase .scatter-figure svg", report)
        self.assertIn("width: min(100%, 40rem);", report)
        self.assertNotIn('viewBox="0 0 800 310"', report)
        self.assertIn('viewBox="0 0 650 310"', report)
        self.assertNotIn(
            'role="region" aria-label="Scrollable JSON details"',
            report,
        )
        self.assertIn("max-height: none !important;", report)
        self.assertIn("table-layout: auto !important;", report)
        self.assertIn("white-space: pre-wrap !important;", report)
        self.assertIn('<p class="table-scroll-hint">Scroll the table horizontally.</p>', report)
        self.assertIn(".showcase .candidate-table {\n  width: 92rem;", report)
        self.assertIn(".showcase .candidate-table caption {", report)
        self.assertIn(".showcase .metadata-table {\n  width: 100%;\n  min-width: 48rem;", report)
        self.assertIn(".showcase .metadata-table th:first-child { width: 13rem; }", report)
        self.assertIn(".showcase th { overflow-wrap: normal; }", report)
        self.assertIn(".showcase td { overflow-wrap: anywhere; }", report)
        self.assertIn("max-height: 18rem;", report)
        self.assertIn(
            ".showcase .table-scroll,\n"
            ".showcase .command {\n"
            "  width: 100%;\n"
            "  max-width: 100%;\n"
            "  min-width: 0;\n"
            "  contain: inline-size;\n"
            "}",
            report,
        )
        self.assertIn(
            '.showcase [data-series-style="0"] .chart-marker {',
            report,
        )
        self.assertNotIn(
            '.showcase .scatter-figure [data-series-style="0"] .chart-marker {',
            report,
        )
        self.assertIn(
            '.showcase .scatter-figure .chart-tick[x="68"] {\n  text-anchor: end;\n}',
            report,
        )
        self.assertNotIn(
            '.showcase .chart-tick[x="68"][y="34"]',
            report,
        )
        for label, shape in expected_shapes.items():
            groups = re.findall(
                rf'<g role="group" aria-label="{re.escape(label)}"[^>]*'
                rf'data-marker-shape="{shape}"[^>]*>(.*?)</g>',
                report,
                re.DOTALL,
            )
            self.assertGreaterEqual(len(groups), 4, label)
            expected_tag = {
                "circle": "circle",
                "square": "rect",
                "triangle": "path",
            }[shape]
            for group in groups:
                self.assertIn(f'<{expected_tag} class="chart-marker', group)
            self.assertRegex(
                report,
                rf'<li[^>]*data-marker-shape="{shape}"[^>]*>.*?'
                rf'<span class="series-name">{re.escape(label)}</span>',
            )

    def test_load_chart_axes_and_slo_label_use_the_actual_plot_domain(self) -> None:
        load = measured_load_sweep()
        report = rendered_showcase()
        chart_metrics = (
            ("requests_per_second", "req/s"),
            ("generated_tokens_per_second", "tok/s"),
            ("e2e_latency_ms_p95", "ms"),
        )

        for metric, unit in chart_metrics:
            maximum = max(float(row[metric]) for row in load["rows"])
            ceiling = f"{maximum * 1.08:,.4f}".rstrip("0").rstrip(".")
            self.assertIn(f">{ceiling} {unit}</text>", report)
        self.assertIn('class="slo-reference-label"', report)
        self.assertIn('text-anchor="start">SLO', report)

    def test_load_section_puts_measured_results_before_exact_contract(self) -> None:
        report = rendered_showcase()
        load_start = report.index('<section class="report-section" aria-labelledby="load-heading">')
        load_end = report.index('<section class="report-section"', load_start + 1)
        load_section = report[load_start:load_end]

        self.assertIn(
            '<div class="load-takeaway" role="note" aria-label="Measured load takeaway">',
            load_section,
        )
        self.assertIn(
            "Every candidate’s highest measured all-gates passing concurrency was C4.",
            load_section,
        )
        self.assertIn(
            "At C4, every candidate remained within the declared 3,000 ms p95 end-to-end ceiling.",
            load_section,
        )
        self.assertIn(
            '<details class="load-contract-disclosure">'
            "<summary>View load-test settings and prompts</summary>"
            '<div class="load-context-grid">',
            load_section,
        )
        self.assertLess(
            load_section.index('class="series-key-wrap"'),
            load_section.index('<div class="chart-grid-layout">'),
        )
        self.assertLess(
            load_section.index('<div class="chart-grid-layout">'),
            load_section.index('<details class="load-contract-disclosure">'),
        )
        self.assertLess(
            load_section.index('<details class="load-contract-disclosure">'),
            load_section.index("View all measured load results"),
        )
        self.assertEqual(load_section.count("Verified test commands"), 1)
        self.assertEqual(load_section.count("Load sweep methodology"), 1)

    def test_out_of_range_slo_is_annotated_without_a_false_reference_line(self) -> None:
        load = deepcopy(measured_load_sweep())
        maximum = max(float(row["e2e_latency_ms_p95"]) for row in load["rows"])
        threshold = maximum * 2
        load["slo"]["max_e2e_latency_ms_p95"] = threshold

        report = rendered_showcase(lock=False, load_sweep=load)

        self.assertIn(
            f"SLO · {threshold:,.0f} ms · above plotted range",
            report,
        )
        self.assertNotIn('<line class="slo-reference-line"', report)
        self.assertIn("latency ceiling is above the plotted measured range", report)

    def test_no_javascript_fallback_css_is_in_the_document_head(self) -> None:
        report = rendered_showcase()
        head, body = report.split("</head>", 1)

        self.assertIn("<noscript>", head)
        self.assertIn(".showcase .theme-toggle { display: none; }", head)
        self.assertIn(".showcase .profile-tabs { display: none; }", head)
        self.assertIn(".showcase .profile-panel[hidden]", head)
        self.assertNotRegex(body, r"<noscript>\s*<style>")
        self.assertIn("JavaScript is unavailable", body)

    def test_theme_toggle_is_accessible_persistent_and_applied_before_styles(self) -> None:
        report = rendered_showcase()
        head, body = report.split("</head>", 1)

        self.assertEqual(report.count("data-theme-toggle"), 2)
        self.assertEqual(
            report.count(
                '<button type="button" class="theme-toggle" data-theme-toggle '
                'aria-pressed="false" hidden>'
            ),
            1,
        )
        self.assertIn('<span class="theme-toggle-label">Dark mode</span>', report)
        self.assertIn("data-theme-state", report)
        self.assertIn('<meta name="color-scheme" content="light dark">', head)
        self.assertLess(head.index("paretopilot.theme.v1"), head.index("<style>"))
        self.assertIn('savedTheme === "light" || savedTheme === "dark"', head)
        self.assertIn('window.matchMedia("(prefers-color-scheme: dark)")', head)
        self.assertIn("window.localStorage.getItem(storageKey)", head)
        self.assertIn("window.localStorage.setItem(storageKey, resolvedTheme)", body)
        self.assertIn("try {", head)
        self.assertIn("try {", body)
        self.assertIn("root.dataset.theme = resolvedTheme;", body)
        self.assertNotIn(".style.colorScheme", report)
        self.assertIn(
            'themeToggle.setAttribute("aria-pressed", String(isDark))',
            body,
        )
        self.assertIn(
            'themeColor.content = isDark ? "#0b1220" : "#13233d"',
            body,
        )
        self.assertIn('html[data-theme="dark"] .showcase {', report)
        self.assertIn(".showcase .theme-toggle { display: none; }", report)
        state_sync = body.index("applyTheme(root.dataset.theme, false);")
        reveal = body.index("themeToggle.hidden = false;")
        listener = body.index('themeToggle.addEventListener("click"')
        self.assertLess(state_sync, reveal)
        self.assertLess(reveal, listener)

        print_theme_selector = ".showcase,\n  html[data-theme] .showcase"
        print_rule_start = report.index(f"{print_theme_selector} {{")
        print_media_start = report.rfind("@media print {", 0, print_rule_start)
        self.assertNotEqual(print_media_start, -1)
        print_css = css_rule_body(report[print_media_start:], "@media print")
        self.assertIn(
            f"{print_theme_selector} {{",
            print_css,
        )
        print_tokens = css_hex_tokens(print_css, print_theme_selector)
        light_tokens = css_hex_tokens(report, ".showcase")
        for token in (
            "--flight-ink",
            "--flight-cobalt",
            "--flight-teal",
            "--flight-amber",
            "--flight-danger",
            "--flight-text-muted",
            "--flight-control-border",
            "--flight-chart-axis",
            "--flight-purple",
            "--flight-cyan",
        ):
            with self.subTest(print_token=token):
                self.assertEqual(print_tokens[token], light_tokens[token])
        self.assertEqual(print_tokens["--flight-command-text"], light_tokens["--flight-ink"])
        for token in (
            "--flight-canvas",
            "--flight-paper",
            "--flight-paper-blue",
            "--flight-command-bg",
        ):
            with self.subTest(print_surface=token):
                self.assertEqual(print_tokens[token], "#ffffff")
        self.assertIn(".showcase .theme-toggle { display: none; }", print_css)
        self.assertIn(".showcase .metadata-table code,", print_css)
        self.assertIn(".showcase .metadata-table,", print_css)

    def test_light_and_dark_theme_tokens_meet_contrast_requirements(self) -> None:
        report = rendered_showcase()
        light_tokens = css_hex_tokens(report, ".showcase")
        dark_tokens = dict(light_tokens)
        dark_tokens.update(css_hex_tokens(report, 'html[data-theme="dark"] .showcase'))
        text_pairs = (
            ("body", "--flight-ink", "--flight-canvas"),
            ("muted", "--flight-text-muted", "--flight-canvas"),
            ("subtle surface text", "--flight-text-subtle", "--flight-paper-blue"),
            ("link", "--flight-cobalt", "--flight-canvas"),
            ("inverse", "--flight-on-dark", "--flight-panel"),
            ("solid cobalt", "--flight-white", "--flight-cobalt-solid"),
            ("ladder warning", "--flight-focus-inverse", "--flight-panel"),
            ("table header", "--flight-white", "--flight-panel"),
            ("striped table row", "--flight-ink", "--flight-paper-blue"),
            ("trust table", "--flight-on-dark", "--flight-panel"),
            ("trust table caption", "--flight-on-dark-muted", "--flight-panel"),
            ("code", "--flight-command-text", "--flight-command-bg"),
            ("success", "--flight-teal", "--flight-teal-soft"),
            ("warning", "--flight-amber", "--flight-amber-soft"),
            ("danger", "--flight-danger", "--flight-danger-soft"),
            ("capacity pass label", "--flight-text-muted", "--flight-paper-blue"),
            ("capacity blocked label", "--flight-text-muted", "--flight-danger-soft"),
            ("capacity q8 selected label", "--flight-text-muted", "--flight-cobalt-soft"),
            ("capacity q4 selected label", "--flight-text-muted", "--flight-teal-soft"),
        )
        non_text_pairs = (
            ("control border", "--flight-control-border", "--flight-panel"),
            ("focus", "--flight-focus", "--flight-canvas"),
            ("inverse focus", "--flight-focus-inverse", "--flight-panel"),
            ("chart axis", "--flight-chart-axis", "--flight-paper"),
            ("cobalt chart", "--flight-cobalt", "--flight-paper"),
            ("slate chart", "--flight-slate", "--flight-paper"),
            ("teal chart", "--flight-teal", "--flight-paper"),
            ("amber chart", "--flight-amber", "--flight-paper"),
            ("purple chart", "--flight-purple", "--flight-paper"),
            ("cyan chart", "--flight-cyan", "--flight-paper"),
        )

        for theme, tokens in (("light", light_tokens), ("dark", dark_tokens)):
            for role, foreground_token, background_token in text_pairs:
                with self.subTest(theme=theme, role=role):
                    self.assertGreaterEqual(
                        contrast_ratio(
                            tokens[foreground_token],
                            tokens[background_token],
                        ),
                        4.5,
                    )
            for role, foreground_token, background_token in non_text_pairs:
                with self.subTest(theme=theme, role=role):
                    self.assertGreaterEqual(
                        contrast_ratio(
                            tokens[foreground_token],
                            tokens[background_token],
                        ),
                        3.0,
                    )

    def test_fixed_dark_trust_table_uses_inverse_text_and_stable_columns(self) -> None:
        report = rendered_showcase()

        self.assertIn(
            ".showcase .trust-section table { color: var(--flight-on-dark); }",
            report,
        )
        self.assertIn(
            ".showcase .trust-section caption { color: var(--flight-on-dark-muted); }",
            report,
        )
        self.assertIn(
            ".showcase .skip-link {\n"
            "  background: var(--flight-panel);\n"
            "  color: var(--flight-on-dark);\n"
            "}",
            report,
        )
        self.assertIn(
            ".showcase .trust-section tbody tr:nth-child(even) {\n"
            "  background: var(--flight-ink-soft);\n"
            "}",
            report,
        )
        self.assertNotIn(
            ".showcase .trust-section table { color: var(--flight-ink); }",
            report,
        )

    def test_tolerance_track_keeps_full_names_values_and_roles_in_text(self) -> None:
        report = rendered_showcase()

        self.assertIn('<figure class="tolerance-visual"', report)
        self.assertIn("Q8 generic reference", report)
        self.assertIn("Q4 + KleidiAI tuned", report)
        self.assertIn("Within tolerance", report)
        self.assertIn("Outside tolerance", report)
        self.assertIn("2,330.9 ms", report)
        self.assertIn("2,354.2 ms", report)
        self.assertIn(
            "Markers show each measured p95 end-to-end latency on the same scale.",
            report,
        )
        self.assertIn("Exact values are in the table below.", report)

    def test_showcase_rejects_a_different_canonical_report(self) -> None:
        with self.assertRaisesRegex(ValidationError, "does not match"):
            rendered_showcase(canonical_html="<!doctype html><title>different</title>")

    def test_lock_and_canonical_report_are_paired_and_hash_bound(self) -> None:
        benchmarks = canonical_benchmarks()
        recommendation = frozen_v11_recommendation(benchmarks)
        canonical = rendered_v11(data=benchmarks)
        kwargs = {
            "policy_profiles": frozen_v11_profiles(benchmarks),
            "load_sweep": measured_load_sweep(),
            "stability_summary": measured_stability(benchmarks),
            "benchmarks_sha256": "a" * 64,
            "recommendation_sha256": "b" * 64,
            "profiles_sha256": "c" * 64,
            "load_sha256": "d" * 64,
            "stability_sha256": "e" * 64,
        }

        with self.assertRaisesRegex(ValidationError, "must be supplied together"):
            render_showcase_v11(
                benchmarks,
                recommendation,
                evidence_lock=evidence_lock(),
                **kwargs,
            )
        with self.assertRaisesRegex(ValidationError, "must be supplied together"):
            render_showcase_v11(
                benchmarks,
                recommendation,
                canonical_html=canonical,
                **kwargs,
            )

        tampered_lock = deepcopy(evidence_lock())
        tampered_lock["review"]["artifacts_sha256"]["benchmark_set"] = "0" * 64
        with self.assertRaisesRegex(ValidationError, "artifact digest does not match"):
            render_showcase_v11(
                benchmarks,
                recommendation,
                evidence_lock=tampered_lock,
                canonical_html=canonical,
                **kwargs,
            )

        preview = render_showcase_v11(benchmarks, recommendation, **kwargs)
        self.assertIn("Unverified preview", preview)
        self.assertIn("Unverified preview · v1.1 report layout", preview)
        self.assertIn("Source run", preview)
        self.assertIn("This preview is not connected to a verified release.", preview)
        self.assertNotIn("View archived v1.1 report", preview)

    def test_showcase_rejects_unsafe_canonical_report_links(self) -> None:
        benchmarks = canonical_benchmarks()
        recommendation = canonical_recommendation(benchmarks)

        for href in (
            "javascript:alert(1)",
            "http://example.com/report.html",
            "https://example.com/evidence/report-v1.1.html",
            "https://github.com/agrovr/ParetoPilot/blob/main/report-v1.1.html",
            "//example.com/report.html",
            "/absolute/report.html",
            "../outside/report.html",
            r"evidence\report.html",
            "evidence/%2e%2e/report-v1.1.html",
            "evidence/report-v1.1.html?download=1",
            "evidence/not-the-canonical-report.html",
        ):
            with (
                self.subTest(href=href),
                self.assertRaisesRegex(
                    ValidationError,
                    "canonical_report_href",
                ),
            ):
                render_showcase_v11(
                    benchmarks,
                    recommendation,
                    canonical_report_href=href,
                )

    def test_showcase_rejects_unverified_or_cross_run_evidence_locks(self) -> None:
        benchmarks = canonical_benchmarks()
        recommendation = frozen_v11_recommendation(benchmarks)
        canonical = rendered_v11(
            data=benchmarks,
            profiles=False,
            load=None,
            stability=None,
        )
        kwargs = {
            "benchmarks_sha256": "a" * 64,
            "recommendation_sha256": "b" * 64,
        }

        unverified = evidence_lock()
        unverified["review"]["all_checksums_verified"] = False
        with self.assertRaisesRegex(ValidationError, "all_checksums_verified"):
            render_showcase_v11(
                benchmarks,
                recommendation,
                evidence_lock=unverified,
                canonical_html=canonical,
                **kwargs,
            )

        cross_run = deepcopy(evidence_lock())
        cross_run["source"]["run_id"] = "another-run"
        with self.assertRaisesRegex(ValidationError, "run id"):
            render_showcase_v11(
                benchmarks,
                recommendation,
                evidence_lock=cross_run,
                canonical_html=canonical,
                **kwargs,
            )

        wrong_release = deepcopy(evidence_lock())
        wrong_release["archive"]["release_url"] = (
            "https://github.com/agrovr/ParetoPilot/releases/tag/v9.9.9"
        )
        wrong_release["review"]["artifacts_sha256"]["report_v1_1"] = hashlib.sha256(
            canonical.encode()
        ).hexdigest()
        with self.assertRaisesRegex(ValidationError, "release tag"):
            render_showcase_v11(
                benchmarks,
                recommendation,
                evidence_lock=wrong_release,
                canonical_html=canonical,
                **kwargs,
            )


if __name__ == "__main__":
    unittest.main()
