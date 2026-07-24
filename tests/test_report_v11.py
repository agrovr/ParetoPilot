from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from paretopilot.analysis import recommend
from paretopilot.domain import BenchmarkSet, Constraints, ValidationError
from paretopilot.load_eval import (
    LoadRequest,
    build_load_evidence_binding,
    combine_load_evaluations,
    evaluate_load,
)
from paretopilot.report import render_report
from paretopilot.report_v11 import render_report_v11
from paretopilot.stability import summarize_stability


def candidate_deployment_argv(candidate_id: str) -> list[str]:
    kleidiai = candidate_id.startswith("q4-kleidiai")
    model = "q8.gguf" if candidate_id == "q8-generic" else "q4.gguf"
    ubatch = "512" if candidate_id == "q4-kleidiai-tuned" else "128"
    return [
        f"./build/{'kleidiai' if kleidiai else 'generic'}/llama-server",
        "--model",
        f"./models/{model}",
        "--threads",
        "4",
        "--ubatch-size",
        ubatch,
        "--parallel",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ]


def canonical_benchmarks(
    *,
    metadata: bool = True,
    classification: str = "canonical",
) -> BenchmarkSet:
    candidate_evidence = {
        candidate_id: {
            "artifacts": {
                "throughput_pass_1": {"path": "one", "sha256": "1" * 64},
                "throughput_pass_2": {"path": "two", "sha256": "2" * 64},
                "server_evaluation_pass_1": {"path": "three", "sha256": "3" * 64},
                "server_evaluation_pass_2": {"path": "four", "sha256": "4" * 64},
                "server_time_pass_1": {"path": "five", "sha256": "5" * 64},
                "server_time_pass_2": {"path": "six", "sha256": "6" * 64},
            }
        }
        for candidate_id in (
            "q8-generic",
            "q4-generic",
            "q4-kleidiai",
            "q4-kleidiai-tuned",
        )
    }
    return BenchmarkSet.from_mapping(
        {
            "schema_version": "1.0",
            "baseline_id": "q8-generic",
            "synthetic": False,
            "metadata": (
                {
                    "classification": classification,
                    "runner": "Ubuntu 24.04 Arm64 · Neoverse-N2",
                    "candidate_evidence": candidate_evidence,
                }
                if metadata
                else {}
            ),
            "candidates": [
                {
                    "id": "q8-generic",
                    "label": "Q8 generic reference",
                    "parameters": {
                        "configuration": {"quantization": "Q8_0", "kleidiai": False},
                        "deployment_argv": candidate_deployment_argv("q8-generic"),
                    },
                    "metrics": {
                        "e2e_latency_ms_p95": 2335.917,
                        "ttft_ms_p95": 553.415,
                        "prompt_tps": 101.7475,
                        "generation_tps": 36.5399,
                        "peak_rss_mib": 3437.613,
                        "model_size_mib": 1806.767,
                        "quality_score": 1.0,
                    },
                },
                {
                    "id": "q4-generic",
                    "label": "Q4 generic",
                    "parameters": {
                        "configuration": {"quantization": "Q4_0", "kleidiai": False},
                        "deployment_argv": candidate_deployment_argv("q4-generic"),
                    },
                    "metrics": {
                        "e2e_latency_ms_p95": 2330.914,
                        "ttft_ms_p95": 483.327,
                        "prompt_tps": 113.546,
                        "generation_tps": 34.713,
                        "peak_rss_mib": 1966.461,
                        "model_size_mib": 1016.834,
                        "quality_score": 1.0,
                    },
                },
                {
                    "id": "q4-kleidiai",
                    "label": "Q4 + KleidiAI",
                    "parameters": {
                        "configuration": {"quantization": "Q4_0", "kleidiai": True},
                        "deployment_argv": candidate_deployment_argv("q4-kleidiai"),
                    },
                    "metrics": {
                        "e2e_latency_ms_p95": 2393.537,
                        "ttft_ms_p95": 472.764,
                        "prompt_tps": 114.102,
                        "generation_tps": 34.568,
                        "peak_rss_mib": 1966.477,
                        "model_size_mib": 1016.834,
                        "quality_score": 1.0,
                    },
                },
                {
                    "id": "q4-kleidiai-tuned",
                    "label": "Q4 + KleidiAI tuned",
                    "parameters": {
                        "configuration": {
                            "quantization": "Q4_0",
                            "kleidiai": True,
                            "ubatch_size": 512,
                        },
                        "deployment_argv": candidate_deployment_argv("q4-kleidiai-tuned"),
                    },
                    "metrics": {
                        "e2e_latency_ms_p95": 2337.799,
                        "ttft_ms_p95": 472.799,
                        "prompt_tps": 131.257,
                        "generation_tps": 34.633,
                        "peak_rss_mib": 1966.477,
                        "model_size_mib": 1016.834,
                        "quality_score": 1.0,
                    },
                },
            ],
        }
    )


def canonical_constraints() -> Constraints:
    return Constraints.from_mapping(
        {
            "min_quality_retention": 1.0,
            "quality_metric": "quality_score",
            "max_values": {
                "e2e_latency_ms_p95": 15000,
                "peak_rss_mib": 4096,
            },
            "min_values": {},
            "objective": {"metric": "e2e_latency_ms_p95", "direction": "min"},
            "objective_tolerance_percent": 1.0,
            "preference_order": [
                "q8-generic",
                "q4-generic",
                "q4-kleidiai",
                "q4-kleidiai-tuned",
            ],
            "frontier_metrics": {
                "e2e_latency_ms_p95": "min",
                "generation_tps": "max",
                "model_size_mib": "min",
                "peak_rss_mib": "min",
                "quality_score": "max",
                "ttft_ms_p95": "min",
            },
        }
    )


def canonical_recommendation(data: BenchmarkSet) -> dict[str, object]:
    recommendation = dict(recommend(data, canonical_constraints()))
    recommendation["input_fingerprints"] = {
        "benchmarks_sha256": "a" * 64,
        "constraints_sha256": "f" * 64,
    }
    return recommendation


def derived_profiles(data: BenchmarkSet) -> dict[str, object]:
    memory_policy = Constraints.from_mapping(
        {
            "min_quality_retention": 1.0,
            "quality_metric": "quality_score",
            "max_values": {},
            "min_values": {},
            "objective": {"metric": "peak_rss_mib", "direction": "min"},
            "objective_tolerance_percent": 0.01,
            "preference_order": [
                "q4-kleidiai-tuned",
                "q4-generic",
                "q4-kleidiai",
                "q8-generic",
            ],
            "frontier_metrics": {
                "e2e_latency_ms_p95": "min",
                "generation_tps": "max",
                "model_size_mib": "min",
                "peak_rss_mib": "min",
                "quality_score": "max",
                "ttft_ms_p95": "min",
            },
        }
    )
    prompt_policy = Constraints.from_mapping(
        {
            "min_quality_retention": 1.0,
            "quality_metric": "quality_score",
            "max_values": {},
            "min_values": {},
            "objective": {"metric": "prompt_tps", "direction": "max"},
            "frontier_metrics": {
                "e2e_latency_ms_p95": "min",
                "generation_tps": "max",
                "model_size_mib": "min",
                "peak_rss_mib": "min",
                "prompt_tps": "max",
                "quality_score": "max",
                "ttft_ms_p95": "min",
            },
        }
    )
    canonical = canonical_recommendation(data)
    memory = recommend(data, memory_policy)
    prompt = recommend(data, prompt_policy)
    canonical_evidence = not data.synthetic and data.metadata.get("classification") == "canonical"
    primary_notice = (
        "This is the canonical predeclared decision."
        if canonical_evidence
        else "This is the primary predeclared policy; the evidence is not canonical."
    )
    derived_notice = (
        "Derived scenario only; it does not replace the canonical decision."
        if canonical_evidence
        else "Derived scenario only; it does not replace the primary predeclared policy."
    )
    return {
        "schema_version": "1.0",
        "source_schema_version": data.schema_version,
        "synthetic_source": data.synthetic,
        "canonical_profile_id": "canonical-balanced",
        "canonical_selected_id": canonical["selected_id"],
        "input_fingerprints": {
            "benchmarks_sha256": "a" * 64,
            "constraints_sha256": "f" * 64,
            "policies_sha256": "9" * 64,
        },
        "profiles": [
            {
                "id": "canonical-balanced",
                "label": "Canonical policy",
                "description": "The predeclared latency objective and simpler-first preference.",
                "classification": "canonical",
                "scenario_notice": primary_notice,
                "recommendation": canonical,
            },
            {
                "id": "memory-first",
                "label": "Memory first",
                "description": "Prefer the smallest measured peak resident memory.",
                "classification": "derived-non-canonical",
                "scenario_notice": derived_notice,
                "recommendation": memory,
            },
            {
                "id": "prompt-throughput",
                "label": "Prompt throughput",
                "description": "Prefer the greatest measured prompt processing throughput.",
                "classification": "derived-non-canonical",
                "scenario_notice": derived_notice,
                "recommendation": prompt,
            },
        ],
    }


def measured_load_sweep() -> dict[str, object]:
    def runner(request: LoadRequest) -> dict[str, object]:
        duration_seconds = 2.15 + request.concurrency * 0.19
        wave = (request.request_index - 1) // request.concurrency
        started = request.concurrency * 100.0 + wave * 2.5
        return {
            "completed": True,
            "ttft_ms": 430.0 + request.concurrency * 45.0,
            "e2e_latency_ms": duration_seconds * 1000.0,
            "generated_tokens": request.output_tokens,
            "error": None,
            "started_at_seconds": started,
            "finished_at_seconds": started + duration_seconds,
        }

    prompts = ("Summarize this note.", "Return a JSON object.")
    slo = {
        "min_completion_rate": 1.0,
        "max_ttft_ms_p95": 900.0,
        "max_e2e_latency_ms_p95": 3000.0,
    }
    evaluations = []
    with TemporaryDirectory() as directory:
        root = Path(directory)
        plan_path = root / "load-plan.json"
        plan_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "prompts": list(prompts),
                    "output_tokens": 32,
                    "warmup_requests_per_level": 1,
                    "measured_requests_per_level": 4,
                    "concurrency_levels": [1, 2, 4],
                    "timeout_seconds": 180.0,
                    "slo": slo,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        for index, (candidate_id, rss) in enumerate(
            (
                ("q8-generic", 3437.6),
                ("q4-generic", 1966.4),
                ("q4-kleidiai", 1966.5),
                ("q4-kleidiai-tuned", 1966.5),
            ),
            start=1,
        ):
            canonical_command = root / f"{candidate_id}-canonical.json"
            load_command = root / f"{candidate_id}-load.json"
            base_argv = candidate_deployment_argv(candidate_id)
            canonical_command.write_text(
                json.dumps({"schema_version": "1.0", "argv": base_argv}) + "\n",
                encoding="utf-8",
            )
            load_argv = list(base_argv)
            load_argv[-1] = str(18080 + index)
            load_command.write_text(
                json.dumps({"schema_version": "1.0", "argv": load_argv}) + "\n",
                encoding="utf-8",
            )
            evaluations.append(
                evaluate_load(
                    candidate_id=candidate_id,
                    prompts=prompts,
                    output_tokens=32,
                    warmup_requests_per_level=1,
                    measured_requests_per_level=4,
                    request_runner=runner,
                    slo=slo,
                    peak_rss_mib_by_concurrency={1: rss, 2: rss, 4: rss},
                    synthetic=False,
                    evidence_binding=build_load_evidence_binding(
                        base_url=f"http://127.0.0.1:{18080 + index}",
                        plan_path=plan_path,
                        server_command_path=load_command,
                        canonical_server_command_path=canonical_command,
                    ),
                )
            )
    return dict(
        combine_load_evaluations(
            evaluations,
            require_evidence_bindings=True,
        )
    )


def measured_stability(
    data: BenchmarkSet,
    *,
    baseline_id: str | None = None,
) -> dict[str, object]:
    passes = []
    for multiplier in (1.0, 1.01):
        passes.append(
            BenchmarkSet.from_mapping(
                {
                    "schema_version": data.schema_version,
                    "baseline_id": baseline_id or data.baseline_id,
                    "synthetic": data.synthetic,
                    "metadata": {},
                    "candidates": [
                        {
                            "id": candidate.candidate_id,
                            "label": candidate.label,
                            "parameters": dict(candidate.parameters),
                            "metrics": {
                                **dict(candidate.metrics),
                                "e2e_latency_ms_p95": (
                                    candidate.metrics["e2e_latency_ms_p95"] * multiplier
                                ),
                                "generation_tps": (
                                    candidate.metrics["generation_tps"] * multiplier
                                ),
                            },
                        }
                        for candidate in data.candidates
                    ],
                }
            )
        )
    return dict(
        summarize_stability(
            passes,
            metric_directions={
                "e2e_latency_ms_p95": "min",
                "generation_tps": "max",
            },
            pass_labels=("measured-A", "measured-B"),
            input_fingerprints={
                "measured-A": "a" * 64,
                "measured-B": "b" * 64,
            },
        )
    )


def rendered_v11(
    *,
    data: BenchmarkSet | None = None,
    profiles: bool = True,
    load: dict[str, object] | None | bool = True,
    stability: dict[str, object] | None | bool = True,
) -> str:
    benchmarks = data or canonical_benchmarks()
    recommendation = canonical_recommendation(benchmarks)
    load_sweep = measured_load_sweep() if load is True else load
    return render_report_v11(
        benchmarks,
        recommendation,
        policy_profiles=derived_profiles(benchmarks) if profiles else None,
        load_sweep=load_sweep,
        stability_summary=(measured_stability(benchmarks) if stability is True else stability),
        benchmarks_sha256="a" * 64,
        recommendation_sha256="b" * 64,
        profiles_sha256="c" * 64 if profiles else "",
        load_sha256="d" * 64 if load is not None else "",
        stability_sha256="e" * 64 if stability is not None else "",
    )


class ReportV11Tests(unittest.TestCase):
    def test_archived_generator_version_is_displayed_but_not_reselected(self) -> None:
        data = canonical_benchmarks()
        recommendation = canonical_recommendation(data)
        recommendation["paretopilot_version"] = "0.9.0"

        rendered = render_report_v11(data, recommendation)

        self.assertIn("0.9.0", rendered)
        invalid = deepcopy(recommendation)
        invalid["paretopilot_version"] = " "
        with self.assertRaisesRegex(ValidationError, "paretopilot_version"):
            render_report_v11(data, invalid)

    def test_output_is_deterministic_and_v1_renderer_has_no_side_effects(self) -> None:
        data = canonical_benchmarks()
        constraints = canonical_constraints()
        recommendation = canonical_recommendation(data)
        before = render_report(
            data,
            constraints,
            recommendation,
            benchmarks_sha256="a" * 64,
            constraints_sha256="c" * 64,
        )

        first = rendered_v11(data=data)
        second = rendered_v11(data=data)
        after = render_report(
            data,
            constraints,
            recommendation,
            benchmarks_sha256="a" * 64,
            constraints_sha256="c" * 64,
        )

        self.assertEqual(first.encode(), second.encode())
        self.assertEqual(before.encode(), after.encode())
        self.assertNotIn("generated_at", first)

    def test_opening_verdict_keeps_canonical_outcome_and_resource_alternative(self) -> None:
        report = rendered_v11()

        self.assertIn("Canonical recommendation", report)
        self.assertIn("Q8 generic reference", report)
        self.assertIn("Measured resource alternative · not selected", report)
        self.assertIn("Q4 + KleidiAI tuned", report)
        self.assertIn("43.7% lower", report)
        self.assertNotIn(">No change<", report)
        self.assertIn("descriptive comparison, not a second recommendation", report)

    def test_exploratory_report_uses_primary_policy_labels(self) -> None:
        report = rendered_v11(data=canonical_benchmarks(classification="exploratory"))

        self.assertIn(
            "Exploratory evidence — not canonical submission evidence.",
            report,
        )
        self.assertIn("Primary predeclared policy", report)
        self.assertIn("does not replace the primary predeclared policy", report)
        self.assertNotIn(">Canonical recommendation<", report)
        self.assertNotIn("Canonical submission decision.", report)

    def test_sections_follow_the_required_evidence_narrative(self) -> None:
        report = rendered_v11()
        headings = (
            "Why this decision held",
            "Aligned tradeoffs",
            "Deployment policy selector",
            "Measured load behavior",
            "Repeat stability",
            "p95 end-to-end latency vs generation throughput",
            "Full evidence table",
            "Trust and reproduction",
        )
        positions = [report.index(heading) for heading in headings]

        self.assertEqual(positions, sorted(positions))

    def test_source_profile_load_and_metadata_text_are_escaped(self) -> None:
        data = canonical_benchmarks(metadata=False)
        raw = {
            "schema_version": "1.0",
            "baseline_id": data.baseline_id,
            "synthetic": data.synthetic,
            "metadata": {"<img>": "<script>alert('metadata')</script>"},
            "candidates": [
                {
                    "id": candidate.candidate_id,
                    "label": (
                        "<img src=x onerror=alert('candidate')>"
                        if candidate.candidate_id == "q8-generic"
                        else candidate.label
                    ),
                    "parameters": dict(candidate.parameters),
                    "metrics": dict(candidate.metrics),
                }
                for candidate in data.candidates
            ],
        }
        unsafe_data = BenchmarkSet.from_mapping(raw)
        recommendation = canonical_recommendation(unsafe_data)
        unsafe_profile = recommend(unsafe_data, canonical_constraints())
        profiles = {
            "unsafe": {
                "label": "<svg onload=alert('profile')>",
                "description": "<script>alert('description')</script>",
                "decision": unsafe_profile,
            }
        }
        load = measured_load_sweep()
        methodology = deepcopy(load["methodology"])
        assert isinstance(methodology, dict)
        prompts = deepcopy(methodology["prompts"])
        assert isinstance(prompts, list)
        malicious_prompt = "<img src=x onerror=alert('load')>"
        malicious_digest = hashlib.sha256(malicious_prompt.encode()).hexdigest()
        prompts[0]["text"] = malicious_prompt
        prompts[0]["sha256"] = malicious_digest
        methodology["prompts"] = prompts
        load["methodology"] = methodology
        load_rows = deepcopy(load["rows"])
        assert isinstance(load_rows, list)
        for row in load_rows:
            for sample in row["samples"]:
                if sample["prompt_index"] == 1:
                    sample["prompt_sha256"] = malicious_digest
        load["rows"] = load_rows

        report = render_report_v11(
            unsafe_data,
            recommendation,
            policy_profiles=profiles,
            load_sweep=load,
        )

        self.assertNotIn("<script>alert", report)
        self.assertNotIn("<img src=x", report)
        self.assertNotIn("<svg onload", report)
        self.assertIn("&lt;script&gt;alert(&#x27;metadata&#x27;)&lt;/script&gt;", report)
        self.assertIn("&lt;svg onload=alert(&#x27;profile&#x27;)&gt;", report)
        self.assertIn("&lt;img src=x onerror=alert(&#x27;load&#x27;)&gt;", report)

    def test_accessible_landmarks_direct_labels_and_data_tables(self) -> None:
        report = rendered_v11()

        self.assertIn('<html lang="en">', report)
        self.assertIn('<a class="skip-link" href="#main-content">', report)
        self.assertIn('<main id="main-content" class="report-main">', report)
        self.assertIn('<footer class="report-footer">', report)
        self.assertIn('role="tablist" aria-label="Deployment policies"', report)
        self.assertGreaterEqual(report.count('role="tab"'), 3)
        self.assertGreaterEqual(report.count('role="tabpanel"'), 3)
        self.assertIn('aria-labelledby="scatter-title scatter-description"', report)
        self.assertIn('aria-labelledby="load-request-throughput-title ', report)
        self.assertIn('class="direct-label"', report)
        self.assertGreaterEqual(report.count("<caption>"), 10)
        self.assertIn(":focus-visible", report)
        self.assertIn('role="region"', report)
        self.assertIn("Selected · Baseline", report)
        self.assertIn("Better ·", report)
        self.assertIn("Tradeoff ·", report)

    def test_policy_interaction_only_switches_precomputed_panels(self) -> None:
        report = rendered_v11()
        script_match = re.search(r"<script>(.*?)</script>", report, re.DOTALL)
        self.assertIsNotNone(script_match)
        assert script_match is not None
        script = script_match.group(1)

        self.assertIn('document.querySelectorAll("[data-profile-target]")', script)
        self.assertIn("panel.hidden =", script)
        self.assertIn('item.setAttribute("aria-selected"', script)
        self.assertNotIn("Math.", script)
        self.assertNotIn(".sort(", script)
        self.assertNotIn(".reduce(", script)
        self.assertNotIn("candidate.metrics", script)
        self.assertNotIn("selected_id", script)
        self.assertIn("Derived policy scenario", report)
        self.assertIn("does not replace the canonical decision", report)
        self.assertIn("Q4 + KleidiAI tuned", report)

    def test_cli_style_recommendation_fingerprints_match_profile_payload(self) -> None:
        data = canonical_benchmarks()
        recommendation = dict(recommend(data, canonical_constraints()))
        recommendation["input_fingerprints"] = {
            "benchmarks_sha256": "a" * 64,
            "constraints_sha256": "f" * 64,
        }
        profiles = derived_profiles(data)
        profiles["input_fingerprints"] = {
            "benchmarks_sha256": "a" * 64,
            "constraints_sha256": "f" * 64,
            "policies_sha256": "9" * 64,
        }

        report = render_report_v11(
            data,
            recommendation,
            policy_profiles=profiles,
            benchmarks_sha256="a" * 64,
            recommendation_sha256="b" * 64,
            profiles_sha256="c" * 64,
        )

        self.assertIn("Canonical policy", report)
        self.assertIn("Policy profiles SHA-256", report)
        self.assertIn("c" * 64, report)

        mismatched = deepcopy(profiles)
        mismatched["input_fingerprints"]["benchmarks_sha256"] = "8" * 64
        with self.assertRaisesRegex(ValidationError, "benchmark fingerprint"):
            render_report_v11(
                data,
                recommendation,
                policy_profiles=mismatched,
                benchmarks_sha256="a" * 64,
            )

    def test_no_optional_data_has_explicit_honest_empty_states(self) -> None:
        report = rendered_v11(profiles=False, load=None)

        self.assertIn("No load sweep was supplied.", report)
        self.assertNotIn("<script>", report)
        self.assertEqual(report.count('role="tab"'), 1)

        no_metadata = rendered_v11(
            data=canonical_benchmarks(metadata=False),
            profiles=False,
            load=None,
            stability=None,
        )
        self.assertIn("Pass-level variation was not supplied", no_metadata)
        self.assertIn("without a repeat-stability claim", no_metadata)

    def test_empty_load_sweep_is_rejected_instead_of_drawing_invented_curves(self) -> None:
        malformed = measured_load_sweep()
        malformed["rows"] = []
        malformed["highest_slo_concurrency"] = {}

        with self.assertRaises(ValidationError):
            rendered_v11(load=malformed)

    def test_load_curves_summary_slo_and_complete_table_are_rendered(self) -> None:
        report = rendered_v11()

        self.assertIn("Request throughput by concurrency", report)
        self.assertIn("Generated-token throughput by concurrency", report)
        self.assertIn("p95 end-to-end latency by concurrency", report)
        self.assertIn("Declared service-level objective", report)
        self.assertIn("Highest SLO-passing concurrency", report)
        self.assertIn("Completion rate", report)
        self.assertIn("100.00%", report)
        self.assertIn("Failed requests", report)
        self.assertIn("Error rate", report)
        self.assertIn("Peak resident memory", report)
        self.assertIn("Validated load-to-deployment command bindings", report)
        self.assertIn("Canonical parallel", report)
        self.assertIn("Load plan SHA-256", report)

    def test_repeat_coverage_is_explicitly_not_a_significance_claim(self) -> None:
        report = rendered_v11(stability=None)

        self.assertIn("2 linked (pass 1, pass 2)", report)
        self.assertIn("Coverage, not variation.", report)
        self.assertIn("not a stability result", report)
        self.assertIn("aggregate values alone do not quantify", report)
        self.assertIn("statistical significance", report)

    def test_validated_stability_renders_pass_deltas_spread_and_consistency(self) -> None:
        report = rendered_v11()

        self.assertIn("Observed passes:</strong> measured-A, measured-B", report)
        self.assertIn("Observed pass directions, relative deltas, and relative spread", report)
        self.assertIn("Pass values and deltas", report)
        self.assertIn("Relative spread", report)
        self.assertIn("+", report)
        self.assertIn("consistent", report)
        self.assertIn("no significance claim", report)
        self.assertIn("not a statistical-significance claim", report)

    def test_scatter_title_is_two_metric_specific_and_has_textual_data(self) -> None:
        report = rendered_v11()

        self.assertIn("p95 end-to-end latency vs generation throughput", report)
        self.assertIn("not the complete multi-objective frontier", report)
        self.assertNotIn("Pareto view", report)
        self.assertIn("p95 end-to-end latency vs generation throughput data", report)
        self.assertIn("Declared frontier", report)

    def test_responsive_flat_styles_and_self_contained_assets(self) -> None:
        report = rendered_v11()

        self.assertIn("@media (min-width: 48rem)", report)
        self.assertIn(".verdict-layout { grid-template-columns:", report)
        self.assertIn("overflow-x: auto", report)
        self.assertIn(".tradeoff-row", report)
        self.assertIn("@media print", report)
        self.assertNotIn("linear-gradient", report)
        self.assertNotIn("radial-gradient", report)
        self.assertNotIn("backdrop-filter", report)
        self.assertNotIn("box-shadow", report)
        self.assertNotIn("@keyframes", report)
        self.assertNotIn("<script src=", report)
        self.assertNotIn('<link rel="stylesheet"', report)

    def test_invalid_profiles_load_rows_recommendation_and_hashes_fail_closed(self) -> None:
        data = canonical_benchmarks()
        recommendation = canonical_recommendation(data)
        duplicate_load = measured_load_sweep()
        duplicate_rows = deepcopy(duplicate_load["rows"])
        assert isinstance(duplicate_rows, list)
        duplicate_rows.append(deepcopy(duplicate_rows[0]))
        duplicate_load["rows"] = duplicate_rows

        non_finite_load = measured_load_sweep()
        non_finite_rows = deepcopy(non_finite_load["rows"])
        assert isinstance(non_finite_rows, list)
        non_finite_rows[0]["requests_per_second"] = math.inf
        non_finite_load["rows"] = non_finite_rows

        invalid_completion_load = measured_load_sweep()
        invalid_completion_rows = deepcopy(invalid_completion_load["rows"])
        assert isinstance(invalid_completion_rows, list)
        invalid_completion_rows[0]["completion_rate"] = 1.1
        invalid_completion_load["rows"] = invalid_completion_rows

        unknown_field_load = measured_load_sweep()
        unknown_field_rows = deepcopy(unknown_field_load["rows"])
        assert isinstance(unknown_field_rows, list)
        unknown_field_rows[0]["guessed_cost"] = 1.0
        unknown_field_load["rows"] = unknown_field_rows
        cases: list[tuple[str, dict[str, object]]] = [
            (
                "unknown profile candidate",
                {"policy_profiles": {"bad": {"selected_id": "missing"}}},
            ),
            (
                "duplicate load identity",
                {"load_sweep": duplicate_load},
            ),
            (
                "non-finite load value",
                {"load_sweep": non_finite_load},
            ),
            (
                "invalid completion rate",
                {"load_sweep": invalid_completion_load},
            ),
            (
                "unknown load field",
                {"load_sweep": unknown_field_load},
            ),
            ("invalid hash", {"benchmarks_sha256": "not-a-digest"}),
        ]
        for label, kwargs in cases:
            with self.subTest(label=label):
                with self.assertRaises(ValidationError):
                    render_report_v11(data, recommendation, **kwargs)

        mismatched = deepcopy(recommendation)
        mismatched["baseline_id"] = "q4-generic"
        with self.assertRaisesRegex(ValidationError, "does not match"):
            render_report_v11(data, mismatched)

        tampered_winner = deepcopy(recommendation)
        tampered_winner["selected_id"] = "q4-generic"
        with self.assertRaisesRegex(ValidationError, "fresh selection"):
            render_report_v11(data, tampered_winner)

        tampered_profiles = derived_profiles(data)
        tampered_profiles["profiles"][1]["recommendation"]["selected_id"] = "q8-generic"
        with self.assertRaisesRegex(ValidationError, "fresh selection"):
            render_report_v11(
                data,
                recommendation,
                policy_profiles=tampered_profiles,
            )

        swapped_load_binding = measured_load_sweep()
        configurations = swapped_load_binding["evidence_bindings"][
            "candidate_server_configurations"
        ]
        configurations["q4-generic"] = deepcopy(configurations["q8-generic"])
        swapped_load_binding["evidence_bindings"]["candidate_request_base_urls"]["q4-generic"] = (
            swapped_load_binding["evidence_bindings"]["candidate_request_base_urls"]["q8-generic"]
        )
        with self.assertRaisesRegex(ValidationError, "deployment_argv"):
            render_report_v11(
                data,
                recommendation,
                load_sweep=swapped_load_binding,
            )

        wrong_baseline_stability = measured_stability(
            data,
            baseline_id="q4-generic",
        )
        with self.assertRaisesRegex(ValidationError, "baseline_id does not match"):
            render_report_v11(
                data,
                recommendation,
                stability_summary=wrong_baseline_stability,
            )

        wrong_synthetic_stability = measured_stability(data)
        wrong_synthetic_stability["synthetic"] = True
        with self.assertRaisesRegex(ValidationError, "synthetic does not match"):
            render_report_v11(
                data,
                recommendation,
                stability_summary=wrong_synthetic_stability,
            )

        wrong_configuration_stability = measured_stability(data)
        wrong_configuration_stability["candidate_configuration_fingerprints"]["q4-generic"] = (
            "0" * 64
        )
        with self.assertRaisesRegex(
            ValidationError,
            "configuration fingerprint does not match",
        ):
            render_report_v11(
                data,
                recommendation,
                stability_summary=wrong_configuration_stability,
            )

        subset = BenchmarkSet(
            schema_version=data.schema_version,
            baseline_id=data.baseline_id,
            candidates=data.candidates[:-1],
            metadata={},
            synthetic=data.synthetic,
        )
        with self.assertRaisesRegex(ValidationError, "candidate ids must match"):
            render_report_v11(
                data,
                recommendation,
                stability_summary=measured_stability(subset),
            )

    def test_deployment_export_fails_closed_for_unsafe_argv(self) -> None:
        data = canonical_benchmarks()
        raw = {
            "schema_version": data.schema_version,
            "baseline_id": data.baseline_id,
            "synthetic": data.synthetic,
            "metadata": dict(data.metadata),
            "candidates": [
                {
                    "id": candidate.candidate_id,
                    "label": candidate.label,
                    "parameters": (
                        {
                            **dict(candidate.parameters),
                            "deployment_argv": ["llama-server\nrm -rf /tmp/example"],
                        }
                        if candidate.candidate_id == "q8-generic"
                        else dict(candidate.parameters)
                    ),
                    "metrics": dict(candidate.metrics),
                }
                for candidate in data.candidates
            ],
        }
        unsafe = BenchmarkSet.from_mapping(raw)
        report = render_report_v11(
            unsafe,
            canonical_recommendation(unsafe),
        )

        self.assertIn("Deployment argument 0 is invalid", report)
        self.assertNotIn('<pre class="command">', report)
        self.assertIn("llama-server\\nrm -rf /tmp/example", report)


if __name__ == "__main__":
    unittest.main()
