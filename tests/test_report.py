from __future__ import annotations

import unittest

from paretopilot.analysis import recommend
from paretopilot.domain import BenchmarkSet, Constraints
from paretopilot.report import render_report


def report_constraints() -> Constraints:
    return Constraints.from_mapping(
        {
            "min_quality_retention": 0.95,
            "quality_metric": "quality_score",
            "max_values": {"e2e_latency_ms_p95": 3500, "peak_rss_mib": 4096},
            "min_values": {},
            "objective": {"metric": "generation_tps", "direction": "max"},
            "frontier_metrics": {
                "generation_tps": "max",
                "e2e_latency_ms_p95": "min",
                "peak_rss_mib": "min",
                "quality_score": "max",
            },
        }
    )


def report_benchmarks(*, synthetic: bool = True, safe_parameters: bool = True) -> BenchmarkSet:
    selected_parameters: dict[str, object] = {
        "quantization": "Q4_0",
        "kleidiai": True,
        "threads": 4,
        "batch_size": 256,
    }
    if safe_parameters:
        selected_parameters.update(
            {
                "runtime_binary": "./build-kleidiai/bin/llama-server",
                "model_path": "/models/qwen 0.5b.gguf",
                "context_size": 2048,
            }
        )
    return BenchmarkSet.from_mapping(
        {
            "schema_version": "1.0",
            "baseline_id": "baseline",
            "synthetic": synthetic,
            "metadata": {
                "runner": "ubuntu-24.04-arm",
                "note": "Controlled A-B-B-A comparison",
            },
            "candidates": [
                {
                    "id": "baseline",
                    "label": "Generic baseline",
                    "parameters": {"quantization": "Q4_0", "threads": 4},
                    "metrics": {
                        "quality_score": 1.0,
                        "e2e_latency_ms_p95": 3200,
                        "generation_tps": 20,
                        "peak_rss_mib": 1800,
                        "energy_joules": 0,
                    },
                },
                {
                    "id": "optimized",
                    "label": "KleidiAI candidate",
                    "parameters": selected_parameters,
                    "metrics": {
                        "quality_score": 1.0,
                        "e2e_latency_ms_p95": 2600,
                        "generation_tps": 24,
                        "peak_rss_mib": 1750,
                    },
                },
                {
                    "id": "fast-low-quality",
                    "label": "Fast but below quality floor",
                    "parameters": {"quantization": "Q2", "threads": 4},
                    "metrics": {
                        "quality_score": 0.8,
                        "e2e_latency_ms_p95": 2000,
                        "generation_tps": 30,
                        "peak_rss_mib": 1500,
                    },
                },
            ],
        }
    )


def rendered_report(
    benchmarks: BenchmarkSet | None = None,
    constraints: Constraints | None = None,
) -> str:
    data = benchmarks or report_benchmarks()
    policy = constraints or report_constraints()
    return render_report(
        data,
        policy,
        recommend(data, policy),
        benchmarks_sha256="a" * 64,
        constraints_sha256="b" * 64,
    )


def with_candidate_parameters(
    data: BenchmarkSet,
    candidate_id: str,
    parameters: dict[str, object],
) -> BenchmarkSet:
    return BenchmarkSet.from_mapping(
        {
            "schema_version": data.schema_version,
            "baseline_id": data.baseline_id,
            "synthetic": data.synthetic,
            "metadata": dict(data.metadata),
            "candidates": [
                {
                    "id": item.candidate_id,
                    "label": item.label,
                    "parameters": (
                        parameters if item.candidate_id == candidate_id else dict(item.parameters)
                    ),
                    "metrics": dict(item.metrics),
                }
                for item in data.candidates
            ],
        }
    )


class ReportTests(unittest.TestCase):
    def test_output_is_byte_deterministic(self) -> None:
        data = report_benchmarks()
        policy = report_constraints()
        recommendation = recommend(data, policy)

        first = render_report(
            data,
            policy,
            recommendation,
            benchmarks_sha256="a" * 64,
            constraints_sha256="b" * 64,
        )
        second = render_report(
            data,
            policy,
            recommendation,
            benchmarks_sha256="a" * 64,
            constraints_sha256="b" * 64,
        )

        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))
        self.assertNotIn("generated_at", first)

    def test_all_source_text_is_html_escaped(self) -> None:
        data = BenchmarkSet.from_mapping(
            {
                "schema_version": "1.0",
                "baseline_id": "baseline<script>",
                "synthetic": True,
                "metadata": {"<img>": "<script>alert('metadata')</script>"},
                "candidates": [
                    {
                        "id": "baseline<script>",
                        "label": "<img src=x onerror=alert('label')>",
                        "parameters": {"model_path": "<unsafe>&'\""},
                        "metrics": {
                            "quality_score": 1.0,
                            "generation_tps": 1.0,
                            "e2e_latency_ms_p95": 1.0,
                            "peak_rss_mib": 1.0,
                        },
                    }
                ],
            }
        )
        policy = report_constraints()
        report = rendered_report(data, policy)

        self.assertNotIn("<script>alert", report)
        self.assertNotIn("<img src=x", report)
        self.assertIn("&lt;img src=x onerror=alert(&#x27;label&#x27;)&gt;", report)
        self.assertIn("&lt;script&gt;alert(&#x27;metadata&#x27;)&lt;/script&gt;", report)
        self.assertIn("&lt;unsafe&gt;&amp;&#x27;\\&quot;", report)

    def test_synthetic_source_has_persistent_warning_banner(self) -> None:
        report = rendered_report()

        self.assertIn('class="synthetic-banner" role="alert"', report)
        self.assertIn("Synthetic evidence — do not cite as measured Arm performance.", report)
        self.assertIn("position: sticky", report)

    def test_measured_source_does_not_show_synthetic_banner(self) -> None:
        report = rendered_report(report_benchmarks(synthetic=False))

        self.assertNotIn('class="synthetic-banner" role="alert"', report)
        self.assertIn("Measured evidence · Decision evidence", report)

    def test_exploratory_source_has_persistent_noncanonical_banner(self) -> None:
        data = report_benchmarks(synthetic=False)
        raw = {
            "schema_version": data.schema_version,
            "baseline_id": data.baseline_id,
            "synthetic": data.synthetic,
            "metadata": {**dict(data.metadata), "classification": "exploratory"},
            "candidates": [
                {
                    "id": item.candidate_id,
                    "label": item.label,
                    "parameters": dict(item.parameters),
                    "metrics": dict(item.metrics),
                }
                for item in data.candidates
            ],
        }

        report = rendered_report(BenchmarkSet.from_mapping(raw))

        self.assertIn("Exploratory evidence — not canonical submission evidence.", report)
        self.assertIn("Exploratory measured evidence · Decision evidence", report)

    def test_objective_tolerance_and_preference_are_visible_guardrails(self) -> None:
        raw = {
            "min_quality_retention": 0.95,
            "quality_metric": "quality_score",
            "max_values": {"e2e_latency_ms_p95": 3500, "peak_rss_mib": 4096},
            "min_values": {},
            "objective": {"metric": "generation_tps", "direction": "max"},
            "objective_tolerance_percent": 1.0,
            "preference_order": ["baseline", "optimized", "fast-low-quality"],
            "frontier_metrics": {
                "generation_tps": "max",
                "e2e_latency_ms_p95": "min",
                "peak_rss_mib": "min",
                "quality_score": "max",
            },
        }

        report = rendered_report(constraints=Constraints.from_mapping(raw))

        self.assertIn("Objective tolerance", report)
        self.assertIn("Within 1.00% of the numeric best", report)
        self.assertIn("baseline → optimized → fast-low-quality", report)

    def test_baseline_selection_is_called_out_clearly(self) -> None:
        data = BenchmarkSet.from_mapping(
            {
                "schema_version": "1.0",
                "baseline_id": "baseline",
                "synthetic": False,
                "candidates": [
                    {
                        "id": "baseline",
                        "label": "Measured baseline",
                        "parameters": {},
                        "metrics": {
                            "quality_score": 1.0,
                            "generation_tps": 25,
                            "e2e_latency_ms_p95": 2000,
                            "peak_rss_mib": 1000,
                        },
                    },
                    {
                        "id": "candidate",
                        "label": "Slower candidate",
                        "parameters": {},
                        "metrics": {
                            "quality_score": 1.0,
                            "generation_tps": 20,
                            "e2e_latency_ms_p95": 2400,
                            "peak_rss_mib": 1100,
                        },
                    },
                ],
            }
        )
        report = rendered_report(data)

        self.assertIn("Baseline retained", report)
        self.assertIn("ParetoPilot retained the measured baseline.", report)
        self.assertIn("Selected · Baseline", report)

    def test_exact_rejection_reason_is_rendered(self) -> None:
        data = report_benchmarks()
        policy = report_constraints()
        result = recommend(data, policy)
        reason = result["rejected"]["fast-low-quality"][0]
        report = render_report(
            data,
            policy,
            result,
            benchmarks_sha256="a" * 64,
            constraints_sha256="b" * 64,
        )

        self.assertIn(reason, report)
        self.assertIn("Rejected", report)

    def test_missing_metric_is_not_rendered_as_zero(self) -> None:
        report = rendered_report()

        self.assertIn('data-metric="energy_joules"', report)
        self.assertIn('<span class="metric-value not-measured">Not measured</span>', report)
        self.assertIn("Insufficient evidence", report)

    def test_safe_deployment_command_is_shell_quoted_and_local_only(self) -> None:
        report = rendered_report(report_benchmarks(safe_parameters=True))

        self.assertIn("Exportable", report)
        self.assertIn("./build-kleidiai/bin/llama-server", report)
        self.assertIn("&#x27;/models/qwen 0.5b.gguf&#x27;", report)
        self.assertIn("--threads 4 --batch-size 256 --host 127.0.0.1", report)
        self.assertIn("--ctx-size 2048", report)

    def test_authoritative_deployment_argv_is_rendered_without_added_flags(self) -> None:
        data = report_benchmarks(synthetic=False, safe_parameters=True)
        selected_parameters = dict(data.by_id("optimized").parameters)
        selected_parameters["deployment_argv"] = [
            "./bin/llama server",
            "--model",
            "/models/qwen 0.5b.gguf",
            "--threads",
            "6",
        ]
        authoritative = with_candidate_parameters(data, "optimized", selected_parameters)

        report = rendered_report(authoritative)

        self.assertIn("Exportable", report)
        self.assertIn(
            "&#x27;./bin/llama server&#x27; --model &#x27;/models/qwen 0.5b.gguf&#x27; --threads 6",
            report,
        )
        self.assertIn("authoritative deployment_argv", report)
        self.assertIn("added no executable, flags, or values", report)
        self.assertNotIn("--host 127.0.0.1", report)
        self.assertNotIn("--batch-size 256", report)

    def test_invalid_authoritative_argv_fails_closed_without_flat_fallback(self) -> None:
        data = report_benchmarks(synthetic=True, safe_parameters=True)
        selected_parameters = dict(data.by_id("optimized").parameters)
        selected_parameters["deployment_argv"] = [
            "llama-server",
            "--model",
            "/models/model.gguf\n--host 0.0.0.0",
        ]
        invalid = with_candidate_parameters(data, "optimized", selected_parameters)

        report = rendered_report(invalid)

        self.assertIn("Not exportable", report)
        self.assertIn("deployment_argv[2] contains a control character", report)
        self.assertIn("failed closed", report)
        self.assertNotIn('<pre class="command">', report)
        self.assertNotIn("--batch-size 256 --host 127.0.0.1", report)

    def test_incomplete_or_unsafe_parameters_do_not_export_a_command(self) -> None:
        data = report_benchmarks(safe_parameters=False)
        report = rendered_report(data)

        self.assertIn("Not exportable", report)
        self.assertIn("runtime_binary is missing", report)
        self.assertIn("model_path is missing", report)
        self.assertNotIn('<pre class="command">', report)

        raw = {
            "schema_version": data.schema_version,
            "baseline_id": data.baseline_id,
            "synthetic": data.synthetic,
            "metadata": dict(data.metadata),
            "candidates": [
                {
                    "id": item.candidate_id,
                    "label": item.label,
                    "parameters": (
                        {
                            **dict(item.parameters),
                            "runtime_binary": "llama-server\nrm -rf /tmp/example",
                            "model_path": "/models/model.gguf",
                        }
                        if item.candidate_id == "optimized"
                        else dict(item.parameters)
                    ),
                    "metrics": dict(item.metrics),
                }
                for item in data.candidates
            ],
        }
        unsafe = BenchmarkSet.from_mapping(raw)
        unsafe_report = rendered_report(unsafe)
        self.assertIn("runtime_binary is invalid", unsafe_report)
        self.assertNotIn('<pre class="command">', unsafe_report)
        self.assertIn("llama-server\\nrm -rf /tmp/example", unsafe_report)

    def test_report_has_accessible_landmarks_tables_and_status_text(self) -> None:
        report = rendered_report()

        self.assertIn('<a class="skip-link" href="#main-content">', report)
        self.assertIn('<header class="report-header">', report)
        self.assertIn('<main id="main-content" class="report-main">', report)
        self.assertIn('<link rel="icon" href="data:,">', report)
        self.assertIn('<footer class="report-footer">', report)
        self.assertGreaterEqual(report.count("<caption>"), 4)
        self.assertIn('role="status" aria-label="Recommendation outcome"', report)
        self.assertIn("Peak resident memory", report)
        self.assertIn("1,750 MiB", report)
        self.assertIn(":focus-visible", report)

    def test_scatter_requires_both_metrics_for_every_candidate(self) -> None:
        complete = rendered_report()
        self.assertIn('<svg viewBox="0 0 800 360" role="img"', complete)
        self.assertIn("pareto-chart-description", complete)

        data = report_benchmarks()
        raw = {
            "schema_version": data.schema_version,
            "baseline_id": data.baseline_id,
            "synthetic": data.synthetic,
            "metadata": dict(data.metadata),
            "candidates": [
                {
                    "id": item.candidate_id,
                    "label": item.label,
                    "parameters": dict(item.parameters),
                    "metrics": (
                        {
                            key: value
                            for key, value in item.metrics.items()
                            if key != "generation_tps"
                        }
                        if item.candidate_id == "fast-low-quality"
                        else dict(item.metrics)
                    ),
                }
                for item in data.candidates
            ],
        }
        incomplete = BenchmarkSet.from_mapping(raw)
        result = recommend(incomplete, report_constraints())
        report = render_report(
            incomplete,
            report_constraints(),
            result,
            benchmarks_sha256="a" * 64,
            constraints_sha256="b" * 64,
        )
        self.assertNotIn('<svg viewBox="0 0 800 360"', report)
        self.assertIn("Scatter plot unavailable.", report)

    def test_current_schema_and_optional_performix_copy_are_supported(self) -> None:
        report = rendered_report()

        self.assertTrue(report.startswith('<!doctype html>\n<html lang="en">'))
        self.assertIn("Benchmark schema</dt><dd>1.0", report)
        self.assertIn("Performix is an optional supplementary profiler", report)
        self.assertNotIn("http://", report)
        self.assertNotIn("https://", report)

    def test_canonical_study_and_server_metric_names_have_labels_units_and_priority(
        self,
    ) -> None:
        data = BenchmarkSet.from_mapping(
            {
                "schema_version": "1.0",
                "baseline_id": "canonical",
                "synthetic": False,
                "candidates": [
                    {
                        "id": "canonical",
                        "label": "Canonical Arm64 baseline",
                        "parameters": {},
                        "metrics": {
                            "generation_tokens_per_second": 35.1134,
                            "prompt_tokens_per_second": 113.6605,
                            "generation_duration_ms": 14.5,
                            "prompt_duration_ms": 9.25,
                            "model_identity_quality_retention": 1.0,
                            "confidence_and_practical_effect_gate": 1.0,
                            "prompt_tps": 110.0,
                            "ttft_ms_p95": 24.0,
                        },
                    }
                ],
            }
        )
        policy = Constraints.from_mapping(
            {
                "min_quality_retention": 1.0,
                "quality_metric": "model_identity_quality_retention",
                "max_values": {},
                "min_values": {"confidence_and_practical_effect_gate": 1.0},
                "objective": {
                    "metric": "generation_tokens_per_second",
                    "direction": "max",
                },
                "frontier_metrics": {
                    "generation_tokens_per_second": "max",
                    "prompt_tokens_per_second": "max",
                    "generation_duration_ms": "min",
                    "prompt_duration_ms": "min",
                    "model_identity_quality_retention": "max",
                },
            }
        )

        report = rendered_report(data, policy)

        self.assertIn("35.1134 tok/s", report)
        self.assertIn("113.6605 tok/s", report)
        self.assertIn("110 tok/s", report)
        self.assertIn("14.5 ms", report)
        self.assertIn("9.25 ms", report)
        self.assertIn("Time to first token (p95)", report)
        self.assertIn("24 ms", report)
        self.assertIn("Model-identity quality retention", report)
        self.assertIn("100.00%", report)
        self.assertIn("Confidence and practical-effect gate", report)
        self.assertIn("Pass (1)", report)
        self.assertLess(
            report.index("35.1134 tok/s"),
            report.index("113.6605 tok/s"),
        )


if __name__ == "__main__":
    unittest.main()
