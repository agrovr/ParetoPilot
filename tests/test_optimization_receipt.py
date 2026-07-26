from __future__ import annotations

import math
import unittest

from paretopilot.decision_passport import build_decision_passport
from paretopilot.domain import BenchmarkSet, Constraints, ValidationError
from paretopilot.optimization_receipt import render_optimization_receipt


def _candidate(
    candidate_id: str,
    label: str,
    attribution_stage: str,
    *,
    latency: float,
    generation: float,
    prompt: float,
    rss: float,
    model_size: float,
    quality: float,
) -> dict[str, object]:
    return {
        "id": candidate_id,
        "label": label,
        "parameters": {
            "configuration": {
                "attribution_stage": attribution_stage,
                "threads": 4,
            }
        },
        "metrics": {
            "e2e_latency_ms_p95": latency,
            "generation_tps": generation,
            "model_size_mib": model_size,
            "peak_rss_mib": rss,
            "prompt_tps": prompt,
            "quality_score": quality,
        },
    }


def _passport(*, synthetic: bool = False) -> dict[str, object]:
    benchmarks = BenchmarkSet.from_mapping(
        {
            "schema_version": "1.0",
            "baseline_id": "q8-generic",
            "synthetic": synthetic,
            "metadata": {
                "classification": "canonical",
                "source": {
                    "repository": "agrovr/ParetoPilot",
                    "revision": "8a9ddce0afa2272c4a4097fe87ef6f06cb7689a9",
                    "workflow": ".github/workflows/candidate-study-arm64.yml",
                    "run_id": "30055662526",
                    "run_attempt": 1,
                    "runner": {
                        "architecture": "arm64",
                        "cpu": "Neoverse-N2",
                        "cpu_count": 4,
                        "os": "Ubuntu 24.04",
                    },
                },
                "runtime": {
                    "name": "llama.cpp",
                    "repository": "https://github.com/ggml-org/llama.cpp",
                    "revision": "67b9b0e7f6ce45d929a4411907d3c48ec719e81c",
                },
                "model_family": {
                    "name": "Qwen2.5-1.5B-Instruct",
                    "repository": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                    "revision": "91cad51170dc346986eccefdc2dd33a9da36ead9",
                },
                "evaluation_suite": {
                    "id": "paretopilot-qwen-behavior-v2",
                    "sha256": "e49c16fba32fd65c947264aef4141026ab68b1fd415ef09eeea6e8ade9a545c7",
                },
            },
            # Deliberately shuffled: the passport, not source order, defines the
            # four-stage receipt path.
            "candidates": [
                _candidate(
                    "q4-kleidiai-tuned",
                    "Q4 with KleidiAI tuned",
                    "runtime-tuning",
                    latency=103.0,
                    generation=36.0,
                    prompt=130.0,
                    rss=480.0,
                    model_size=500.0,
                    quality=0.96,
                ),
                _candidate(
                    "q8-generic",
                    "Q8 generic reference",
                    "reference",
                    latency=100.0,
                    generation=50.0,
                    prompt=100.0,
                    rss=1000.0,
                    model_size=1000.0,
                    quality=1.0,
                ),
                _candidate(
                    "q4-kleidiai",
                    "Q4 with KleidiAI",
                    "arm-kernel",
                    latency=106.0,
                    generation=40.0,
                    prompt=120.0,
                    rss=490.0,
                    model_size=500.0,
                    quality=0.96,
                ),
                _candidate(
                    "q4-generic",
                    "Q4 generic",
                    "quantization",
                    latency=104.0,
                    generation=35.0,
                    prompt=110.0,
                    rss=500.0,
                    model_size=500.0,
                    quality=0.96,
                ),
            ],
        }
    )
    constraints = Constraints.from_mapping(
        {
            "min_quality_retention": 0.9,
            "quality_metric": "quality_score",
            "max_values": {"peak_rss_mib": 2000.0},
            "min_values": {"quality_score": 0.9},
            "objective": {
                "metric": "e2e_latency_ms_p95",
                "direction": "min",
            },
            "objective_tolerance_percent": 5.0,
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
                "prompt_tps": "max",
                "quality_score": "max",
            },
        }
    )
    return dict(build_decision_passport(benchmarks, constraints))


def _generic_synthetic_passport() -> dict[str, object]:
    benchmarks = BenchmarkSet.from_mapping(
        {
            "schema_version": "1.0",
            "baseline_id": "baseline",
            "synthetic": True,
            "candidates": [
                {
                    "id": "baseline",
                    "label": "Generic baseline",
                    "parameters": {},
                    "metrics": {"latency_ms": 10.0, "quality": 1.0},
                },
                {
                    "id": "candidate",
                    "label": "Generic candidate",
                    "parameters": {},
                    "metrics": {"latency_ms": 9.0, "quality": 1.0},
                },
            ],
        }
    )
    constraints = Constraints.from_mapping(
        {
            "min_quality_retention": 0.9,
            "quality_metric": "quality",
            "max_values": {},
            "min_values": {"quality": 0.9},
            "objective": {"metric": "latency_ms", "direction": "min"},
            "objective_tolerance_percent": 0.0,
            "preference_order": ["candidate", "baseline"],
            "frontier_metrics": {"latency_ms": "min", "quality": "max"},
        }
    )
    return dict(build_decision_passport(benchmarks, constraints))


class OptimizationReceiptTests(unittest.TestCase):
    def test_renders_complete_deterministic_receipt_from_real_passport(self) -> None:
        passport = _passport()

        first = render_optimization_receipt(passport)
        second = render_optimization_receipt(passport)

        self.assertEqual(first, second)
        self.assertTrue(first.endswith("\n"))
        self.assertFalse(first.endswith("\n\n"))
        self.assertNotIn("\r", first)
        self.assertIn("# ParetoPilot decision summary", first)
        self.assertIn("## Decision", first)
        self.assertIn("Q8 generic reference (q8\\-generic)", first)
        self.assertIn("## How the cutoff was applied", first)
        self.assertIn("| Objective | End-to-end latency p95 (ms) |", first)
        self.assertIn("| Shortlist boundary | 105 (at\\-or\\-below) |", first)
        self.assertIn("| Margin to shortlist boundary | 5 (4.76% of boundary) |", first)
        self.assertIn(
            "Displayed measurements are rounded for readability; exact values remain",
            first,
        )
        self.assertIn("## Configuration comparison", first)
        for heading in (
            "### Stage 1 — Reference",
            "### Stage 2 — Quantization",
            "### Stage 3 — KleidiAI build",
            "### Stage 4 — Runtime tuning",
        ):
            self.assertIn(heading, first)
        self.assertEqual(first.count("Adjacent delta from "), 3)
        self.assertIn(
            "| End-to-end latency p95 (ms) | Minimize | 100 | 104 | +4 | +4% | Tradeoff |",
            first,
        )
        self.assertIn("## Resource alternative", first)
        self.assertIn("Lower-resource alternative", first)
        self.assertIn("Compared with: Q8 generic reference (q8\\-generic).", first)
        self.assertIn(
            "| Peak RSS (MiB) | Minimize | 1,000 | 480 | -520 | -52% | Improved |",
            first,
        )
        self.assertNotIn("4.7619047619", first)
        self.assertNotIn("e2e\\_latency\\_ms\\_p95", first)
        self.assertIn("## Source and verification details", first)
        self.assertIn("| Evaluation suite SHA-256 | e49c16f", first)
        self.assertIn("**Verification scope:**", first)
        self.assertIn("**Measured Arm64 result.**", first)
        self.assertIn("See the limits below", first)
        self.assertIn("**Limits:**", first)
        self.assertNotIn("Published canonical outputs changed", first)
        self.assertNotIn("scope below remains authoritative", first)
        self.assertNotIn("**Boundary caveat:**", first)
        self.assertNotIn("Generated at", first)
        self.assertNotIn("Timestamp", first)

    def test_escapes_markdown_and_html_controlled_source_text(self) -> None:
        passport = _passport()
        dangerous = "Q8 | *bold* [link](javascript:alert(1))\r\n# title <script>"
        passport["selected_decision"]["label"] = dangerous
        passport["selected_decision"]["reason"] = "Keep _baseline_ | [click](bad)"
        passport["ladder"][0]["label"] = dangerous
        passport["provenance"]["source"]["repository"] = "<img src=x onerror=alert(1)>"
        passport["method"]["current_boundary_caveat"] = "Scope | *only* <iframe>"

        rendered = render_optimization_receipt(passport)

        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<img", rendered)
        self.assertNotIn("<iframe>", rendered)
        self.assertNotIn("\r", rendered)
        self.assertIn(
            "Q8 \\| \\*bold\\* \\[link\\]\\(javascript:alert\\(1\\)\\) \\# title &lt;script&gt;",
            rendered,
        )
        self.assertIn("Keep \\_baseline\\_ \\| \\[click\\]\\(bad\\)", rendered)
        self.assertIn("&lt;img src=x onerror=alert\\(1\\)&gt;", rendered)

    def test_absent_optional_evidence_is_not_inferred(self) -> None:
        passport = _passport()
        passport["evidence_grade"] = "measured-unattributed"
        passport["provenance"]["attribution_complete"] = False
        passport["provenance"]["runner"]["cpu"] = None
        passport["provenance"]["run"]["id"] = None
        passport["provenance"]["evaluation_suite"]["sha256"] = None
        passport["provenance"]["issues"] = [
            "metadata.source.runner.cpu is missing or invalid",
            "metadata.source.run_id is missing or invalid",
            "metadata.evaluation_suite.sha256 is missing or invalid",
        ]
        passport["ladder"][1]["objective_value"] = None
        passport["closest_outside_shortlist"]["objective_value"] = None
        passport["closest_outside_shortlist"]["shortfall_to_shortlist"]["percent_of_boundary"] = (
            None
        )
        passport["ladder"][1]["delta_from_previous"]["metrics"][0]["percent"] = None

        rendered = render_optimization_receipt(passport)

        self.assertIn("| Runner CPU | Not measured |", rendered)
        self.assertIn("| Run ID | Not measured |", rendered)
        self.assertIn("| Evaluation suite SHA-256 | Not measured |", rendered)
        self.assertIn("| Objective (End-to-end latency p95 (ms)) | Not measured |", rendered)
        self.assertGreaterEqual(rendered.count("Not measured"), 6)
        self.assertNotIn("None", rendered.split("## Source and verification details", 1)[1])

    def test_synthetic_receipt_uses_fixture_safe_language(self) -> None:
        passport = _passport(synthetic=True)

        rendered = render_optimization_receipt(passport)

        self.assertIn("**Synthetic example.**", rendered)
        self.assertIn("Displayed example values are rounded for readability", rendered)
        self.assertIn("example data, not measured Arm64 or deployment results", rendered)
        self.assertIn("Each stage changes one configuration setting.", rendered)
        self.assertIn(
            "The synthetic example is not a measured deployment benchmark",
            rendered,
        )
        self.assertNotIn("Lower-resource alternative", rendered)
        self.assertNotIn("**Arm64-attributed source evidence.**", rendered)

    def test_generic_synthetic_passport_keeps_unrecognized_stages_plain(self) -> None:
        passport = _generic_synthetic_passport()
        passport["input_fingerprints"] = {
            "constraints_sha256": "b" * 64,
            "benchmarks_sha256": "a" * 64,
        }
        self.assertEqual(len(passport["ladder"]), 2)
        self.assertTrue(all(stage["attribution_stage"] is None for stage in passport["ladder"]))
        self.assertTrue(
            all(not stage["recognized_attribution_stage"] for stage in passport["ladder"])
        )

        rendered = render_optimization_receipt(passport)

        self.assertIn("## Compared configurations", rendered)
        self.assertNotIn("## Configuration comparison", rendered)
        self.assertIn("### Stage 1\n", rendered)
        self.assertIn("### Stage 2\n", rendered)
        self.assertNotIn("### Stage 1 —", rendered)
        self.assertEqual(rendered.count("| What changed | Not measured |"), 2)
        self.assertEqual(rendered.count("Adjacent delta from "), 1)
        self.assertIn("**Synthetic example.**", rendered)
        self.assertIn("| Objective | latency\\_ms |", rendered)
        self.assertNotIn("Latency (ms)", rendered)
        self.assertNotIn("Reference", rendered)
        self.assertNotIn("Quantization", rendered)
        self.assertIn(
            f"| Input fingerprint (benchmarks\\_sha256) | {'a' * 64} |",
            rendered,
        )
        self.assertIn(
            f"| Input fingerprint (constraints\\_sha256) | {'b' * 64} |",
            rendered,
        )
        self.assertLess(rendered.index("a" * 64), rendered.index("b" * 64))

    def test_fails_closed_on_malformed_required_structure(self) -> None:
        cases: list[tuple[str, object, str]] = []

        missing_objective = _passport()
        del missing_objective["objective"]
        cases.append(("missing objective", missing_objective, "passport.objective is required"))

        empty_ladder = _passport()
        empty_ladder["ladder"].clear()
        cases.append(("empty ladder", empty_ladder, "at least one stage"))

        broken_link = _passport()
        broken_link["ladder"][2]["delta_from_previous"]["previous_candidate_id"] = "wrong"
        cases.append(("broken adjacency", broken_link, "immediately preceding stage"))

        nonfinite_delta = _passport()
        nonfinite_delta["ladder"][1]["delta_from_previous"]["metrics"][0]["current"] = math.inf
        cases.append(("nonfinite delta", nonfinite_delta, "must be finite"))

        wrong_baseline = _passport()
        wrong_baseline["ladder"][0]["baseline"] = False
        cases.append(("missing baseline marker", wrong_baseline, "declared baseline"))

        synthetic_alternative = _passport(synthetic=True)
        synthetic_alternative["resource_alternative"] = _passport()["resource_alternative"]
        cases.append(
            (
                "synthetic alternative",
                synthetic_alternative,
                "must be null for synthetic fixture evidence",
            )
        )

        for name, malformed, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValidationError, message):
                    render_optimization_receipt(malformed)

    def test_requires_mapping_input(self) -> None:
        with self.assertRaisesRegex(TypeError, "Decision Passport mapping"):
            render_optimization_receipt([])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
