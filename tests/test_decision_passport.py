from __future__ import annotations

from copy import deepcopy
import json
import math
import unittest

from paretopilot.decision_passport import build_decision_passport
from paretopilot.domain import BenchmarkSet, Constraints, ValidationError


def _metadata(*, architecture: object = "arm64") -> dict[str, object]:
    return {
        "classification": "canonical",
        "source": {
            "repository": "agrovr/ParetoPilot",
            "revision": "8a9ddce0afa2272c4a4097fe87ef6f06cb7689a9",
            "workflow": ".github/workflows/candidate-study-arm64.yml",
            "run_id": "30055662526",
            "run_attempt": 1,
            "runner": {
                "architecture": architecture,
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
    }


def _candidate(
    candidate_id: str,
    label: str,
    attribution_stage: str,
    *,
    e2e: float,
    ttft: float,
    prompt: float,
    generation: float,
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
            "e2e_latency_ms_p95": e2e,
            "generation_tps": generation,
            "model_size_mib": model_size,
            "peak_rss_mib": rss,
            "prompt_tps": prompt,
            "quality_score": quality,
            "ttft_ms_p95": ttft,
        },
    }


def _benchmark_mapping(
    *,
    synthetic: bool = False,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    # Deliberately not in attribution order. The production canonical benchmark
    # preserves this source order, while the passport must present the four
    # optimization stages from reference through runtime tuning.
    return {
        "schema_version": "1.0",
        "baseline_id": "q8-generic",
        "synthetic": synthetic,
        "metadata": _metadata() if metadata is None else metadata,
        "candidates": [
            _candidate(
                "q4-generic",
                "Q4 generic",
                "quantization",
                e2e=2311.125148,
                ttft=483.113115,
                prompt=113.8210,
                generation=35.01235,
                rss=1966.472656,
                model_size=1016.833527,
                quality=20 / 24,
            ),
            _candidate(
                "q4-kleidiai",
                "Q4 with KleidiAI",
                "arm-kernel",
                e2e=2299.454336,
                ttft=470.402254,
                prompt=114.4480,
                generation=35.37635,
                rss=1966.484375,
                model_size=1016.833527,
                quality=20 / 24,
            ),
            _candidate(
                "q4-kleidiai-tuned",
                "Q4 with KleidiAI tuned",
                "runtime-tuning",
                e2e=2307.715263,
                ttft=469.968079,
                prompt=131.4565,
                generation=35.09590,
                rss=1966.480469,
                model_size=1016.833527,
                quality=20 / 24,
            ),
            _candidate(
                "q8-generic",
                "Q8 generic reference",
                "reference",
                e2e=2231.932869,
                ttft=545.373894,
                prompt=102.6185,
                generation=38.72645,
                rss=3437.597656,
                model_size=1806.766632,
                quality=21 / 24,
            ),
        ],
    }


def _constraints() -> Constraints:
    return Constraints.from_mapping(
        {
            "min_quality_retention": 0.95,
            "quality_metric": "quality_score",
            "max_values": {
                "e2e_latency_ms_p95": 15000,
                "peak_rss_mib": 4096,
            },
            "min_values": {"quality_score": 0.8},
            "objective": {
                "metric": "e2e_latency_ms_p95",
                "direction": "min",
            },
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


class DecisionPassportTests(unittest.TestCase):
    def test_builds_deterministic_arm64_four_stage_decision_context(self) -> None:
        benchmarks = BenchmarkSet.from_mapping(_benchmark_mapping())

        first = build_decision_passport(benchmarks, _constraints())
        second = build_decision_passport(benchmarks, _constraints())

        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False, sort_keys=True)
        self.assertEqual(first["evidence_grade"], "arm64-attributed")
        self.assertTrue(first["provenance"]["attribution_complete"])
        self.assertEqual(
            [stage["candidate_id"] for stage in first["ladder"]],
            [
                "q8-generic",
                "q4-generic",
                "q4-kleidiai",
                "q4-kleidiai-tuned",
            ],
        )
        self.assertTrue(first["selected_decision"]["baseline_retained"])
        self.assertTrue(first["selected_decision"]["numeric_best"])

        objective = first["objective"]
        expected_boundary = 2231.932869 * 1.01
        self.assertAlmostEqual(objective["shortlist_boundary"], expected_boundary)
        self.assertAlmostEqual(
            objective["selected_runway"]["absolute"],
            expected_boundary - 2231.932869,
        )

        quantization_delta = first["ladder"][1]["delta_from_previous"]
        changes = {row["metric"]: row for row in quantization_delta["metrics"]}
        self.assertLess(changes["model_size_mib"]["absolute"], 0)
        self.assertEqual(changes["model_size_mib"]["effect"], "improved")
        self.assertEqual(changes["e2e_latency_ms_p95"]["effect"], "tradeoff")

        closest = first["closest_outside_shortlist"]
        self.assertEqual(closest["candidate_id"], "q4-kleidiai")
        self.assertGreater(closest["shortfall_to_shortlist"]["absolute"], 0)

        alternative = first["resource_alternative"]
        self.assertEqual(alternative["candidate_id"], "q4-kleidiai-tuned")
        self.assertTrue(alternative["is_secondary_not_recommendation"])
        improvements = {row["metric"] for row in alternative["improvements"]}
        tradeoffs = {row["metric"] for row in alternative["tradeoffs"]}
        self.assertTrue({"model_size_mib", "peak_rss_mib", "prompt_tps"} <= improvements)
        self.assertTrue({"e2e_latency_ms_p95", "generation_tps", "quality_score"} <= tradeoffs)
        self.assertIn(
            "applies only to the supplied benchmark",
            first["method"]["current_boundary_caveat"],
        )
        self.assertFalse(first["method"]["canonical_outputs_modified"])

    def test_synthetic_flag_overrides_complete_arm64_metadata(self) -> None:
        benchmarks = BenchmarkSet.from_mapping(_benchmark_mapping(synthetic=True))

        passport = build_decision_passport(benchmarks, _constraints())

        self.assertEqual(passport["evidence_grade"], "synthetic")
        self.assertFalse(passport["provenance"]["attribution_complete"])
        self.assertEqual(
            passport["provenance"]["issues"][0],
            "benchmark set is explicitly synthetic",
        )
        self.assertIsNone(passport["resource_alternative"])
        self.assertIn("synthetic example data", passport["method"]["resource_alternative"])
        self.assertNotIn(
            "measured comparison",
            passport["method"]["resource_alternative"],
        )
        caveat = passport["method"]["current_boundary_caveat"]
        self.assertIn("software example", caveat)
        self.assertIn("not measured Arm64 evidence", caveat)

    def test_arm64_attribution_requires_every_explicit_identity(self) -> None:
        aarch64 = BenchmarkSet.from_mapping(
            _benchmark_mapping(metadata=_metadata(architecture="AARCH64"))
        )
        self.assertEqual(
            build_decision_passport(aarch64, _constraints())["evidence_grade"],
            "arm64-attributed",
        )

        incomplete_metadata = _metadata()
        incomplete_metadata["source"]["runner"]["cpu"] = ""
        incomplete_metadata["source"]["run_id"] = math.inf
        incomplete_metadata["evaluation_suite"]["sha256"] = "not-a-digest"
        incomplete = BenchmarkSet.from_mapping(_benchmark_mapping(metadata=incomplete_metadata))

        passport = build_decision_passport(incomplete, _constraints())

        self.assertEqual(passport["evidence_grade"], "measured-unattributed")
        self.assertFalse(passport["provenance"]["attribution_complete"])
        self.assertIsNone(passport["provenance"]["runner"]["cpu"])
        self.assertIsNone(passport["provenance"]["run"]["id"])
        self.assertIsNone(passport["provenance"]["evaluation_suite"]["sha256"])
        self.assertNotIn("Infinity", json.dumps(passport, allow_nan=False))
        self.assertEqual(
            passport["provenance"]["issues"],
            [
                "metadata.source.runner.cpu is missing or invalid",
                "metadata.source.run_id is missing or invalid",
                "metadata.evaluation_suite.sha256 is missing or invalid",
            ],
        )

        wrong_architecture = BenchmarkSet.from_mapping(
            _benchmark_mapping(metadata=_metadata(architecture="x86_64"))
        )
        wrong_passport = build_decision_passport(wrong_architecture, _constraints())
        self.assertEqual(wrong_passport["evidence_grade"], "measured-unattributed")
        self.assertIn(
            "metadata.source.runner.architecture is not a recognized Arm64 architecture",
            wrong_passport["provenance"]["issues"],
        )

    def test_arm64_attribution_requires_positive_cpu_count_and_run_attempt(self) -> None:
        incomplete_metadata = _metadata()
        del incomplete_metadata["source"]["runner"]["cpu_count"]
        incomplete_metadata["source"]["run_attempt"] = 0
        benchmarks = BenchmarkSet.from_mapping(_benchmark_mapping(metadata=incomplete_metadata))

        passport = build_decision_passport(benchmarks, _constraints())

        self.assertEqual(passport["evidence_grade"], "measured-unattributed")
        self.assertFalse(passport["provenance"]["attribution_complete"])
        self.assertIsNone(passport["provenance"]["runner"]["cpu_count"])
        self.assertIsNone(passport["provenance"]["run"]["attempt"])
        self.assertIn(
            "checks whether required source metadata is present",
            passport["provenance"]["verification_scope"],
        )
        self.assertEqual(
            passport["provenance"]["issues"],
            [
                "metadata.source.runner.cpu_count is missing or invalid",
                "metadata.source.run_attempt is missing or invalid",
            ],
        )

    def test_passport_uses_recommendation_tolerance_at_near_boundary(self) -> None:
        cases = (
            ("min", 1000.0, 1010.0000005),
            ("max", 1000.0, 989.9999995),
        )
        for direction, best_value, near_boundary_value in cases:
            with self.subTest(direction=direction):
                benchmarks = BenchmarkSet.from_mapping(
                    {
                        "schema_version": "1.0",
                        "baseline_id": "numeric-best",
                        "synthetic": True,
                        "candidates": [
                            {
                                "id": "numeric-best",
                                "label": "Numeric best",
                                "parameters": {},
                                "metrics": {"objective": best_value, "quality": 0.9},
                            },
                            {
                                "id": "preferred",
                                "label": "Preferred near boundary",
                                "parameters": {},
                                "metrics": {
                                    "objective": near_boundary_value,
                                    "quality": 1.0,
                                },
                            },
                        ],
                    }
                )
                constraints = Constraints.from_mapping(
                    {
                        "min_quality_retention": 0.0,
                        "quality_metric": "quality",
                        "max_values": {},
                        "min_values": {},
                        "objective": {"metric": "objective", "direction": direction},
                        "objective_tolerance_percent": 1.0,
                        "preference_order": ["preferred", "numeric-best"],
                        "frontier_metrics": {
                            "objective": direction,
                            "quality": "max",
                        },
                    }
                )

                passport = build_decision_passport(benchmarks, constraints)

                self.assertEqual(
                    passport["selected_decision"]["candidate_id"],
                    "preferred",
                )
                self.assertEqual(
                    passport["objective"]["selected_runway"]["absolute"],
                    0.0,
                )

    def test_rejected_candidate_missing_objective_is_explicitly_unavailable(self) -> None:
        raw = _benchmark_mapping()
        candidates = raw["candidates"]
        assert isinstance(candidates, list)
        missing = next(candidate for candidate in candidates if candidate["id"] == "q4-generic")
        missing["metrics"].pop("e2e_latency_ms_p95")

        passport = build_decision_passport(BenchmarkSet.from_mapping(raw), _constraints())
        stage = next(stage for stage in passport["ladder"] if stage["candidate_id"] == "q4-generic")

        self.assertIsNone(stage["objective_value"])
        self.assertFalse(stage["eligible"])
        self.assertIn(
            "missing metric e2e_latency_ms_p95",
            stage["constraint_violations"],
        )

    def test_resource_alternative_never_selects_a_rejected_candidate(self) -> None:
        raw = _benchmark_mapping()
        candidates = raw["candidates"]
        assert isinstance(candidates, list)
        for candidate in candidates:
            if candidate["id"] != "q8-generic":
                candidate["metrics"]["quality_score"] = 0.5

        passport = build_decision_passport(BenchmarkSet.from_mapping(raw), _constraints())

        self.assertEqual(passport["selected_decision"]["candidate_id"], "q8-generic")
        self.assertIsNone(passport["resource_alternative"])

    def test_missing_optional_metrics_produce_explicit_noncomparability(self) -> None:
        raw = _benchmark_mapping()
        candidates = raw["candidates"]
        assert isinstance(candidates, list)
        for candidate in candidates:
            metrics = candidate["metrics"]
            assert isinstance(metrics, dict)
            for metric in (
                "generation_tps",
                "model_size_mib",
                "peak_rss_mib",
                "prompt_tps",
                "ttft_ms_p95",
            ):
                metrics.pop(metric)
        constraints = Constraints.from_mapping(
            {
                "min_quality_retention": 0.95,
                "quality_metric": "quality_score",
                "max_values": {},
                "min_values": {"quality_score": 0.8},
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
                    "quality_score": "max",
                },
            }
        )

        passport = build_decision_passport(BenchmarkSet.from_mapping(raw), constraints)

        self.assertIsNone(passport["resource_alternative"])
        for stage in passport["ladder"][1:]:
            delta = stage["delta_from_previous"]
            self.assertEqual(
                {row["metric"] for row in delta["metrics"]},
                {"e2e_latency_ms_p95", "quality_score"},
            )
            self.assertEqual(delta["not_comparable_metrics"], [])

    def test_unrecognized_stage_falls_back_without_copying_nonfinite_parameters(self) -> None:
        raw = _benchmark_mapping()
        candidates = raw["candidates"]
        assert isinstance(candidates, list)
        candidates[0]["parameters"]["configuration"]["attribution_stage"] = math.inf
        candidates[0]["parameters"]["unrelated_nonfinite"] = math.inf
        benchmarks = BenchmarkSet.from_mapping(raw)

        passport = build_decision_passport(benchmarks, _constraints())

        self.assertEqual(passport["ladder"][-1]["candidate_id"], "q4-generic")
        self.assertIsNone(passport["ladder"][-1]["attribution_stage"])
        json.dumps(passport, allow_nan=False)

    def test_domain_rejects_nonfinite_measured_metrics_before_passport_build(self) -> None:
        raw = deepcopy(_benchmark_mapping())
        candidates = raw["candidates"]
        assert isinstance(candidates, list)
        candidates[0]["metrics"]["e2e_latency_ms_p95"] = math.inf

        with self.assertRaisesRegex(ValidationError, "must be finite"):
            BenchmarkSet.from_mapping(raw)

    def test_requires_validated_domain_inputs(self) -> None:
        benchmarks = BenchmarkSet.from_mapping(_benchmark_mapping())
        constraints = _constraints()

        with self.assertRaisesRegex(TypeError, "validated BenchmarkSet"):
            build_decision_passport({}, constraints)
        with self.assertRaisesRegex(TypeError, "validated Constraints"):
            build_decision_passport(benchmarks, {})


if __name__ == "__main__":
    unittest.main()
