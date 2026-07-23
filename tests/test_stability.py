from __future__ import annotations

from copy import deepcopy
import unittest

from paretopilot.domain import BenchmarkSet, ValidationError
from paretopilot.stability import summarize_stability, validate_stability_summary


def _benchmark(
    *,
    baseline_latency: float,
    baseline_throughput: float,
    optimized_latency: float,
    optimized_throughput: float,
    unstable_latency: float,
    unstable_throughput: float,
    baseline_id: str = "baseline",
    synthetic: bool = False,
    parameter_marker: str = "same",
) -> BenchmarkSet:
    return BenchmarkSet.from_mapping(
        {
            "schema_version": "1.0",
            "baseline_id": baseline_id,
            "synthetic": synthetic,
            "metadata": {},
            "candidates": [
                {
                    "id": "baseline",
                    "label": "Baseline",
                    "parameters": {"marker": parameter_marker},
                    "metrics": {
                        "latency": baseline_latency,
                        "throughput": baseline_throughput,
                    },
                },
                {
                    "id": "optimized",
                    "label": "Optimized",
                    "parameters": {"marker": parameter_marker},
                    "metrics": {
                        "latency": optimized_latency,
                        "throughput": optimized_throughput,
                    },
                },
                {
                    "id": "unstable",
                    "label": "Unstable",
                    "parameters": {"marker": parameter_marker},
                    "metrics": {
                        "latency": unstable_latency,
                        "throughput": unstable_throughput,
                    },
                },
            ],
        }
    )


def _passes() -> tuple[BenchmarkSet, BenchmarkSet]:
    return (
        _benchmark(
            baseline_latency=100.0,
            baseline_throughput=50.0,
            optimized_latency=80.0,
            optimized_throughput=60.0,
            unstable_latency=90.0,
            unstable_throughput=55.0,
        ),
        _benchmark(
            baseline_latency=102.0,
            baseline_throughput=49.0,
            optimized_latency=81.0,
            optimized_throughput=61.0,
            unstable_latency=110.0,
            unstable_throughput=45.0,
        ),
    )


class StabilityTests(unittest.TestCase):
    def test_summarizes_consistent_mixed_and_no_change_without_significance(self) -> None:
        summary = summarize_stability(
            _passes(),
            metric_directions={"throughput": "max", "latency": "min"},
            pass_labels=("A", "B"),
        )

        validate_stability_summary(summary)
        rows = {(row["candidate_id"], row["metric"]): row for row in summary["rows"]}
        optimized_latency = rows[("optimized", "latency")]
        unstable_latency = rows[("unstable", "latency")]
        baseline_throughput = rows[("baseline", "throughput")]

        self.assertEqual(optimized_latency["consistency"], "consistent")
        self.assertEqual(optimized_latency["overall_direction"], "improved")
        self.assertAlmostEqual(
            optimized_latency["relative_spread_percent"],
            (1.0 / 80.5) * 100.0,
        )
        self.assertEqual(
            [value["comparison"] for value in unstable_latency["pass_values"]],
            ["improved", "regressed"],
        )
        self.assertEqual(unstable_latency["consistency"], "mixed")
        self.assertEqual(unstable_latency["overall_direction"], "mixed")
        self.assertEqual(baseline_throughput["consistency"], "no change")
        self.assertEqual(baseline_throughput["overall_direction"], "no change")
        self.assertNotIn("significant", summary["method"])
        self.assertEqual(summary["pass_labels"], ["A", "B"])
        self.assertEqual(
            [(row["candidate_id"], row["metric"]) for row in summary["rows"]],
            sorted((row["candidate_id"], row["metric"]) for row in summary["rows"]),
        )

    def test_zero_baseline_delta_is_reported_honestly(self) -> None:
        first = _benchmark(
            baseline_latency=0.0,
            baseline_throughput=0.0,
            optimized_latency=1.0,
            optimized_throughput=1.0,
            unstable_latency=0.0,
            unstable_throughput=0.0,
        )
        second = _benchmark(
            baseline_latency=0.0,
            baseline_throughput=0.0,
            optimized_latency=2.0,
            optimized_throughput=2.0,
            unstable_latency=0.0,
            unstable_throughput=0.0,
        )

        summary = summarize_stability(
            (first, second),
            metric_directions={"latency": "min"},
        )
        optimized = next(row for row in summary["rows"] if row["candidate_id"] == "optimized")

        self.assertEqual(
            [value["relative_delta_percent"] for value in optimized["pass_values"]],
            [None, None],
        )
        self.assertEqual(optimized["consistency"], "consistent")
        self.assertEqual(optimized["overall_direction"], "regressed")

    def test_rejects_incompatible_passes_and_metric_contracts(self) -> None:
        first, second = _passes()
        different_baseline = BenchmarkSet(
            schema_version=second.schema_version,
            baseline_id="optimized",
            candidates=second.candidates,
            metadata=second.metadata,
            synthetic=second.synthetic,
        )
        different_candidates = BenchmarkSet(
            schema_version=second.schema_version,
            baseline_id=second.baseline_id,
            candidates=second.candidates[:-1],
            metadata=second.metadata,
            synthetic=second.synthetic,
        )
        synthetic = BenchmarkSet(
            schema_version=second.schema_version,
            baseline_id=second.baseline_id,
            candidates=second.candidates,
            metadata=second.metadata,
            synthetic=True,
        )
        different_parameters = _benchmark(
            baseline_latency=102.0,
            baseline_throughput=49.0,
            optimized_latency=81.0,
            optimized_throughput=61.0,
            unstable_latency=110.0,
            unstable_throughput=45.0,
            parameter_marker="changed",
        )
        scenarios = [
            ((first,), {"latency": "min"}, "from 2 to 8"),
            ((first, different_baseline), {"latency": "min"}, "baseline_id"),
            ((first, different_candidates), {"latency": "min"}, "candidate ids"),
            ((first, synthetic), {"latency": "min"}, "synthetic flag"),
            ((first, different_parameters), {"latency": "min"}, "parameters"),
            ((first, second), {"missing": "min"}, "missing metrics"),
            ((first, second), {"latency": "sideways"}, "must be 'min' or 'max'"),
        ]
        for passes, directions, message in scenarios:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValidationError, message):
                    summarize_stability(passes, metric_directions=directions)

    def test_serialized_summary_validator_recomputes_labels_and_spread(self) -> None:
        summary = summarize_stability(
            _passes(),
            metric_directions={"latency": "min"},
        )
        scenarios = []

        label = deepcopy(summary)
        label["rows"][1]["consistency"] = "mixed"
        scenarios.append((label, "consistency does not match"))

        spread = deepcopy(summary)
        spread["rows"][1]["relative_spread_percent"] += 1.0
        scenarios.append((spread, "relative_spread_percent does not match"))

        comparison = deepcopy(summary)
        comparison["rows"][1]["pass_values"][0]["comparison"] = "regressed"
        scenarios.append((comparison, "comparison does not match"))

        unknown = deepcopy(summary)
        unknown["rows"][1]["statistically_significant"] = True
        scenarios.append((unknown, "unknown fields"))

        contradictory_baseline = deepcopy(summary)
        optimized_latency = next(
            row
            for row in contradictory_baseline["rows"]
            if row["candidate_id"] == "optimized" and row["metric"] == "latency"
        )
        optimized_latency["pass_values"][0]["baseline_value"] = 1000.0
        optimized_latency["pass_values"][0]["relative_delta_percent"] = 92.0
        scenarios.append((contradictory_baseline, "inconsistent across candidates"))

        inputs = summarize_stability(
            _passes(),
            metric_directions={"latency": "min"},
            pass_labels=("A", "B"),
            input_fingerprints={"A": "a" * 64, "B": "b" * 64},
        )
        validate_stability_summary(inputs, require_input_fingerprints=True)
        missing_inputs = deepcopy(inputs)
        missing_inputs["input_fingerprints"] = {}
        with self.assertRaisesRegex(ValidationError, "required"):
            validate_stability_summary(
                missing_inputs,
                require_input_fingerprints=True,
            )

        for payload, message in scenarios:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValidationError, message):
                    validate_stability_summary(payload)


if __name__ == "__main__":
    unittest.main()
