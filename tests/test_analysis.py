from __future__ import annotations

import unittest

from paretopilot.analysis import evaluate_constraints, pareto_frontier, recommend
from paretopilot.domain import BenchmarkSet, Constraints, ValidationError


def benchmark_set() -> BenchmarkSet:
    return BenchmarkSet.from_mapping(
        {
            "schema_version": "1.0",
            "baseline_id": "baseline",
            "synthetic": True,
            "candidates": [
                {
                    "id": "baseline",
                    "parameters": {},
                    "metrics": {
                        "quality_score": 1.0,
                        "latency": 100.0,
                        "throughput": 10.0,
                        "peak_rss_mib": 1000.0,
                    },
                },
                {
                    "id": "fast-low-quality",
                    "parameters": {},
                    "metrics": {
                        "quality_score": 0.80,
                        "latency": 40.0,
                        "throughput": 25.0,
                        "peak_rss_mib": 500.0,
                    },
                },
                {
                    "id": "balanced",
                    "parameters": {},
                    "metrics": {
                        "quality_score": 0.97,
                        "latency": 60.0,
                        "throughput": 20.0,
                        "peak_rss_mib": 600.0,
                    },
                },
                {
                    "id": "dominated",
                    "parameters": {},
                    "metrics": {
                        "quality_score": 0.96,
                        "latency": 70.0,
                        "throughput": 18.0,
                        "peak_rss_mib": 700.0,
                    },
                },
            ],
        }
    )


def constraints() -> Constraints:
    return Constraints.from_mapping(
        {
            "min_quality_retention": 0.95,
            "quality_metric": "quality_score",
            "max_values": {"peak_rss_mib": 900},
            "objective": {"metric": "throughput", "direction": "max"},
            "frontier_metrics": {
                "quality_score": "max",
                "latency": "min",
                "throughput": "max",
                "peak_rss_mib": "min",
            },
        }
    )


class AnalysisTests(unittest.TestCase):
    def test_quality_and_memory_constraints_are_explained(self) -> None:
        evaluations = {
            item.candidate.candidate_id: item
            for item in evaluate_constraints(benchmark_set(), constraints())
        }
        self.assertFalse(evaluations["baseline"].eligible)
        self.assertIn("exceeds maximum", evaluations["baseline"].violations[0])
        self.assertFalse(evaluations["fast-low-quality"].eligible)
        self.assertIn("quality retention", evaluations["fast-low-quality"].violations[0])
        self.assertTrue(evaluations["balanced"].eligible)

    def test_pareto_frontier_removes_dominated_candidate(self) -> None:
        data = benchmark_set()
        directions = {
            "quality_score": "max",
            "latency": "min",
            "throughput": "max",
            "peak_rss_mib": "min",
        }
        frontier = pareto_frontier((data.by_id("balanced"), data.by_id("dominated")), directions)
        self.assertEqual([item.candidate_id for item in frontier], ["balanced"])

    def test_recommendation_selects_best_eligible_frontier_candidate(self) -> None:
        result = recommend(benchmark_set(), constraints())
        self.assertEqual(result["selected_id"], "balanced")
        self.assertEqual(result["frontier_ids"], ["balanced"])
        self.assertIn("fast-low-quality", result["rejected"])
        self.assertEqual(result["constraints"]["min_quality_retention"], 0.95)
        self.assertEqual(result["paretopilot_version"], "0.2.0")

    def test_quality_retention_requires_positive_baseline(self) -> None:
        data = BenchmarkSet.from_mapping(
            {
                "schema_version": "1.0",
                "baseline_id": "baseline",
                "synthetic": True,
                "candidates": [
                    candidate
                    for candidate in [
                        {
                            "id": "baseline",
                            "metrics": {
                                "quality_score": 0.0,
                                "latency": 1.0,
                                "throughput": 1.0,
                                "peak_rss_mib": 1.0,
                            },
                        }
                    ]
                ],
            }
        )
        with self.assertRaisesRegex(ValidationError, "greater than zero"):
            evaluate_constraints(data, constraints())

    def test_missing_frontier_metric_rejects_only_that_candidate(self) -> None:
        data = benchmark_set()
        raw = {
            "schema_version": data.schema_version,
            "baseline_id": data.baseline_id,
            "synthetic": True,
            "candidates": [
                {
                    "id": item.candidate_id,
                    "parameters": dict(item.parameters),
                    "metrics": (
                        {k: v for k, v in item.metrics.items() if k != "throughput"}
                        if item.candidate_id == "dominated"
                        else dict(item.metrics)
                    ),
                }
                for item in data.candidates
            ],
        }
        evaluations = {
            item.candidate.candidate_id: item
            for item in evaluate_constraints(BenchmarkSet.from_mapping(raw), constraints())
        }
        self.assertIn("missing metric throughput", evaluations["dominated"].violations)

    def test_objective_tie_break_is_independent_of_frontier_key_order(self) -> None:
        data = BenchmarkSet.from_mapping(
            {
                "schema_version": "1.0",
                "baseline_id": "baseline",
                "synthetic": True,
                "candidates": [
                    {
                        "id": "baseline",
                        "metrics": {
                            "quality_score": 1.0,
                            "throughput": 10.0,
                            "latency": 10.0,
                        },
                    },
                    {
                        "id": "z-candidate",
                        "metrics": {
                            "quality_score": 1.0,
                            "throughput": 20.0,
                            "latency": 5.0,
                        },
                    },
                    {
                        "id": "a-candidate",
                        "metrics": {
                            "quality_score": 0.98,
                            "throughput": 20.0,
                            "latency": 4.0,
                        },
                    },
                ],
            }
        )
        common = {
            "min_quality_retention": 0.95,
            "objective": {"metric": "throughput", "direction": "max"},
        }
        first = Constraints.from_mapping(
            {
                **common,
                "frontier_metrics": {
                    "throughput": "max",
                    "quality_score": "max",
                    "latency": "min",
                },
            }
        )
        second = Constraints.from_mapping(
            {
                **common,
                "frontier_metrics": {
                    "latency": "min",
                    "quality_score": "max",
                    "throughput": "max",
                },
            }
        )

        self.assertEqual(recommend(data, first)["selected_id"], "a-candidate")
        self.assertEqual(recommend(data, second)["selected_id"], "a-candidate")

    def test_max_objective_tolerance_includes_boundary_and_applies_preference(self) -> None:
        data = self._tolerance_benchmarks(
            "throughput", baseline_value=80.0, values=(100.0, 90.0, 89.9)
        )
        configured = self._tolerance_constraints(
            "throughput",
            "max",
            tolerance=10.0,
            preference_order=("boundary", "outside", "best", "baseline"),
        )

        result = recommend(data, configured)

        self.assertEqual(result["selected_id"], "boundary")
        self.assertEqual(result["selection"]["numeric_best_id"], "best")
        self.assertEqual(result["selection"]["numeric_best_value"], 100.0)
        self.assertEqual(result["selection"]["shortlist_ids"], ["best", "boundary"])
        self.assertTrue(result["selection"]["preference_changed_winner"])
        self.assertEqual(result["constraints"]["objective_tolerance_percent"], 10.0)
        self.assertEqual(
            result["constraints"]["preference_order"],
            ["boundary", "outside", "best", "baseline"],
        )

    def test_min_objective_tolerance_includes_boundary(self) -> None:
        data = self._tolerance_benchmarks(
            "latency", baseline_value=130.0, values=(100.0, 110.0, 110.1)
        )
        configured = self._tolerance_constraints(
            "latency",
            "min",
            tolerance=10.0,
            preference_order=("boundary", "outside", "best", "baseline"),
        )

        result = recommend(data, configured)

        self.assertEqual(result["selected_id"], "boundary")
        self.assertEqual(result["selection"]["shortlist_ids"], ["best", "boundary"])

    def test_tolerance_without_preference_uses_deterministic_candidate_id(self) -> None:
        data = BenchmarkSet.from_mapping(
            {
                "schema_version": "1.0",
                "baseline_id": "baseline",
                "synthetic": True,
                "candidates": [
                    {
                        "id": "z-best",
                        "metrics": {"quality_score": 0.97, "throughput": 100.0},
                    },
                    {
                        "id": "baseline",
                        "metrics": {"quality_score": 1.0, "throughput": 80.0},
                    },
                    {
                        "id": "a-near",
                        "metrics": {"quality_score": 0.98, "throughput": 95.0},
                    },
                ],
            }
        )
        configured = self._tolerance_constraints("throughput", "max", tolerance=10.0)

        result = recommend(data, configured)

        self.assertEqual(result["selected_id"], "a-near")
        self.assertEqual(result["selection"]["numeric_best_id"], "z-best")
        self.assertEqual(result["selection"]["shortlist_ids"], ["a-near", "z-best"])
        self.assertFalse(result["selection"]["preference_order_applied"])
        self.assertFalse(result["selection"]["preference_changed_winner"])

    def test_zero_objective_value_has_a_zero_width_relative_tolerance(self) -> None:
        cases = (
            ("latency", "min", 5.0, (0.0, 0.1, 1.0)),
            ("throughput", "max", -5.0, (0.0, -0.1, -1.0)),
        )
        for metric, direction, baseline_value, values in cases:
            with self.subTest(direction=direction):
                data = self._tolerance_benchmarks(
                    metric,
                    baseline_value=baseline_value,
                    values=values,
                )
                configured = self._tolerance_constraints(
                    metric,
                    direction,
                    tolerance=100.0,
                    preference_order=("outside", "boundary", "best", "baseline"),
                )

                result = recommend(data, configured)

                self.assertEqual(result["selected_id"], "best")
                self.assertEqual(result["selection"]["shortlist_ids"], ["best"])
                self.assertFalse(result["selection"]["preference_changed_winner"])

    def test_preference_order_must_cover_every_benchmark_candidate(self) -> None:
        data = self._tolerance_benchmarks(
            "throughput", baseline_value=80.0, values=(100.0, 95.0, 89.0)
        )
        configured = self._tolerance_constraints(
            "throughput",
            "max",
            tolerance=10.0,
            preference_order=("best", "boundary", "unknown"),
        )

        with self.assertRaisesRegex(
            ValidationError,
            r"cover exactly.*missing: baseline, outside.*unknown: unknown",
        ):
            recommend(data, configured)

    @staticmethod
    def _tolerance_benchmarks(
        objective_metric: str,
        *,
        baseline_value: float,
        values: tuple[float, float, float],
    ) -> BenchmarkSet:
        objective_values = dict(zip(("best", "boundary", "outside"), values, strict=True))
        candidates = [
            {
                "id": "baseline",
                "metrics": {
                    "quality_score": 1.0,
                    objective_metric: baseline_value,
                },
            }
        ]
        for index, candidate_id in enumerate(("best", "boundary", "outside")):
            candidates.append(
                {
                    "id": candidate_id,
                    "metrics": {
                        # The quality tradeoff keeps each objective candidate on the frontier.
                        "quality_score": 0.97 + (index * 0.01),
                        objective_metric: objective_values[candidate_id],
                    },
                }
            )
        return BenchmarkSet.from_mapping(
            {
                "schema_version": "1.0",
                "baseline_id": "baseline",
                "synthetic": True,
                "candidates": candidates,
            }
        )

    @staticmethod
    def _tolerance_constraints(
        objective_metric: str,
        direction: str,
        *,
        tolerance: float,
        preference_order: tuple[str, ...] = (),
    ) -> Constraints:
        return Constraints.from_mapping(
            {
                "min_quality_retention": 0.0,
                "objective": {"metric": objective_metric, "direction": direction},
                "frontier_metrics": {
                    objective_metric: direction,
                    "quality_score": "max",
                },
                "objective_tolerance_percent": tolerance,
                "preference_order": list(preference_order),
            }
        )


if __name__ == "__main__":
    unittest.main()
