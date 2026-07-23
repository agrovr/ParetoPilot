from __future__ import annotations

import unittest

from paretopilot.domain import BenchmarkSet, Constraints, ValidationError


def candidate(candidate_id: str, **metrics: float) -> dict:
    return {"id": candidate_id, "parameters": {}, "metrics": metrics}


class BenchmarkSetTests(unittest.TestCase):
    def test_requires_existing_baseline(self) -> None:
        raw = {
            "schema_version": "1.0",
            "baseline_id": "missing",
            "synthetic": True,
            "candidates": [candidate("actual", quality_score=1.0)],
        }
        with self.assertRaisesRegex(ValidationError, "baseline_id"):
            BenchmarkSet.from_mapping(raw)

    def test_rejects_duplicate_candidate_ids(self) -> None:
        raw = {
            "schema_version": "1.0",
            "baseline_id": "same",
            "synthetic": True,
            "candidates": [
                candidate("same", quality_score=1.0),
                candidate("same", quality_score=0.9),
            ],
        }
        with self.assertRaisesRegex(ValidationError, "unique"):
            BenchmarkSet.from_mapping(raw)

    def test_rejects_non_object_candidate(self) -> None:
        raw = {
            "schema_version": "1.0",
            "baseline_id": "baseline",
            "synthetic": True,
            "candidates": ["baseline"],
        }
        with self.assertRaisesRegex(ValidationError, r"candidates\[0\] must be an object"):
            BenchmarkSet.from_mapping(raw)

    def test_requires_explicit_synthetic_label(self) -> None:
        raw = {
            "schema_version": "1.0",
            "baseline_id": "baseline",
            "candidates": [candidate("baseline", quality_score=1.0)],
        }
        with self.assertRaisesRegex(ValidationError, "must be explicitly set"):
            BenchmarkSet.from_mapping(raw)

    def test_rejects_unknown_top_level_field(self) -> None:
        raw = {
            "schema_version": "1.0",
            "baseline_id": "baseline",
            "synthetic": True,
            "synthethic": True,
            "candidates": [candidate("baseline", quality_score=1.0)],
        }
        with self.assertRaisesRegex(ValidationError, "unknown fields: synthethic"):
            BenchmarkSet.from_mapping(raw)

    def test_constraints_validate_retention_range(self) -> None:
        raw = {
            "min_quality_retention": 1.1,
            "objective": {"metric": "latency", "direction": "min"},
            "frontier_metrics": {"latency": "min"},
        }
        with self.assertRaisesRegex(ValidationError, "between 0 and 1"):
            Constraints.from_mapping(raw)

    def test_constraints_reject_conflicting_objective_direction(self) -> None:
        raw = {
            "objective": {"metric": "latency", "direction": "min"},
            "frontier_metrics": {"latency": "max"},
        }
        with self.assertRaisesRegex(ValidationError, "must match"):
            Constraints.from_mapping(raw)

    def test_constraints_reject_impossible_bounds(self) -> None:
        raw = {
            "min_values": {"throughput": 20},
            "max_values": {"throughput": 10},
            "objective": {"metric": "throughput", "direction": "max"},
            "frontier_metrics": {"throughput": "max"},
        }
        with self.assertRaisesRegex(ValidationError, "cannot exceed"):
            Constraints.from_mapping(raw)

    def test_constraints_reject_minimized_quality_metric(self) -> None:
        raw = {
            "quality_metric": "quality_score",
            "objective": {"metric": "latency", "direction": "min"},
            "frontier_metrics": {
                "latency": "min",
                "quality_score": "min",
            },
        }
        with self.assertRaisesRegex(ValidationError, "must use direction 'max'"):
            Constraints.from_mapping(raw)

    def test_constraints_default_to_strict_objective_selection(self) -> None:
        parsed = Constraints.from_mapping(
            {
                "objective": {"metric": "latency", "direction": "min"},
                "frontier_metrics": {"latency": "min"},
            }
        )

        self.assertEqual(parsed.objective_tolerance_percent, 0.0)
        self.assertEqual(parsed.preference_order, ())

    def test_constraints_accept_objective_tolerance_and_preference_order(self) -> None:
        parsed = Constraints.from_mapping(
            {
                "objective": {"metric": "latency", "direction": "min"},
                "frontier_metrics": {"latency": "min"},
                "objective_tolerance_percent": 7.5,
                "preference_order": ["low-memory", "fastest"],
            }
        )

        self.assertEqual(parsed.objective_tolerance_percent, 7.5)
        self.assertEqual(parsed.preference_order, ("low-memory", "fastest"))

    def test_constraints_reject_invalid_objective_tolerances(self) -> None:
        for value in (-0.1, 100.1, float("inf"), float("nan"), True):
            with self.subTest(value=value):
                raw = {
                    "objective": {"metric": "latency", "direction": "min"},
                    "frontier_metrics": {"latency": "min"},
                    "objective_tolerance_percent": value,
                }
                with self.assertRaises(ValidationError):
                    Constraints.from_mapping(raw)

    def test_constraints_reject_invalid_preference_order(self) -> None:
        invalid_orders = (
            "candidate-a",
            ["candidate-a", "candidate-a"],
            ["candidate-a", ""],
            ["candidate-a", 2],
        )
        for value in invalid_orders:
            with self.subTest(value=value):
                raw = {
                    "objective": {"metric": "latency", "direction": "min"},
                    "frontier_metrics": {"latency": "min"},
                    "preference_order": value,
                }
                with self.assertRaises(ValidationError):
                    Constraints.from_mapping(raw)


if __name__ == "__main__":
    unittest.main()
