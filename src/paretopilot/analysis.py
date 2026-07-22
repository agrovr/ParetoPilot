"""Constraint evaluation, Pareto analysis, and deterministic recommendation logic."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from paretopilot import __version__
from paretopilot.domain import BenchmarkSet, Candidate, Constraints, Direction, ValidationError


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: Candidate
    violations: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return not self.violations


def evaluate_constraints(
    benchmarks: BenchmarkSet,
    constraints: Constraints,
) -> tuple[CandidateEvaluation, ...]:
    baseline = benchmarks.baseline
    if constraints.quality_metric not in baseline.metrics:
        raise ValidationError(
            f"baseline is missing quality metric {constraints.quality_metric!r}"
        )

    baseline_quality = baseline.metrics[constraints.quality_metric]
    if baseline_quality <= 0:
        raise ValidationError(
            f"baseline quality metric {constraints.quality_metric!r} must be greater than zero"
        )
    quality_floor = baseline_quality * constraints.min_quality_retention

    required_metrics = {
        constraints.quality_metric,
        constraints.objective.metric,
        *constraints.frontier_metrics,
        *constraints.max_values,
        *constraints.min_values,
    }

    evaluations: list[CandidateEvaluation] = []
    for candidate in benchmarks.candidates:
        violations = [
            f"missing metric {metric}"
            for metric in sorted(required_metrics - set(candidate.metrics))
        ]
        quality = candidate.metrics.get(constraints.quality_metric)
        if quality is not None and quality < quality_floor and not math.isclose(
            quality, quality_floor
        ):
            retention = quality / baseline_quality
            violations.append(
                f"quality retention {retention:.4f} is below {constraints.min_quality_retention:.4f}"
            )

        for metric, maximum in constraints.max_values.items():
            value = candidate.metrics.get(metric)
            if value is not None and value > maximum and not math.isclose(value, maximum):
                violations.append(f"{metric}={value:g} exceeds maximum {maximum:g}")

        for metric, minimum in constraints.min_values.items():
            value = candidate.metrics.get(metric)
            if value is not None and value < minimum and not math.isclose(value, minimum):
                violations.append(f"{metric}={value:g} is below minimum {minimum:g}")

        evaluations.append(CandidateEvaluation(candidate, tuple(violations)))
    return tuple(evaluations)


def dominates(
    left: Candidate,
    right: Candidate,
    directions: Mapping[str, Direction],
) -> bool:
    """Return True when left is no worse everywhere and strictly better somewhere."""

    strictly_better = False
    for metric, direction in directions.items():
        if metric not in left.metrics:
            raise ValidationError(
                f"candidate {left.candidate_id!r} is missing frontier metric {metric!r}"
            )
        if metric not in right.metrics:
            raise ValidationError(
                f"candidate {right.candidate_id!r} is missing frontier metric {metric!r}"
            )
        left_value = left.metrics[metric]
        right_value = right.metrics[metric]

        equal = math.isclose(left_value, right_value, rel_tol=1e-9, abs_tol=1e-12)
        if direction == "min":
            if left_value > right_value and not equal:
                return False
            strictly_better = strictly_better or (left_value < right_value and not equal)
        elif direction == "max":
            if left_value < right_value and not equal:
                return False
            strictly_better = strictly_better or (left_value > right_value and not equal)
        else:
            raise ValidationError(f"unsupported direction {direction!r} for {metric!r}")
    return strictly_better


def pareto_frontier(
    candidates: Iterable[Candidate],
    directions: Mapping[str, Direction],
) -> tuple[Candidate, ...]:
    items = tuple(candidates)
    for candidate in items:
        missing = sorted(set(directions) - set(candidate.metrics))
        if missing:
            raise ValidationError(
                f"candidate {candidate.candidate_id!r} is missing frontier metrics: "
                + ", ".join(missing)
            )

    frontier = [
        candidate
        for candidate in items
        if not any(
            dominates(other, candidate, directions)
            for other in items
            if other.candidate_id != candidate.candidate_id
        )
    ]
    return tuple(sorted(frontier, key=lambda candidate: candidate.candidate_id))


def _selection_key(candidate: Candidate, constraints: Constraints) -> tuple[Any, ...]:
    value = candidate.metrics[constraints.objective.metric]
    directed_value = value if constraints.objective.direction == "min" else -value
    return directed_value, candidate.candidate_id


def _deltas(selected: Candidate, baseline: Candidate) -> Mapping[str, Mapping[str, float | None]]:
    deltas: dict[str, Mapping[str, float | None]] = {}
    for metric in sorted(set(selected.metrics) & set(baseline.metrics)):
        baseline_value = baseline.metrics[metric]
        selected_value = selected.metrics[metric]
        absolute = selected_value - baseline_value
        percent = None if baseline_value == 0 else (absolute / baseline_value) * 100.0
        deltas[metric] = {
            "baseline": baseline_value,
            "selected": selected_value,
            "absolute": absolute,
            "percent": percent,
        }
    return deltas


def recommend(benchmarks: BenchmarkSet, constraints: Constraints) -> Mapping[str, Any]:
    evaluations = evaluate_constraints(benchmarks, constraints)
    eligible = tuple(item.candidate for item in evaluations if item.eligible)
    if not eligible:
        violation_summary = {
            item.candidate.candidate_id: list(item.violations) for item in evaluations
        }
        raise ValidationError(f"no candidate satisfies the constraints: {violation_summary}")

    frontier = pareto_frontier(eligible, constraints.frontier_metrics)
    selected = min(frontier, key=lambda candidate: _selection_key(candidate, constraints))

    return {
        "schema_version": "1.0",
        "paretopilot_version": __version__,
        "source_schema_version": benchmarks.schema_version,
        "synthetic_source": benchmarks.synthetic,
        "source_metadata": dict(benchmarks.metadata),
        "baseline_id": benchmarks.baseline_id,
        "selected_id": selected.candidate_id,
        "objective": {
            "metric": constraints.objective.metric,
            "direction": constraints.objective.direction,
        },
        "constraints": {
            "min_quality_retention": constraints.min_quality_retention,
            "quality_metric": constraints.quality_metric,
            "max_values": dict(sorted(constraints.max_values.items())),
            "min_values": dict(sorted(constraints.min_values.items())),
            "objective": {
                "metric": constraints.objective.metric,
                "direction": constraints.objective.direction,
            },
            "frontier_metrics": dict(sorted(constraints.frontier_metrics.items())),
        },
        "eligible_ids": sorted(candidate.candidate_id for candidate in eligible),
        "frontier_ids": [candidate.candidate_id for candidate in frontier],
        "rejected": {
            item.candidate.candidate_id: list(item.violations)
            for item in evaluations
            if not item.eligible
        },
        "selected": {
            "id": selected.candidate_id,
            "label": selected.label,
            "parameters": dict(selected.parameters),
            "metrics": dict(selected.metrics),
        },
        "deltas_vs_baseline": _deltas(selected, benchmarks.baseline),
    }
