"""Validated domain objects for benchmark evidence and selection constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence


Direction = str


class ValidationError(ValueError):
    """Raised when benchmark evidence does not satisfy the canonical contract."""


def _reject_unknown_fields(
    raw: Mapping[str, Any], *, allowed: set[str], context: str
) -> None:
    unknown = sorted(str(name) for name in set(raw) - allowed)
    if unknown:
        raise ValidationError(f"{context} contains unknown fields: {', '.join(unknown)}")


def _finite_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValidationError(f"{field_name} must be finite")
    return converted


def _direction(value: Any, *, field_name: str) -> Direction:
    if value not in {"min", "max"}:
        raise ValidationError(f"{field_name} must be 'min' or 'max'")
    return str(value)


@dataclass(frozen=True)
class Candidate:
    """One fully described inference configuration and its aggregate metrics."""

    candidate_id: str
    label: str
    parameters: Mapping[str, Any]
    metrics: Mapping[str, float]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Candidate":
        _reject_unknown_fields(
            raw,
            allowed={"id", "label", "parameters", "metrics"},
            context="candidate",
        )
        candidate_id = raw.get("id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValidationError("candidate.id must be a non-empty string")

        label = raw.get("label", candidate_id)
        if not isinstance(label, str) or not label.strip():
            raise ValidationError(f"candidate {candidate_id!r} label must be a non-empty string")

        parameters = raw.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValidationError(f"candidate {candidate_id!r} parameters must be an object")

        raw_metrics = raw.get("metrics")
        if not isinstance(raw_metrics, Mapping) or not raw_metrics:
            raise ValidationError(f"candidate {candidate_id!r} metrics must be a non-empty object")

        metrics: dict[str, float] = {}
        for name, value in raw_metrics.items():
            if not isinstance(name, str) or not name:
                raise ValidationError(f"candidate {candidate_id!r} contains an invalid metric name")
            metrics[name] = _finite_number(
                value,
                field_name=f"candidate {candidate_id!r} metric {name!r}",
            )

        return cls(
            candidate_id=candidate_id,
            label=label,
            parameters=dict(parameters),
            metrics=metrics,
        )


@dataclass(frozen=True)
class BenchmarkSet:
    """A set of candidates measured under one controlled benchmark context."""

    schema_version: str
    baseline_id: str
    candidates: tuple[Candidate, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    synthetic: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BenchmarkSet":
        _reject_unknown_fields(
            raw,
            allowed={"schema_version", "baseline_id", "candidates", "metadata", "synthetic"},
            context="benchmark set",
        )
        schema_version = raw.get("schema_version")
        if schema_version != "1.0":
            raise ValidationError("schema_version must currently be '1.0'")

        baseline_id = raw.get("baseline_id")
        if not isinstance(baseline_id, str) or not baseline_id.strip():
            raise ValidationError("baseline_id must be a non-empty string")

        raw_candidates = raw.get("candidates")
        if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
            raise ValidationError("candidates must be an array")
        candidates_list: list[Candidate] = []
        for index, item in enumerate(raw_candidates):
            if not isinstance(item, Mapping):
                raise ValidationError(f"candidates[{index}] must be an object")
            candidates_list.append(Candidate.from_mapping(item))
        candidates = tuple(candidates_list)
        if not candidates:
            raise ValidationError("candidates must contain at least one candidate")

        ids = [candidate.candidate_id for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise ValidationError("candidate ids must be unique")
        if baseline_id not in ids:
            raise ValidationError("baseline_id must refer to one candidate")

        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValidationError("metadata must be an object")

        if "synthetic" not in raw:
            raise ValidationError("synthetic must be explicitly set to true or false")
        synthetic = raw["synthetic"]
        if not isinstance(synthetic, bool):
            raise ValidationError("synthetic must be a boolean")

        return cls(
            schema_version=schema_version,
            baseline_id=baseline_id,
            candidates=candidates,
            metadata=dict(metadata),
            synthetic=synthetic,
        )

    @property
    def baseline(self) -> Candidate:
        return self.by_id(self.baseline_id)

    def by_id(self, candidate_id: str) -> Candidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise KeyError(candidate_id)


@dataclass(frozen=True)
class Objective:
    metric: str
    direction: Direction

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Objective":
        _reject_unknown_fields(
            raw,
            allowed={"metric", "direction"},
            context="objective",
        )
        metric = raw.get("metric")
        if not isinstance(metric, str) or not metric.strip():
            raise ValidationError("objective.metric must be a non-empty string")
        return cls(
            metric=metric,
            direction=_direction(raw.get("direction"), field_name="objective.direction"),
        )


@dataclass(frozen=True)
class Constraints:
    """Quality/resource gates and the objective used to select a winner."""

    min_quality_retention: float
    quality_metric: str
    max_values: Mapping[str, float]
    min_values: Mapping[str, float]
    objective: Objective
    frontier_metrics: Mapping[str, Direction]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Constraints":
        _reject_unknown_fields(
            raw,
            allowed={
                "min_quality_retention",
                "quality_metric",
                "max_values",
                "min_values",
                "objective",
                "frontier_metrics",
            },
            context="constraints",
        )
        min_quality_retention = _finite_number(
            raw.get("min_quality_retention", 1.0),
            field_name="min_quality_retention",
        )
        if not 0.0 <= min_quality_retention <= 1.0:
            raise ValidationError("min_quality_retention must be between 0 and 1")

        quality_metric = raw.get("quality_metric", "quality_score")
        if not isinstance(quality_metric, str) or not quality_metric.strip():
            raise ValidationError("quality_metric must be a non-empty string")

        max_values = cls._metric_thresholds(raw.get("max_values", {}), "max_values")
        min_values = cls._metric_thresholds(raw.get("min_values", {}), "min_values")
        for metric in sorted(set(max_values) & set(min_values)):
            if min_values[metric] > max_values[metric]:
                raise ValidationError(
                    f"minimum for {metric!r} cannot exceed its maximum"
                )

        raw_objective = raw.get("objective")
        if not isinstance(raw_objective, Mapping):
            raise ValidationError("objective must be an object")
        objective = Objective.from_mapping(raw_objective)

        raw_frontier = raw.get("frontier_metrics")
        if not isinstance(raw_frontier, Mapping) or not raw_frontier:
            raise ValidationError("frontier_metrics must be a non-empty object")
        frontier_metrics = {
            str(name): _direction(direction, field_name=f"frontier_metrics.{name}")
            for name, direction in raw_frontier.items()
            if isinstance(name, str) and name
        }
        if len(frontier_metrics) != len(raw_frontier):
            raise ValidationError("frontier_metrics contains an invalid metric name")
        if objective.metric not in frontier_metrics:
            raise ValidationError("objective.metric must also appear in frontier_metrics")
        if frontier_metrics[objective.metric] != objective.direction:
            raise ValidationError(
                "objective.direction must match its direction in frontier_metrics"
            )
        if (
            quality_metric in frontier_metrics
            and frontier_metrics[quality_metric] != "max"
        ):
            raise ValidationError(
                "quality_metric must use direction 'max' in frontier_metrics"
            )

        return cls(
            min_quality_retention=min_quality_retention,
            quality_metric=quality_metric,
            max_values=max_values,
            min_values=min_values,
            objective=objective,
            frontier_metrics=frontier_metrics,
        )

    @staticmethod
    def _metric_thresholds(raw: Any, field_name: str) -> Mapping[str, float]:
        if not isinstance(raw, Mapping):
            raise ValidationError(f"{field_name} must be an object")
        thresholds: dict[str, float] = {}
        for name, value in raw.items():
            if not isinstance(name, str) or not name:
                raise ValidationError(f"{field_name} contains an invalid metric name")
            thresholds[name] = _finite_number(value, field_name=f"{field_name}.{name}")
        return thresholds
