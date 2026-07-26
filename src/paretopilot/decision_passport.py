"""Deterministic, supplementary context for one ParetoPilot decision.

The decision passport does not participate in candidate selection.  It
recomputes the canonical recommendation, then describes the supplied evidence,
the objective boundary, the ordered optimization ladder, and measured
alternatives.  No timestamp or inferred hardware claim is added, so callers
can hash and archive the result independently of the locked recommendation.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from paretopilot import __version__
from paretopilot.analysis import recommend
from paretopilot.domain import BenchmarkSet, Candidate, Constraints, ValidationError


__all__ = ["build_decision_passport"]


_ATTRIBUTION_ORDER = {
    "reference": 0,
    "quantization": 1,
    "arm-kernel": 2,
    "runtime-tuning": 3,
}
_ARM64_ARCHITECTURES = frozenset({"arm64", "aarch64"})
_RESOURCE_METRICS = ("peak_rss_mib", "model_size_mib")
_KNOWN_METRIC_DIRECTIONS = {
    "e2e_latency_ms_p95": "min",
    "generation_tokens_per_second": "max",
    "generation_tps": "max",
    "model_size_mib": "min",
    "peak_rss_mib": "min",
    "prompt_tokens_per_second": "max",
    "prompt_tps": "max",
    "quality_score": "max",
    "requests_per_second": "max",
    "ttft_ms_p50": "min",
    "ttft_ms_p95": "min",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_CURRENT_BOUNDARY_CAVEAT = (
    "This result applies only to the supplied benchmark, model, runner, and workload. "
    "It does not predict performance, energy use, or cost on other systems, and it "
    "does not establish statistical significance."
)
_SYNTHETIC_BOUNDARY_CAVEAT = (
    "This is a software example, not measured Arm64 evidence. It does not predict "
    "deployment performance or establish statistical significance."
)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


def _identifier(value: object) -> str | None:
    text = _text(value)
    if text is not None:
        return text
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return str(value)
    return None


def _positive_integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _sha256(value: object) -> str | None:
    text = _text(value)
    if text is None or _SHA256_PATTERN.fullmatch(text) is None:
        return None
    return text.lower()


def _required_text(
    value: object,
    *,
    field: str,
    issues: list[str],
) -> str | None:
    normalized = _text(value)
    if normalized is None:
        issues.append(f"{field} is missing or invalid")
    return normalized


def _provenance(benchmarks: BenchmarkSet) -> tuple[str, dict[str, Any]]:
    """Return an evidence grade and a bounded provenance mapping."""

    metadata = _mapping(benchmarks.metadata)
    source = _mapping(metadata.get("source"))
    runner = _mapping(source.get("runner"))
    runtime = _mapping(metadata.get("runtime"))
    model = _mapping(metadata.get("model_family"))
    evaluation_suite = _mapping(metadata.get("evaluation_suite"))
    issues: list[str] = []

    raw_architecture = _required_text(
        runner.get("architecture"),
        field="metadata.source.runner.architecture",
        issues=issues,
    )
    normalized_architecture = raw_architecture.casefold() if raw_architecture is not None else None
    architecture = (
        "arm64" if normalized_architecture in _ARM64_ARCHITECTURES else normalized_architecture
    )
    if normalized_architecture is not None and normalized_architecture not in _ARM64_ARCHITECTURES:
        issues.append("metadata.source.runner.architecture is not a recognized Arm64 architecture")

    cpu = _required_text(
        runner.get("cpu"),
        field="metadata.source.runner.cpu",
        issues=issues,
    )
    operating_system = _required_text(
        runner.get("os"),
        field="metadata.source.runner.os",
        issues=issues,
    )
    cpu_count = _positive_integer(runner.get("cpu_count"))
    if cpu_count is None:
        issues.append("metadata.source.runner.cpu_count is missing or invalid")

    run_id = _identifier(source.get("run_id"))
    if run_id is None:
        issues.append("metadata.source.run_id is missing or invalid")
    run_attempt = _positive_integer(source.get("run_attempt"))
    if run_attempt is None:
        issues.append("metadata.source.run_attempt is missing or invalid")

    source_repository = _required_text(
        source.get("repository"),
        field="metadata.source.repository",
        issues=issues,
    )
    source_revision = _required_text(
        source.get("revision"),
        field="metadata.source.revision",
        issues=issues,
    )
    source_workflow = _required_text(
        source.get("workflow"),
        field="metadata.source.workflow",
        issues=issues,
    )

    runtime_name = _required_text(
        runtime.get("name"),
        field="metadata.runtime.name",
        issues=issues,
    )
    runtime_repository = _required_text(
        runtime.get("repository"),
        field="metadata.runtime.repository",
        issues=issues,
    )
    runtime_revision = _required_text(
        runtime.get("revision"),
        field="metadata.runtime.revision",
        issues=issues,
    )

    model_name = _required_text(
        model.get("name"),
        field="metadata.model_family.name",
        issues=issues,
    )
    model_repository = _required_text(
        model.get("repository"),
        field="metadata.model_family.repository",
        issues=issues,
    )
    model_revision = _required_text(
        model.get("revision"),
        field="metadata.model_family.revision",
        issues=issues,
    )

    evaluation_id = _required_text(
        evaluation_suite.get("id"),
        field="metadata.evaluation_suite.id",
        issues=issues,
    )
    evaluation_sha256 = _sha256(evaluation_suite.get("sha256"))
    if evaluation_sha256 is None:
        issues.append("metadata.evaluation_suite.sha256 is missing or invalid")

    if benchmarks.synthetic:
        issues.insert(0, "benchmark set is explicitly synthetic")

    attribution_complete = not issues and not benchmarks.synthetic
    if benchmarks.synthetic:
        grade = "synthetic"
    elif attribution_complete:
        grade = "arm64-attributed"
    else:
        grade = "measured-unattributed"

    classification = _text(metadata.get("classification"))
    return grade, {
        "classification": classification,
        "attribution_complete": attribution_complete,
        "verification_scope": (
            "checks whether required source metadata is present; it does not independently "
            "authenticate that metadata or bind it to candidate artifacts"
        ),
        "runner": {
            "architecture": architecture,
            "reported_architecture": raw_architecture,
            "cpu": cpu,
            "cpu_count": cpu_count,
            "os": operating_system,
        },
        "run": {
            "id": run_id,
            "attempt": run_attempt,
        },
        "source": {
            "repository": source_repository,
            "revision": source_revision,
            "workflow": source_workflow,
        },
        "runtime": {
            "name": runtime_name,
            "repository": runtime_repository,
            "revision": runtime_revision,
        },
        "model": {
            "name": model_name,
            "repository": model_repository,
            "revision": model_revision,
        },
        "evaluation_suite": {
            "id": evaluation_id,
            "sha256": evaluation_sha256,
        },
        "issues": issues,
    }


def _declared_attribution_stage(candidate: Candidate) -> str | None:
    configuration = _mapping(candidate.parameters.get("configuration"))
    return _text(configuration.get("attribution_stage"))


def _ordered_candidates(benchmarks: BenchmarkSet) -> tuple[Candidate, ...]:
    indexed = tuple(enumerate(benchmarks.candidates))

    def order(item: tuple[int, Candidate]) -> tuple[int, int, int, str]:
        input_index, candidate = item
        stage = _declared_attribution_stage(candidate)
        if stage in _ATTRIBUTION_ORDER:
            return 0, _ATTRIBUTION_ORDER[stage], input_index, candidate.candidate_id
        return 1, input_index, input_index, candidate.candidate_id

    return tuple(candidate for _, candidate in sorted(indexed, key=order))


def _metric_directions(constraints: Constraints) -> dict[str, str]:
    directions = dict(_KNOWN_METRIC_DIRECTIONS)
    directions.update(constraints.frontier_metrics)
    directions[constraints.objective.metric] = constraints.objective.direction
    directions[constraints.quality_metric] = "max"
    for metric in _RESOURCE_METRICS:
        directions.setdefault(metric, "min")
    return directions


def _percent_change(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return (current - previous) / previous * 100.0


def _effect(previous: float, current: float, direction: str | None) -> str:
    if math.isclose(previous, current, rel_tol=1e-9, abs_tol=1e-12):
        return "held"
    if direction == "min":
        return "improved" if current < previous else "tradeoff"
    if direction == "max":
        return "improved" if current > previous else "tradeoff"
    return "changed"


def _metric_changes(
    previous: Candidate,
    current: Candidate,
    directions: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in sorted(set(previous.metrics) & set(current.metrics)):
        previous_value = previous.metrics[metric]
        current_value = current.metrics[metric]
        direction = directions.get(metric)
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "previous": previous_value,
                "current": current_value,
                "absolute": current_value - previous_value,
                "percent": _percent_change(current_value, previous_value),
                "effect": _effect(previous_value, current_value, direction),
            }
        )
    return rows


def _candidate_status(
    candidate: Candidate,
    *,
    recommendation: Mapping[str, Any],
) -> dict[str, Any]:
    eligible_ids = set(recommendation["eligible_ids"])
    frontier_ids = set(recommendation["frontier_ids"])
    shortlist_ids = set(_mapping(recommendation["selection"])["shortlist_ids"])
    raw_rejected = _mapping(recommendation["rejected"])
    reasons = raw_rejected.get(candidate.candidate_id, ())
    if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes, bytearray)):
        reasons = (str(reasons),)
    return {
        "eligible": candidate.candidate_id in eligible_ids,
        "frontier": candidate.candidate_id in frontier_ids,
        "shortlisted": candidate.candidate_id in shortlist_ids,
        "selected": candidate.candidate_id == recommendation["selected_id"],
        "baseline": candidate.candidate_id == recommendation["baseline_id"],
        "constraint_violations": [str(reason) for reason in reasons],
    }


def _objective_context(
    benchmarks: BenchmarkSet,
    constraints: Constraints,
    recommendation: Mapping[str, Any],
) -> dict[str, Any]:
    selection = _mapping(recommendation["selection"])
    metric = constraints.objective.metric
    direction = constraints.objective.direction
    numeric_best_value = float(selection["numeric_best_value"])
    tolerance_percent = float(selection["objective_tolerance_percent"])
    margin = abs(numeric_best_value) * tolerance_percent / 100.0
    boundary = numeric_best_value + margin if direction == "min" else numeric_best_value - margin
    selected = benchmarks.by_id(str(recommendation["selected_id"]))
    selected_value = selected.metrics[metric]
    runway = boundary - selected_value if direction == "min" else selected_value - boundary
    if runway < 0:
        if math.isclose(selected_value, boundary):
            runway = 0.0
        else:
            raise ValidationError(
                "recomputed selected candidate falls outside the objective shortlist"
            )
    runway_percent = None if boundary == 0 else runway / abs(boundary) * 100.0
    return {
        "metric": metric,
        "direction": direction,
        "numeric_best_id": str(selection["numeric_best_id"]),
        "numeric_best_value": numeric_best_value,
        "tolerance_percent": tolerance_percent,
        "tolerance_margin": margin,
        "shortlist_boundary": boundary,
        "boundary_rule": "at-or-below" if direction == "min" else "at-or-above",
        "selected_value": selected_value,
        "selected_runway": {
            "absolute": runway,
            "percent_of_boundary": runway_percent,
        },
        "shortlist_ids": [str(value) for value in selection["shortlist_ids"]],
    }


def _ladder(
    benchmarks: BenchmarkSet,
    constraints: Constraints,
    recommendation: Mapping[str, Any],
    ordered: Sequence[Candidate],
) -> list[dict[str, Any]]:
    directions = _metric_directions(constraints)
    stages: list[dict[str, Any]] = []
    previous: Candidate | None = None
    for index, candidate in enumerate(ordered, start=1):
        declared_stage = _declared_attribution_stage(candidate)
        delta: dict[str, Any] | None = None
        if previous is not None:
            shared_metrics = set(previous.metrics) & set(candidate.metrics)
            delta = {
                "previous_candidate_id": previous.candidate_id,
                "metrics": _metric_changes(previous, candidate, directions),
                "not_comparable_metrics": sorted(
                    (set(previous.metrics) | set(candidate.metrics)) - shared_metrics
                ),
            }
        stages.append(
            {
                "stage": index,
                "attribution_stage": declared_stage,
                "recognized_attribution_stage": declared_stage in _ATTRIBUTION_ORDER,
                "candidate_id": candidate.candidate_id,
                "label": candidate.label,
                "objective_value": candidate.metrics.get(constraints.objective.metric),
                **_candidate_status(candidate, recommendation=recommendation),
                "delta_from_previous": delta,
            }
        )
        previous = candidate
    return stages


def _outside_shortlist(
    benchmarks: BenchmarkSet,
    constraints: Constraints,
    recommendation: Mapping[str, Any],
    objective: Mapping[str, Any],
    stage_positions: Mapping[str, int],
) -> dict[str, Any] | None:
    shortlist_ids = set(objective["shortlist_ids"])
    frontier_ids = set(recommendation["frontier_ids"])
    boundary = float(objective["shortlist_boundary"])
    metric = constraints.objective.metric
    direction = constraints.objective.direction
    candidates: list[tuple[float, int, str, Candidate]] = []
    for candidate in benchmarks.candidates:
        if candidate.candidate_id not in frontier_ids or candidate.candidate_id in shortlist_ids:
            continue
        value = candidate.metrics[metric]
        shortfall = value - boundary if direction == "min" else boundary - value
        if shortfall < 0 and math.isclose(shortfall, 0.0, rel_tol=1e-9, abs_tol=1e-12):
            shortfall = 0.0
        if shortfall < 0:
            raise ValidationError(
                "frontier candidate outside shortlist has an invalid objective gap"
            )
        candidates.append(
            (
                shortfall,
                stage_positions[candidate.candidate_id],
                candidate.candidate_id,
                candidate,
            )
        )
    if not candidates:
        return None
    shortfall, stage, _, candidate = min(candidates)
    shortfall_percent = None if boundary == 0 else shortfall / abs(boundary) * 100.0
    return {
        "candidate_id": candidate.candidate_id,
        "label": candidate.label,
        "stage": stage,
        "objective_value": candidate.metrics[metric],
        "shortfall_to_shortlist": {
            "absolute": shortfall,
            "percent_of_boundary": shortfall_percent,
        },
    }


def _prompt_value(candidate: Candidate) -> float:
    for metric in ("prompt_tokens_per_second", "prompt_tps"):
        if metric in candidate.metrics:
            return candidate.metrics[metric]
    return -math.inf


def _resource_gain(candidate: Candidate, baseline: Candidate) -> bool:
    return any(
        metric in candidate.metrics
        and metric in baseline.metrics
        and candidate.metrics[metric] < baseline.metrics[metric]
        and not math.isclose(
            candidate.metrics[metric],
            baseline.metrics[metric],
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        for metric in _RESOURCE_METRICS
    )


def _resource_alternative(
    benchmarks: BenchmarkSet,
    constraints: Constraints,
    recommendation: Mapping[str, Any],
    stage_positions: Mapping[str, int],
) -> dict[str, Any] | None:
    baseline = benchmarks.baseline
    selected_id = str(recommendation["selected_id"])
    candidates = [
        candidate
        for candidate in benchmarks.candidates
        if candidate.candidate_id not in {baseline.candidate_id, selected_id}
        and _resource_gain(candidate, baseline)
    ]
    if not candidates:
        return None

    eligible_ids = set(recommendation["eligible_ids"])
    eligible = [candidate for candidate in candidates if candidate.candidate_id in eligible_ids]
    if not eligible:
        return None
    pool = eligible

    lowest: dict[str, float] = {
        metric: min(candidate.metrics[metric] for candidate in pool if metric in candidate.metrics)
        for metric in _RESOURCE_METRICS
        if any(metric in candidate.metrics for candidate in pool)
    }
    resource_shortlist = [
        candidate
        for candidate in pool
        if all(
            metric in candidate.metrics and candidate.metrics[metric] <= value + abs(value) * 0.001
            for metric, value in lowest.items()
        )
    ]

    def resource_score(candidate: Candidate) -> float:
        scores = [
            candidate.metrics[metric] / value
            for metric, value in lowest.items()
            if metric in candidate.metrics and value > 0
        ]
        return sum(scores) / len(scores) if scores else math.inf

    def candidate_key(candidate: Candidate) -> tuple[float, float, float, float, int, str]:
        return (
            -_prompt_value(candidate),
            candidate.metrics.get("e2e_latency_ms_p95", math.inf),
            candidate.metrics.get("peak_rss_mib", math.inf),
            candidate.metrics.get("model_size_mib", math.inf),
            stage_positions[candidate.candidate_id],
            candidate.candidate_id,
        )

    if resource_shortlist:
        alternative = min(resource_shortlist, key=candidate_key)
    else:
        alternative = min(
            pool,
            key=lambda candidate: (
                resource_score(candidate),
                *candidate_key(candidate),
            ),
        )

    changes = _metric_changes(baseline, alternative, _metric_directions(constraints))
    improvements = [row for row in changes if row["effect"] == "improved"]
    tradeoffs = [row for row in changes if row["effect"] == "tradeoff"]
    return {
        "candidate_id": alternative.candidate_id,
        "label": alternative.label,
        "stage": stage_positions[alternative.candidate_id],
        **_candidate_status(alternative, recommendation=recommendation),
        "improvements": improvements,
        "tradeoffs": tradeoffs,
        "other_changes": [row for row in changes if row["effect"] not in {"improved", "tradeoff"}],
        "is_secondary_not_recommendation": True,
    }


def build_decision_passport(
    benchmarks: BenchmarkSet,
    constraints: Constraints,
) -> Mapping[str, Any]:
    """Build a deterministic supplementary decision passport.

    Selection is recomputed with :func:`paretopilot.analysis.recommend`.  The
    returned mapping never modifies the recommendation, source benchmark, or
    canonical renderers.
    """

    if not isinstance(benchmarks, BenchmarkSet):
        raise TypeError("benchmarks must be a validated BenchmarkSet")
    if not isinstance(constraints, Constraints):
        raise TypeError("constraints must be validated Constraints")

    recommendation = recommend(benchmarks, constraints)
    selected = benchmarks.by_id(str(recommendation["selected_id"]))
    ordered = _ordered_candidates(benchmarks)
    stage_positions = {
        candidate.candidate_id: index for index, candidate in enumerate(ordered, start=1)
    }
    evidence_grade, provenance = _provenance(benchmarks)
    objective = _objective_context(benchmarks, constraints, recommendation)
    ladder = _ladder(benchmarks, constraints, recommendation, ordered)
    synthetic_evidence = evidence_grade == "synthetic"

    passport: dict[str, Any] = {
        "schema_version": "1.0",
        "paretopilot_version": __version__,
        "evidence_grade": evidence_grade,
        "provenance": provenance,
        "selected_decision": {
            "candidate_id": selected.candidate_id,
            "label": selected.label,
            "baseline_id": benchmarks.baseline_id,
            "baseline_retained": selected.candidate_id == benchmarks.baseline_id,
            "numeric_best": selected.candidate_id == objective["numeric_best_id"],
            "reason": _mapping(recommendation["selection"])["reason"],
        },
        "objective": objective,
        "ladder": ladder,
        "closest_outside_shortlist": _outside_shortlist(
            benchmarks,
            constraints,
            recommendation,
            objective,
            stage_positions,
        ),
        "resource_alternative": (
            None
            if synthetic_evidence
            else _resource_alternative(
                benchmarks,
                constraints,
                recommendation,
                stage_positions,
            )
        ),
        "method": {
            "selection": "recomputed with paretopilot.analysis.recommend",
            "ladder_order": (
                "recognized attribution stages reference, quantization, arm-kernel, "
                "and runtime-tuning; then stable input order and candidate id"
            ),
            "resource_alternative": (
                "not produced for synthetic example data"
                if synthetic_evidence
                else "measured comparison only; it does not change the selected configuration"
            ),
            "current_boundary_caveat": (
                _SYNTHETIC_BOUNDARY_CAVEAT if synthetic_evidence else _CURRENT_BOUNDARY_CAVEAT
            ),
            "canonical_outputs_modified": False,
        },
    }
    try:
        json.dumps(passport, allow_nan=False, ensure_ascii=False, sort_keys=True)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValidationError(f"decision passport is not strict JSON: {exc}") from exc
    return passport
