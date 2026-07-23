"""Additive v1.1 evidence-first HTML reports.

This module deliberately does not replace :mod:`paretopilot.report`. The v1.0
renderer preserves compatibility with the published v1.0 evidence, while the
``report-v11`` command selects this richer report explicitly.

The v1.1 document is still deterministic, dependency-free, self-contained, and
offline-reviewable.  Browser JavaScript is limited to revealing pre-rendered
policy panels; all recommendations are computed before this renderer is called.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
import math
import re
import shlex
from collections.abc import Mapping, Sequence
from typing import Any

from paretopilot.analysis import recommend
from paretopilot.domain import BenchmarkSet, Candidate, Constraints, ValidationError
from paretopilot.load_eval import (
    validate_combined_load_evaluation,
    validate_load_evaluation,
)
from paretopilot.stability import (
    candidate_configuration_fingerprint,
    validate_stability_summary,
)


__all__ = ["render_report_v11"]


_METRIC_LABELS = {
    "completion_rate": "Completion rate",
    "e2e_latency_ms_p95": "End-to-end latency (p95)",
    "e2e_latency_ms_p50": "End-to-end latency (p50)",
    "error_rate": "Error rate",
    "failed_requests": "Failed requests",
    "generated_tokens_per_second": "Generated-token throughput",
    "generation_tokens_per_second": "Generation throughput",
    "generation_tps": "Generation throughput",
    "model_size_mib": "Model size",
    "peak_rss_mib": "Peak resident memory",
    "prompt_tokens_per_second": "Prompt processing throughput",
    "prompt_tps": "Prompt processing throughput",
    "quality_score": "Quality score",
    "requests_per_second": "Request throughput",
    "request_count": "Measured requests",
    "completed_requests": "Completed requests",
    "ttft_ms_p50": "Time to first token (p50)",
    "ttft_ms_p95": "Time to first token (p95)",
    "wall_time_seconds": "Wall time",
}

_METRIC_PRIORITY = (
    "e2e_latency_ms_p95",
    "ttft_ms_p95",
    "prompt_tokens_per_second",
    "prompt_tps",
    "generation_tokens_per_second",
    "generation_tps",
    "requests_per_second",
    "generated_tokens_per_second",
    "peak_rss_mib",
    "model_size_mib",
    "completion_rate",
    "error_rate",
    "failed_requests",
    "quality_score",
)

_LOAD_ROW_FIELDS = {
    "candidate_id",
    "concurrency",
    "request_count",
    "completed_requests",
    "failed_requests",
    "requests_per_second",
    "generated_tokens_per_second",
    "wall_time_seconds",
    "ttft_ms_p50",
    "e2e_latency_ms_p95",
    "ttft_ms_p95",
    "e2e_latency_ms_p50",
    "completion_rate",
    "error_rate",
    "peak_rss_mib",
    "slo_met",
    "slo_failures",
    "samples",
}

_LOAD_METRICS = (
    "requests_per_second",
    "generated_tokens_per_second",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "e2e_latency_ms_p50",
    "e2e_latency_ms_p95",
    "completion_rate",
    "error_rate",
    "peak_rss_mib",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PASS_RE = re.compile(r"^(throughput|server_evaluation|server_time)_pass_(\d+)$")


@dataclass(frozen=True)
class _DecisionProfile:
    profile_id: str
    label: str
    description: str
    selected_id: str
    objective_metric: str
    objective_direction: str
    reason: str
    derived: bool
    scenario_notice: str = ""


@dataclass(frozen=True)
class _LoadRow:
    candidate_id: str
    concurrency: int
    metrics: Mapping[str, float]
    request_count: int
    completed_requests: int
    failed_requests: int
    wall_time_seconds: float
    slo_met: bool
    slo_failures: tuple[str, ...]
    samples: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _LoadSweep:
    rows: tuple[_LoadRow, ...]
    methodology: Mapping[str, Any]
    slo: Mapping[str, Any]
    highest_slo_concurrency: Mapping[str, int | None]
    evidence_bindings: Mapping[str, Any] | None
    synthetic: bool


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _normalise_json_value(value: Any) -> Any:
    """Return a stable JSON-safe representation for source metadata."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_json_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_normalise_json_value(item) for item in value), key=str)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalise_json_value(item) for item in value]
    return f"<{type(value).__name__}>"


def _stable_json(value: Any) -> str:
    return json.dumps(
        _normalise_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(", ", ": "),
    )


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{context} must be an object")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{context} must be a non-empty string")
    return value


def _finite_number(value: Any, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{context} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValidationError(f"{context} must be finite")
    if minimum is not None and converted < minimum:
        raise ValidationError(f"{context} must be at least {minimum:g}")
    return converted


def _positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{context} must be a positive integer")
    return value


def _direction(value: Any, context: str) -> str:
    if value not in {"min", "max"}:
        raise ValidationError(f"{context} must be 'min' or 'max'")
    return str(value)


def _sequence_of_ids(value: Any, context: str, valid_ids: set[str]) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValidationError(f"{context} must be an array")
    result: list[str] = []
    for index, item in enumerate(value):
        candidate_id = _text(item, f"{context}[{index}]")
        if candidate_id not in valid_ids:
            raise ValidationError(f"{context}[{index}] is not present in benchmarks")
        result.append(candidate_id)
    if len(result) != len(set(result)):
        raise ValidationError(f"{context} must not contain duplicate candidate ids")
    return tuple(result)


def _metric_label(metric: str) -> str:
    return _METRIC_LABELS.get(metric, metric.replace("_", " ").strip().capitalize())


def _metric_unit(metric: str) -> str:
    if metric in {"completion_rate", "error_rate"}:
        return "%"
    if metric == "wall_time_seconds":
        return " s"
    if metric.endswith("_ms") or "_ms_" in metric:
        return " ms"
    if metric.endswith("_mib") or metric.endswith("_mb"):
        return " MiB"
    if metric in {"requests_per_second"}:
        return " req/s"
    if metric.endswith("_tps") or metric.endswith("_tokens_per_second"):
        return " tok/s"
    return ""


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-9):
        return f"{value:,.0f}"
    if abs(value) >= 1_000_000 or (0 < abs(value) < 0.0001):
        return f"{value:.4g}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _format_metric(metric: str, value: float) -> str:
    if metric in {"completion_rate", "error_rate"}:
        return f"{value * 100:.2f}%"
    return f"{_format_number(value)}{_metric_unit(metric)}"


def _ordered_metrics(benchmarks: BenchmarkSet) -> tuple[str, ...]:
    metrics = {metric for candidate in benchmarks.candidates for metric in candidate.metrics}
    priority = [metric for metric in _METRIC_PRIORITY if metric in metrics]
    return tuple(priority + sorted(metrics - set(priority)))


def _validated_recommendation(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
) -> tuple[Candidate, Mapping[str, Any], Mapping[str, Any]]:
    raw = _mapping(recommendation, "recommendation")
    _text(
        raw.get("paretopilot_version"),
        "recommendation.paretopilot_version",
    )
    valid_ids = {candidate.candidate_id for candidate in benchmarks.candidates}
    selected_id = _text(raw.get("selected_id"), "recommendation.selected_id")
    if selected_id not in valid_ids:
        raise ValidationError("recommendation.selected_id is not present in benchmarks")
    baseline_id = raw.get("baseline_id", benchmarks.baseline_id)
    if baseline_id != benchmarks.baseline_id:
        raise ValidationError("recommendation.baseline_id does not match benchmarks.baseline_id")

    objective = _mapping(raw.get("objective"), "recommendation.objective")
    metric = _text(objective.get("metric"), "recommendation.objective.metric")
    _direction(objective.get("direction"), "recommendation.objective.direction")
    if metric not in benchmarks.by_id(selected_id).metrics:
        raise ValidationError(
            "recommendation objective metric is missing from the selected candidate"
        )

    selection = _mapping(raw.get("selection"), "recommendation.selection")
    numeric_best_id = _text(
        selection.get("numeric_best_id"),
        "recommendation.selection.numeric_best_id",
    )
    if numeric_best_id not in valid_ids:
        raise ValidationError(
            "recommendation.selection.numeric_best_id is not present in benchmarks"
        )
    tolerance = _finite_number(
        selection.get("objective_tolerance_percent"),
        "recommendation.selection.objective_tolerance_percent",
        minimum=0.0,
    )
    if tolerance > 100.0:
        raise ValidationError(
            "recommendation.selection.objective_tolerance_percent must not exceed 100"
        )
    _sequence_of_ids(
        selection.get("shortlist_ids"),
        "recommendation.selection.shortlist_ids",
        valid_ids,
    )
    _text(selection.get("reason"), "recommendation.selection.reason")

    constraints = _mapping(raw.get("constraints"), "recommendation.constraints")
    constraints_objective = _mapping(
        constraints.get("objective"),
        "recommendation.constraints.objective",
    )
    if constraints_objective.get("metric") != metric or constraints_objective.get(
        "direction"
    ) != objective.get("direction"):
        raise ValidationError(
            "recommendation objective does not match recommendation.constraints.objective"
        )

    for field in ("eligible_ids", "frontier_ids"):
        _sequence_of_ids(raw.get(field), f"recommendation.{field}", valid_ids)
    rejected = _mapping(raw.get("rejected"), "recommendation.rejected")
    for candidate_id, reasons in rejected.items():
        if candidate_id not in valid_ids:
            raise ValidationError(
                f"recommendation.rejected candidate {candidate_id!r} is not in benchmarks"
            )
        if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes, bytearray)):
            raise ValidationError(f"recommendation.rejected.{candidate_id} must be an array")
        for index, reason in enumerate(reasons):
            _text(reason, f"recommendation.rejected.{candidate_id}[{index}]")

    try:
        reconstructed_constraints = Constraints.from_mapping(constraints)
        expected = dict(recommend(benchmarks, reconstructed_constraints))
    except ValidationError as exc:
        raise ValidationError(f"recommendation constraints are invalid: {exc}") from exc
    supplied = dict(raw)
    supplied.pop("input_fingerprints", None)
    supplied.pop("paretopilot_version", None)
    expected.pop("paretopilot_version", None)
    if _stable_json(supplied) != _stable_json(expected):
        raise ValidationError(
            "recommendation does not match a fresh selection from its benchmark set "
            "and declared constraints"
        )

    return benchmarks.by_id(selected_id), selection, constraints


def _validate_recommendation_fingerprints(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
    *,
    benchmarks_sha256: str,
) -> None:
    raw_fingerprints = recommendation.get("input_fingerprints")
    if raw_fingerprints is None:
        if benchmarks.synthetic:
            return
        raise ValidationError("measured recommendation must include input_fingerprints")
    fingerprints = _mapping(
        raw_fingerprints,
        "recommendation.input_fingerprints",
    )
    _exact_fields(
        fingerprints,
        {"benchmarks_sha256", "constraints_sha256"},
        "recommendation.input_fingerprints",
    )
    benchmark_digest = _hash_value(
        fingerprints.get("benchmarks_sha256"),
        "recommendation.input_fingerprints.benchmarks_sha256",
    )
    constraints_digest = _hash_value(
        fingerprints.get("constraints_sha256"),
        "recommendation.input_fingerprints.constraints_sha256",
    )
    if not benchmark_digest or not constraints_digest:
        raise ValidationError("recommendation input fingerprints must be non-empty SHA-256 digests")
    if benchmarks_sha256 and benchmark_digest != benchmarks_sha256:
        raise ValidationError("recommendation benchmark fingerprint does not match report input")


def _decision_from_mapping(
    profile_id: str,
    raw_profile: Mapping[str, Any],
    benchmarks: BenchmarkSet,
) -> _DecisionProfile:
    """Accept a recommendation mapping or a small envelope around one."""

    valid_ids = {candidate.candidate_id for candidate in benchmarks.candidates}
    if "recommendation" in raw_profile:
        decision = _mapping(
            raw_profile.get("recommendation"),
            f"policy_profiles.{profile_id}.recommendation",
        )
        label = _text(
            raw_profile.get("label", profile_id.replace("-", " ").title()),
            f"policy_profiles.{profile_id}.label",
        )
        description = _text(
            raw_profile.get(
                "description",
                "A precomputed alternative policy applied to the same measured evidence.",
            ),
            f"policy_profiles.{profile_id}.description",
        )
    elif "decision" in raw_profile:
        decision = _mapping(raw_profile.get("decision"), f"policy_profiles.{profile_id}.decision")
        label = _text(
            raw_profile.get("label", profile_id.replace("-", " ").title()),
            f"policy_profiles.{profile_id}.label",
        )
        description = _text(
            raw_profile.get(
                "description",
                "A precomputed alternative policy applied to the same measured evidence.",
            ),
            f"policy_profiles.{profile_id}.description",
        )
    else:
        decision = raw_profile
        label = _text(
            raw_profile.get("label", profile_id.replace("-", " ").title()),
            f"policy_profiles.{profile_id}.label",
        )
        description = _text(
            raw_profile.get(
                "description",
                "A precomputed alternative policy applied to the same measured evidence.",
            ),
            f"policy_profiles.{profile_id}.description",
        )

    _validated_recommendation(benchmarks, decision)
    selected_id = _text(
        decision.get("selected_id"),
        f"policy_profiles.{profile_id}.selected_id",
    )
    if selected_id not in valid_ids:
        raise ValidationError(
            f"policy_profiles.{profile_id}.selected_id is not present in benchmarks"
        )
    objective = _mapping(
        decision.get("objective"),
        f"policy_profiles.{profile_id}.objective",
    )
    objective_metric = _text(
        objective.get("metric"),
        f"policy_profiles.{profile_id}.objective.metric",
    )
    objective_direction = _direction(
        objective.get("direction"),
        f"policy_profiles.{profile_id}.objective.direction",
    )
    if objective_metric not in benchmarks.by_id(selected_id).metrics:
        raise ValidationError(
            f"policy_profiles.{profile_id} objective metric is missing from its selection"
        )

    raw_selection = decision.get("selection")
    if isinstance(raw_selection, Mapping):
        raw_reason = raw_selection.get("reason")
    else:
        raw_reason = decision.get("reason")
    reason = _text(raw_reason, f"policy_profiles.{profile_id}.reason")
    return _DecisionProfile(
        profile_id=profile_id,
        label=label,
        description=description,
        selected_id=selected_id,
        objective_metric=objective_metric,
        objective_direction=objective_direction,
        reason=reason,
        derived=True,
        scenario_notice=(
            _text(
                raw_profile.get("scenario_notice"),
                f"policy_profiles.{profile_id}.scenario_notice",
            )
            if "scenario_notice" in raw_profile
            else ""
        ),
    )


def _normalise_profiles(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
    policy_profiles: Mapping[str, Any] | None,
    *,
    benchmarks_sha256: str,
) -> tuple[_DecisionProfile, ...]:
    selected, selection, _ = _validated_recommendation(benchmarks, recommendation)
    objective = _mapping(recommendation["objective"], "recommendation.objective")
    canonical = _DecisionProfile(
        profile_id="canonical",
        label="Canonical policy",
        description="The predeclared objective, tolerance, constraints, and preference order.",
        selected_id=selected.candidate_id,
        objective_metric=str(objective["metric"]),
        objective_direction=str(objective["direction"]),
        reason=str(selection["reason"]),
        derived=False,
    )
    if policy_profiles is None:
        return (canonical,)

    raw_container = _mapping(policy_profiles, "policy_profiles")
    if "profiles" in raw_container and isinstance(raw_container.get("profiles"), Sequence):
        allowed = {
            "schema_version",
            "source_schema_version",
            "synthetic_source",
            "canonical_profile_id",
            "canonical_selected_id",
            "profiles",
            "input_fingerprints",
        }
        unknown = sorted(set(raw_container) - allowed)
        if unknown:
            raise ValidationError("policy_profiles contains unknown fields: " + ", ".join(unknown))
        if raw_container.get("schema_version") != "1.0":
            raise ValidationError("policy_profiles.schema_version must be '1.0'")
        if raw_container.get("source_schema_version") != benchmarks.schema_version:
            raise ValidationError("policy_profiles.source_schema_version does not match benchmarks")
        if raw_container.get("synthetic_source") is not benchmarks.synthetic:
            raise ValidationError("policy_profiles.synthetic_source does not match benchmarks")
        canonical_profile_id = _text(
            raw_container.get("canonical_profile_id"),
            "policy_profiles.canonical_profile_id",
        )
        if raw_container.get("canonical_selected_id") != selected.candidate_id:
            raise ValidationError(
                "policy_profiles.canonical_selected_id does not match recommendation"
            )
        if not benchmarks.synthetic and "input_fingerprints" not in raw_container:
            raise ValidationError("measured policy_profiles must include input_fingerprints")
        if "input_fingerprints" in raw_container:
            fingerprints = _mapping(
                raw_container.get("input_fingerprints"),
                "policy_profiles.input_fingerprints",
            )
            allowed_fingerprints = {
                "benchmarks_sha256",
                "constraints_sha256",
                "policies_sha256",
            }
            unknown_fingerprints = sorted(set(fingerprints) - allowed_fingerprints)
            if unknown_fingerprints:
                raise ValidationError(
                    "policy_profiles.input_fingerprints contains unknown fields: "
                    + ", ".join(unknown_fingerprints)
                )
            for name in allowed_fingerprints:
                if name not in fingerprints:
                    raise ValidationError(f"policy_profiles.input_fingerprints.{name} is required")
                _hash_value(
                    fingerprints[name],
                    f"policy_profiles.input_fingerprints.{name}",
                )
            if benchmarks_sha256 and fingerprints["benchmarks_sha256"] != benchmarks_sha256:
                raise ValidationError(
                    "policy_profiles benchmark fingerprint does not match report input"
                )
            recommendation_fingerprints = recommendation.get("input_fingerprints")
            if isinstance(recommendation_fingerprints, Mapping):
                expected_constraints = recommendation_fingerprints.get("constraints_sha256")
                if (
                    isinstance(expected_constraints, str)
                    and fingerprints["constraints_sha256"] != expected_constraints
                ):
                    raise ValidationError(
                        "policy_profiles constraints fingerprint does not match recommendation"
                    )
        raw_entries = raw_container.get("profiles")
        assert isinstance(raw_entries, Sequence)
        entries: list[tuple[str, Mapping[str, Any]]] = []
        seen_ids: set[str] = set()
        for index, raw_entry_value in enumerate(raw_entries):
            context = f"policy_profiles.profiles[{index}]"
            raw_entry = _mapping(raw_entry_value, context)
            expected = {
                "id",
                "label",
                "description",
                "classification",
                "scenario_notice",
                "recommendation",
            }
            missing = sorted(expected - set(raw_entry))
            unknown_entry = sorted(set(raw_entry) - expected)
            if missing or unknown_entry:
                details: list[str] = []
                if missing:
                    details.append("missing fields: " + ", ".join(missing))
                if unknown_entry:
                    details.append("unknown fields: " + ", ".join(unknown_entry))
                raise ValidationError(f"{context} has " + "; ".join(details))
            profile_id = _text(raw_entry.get("id"), f"{context}.id")
            if profile_id in seen_ids:
                raise ValidationError("policy_profiles profile ids must be unique")
            seen_ids.add(profile_id)
            entries.append((profile_id, raw_entry))

        canonical_entries = [
            entry for profile_id, entry in entries if profile_id == canonical_profile_id
        ]
        if len(canonical_entries) != 1:
            raise ValidationError(
                "policy_profiles must contain exactly one canonical_profile_id entry"
            )
        canonical_entry = canonical_entries[0]
        if canonical_entry.get("classification") != "canonical":
            raise ValidationError(
                "policy_profiles canonical entry classification must be 'canonical'"
            )
        canonical_recommendation = _mapping(
            canonical_entry.get("recommendation"),
            "policy_profiles canonical recommendation",
        )
        comparable_recommendation = dict(recommendation)
        comparable_recommendation.pop("input_fingerprints", None)
        comparable_canonical = dict(canonical_recommendation)
        comparable_canonical.pop("input_fingerprints", None)
        if _stable_json(comparable_canonical) != _stable_json(comparable_recommendation):
            raise ValidationError(
                "policy_profiles canonical recommendation does not match recommendation"
            )
        canonical = _DecisionProfile(
            profile_id=canonical_profile_id,
            label=_text(canonical_entry.get("label"), "policy_profiles canonical label"),
            description=_text(
                canonical_entry.get("description"),
                "policy_profiles canonical description",
            ),
            selected_id=selected.candidate_id,
            objective_metric=str(objective["metric"]),
            objective_direction=str(objective["direction"]),
            reason=str(selection["reason"]),
            derived=False,
            scenario_notice=_text(
                canonical_entry.get("scenario_notice"),
                "policy_profiles canonical scenario_notice",
            ),
        )
        profiles = [canonical]
        for profile_id, entry in entries:
            if profile_id == canonical_profile_id:
                continue
            if entry.get("classification") != "derived-non-canonical":
                raise ValidationError(
                    f"policy_profiles.{profile_id} classification must be 'derived-non-canonical'"
                )
            profiles.append(_decision_from_mapping(profile_id, entry, benchmarks))
        return tuple(profiles)

    if "profiles" in raw_container:
        if raw_container.get("schema_version") != "1.0":
            raise ValidationError("policy_profiles.schema_version must be '1.0'")
        unknown = sorted(set(raw_container) - {"schema_version", "profiles"})
        if unknown:
            raise ValidationError("policy_profiles contains unknown fields: " + ", ".join(unknown))
        raw_profiles = _mapping(raw_container.get("profiles"), "policy_profiles.profiles")
    else:
        raw_profiles = raw_container

    profiles: list[_DecisionProfile] = [canonical]
    for raw_id in sorted(raw_profiles, key=str):
        profile_id = _text(raw_id, "policy_profiles profile id")
        if profile_id == "canonical":
            raise ValidationError("policy_profiles must not replace the canonical recommendation")
        raw_profile = _mapping(
            raw_profiles[raw_id],
            f"policy_profiles.{profile_id}",
        )
        profiles.append(_decision_from_mapping(profile_id, raw_profile, benchmarks))
    return tuple(profiles)


def _nonnegative_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{context} must be a non-negative integer")
    return value


def _exact_fields(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        if unknown:
            details.append("unknown fields: " + ", ".join(unknown))
        raise ValidationError(f"{context} has " + "; ".join(details))


def _normalise_load_methodology(
    raw_value: Any,
) -> tuple[Mapping[str, Any], tuple[int, ...], Mapping[int, str]]:
    raw = _mapping(raw_value, "load_sweep.methodology")
    expected = {
        "concurrency_levels",
        "prompts",
        "output_tokens",
        "warmup_requests_per_level",
        "measured_requests_per_level",
        "prompt_schedule",
        "client_scheduler",
    }
    _exact_fields(raw, expected, "load_sweep.methodology")
    raw_levels = raw.get("concurrency_levels")
    if not isinstance(raw_levels, Sequence) or isinstance(raw_levels, (str, bytes, bytearray)):
        raise ValidationError("load_sweep.methodology.concurrency_levels must be an array")
    levels = tuple(
        _positive_integer(value, f"load_sweep.methodology.concurrency_levels[{index}]")
        for index, value in enumerate(raw_levels)
    )
    if not levels or len(levels) != len(set(levels)) or levels != tuple(sorted(levels)):
        raise ValidationError(
            "load_sweep.methodology.concurrency_levels must be unique and increasing"
        )

    raw_prompts = raw.get("prompts")
    if not isinstance(raw_prompts, Sequence) or isinstance(raw_prompts, (str, bytes, bytearray)):
        raise ValidationError("load_sweep.methodology.prompts must be an array")
    prompts: list[Mapping[str, Any]] = []
    prompt_hashes: dict[int, str] = {}
    for index, raw_prompt_value in enumerate(raw_prompts):
        context = f"load_sweep.methodology.prompts[{index}]"
        raw_prompt = _mapping(raw_prompt_value, context)
        _exact_fields(raw_prompt, {"index", "text", "sha256"}, context)
        prompt_index = _nonnegative_integer(raw_prompt.get("index"), f"{context}.index")
        if prompt_index in prompt_hashes:
            raise ValidationError("load_sweep.methodology prompt indexes must be unique")
        prompt_text = _text(raw_prompt.get("text"), f"{context}.text")
        digest = _text(raw_prompt.get("sha256"), f"{context}.sha256")
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValidationError(f"{context}.sha256 must be a lowercase SHA-256 digest")
        prompt_hashes[prompt_index] = digest
        prompts.append({"index": prompt_index, "text": prompt_text, "sha256": digest})
    if not prompts:
        raise ValidationError("load_sweep.methodology.prompts must not be empty")

    output_tokens = _positive_integer(
        raw.get("output_tokens"),
        "load_sweep.methodology.output_tokens",
    )
    warmups = _nonnegative_integer(
        raw.get("warmup_requests_per_level"),
        "load_sweep.methodology.warmup_requests_per_level",
    )
    measured = _positive_integer(
        raw.get("measured_requests_per_level"),
        "load_sweep.methodology.measured_requests_per_level",
    )
    if raw.get("prompt_schedule") != "round-robin":
        raise ValidationError("load_sweep.methodology.prompt_schedule must be 'round-robin'")
    if raw.get("client_scheduler") != "bounded thread pool":
        raise ValidationError(
            "load_sweep.methodology.client_scheduler must be 'bounded thread pool'"
        )
    methodology = {
        "client_scheduler": "bounded thread pool",
        "concurrency_levels": list(levels),
        "measured_requests_per_level": measured,
        "output_tokens": output_tokens,
        "prompt_schedule": "round-robin",
        "prompts": prompts,
        "warmup_requests_per_level": warmups,
    }
    return methodology, levels, prompt_hashes


def _normalise_load_slo(raw_value: Any) -> Mapping[str, float]:
    raw = _mapping(raw_value, "load_sweep.slo")
    expected = {
        "min_completion_rate",
        "max_ttft_ms_p95",
        "max_e2e_latency_ms_p95",
    }
    _exact_fields(raw, expected, "load_sweep.slo")
    completion = _finite_number(
        raw.get("min_completion_rate"),
        "load_sweep.slo.min_completion_rate",
        minimum=0.0,
    )
    if completion > 1.0:
        raise ValidationError("load_sweep.slo.min_completion_rate must not exceed 1")
    return {
        "min_completion_rate": completion,
        "max_ttft_ms_p95": _finite_number(
            raw.get("max_ttft_ms_p95"),
            "load_sweep.slo.max_ttft_ms_p95",
            minimum=0.0,
        ),
        "max_e2e_latency_ms_p95": _finite_number(
            raw.get("max_e2e_latency_ms_p95"),
            "load_sweep.slo.max_e2e_latency_ms_p95",
            minimum=0.0,
        ),
    }


def _normalise_load_samples(
    raw_value: Any,
    *,
    context: str,
    prompt_hashes: Mapping[int, str],
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes, bytearray)):
        raise ValidationError(f"{context} must be an array")
    samples: list[Mapping[str, Any]] = []
    indexes: set[int] = set()
    expected = {
        "index",
        "prompt_index",
        "prompt_sha256",
        "completed",
        "ttft_ms",
        "e2e_latency_ms",
        "generated_tokens",
        "error",
        "started_at_seconds",
        "finished_at_seconds",
    }
    for position, raw_sample_value in enumerate(raw_value):
        sample_context = f"{context}[{position}]"
        raw_sample = _mapping(raw_sample_value, sample_context)
        _exact_fields(raw_sample, expected, sample_context)
        sample_index = _nonnegative_integer(
            raw_sample.get("index"),
            f"{sample_context}.index",
        )
        if sample_index in indexes:
            raise ValidationError(f"{context} sample indexes must be unique")
        indexes.add(sample_index)
        prompt_index = _nonnegative_integer(
            raw_sample.get("prompt_index"),
            f"{sample_context}.prompt_index",
        )
        if prompt_index not in prompt_hashes:
            raise ValidationError(f"{sample_context}.prompt_index is not declared")
        prompt_sha256 = _text(
            raw_sample.get("prompt_sha256"),
            f"{sample_context}.prompt_sha256",
        )
        if prompt_sha256 != prompt_hashes[prompt_index]:
            raise ValidationError(
                f"{sample_context}.prompt_sha256 does not match the declared prompt"
            )
        completed = raw_sample.get("completed")
        if not isinstance(completed, bool):
            raise ValidationError(f"{sample_context}.completed must be a boolean")

        numeric_values: dict[str, float | int | None] = {}
        for field in ("ttft_ms", "e2e_latency_ms"):
            value = raw_sample.get(field)
            numeric_values[field] = (
                None
                if value is None
                else _finite_number(value, f"{sample_context}.{field}", minimum=0.0)
            )
        generated_value = raw_sample.get("generated_tokens")
        numeric_values["generated_tokens"] = (
            None
            if generated_value is None
            else _nonnegative_integer(
                generated_value,
                f"{sample_context}.generated_tokens",
            )
        )
        error = raw_sample.get("error")
        if error is not None and not isinstance(error, str):
            raise ValidationError(f"{sample_context}.error must be a string or null")
        if completed:
            if any(numeric_values[field] is None for field in numeric_values):
                raise ValidationError(
                    f"{sample_context} completed samples require all measurements"
                )
            if error not in {None, ""}:
                raise ValidationError(
                    f"{sample_context} completed samples must not contain an error"
                )
        elif not isinstance(error, str) or not error.strip():
            raise ValidationError(f"{sample_context} failed samples require a non-empty error")
        started = _finite_number(
            raw_sample.get("started_at_seconds"),
            f"{sample_context}.started_at_seconds",
            minimum=0.0,
        )
        finished = _finite_number(
            raw_sample.get("finished_at_seconds"),
            f"{sample_context}.finished_at_seconds",
            minimum=0.0,
        )
        if finished < started:
            raise ValidationError(f"{sample_context}.finished_at_seconds must not precede start")
        samples.append(
            {
                "completed": completed,
                "e2e_latency_ms": numeric_values["e2e_latency_ms"],
                "error": error,
                "finished_at_seconds": finished,
                "generated_tokens": numeric_values["generated_tokens"],
                "index": sample_index,
                "prompt_index": prompt_index,
                "prompt_sha256": prompt_sha256,
                "started_at_seconds": started,
                "ttft_ms": numeric_values["ttft_ms"],
            }
        )
    return tuple(samples)


def _normalise_load_sweep(
    benchmarks: BenchmarkSet,
    load_sweep: Mapping[str, Any] | None,
) -> _LoadSweep | None:
    if load_sweep is None:
        return None
    raw = _mapping(load_sweep, "load_sweep")
    single_candidate = "candidate_id" in raw
    require_bindings = not benchmarks.synthetic
    if single_candidate:
        validate_load_evaluation(
            raw,
            require_evidence_binding=require_bindings,
        )
    else:
        validate_combined_load_evaluation(
            raw,
            require_evidence_bindings=require_bindings,
        )
    synthetic = bool(raw["synthetic"])
    if synthetic is not benchmarks.synthetic:
        raise ValidationError("load_sweep.synthetic does not match benchmarks")
    methodology = _mapping(raw["methodology"], "load_sweep.methodology")
    slo = _mapping(raw["slo"], "load_sweep.slo")
    raw_rows = raw["rows"]
    assert isinstance(raw_rows, Sequence)
    valid_ids = {candidate.candidate_id for candidate in benchmarks.candidates}
    rows: list[_LoadRow] = []
    for index, raw_row_value in enumerate(raw_rows):
        context = f"load_sweep.rows[{index}]"
        raw_row = _mapping(raw_row_value, context)
        candidate_id = str(raw_row["candidate_id"])
        if candidate_id not in valid_ids:
            raise ValidationError(f"{context}.candidate_id is not present in benchmarks")
        metrics = {
            metric: float(raw_row[metric])
            for metric in _LOAD_METRICS
            if raw_row.get(metric) is not None
        }
        raw_samples = raw_row["samples"]
        raw_failures = raw_row["slo_failures"]
        assert isinstance(raw_samples, Sequence)
        assert isinstance(raw_failures, Sequence)
        rows.append(
            _LoadRow(
                candidate_id,
                int(raw_row["concurrency"]),
                metrics,
                int(raw_row["request_count"]),
                int(raw_row["completed_requests"]),
                int(raw_row["failed_requests"]),
                float(raw_row["wall_time_seconds"]),
                bool(raw_row["slo_met"]),
                tuple(str(value) for value in raw_failures),
                tuple(_mapping(sample, f"{context}.samples") for sample in raw_samples),
            )
        )
    if single_candidate:
        raw_highest: Mapping[str, Any] = {str(raw["candidate_id"]): raw["highest_slo_concurrency"]}
    else:
        raw_highest = _mapping(
            raw["highest_slo_concurrency"],
            "load_sweep.highest_slo_concurrency",
        )
    highest = {
        str(candidate_id): None if value is None else int(value)
        for candidate_id, value in raw_highest.items()
    }
    unknown_highest = sorted(set(highest) - valid_ids)
    if unknown_highest:
        raise ValidationError(
            "load_sweep.highest_slo_concurrency contains unknown candidates: "
            + ", ".join(unknown_highest)
        )
    measured_candidate_ids = {row.candidate_id for row in rows}
    if require_bindings and measured_candidate_ids != valid_ids:
        missing = sorted(valid_ids - measured_candidate_ids)
        unknown = sorted(measured_candidate_ids - valid_ids)
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise ValidationError(
            "measured load_sweep must cover every benchmark candidate"
            + (f" ({'; '.join(details)})" if details else "")
        )
    raw_bindings = raw["evidence_binding"] if single_candidate else raw["evidence_bindings"]
    normalized_bindings = (
        _mapping(raw_bindings, "load_sweep evidence bindings") if raw_bindings is not None else None
    )
    if require_bindings:
        assert normalized_bindings is not None
        if single_candidate:
            binding_configurations = {
                str(raw["candidate_id"]): _mapping(
                    normalized_bindings.get("server_configuration"),
                    "load_sweep server_configuration",
                )
            }
        else:
            binding_configurations = _mapping(
                normalized_bindings.get("candidate_server_configurations"),
                "load_sweep candidate_server_configurations",
            )
        for candidate_id in sorted(valid_ids):
            expected_argv = benchmarks.by_id(candidate_id).parameters.get("deployment_argv")
            if not isinstance(expected_argv, Sequence) or isinstance(
                expected_argv,
                (str, bytes, bytearray),
            ):
                raise ValidationError(
                    f"benchmark candidate {candidate_id!r} is missing a deployment_argv "
                    "needed to bind measured load evidence"
                )
            configuration = _mapping(
                binding_configurations.get(candidate_id),
                f"load_sweep binding for {candidate_id}",
            )
            canonical_argv = configuration.get("canonical_server_argv")
            if list(canonical_argv) != list(expected_argv):
                raise ValidationError(
                    f"load_sweep canonical server command for {candidate_id!r} does not "
                    "match the benchmark candidate deployment_argv"
                )

    return _LoadSweep(
        tuple(rows),
        dict(methodology),
        dict(slo),
        highest,
        normalized_bindings,
        synthetic,
    )


def _frontier_directions(recommendation: Mapping[str, Any]) -> Mapping[str, str]:
    constraints = recommendation.get("constraints")
    if not isinstance(constraints, Mapping):
        return {}
    raw = constraints.get("frontier_metrics")
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(metric): str(direction)
        for metric, direction in raw.items()
        if isinstance(metric, str) and direction in {"min", "max"}
    }


def _resource_alternative(
    benchmarks: BenchmarkSet,
    selected: Candidate,
) -> Candidate | None:
    """Choose a descriptive low-resource alternative without changing the decision."""

    candidates = [
        candidate
        for candidate in benchmarks.candidates
        if candidate.candidate_id != selected.candidate_id
        and ("peak_rss_mib" in candidate.metrics or "model_size_mib" in candidate.metrics)
    ]
    if not candidates:
        return None

    lowest_rss = min(candidate.metrics.get("peak_rss_mib", math.inf) for candidate in candidates)
    lowest_model = min(
        candidate.metrics.get("model_size_mib", math.inf) for candidate in candidates
    )
    resource_shortlist = [
        candidate
        for candidate in candidates
        if (
            candidate.metrics.get("peak_rss_mib", math.inf) <= lowest_rss + abs(lowest_rss) * 0.001
            and candidate.metrics.get("model_size_mib", math.inf)
            <= lowest_model + abs(lowest_model) * 0.001
        )
    ]

    def prompt_value(candidate: Candidate) -> float:
        prompt_metric = (
            candidate.metrics.get("prompt_tokens_per_second")
            if "prompt_tokens_per_second" in candidate.metrics
            else candidate.metrics.get("prompt_tps")
        )
        return prompt_metric if prompt_metric is not None else -math.inf

    def shortlist_key(candidate: Candidate) -> tuple[float, float, float, str]:
        return (
            -prompt_value(candidate),
            candidate.metrics.get("e2e_latency_ms_p95", math.inf),
            candidate.metrics.get("peak_rss_mib", math.inf),
            candidate.candidate_id,
        )

    if resource_shortlist:
        alternative = min(resource_shortlist, key=shortlist_key)
    else:

        def resource_score(candidate: Candidate) -> float:
            rss = candidate.metrics.get("peak_rss_mib", math.inf)
            model = candidate.metrics.get("model_size_mib", math.inf)
            rss_score = rss / lowest_rss if math.isfinite(rss) and lowest_rss > 0 else 1.0
            model_score = model / lowest_model if math.isfinite(model) and lowest_model > 0 else 1.0
            return rss_score + model_score

        alternative = min(
            candidates,
            key=lambda candidate: (
                resource_score(candidate),
                -prompt_value(candidate),
                candidate.candidate_id,
            ),
        )
    baseline = benchmarks.baseline
    has_resource_gain = any(
        metric in baseline.metrics
        and metric in alternative.metrics
        and alternative.metrics[metric] < baseline.metrics[metric]
        and not math.isclose(
            alternative.metrics[metric],
            baseline.metrics[metric],
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        for metric in ("peak_rss_mib", "model_size_mib")
    )
    return alternative if has_resource_gain else None


def _percent_delta(candidate_value: float, baseline_value: float) -> float | None:
    if baseline_value == 0:
        return None
    return (candidate_value - baseline_value) / baseline_value * 100.0


def _effect_text(
    metric: str,
    candidate_value: float,
    baseline_value: float,
    direction: str | None,
) -> tuple[str, str]:
    delta = candidate_value - baseline_value
    if math.isclose(delta, 0.0, rel_tol=1e-9, abs_tol=1e-12):
        return "Held", "effect-held"
    percent = _percent_delta(candidate_value, baseline_value)
    change = (
        f"{abs(percent):.2f}% {'higher' if delta > 0 else 'lower'}"
        if percent is not None
        else f"{_format_number(abs(delta))} {'higher' if delta > 0 else 'lower'}"
    )
    improved = (direction == "max" and delta > 0) or (direction == "min" and delta < 0)
    regressed = (direction == "max" and delta < 0) or (direction == "min" and delta > 0)
    if improved:
        return f"Better · {change}", "effect-better"
    if regressed:
        return f"Tradeoff · {change}", "effect-tradeoff"
    return f"Changed · {change}", "effect-held"


def _evidence_banner(benchmarks: BenchmarkSet) -> str:
    classification = benchmarks.metadata.get("classification")
    if benchmarks.synthetic:
        return (
            '<aside class="evidence-banner" role="alert">'
            "<strong>Synthetic evidence — do not cite as measured Arm performance.</strong> "
            "This report demonstrates the decision workflow only."
            "</aside>"
        )
    if classification == "exploratory":
        return (
            '<aside class="evidence-banner" role="alert">'
            "<strong>Exploratory evidence — not canonical submission evidence.</strong> "
            "Rerun the study canonically before citing its measurements."
            "</aside>"
        )
    return ""


def _source_type(benchmarks: BenchmarkSet) -> str:
    if benchmarks.synthetic:
        return "Synthetic fixture"
    classification = benchmarks.metadata.get("classification")
    if classification == "canonical":
        return "Canonical measured evidence"
    if classification == "exploratory":
        return "Exploratory measured evidence"
    return "Measured evidence"


def _resource_facts(baseline: Candidate, alternative: Candidate) -> str:
    facts: list[str] = []
    for metric in (
        "model_size_mib",
        "peak_rss_mib",
        "ttft_ms_p95",
        "prompt_tokens_per_second",
        "prompt_tps",
    ):
        if metric not in baseline.metrics or metric not in alternative.metrics:
            continue
        baseline_value = baseline.metrics[metric]
        alternative_value = alternative.metrics[metric]
        delta = _percent_delta(alternative_value, baseline_value)
        if delta is None or math.isclose(delta, 0.0, rel_tol=1e-9, abs_tol=1e-12):
            continue
        word = "higher" if delta > 0 else "lower"
        facts.append(
            "<li>"
            f"<strong>{abs(delta):.1f}% {word}</strong> "
            f"{_escape(_metric_label(metric).lower())}"
            "</li>"
        )
        if len(facts) == 3:
            break
    if not facts:
        return "<p>No distinct measured resource gain is present.</p>"
    return '<ul class="fact-list">' + "".join(facts) + "</ul>"


def _verdict_section(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
    selected: Candidate,
    selection: Mapping[str, Any],
    alternative: Candidate | None,
) -> str:
    objective = _mapping(recommendation["objective"], "recommendation.objective")
    baseline_retained = selected.candidate_id == benchmarks.baseline_id
    preference_changed = selection.get("preference_changed_winner") is True
    if baseline_retained and preference_changed:
        verdict = (
            "ParetoPilot retained the measured baseline under the predeclared "
            "objective tolerance and preference policy."
        )
    elif baseline_retained:
        verdict = (
            "ParetoPilot retained the measured baseline because no alternative "
            "delivered a better eligible objective result on the declared frontier."
        )
    else:
        verdict = (
            f"{selected.label} was selected over the baseline for the declared "
            f"{_metric_label(str(objective['metric'])).lower()} objective."
        )

    if alternative is None:
        alternative_markup = (
            '<section class="verdict-column alternative-column" '
            'aria-labelledby="resource-alternative-heading">'
            '<p class="context-label">Measured resource alternative</p>'
            '<h2 id="resource-alternative-heading">No distinct alternative</h2>'
            "<p>The benchmark set does not contain another candidate with a measured "
            "memory or model-size reduction.</p>"
            "</section>"
        )
    else:
        alternative_markup = (
            '<section class="verdict-column alternative-column" '
            'aria-labelledby="resource-alternative-heading">'
            '<p class="context-label">Measured resource alternative · not selected</p>'
            f'<h2 id="resource-alternative-heading">{_escape(alternative.label)}</h2>'
            f"{_resource_facts(benchmarks.baseline, alternative)}"
            '<p class="column-note">This is a descriptive comparison, not a second '
            "recommendation. Its tradeoffs remain visible below.</p>"
            "</section>"
        )

    return (
        '<section class="verdict-layout" aria-label="Decision summary">'
        '<section class="verdict-column canonical-column" aria-labelledby="canonical-heading">'
        '<p class="context-label">Canonical recommendation</p>'
        f'<h2 id="canonical-heading">{_escape(selected.label)}</h2>'
        f'<p class="verdict-copy">{_escape(verdict)}</p>'
        '<dl class="decision-pair">'
        f"<div><dt>Objective</dt><dd>{_escape(_metric_label(str(objective['metric'])))}</dd></div>"
        f"<div><dt>Direction</dt><dd>{'Minimize' if objective['direction'] == 'min' else 'Maximize'}</dd></div>"
        "</dl>"
        "</section>"
        f"{alternative_markup}"
        "</section>"
    )


def _tolerance_rows(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> str:
    objective = _mapping(recommendation["objective"], "recommendation.objective")
    metric = str(objective["metric"])
    shortlist = set(str(item) for item in selection["shortlist_ids"])
    numeric_best_id = str(selection["numeric_best_id"])
    rows: list[str] = []
    for candidate in benchmarks.candidates:
        value = candidate.metrics.get(metric)
        if value is None:
            continue
        labels: list[str] = []
        if candidate.candidate_id == numeric_best_id:
            labels.append("Numeric best")
        if candidate.candidate_id in shortlist:
            labels.append("Within tolerance")
        if candidate.candidate_id == recommendation["selected_id"]:
            labels.append("Selected")
        if not labels:
            labels.append("Outside shortlist")
        rows.append(
            "<tr>"
            f'<th scope="row">{_escape(candidate.label)}</th>'
            f"<td>{_escape(_format_metric(metric, value))}</td>"
            f"<td>{_escape(' · '.join(labels))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _why_section(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> str:
    objective = _mapping(recommendation["objective"], "recommendation.objective")
    metric = str(objective["metric"])
    direction = str(objective["direction"])
    numeric_best = benchmarks.by_id(str(selection["numeric_best_id"]))
    best_value = numeric_best.metrics[metric]
    tolerance = float(selection["objective_tolerance_percent"])
    margin = abs(best_value) * tolerance / 100.0
    boundary = best_value + margin if direction == "min" else best_value - margin
    boundary_word = "at or below" if direction == "min" else "at or above"

    return (
        '<section class="report-section" aria-labelledby="why-heading">'
        '<div class="section-heading"><h2 id="why-heading">Why this decision held</h2>'
        "<p>The tolerance and preference policy are part of the decision, not a "
        "post-hoc explanation.</p></div>"
        '<div class="why-layout">'
        '<div class="reason-block">'
        '<p class="context-label">Predeclared objective tolerance</p>'
        f'<p class="reason-number">{_escape(f"{tolerance:.2f}%")}</p>'
        f"<p>The numeric best was <strong>{_escape(numeric_best.label)}</strong> at "
        f"<strong>{_escape(_format_metric(metric, best_value))}</strong>. The shortlist "
        f"accepted values {boundary_word} "
        f"<strong>{_escape(_format_metric(metric, boundary))}</strong>.</p>"
        f'<p class="reason-copy">{_escape(selection["reason"])}</p>'
        "</div>"
        '<div class="table-scroll" tabindex="0" role="region" '
        'aria-label="Scrollable objective tolerance comparison">'
        '<table class="compact-table tolerance-table">'
        "<caption>Objective values and shortlist membership</caption>"
        '<thead><tr><th scope="col">Candidate</th><th scope="col">'
        f'{_escape(_metric_label(metric))}</th><th scope="col">Decision role</th></tr></thead>'
        f"<tbody>{_tolerance_rows(benchmarks, recommendation, selection)}</tbody>"
        "</table></div>"
        "</div>"
        "</section>"
    )


def _tradeoff_rows(
    baseline: Candidate,
    alternative: Candidate,
    directions: Mapping[str, str],
) -> str:
    rows: list[str] = []
    for metric in _METRIC_PRIORITY:
        if metric not in baseline.metrics or metric not in alternative.metrics:
            continue
        baseline_value = baseline.metrics[metric]
        alternative_value = alternative.metrics[metric]
        if math.isclose(
            baseline_value,
            alternative_value,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            continue
        effect, effect_class = _effect_text(
            metric,
            alternative_value,
            baseline_value,
            directions.get(metric),
        )
        rows.append(
            '<div class="tradeoff-row" role="row">'
            f'<div class="tradeoff-metric" role="rowheader">{_escape(_metric_label(metric))}</div>'
            '<div class="tradeoff-value baseline-value" role="cell">'
            f'<span class="value-label">Baseline</span>{_escape(_format_metric(metric, baseline_value))}'
            "</div>"
            '<div class="tradeoff-connector" aria-hidden="true">'
            '<span class="baseline-marker"></span><span class="connector-line"></span>'
            '<span class="alternative-marker"></span></div>'
            '<div class="tradeoff-value alternative-value" role="cell">'
            f'<span class="value-label">Alternative</span>{_escape(_format_metric(metric, alternative_value))}'
            "</div>"
            f'<div class="tradeoff-effect {effect_class}" role="cell">{_escape(effect)}</div>'
            "</div>"
        )
    return "".join(rows)


def _tradeoffs_section(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
    alternative: Candidate | None,
) -> str:
    if alternative is None:
        body = (
            '<div class="no-data" role="status"><strong>No resource comparison available.</strong> '
            "At least two candidates with shared measurements are required.</div>"
        )
    else:
        rows = _tradeoff_rows(
            benchmarks.baseline,
            alternative,
            _frontier_directions(recommendation),
        )
        body = (
            '<div class="tradeoff-board" role="table" '
            'aria-label="Baseline and resource alternative tradeoffs">'
            '<div class="sr-only" role="row"><span role="columnheader">Metric</span>'
            '<span role="columnheader">Baseline</span><span role="columnheader">Visual link</span>'
            '<span role="columnheader">Alternative</span><span role="columnheader">Effect</span></div>'
            f"{rows}"
            "</div>"
            if rows
            else (
                '<div class="no-data" role="status"><strong>No differing shared metrics.</strong> '
                "The complete evidence table still lists every measurement.</div>"
            )
        )
    alternative_name = alternative.label if alternative is not None else "resource alternative"
    return (
        '<section class="report-section" aria-labelledby="tradeoffs-heading">'
        '<div class="section-heading"><h2 id="tradeoffs-heading">Aligned tradeoffs</h2>'
        f"<p>The measured baseline is compared with {_escape(alternative_name)}. "
        "“Better” and “Tradeoff” use only declared frontier directions.</p></div>"
        f"{body}"
        "</section>"
    )


def _profile_panel(
    profile: _DecisionProfile,
    benchmarks: BenchmarkSet,
    *,
    index: int,
) -> str:
    selected = benchmarks.by_id(profile.selected_id)
    metrics = [metric for metric in _METRIC_PRIORITY if metric in selected.metrics][:5]
    metric_items = "".join(
        "<li>"
        f"<span>{_escape(_metric_label(metric))}</span>"
        f"<strong>{_escape(_format_metric(metric, selected.metrics[metric]))}</strong>"
        "</li>"
        for metric in metrics
    )
    origin = "Derived policy scenario" if profile.derived else "Canonical recommendation"
    notice = profile.scenario_notice or (
        "This scenario applies a different precomputed policy to the same evidence. "
        "It does not replace the canonical decision."
        if profile.derived
        else "This is the canonical predeclared decision."
    )
    derived_note = f'<p class="derived-note">{_escape(notice)}</p>'
    hidden = " hidden" if index else ""
    return (
        f'<section id="profile-panel-{index}" class="profile-panel" role="tabpanel" '
        f'aria-labelledby="profile-tab-{index}" data-profile-panel="{index}"{hidden}>'
        '<div class="profile-decision">'
        f'<p class="context-label">{_escape(origin)}</p>'
        f"<h3>{_escape(selected.label)}</h3>"
        f"<p>{_escape(profile.description)}</p>"
        '<dl class="decision-pair">'
        f"<div><dt>Objective</dt><dd>{_escape(_metric_label(profile.objective_metric))}</dd></div>"
        f"<div><dt>Direction</dt><dd>{'Minimize' if profile.objective_direction == 'min' else 'Maximize'}</dd></div>"
        "</dl>"
        f'<p class="profile-reason">{_escape(profile.reason)}</p>'
        f"{derived_note}"
        "</div>"
        f'<ul class="profile-metrics" aria-label="{_escape(selected.label)} key measurements">'
        f"{metric_items}</ul>"
        "</section>"
    )


def _policy_section(
    profiles: tuple[_DecisionProfile, ...],
    benchmarks: BenchmarkSet,
) -> tuple[str, str]:
    tabs = "".join(
        f'<button id="profile-tab-{index}" type="button" role="tab" '
        f'aria-selected="{"true" if index == 0 else "false"}" '
        f'aria-controls="profile-panel-{index}" data-profile-target="{index}" '
        f'tabindex="{"0" if index == 0 else "-1"}">'
        f"{_escape(profile.label)}"
        f"<span>{'Canonical' if not profile.derived else 'Derived'}</span>"
        "</button>"
        for index, profile in enumerate(profiles)
    )
    panels = "".join(
        _profile_panel(profile, benchmarks, index=index) for index, profile in enumerate(profiles)
    )
    if len(profiles) > 1:
        script = """
  <script>
  (() => {
    "use strict";
    const tabs = document.querySelectorAll("[data-profile-target]");
    const panels = document.querySelectorAll("[data-profile-panel]");
    for (const tab of tabs) {
      tab.addEventListener("click", () => {
        const target = tab.getAttribute("data-profile-target");
        for (const item of tabs) {
          const active = item === tab;
          item.setAttribute("aria-selected", active ? "true" : "false");
          item.setAttribute("tabindex", active ? "0" : "-1");
        }
        for (const panel of panels) {
          panel.hidden = panel.getAttribute("data-profile-panel") !== target;
        }
        tab.focus();
      });
      tab.addEventListener("keydown", (event) => {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
        event.preventDefault();
        const items = Array.from(tabs);
        const step = event.key === "ArrowRight" ? 1 : -1;
        const next = (items.indexOf(tab) + step + items.length) % items.length;
        items[next].click();
      });
    }
  })();
  </script>
"""
    else:
        script = ""
    section = (
        '<section class="report-section" aria-labelledby="policies-heading">'
        '<div class="section-heading"><h2 id="policies-heading">Deployment policy selector</h2>'
        "<p>Every panel below is precomputed. The browser only switches views; it does "
        "not recalculate or rank candidates.</p></div>"
        f'<div class="profile-tabs" role="tablist" aria-label="Deployment policies">{tabs}</div>'
        f"{panels}"
        "</section>"
    )
    return section, script


def _scaled_positions(
    values: Sequence[float],
    start: float,
    end: float,
    *,
    domain_min: float | None = None,
    domain_max: float | None = None,
) -> list[float]:
    low = min(values) if domain_min is None else domain_min
    high = max(values) if domain_max is None else domain_max
    if math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-12):
        return [(start + end) / 2.0 for _ in values]
    return [start + (value - low) / (high - low) * (end - start) for value in values]


def _marker(
    x: float,
    y: float,
    *,
    series_index: int,
    css_class: str,
) -> str:
    shape = series_index % 3
    if shape == 1:
        return (
            f'<rect class="{css_class}" x="{x - 4:.2f}" y="{y - 4:.2f}" '
            'width="8" height="8"></rect>'
        )
    if shape == 2:
        return (
            f'<path class="{css_class}" d="M {x:.2f} {y - 5:.2f} '
            f'L {x + 5:.2f} {y + 4:.2f} L {x - 5:.2f} {y + 4:.2f} Z"></path>'
        )
    return f'<circle class="{css_class}" cx="{x:.2f}" cy="{y:.2f}" r="4.5"></circle>'


def _chart_table(
    rows: Sequence[_LoadRow],
    benchmarks: BenchmarkSet,
    metric: str,
    *,
    caption: str,
) -> str:
    table_rows = "".join(
        "<tr>"
        f'<th scope="row">{_escape(benchmarks.by_id(row.candidate_id).label)}</th>'
        f"<td>{row.concurrency}</td>"
        f"<td>{_escape(_format_metric(metric, row.metrics[metric]))}</td>"
        "</tr>"
        for row in rows
        if metric in row.metrics
    )
    return (
        '<table class="sr-only">'
        f"<caption>{_escape(caption)}</caption>"
        '<thead><tr><th scope="col">Candidate</th><th scope="col">Concurrency</th>'
        f'<th scope="col">{_escape(_metric_label(metric))}</th></tr></thead>'
        f"<tbody>{table_rows}</tbody></table>"
    )


def _line_chart(
    load: _LoadSweep,
    benchmarks: BenchmarkSet,
    *,
    metric: str,
    chart_id: str,
    title: str,
    description: str,
) -> str:
    metric_rows = [row for row in load.rows if metric in row.metrics]
    if not metric_rows:
        return ""
    candidate_ids = sorted({row.candidate_id for row in metric_rows})
    concurrency_values = [float(row.concurrency) for row in metric_rows]
    metric_values = [row.metrics[metric] for row in metric_rows]
    x_positions = _scaled_positions(
        concurrency_values,
        76.0,
        590.0,
        domain_min=min(concurrency_values),
        domain_max=max(concurrency_values),
    )
    maximum = max(metric_values)
    y_domain_max = maximum * 1.08 if maximum > 0 else 1.0
    y_positions = _scaled_positions(
        metric_values,
        258.0,
        30.0,
        domain_min=0.0,
        domain_max=y_domain_max,
    )
    positions = {
        (row.candidate_id, row.concurrency): (x_pos, y_pos)
        for row, x_pos, y_pos in zip(
            metric_rows,
            x_positions,
            y_positions,
            strict=True,
        )
    }
    series_rows = {
        candidate_id: sorted(
            (row for row in metric_rows if row.candidate_id == candidate_id),
            key=lambda item: item.concurrency,
        )
        for candidate_id in candidate_ids
    }
    last_points = [
        positions[(candidate_id, series_rows[candidate_id][-1].concurrency)]
        for candidate_id in candidate_ids
    ]
    label_positions = _adjust_label_positions([point[1] for point in last_points])
    series_markup: list[str] = []
    for series_index, (candidate_id, label_y) in enumerate(
        zip(candidate_ids, label_positions, strict=True)
    ):
        rows = series_rows[candidate_id]
        points = [positions[(row.candidate_id, row.concurrency)] for row in rows]
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        css_class = f"series-{series_index % 6}"
        line = (
            f'<polyline class="chart-line {css_class}" points="{point_text}"></polyline>'
            if len(points) > 1
            else ""
        )
        marks = "".join(
            _marker(x, y, series_index=series_index, css_class=f"chart-marker {css_class}")
            for x, y in points
        )
        last_x, last_y = points[-1]
        label = benchmarks.by_id(candidate_id).label
        series_markup.append(
            f'<g aria-label="{_escape(label)}">{line}{marks}'
            f'<line class="label-leader {css_class}" x1="{last_x + 6:.2f}" '
            f'y1="{last_y:.2f}" x2="614" y2="{label_y:.2f}"></line>'
            f'<text class="direct-label" x="620" y="{label_y + 4:.2f}">'
            f"{_escape(label)}</text></g>"
        )

    x_low = min(int(row.concurrency) for row in metric_rows)
    x_high = max(int(row.concurrency) for row in metric_rows)
    return (
        '<figure class="chart-figure">'
        f'<svg viewBox="0 0 800 310" role="img" '
        f'aria-labelledby="{chart_id}-title {chart_id}-description">'
        f'<title id="{chart_id}-title">{_escape(title)}</title>'
        f'<desc id="{chart_id}-description">{_escape(description)}</desc>'
        '<line class="chart-axis" x1="76" y1="258" x2="590" y2="258"></line>'
        '<line class="chart-axis" x1="76" y1="30" x2="76" y2="258"></line>'
        '<line class="chart-grid" x1="76" y1="144" x2="590" y2="144"></line>'
        f'<text class="chart-tick" x="76" y="278">{x_low}</text>'
        f'<text class="chart-tick chart-tick-end" x="590" y="278">{x_high}</text>'
        '<text class="chart-axis-label" x="333" y="302">Concurrent requests</text>'
        '<text class="chart-tick" x="66" y="262">0</text>'
        f'<text class="chart-tick" x="66" y="34">{_escape(_format_metric(metric, maximum))}</text>'
        f"{''.join(series_markup)}"
        "</svg>"
        f"<figcaption>{_escape(description)}</figcaption>"
        f"{_chart_table(metric_rows, benchmarks, metric, caption=title + ' data')}"
        "</figure>"
    )


def _load_table(load: _LoadSweep, benchmarks: BenchmarkSet) -> str:
    available = [
        metric for metric in _LOAD_METRICS if any(metric in row.metrics for row in load.rows)
    ]
    metric_headers = "".join(
        f'<th scope="col">{_escape(_metric_label(metric))}</th>' for metric in available
    )
    rows = []
    for row in load.rows:
        cells = "".join(
            "<td>"
            + (
                _escape(_format_metric(metric, row.metrics[metric]))
                if metric in row.metrics
                else '<span class="not-measured">Not measured</span>'
            )
            + "</td>"
            for metric in available
        )
        rows.append(
            "<tr>"
            f'<th scope="row">{_escape(benchmarks.by_id(row.candidate_id).label)}</th>'
            f"<td>{row.concurrency}</td>"
            f"<td>{row.request_count}</td>"
            f"<td>{row.completed_requests}</td>"
            f"<td>{row.failed_requests}</td>"
            f"{cells}"
            f"<td>{'Met' if row.slo_met else 'Not met'}</td>"
            "<td><details><summary>View samples</summary>"
            f'<code class="json-block">{_escape(_stable_json(row.samples))}</code>'
            "</details></td></tr>"
        )
    return (
        '<div class="table-scroll" tabindex="0" role="region" '
        'aria-label="Scrollable load sweep evidence">'
        '<table class="evidence-table load-table">'
        "<caption>Validated measurements at each concurrency level</caption>"
        '<thead><tr><th scope="col">Candidate</th><th scope="col">Concurrency</th>'
        '<th scope="col">Measured requests</th><th scope="col">Completed requests</th>'
        '<th scope="col">Failed requests</th>'
        f'{metric_headers}<th scope="col">SLO status</th>'
        '<th scope="col">Samples</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _key_value_table(
    values: Mapping[str, Any],
    *,
    caption: str,
    table_class: str = "compact-table",
) -> str:
    rows = "".join(
        "<tr>"
        f'<th scope="row">{_escape(_metric_label(str(key)))}</th>'
        f"<td><code>{_escape(_stable_json(values[key]))}</code></td>"
        "</tr>"
        for key in sorted(values, key=str)
    )
    return (
        f'<table class="{table_class}"><caption>{_escape(caption)}</caption>'
        '<thead><tr><th scope="col">Field</th><th scope="col">Value</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>"
    )


def _load_binding_table(load: _LoadSweep, benchmarks: BenchmarkSet) -> str:
    bindings = load.evidence_bindings
    if bindings is None:
        return (
            '<div class="no-data" role="status"><strong>No load command binding was '
            "supplied.</strong> This is permitted only for synthetic report fixtures.</div>"
        )
    if "candidate_server_configurations" in bindings:
        configurations = _mapping(
            bindings.get("candidate_server_configurations"),
            "load evidence candidate_server_configurations",
        )
        request_base_urls = _mapping(
            bindings.get("candidate_request_base_urls"),
            "load evidence candidate_request_base_urls",
        )
    else:
        configurations = {
            load.rows[0].candidate_id: _mapping(
                bindings.get("server_configuration"),
                "load evidence server_configuration",
            )
        }
        request_base_urls = {load.rows[0].candidate_id: str(bindings["request_base_url"])}
    rows: list[str] = []
    for candidate_id in sorted(configurations):
        configuration = _mapping(
            configurations[candidate_id],
            f"load evidence configuration {candidate_id}",
        )
        differences = configuration.get("differing_binding_options")
        assert isinstance(differences, Sequence)
        rows.append(
            "<tr>"
            f'<th scope="row">{_escape(benchmarks.by_id(str(candidate_id)).label)}</th>'
            f"<td><code>{_escape(request_base_urls[candidate_id])}</code></td>"
            f"<td>{_escape(configuration['canonical_parallel'])}</td>"
            f"<td>{_escape(', '.join(str(value) for value in differences) or 'none')}</td>"
            f"<td>{_hash_markup(str(configuration['load_server_command_sha256']))}</td>"
            f"<td>{_hash_markup(str(configuration['canonical_server_command_sha256']))}</td>"
            "</tr>"
        )
    return (
        '<p class="evidence-limit"><strong>Configuration binding:</strong> every measured '
        "load command was validated as materially identical to its canonical deployment "
        "command. Only the declared host or port binding may differ.</p>"
        '<dl class="hash-list binding-plan"><div><dt>Load plan SHA-256</dt>'
        f"<dd>{_hash_markup(str(bindings['plan_sha256']))}</dd></div></dl>"
        '<div class="table-scroll" tabindex="0" role="region" '
        'aria-label="Scrollable load command bindings"><table class="compact-table">'
        "<caption>Validated load-to-deployment command bindings</caption>"
        '<thead><tr><th scope="col">Candidate</th><th scope="col">Request endpoint</th>'
        '<th scope="col">Canonical parallel</th>'
        '<th scope="col">Allowed differences used</th><th scope="col">Load command SHA-256</th>'
        '<th scope="col">Canonical command SHA-256</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _load_section(load: _LoadSweep | None, benchmarks: BenchmarkSet) -> str:
    if load is None:
        content = (
            '<div class="no-data" role="status"><strong>No load sweep was supplied.</strong> '
            "The canonical single-request evidence remains valid, but this report does not "
            "infer concurrent cloud behavior.</div>"
        )
    elif not load.rows:
        content = (
            '<div class="no-data" role="status"><strong>The load sweep contains no rows.</strong> '
            "No concurrency curve can be drawn.</div>"
        )
    else:
        chart_specs = (
            (
                "requests_per_second",
                "load-request-throughput",
                "Request throughput by concurrency",
                "Measured completed requests per second as concurrent request count increases.",
            ),
            (
                "generated_tokens_per_second",
                "load-token-throughput",
                "Generated-token throughput by concurrency",
                "Measured generated tokens per second as concurrent request count increases.",
            ),
            (
                "e2e_latency_ms_p95",
                "load-tail-latency",
                "p95 end-to-end latency by concurrency",
                "Measured p95 end-to-end response latency as concurrent request count increases.",
            ),
        )
        charts = "".join(
            _line_chart(
                load,
                benchmarks,
                metric=metric,
                chart_id=chart_id,
                title=title,
                description=description,
            )
            for metric, chart_id, title, description in chart_specs
            if any(metric in row.metrics for row in load.rows)
        )
        context_tables: list[str] = []
        context_tables.append(
            '<div class="load-context load-binding"><h3>Command and plan binding</h3>'
            + _load_binding_table(load, benchmarks)
            + "</div>"
        )
        highest_labels = {
            benchmarks.by_id(candidate_id).label: (
                "No measured level met the SLO" if value is None else value
            )
            for candidate_id, value in load.highest_slo_concurrency.items()
        }
        context_tables.append(
            '<div class="load-context"><h3>Highest SLO-passing concurrency</h3>'
            + _key_value_table(
                highest_labels,
                caption="Highest measured concurrency meeting all declared SLO gates",
            )
            + "</div>"
        )
        if load.slo:
            context_tables.append(
                '<div class="load-context"><h3>Declared service-level objective</h3>'
                + _key_value_table(load.slo, caption="Load sweep service-level objective")
                + "</div>"
            )
        context_tables.append(
            '<div class="load-context"><h3>Measured methodology</h3>'
            + _key_value_table(load.methodology, caption="Load sweep methodology")
            + "</div>"
        )
        context_markup = (
            '<div class="load-context-grid">' + "".join(context_tables) + "</div>"
            if context_tables
            else ""
        )
        content = (
            f"{context_markup}"
            f'<div class="chart-grid-layout">{charts}</div>'
            f"{_load_table(load, benchmarks)}"
        )

    return (
        '<section class="report-section" aria-labelledby="load-heading">'
        '<div class="section-heading"><h2 id="load-heading">Measured load behavior</h2>'
        "<p>Curves appear only for supplied, validated measurements. Missing values are "
        "not interpolated or estimated.</p></div>"
        f"{content}"
        "</section>"
    )


def _repeat_coverage(benchmarks: BenchmarkSet) -> Mapping[str, Mapping[str, tuple[int, ...]]]:
    raw_evidence = benchmarks.metadata.get("candidate_evidence")
    if not isinstance(raw_evidence, Mapping):
        return {}
    result: dict[str, Mapping[str, tuple[int, ...]]] = {}
    for candidate in benchmarks.candidates:
        raw_candidate = raw_evidence.get(candidate.candidate_id)
        if not isinstance(raw_candidate, Mapping):
            continue
        raw_artifacts = raw_candidate.get("artifacts")
        if not isinstance(raw_artifacts, Mapping):
            continue
        passes: dict[str, set[int]] = {
            "throughput": set(),
            "server_evaluation": set(),
            "server_time": set(),
        }
        for raw_name in raw_artifacts:
            if not isinstance(raw_name, str):
                continue
            match = _PASS_RE.fullmatch(raw_name)
            if match is not None:
                passes[match.group(1)].add(int(match.group(2)))
        if any(passes.values()):
            result[candidate.candidate_id] = {
                name: tuple(sorted(numbers)) for name, numbers in passes.items()
            }
    return result


def _normalise_stability_summary(
    benchmarks: BenchmarkSet,
    stability_summary: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if stability_summary is None:
        return None
    raw = _mapping(stability_summary, "stability_summary")
    validate_stability_summary(
        raw,
        require_input_fingerprints=not benchmarks.synthetic,
    )
    if raw["baseline_id"] != benchmarks.baseline_id:
        raise ValidationError("stability_summary.baseline_id does not match benchmarks.baseline_id")
    if raw["synthetic"] is not benchmarks.synthetic:
        raise ValidationError("stability_summary.synthetic does not match benchmarks.synthetic")
    benchmark_ids = {candidate.candidate_id for candidate in benchmarks.candidates}
    summary_ids = {str(row["candidate_id"]) for row in raw["rows"] if isinstance(row, Mapping)}
    if summary_ids != benchmark_ids:
        raise ValidationError("stability_summary candidate ids must match benchmarks exactly")
    directions = _mapping(
        raw["metric_directions"],
        "stability_summary.metric_directions",
    )
    for candidate in benchmarks.candidates:
        missing = sorted(set(directions) - set(candidate.metrics))
        if missing:
            raise ValidationError(
                f"stability_summary metrics are missing from candidate "
                f"{candidate.candidate_id!r}: {', '.join(missing)}"
            )
    configuration_fingerprints = _mapping(
        raw["candidate_configuration_fingerprints"],
        "stability_summary.candidate_configuration_fingerprints",
    )
    for candidate in benchmarks.candidates:
        if configuration_fingerprints.get(
            candidate.candidate_id
        ) != candidate_configuration_fingerprint(candidate):
            raise ValidationError(
                "stability_summary candidate configuration fingerprint does not "
                f"match benchmark candidate {candidate.candidate_id!r}"
            )
    return raw


def _pass_text(values: tuple[int, ...]) -> str:
    if not values:
        return "Not linked"
    return f"{len(values)} linked ({', '.join(f'pass {value}' for value in values)})"


def _stability_pass_values(row: Mapping[str, Any]) -> str:
    values = []
    metric = str(row["metric"])
    for raw_pass in row["pass_values"]:
        pass_value = _mapping(raw_pass, "stability pass value")
        delta = pass_value["relative_delta_percent"]
        delta_text = "percentage unavailable" if delta is None else f"{float(delta):+.2f}%"
        values.append(
            "<li>"
            f"<strong>{_escape(pass_value['pass_label'])}</strong>"
            f"<span>{_escape(_format_metric(metric, float(pass_value['candidate_value'])))} "
            f"vs {_escape(_format_metric(metric, float(pass_value['baseline_value'])))}</span>"
            f"<span>{_escape(delta_text)} · {_escape(pass_value['comparison'])}</span>"
            "</li>"
        )
    return '<ul class="pass-deltas">' + "".join(values) + "</ul>"


def _stability_table(
    benchmarks: BenchmarkSet,
    stability_summary: Mapping[str, Any],
) -> str:
    rows = []
    for raw_row in stability_summary["rows"]:
        row = _mapping(raw_row, "stability row")
        spread = row["relative_spread_percent"]
        spread_text = "Percentage unavailable" if spread is None else f"{float(spread):.2f}%"
        rows.append(
            "<tr>"
            f'<th scope="row">{_escape(benchmarks.by_id(str(row["candidate_id"])).label)}</th>'
            f"<td>{_escape(_metric_label(str(row['metric'])))}</td>"
            f"<td>{'Lower is better' if row['direction'] == 'min' else 'Higher is better'}</td>"
            f"<td>{_stability_pass_values(row)}</td>"
            f"<td>{_escape(spread_text)}</td>"
            f"<td>{_escape(row['consistency'])}</td>"
            f"<td>{_escape(row['overall_direction'])}</td>"
            "</tr>"
        )
    return (
        '<div class="table-scroll" tabindex="0" role="region" '
        'aria-label="Scrollable pass-level stability evidence">'
        '<table class="evidence-table stability-table">'
        "<caption>Observed pass directions, relative deltas, and relative spread</caption>"
        '<thead><tr><th scope="col">Candidate</th><th scope="col">Metric</th>'
        '<th scope="col">Direction</th><th scope="col">Pass values and deltas</th>'
        '<th scope="col">Relative spread</th><th scope="col">Consistency</th>'
        '<th scope="col">Overall direction</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _repeat_section(
    benchmarks: BenchmarkSet,
    stability_summary: Mapping[str, Any] | None,
) -> str:
    if stability_summary is not None:
        pass_labels = ", ".join(_escape(label) for label in stability_summary["pass_labels"])
        intro = (
            "Validated pass-level values show whether each measured direction repeated "
            "and how widely the candidate values spread."
        )
        content = (
            '<div class="stability-method">'
            f"<p><strong>Observed passes:</strong> {pass_labels}</p>"
            f"<p><strong>Method:</strong> {_escape(stability_summary['method'])}.</p>"
            "</div>"
            f"{_stability_table(benchmarks, stability_summary)}"
            '<p class="evidence-limit"><strong>Interpretation limit:</strong> consistency '
            "and relative spread describe these observed passes only. They are not a "
            "statistical-significance claim.</p>"
        )
    else:
        coverage = _repeat_coverage(benchmarks)
        intro = (
            "No validated pass-level stability summary was supplied. Linked artifacts "
            "below show coverage only, not observed variation."
        )
        if not coverage:
            content = (
                '<div class="no-data" role="status"><strong>Pass-level variation was not '
                "supplied, and repeat artifacts are not linked.</strong> Aggregate "
                "measurements are shown without a repeat-stability claim.</div>"
            )
        else:
            rows = []
            for candidate in benchmarks.candidates:
                candidate_coverage = coverage.get(candidate.candidate_id)
                if candidate_coverage is None:
                    row = (
                        "<tr>"
                        f'<th scope="row">{_escape(candidate.label)}</th>'
                        '<td colspan="3" class="not-measured">No pass artifacts linked</td>'
                        "</tr>"
                    )
                else:
                    row = (
                        "<tr>"
                        f'<th scope="row">{_escape(candidate.label)}</th>'
                        f"<td>{_escape(_pass_text(candidate_coverage['throughput']))}</td>"
                        f"<td>{_escape(_pass_text(candidate_coverage['server_evaluation']))}</td>"
                        f"<td>{_escape(_pass_text(candidate_coverage['server_time']))}</td>"
                        "</tr>"
                    )
                rows.append(row)
            content = (
                '<div class="no-data coverage-notice" role="status"><strong>Coverage, '
                "not variation.</strong> Pass links prove artifacts were recorded; they "
                "do not reveal consistency or spread.</div>"
                '<div class="table-scroll" tabindex="0" role="region" '
                'aria-label="Scrollable repeat artifact coverage">'
                '<table class="compact-table repeat-table">'
                "<caption>Artifact coverage only; this is not a stability result</caption>"
                '<thead><tr><th scope="col">Candidate</th>'
                '<th scope="col">Throughput</th><th scope="col">Latency and quality</th>'
                '<th scope="col">Peak memory</th></tr></thead>'
                f"<tbody>{''.join(rows)}</tbody></table></div>"
                '<p class="evidence-limit"><strong>Interpretation limit:</strong> linked '
                "pass coverage proves repetition was recorded; aggregate values alone do "
                "not quantify between-pass variance or statistical significance.</p>"
            )
    return (
        '<section class="report-section" aria-labelledby="repeat-heading">'
        '<div class="section-heading"><h2 id="repeat-heading">Repeat stability</h2>'
        f"<p>{_escape(intro)}</p></div>"
        f"{content}"
        "</section>"
    )


def _generation_metric(benchmarks: BenchmarkSet) -> str | None:
    for metric in ("generation_tokens_per_second", "generation_tps"):
        if all(metric in candidate.metrics for candidate in benchmarks.candidates):
            return metric
    return None


def _adjust_label_positions(raw_positions: Sequence[float]) -> list[float]:
    if not raw_positions:
        return []
    indexed = sorted(enumerate(raw_positions), key=lambda item: item[1])
    adjusted: list[tuple[int, float]] = []
    minimum_gap = 21.0
    for index, raw in indexed:
        value = max(28.0, raw)
        if adjusted:
            value = max(value, adjusted[-1][1] + minimum_gap)
        adjusted.append((index, value))
    overflow = max(0.0, adjusted[-1][1] - 260.0)
    if overflow:
        adjusted = [(index, value - overflow) for index, value in adjusted]
    return [dict(adjusted)[index] for index in range(len(raw_positions))]


def _scatter_section(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
) -> str:
    generation_metric = _generation_metric(benchmarks)
    latency_metric = "e2e_latency_ms_p95"
    can_plot = (
        len(benchmarks.candidates) >= 2
        and generation_metric is not None
        and all(latency_metric in candidate.metrics for candidate in benchmarks.candidates)
    )
    title = "p95 end-to-end latency vs generation throughput"
    if not can_plot:
        content = (
            '<div class="no-data" role="status"><strong>Two-dimensional comparison '
            "unavailable.</strong> Every candidate needs p95 end-to-end latency and one "
            "common generation-throughput metric. The evidence table remains authoritative."
            "</div>"
        )
    else:
        assert generation_metric is not None
        candidates = list(benchmarks.candidates)
        latency = [candidate.metrics[latency_metric] for candidate in candidates]
        generation = [candidate.metrics[generation_metric] for candidate in candidates]
        x_positions = _scaled_positions(latency, 78.0, 548.0)
        y_positions = _scaled_positions(generation, 254.0, 30.0)
        label_positions = _adjust_label_positions(y_positions)
        selected_id = str(recommendation["selected_id"])
        frontier_ids = set(str(item) for item in recommendation.get("frontier_ids", ()))
        points: list[str] = []
        table_rows: list[str] = []
        for index, (candidate, x_pos, y_pos, label_y) in enumerate(
            zip(candidates, x_positions, y_positions, label_positions, strict=True)
        ):
            classes = [f"series-{index % 6}"]
            roles = []
            if candidate.candidate_id == benchmarks.baseline_id:
                classes.append("point-baseline")
                roles.append("Baseline")
            if candidate.candidate_id == selected_id:
                classes.append("point-selected")
                roles.append("Selected")
            if candidate.candidate_id in frontier_ids:
                roles.append("Declared frontier")
            if not roles:
                roles.append("Candidate")
            css_class = " ".join(classes)
            points.append(
                "<g>"
                + _marker(
                    x_pos,
                    y_pos,
                    series_index=index,
                    css_class=f"chart-marker {css_class}",
                )
                + f'<line class="scatter-leader {css_class}" x1="{x_pos + 6:.2f}" '
                f'y1="{y_pos:.2f}" x2="586" y2="{label_y:.2f}"></line>'
                f'<text class="direct-label" x="592" y="{label_y + 4:.2f}">'
                f"{_escape(candidate.label)}</text></g>"
            )
            table_rows.append(
                "<tr>"
                f'<th scope="row">{_escape(candidate.label)}</th>'
                f"<td>{_escape(_format_metric(latency_metric, latency[index]))}</td>"
                f"<td>{_escape(_format_metric(generation_metric, generation[index]))}</td>"
                f"<td>{_escape(' · '.join(roles))}</td>"
                "</tr>"
            )

        description = (
            "Each labeled point is one measured candidate. Left is lower p95 end-to-end "
            "latency; higher is greater generation throughput. This is a two-metric view, "
            "not the complete multi-objective frontier."
        )
        content = (
            '<figure class="chart-figure scatter-figure">'
            '<svg viewBox="0 0 800 310" role="img" '
            'aria-labelledby="scatter-title scatter-description">'
            f'<title id="scatter-title">{title}</title>'
            f'<desc id="scatter-description">{description}</desc>'
            '<line class="chart-axis" x1="78" y1="254" x2="548" y2="254"></line>'
            '<line class="chart-axis" x1="78" y1="30" x2="78" y2="254"></line>'
            f'<text class="chart-tick" x="78" y="276">{_escape(_format_number(min(latency)))}</text>'
            f'<text class="chart-tick chart-tick-end" x="548" y="276">'
            f"{_escape(_format_number(max(latency)))}</text>"
            '<text class="chart-axis-label" x="313" y="302">p95 end-to-end latency (ms)</text>'
            f'<text class="chart-tick" x="68" y="258">{_escape(_format_number(min(generation)))}</text>'
            f'<text class="chart-tick" x="68" y="34">{_escape(_format_number(max(generation)))}</text>'
            f"{''.join(points)}"
            "</svg>"
            f"<figcaption>{description}</figcaption>"
            "</figure>"
            '<div class="table-scroll" tabindex="0" role="region" '
            'aria-label="Scrollable latency and generation data">'
            '<table class="compact-table scatter-table">'
            f"<caption>{title} data</caption>"
            '<thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">p95 end-to-end latency</th>'
            '<th scope="col">Generation throughput</th><th scope="col">Decision role</th>'
            f"</tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>"
        )

    return (
        '<section class="report-section" aria-labelledby="scatter-heading">'
        f'<div class="section-heading"><h2 id="scatter-heading">{title}</h2>'
        "<p>This accurately scoped two-dimensional view complements, but does not "
        "replace, the full constraint and frontier analysis.</p></div>"
        f"{content}"
        "</section>"
    )


def _candidate_status(
    candidate_id: str,
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
) -> str:
    labels: list[str] = []
    if candidate_id == recommendation["selected_id"]:
        labels.append("Selected")
    if candidate_id == benchmarks.baseline_id:
        labels.append("Baseline")
    if candidate_id in set(str(item) for item in recommendation.get("frontier_ids", ())):
        labels.append("Declared frontier")
    if candidate_id in set(str(item) for item in recommendation.get("eligible_ids", ())):
        labels.append("Eligible")
    if candidate_id in recommendation.get("rejected", {}):
        labels.append("Rejected")
    return " · ".join(labels) if labels else "Not classified"


def _evidence_table(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
) -> str:
    metrics = _ordered_metrics(benchmarks)
    headers = "".join(
        f'<th scope="col">{_escape(_metric_label(metric))}</th>' for metric in metrics
    )
    rejected = recommendation.get("rejected", {})
    rows: list[str] = []
    for candidate in benchmarks.candidates:
        metric_cells = "".join(
            "<td>"
            + (
                _escape(_format_metric(metric, candidate.metrics[metric]))
                if metric in candidate.metrics
                else '<span class="not-measured">Not measured</span>'
            )
            + "</td>"
            for metric in metrics
        )
        reasons = rejected.get(candidate.candidate_id, ())
        reason_markup = (
            '<ul class="reason-list">'
            + "".join(f"<li>{_escape(reason)}</li>" for reason in reasons)
            + "</ul>"
            if reasons
            else '<span class="quiet">No declared constraint violations.</span>'
        )
        rows.append(
            "<tr>"
            '<th scope="row">'
            f'<span class="candidate-label">{_escape(candidate.label)}</span>'
            f"<code>{_escape(candidate.candidate_id)}</code></th>"
            f"<td>{_escape(_candidate_status(candidate.candidate_id, benchmarks, recommendation))}</td>"
            "<td><details><summary>View configuration</summary>"
            f'<code class="json-block">{_escape(_stable_json(candidate.parameters))}</code>'
            "</details></td>"
            f"{metric_cells}<td>{reason_markup}</td>"
            "</tr>"
        )
    return (
        '<div class="table-scroll" tabindex="0" role="region" '
        'aria-label="Scrollable full candidate evidence">'
        '<table class="evidence-table candidate-table">'
        "<caption>All candidate measurements, parameters, statuses, and rejection reasons</caption>"
        '<thead><tr><th scope="col">Candidate</th><th scope="col">Decision status</th>'
        f'<th scope="col">Parameters</th>{headers}<th scope="col">Constraint result</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _evidence_section(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
) -> str:
    return (
        '<section class="report-section" aria-labelledby="evidence-heading">'
        '<div class="section-heading"><h2 id="evidence-heading">Full evidence table</h2>'
        "<p>Missing values remain explicitly unmeasured. Rejection reasons are copied "
        "from the canonical recommendation record.</p></div>"
        f"{_evidence_table(benchmarks, recommendation)}"
        "</section>"
    )


def _deployment_command(candidate: Candidate, *, synthetic: bool) -> tuple[str | None, str]:
    if synthetic:
        return None, "Deployment export is disabled for synthetic evidence."
    raw = candidate.parameters.get("deployment_argv")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None, "No validated deployment argv is present for this candidate."
    if not raw or len(raw) > 256:
        return None, "The deployment argv must contain between 1 and 256 arguments."
    argv: list[str] = []
    for index, item in enumerate(raw):
        if (
            not isinstance(item, str)
            or not item
            or len(item) > 4096
            or "\x00" in item
            or "\r" in item
            or "\n" in item
        ):
            return (
                None,
                f"Deployment argument {index} is invalid; no command was exported.",
            )
        argv.append(item)
    return shlex.join(argv), ""


def _hash_value(value: str, context: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{context} must be a string")
    if value and _SHA256_RE.fullmatch(value) is None:
        raise ValidationError(f"{context} must be empty or a lowercase SHA-256 digest")
    return value


def _hash_markup(value: str) -> str:
    if not value:
        return '<span class="not-measured">Not supplied</span>'
    return f"<code>{_escape(value)}</code>"


def _metadata_table(benchmarks: BenchmarkSet) -> str:
    if not benchmarks.metadata:
        return (
            '<div class="no-data" role="status"><strong>No benchmark metadata was '
            "supplied.</strong></div>"
        )
    return (
        '<div class="table-scroll" tabindex="0" role="region" '
        'aria-label="Scrollable benchmark metadata">'
        + _key_value_table(
            benchmarks.metadata,
            caption="Source-supplied benchmark metadata",
            table_class="metadata-table",
        )
        + "</div>"
    )


def _trust_section(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
    selected: Candidate,
    *,
    benchmarks_sha256: str,
    recommendation_sha256: str,
    profiles_sha256: str,
    load_sha256: str,
    stability_sha256: str,
) -> str:
    command, command_reason = _deployment_command(selected, synthetic=benchmarks.synthetic)
    if command is None:
        deployment = (
            '<div class="no-data deployment-state" role="status">'
            f"<strong>No deployment command exported.</strong> {_escape(command_reason)}</div>"
        )
    else:
        deployment = (
            "<p>The command below reproduces only the selected server configuration. "
            "It was rendered from the validated argv vector; no flags or paths were guessed.</p>"
            f'<pre class="command"><code>{_escape(command)}</code></pre>'
        )
    paretopilot_version = recommendation.get("paretopilot_version", "Not supplied")
    return (
        '<section class="report-section trust-section" aria-labelledby="trust-heading">'
        '<div class="section-heading"><h2 id="trust-heading">Trust and reproduction</h2>'
        "<p>Hashes and source metadata anchor this deterministic document. No generation "
        "timestamp or network-loaded asset is added.</p></div>"
        '<div class="trust-grid">'
        '<div class="trust-block"><h3>Source anchors</h3>'
        '<dl class="hash-list">'
        f"<div><dt>Benchmark SHA-256</dt><dd>{_hash_markup(benchmarks_sha256)}</dd></div>"
        f"<div><dt>Recommendation SHA-256</dt><dd>{_hash_markup(recommendation_sha256)}</dd></div>"
        f"<div><dt>Policy profiles SHA-256</dt><dd>{_hash_markup(profiles_sha256)}</dd></div>"
        f"<div><dt>Load sweep SHA-256</dt><dd>{_hash_markup(load_sha256)}</dd></div>"
        f"<div><dt>Stability summary SHA-256</dt><dd>{_hash_markup(stability_sha256)}</dd></div>"
        f"<div><dt>Benchmark schema</dt><dd>{_escape(benchmarks.schema_version)}</dd></div>"
        f"<div><dt>ParetoPilot version</dt><dd>{_escape(paretopilot_version)}</dd></div>"
        f"<div><dt>Baseline ID</dt><dd><code>{_escape(benchmarks.baseline_id)}</code></dd></div>"
        f"<div><dt>Selected ID</dt><dd><code>{_escape(selected.candidate_id)}</code></dd></div>"
        "</dl></div>"
        '<div class="trust-block"><h3>Selected deployment argv</h3>'
        f"{deployment}</div>"
        "</div>"
        '<details class="metadata-details"><summary>View full source metadata</summary>'
        f"{_metadata_table(benchmarks)}</details>"
        '<div class="reproduction-note"><h3>Reproduction contract</h3>'
        "<ol><li>Verify the evidence archive and recorded source hashes.</li>"
        "<li>Rebuild the benchmark set and canonical recommendation from the validated inputs.</li>"
        "<li>Render into a fresh path and compare the resulting artifact hash.</li></ol>"
        "<p>The v1.1 renderer is opt-in. It does not alter the canonical v1.0 report path "
        "or its byte output.</p></div>"
        "</section>"
    )


_CSS = """
:root {
  color-scheme: light;
  --bg: oklch(1 0 0);
  --surface: oklch(0.972 0.008 230);
  --surface-strong: oklch(0.936 0.014 230);
  --ink: oklch(0.225 0.028 230);
  --muted: oklch(0.43 0.028 230);
  --line: oklch(0.86 0.018 230);
  --line-strong: oklch(0.64 0.045 230);
  --accent: oklch(0.49 0.15 230);
  --accent-dark: oklch(0.32 0.12 230);
  --accent-soft: oklch(0.94 0.035 230);
  --success: oklch(0.39 0.105 155);
  --success-soft: oklch(0.95 0.035 155);
  --warning: oklch(0.39 0.10 75);
  --warning-soft: oklch(0.95 0.055 80);
  --danger: oklch(0.43 0.14 28);
  --danger-soft: oklch(0.95 0.035 28);
  --focus: oklch(0.56 0.18 230);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 1rem;
  line-height: 1.55;
}
button, input, select { font: inherit; }
button { color: inherit; }
code, pre { font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; }
code { overflow-wrap: anywhere; }
h1, h2, h3 { line-height: 1.18; letter-spacing: -.025em; text-wrap: balance; }
h1 { max-width: 25ch; margin: 0; font-size: 2.45rem; }
h2 { margin: 0; font-size: 1.55rem; }
h3 { margin: 0 0 .55rem; font-size: 1.08rem; }
p { max-width: 72ch; text-wrap: pretty; }
a { color: var(--accent-dark); }
button:focus-visible, a:focus-visible, summary:focus-visible, [tabindex]:focus-visible {
  outline: 3px solid var(--focus);
  outline-offset: 3px;
}
.sr-only {
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  padding: 0 !important;
  margin: -1px !important;
  overflow: hidden !important;
  clip: rect(0, 0, 0, 0) !important;
  white-space: nowrap !important;
  border: 0 !important;
}
.skip-link {
  position: fixed;
  inset: 0 auto auto 0;
  transform: translateY(-130%);
  z-index: 30;
  padding: .7rem .9rem;
  background: var(--ink);
  color: var(--bg);
}
.skip-link:focus { transform: translateY(0); }
.evidence-banner {
  position: sticky;
  top: 0;
  z-index: 20;
  padding: .7rem max(.9rem, calc((100vw - 78rem) / 2));
  border-bottom: 1px solid var(--warning);
  background: var(--warning-soft);
  text-align: center;
}
.report-header, .report-main, .report-footer {
  width: min(calc(100% - 1.5rem), 78rem);
  margin-inline: auto;
}
.report-header { padding: 2.8rem 0 2.4rem; }
.brand-line {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: .7rem 1.5rem;
  margin-bottom: 2rem;
  padding-bottom: .85rem;
  border-bottom: 1px solid var(--line);
}
.brand { font-size: 1.05rem; font-weight: 760; letter-spacing: -.015em; }
.source-type { color: var(--muted); font-size: .92rem; }
.report-lede { margin: .85rem 0 0; color: var(--muted); font-size: 1.08rem; }
.verdict-layout {
  display: grid;
  margin-top: 2rem;
  border-block: 1px solid var(--line-strong);
}
.verdict-column { padding: 1.35rem 0; }
.verdict-column + .verdict-column { border-top: 1px solid var(--line); }
.canonical-column h2 { font-size: 1.55rem; }
.alternative-column h2 { font-size: 1.35rem; }
.context-label {
  margin: 0 0 .35rem;
  color: var(--muted);
  font-size: .84rem;
  font-weight: 720;
}
.verdict-copy { margin-bottom: 1rem; }
.column-note, .derived-note, .evidence-limit {
  color: var(--muted);
  font-size: .92rem;
}
.fact-list { margin: .85rem 0; padding-left: 1.2rem; }
.fact-list li + li { margin-top: .35rem; }
.decision-pair {
  display: flex;
  flex-wrap: wrap;
  gap: .8rem 1.8rem;
  margin: 0;
}
.decision-pair div { min-width: 9rem; }
.decision-pair dt, .hash-list dt {
  color: var(--muted);
  font-size: .8rem;
}
.decision-pair dd, .hash-list dd { margin: .15rem 0 0; font-weight: 680; }
.report-section {
  padding: 2.5rem 0;
  border-top: 1px solid var(--line);
}
.section-heading {
  display: grid;
  gap: .45rem;
  margin-bottom: 1.35rem;
}
.section-heading p { margin: 0; color: var(--muted); }
.why-layout { display: grid; gap: 1.3rem; }
.reason-block {
  padding: 1.15rem;
  border: 1px solid var(--line-strong);
  border-radius: 10px;
  background: var(--surface);
}
.reason-block p:last-child { margin-bottom: 0; }
.reason-number { margin: 0; font-size: 2rem; font-weight: 760; letter-spacing: -.03em; }
.reason-copy { padding-top: .75rem; border-top: 1px solid var(--line); }
.table-scroll {
  max-width: 100%;
  overflow-x: auto;
  overscroll-behavior-inline: contain;
}
table { width: 100%; border-collapse: collapse; font-size: .9rem; }
caption { padding: 0 0 .65rem; color: var(--muted); text-align: left; }
th, td {
  padding: .72rem;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}
thead th { background: var(--surface); color: var(--ink); font-weight: 720; }
tbody tr:last-child > * { border-bottom: 0; }
.compact-table { min-width: 40rem; }
.tradeoff-board {
  border-block: 1px solid var(--line);
}
.tradeoff-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: .45rem 1rem;
  padding: 1rem 0;
  border-bottom: 1px solid var(--line);
}
.tradeoff-row:last-child { border-bottom: 0; }
.tradeoff-metric { grid-column: 1 / -1; font-weight: 720; }
.tradeoff-value { font-variant-numeric: tabular-nums; }
.value-label { display: block; color: var(--muted); font-size: .78rem; }
.tradeoff-connector { display: none; }
.tradeoff-effect {
  grid-column: 1 / -1;
  width: fit-content;
  padding: .15rem .5rem;
  border-radius: 999px;
  font-size: .82rem;
  font-weight: 700;
}
.effect-better { background: var(--success-soft); color: var(--success); }
.effect-tradeoff { background: var(--warning-soft); color: var(--warning); }
.effect-held { background: var(--surface-strong); color: var(--muted); }
.profile-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: .45rem;
  padding-bottom: .85rem;
  border-bottom: 1px solid var(--line);
}
.profile-tabs button {
  padding: .58rem .72rem;
  border: 1px solid var(--line-strong);
  border-radius: 8px;
  background: var(--bg);
  cursor: pointer;
  text-align: left;
}
.profile-tabs button span { display: block; color: var(--muted); font-size: .72rem; }
.profile-tabs button[aria-selected="true"] {
  border-color: var(--accent);
  background: var(--accent-soft);
  color: var(--accent-dark);
}
.profile-panel { padding-top: 1.25rem; }
.profile-panel[hidden] { display: none; }
.profile-decision { min-width: 0; }
.profile-reason {
  margin-top: 1rem;
  padding-top: .8rem;
  border-top: 1px solid var(--line);
}
.profile-metrics {
  margin: 1.1rem 0 0;
  padding: 0;
  list-style: none;
  border-block: 1px solid var(--line);
}
.profile-metrics li {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  padding: .62rem 0;
  border-bottom: 1px solid var(--line);
}
.profile-metrics li:last-child { border-bottom: 0; }
.profile-metrics span { color: var(--muted); }
.profile-metrics strong { text-align: right; font-variant-numeric: tabular-nums; }
.load-context-grid, .chart-grid-layout { display: grid; gap: 1.2rem; }
.load-context { min-width: 0; }
.chart-grid-layout { margin: 1.2rem 0; }
.chart-figure { min-width: 0; margin: 0; }
.chart-figure svg {
  display: block;
  width: 100%;
  min-width: 44rem;
  height: auto;
}
.chart-figure figcaption { margin-top: .55rem; color: var(--muted); font-size: .86rem; }
.chart-axis { stroke: var(--line-strong); stroke-width: 1.4; }
.chart-grid { stroke: var(--line); stroke-width: 1; stroke-dasharray: 3 4; }
.chart-tick, .chart-axis-label, .direct-label {
  fill: var(--muted);
  font-family: ui-sans-serif, system-ui, sans-serif;
  font-size: 11px;
}
.chart-tick { text-anchor: end; }
.chart-tick-end { text-anchor: end; }
.chart-axis-label { text-anchor: middle; font-size: 12px; }
.direct-label { fill: var(--ink); font-size: 10.5px; }
.chart-line, .label-leader, .scatter-leader {
  fill: none;
  stroke-width: 2;
}
.label-leader, .scatter-leader { stroke-width: 1; }
.chart-marker { stroke: var(--bg); stroke-width: 2; }
.series-0 { stroke: oklch(0.42 0.15 230); fill: oklch(0.42 0.15 230); }
.series-1 { stroke: oklch(0.39 0.11 155); fill: oklch(0.39 0.11 155); stroke-dasharray: 6 3; }
.series-2 { stroke: oklch(0.44 0.13 40); fill: oklch(0.44 0.13 40); stroke-dasharray: 2 3; }
.series-3 { stroke: oklch(0.39 0.11 300); fill: oklch(0.39 0.11 300); stroke-dasharray: 9 3 2 3; }
.series-4 { stroke: oklch(0.39 0.09 90); fill: oklch(0.39 0.09 90); stroke-dasharray: 4 4; }
.series-5 { stroke: oklch(0.38 0.08 200); fill: oklch(0.38 0.08 200); stroke-dasharray: 10 4; }
.point-selected { stroke-width: 3; }
.point-baseline { stroke: var(--ink); }
.load-table, .candidate-table { min-width: 72rem; }
.no-data {
  padding: 1rem;
  border: 1px solid var(--line-strong);
  border-radius: 10px;
  background: var(--surface);
}
.evidence-limit {
  margin: 1rem 0 0;
  padding: .8rem 0;
  border-block: 1px solid var(--line);
}
.coverage-notice { margin-bottom: 1rem; }
.stability-method {
  display: flex;
  flex-wrap: wrap;
  gap: .2rem 1.5rem;
  margin-bottom: 1rem;
  color: var(--muted);
}
.stability-method p { margin: 0; }
.stability-table { min-width: 72rem; }
.pass-deltas { min-width: 15rem; margin: 0; padding: 0; list-style: none; }
.pass-deltas li { display: grid; gap: .08rem; }
.pass-deltas li + li {
  margin-top: .5rem;
  padding-top: .5rem;
  border-top: 1px solid var(--line);
}
.pass-deltas span { color: var(--muted); font-size: .82rem; }
.scatter-figure { margin-bottom: 1rem; overflow-x: auto; }
.candidate-label { display: block; min-width: 12rem; font-weight: 720; }
.candidate-table details { min-width: 10rem; }
summary {
  width: fit-content;
  cursor: pointer;
  color: var(--accent-dark);
  font-weight: 700;
}
details[open] summary { margin-bottom: .7rem; }
.json-block {
  display: block;
  width: min(28rem, 72vw);
  max-height: 20rem;
  padding: .7rem;
  overflow: auto;
  border-radius: 6px;
  background: var(--surface);
  white-space: pre-wrap;
}
.reason-list { min-width: 14rem; margin: 0; padding-left: 1.1rem; }
.quiet, .not-measured { color: var(--muted); }
.trust-grid { display: grid; gap: 1.4rem; }
.trust-block { min-width: 0; }
.hash-list { margin: 0; }
.hash-list > div { padding: .62rem 0; border-bottom: 1px solid var(--line); }
.hash-list dd { overflow-wrap: anywhere; }
.command {
  max-width: 100%;
  margin: 1rem 0 0;
  padding: 1rem;
  overflow-x: auto;
  border-radius: 8px;
  background: var(--ink);
  color: var(--bg);
  white-space: pre;
}
.metadata-details { display: block; margin-top: 1.5rem; }
.metadata-table { min-width: 42rem; }
.metadata-table code { white-space: pre-wrap; }
.reproduction-note {
  margin-top: 1.5rem;
  padding-top: 1.2rem;
  border-top: 1px solid var(--line);
}
.reproduction-note ol { max-width: 72ch; padding-left: 1.25rem; }
.report-footer {
  padding: 1.35rem 0 2.5rem;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: .9rem;
}
@media (min-width: 48rem) {
  .report-header { padding-top: 3.8rem; }
  .verdict-layout { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
  .verdict-column { padding: 1.5rem; }
  .verdict-column:first-child { padding-left: 0; }
  .verdict-column + .verdict-column {
    border-top: 0;
    border-left: 1px solid var(--line);
  }
  .why-layout { grid-template-columns: minmax(17rem, .72fr) minmax(25rem, 1.28fr); }
  .profile-panel { grid-template-columns: minmax(0, 1.2fr) minmax(15rem, .8fr); gap: 2rem; }
  .profile-panel:not([hidden]) { display: grid; }
  .profile-metrics { margin-top: 0; }
  .load-context-grid, .chart-grid-layout {
    grid-template-columns: repeat(auto-fit, minmax(25rem, 1fr));
  }
  .trust-grid { grid-template-columns: minmax(0, .85fr) minmax(0, 1.15fr); }
}
@media (min-width: 52rem) {
  .tradeoff-row {
    grid-template-columns: minmax(12rem, 1.1fr) minmax(8rem, .8fr) 4.5rem minmax(8rem, .8fr) minmax(10rem, 1fr);
    align-items: center;
    gap: .8rem;
  }
  .tradeoff-metric, .tradeoff-effect { grid-column: auto; }
  .tradeoff-connector { display: flex; align-items: center; }
  .connector-line { flex: 1; height: 1px; background: var(--line-strong); }
  .baseline-marker, .alternative-marker {
    width: 8px;
    height: 8px;
    border: 2px solid var(--accent-dark);
  }
  .baseline-marker { border-radius: 50%; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
}
@media print {
  .skip-link, .profile-tabs { display: none; }
  .evidence-banner { position: static; }
  .report-header, .report-main, .report-footer { width: 100%; }
  .table-scroll, .scatter-figure { overflow: visible; }
  .profile-panel[hidden] { display: block; }
  .candidate-table, .load-table, .compact-table, .metadata-table, .chart-figure svg {
    min-width: 0;
  }
}
"""


def render_report_v11(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
    *,
    policy_profiles: Mapping[str, Any] | None = None,
    load_sweep: Mapping[str, Any] | None = None,
    stability_summary: Mapping[str, Any] | None = None,
    benchmarks_sha256: str = "",
    recommendation_sha256: str = "",
    profiles_sha256: str = "",
    load_sha256: str = "",
    stability_sha256: str = "",
) -> str:
    """Render one opt-in, deterministic, self-contained v1.1 decision report.

    ``recommendation`` must be the canonical mapping produced by
    :func:`paretopilot.analysis.recommend`. Supplied policy profiles contain
    precomputed decisions; the renderer recomputes each selection only as a
    fail-closed integrity check. Load-sweep rows and pass-level stability
    summaries are validated and visualized without interpolation or significance
    claims.
    """

    if not isinstance(benchmarks, BenchmarkSet):
        raise TypeError("benchmarks must be a BenchmarkSet")
    recommendation = _mapping(recommendation, "recommendation")
    benchmarks_sha256 = _hash_value(benchmarks_sha256, "benchmarks_sha256")
    recommendation_sha256 = _hash_value(
        recommendation_sha256,
        "recommendation_sha256",
    )
    profiles_sha256 = _hash_value(profiles_sha256, "profiles_sha256")
    load_sha256 = _hash_value(load_sha256, "load_sha256")
    stability_sha256 = _hash_value(stability_sha256, "stability_sha256")
    _validate_recommendation_fingerprints(
        benchmarks,
        recommendation,
        benchmarks_sha256=benchmarks_sha256,
    )
    selected, selection, _ = _validated_recommendation(benchmarks, recommendation)
    profiles = _normalise_profiles(
        benchmarks,
        recommendation,
        policy_profiles,
        benchmarks_sha256=benchmarks_sha256,
    )
    load = _normalise_load_sweep(benchmarks, load_sweep)
    stability = _normalise_stability_summary(benchmarks, stability_summary)
    alternative = _resource_alternative(benchmarks, selected)
    policy_section, interaction_script = _policy_section(profiles, benchmarks)

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="color-scheme" content="light">\n'
        '<link rel="icon" href="data:,">\n'
        "<title>ParetoPilot v1.1 deployment decision report</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        '<a class="skip-link" href="#main-content">Skip to report</a>\n'
        f"{_evidence_banner(benchmarks)}\n"
        '<header class="report-header">\n'
        '<div class="brand-line"><span class="brand">ParetoPilot</span>'
        f'<span class="source-type">{_escape(_source_type(benchmarks))} · v1.1 view</span></div>\n'
        "<h1>Arm64 deployment decision evidence</h1>\n"
        '<p class="report-lede">One measured study can support different deployment '
        "priorities without pretending there is one universal winner.</p>\n"
        f"{_verdict_section(benchmarks, recommendation, selected, selection, alternative)}\n"
        "</header>\n"
        '<main id="main-content" class="report-main">\n'
        f"{_why_section(benchmarks, recommendation, selection)}\n"
        f"{_tradeoffs_section(benchmarks, recommendation, alternative)}\n"
        f"{policy_section}\n"
        f"{_load_section(load, benchmarks)}\n"
        f"{_repeat_section(benchmarks, stability)}\n"
        f"{_scatter_section(benchmarks, recommendation)}\n"
        f"{_evidence_section(benchmarks, recommendation)}\n"
        f"{_trust_section(benchmarks, recommendation, selected, benchmarks_sha256=benchmarks_sha256, recommendation_sha256=recommendation_sha256, profiles_sha256=profiles_sha256, load_sha256=load_sha256, stability_sha256=stability_sha256)}\n"
        "</main>\n"
        '<footer class="report-footer">ParetoPilot keeps the canonical recommendation, '
        "derived policy scenarios, measurements, and evidence limits visibly separate.</footer>\n"
        f"{interaction_script}"
        "</body>\n"
        "</html>\n"
    )
