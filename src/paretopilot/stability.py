"""Deterministic pass-level stability summaries without significance claims.

The input objects are already-validated :class:`~paretopilot.domain.BenchmarkSet`
instances.  This module describes observed direction consistency and relative
spread only; two or a few benchmark passes are never labeled statistically
significant.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Any, Mapping, Sequence

from paretopilot.domain import BenchmarkSet, ValidationError


_COMPARISONS = {"improved", "regressed", "no change"}
_CONSISTENCY_LABELS = {"consistent", "mixed", "no change"}


def summarize_stability(
    passes: Sequence[BenchmarkSet],
    *,
    metric_directions: Mapping[str, str],
    pass_labels: Sequence[str] | None = None,
    input_fingerprints: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    """Summarize observed candidate directions across two to eight passes."""

    if not isinstance(passes, Sequence) or isinstance(passes, (str, bytes)):
        raise ValidationError("stability passes must be an array")
    if not 2 <= len(passes) <= 8:
        raise ValidationError("stability summary requires from 2 to 8 passes")
    if any(not isinstance(benchmark, BenchmarkSet) for benchmark in passes):
        raise ValidationError("stability passes must contain validated BenchmarkSet objects")
    directions = _validate_directions(metric_directions)
    labels = _validate_pass_labels(pass_labels, len(passes))
    source_fingerprints = _validate_input_fingerprints(
        input_fingerprints or {},
        labels,
    )

    first = passes[0]
    baseline_id = first.baseline_id
    candidate_ids = sorted(candidate.candidate_id for candidate in first.candidates)
    first_ids = set(candidate_ids)
    configuration_fingerprints = {
        candidate_id: candidate_configuration_fingerprint(first.by_id(candidate_id))
        for candidate_id in candidate_ids
    }
    for index, benchmark in enumerate(passes, start=1):
        if benchmark.schema_version != first.schema_version:
            raise ValidationError(
                f"stability pass {index} schema version does not match the first pass"
            )
        if benchmark.baseline_id != baseline_id:
            raise ValidationError(
                f"stability pass {index} baseline_id does not match the first pass"
            )
        if benchmark.synthetic is not first.synthetic:
            raise ValidationError(
                f"stability pass {index} synthetic flag does not match the first pass"
            )
        ids = {candidate.candidate_id for candidate in benchmark.candidates}
        if ids != first_ids:
            raise ValidationError(
                f"stability pass {index} candidate ids do not match the first pass"
            )
        for candidate_id in candidate_ids:
            candidate = benchmark.by_id(candidate_id)
            if (
                candidate_configuration_fingerprint(candidate)
                != configuration_fingerprints[candidate_id]
            ):
                raise ValidationError(
                    f"stability pass {index} candidate {candidate_id!r} label or "
                    "parameters do not match the first pass"
                )
            missing = sorted(set(directions) - set(candidate.metrics))
            if missing:
                raise ValidationError(
                    f"stability pass {index} candidate {candidate_id!r} is missing "
                    f"metrics: {', '.join(missing)}"
                )

    rows: list[Mapping[str, Any]] = []
    for candidate_id in candidate_ids:
        for metric, direction in directions.items():
            pass_values: list[Mapping[str, Any]] = []
            candidate_values: list[float] = []
            for label, benchmark in zip(labels, passes, strict=True):
                candidate_value = float(benchmark.by_id(candidate_id).metrics[metric])
                baseline_value = float(benchmark.baseline.metrics[metric])
                comparison = _comparison(
                    candidate_value,
                    baseline_value,
                    direction,
                )
                candidate_values.append(candidate_value)
                pass_values.append(
                    {
                        "pass_label": label,
                        "candidate_value": candidate_value,
                        "baseline_value": baseline_value,
                        "relative_delta_percent": _relative_delta_percent(
                            candidate_value,
                            baseline_value,
                            direction,
                        ),
                        "comparison": comparison,
                    }
                )
            comparisons = {
                str(value["comparison"])
                for value in pass_values
                if value["comparison"] != "no change"
            }
            if not comparisons:
                consistency = "no change"
                overall_direction = "no change"
            elif len(comparisons) == 1:
                consistency = "consistent"
                overall_direction = next(iter(comparisons))
            else:
                consistency = "mixed"
                overall_direction = "mixed"
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "metric": metric,
                    "direction": direction,
                    "pass_values": pass_values,
                    "relative_spread_percent": _relative_spread_percent(candidate_values),
                    "consistency": consistency,
                    "overall_direction": overall_direction,
                }
            )

    summary: Mapping[str, Any] = {
        "schema_version": "1.0",
        "baseline_id": baseline_id,
        "synthetic": first.synthetic,
        "pass_labels": list(labels),
        "metric_directions": directions,
        "method": "observed pass direction and relative spread; no significance claim",
        "input_fingerprints": source_fingerprints,
        "candidate_configuration_fingerprints": configuration_fingerprints,
        "rows": rows,
    }
    validate_stability_summary(summary)
    return summary


def validate_stability_summary(
    raw: Mapping[str, Any],
    *,
    require_input_fingerprints: bool = False,
) -> None:
    """Strictly validate and recompute a serialized stability summary."""

    if not isinstance(require_input_fingerprints, bool):
        raise ValidationError("require_input_fingerprints must be a boolean")
    if not isinstance(raw, Mapping):
        raise ValidationError("stability summary must be an object")
    _require_exact_fields(
        raw,
        {
            "schema_version",
            "baseline_id",
            "synthetic",
            "pass_labels",
            "metric_directions",
            "method",
            "input_fingerprints",
            "candidate_configuration_fingerprints",
            "rows",
        },
        "stability summary",
    )
    if raw.get("schema_version") != "1.0":
        raise ValidationError("stability summary schema_version must be '1.0'")
    baseline_id = _nonempty_text(raw.get("baseline_id"), "stability baseline_id")
    if not isinstance(raw.get("synthetic"), bool):
        raise ValidationError("stability summary synthetic must be a boolean")
    labels_value = raw.get("pass_labels")
    if not isinstance(labels_value, list):
        raise ValidationError("stability summary pass_labels must be an array")
    labels = _validate_pass_labels(labels_value, len(labels_value))
    if not 2 <= len(labels) <= 8:
        raise ValidationError("stability summary requires from 2 to 8 pass labels")
    directions = _validate_directions(raw.get("metric_directions"))
    input_fingerprints = _validate_input_fingerprints(
        raw.get("input_fingerprints"),
        labels,
    )
    if require_input_fingerprints and not input_fingerprints:
        raise ValidationError("stability summary input_fingerprints are required")
    raw_configuration_fingerprints = _mapping(
        raw.get("candidate_configuration_fingerprints"),
        "stability summary candidate_configuration_fingerprints",
    )
    configuration_fingerprints = {
        _nonempty_text(candidate_id, "stability configuration candidate id"): _sha256_digest(
            digest,
            f"stability configuration fingerprint {candidate_id}",
        )
        for candidate_id, digest in raw_configuration_fingerprints.items()
    }
    if list(configuration_fingerprints) != sorted(configuration_fingerprints):
        raise ValidationError(
            "stability summary candidate_configuration_fingerprints must be in canonical order"
        )
    if raw.get("method") != "observed pass direction and relative spread; no significance claim":
        raise ValidationError("stability summary method is not recognized")
    rows = raw.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValidationError("stability summary rows must be a non-empty array")

    seen_pairs: set[tuple[str, str]] = set()
    candidate_ids: set[str] = set()
    pass_candidate_values: dict[tuple[str, str, str], float] = {}
    pass_baseline_values: dict[tuple[str, str, str], float] = {}
    for index, row_value in enumerate(rows):
        context = f"stability summary rows[{index}]"
        row = _mapping(row_value, context)
        _require_exact_fields(
            row,
            {
                "candidate_id",
                "metric",
                "direction",
                "pass_values",
                "relative_spread_percent",
                "consistency",
                "overall_direction",
            },
            context,
        )
        candidate_id = _nonempty_text(row.get("candidate_id"), f"{context}.candidate_id")
        candidate_ids.add(candidate_id)
        metric = _nonempty_text(row.get("metric"), f"{context}.metric")
        if metric not in directions:
            raise ValidationError(f"{context}.metric is not declared")
        direction = row.get("direction")
        if direction != directions[metric]:
            raise ValidationError(f"{context}.direction does not match metric_directions")
        pair = (candidate_id, metric)
        if pair in seen_pairs:
            raise ValidationError(f"stability summary contains duplicate row {pair!r}")
        seen_pairs.add(pair)

        pass_values_raw = row.get("pass_values")
        if not isinstance(pass_values_raw, list) or len(pass_values_raw) != len(labels):
            raise ValidationError(f"{context}.pass_values must contain exactly one value per pass")
        candidate_values: list[float] = []
        comparisons: list[str] = []
        for pass_index, (label, pass_value_raw) in enumerate(
            zip(labels, pass_values_raw, strict=True)
        ):
            pass_context = f"{context}.pass_values[{pass_index}]"
            pass_value = _mapping(pass_value_raw, pass_context)
            _require_exact_fields(
                pass_value,
                {
                    "pass_label",
                    "candidate_value",
                    "baseline_value",
                    "relative_delta_percent",
                    "comparison",
                },
                pass_context,
            )
            if pass_value.get("pass_label") != label:
                raise ValidationError(f"{pass_context}.pass_label is not in canonical order")
            candidate_value = _finite_number(
                pass_value.get("candidate_value"),
                f"{pass_context}.candidate_value",
            )
            baseline_value = _finite_number(
                pass_value.get("baseline_value"),
                f"{pass_context}.baseline_value",
            )
            expected_comparison = _comparison(candidate_value, baseline_value, direction)
            if pass_value.get("comparison") != expected_comparison:
                raise ValidationError(f"{pass_context}.comparison does not match its values")
            expected_delta = _relative_delta_percent(
                candidate_value,
                baseline_value,
                direction,
            )
            actual_delta = pass_value.get("relative_delta_percent")
            if expected_delta is None:
                if actual_delta is not None:
                    raise ValidationError(f"{pass_context}.relative_delta_percent must be null")
            elif not _numbers_match(actual_delta, expected_delta):
                raise ValidationError(
                    f"{pass_context}.relative_delta_percent does not match its values"
                )
            candidate_values.append(candidate_value)
            comparisons.append(expected_comparison)
            pass_candidate_values[(candidate_id, metric, label)] = candidate_value
            pass_baseline_values[(candidate_id, metric, label)] = baseline_value

        expected_spread = _relative_spread_percent(candidate_values)
        actual_spread = row.get("relative_spread_percent")
        if expected_spread is None:
            if actual_spread is not None:
                raise ValidationError(f"{context}.relative_spread_percent must be null")
        elif not _numbers_match(actual_spread, expected_spread):
            raise ValidationError(f"{context}.relative_spread_percent does not match pass values")
        changed = {value for value in comparisons if value != "no change"}
        if not changed:
            expected_consistency = "no change"
            expected_direction = "no change"
        elif len(changed) == 1:
            expected_consistency = "consistent"
            expected_direction = next(iter(changed))
        else:
            expected_consistency = "mixed"
            expected_direction = "mixed"
        if row.get("consistency") not in _CONSISTENCY_LABELS:
            raise ValidationError(f"{context}.consistency is not recognized")
        if row.get("consistency") != expected_consistency:
            raise ValidationError(f"{context}.consistency does not match pass directions")
        if row.get("overall_direction") != expected_direction:
            raise ValidationError(f"{context}.overall_direction does not match pass directions")

    expected_pairs = {
        (candidate_id, metric) for candidate_id in candidate_ids for metric in directions
    }
    if seen_pairs != expected_pairs:
        raise ValidationError("stability summary rows must contain every candidate and metric")
    canonical_pairs = sorted(seen_pairs)
    actual_pairs = [(str(row["candidate_id"]), str(row["metric"])) for row in rows]
    if actual_pairs != canonical_pairs:
        raise ValidationError("stability summary rows must be in canonical order")
    if baseline_id not in candidate_ids:
        raise ValidationError("stability summary baseline_id must identify one candidate")
    if set(configuration_fingerprints) != candidate_ids:
        raise ValidationError(
            "stability summary candidate_configuration_fingerprints must cover every candidate"
        )
    for metric in directions:
        for label in labels:
            baseline_candidate_value = pass_candidate_values[(baseline_id, metric, label)]
            baseline_row_value = pass_baseline_values[(baseline_id, metric, label)]
            if not math.isclose(
                baseline_candidate_value,
                baseline_row_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValidationError(
                    "stability summary baseline candidate value must equal its "
                    f"baseline_value for metric {metric!r} pass {label!r}"
                )
            for candidate_id in candidate_ids:
                candidate_baseline = pass_baseline_values[(candidate_id, metric, label)]
                if not math.isclose(
                    candidate_baseline,
                    baseline_candidate_value,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValidationError(
                        "stability summary baseline_value is inconsistent across "
                        f"candidates for metric {metric!r} pass {label!r}"
                    )


def candidate_configuration_fingerprint(candidate: Any) -> str:
    """Hash the label and deployment parameters that identify one candidate."""

    payload = json.dumps(
        {
            "label": candidate.label,
            "parameters": candidate.parameters,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_input_fingerprints(
    value: Any,
    labels: Sequence[str],
) -> Mapping[str, str]:
    fingerprints = _mapping(value, "stability input_fingerprints")
    if fingerprints and set(fingerprints) != set(labels):
        raise ValidationError(
            "stability input_fingerprints must contain exactly one digest per pass label"
        )
    if list(fingerprints) != [label for label in labels if label in fingerprints]:
        raise ValidationError("stability input_fingerprints must follow pass label order")
    return {
        label: _sha256_digest(
            fingerprints[label],
            f"stability input_fingerprints.{label}",
        )
        for label in labels
        if label in fingerprints
    }


def _sha256_digest(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValidationError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _comparison(candidate: float, baseline: float, direction: str) -> str:
    if math.isclose(candidate, baseline, rel_tol=1e-12, abs_tol=1e-12):
        return "no change"
    improved = candidate > baseline if direction == "max" else candidate < baseline
    return "improved" if improved else "regressed"


def _relative_delta_percent(
    candidate: float,
    baseline: float,
    direction: str,
) -> float | None:
    if math.isclose(candidate, baseline, rel_tol=1e-12, abs_tol=1e-12):
        return 0.0
    if math.isclose(baseline, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return None
    raw = ((candidate - baseline) / abs(baseline)) * 100.0
    return raw if direction == "max" else -raw


def _relative_spread_percent(values: Sequence[float]) -> float | None:
    mean = float(statistics.fmean(values))
    spread = max(values) - min(values)
    if math.isclose(mean, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return 0.0 if math.isclose(spread, 0.0, abs_tol=1e-12) else None
    return (spread / abs(mean)) * 100.0


def _validate_directions(value: Any) -> Mapping[str, str]:
    directions = _mapping(value, "metric_directions")
    if not 1 <= len(directions) <= 32:
        raise ValidationError("metric_directions must contain from 1 to 32 metrics")
    if any(not isinstance(metric, str) or not metric.strip() for metric in directions):
        raise ValidationError("metric_directions contains an invalid metric name")
    normalized: dict[str, str] = {}
    for metric in sorted(directions):
        name = _nonempty_text(metric, "metric_directions metric")
        direction = directions[metric]
        if direction not in {"min", "max"}:
            raise ValidationError(f"metric_directions.{name} must be 'min' or 'max'")
        normalized[name] = str(direction)
    return normalized


def _validate_pass_labels(
    value: Sequence[str] | None,
    pass_count: int,
) -> tuple[str, ...]:
    if value is None:
        return tuple(f"pass-{index}" for index in range(1, pass_count + 1))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValidationError("pass_labels must be an array")
    if len(value) != pass_count:
        raise ValidationError("pass_labels must contain exactly one label per pass")
    labels = tuple(
        _nonempty_text(label, f"pass_labels[{index}]") for index, label in enumerate(value)
    )
    if len(set(labels)) != len(labels):
        raise ValidationError("pass_labels must be unique")
    return labels


def _numbers_match(value: Any, expected: float) -> bool:
    try:
        actual = _finite_number(value, "numeric aggregate")
    except ValidationError:
        return False
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{context} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValidationError(f"{context} must be a finite number")
    return converted


def _nonempty_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{context} must be a non-empty string")
    if len(value) > 128:
        raise ValidationError(f"{context} must contain at most 128 characters")
    return value


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{context} must be an object")
    return value


def _require_exact_fields(
    raw: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(raw)
    missing = sorted(expected - actual)
    unknown = sorted(str(key) for key in actual - expected)
    details: list[str] = []
    if missing:
        details.append("missing fields: " + ", ".join(missing))
    if unknown:
        details.append("unknown fields: " + ", ".join(unknown))
    if details:
        raise ValidationError(f"{context} has " + "; ".join(details))
