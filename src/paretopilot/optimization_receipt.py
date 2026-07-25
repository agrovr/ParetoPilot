"""Deterministic Markdown receipts for validated Decision Passports.

The renderer is presentation-only: it performs no I/O, adds no clock value,
and never invents an unavailable measurement. Required structural errors fail
closed; optional evidence is shown as ``Not measured``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any

from paretopilot.domain import ValidationError


__all__ = ["render_optimization_receipt"]


_STAGE_LABELS = {
    "reference": "Reference",
    "quantization": "Quantization",
    "arm-kernel": "KleidiAI build",
    "runtime-tuning": "Runtime tuning",
}
_GRADES = frozenset({"arm64-attributed", "measured-unattributed", "synthetic"})
_EFFECTS = frozenset({"changed", "held", "improved", "tradeoff"})
_METRIC_LABELS = {
    "e2e_latency_ms_p95": "End-to-end latency p95 (ms)",
    "generation_tokens_per_second": "Generation throughput (tokens/s)",
    "generation_tps": "Generation throughput (tokens/s)",
    "model_size_mib": "Model size (MiB)",
    "peak_rss_mib": "Peak RSS (MiB)",
    "prompt_tokens_per_second": "Prompt throughput (tokens/s)",
    "prompt_tps": "Prompt throughput (tokens/s)",
    "quality_score": "Quality score",
    "requests_per_second": "Request throughput (requests/s)",
    "ttft_ms_p50": "Time to first token p50 (ms)",
    "ttft_ms_p95": "Time to first token p95 (ms)",
}
_MD_CONTROL = re.compile(r"([\\`*_[\]{}()#+\-.!|~])")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_NOT_MEASURED = "Not measured"


def _error(path: str, message: str) -> ValidationError:
    return ValidationError(f"decision passport {path} {message}")


def _required(raw: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in raw:
        raise _error(f"{path}.{key}", "is required")
    return raw[key]


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "must be an object")
    return value


def _sequence(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise _error(path, "must be an array")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(path, "must be a non-empty string")
    return value.strip()


def _optional_text(value: object, path: str) -> str | None:
    return None if value is None else _text(value, path)


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise _error(path, "must be a boolean")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _error(path, "must be a positive integer")
    return value


def _optional_integer(value: object, path: str) -> int | None:
    return None if value is None else _integer(value, path)


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, "must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise _error(path, "must be finite")
    return result


def _optional_number(value: object, path: str) -> float | None:
    return None if value is None else _number(value, path)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def _text_list(value: object, path: str) -> list[str]:
    return [_text(item, f"{path}[{index}]") for index, item in enumerate(_sequence(value, path))]


def _escape(value: str) -> str:
    """Escape source-controlled text for inline and table-cell Markdown."""

    normalized = " ".join(value.replace("\r", " ").replace("\n", " ").split())
    normalized = normalized.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _MD_CONTROL.sub(r"\\\1", normalized)


def _display_text(value: str | None) -> str:
    return _NOT_MEASURED if value is None else _escape(value)


def _format_number(
    value: float | None,
    *,
    decimals: int = 4,
    signed: bool = False,
) -> str:
    if value is None:
        return _NOT_MEASURED
    value = 0.0 if value == 0 else value
    if value and abs(value) < 10 ** (-decimals):
        rendered = format(value, ".4g")
    else:
        rendered = f"{value:,.{decimals}f}".rstrip("0").rstrip(".")
    return f"+{rendered}" if signed and value > 0 else rendered


def _format_percent(value: float | None, *, signed: bool = False) -> str:
    rendered = _format_number(value, decimals=2, signed=signed)
    return rendered if value is None else f"{rendered}%"


def _candidate(label: str, candidate_id: str) -> str:
    return f"{_escape(label)} ({_escape(candidate_id)})"


def _direction(value: str | None) -> str:
    return {"min": "Minimize", "max": "Maximize", None: "Not declared"}[value]


def _metric_label(metric: str) -> str:
    return _METRIC_LABELS.get(metric, _escape(metric))


def _table(rows: Sequence[tuple[str, str]]) -> list[str]:
    return [
        "| Field | Value |",
        "| --- | --- |",
        *(f"| {key} | {value} |" for key, value in rows),
    ]


def _validate_change(value: object, path: str) -> dict[str, Any]:
    raw = _mapping(value, path)
    metric = _text(_required(raw, "metric", path), f"{path}.metric")
    direction = raw.get("direction")
    if direction not in {None, "min", "max"}:
        raise _error(f"{path}.direction", "must be 'min', 'max', or null")
    previous = _optional_number(raw.get("previous"), f"{path}.previous")
    current = _optional_number(raw.get("current"), f"{path}.current")
    absolute = _optional_number(raw.get("absolute"), f"{path}.absolute")
    percent = _optional_number(raw.get("percent"), f"{path}.percent")
    effect = _text(_required(raw, "effect", path), f"{path}.effect")
    if effect not in _EFFECTS:
        raise _error(f"{path}.effect", "has an unsupported value")

    measurements = (previous, current, absolute)
    if any(item is None for item in measurements) and any(
        item is not None for item in measurements
    ):
        raise _error(path, "must provide previous, current, and absolute together")
    if previous is not None and current is not None and absolute is not None:
        if not _close(absolute, current - previous):
            raise _error(f"{path}.absolute", "does not match current minus previous")
        expected_percent = None if previous == 0 else absolute / previous * 100.0
        if expected_percent is None and percent is not None:
            raise _error(f"{path}.percent", "must be null when previous is zero")
        if (
            expected_percent is not None
            and percent is not None
            and not _close(percent, expected_percent)
        ):
            raise _error(f"{path}.percent", "does not match the adjacent values")

    return {
        "metric": metric,
        "direction": direction,
        "previous": previous,
        "current": current,
        "absolute": absolute,
        "percent": percent,
        "effect": effect,
    }


def _validate_changes(value: object, path: str) -> list[dict[str, Any]]:
    rows = [
        _validate_change(item, f"{path}[{index}]")
        for index, item in enumerate(_sequence(value, path))
    ]
    metrics = [str(row["metric"]) for row in rows]
    if len(metrics) != len(set(metrics)):
        raise _error(path, "must not repeat a metric")
    return sorted(rows, key=lambda row: str(row["metric"]))


def _validate_objective(value: object) -> dict[str, Any]:
    path = "objective"
    raw = _mapping(value, path)
    direction = _text(_required(raw, "direction", path), f"{path}.direction")
    if direction not in {"min", "max"}:
        raise _error(f"{path}.direction", "must be 'min' or 'max'")
    boundary_rule = _text(_required(raw, "boundary_rule", path), f"{path}.boundary_rule")
    expected_rule = "at-or-below" if direction == "min" else "at-or-above"
    if boundary_rule != expected_rule:
        raise _error(f"{path}.boundary_rule", f"must be {expected_rule!r}")

    best = _number(_required(raw, "numeric_best_value", path), f"{path}.numeric_best_value")
    tolerance = _number(_required(raw, "tolerance_percent", path), f"{path}.tolerance_percent")
    margin = _number(_required(raw, "tolerance_margin", path), f"{path}.tolerance_margin")
    boundary = _number(_required(raw, "shortlist_boundary", path), f"{path}.shortlist_boundary")
    selected_value = _number(_required(raw, "selected_value", path), f"{path}.selected_value")
    if not 0 <= tolerance <= 100:
        raise _error(f"{path}.tolerance_percent", "must be between 0 and 100")
    if not _close(margin, abs(best) * tolerance / 100.0):
        raise _error(f"{path}.tolerance_margin", "does not match the tolerance")
    expected_boundary = best + margin if direction == "min" else best - margin
    if not _close(boundary, expected_boundary):
        raise _error(f"{path}.shortlist_boundary", "does not match the tolerance")

    runway_path = f"{path}.selected_runway"
    runway_raw = _mapping(_required(raw, "selected_runway", path), runway_path)
    runway = _number(_required(runway_raw, "absolute", runway_path), f"{runway_path}.absolute")
    runway_percent = _optional_number(
        _required(runway_raw, "percent_of_boundary", runway_path),
        f"{runway_path}.percent_of_boundary",
    )
    expected_runway = boundary - selected_value if direction == "min" else selected_value - boundary
    if expected_runway < 0 and _close(expected_runway, 0):
        expected_runway = 0.0
    if runway < 0 or not _close(runway, expected_runway):
        raise _error(f"{runway_path}.absolute", "does not match the boundary")

    shortlist = _text_list(_required(raw, "shortlist_ids", path), f"{path}.shortlist_ids")
    if not shortlist or len(shortlist) != len(set(shortlist)):
        raise _error(f"{path}.shortlist_ids", "must contain unique candidate ids")
    return {
        "metric": _text(_required(raw, "metric", path), f"{path}.metric"),
        "direction": direction,
        "numeric_best_id": _text(
            _required(raw, "numeric_best_id", path), f"{path}.numeric_best_id"
        ),
        "numeric_best_value": best,
        "tolerance_percent": tolerance,
        "tolerance_margin": margin,
        "shortlist_boundary": boundary,
        "boundary_rule": boundary_rule,
        "selected_value": selected_value,
        "selected_runway": {"absolute": runway, "percent_of_boundary": runway_percent},
        "shortlist_ids": shortlist,
    }


def _validate_stages(value: object) -> list[dict[str, Any]]:
    raw_stages = _sequence(value, "ladder")
    if not raw_stages:
        raise _error("ladder", "must contain at least one stage")
    stages: list[dict[str, Any]] = []
    for offset, value in enumerate(raw_stages):
        path = f"ladder[{offset}]"
        raw = _mapping(value, path)
        stage_number = _integer(_required(raw, "stage", path), f"{path}.stage")
        if stage_number != offset + 1:
            raise _error(f"{path}.stage", f"must be {offset + 1}")
        attribution = _optional_text(
            _required(raw, "attribution_stage", path), f"{path}.attribution_stage"
        )
        recognized = _boolean(
            _required(raw, "recognized_attribution_stage", path),
            f"{path}.recognized_attribution_stage",
        )
        if recognized != (attribution in _STAGE_LABELS):
            raise _error(
                f"{path}.recognized_attribution_stage",
                "does not match the declared attribution stage",
            )

        delta_value = _required(raw, "delta_from_previous", path)
        delta: dict[str, Any] | None
        if offset == 0:
            if delta_value is not None:
                raise _error(f"{path}.delta_from_previous", "must be null")
            delta = None
        else:
            delta_path = f"{path}.delta_from_previous"
            delta_raw = _mapping(delta_value, delta_path)
            previous_id = _text(
                _required(delta_raw, "previous_candidate_id", delta_path),
                f"{delta_path}.previous_candidate_id",
            )
            if previous_id != stages[-1]["candidate_id"]:
                raise _error(
                    f"{delta_path}.previous_candidate_id",
                    "must identify the immediately preceding stage",
                )
            delta = {
                "previous_candidate_id": previous_id,
                "metrics": _validate_changes(
                    _required(delta_raw, "metrics", delta_path), f"{delta_path}.metrics"
                ),
                "not_comparable_metrics": sorted(
                    _text_list(
                        _required(delta_raw, "not_comparable_metrics", delta_path),
                        f"{delta_path}.not_comparable_metrics",
                    )
                ),
            }

        stages.append(
            {
                "stage": stage_number,
                "attribution_stage": attribution,
                "stage_label": _STAGE_LABELS.get(attribution),
                "candidate_id": _text(_required(raw, "candidate_id", path), f"{path}.candidate_id"),
                "label": _text(_required(raw, "label", path), f"{path}.label"),
                "objective_value": _optional_number(
                    _required(raw, "objective_value", path), f"{path}.objective_value"
                ),
                **{
                    key: _boolean(_required(raw, key, path), f"{path}.{key}")
                    for key in ("eligible", "frontier", "shortlisted", "selected", "baseline")
                },
                "constraint_violations": _text_list(
                    _required(raw, "constraint_violations", path),
                    f"{path}.constraint_violations",
                ),
                "delta_from_previous": delta,
            }
        )
    ids = [str(stage["candidate_id"]) for stage in stages]
    if len(ids) != len(set(ids)):
        raise _error("ladder", "must contain unique candidate ids")
    return stages


def _validate_provenance(value: object, grade: str) -> dict[str, Any]:
    path = "provenance"
    raw = _mapping(value, path)
    complete = _boolean(
        _required(raw, "attribution_complete", path), f"{path}.attribution_complete"
    )
    if complete != (grade == "arm64-attributed"):
        raise _error(f"{path}.attribution_complete", f"does not match {grade} evidence")

    result: dict[str, Any] = {
        "classification": _optional_text(
            _required(raw, "classification", path), f"{path}.classification"
        ),
        "attribution_complete": complete,
        "verification_scope": _text(
            _required(raw, "verification_scope", path), f"{path}.verification_scope"
        ),
        "issues": _text_list(_required(raw, "issues", path), f"{path}.issues"),
    }
    field_groups = {
        "runner": ("architecture", "reported_architecture", "cpu", "os"),
        "source": ("repository", "revision", "workflow"),
        "runtime": ("name", "repository", "revision"),
        "model": ("name", "repository", "revision"),
        "evaluation_suite": ("id", "sha256"),
    }
    for group, fields in field_groups.items():
        group_path = f"{path}.{group}"
        group_raw = _mapping(_required(raw, group, path), group_path)
        result[group] = {
            field: _optional_text(_required(group_raw, field, group_path), f"{group_path}.{field}")
            for field in fields
        }
    run_path = f"{path}.run"
    run_raw = _mapping(_required(raw, "run", path), run_path)
    result["run"] = {
        "id": _optional_text(_required(run_raw, "id", run_path), f"{run_path}.id"),
        "attempt": _optional_integer(
            _required(run_raw, "attempt", run_path), f"{run_path}.attempt"
        ),
    }
    runner_raw = _mapping(_required(raw, "runner", path), f"{path}.runner")
    result["runner"]["cpu_count"] = _optional_integer(
        _required(runner_raw, "cpu_count", f"{path}.runner"),
        f"{path}.runner.cpu_count",
    )
    digest = result["evaluation_suite"]["sha256"]
    if digest is not None and _SHA256.fullmatch(digest) is None:
        raise _error(f"{path}.evaluation_suite.sha256", "must be a SHA-256 digest or null")
    if grade == "arm64-attributed":
        if result["runner"]["architecture"] != "arm64":
            raise _error(f"{path}.runner.architecture", "must be arm64")
        if result["issues"]:
            raise _error(f"{path}.issues", "must be empty for attributed evidence")
    return result


def _validate_optional_candidate(
    value: object,
    path: str,
    stages: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if value is None:
        return None
    raw = _mapping(value, path)
    candidate_id = _text(_required(raw, "candidate_id", path), f"{path}.candidate_id")
    stage = next((item for item in stages if item["candidate_id"] == candidate_id), None)
    if stage is None:
        raise _error(f"{path}.candidate_id", "must identify a ladder candidate")
    number = _integer(_required(raw, "stage", path), f"{path}.stage")
    label = _text(_required(raw, "label", path), f"{path}.label")
    if number != stage["stage"] or label != stage["label"]:
        raise _error(path, "does not match its ladder stage")
    return {"raw": raw, "candidate_id": candidate_id, "stage": number, "label": label}


def _validate_passport(passport: Mapping[str, Any]) -> dict[str, Any]:
    schema = _text(_required(passport, "schema_version", "passport"), "schema_version")
    if schema != "1.0":
        raise _error("schema_version", "must be '1.0'")
    grade = _text(_required(passport, "evidence_grade", "passport"), "evidence_grade")
    if grade not in _GRADES:
        raise _error("evidence_grade", "has an unsupported value")

    selected_path = "selected_decision"
    selected_raw = _mapping(_required(passport, "selected_decision", "passport"), selected_path)
    selected = {
        key: _text(_required(selected_raw, key, selected_path), f"{selected_path}.{key}")
        for key in ("candidate_id", "label", "baseline_id", "reason")
    } | {
        key: _boolean(_required(selected_raw, key, selected_path), f"{selected_path}.{key}")
        for key in ("baseline_retained", "numeric_best")
    }
    objective = _validate_objective(_required(passport, "objective", "passport"))
    stages = _validate_stages(_required(passport, "ladder", "passport"))
    selected_stages = [stage for stage in stages if stage["selected"]]
    baseline_stages = [stage for stage in stages if stage["baseline"]]
    if len(selected_stages) != 1 or selected_stages[0]["candidate_id"] != selected["candidate_id"]:
        raise _error("ladder", "must mark exactly the selected decision")
    if len(baseline_stages) != 1 or baseline_stages[0]["candidate_id"] != selected["baseline_id"]:
        raise _error("ladder", "must mark exactly the declared baseline")
    if selected_stages[0]["label"] != selected["label"]:
        raise _error("selected_decision.label", "does not match the selected ladder stage")
    if selected["baseline_retained"] != (selected["candidate_id"] == selected["baseline_id"]):
        raise _error("selected_decision.baseline_retained", "does not match the candidate ids")
    if selected["numeric_best"] != (selected["candidate_id"] == objective["numeric_best_id"]):
        raise _error("selected_decision.numeric_best", "does not match the numeric-best id")
    if objective["selected_value"] != selected_stages[0]["objective_value"]:
        raise _error("objective.selected_value", "does not match the selected ladder stage")
    stage_ids = {str(stage["candidate_id"]) for stage in stages}
    if objective["numeric_best_id"] not in stage_ids:
        raise _error("objective.numeric_best_id", "must identify a ladder candidate")
    if {selected["candidate_id"], objective["numeric_best_id"]} - set(objective["shortlist_ids"]):
        raise _error("objective.shortlist_ids", "must contain selected and numeric-best ids")

    closest = _validate_optional_candidate(
        _required(passport, "closest_outside_shortlist", "passport"),
        "closest_outside_shortlist",
        stages,
    )
    if closest is not None:
        raw = closest["raw"]
        closest["objective_value"] = _optional_number(
            _required(raw, "objective_value", "closest_outside_shortlist"),
            "closest_outside_shortlist.objective_value",
        )
        gap_path = "closest_outside_shortlist.shortfall_to_shortlist"
        gap = _mapping(
            _required(raw, "shortfall_to_shortlist", "closest_outside_shortlist"),
            gap_path,
        )
        closest["shortfall"] = {
            "absolute": _number(_required(gap, "absolute", gap_path), f"{gap_path}.absolute"),
            "percent": _optional_number(
                _required(gap, "percent_of_boundary", gap_path),
                f"{gap_path}.percent_of_boundary",
            ),
        }
        if closest["candidate_id"] in objective["shortlist_ids"]:
            raise _error("closest_outside_shortlist.candidate_id", "must be outside the shortlist")

    alternative = _validate_optional_candidate(
        _required(passport, "resource_alternative", "passport"),
        "resource_alternative",
        stages,
    )
    if grade == "synthetic" and alternative is not None:
        raise _error("resource_alternative", "must be null for synthetic fixture evidence")
    if alternative is not None:
        raw = alternative["raw"]
        for flag in ("eligible", "frontier", "shortlisted", "selected", "baseline"):
            alternative[flag] = _boolean(
                _required(raw, flag, "resource_alternative"), f"resource_alternative.{flag}"
            )
        if (
            not alternative["eligible"]
            or alternative["selected"]
            or alternative["baseline"]
            or alternative["candidate_id"] in {selected["candidate_id"], selected["baseline_id"]}
        ):
            raise _error("resource_alternative", "must be an eligible secondary candidate")
        if not _boolean(
            _required(raw, "is_secondary_not_recommendation", "resource_alternative"),
            "resource_alternative.is_secondary_not_recommendation",
        ):
            raise _error("resource_alternative.is_secondary_not_recommendation", "must be true")
        alternative["constraint_violations"] = _text_list(
            _required(raw, "constraint_violations", "resource_alternative"),
            "resource_alternative.constraint_violations",
        )
        alternative["changes"] = []
        seen: set[str] = set()
        for group, effects in (
            ("improvements", {"improved"}),
            ("tradeoffs", {"tradeoff"}),
            ("other_changes", {"changed", "held"}),
        ):
            rows = _validate_changes(
                _required(raw, group, "resource_alternative"),
                f"resource_alternative.{group}",
            )
            if any(str(row["effect"]) not in effects for row in rows):
                raise _error(f"resource_alternative.{group}", "contains the wrong effect")
            for row in rows:
                if row["metric"] in seen:
                    raise _error("resource_alternative", "must not repeat a delta metric")
                seen.add(str(row["metric"]))
            alternative["changes"].extend(rows)
        alternative["changes"].sort(key=lambda row: str(row["metric"]))

    method_path = "method"
    method_raw = _mapping(_required(passport, "method", "passport"), method_path)
    method = {
        key: _text(_required(method_raw, key, method_path), f"{method_path}.{key}")
        for key in (
            "selection",
            "ladder_order",
            "resource_alternative",
            "current_boundary_caveat",
        )
    }
    if _boolean(
        _required(method_raw, "canonical_outputs_modified", method_path),
        f"{method_path}.canonical_outputs_modified",
    ):
        raise _error(f"{method_path}.canonical_outputs_modified", "must be false")

    input_fingerprints: dict[str, str] = {}
    if "input_fingerprints" in passport:
        fingerprint_path = "input_fingerprints"
        fingerprint_raw = _mapping(passport["input_fingerprints"], fingerprint_path)
        if not fingerprint_raw:
            raise _error(fingerprint_path, "must not be empty when supplied")
        for raw_name, raw_digest in fingerprint_raw.items():
            name = _text(raw_name, f"{fingerprint_path} key")
            digest = _text(raw_digest, f"{fingerprint_path}.{name}")
            if _SHA256.fullmatch(digest) is None:
                raise _error(
                    f"{fingerprint_path}.{name}",
                    "must be a SHA-256 digest",
                )
            input_fingerprints[name] = digest.lower()

    return {
        "schema_version": schema,
        "paretopilot_version": _text(
            _required(passport, "paretopilot_version", "passport"), "paretopilot_version"
        ),
        "evidence_grade": grade,
        "provenance": _validate_provenance(_required(passport, "provenance", "passport"), grade),
        "selected_decision": selected,
        "objective": objective,
        "ladder": stages,
        "closest_outside_shortlist": closest,
        "resource_alternative": alternative,
        "method": method,
        "input_fingerprints": dict(sorted(input_fingerprints.items())),
    }


def _status(stage: Mapping[str, Any]) -> str:
    labels = [
        label
        for key, label in (
            ("baseline", "Baseline"),
            ("selected", "Selected"),
            ("eligible", "Eligible"),
            ("frontier", "Pareto frontier"),
            ("shortlisted", "Objective shortlist"),
        )
        if stage[key]
    ]
    if not stage["eligible"]:
        labels.append("Rejected")
    return ", ".join(labels)


def _delta_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    result = [
        "| Metric | Direction | Previous | Current | Delta | Delta percent | Effect |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    if not rows:
        result.append(
            f"| {_NOT_MEASURED} | Not declared | {_NOT_MEASURED} | {_NOT_MEASURED} | "
            f"{_NOT_MEASURED} | {_NOT_MEASURED} | No comparable metric |"
        )
    for row in rows:
        result.append(
            "| "
            + " | ".join(
                (
                    _metric_label(str(row["metric"])),
                    _direction(row["direction"]),
                    _format_number(row["previous"]),
                    _format_number(row["current"]),
                    _format_number(row["absolute"], signed=True),
                    _format_percent(row["percent"], signed=True),
                    str(row["effect"]).capitalize(),
                )
            )
            + " |"
        )
    return result


def _evidence_statement(grade: str) -> str:
    if grade == "synthetic":
        return (
            "**Synthetic fixture only.** Values in this receipt are fixture values, "
            "not measured Arm64 or deployment evidence."
        )
    if grade == "arm64-attributed":
        return (
            "**Arm64-attributed source evidence.** Attribution means the required "
            "source metadata is complete; the scope below remains authoritative."
        )
    return (
        "**Measured but unattributed source evidence.** Arm64 attribution is incomplete, "
        "so this receipt does not claim a verified Arm64 result."
    )


def render_optimization_receipt(passport: Mapping[str, Any]) -> str:
    """Render a strict, deterministic Markdown optimization receipt."""

    if not isinstance(passport, Mapping):
        raise TypeError("passport must be a validated Decision Passport mapping")
    data = _validate_passport(passport)
    selected = data["selected_decision"]
    objective = data["objective"]
    stages = data["ladder"]
    baseline = next(stage for stage in stages if stage["baseline"])

    lines = [
        "# ParetoPilot Optimization Receipt",
        "",
        _evidence_statement(data["evidence_grade"]),
        "",
        (
            "Displayed fixture values are rounded for readability; exact values remain in "
            "the Decision Passport and other machine-readable artifacts."
            if data["evidence_grade"] == "synthetic"
            else (
                "Displayed measurements are rounded for readability; exact values remain in "
                "the Decision Passport and other machine-readable artifacts."
            )
        ),
        "",
        "## Decision",
        "",
        *_table(
            (
                ("Baseline", _candidate(baseline["label"], baseline["candidate_id"])),
                ("Selected", _candidate(selected["label"], selected["candidate_id"])),
                ("Baseline retained", "Yes" if selected["baseline_retained"] else "No"),
                ("Numeric objective winner", "Yes" if selected["numeric_best"] else "No"),
                ("Decision reason", _escape(selected["reason"])),
            )
        ),
        "",
        "## Objective boundary",
        "",
    ]
    numeric_best = next(
        stage for stage in stages if stage["candidate_id"] == objective["numeric_best_id"]
    )
    closest = data["closest_outside_shortlist"]
    closest_text = _NOT_MEASURED
    if closest is not None:
        closest_text = (
            f"{_candidate(closest['label'], closest['candidate_id'])}; "
            f"value {_format_number(closest['objective_value'])}; shortfall "
            f"{_format_number(closest['shortfall']['absolute'])} "
            f"({_format_percent(closest['shortfall']['percent'])})"
        )
    lines.extend(
        _table(
            (
                ("Objective", _metric_label(objective["metric"])),
                ("Direction", _direction(objective["direction"])),
                (
                    "Numeric best",
                    f"{_candidate(numeric_best['label'], numeric_best['candidate_id'])} "
                    f"at {_format_number(objective['numeric_best_value'])}",
                ),
                ("Predeclared tolerance", _format_percent(objective["tolerance_percent"])),
                ("Tolerance margin", _format_number(objective["tolerance_margin"])),
                (
                    "Shortlist boundary",
                    f"{_format_number(objective['shortlist_boundary'])} "
                    f"({_escape(objective['boundary_rule'])})",
                ),
                ("Selected value", _format_number(objective["selected_value"])),
                (
                    "Selected runway to boundary",
                    f"{_format_number(objective['selected_runway']['absolute'])} "
                    f"({_format_percent(objective['selected_runway']['percent_of_boundary'])} "
                    "of boundary)",
                ),
                ("Closest candidate outside shortlist", closest_text),
            )
        )
    )

    value_kind = "fixture-value" if data["evidence_grade"] == "synthetic" else "supplied benchmark"
    canonical_path = [stage["attribution_stage"] for stage in stages] == list(_STAGE_LABELS)
    lines.extend(
        (
            "",
            ("## Four-stage optimization path" if canonical_path else "## Optimization path"),
            "",
            f"This is the ordered {value_kind} path. Each delta compares only adjacent stages.",
        )
    )
    for stage in stages:
        violations = stage["constraint_violations"]
        lines.extend(
            (
                "",
                (
                    f"### Stage {stage['stage']} — {stage['stage_label']}"
                    if stage["stage_label"] is not None
                    else f"### Stage {stage['stage']}"
                ),
                "",
                *_table(
                    (
                        ("Candidate", _candidate(stage["label"], stage["candidate_id"])),
                        (
                            "Attribution stage",
                            _display_text(stage["attribution_stage"]),
                        ),
                        (
                            f"Objective ({_metric_label(objective['metric'])})",
                            _format_number(stage["objective_value"]),
                        ),
                        ("Status", _status(stage)),
                        (
                            "Constraint violations",
                            (
                                ", ".join(_escape(item) for item in violations)
                                if violations
                                else "None"
                            ),
                        ),
                    )
                ),
            )
        )
        delta = stage["delta_from_previous"]
        if delta is None:
            lines.extend(("", "Starting point; no adjacent-stage delta applies."))
            continue
        previous = next(
            item for item in stages if item["candidate_id"] == delta["previous_candidate_id"]
        )
        lines.extend(
            (
                "",
                f"Adjacent delta from {_candidate(previous['label'], previous['candidate_id'])}:",
                "",
                *_delta_table(delta["metrics"]),
                "",
                (
                    "**Not comparable between these stages:** "
                    + (
                        ", ".join(_escape(metric) for metric in delta["not_comparable_metrics"])
                        if delta["not_comparable_metrics"]
                        else "None"
                    )
                ),
            )
        )

    lines.extend(("", "## Resource alternative", ""))
    alternative = data["resource_alternative"]
    if data["evidence_grade"] == "synthetic":
        lines.append(
            "Not applicable. Synthetic fixture values are not deployment benchmark evidence, "
            "so this receipt does not claim a resource alternative."
        )
    elif alternative is None:
        lines.append(
            "No eligible resource alternative was identified in the supplied benchmark set."
        )
    else:
        lines.extend(
            (
                f"**Secondary comparison, not the recommendation:** "
                f"{_candidate(alternative['label'], alternative['candidate_id'])}",
                "",
                f"Baseline for every delta: "
                f"{_candidate(baseline['label'], baseline['candidate_id'])}.",
                "",
                *_delta_table(alternative["changes"]),
            )
        )

    provenance = data["provenance"]
    runner = provenance["runner"]
    run = provenance["run"]
    source = provenance["source"]
    runtime = provenance["runtime"]
    model = provenance["model"]
    suite = provenance["evaluation_suite"]
    provenance_rows = (
        ("Evidence grade", _escape(data["evidence_grade"])),
        (
            "Attribution metadata complete",
            "Yes" if provenance["attribution_complete"] else "No",
        ),
        ("Classification", _display_text(provenance["classification"])),
        ("Runner architecture", _display_text(runner["architecture"])),
        ("Runner reported architecture", _display_text(runner["reported_architecture"])),
        ("Runner CPU", _display_text(runner["cpu"])),
        (
            "Runner CPU count",
            _NOT_MEASURED if runner["cpu_count"] is None else str(runner["cpu_count"]),
        ),
        ("Runner OS", _display_text(runner["os"])),
        ("Run ID", _display_text(run["id"])),
        ("Run attempt", _NOT_MEASURED if run["attempt"] is None else str(run["attempt"])),
        ("Source repository", _display_text(source["repository"])),
        ("Source revision", _display_text(source["revision"])),
        ("Source workflow", _display_text(source["workflow"])),
        ("Runtime", _display_text(runtime["name"])),
        ("Runtime repository", _display_text(runtime["repository"])),
        ("Runtime revision", _display_text(runtime["revision"])),
        ("Model", _display_text(model["name"])),
        ("Model repository", _display_text(model["repository"])),
        ("Model revision", _display_text(model["revision"])),
        ("Evaluation suite", _display_text(suite["id"])),
        ("Evaluation suite SHA-256", _display_text(suite["sha256"])),
        ("Receipt schema", _escape(data["schema_version"])),
        ("ParetoPilot version", _escape(data["paretopilot_version"])),
        *(
            (
                f"Input fingerprint ({_escape(name)})",
                _escape(digest),
            )
            for name, digest in data["input_fingerprints"].items()
        ),
    )
    issues = provenance["issues"]
    lines.extend(
        (
            "",
            "## Provenance, fingerprints, and scope",
            "",
            *_table(provenance_rows),
            "",
            f"**Verification scope:** {_escape(provenance['verification_scope'])}",
            "",
            (
                "**Attribution issues:** "
                + ("; ".join(_escape(issue) for issue in issues) if issues else "None")
            ),
            "",
            f"**Boundary caveat:** {_escape(data['method']['current_boundary_caveat'])}",
            "",
            "**Canonical outputs modified:** No",
        )
    )
    return "\n".join(lines).rstrip() + "\n"
