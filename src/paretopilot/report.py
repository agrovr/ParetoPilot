"""Deterministic, dependency-free HTML decision reports.

The report is intentionally static: it can be reviewed offline, archived beside the
benchmark evidence, and hashed without timestamps or network-loaded assets changing it.
"""

from __future__ import annotations

import html
import json
import math
import shlex
from collections.abc import Mapping, Sequence
from typing import Any

from paretopilot.domain import BenchmarkSet, Candidate, Constraints


__all__ = ["render_report"]


_METRIC_LABELS = {
    "confidence_and_practical_effect_gate": "Confidence and practical-effect gate",
    "e2e_latency_ms_p95": "End-to-end latency (p95)",
    "generation_duration_ms": "Generation duration",
    "generation_tokens_per_second": "Generation throughput",
    "generation_tps": "Generation throughput",
    "model_identity_quality_retention": "Model-identity quality retention",
    "model_size_mib": "Model size",
    "peak_rss_mib": "Peak resident memory",
    "prompt_duration_ms": "Prompt processing duration",
    "prompt_tokens_per_second": "Prompt processing throughput",
    "prompt_tps": "Prompt processing throughput",
    "quality_score": "Quality score",
    "requests_per_second": "Requests per second",
    "ttft_ms_p50": "Time to first token (p50)",
    "ttft_ms_p95": "Time to first token (p95)",
}

_METRIC_PRIORITY = (
    "generation_tokens_per_second",
    "generation_tps",
    "prompt_tokens_per_second",
    "prompt_tps",
    "generation_duration_ms",
    "prompt_duration_ms",
    "e2e_latency_ms_p95",
    "ttft_ms_p95",
    "ttft_ms_p50",
    "model_identity_quality_retention",
    "quality_score",
    "confidence_and_practical_effect_gate",
    "peak_rss_mib",
    "model_size_mib",
    "requests_per_second",
)

_METRIC_UNITS = {
    "generation_tokens_per_second": " tok/s",
    "generation_tps": " tok/s",
    "prompt_tokens_per_second": " tok/s",
    "prompt_tps": " tok/s",
    "generation_duration_ms": " ms",
    "prompt_duration_ms": " ms",
    "ttft_ms_p95": " ms",
}

_REQUIRED_DEPLOYMENT_PARAMETERS = (
    "runtime_binary",
    "model_path",
    "threads",
    "batch_size",
)

_MAX_DEPLOYMENT_ARGV_ITEMS = 128
_MAX_DEPLOYMENT_ARGUMENT_LENGTH = 4096
_MAX_DEPLOYMENT_ARGV_LENGTH = 32768


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _normalise_json_value(value: Any) -> Any:
    """Convert JSON-like evidence values to a stable, serialisable representation."""

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


def _metric_label(metric: str) -> str:
    if metric in _METRIC_LABELS:
        return _METRIC_LABELS[metric]
    return metric.replace("_", " ").strip().capitalize()


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-9):
        return f"{value:,.0f}"
    if abs(value) >= 1_000_000 or (0 < abs(value) < 0.0001):
        return f"{value:.4g}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _metric_unit(metric: str) -> str:
    if metric in _METRIC_UNITS:
        return _METRIC_UNITS[metric]
    if metric.endswith("_ms") or "_ms_" in metric:
        return " ms"
    if metric.endswith("_mib") or metric.endswith("_mb"):
        return " MiB"
    if metric.endswith("_tps") or metric.endswith("_tokens_per_second"):
        return " tok/s"
    if metric == "requests_per_second":
        return " req/s"
    return ""


def _formatted_metric_text(metric: str, value: float) -> str:
    if metric == "model_identity_quality_retention":
        return f"{value * 100:.2f}%"
    if metric == "confidence_and_practical_effect_gate":
        if math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12):
            return "Pass (1)"
        if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12):
            return "Fail (0)"
    return f"{_format_number(value)}{_metric_unit(metric)}"


def _metric_value(metric: str, value: float | None) -> str:
    if value is None:
        return '<span class="metric-value not-measured">Not measured</span>'
    return f'<span class="metric-value">{_escape(_formatted_metric_text(metric, value))}</span>'


def _ordered_metrics(benchmarks: BenchmarkSet, constraints: Constraints) -> tuple[str, ...]:
    metrics = {
        constraints.quality_metric,
        constraints.objective.metric,
        *constraints.frontier_metrics,
        *constraints.max_values,
        *constraints.min_values,
    }
    for candidate in benchmarks.candidates:
        metrics.update(candidate.metrics)
    priority = [metric for metric in _METRIC_PRIORITY if metric in metrics]
    return tuple(priority + sorted(metrics - set(priority)))


def _mapping_of_rejections(recommendation: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    raw = recommendation.get("rejected", {})
    if not isinstance(raw, Mapping):
        return {}
    rejected: dict[str, tuple[str, ...]] = {}
    for candidate_id, reasons in raw.items():
        if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes, bytearray)):
            rejected[str(candidate_id)] = tuple(str(reason) for reason in reasons)
        else:
            rejected[str(candidate_id)] = (str(reasons),)
    return rejected


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return set()
    return {str(item) for item in value}


def _status_for_candidate(
    candidate: Candidate,
    *,
    selected_id: str,
    baseline_id: str,
    rejected: Mapping[str, tuple[str, ...]],
    eligible_ids: set[str],
    frontier_ids: set[str],
) -> tuple[str, str]:
    candidate_id = candidate.candidate_id
    labels: list[str] = []
    css_class = "status-neutral"
    if candidate_id == selected_id:
        labels.append("Selected")
        css_class = "status-selected"
    elif candidate_id in rejected:
        labels.append("Rejected")
        css_class = "status-rejected"
    elif candidate_id in frontier_ids:
        labels.append("Pareto frontier")
        css_class = "status-frontier"
    elif candidate_id in eligible_ids:
        labels.append("Eligible")
        css_class = "status-eligible"
    else:
        labels.append("Not classified")
    if candidate_id == baseline_id:
        labels.append("Baseline")
    return " · ".join(labels), css_class


def _candidate_table(
    benchmarks: BenchmarkSet,
    constraints: Constraints,
    recommendation: Mapping[str, Any],
    *,
    selected_id: str,
) -> str:
    metrics = _ordered_metrics(benchmarks, constraints)
    rejected = _mapping_of_rejections(recommendation)
    eligible_ids = _string_set(recommendation.get("eligible_ids"))
    frontier_ids = _string_set(recommendation.get("frontier_ids"))

    metric_headers = "".join(
        f'<th scope="col">{_escape(_metric_label(metric))}</th>' for metric in metrics
    )
    rows: list[str] = []
    for candidate in benchmarks.candidates:
        status, status_class = _status_for_candidate(
            candidate,
            selected_id=selected_id,
            baseline_id=benchmarks.baseline_id,
            rejected=rejected,
            eligible_ids=eligible_ids,
            frontier_ids=frontier_ids,
        )
        reason_items = rejected.get(candidate.candidate_id, ())
        if reason_items:
            reasons = (
                '<ul class="reason-list">'
                + "".join(f"<li>{_escape(reason)}</li>" for reason in reason_items)
                + "</ul>"
            )
        else:
            reasons = '<span class="quiet">No declared constraint violations.</span>'
        metric_cells = "".join(
            f'<td data-metric="{_escape(metric)}">'
            f"{_metric_value(metric, candidate.metrics.get(metric))}</td>"
            for metric in metrics
        )
        rows.append(
            '<tr class="candidate-row">'
            '<th scope="row">'
            f'<span class="candidate-name">{_escape(candidate.label)}</span>'
            f'<code class="candidate-id">{_escape(candidate.candidate_id)}</code>'
            "</th>"
            f'<td><span class="status {status_class}">{_escape(status)}</span></td>'
            '<td><details class="evidence-details parameter-details">'
            "<summary>View configuration</summary>"
            f'<code class="parameter-json">{_escape(_stable_json(candidate.parameters))}</code>'
            "</details></td>"
            f"{metric_cells}"
            f"<td>{reasons}</td>"
            "</tr>"
        )

    return (
        '<div class="table-scroll" tabindex="0" role="region" '
        'aria-label="Scrollable candidate comparison">'
        '<table class="candidate-table">'
        "<caption>Candidate measurements, eligibility, and exact rejection reasons</caption>"
        "<thead><tr>"
        '<th scope="col">Candidate</th>'
        '<th scope="col">Decision status</th>'
        '<th scope="col">Parameters</th>'
        f"{metric_headers}"
        '<th scope="col">Constraint result</th>'
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def _guardrail_rows(constraints: Constraints) -> str:
    rows = [
        (
            "Quality retention floor",
            constraints.quality_metric,
            f"At least {constraints.min_quality_retention * 100:.2f}% of baseline",
        ),
        (
            "Selection objective",
            constraints.objective.metric,
            "Maximize" if constraints.objective.direction == "max" else "Minimize",
        ),
        (
            "Objective tolerance",
            constraints.objective.metric,
            f"Within {constraints.objective_tolerance_percent:.2f}% of the numeric best",
        ),
    ]
    if constraints.preference_order:
        rows.append(
            (
                "Within-tolerance preference",
                constraints.objective.metric,
                " → ".join(constraints.preference_order),
            )
        )
    rows.extend(
        ("Maximum", metric, _formatted_metric_text(metric, value))
        for metric, value in sorted(constraints.max_values.items())
    )
    rows.extend(
        ("Minimum", metric, _formatted_metric_text(metric, value))
        for metric, value in sorted(constraints.min_values.items())
    )
    rows.extend(
        ("Pareto frontier", metric, direction)
        for metric, direction in sorted(constraints.frontier_metrics.items())
    )
    return "".join(
        "<tr>"
        f"<td>{_escape(kind)}</td>"
        f'<th scope="row">{_escape(_metric_label(metric))}</th>'
        f"<td>{_escape(value)}</td>"
        "</tr>"
        for kind, metric, value in rows
    )


def _impact_rows(
    benchmarks: BenchmarkSet,
    constraints: Constraints,
    selected: Candidate,
) -> str:
    baseline = benchmarks.baseline
    rows: list[str] = []
    for metric in _ordered_metrics(benchmarks, constraints):
        baseline_value = baseline.metrics.get(metric)
        selected_value = selected.metrics.get(metric)
        if baseline_value is None or selected_value is None:
            change = '<span class="not-measured">Not measured</span>'
            assessment = "Insufficient evidence"
            assessment_class = "status-neutral"
        else:
            absolute = selected_value - baseline_value
            if math.isclose(absolute, 0.0, rel_tol=1e-9, abs_tol=1e-12):
                change = "No change"
                assessment = "Held"
                assessment_class = "status-neutral"
            elif baseline_value == 0:
                change = f"{_format_number(absolute)} (percentage unavailable)"
                assessment = "Changed"
                assessment_class = "status-neutral"
            else:
                percent = absolute / baseline_value * 100.0
                change = f"{percent:+.2f}%"
                direction = constraints.frontier_metrics.get(metric)
                improved = (direction == "max" and absolute > 0) or (
                    direction == "min" and absolute < 0
                )
                if direction is None:
                    assessment = "Changed"
                    assessment_class = "status-neutral"
                elif improved:
                    assessment = "Improved"
                    assessment_class = "status-selected"
                else:
                    assessment = "Regressed"
                    assessment_class = "status-rejected"
        rows.append(
            "<tr>"
            f'<th scope="row">{_escape(_metric_label(metric))}</th>'
            f"<td>{_metric_value(metric, baseline_value)}</td>"
            f"<td>{_metric_value(metric, selected_value)}</td>"
            f"<td>{change}</td>"
            f'<td><span class="status {assessment_class}">{_escape(assessment)}</span></td>'
            "</tr>"
        )
    return "".join(rows)


def _plot_coordinates(values: list[float], start: float, end: float) -> list[float]:
    low = min(values)
    high = max(values)
    if math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-12):
        return [(start + end) / 2.0 for _ in values]
    return [start + (value - low) / (high - low) * (end - start) for value in values]


def _pareto_visualisation(benchmarks: BenchmarkSet, *, selected_id: str) -> str:
    required = {"generation_tps", "e2e_latency_ms_p95"}
    can_plot = len(benchmarks.candidates) >= 2 and all(
        required <= set(candidate.metrics) for candidate in benchmarks.candidates
    )
    if not can_plot:
        return (
            '<div class="plot-fallback" role="status">'
            "<strong>Scatter plot unavailable.</strong> Every candidate needs both "
            "generation throughput and p95 end-to-end latency. The candidate table remains "
            "the complete textual comparison; unavailable values are marked Not measured."
            "</div>"
        )

    candidates = list(benchmarks.candidates)
    latency = [item.metrics["e2e_latency_ms_p95"] for item in candidates]
    throughput = [item.metrics["generation_tps"] for item in candidates]
    x_positions = _plot_coordinates(latency, 82.0, 750.0)
    y_raw = _plot_coordinates(throughput, 310.0, 34.0)

    point_markup: list[str] = []
    for candidate, x_pos, y_pos in zip(candidates, x_positions, y_raw, strict=True):
        classes = ["chart-point"]
        if candidate.candidate_id == benchmarks.baseline_id:
            classes.append("chart-point-baseline")
        if candidate.candidate_id == selected_id:
            classes.append("chart-point-selected")
        detail = (
            f"{candidate.label}: {_format_number(candidate.metrics['generation_tps'])} tokens "
            f"per second, {_format_number(candidate.metrics['e2e_latency_ms_p95'])} milliseconds "
            "p95 latency"
        )
        label_x = min(x_pos + 10.0, 724.0)
        point_markup.append(
            f'<g class="{" ".join(classes)}">'
            f"<title>{_escape(detail)}</title>"
            f'<circle cx="{x_pos:.2f}" cy="{y_pos:.2f}" r="6"></circle>'
            f'<text x="{label_x:.2f}" y="{max(y_pos - 10.0, 18.0):.2f}">'
            f"{_escape(candidate.label)}</text>"
            "</g>"
        )

    return (
        '<figure class="pareto-figure">'
        '<div class="chart-scroll" tabindex="0" role="region" '
        'aria-label="Scrollable throughput and latency chart">'
        '<svg viewBox="0 0 800 360" role="img" '
        'aria-labelledby="pareto-chart-title pareto-chart-description">'
        '<title id="pareto-chart-title">Candidate throughput and latency</title>'
        '<desc id="pareto-chart-description">Higher on the chart is more generation throughput; '
        "farther left is lower p95 end-to-end latency. Selected and baseline status is shown "
        "visually; one candidate may hold both roles.</desc>"
        '<line class="chart-axis" x1="82" y1="310" x2="750" y2="310"></line>'
        '<line class="chart-axis" x1="82" y1="34" x2="82" y2="310"></line>'
        '<text class="chart-axis-label" x="416" y="348" text-anchor="middle">'
        "p95 end-to-end latency (ms) — lower is better</text>"
        '<text class="chart-axis-label" x="20" y="172" text-anchor="middle" '
        'transform="rotate(-90 20 172)">Generation throughput (tok/s) — higher is better</text>'
        f'<text class="chart-tick" x="82" y="329">{_escape(_format_number(min(latency)))}</text>'
        f'<text class="chart-tick" x="750" y="329" text-anchor="end">'
        f"{_escape(_format_number(max(latency)))}</text>"
        f'<text class="chart-tick" x="72" y="310" text-anchor="end">'
        f"{_escape(_format_number(min(throughput)))}</text>"
        f'<text class="chart-tick" x="72" y="40" text-anchor="end">'
        f"{_escape(_format_number(max(throughput)))}</text>"
        f"{''.join(point_markup)}"
        "</svg></div>"
        "<figcaption>Each point is one candidate. The table below is the authoritative "
        "measurement and eligibility record.</figcaption>"
        "</figure>"
    )


def _positive_int(value: Any, *, maximum: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value <= 0 or value > maximum:
        return None
    return value


def _safe_text_parameter(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return None
    return value


def _authoritative_deployment_argv(
    value: Any,
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None, ("deployment_argv must be a non-empty array of strings",)
    if not value:
        return None, ("deployment_argv must not be empty",)
    if len(value) > _MAX_DEPLOYMENT_ARGV_ITEMS:
        return None, (f"deployment_argv exceeds the {_MAX_DEPLOYMENT_ARGV_ITEMS}-item limit",)

    argv: list[str] = []
    total_length = 0
    for index, argument in enumerate(value):
        if not isinstance(argument, str):
            return None, (f"deployment_argv[{index}] must be a string",)
        if not argument:
            return None, (f"deployment_argv[{index}] must not be empty",)
        if len(argument) > _MAX_DEPLOYMENT_ARGUMENT_LENGTH:
            return None, (
                f"deployment_argv[{index}] exceeds the "
                f"{_MAX_DEPLOYMENT_ARGUMENT_LENGTH}-character limit",
            )
        if any(ord(character) < 32 or ord(character) == 127 for character in argument):
            return None, (f"deployment_argv[{index}] contains a control character",)
        argv.append(argument)
        total_length += len(argument)

    if total_length > _MAX_DEPLOYMENT_ARGV_LENGTH:
        return None, (
            f"deployment_argv exceeds the {_MAX_DEPLOYMENT_ARGV_LENGTH}-character total limit",
        )
    return tuple(argv), ()


def _deployment_export(
    candidate: Candidate,
    *,
    allow_flat_fallback: bool,
) -> tuple[str | None, tuple[str, ...], str]:
    parameters = candidate.parameters
    if "deployment_argv" in parameters:
        argv, problems = _authoritative_deployment_argv(parameters["deployment_argv"])
        if argv is None:
            return None, problems, "authoritative-invalid"
        return shlex.join(argv), (), "authoritative"

    if not allow_flat_fallback:
        return (
            None,
            ("deployment_argv is missing",),
            "authoritative-missing",
        )

    problems: list[str] = []

    runtime_binary = _safe_text_parameter(parameters.get("runtime_binary"))
    model_path = _safe_text_parameter(parameters.get("model_path"))
    threads = _positive_int(parameters.get("threads"), maximum=4096)
    batch_size = _positive_int(parameters.get("batch_size"), maximum=1_048_576)

    for key, value in (
        ("runtime_binary", runtime_binary),
        ("model_path", model_path),
        ("threads", threads),
        ("batch_size", batch_size),
    ):
        if value is None:
            state = "missing" if key not in parameters else "invalid"
            problems.append(f"{key} is {state}")

    context_size: int | None = None
    if "context_size" in parameters:
        context_size = _positive_int(parameters["context_size"], maximum=1_048_576)
        if context_size is None:
            problems.append("context_size is invalid")

    if problems:
        return None, tuple(problems), "synthetic-fallback-invalid"

    assert runtime_binary is not None
    assert model_path is not None
    assert threads is not None
    assert batch_size is not None
    argv = [
        runtime_binary,
        "--model",
        model_path,
        "--threads",
        str(threads),
        "--batch-size",
        str(batch_size),
        "--host",
        "127.0.0.1",
    ]
    if context_size is not None:
        argv.extend(("--ctx-size", str(context_size)))
    return shlex.join(argv), (), "synthetic-fallback"


def _deployment_section(selected: Candidate, *, synthetic_source: bool) -> str:
    command, problems, export_mode = _deployment_export(
        selected,
        allow_flat_fallback=synthetic_source,
    )
    required = ", ".join(_REQUIRED_DEPLOYMENT_PARAMETERS)
    build_note = ""
    if "kleidiai" in selected.parameters:
        enabled = selected.parameters["kleidiai"] is True
        build_note = (
            '<p class="quiet">Build property: KleidiAI must be '
            f"<strong>{'enabled' if enabled else 'disabled'}</strong>. This is a build-time "
            "property and is not invented as a runtime flag.</p>"
        )
    if command is None:
        problem_items = "".join(f"<li>{_escape(problem)}</li>" for problem in problems)
        if export_mode == "authoritative-invalid":
            policy_note = (
                "An authoritative deployment_argv was present but invalid, so ParetoPilot "
                "failed closed and did not try another command source."
            )
        elif export_mode == "authoritative-missing":
            policy_note = (
                "Measured evidence requires an authoritative deployment_argv. Flat-parameter "
                "command synthesis is reserved for explicitly synthetic fixtures."
            )
        else:
            policy_note = (
                f"Required synthetic-fixture parameters: {required}. Compatibility commands "
                "bind to 127.0.0.1 by default."
            )
        return (
            '<div class="deployment-state" role="status">'
            '<span class="status status-rejected">Not exportable</span>'
            "<p>ParetoPilot will not guess a launch command. The selected candidate lacks a "
            "complete, validated deployment parameter set.</p>"
            f'<ul class="reason-list">{problem_items}</ul>'
            f'<p class="quiet">{_escape(policy_note)}</p>'
            f"{build_note}</div>"
        )
    if export_mode == "authoritative":
        command_note = (
            "Rendered from the selected candidate’s authoritative deployment_argv. "
            "ParetoPilot added no executable, flags, or values."
        )
    else:
        command_note = (
            "Synthetic-fixture compatibility export derived from the validated flat parameters. "
            "It binds to 127.0.0.1; network exposure requires a separate decision."
        )
    return (
        '<div class="deployment-state" role="status">'
        '<span class="status status-selected">Exportable</span>'
        "<p>Validated POSIX launch command for the selected candidate. Review paths on the "
        "target Arm64 host before running it.</p>"
        f'<pre class="command"><code>{_escape(command)}</code></pre>'
        f'<p class="quiet">{_escape(command_note)}</p>'
        f"{build_note}</div>"
    )


def _metadata_rows(benchmarks: BenchmarkSet) -> str:
    if not benchmarks.metadata:
        return '<tr><td colspan="2" class="not-measured">Not provided</td></tr>'
    return "".join(
        "<tr>"
        f'<th scope="row">{_escape(key)}</th>'
        f"<td><code>{_escape(_stable_json(benchmarks.metadata[key]))}</code></td>"
        "</tr>"
        for key in sorted(benchmarks.metadata, key=str)
    )


def _source_hash(value: str) -> str:
    return _escape(value) if value.strip() else '<span class="not-measured">Not provided</span>'


def render_report(
    benchmarks: BenchmarkSet,
    constraints: Constraints,
    recommendation: Mapping[str, Any],
    *,
    benchmarks_sha256: str,
    constraints_sha256: str,
) -> str:
    """Render one deterministic, self-contained HTML decision report.

    ``recommendation`` is expected to be the mapping returned by
    :func:`paretopilot.analysis.recommend`. The function deliberately does not add a
    generation timestamp; source hashes provide the immutable provenance anchor.
    """

    selected_id = recommendation.get("selected_id")
    if not isinstance(selected_id, str):
        raise ValueError("recommendation.selected_id must be a string")
    try:
        selected = benchmarks.by_id(selected_id)
    except KeyError as error:
        raise ValueError("recommendation.selected_id is not present in benchmarks") from error

    recommendation_baseline = recommendation.get("baseline_id", benchmarks.baseline_id)
    if recommendation_baseline != benchmarks.baseline_id:
        raise ValueError("recommendation.baseline_id does not match benchmarks.baseline_id")
    if not isinstance(benchmarks_sha256, str) or not isinstance(constraints_sha256, str):
        raise TypeError("source hashes must be strings")

    baseline_retained = selected_id == benchmarks.baseline_id
    verdict_status = "Baseline retained" if baseline_retained else "Configuration selected"
    selection = recommendation.get("selection")
    selection_reason = str(selection.get("reason", "")) if isinstance(selection, Mapping) else ""
    preference_changed_winner = (
        selection.get("preference_changed_winner") is True
        if isinstance(selection, Mapping)
        else False
    )
    if baseline_retained:
        if preference_changed_winner:
            verdict_text = (
                "ParetoPilot retained the measured baseline under the predeclared objective "
                "tolerance and preference policy."
            )
        else:
            verdict_text = (
                "ParetoPilot retained the measured baseline. No alternative delivered a better "
                "eligible objective result on the declared Pareto frontier."
            )
    else:
        verdict_text = (
            f"{selected.label} was selected over the baseline for the declared "
            f"{_metric_label(constraints.objective.metric).lower()} objective."
        )
    if selection_reason:
        verdict_text += f" {selection_reason}"

    classification = benchmarks.metadata.get("classification")
    evidence_banner = ""
    if benchmarks.synthetic:
        evidence_banner = (
            '<aside class="synthetic-banner" role="alert">'
            "<strong>Synthetic evidence — do not cite as measured Arm performance.</strong> "
            "This report demonstrates the workflow only."
            "</aside>"
        )
    elif classification == "exploratory":
        evidence_banner = (
            '<aside class="synthetic-banner" role="alert">'
            "<strong>Exploratory evidence — not canonical submission evidence.</strong> "
            "This report came from a branch or non-default experiment input and must be rerun "
            "canonically before its measurements are cited."
            "</aside>"
        )

    if benchmarks.synthetic:
        source_type = "Synthetic fixture"
    elif classification == "canonical":
        source_type = "Canonical measured evidence"
    elif classification == "exploratory":
        source_type = "Exploratory measured evidence"
    else:
        source_type = "Measured evidence"
    paretopilot_version = recommendation.get("paretopilot_version", "Not provided")
    css = """
    :root {
      color-scheme: light;
      --bg: oklch(1 0 0);
      --surface: oklch(0.972 0.008 230);
      --surface-strong: oklch(0.936 0.014 230);
      --ink: oklch(0.225 0.028 230);
      --muted: oklch(0.455 0.025 230);
      --line: oklch(0.86 0.018 230);
      --line-strong: oklch(0.68 0.045 230);
      --accent: oklch(0.49 0.15 230);
      --accent-dark: oklch(0.34 0.12 230);
      --accent-soft: oklch(0.94 0.035 230);
      --success: oklch(0.43 0.115 155);
      --success-soft: oklch(0.95 0.035 155);
      --danger: oklch(0.45 0.15 28);
      --danger-soft: oklch(0.95 0.035 28);
      --warning: oklch(0.43 0.11 80);
      --warning-soft: oklch(0.95 0.055 80);
      --focus: oklch(0.58 0.18 230);
      --radius: 12px;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      font-size: 1rem;
      line-height: 1.55;
    }
    a { color: var(--accent-dark); }
    a:focus-visible, [tabindex]:focus-visible {
      outline: 3px solid var(--focus);
      outline-offset: 3px;
    }
    .skip-link {
      position: fixed;
      inset: 0 auto auto 0;
      transform: translateY(-130%);
      z-index: 30;
      padding: .65rem .9rem;
      background: var(--ink);
      color: var(--bg);
    }
    .skip-link:focus { transform: translateY(0); }
    .synthetic-banner {
      position: sticky;
      top: 0;
      z-index: 20;
      padding: .7rem max(.9rem, calc((100vw - 78rem) / 2));
      border-bottom: 1px solid var(--warning);
      background: var(--warning-soft);
      color: var(--ink);
      text-align: center;
    }
    .report-header, .report-main, .report-footer {
      width: min(calc(100% - 1.5rem), 78rem);
      margin-inline: auto;
    }
    .report-header { padding: 3rem 0 2.25rem; }
    .brand-line {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: .75rem 1.5rem;
      margin-bottom: 2rem;
      padding-bottom: .85rem;
      border-bottom: 1px solid var(--line);
    }
    .brand { font-size: 1.05rem; font-weight: 750; letter-spacing: -.015em; }
    .source-type { color: var(--muted); font-size: .92rem; }
    h1, h2, h3 { line-height: 1.18; letter-spacing: -.025em; text-wrap: balance; }
    h1 { max-width: 22ch; margin: 0; font-size: 2.25rem; }
    h2 { margin: 0 0 .65rem; font-size: 1.45rem; }
    h3 { margin: 0 0 .4rem; font-size: 1.08rem; }
    p { max-width: 72ch; text-wrap: pretty; }
    .verdict {
      display: grid;
      gap: 1rem;
      margin-top: 1.5rem;
      padding: 1.25rem;
      border: 1px solid var(--accent);
      border-radius: var(--radius);
      background: var(--accent-soft);
    }
    .verdict p { margin: 0; }
    .selected-name { display: block; margin-top: .2rem; font-size: 1.25rem; font-weight: 720; }
    .section {
      padding: 2.2rem 0;
      border-top: 1px solid var(--line);
    }
    .section-intro { margin: 0 0 1.25rem; color: var(--muted); }
    .status {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      min-height: 1.75rem;
      padding: .2rem .55rem;
      border-radius: 999px;
      font-size: .82rem;
      font-weight: 700;
      white-space: nowrap;
    }
    .status-selected { background: var(--success-soft); color: var(--success); }
    .status-rejected { background: var(--danger-soft); color: var(--danger); }
    .status-frontier { background: var(--accent-soft); color: var(--accent-dark); }
    .status-eligible, .status-neutral { background: var(--surface-strong); color: var(--ink); }
    .table-scroll, .chart-scroll {
      max-width: 100%;
      overflow-x: auto;
      overscroll-behavior-inline: contain;
    }
    table { width: 100%; border-collapse: collapse; font-size: .9rem; }
    caption { padding: 0 0 .65rem; color: var(--muted); text-align: left; }
    th, td {
      padding: .75rem;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    thead th { background: var(--surface); color: var(--ink); font-weight: 700; }
    tbody tr:last-child > * { border-bottom: 0; }
    .candidate-table { min-width: 72rem; }
    .candidate-name { display: block; min-width: 12rem; font-weight: 700; }
    .candidate-id { display: block; margin-top: .2rem; color: var(--muted); }
    code, pre { font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; }
    code { overflow-wrap: anywhere; }
    .evidence-details summary {
      width: fit-content;
      cursor: pointer;
      color: var(--accent-dark);
      font-weight: 700;
      white-space: nowrap;
    }
    .evidence-details[open] summary { margin-bottom: .75rem; }
    .parameter-details { min-width: 10rem; }
    .parameter-json {
      display: block;
      width: min(28rem, 70vw);
      max-height: 20rem;
      padding: .75rem;
      overflow: auto;
      border-radius: 6px;
      background: var(--surface);
      white-space: pre-wrap;
    }
    .quiet, .not-measured { color: var(--muted); }
    .reason-list { min-width: 15rem; margin: 0; padding-left: 1.1rem; }
    .reason-list li + li { margin-top: .3rem; }
    .impact-table, .guardrail-table, .metadata-table { min-width: 42rem; }
    .plot-fallback, .deployment-state {
      padding: 1rem;
      border-radius: var(--radius);
      background: var(--surface);
    }
    .plot-fallback { border: 1px solid var(--line-strong); }
    .pareto-figure { margin: 1.25rem 0 0; }
    .chart-scroll { border-bottom: 1px solid var(--line); }
    svg { display: block; width: 100%; min-width: 42rem; height: auto; }
    .chart-axis { stroke: var(--line-strong); stroke-width: 1.5; }
    .chart-axis-label, .chart-tick, .chart-point text {
      fill: var(--muted);
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      font-size: 11px;
    }
    .chart-axis-label { font-size: 12px; }
    .chart-point circle { fill: var(--accent); stroke: var(--bg); stroke-width: 3; }
    .chart-point-baseline circle { fill: var(--muted); }
    .chart-point-selected circle { fill: var(--success); r: 8px; }
    figcaption { margin-top: .65rem; color: var(--muted); font-size: .9rem; }
    .deployment-state p { margin-bottom: 0; }
    .command {
      margin: 1rem 0 0;
      padding: 1rem;
      overflow-x: auto;
      border-radius: 8px;
      background: var(--ink);
      color: var(--bg);
      white-space: pre;
    }
    .provenance-grid { display: grid; gap: 1.25rem; }
    .provenance-block { min-width: 0; }
    .hash-list { margin: 0; }
    .hash-list > div { padding: .65rem 0; border-bottom: 1px solid var(--line); }
    .hash-list dt { color: var(--muted); font-size: .85rem; }
    .hash-list dd { margin: .2rem 0 0; overflow-wrap: anywhere; }
    .report-footer {
      padding: 1.4rem 0 2.5rem;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: .9rem;
    }
    @media (min-width: 48rem) {
      .report-header { padding-top: 4rem; }
      .verdict { grid-template-columns: minmax(14rem, .7fr) minmax(22rem, 1.3fr); align-items: center; }
      .provenance-grid { grid-template-columns: minmax(0, .85fr) minmax(0, 1.15fr); }
    }
    @media (min-width: 80rem) {
      h1 { font-size: 2.6rem; }
      .section { padding-block: 2.75rem; }
    }
    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
      *, *::before, *::after { transition-duration: .01ms !important; }
    }
    @media print {
      .skip-link { display: none; }
      .synthetic-banner { position: static; }
      .report-header, .report-main, .report-footer { width: 100%; }
      .table-scroll, .chart-scroll { overflow: visible; }
      .candidate-table, .impact-table, .guardrail-table, .metadata-table, svg { min-width: 0; }
    }
    """

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<link rel="icon" href="data:,">\n'
        "<title>ParetoPilot decision report</title>\n"
        f"<style>{css}</style>\n"
        "</head>\n"
        "<body>\n"
        '<a class="skip-link" href="#main-content">Skip to report</a>\n'
        f"{evidence_banner}\n"
        '<header class="report-header">\n'
        '<div class="brand-line"><span class="brand">ParetoPilot</span>'
        f'<span class="source-type">{_escape(source_type)} · Decision evidence</span></div>\n'
        "<h1>Arm inference decision report</h1>\n"
        '<div class="verdict" role="status" aria-label="Recommendation outcome">\n'
        f'<div><span class="status status-selected">{_escape(verdict_status)}</span>'
        f'<span class="selected-name">{_escape(selected.label)}</span></div>\n'
        f"<p>{_escape(verdict_text)}</p>\n"
        "</div>\n"
        "</header>\n"
        '<main id="main-content" class="report-main">\n'
        '<section class="section" aria-labelledby="impact-heading">\n'
        '<h2 id="impact-heading">Measured impact</h2>\n'
        '<p class="section-intro">Selected-candidate values are compared directly with the '
        "declared baseline. Directional assessments only use frontier directions.</p>\n"
        '<div class="table-scroll" tabindex="0" role="region" '
        'aria-label="Scrollable impact comparison">\n'
        '<table class="impact-table"><caption>Selected candidate versus baseline</caption>\n'
        '<thead><tr><th scope="col">Metric</th><th scope="col">Baseline</th>'
        '<th scope="col">Selected</th><th scope="col">Change</th>'
        '<th scope="col">Assessment</th></tr></thead>\n'
        f"<tbody>{_impact_rows(benchmarks, constraints, selected)}</tbody></table></div>\n"
        "</section>\n"
        '<section class="section" aria-labelledby="guardrails-heading">\n'
        '<h2 id="guardrails-heading">Declared guardrails</h2>\n'
        '<p class="section-intro">These gates were declared before selection. Performix is an '
        "optional supplementary profiler and does not gate this recommendation.</p>\n"
        '<div class="table-scroll" tabindex="0" role="region" '
        'aria-label="Scrollable declared guardrails">\n'
        '<table class="guardrail-table"><caption>Constraint and Pareto policy</caption>\n'
        '<thead><tr><th scope="col">Rule</th><th scope="col">Metric</th>'
        '<th scope="col">Threshold or direction</th></tr></thead>\n'
        f"<tbody>{_guardrail_rows(constraints)}</tbody></table></div>\n"
        "</section>\n"
        '<section class="section" aria-labelledby="frontier-heading">\n'
        '<h2 id="frontier-heading">Pareto view</h2>\n'
        '<p class="section-intro">The scatter is rendered only when every candidate has both '
        "generation throughput and p95 end-to-end latency.</p>\n"
        f"{_pareto_visualisation(benchmarks, selected_id=selected_id)}\n"
        "</section>\n"
        '<section class="section" aria-labelledby="candidates-heading">\n'
        '<h2 id="candidates-heading">Candidate evidence</h2>\n'
        '<p class="section-intro">Missing measurements are never inferred. Rejection reasons '
        "are reproduced exactly from the recommendation record.</p>\n"
        f"{_candidate_table(benchmarks, constraints, recommendation, selected_id=selected_id)}\n"
        "</section>\n"
        '<section class="section" aria-labelledby="deployment-heading">\n'
        '<h2 id="deployment-heading">Deployment handoff</h2>\n'
        '<p class="section-intro">A command is emitted only from an allowlisted, validated '
        "parameter set. ParetoPilot does not guess paths or resource settings.</p>\n"
        f"{_deployment_section(selected, synthetic_source=benchmarks.synthetic)}\n"
        "</section>\n"
        '<section class="section" aria-labelledby="provenance-heading">\n'
        '<h2 id="provenance-heading">Provenance</h2>\n'
        '<p class="section-intro">Source hashes anchor this report to its benchmark and '
        "constraint inputs. No generation timestamp is added, preserving deterministic output.</p>\n"
        '<div class="provenance-grid">\n'
        '<div class="provenance-block"><h3>Source record</h3>\n'
        '<dl class="hash-list">\n'
        f"<div><dt>Benchmark SHA-256</dt><dd><code>{_source_hash(benchmarks_sha256)}</code></dd></div>\n"
        f"<div><dt>Constraints SHA-256</dt><dd><code>{_source_hash(constraints_sha256)}</code></dd></div>\n"
        f"<div><dt>Benchmark schema</dt><dd>{_escape(benchmarks.schema_version)}</dd></div>\n"
        f"<div><dt>ParetoPilot version</dt><dd>{_escape(paretopilot_version)}</dd></div>\n"
        f"<div><dt>Baseline ID</dt><dd><code>{_escape(benchmarks.baseline_id)}</code></dd></div>\n"
        f"<div><dt>Selected ID</dt><dd><code>{_escape(selected_id)}</code></dd></div>\n"
        "</dl></div>\n"
        '<div class="provenance-block"><h3>Benchmark metadata</h3>\n'
        '<details class="evidence-details metadata-details">\n'
        "<summary>View full source metadata</summary>\n"
        '<div class="table-scroll" tabindex="0" role="region" '
        'aria-label="Scrollable benchmark metadata">\n'
        '<table class="metadata-table"><caption>Source-supplied metadata</caption>\n'
        '<thead><tr><th scope="col">Field</th><th scope="col">Value</th></tr></thead>\n'
        f"<tbody>{_metadata_rows(benchmarks)}</tbody></table></div></details></div>\n"
        "</div>\n"
        "</section>\n"
        "</main>\n"
        '<footer class="report-footer">ParetoPilot preserves measurements, constraints, and '
        "selection rationale in one offline-reviewable artifact.</footer>\n"
        "</body>\n"
        "</html>\n"
    )
