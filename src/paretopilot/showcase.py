"""Judge-facing presentation view for locked ParetoPilot v1.1 evidence.

The canonical :mod:`paretopilot.report_v11` document is an immutable evidence
artifact whose bytes are bound into the v1.1 release.  This module deliberately
keeps that renderer untouched.  It first renders and, when supplied, byte-checks
the canonical document, then adds a separate editorial presentation layer for
GitHub Pages.

All measurements, recommendations, policy panels, load rows, stability rows,
source hashes, and accessible tables still come from the validated canonical
renderer.  The additions here are presentation-only: a provenance strip, an
objective-tolerance track, stable chart styling, responsive legends, and a
stronger visual hierarchy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import html
import math
import re
from typing import Any
from urllib.parse import urlsplit

from paretopilot.capacity_eval import validate_capacity_study
from paretopilot.decision_passport import build_decision_passport
from paretopilot.domain import BenchmarkSet, Candidate, Constraints, ValidationError
from paretopilot.report_v11 import render_report_v11


__all__ = ["render_showcase_v11"]


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHART_MARKER_RE = re.compile(
    r'<(?P<tag>circle|rect|path) class="(?P<classes>chart-marker[^"]*)"'
    r"(?P<attributes>[^>]*)></(?P=tag)>"
)
_SERIES_GROUP_RE = re.compile(
    r'(?P<open><g[^>]*data-series-style="(?P<style>\d+)"[^>]*>)'
    r"(?P<body>.*?)</g>",
    re.DOTALL,
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    return value


def _replace_once(document: str, needle: str, replacement: str, label: str) -> str:
    count = document.count(needle)
    if count != 1:
        raise ValidationError(
            f"canonical report anchor {label!r} occurred {count} times; expected exactly one"
        )
    return document.replace(needle, replacement, 1)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _format_number(value: float, *, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def _metric_label(metric: str) -> str:
    labels = {
        "e2e_latency_ms_p95": "p95 end-to-end latency",
        "ttft_ms_p95": "p95 time to first token",
        "prompt_tokens_per_second": "prompt processing throughput",
        "prompt_tps": "prompt processing throughput",
        "generation_tokens_per_second": "generation throughput",
        "generation_tps": "generation throughput",
        "peak_rss_mib": "peak resident memory",
        "model_size_mib": "model size",
        "quality_score": "quality score",
    }
    return labels.get(metric, metric.replace("_", " "))


def _metric_value(metric: str, value: float) -> str:
    if metric.endswith("_ms_p95") or metric.endswith("_ms_p50"):
        return f"{_format_number(value)} ms"
    if metric.endswith("_mib"):
        return f"{_format_number(value)} MiB"
    if metric in {
        "prompt_tokens_per_second",
        "prompt_tps",
        "generation_tokens_per_second",
        "generation_tps",
    }:
        return f"{_format_number(value, digits=2)} tok/s"
    if metric == "quality_score":
        return f"{value:.2f}"
    return _format_number(value, digits=2)


def _percent_delta(value: float, baseline: float) -> float | None:
    if math.isclose(baseline, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return None
    return ((value - baseline) / abs(baseline)) * 100.0


def _source_context(benchmarks: BenchmarkSet) -> Mapping[str, str]:
    metadata = _mapping(benchmarks.metadata, "benchmark metadata")
    source = metadata.get("source")
    source_mapping = source if isinstance(source, Mapping) else {}
    runner = source_mapping.get("runner")
    runner_mapping = runner if isinstance(runner, Mapping) else {}
    return {
        "run_id": str(source_mapping.get("run_id", "not supplied")),
        "cpu": str(runner_mapping.get("cpu", "Arm64 CPU")),
        "architecture": str(runner_mapping.get("architecture", "arm64")),
        "cpu_count": str(runner_mapping.get("cpu_count", "not supplied")),
        "os": str(runner_mapping.get("os", "Linux")),
    }


def _proof_context(
    evidence_lock: Mapping[str, Any] | None,
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
    *,
    canonical_sha256: str,
    benchmarks_sha256: str,
    recommendation_sha256: str,
    profiles_sha256: str,
    load_sha256: str,
    stability_sha256: str,
) -> Mapping[str, str]:
    if evidence_lock is None:
        return {}

    lock = _mapping(evidence_lock, "evidence_lock")
    if lock.get("schema_version") != "1.1" or lock.get("classification") != "canonical":
        raise ValidationError("evidence_lock must be canonical schema 1.1")

    source = _mapping(lock.get("source"), "evidence_lock.source")
    benchmark_source = _source_context(benchmarks)
    if str(source.get("run_id")) != benchmark_source["run_id"]:
        raise ValidationError("evidence_lock run id does not match benchmark metadata")

    review = _mapping(lock.get("review"), "evidence_lock.review")
    for flag in (
        "all_checksums_verified",
        "exact_file_coverage",
        "status_complete",
        "measurement_valid",
        "valid_evidence",
    ):
        if review.get(flag) is not True:
            raise ValidationError(f"evidence_lock review flag is not true: {flag}")

    replay = _mapping(review.get("replay"), "evidence_lock.review.replay")
    for flag in (
        "valid",
        "decision_reproduced",
        "fully_reproduced",
        "report_matches_archive",
    ):
        if replay.get(flag) is not True:
            raise ValidationError(f"evidence_lock replay flag is not true: {flag}")
    differences = replay.get("differences")
    warnings = replay.get("warnings")
    if differences != [] or warnings != []:
        raise ValidationError("evidence_lock replay must have no differences or warnings")

    comparisons = replay.get("authoritative_comparisons")
    if not isinstance(comparisons, Sequence) or isinstance(comparisons, (str, bytes)):
        raise ValidationError("evidence_lock authoritative comparisons must be a list")
    comparison_names = [str(item) for item in comparisons]
    if not comparison_names or len(set(comparison_names)) != len(comparison_names):
        raise ValidationError("evidence_lock authoritative comparisons must be unique")

    checksum_entries = review.get("checksum_entries")
    if not isinstance(checksum_entries, int) or isinstance(checksum_entries, bool):
        raise ValidationError("evidence_lock checksum_entries must be an integer")
    if checksum_entries <= 0:
        raise ValidationError("evidence_lock checksum_entries must be positive")

    checksum_digest = review.get("checksum_manifest_sha256")
    if not isinstance(checksum_digest, str) or _SHA256_RE.fullmatch(checksum_digest) is None:
        raise ValidationError("evidence_lock checksum manifest digest is invalid")

    replay_checksum_entries = replay.get("checksum_entry_count")
    replay_checksum_digest = replay.get("checksum_manifest_sha256")
    if replay_checksum_entries != checksum_entries:
        raise ValidationError("evidence_lock replay checksum count does not match review")
    if replay_checksum_digest != checksum_digest:
        raise ValidationError("evidence_lock replay checksum digest does not match review")
    if str(replay.get("selected_id")) != str(recommendation.get("selected_id")):
        raise ValidationError("evidence_lock replay selection does not match recommendation")

    artifact_digests = _mapping(
        review.get("artifacts_sha256"), "evidence_lock.review.artifacts_sha256"
    )
    expected_digests = {
        "benchmark_set": benchmarks_sha256,
        "recommendation": recommendation_sha256,
        "report_v1_1": canonical_sha256,
    }
    optional_digests = {
        "policy_profiles": profiles_sha256,
        "load_evaluation": load_sha256,
        "repeat_stability": stability_sha256,
    }
    expected_digests.update({name: digest for name, digest in optional_digests.items() if digest})
    for name, digest in expected_digests.items():
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValidationError(f"showcase input digest is invalid: {name}")
        if artifact_digests.get(name) != digest:
            raise ValidationError(
                f"evidence_lock artifact digest does not match showcase input: {name}"
            )

    archive = _mapping(lock.get("archive"), "evidence_lock.archive")
    release_url = archive.get("release_url")
    release_tag = archive.get("release_tag")
    if not isinstance(release_url, str) or not release_url.startswith("https://github.com/"):
        raise ValidationError("evidence_lock release_url must be a GitHub HTTPS URL")
    if not isinstance(release_tag, str) or not release_tag:
        raise ValidationError("evidence_lock release_tag must be non-empty")
    archive_digest = archive.get("sha256")
    if not isinstance(archive_digest, str) or _SHA256_RE.fullmatch(archive_digest) is None:
        raise ValidationError("evidence_lock archive digest is invalid")

    return {
        "archive_sha256": archive_digest,
        "checksum_entries": str(checksum_entries),
        "checksum_manifest_sha256": checksum_digest,
        "comparison_count": str(len(comparison_names)),
        "report_sha256": canonical_sha256,
        "release_tag": release_tag,
        "release_url": release_url,
    }


def _candidate_style_order(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
) -> tuple[Candidate, ...]:
    selected_id = str(recommendation["selected_id"])
    selected = benchmarks.by_id(selected_id)
    remaining = [
        candidate for candidate in benchmarks.candidates if candidate.candidate_id != selected_id
    ]
    if selected_id != benchmarks.baseline_id:
        baseline = benchmarks.baseline
        remaining = [baseline] + [
            candidate for candidate in remaining if candidate.candidate_id != baseline.candidate_id
        ]
    return (selected, *remaining)


def _marker_shape(style_index: int) -> str:
    return ("circle", "circle", "square", "triangle", "diamond", "bar")[style_index % 6]


def _legend_swatch(style_index: int) -> str:
    shape = _marker_shape(style_index)
    markers = {
        "circle": '<circle class="legend-marker" cx="34" cy="8" r="4"></circle>',
        "square": '<rect class="legend-marker" x="30" y="4" width="8" height="8"></rect>',
        "triangle": '<path class="legend-marker" d="M 34 3 L 39 12 L 29 12 Z"></path>',
        "diamond": '<path class="legend-marker" d="M 34 3 L 39 8 L 34 13 L 29 8 Z"></path>',
        "bar": '<rect class="legend-marker" x="29" y="5" width="10" height="6"></rect>',
    }
    marker = markers[shape]
    if style_index % 6 == 0:
        marker = marker.replace(
            'class="legend-marker"', 'class="legend-marker legend-marker-selected"'
        )
    return (
        f'<svg class="series-swatch" data-marker-shape="{shape}" '
        'viewBox="0 0 44 16" aria-hidden="true" '
        'focusable="false">'
        '<line class="legend-line" x1="1" y1="8" x2="43" y2="8"></line>'
        f"{marker}</svg>"
    )


def _attribute_values(attributes: str) -> Mapping[str, str]:
    return dict(re.findall(r'([a-z]+)="([^"]+)"', attributes))


def _chart_marker_center(match: re.Match[str]) -> tuple[float, float]:
    attributes = _attribute_values(match.group("attributes"))
    tag = match.group("tag")
    try:
        if tag == "circle":
            center = (float(attributes["cx"]), float(attributes["cy"]))
        elif tag == "rect":
            center = (
                float(attributes["x"]) + float(attributes["width"]) / 2.0,
                float(attributes["y"]) + float(attributes["height"]) / 2.0,
            )
        else:
            path_values = [
                float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", attributes["d"])
            ]
            if len(path_values) != 6:
                raise ValueError
            center = (path_values[0], path_values[1] + 5.0)
    except (KeyError, ValueError) as exc:
        raise ValidationError("canonical chart marker geometry is invalid") from exc
    if not all(math.isfinite(value) for value in center):
        raise ValidationError("canonical chart marker geometry must be finite")
    return center


def _styled_chart_marker(match: re.Match[str], *, style_index: int) -> str:
    x, y = _chart_marker_center(match)
    classes = match.group("classes")
    shape = _marker_shape(style_index)
    if shape == "circle":
        return f'<circle class="{classes}" cx="{x:.2f}" cy="{y:.2f}" r="4.5"></circle>'
    if shape == "square":
        return (
            f'<rect class="{classes}" x="{x - 4:.2f}" y="{y - 4:.2f}" width="8" height="8"></rect>'
        )
    if shape == "triangle":
        return (
            f'<path class="{classes}" d="M {x:.2f} {y - 5:.2f} '
            f'L {x + 5:.2f} {y + 4:.2f} L {x - 5:.2f} {y + 4:.2f} Z"></path>'
        )
    if shape == "diamond":
        return (
            f'<path class="{classes}" d="M {x:.2f} {y - 5:.2f} '
            f"L {x + 5:.2f} {y:.2f} L {x:.2f} {y + 5:.2f} "
            f'L {x - 5:.2f} {y:.2f} Z"></path>'
        )
    return f'<rect class="{classes}" x="{x - 5:.2f}" y="{y - 3:.2f}" width="10" height="6"></rect>'


def _normalize_chart_markers(document: str) -> str:
    def replace_group(match: re.Match[str]) -> str:
        style_index = int(match.group("style"))
        body = _CHART_MARKER_RE.sub(
            lambda marker: _styled_chart_marker(marker, style_index=style_index),
            match.group("body"),
        )
        return f"{match.group('open')}{body}</g>"

    return _SERIES_GROUP_RE.sub(replace_group, document)


def _series_key(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, Mapping[str, int]]:
    selected_id = str(recommendation["selected_id"])
    frontier_ids = {str(item) for item in recommendation.get("frontier_ids", ())}
    ordered = _candidate_style_order(benchmarks, recommendation)
    style_by_id = {candidate.candidate_id: index % 6 for index, candidate in enumerate(ordered)}
    items: list[str] = []
    for candidate in ordered:
        roles: list[str] = []
        if candidate.candidate_id == selected_id:
            roles.append("Selected")
        if candidate.candidate_id == benchmarks.baseline_id:
            roles.append("Baseline")
        if candidate.candidate_id in frontier_ids:
            roles.append("Frontier")
        role_markup = (
            f'<span class="series-role">{_escape(" · ".join(roles))}</span>' if roles else ""
        )
        style_index = style_by_id[candidate.candidate_id]
        marker_shape = _marker_shape(style_index)
        items.append(
            f'<li data-series-style="{style_index}" data-marker-shape="{marker_shape}">'
            f"{_legend_swatch(style_index)}"
            f'<span class="series-name">{_escape(candidate.label)}</span>'
            f"{role_markup}</li>"
        )
    return (
        f'<div class="series-key-wrap"><p>{_escape(label)}</p>'
        f'<ul class="series-key" aria-label="{_escape(label)}">{"".join(items)}</ul></div>',
        style_by_id,
    )


def _tag_chart_series(
    document: str,
    benchmarks: BenchmarkSet,
    style_by_id: Mapping[str, int],
) -> str:
    tagged = document
    for candidate in benchmarks.candidates:
        style_index = style_by_id[candidate.candidate_id]
        needle = f'<g aria-label="{_escape(candidate.label)}">'
        replacement = (
            f'<g aria-label="{_escape(candidate.label)}" '
            f'data-series-style="{style_index}" '
            f'data-marker-shape="{_marker_shape(style_index)}">'
        )
        tagged = tagged.replace(needle, replacement)

    scatter_start = tagged.find('<figure class="chart-figure scatter-figure">')
    if scatter_start == -1:
        return tagged
    scatter_end = tagged.find("</svg>", scatter_start)
    if scatter_end == -1:
        raise ValidationError("canonical scatter chart is missing its closing SVG tag")
    scatter = tagged[scatter_start:scatter_end]
    group_count = scatter.count("<g>")
    if group_count != len(benchmarks.candidates):
        raise ValidationError(
            "canonical scatter chart candidate group count does not match benchmarks"
        )
    for candidate in benchmarks.candidates:
        style_index = style_by_id[candidate.candidate_id]
        scatter = scatter.replace(
            "<g>",
            (
                f'<g aria-label="{_escape(candidate.label)}" '
                f'data-series-style="{style_index}" '
                f'data-marker-shape="{_marker_shape(style_index)}">'
            ),
            1,
        )
    tagged = tagged[:scatter_start] + scatter + tagged[scatter_end:]
    return _normalize_chart_markers(tagged)


def _wrap_table_region(document: str, *, aria_label: str, summary: str) -> str:
    """Collapse one long canonical evidence table without changing its contents."""

    anchor = (
        f'<div class="table-scroll" tabindex="0" role="region" aria-label="{_escape(aria_label)}">'
    )
    start = document.find(anchor)
    if start == -1:
        raise ValidationError(f"canonical table region is missing: {aria_label}")
    end = document.find("</div>", start)
    if end == -1:
        raise ValidationError(f"canonical table region is not closed: {aria_label}")
    end += len("</div>")
    region = document[start:end]
    disclosure = (
        f'<details class="data-disclosure"><summary>{_escape(summary)}</summary>{region}</details>'
    )
    return document[:start] + disclosure + document[end:]


def _add_section_kickers(document: str) -> str:
    stages = (
        ("why-heading", "01 · Decision rule"),
        ("tradeoffs-heading", "02 · Honest tradeoffs"),
        ("policies-heading", "03 · Policy lenses"),
        ("load-heading", "04 · Load test"),
        ("repeat-heading", "05 · Repeatability"),
        ("scatter-heading", "06 · Two-metric view"),
        ("evidence-heading", "07 · Evidence matrix"),
        ("trust-heading", "08 · Reproduction"),
    )
    result = document
    for heading_id, label in stages:
        needle = f'<h2 id="{heading_id}">'
        replacement = (
            f'<div class="section-title"><p class="section-kicker">{_escape(label)}</p>{needle}'
        )
        result = _replace_once(result, needle, replacement, f"{heading_id} heading")
        heading_close = result.find("</h2>", result.find(replacement))
        if heading_close == -1:
            raise ValidationError(f"canonical section heading is not closed: {heading_id}")
        heading_close += len("</h2>")
        result = result[:heading_close] + "</div>" + result[heading_close:]
    return result


def _label_chart_scroller(document: str, *, title_id: str, label: str) -> str:
    anchor = f'aria-labelledby="{title_id} '
    svg_start = document.find(anchor)
    if svg_start == -1:
        return document
    figure_start = document.rfind('<figure class="chart-figure', 0, svg_start)
    if figure_start == -1:
        raise ValidationError(f"chart figure is missing for {title_id}")
    figure_end = document.find(">", figure_start)
    if figure_end == -1:
        raise ValidationError(f"chart figure start tag is not closed for {title_id}")
    tag = document[figure_start : figure_end + 1]
    if "tabindex=" in tag:
        return document
    labelled_tag = (
        tag[:-1]
        + f' tabindex="0" role="region" aria-label="{_escape(f"Scrollable chart: {label}")}">'
    )
    scroll_hint = (
        '<p class="chart-scroll-hint" aria-hidden="true">Scroll the plot horizontally.</p>'
    )
    return document[:figure_start] + labelled_tag + scroll_hint + document[figure_end + 1 :]


def _label_interactive_regions(
    document: str,
    benchmarks: BenchmarkSet,
    load_sweep: Mapping[str, Any] | None,
) -> str:
    result = document.replace(
        '<main id="main-content" class="report-main">',
        '<main id="main-content" class="report-main" tabindex="-1">',
        1,
    )
    result = re.sub(
        r'(<section id="profile-panel-\d+" class="profile-panel" role="tabpanel")',
        r'\1 tabindex="0"',
        result,
    )
    for title_id, label in (
        ("load-request-throughput-title", "Request throughput by concurrency"),
        ("load-token-throughput-title", "Generated-token throughput by concurrency"),
        ("load-tail-latency-title", "p95 end-to-end latency by concurrency"),
        ("scatter-title", "Latency versus generation throughput"),
    ):
        result = _label_chart_scroller(result, title_id=title_id, label=label)

    for candidate in benchmarks.candidates:
        result = result.replace(
            "<summary>View configuration</summary>",
            f"<summary>View configuration for {_escape(candidate.label)}</summary>",
            1,
        )

    if load_sweep is not None:
        rows = load_sweep.get("rows")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            for raw_row in rows:
                row = _mapping(raw_row, "load_sweep row")
                candidate = benchmarks.by_id(str(row.get("candidate_id")))
                concurrency = row.get("concurrency")
                result = result.replace(
                    "<summary>View samples</summary>",
                    (
                        f"<summary>View {_escape(candidate.label)} samples at "
                        f"concurrency {_escape(concurrency)}</summary>"
                    ),
                    1,
                )

    result = result.replace(
        '<code class="json-block">',
        '<code class="json-block" tabindex="0">',
    )
    result = result.replace(
        '<pre class="command">',
        (
            '<pre class="command" tabindex="0" role="region" '
            'aria-label="Selected deployment command">'
        ),
    )
    result = result.replace(
        '<span role="columnheader">Visual link</span>',
        "",
    )
    return result


def _add_load_slo_reference(
    document: str,
    load_sweep: Mapping[str, Any] | None,
    *,
    synthetic: bool,
) -> str:
    if load_sweep is None:
        return document
    slo = load_sweep.get("slo")
    rows = load_sweep.get("rows")
    if not isinstance(slo, Mapping) or not isinstance(rows, Sequence):
        return document
    raw_threshold = slo.get("max_e2e_latency_ms_p95")
    if not isinstance(raw_threshold, (int, float)) or isinstance(raw_threshold, bool):
        return document
    values: list[float] = []
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            continue
        raw_value = raw_row.get("e2e_latency_ms_p95")
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            values.append(float(raw_value))
    if not values:
        return document
    maximum = max(values)
    y_domain_max = maximum * 1.08 if maximum > 0 else 1.0
    threshold = float(raw_threshold)
    title_anchor = '<title id="load-tail-latency-title">'
    title_start = document.find(title_anchor)
    if title_start == -1:
        return document
    insert_anchor = '<text class="chart-tick" x="76" y="278">'
    insert_at = document.find(insert_anchor, title_start)
    if insert_at == -1:
        raise ValidationError("tail-latency chart tick anchor is missing")
    if threshold > y_domain_max:
        threshold_markup = (
            '<text class="slo-reference-label slo-reference-label-above" '
            'x="333" y="18" text-anchor="middle">'
            f"SLO · {_escape(_format_number(threshold, digits=0))} ms · "
            "above plotted range</text>"
        )
    else:
        y_position = 258.0 + (threshold / y_domain_max) * (30.0 - 258.0)
        y_position = min(258.0, max(30.0, y_position))
        threshold_markup = (
            f'<line class="slo-reference-line" x1="76" y1="{y_position:.2f}" '
            f'x2="590" y2="{y_position:.2f}"></line>'
            f'<text class="slo-reference-label" x="84" y="{y_position - 7:.2f}" '
            'text-anchor="start">'
            f"SLO · {_escape(_format_number(threshold, digits=0))} ms</text>"
        )
    result = document[:insert_at] + threshold_markup + document[insert_at:]
    old_description = (
        "Measured p95 end-to-end response latency as concurrent request count increases."
    )
    display_description = (
        "Synthetic fixture p95 end-to-end response latency as concurrent request count increases."
        if synthetic
        else old_description
    )
    if threshold > y_domain_max:
        new_description = (
            f"{display_description} The declared {_format_number(threshold, digits=0)} ms "
            f"latency ceiling is above the plotted {'fixture' if synthetic else 'measured'} "
            "range; a passing level must "
            "also satisfy the TTFT and completion-rate gates."
        )
    else:
        new_description = (
            f"{display_description} The amber line marks the declared "
            f"{_format_number(threshold, digits=0)} ms latency ceiling; a passing level "
            "must also satisfy the TTFT and completion-rate gates."
        )
    return result.replace(old_description, new_description)


def _correct_load_axis_ceilings(
    document: str,
    load_sweep: Mapping[str, Any] | None,
) -> str:
    """Label each load chart with the domain ceiling actually used to plot it."""

    if load_sweep is None:
        return document
    rows = load_sweep.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return document
    charts = (
        ("load-request-throughput-title", "requests_per_second", "req/s"),
        (
            "load-token-throughput-title",
            "generated_tokens_per_second",
            "tok/s",
        ),
        ("load-tail-latency-title", "e2e_latency_ms_p95", "ms"),
    )
    result = document
    tick_anchor = '<text class="chart-tick" x="66" y="34">'
    for title_id, metric, unit in charts:
        values: list[float] = []
        for raw_row in rows:
            if not isinstance(raw_row, Mapping):
                continue
            raw_value = raw_row.get(metric)
            if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
                values.append(float(raw_value))
        if not values:
            continue
        maximum = max(values)
        domain_ceiling = maximum * 1.08 if maximum > 0 else 1.0
        title_start = result.find(f'<title id="{title_id}">')
        if title_start == -1:
            raise ValidationError(f"load chart title is missing: {title_id}")
        tick_start = result.find(tick_anchor, title_start)
        if tick_start == -1:
            raise ValidationError(f"load chart top tick is missing: {title_id}")
        value_start = tick_start + len(tick_anchor)
        value_end = result.find("</text>", value_start)
        if value_end == -1:
            raise ValidationError(f"load chart top tick is not closed: {title_id}")
        formatted = f"{domain_ceiling:,.4f}".rstrip("0").rstrip(".")
        result = result[:value_start] + f"{formatted} {unit}" + result[value_end:]
    return result


def _validated_href(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label} must be a non-empty string")
    if "\\" in value:
        raise ValidationError(f"{label} must use URL path separators")
    parsed = urlsplit(value)
    if parsed.query or parsed.fragment:
        raise ValidationError(f"{label} must not contain a query or fragment")
    if "%" in parsed.path:
        raise ValidationError(f"{label} must not contain encoded path segments")
    if parsed.scheme:
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValidationError(f"{label} must be a relative path or an HTTPS URL")
        raw_parts = parsed.path.split("/")[1:]
    else:
        if parsed.netloc or value.startswith(("/", "#")):
            raise ValidationError(f"{label} must be a relative path or an HTTPS URL")
        raw_parts = parsed.path.split("/")
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValidationError(f"{label} must be a safe file path")
    return value


def _validated_evidence_href(value: str, *, label: str, filename: str) -> str:
    validated = _validated_href(value, label=label)
    if urlsplit(validated).path.rsplit("/", 1)[-1] != filename:
        raise ValidationError(f"{label} must point to {filename}")
    return validated


def _validated_report_href(value: str) -> str:
    validated = _validated_href(value, label="canonical_report_href")
    if urlsplit(validated).path.rsplit("/", 1)[-1] != "report-v1.1.html":
        raise ValidationError("canonical_report_href must point to report-v1.1.html")
    return validated


def _add_release_hashes(document: str, proof: Mapping[str, str]) -> str:
    if not proof:
        return document
    document = _replace_once(
        document,
        "<h3>Source anchors</h3>",
        "<h3>Source and release anchors</h3>",
        "trust anchor heading",
    )
    anchor = "<div><dt>Benchmark schema</dt>"
    release_hashes = (
        "<div><dt>Canonical report SHA-256</dt>"
        f"<dd><code>{_escape(proof['report_sha256'])}</code></dd></div>"
        "<div><dt>Evidence archive SHA-256</dt>"
        f"<dd><code>{_escape(proof['archive_sha256'])}</code></dd></div>"
        "<div><dt>Checksum manifest SHA-256</dt>"
        f"<dd><code>{_escape(proof['checksum_manifest_sha256'])}</code></dd></div>"
    )
    return _replace_once(
        document,
        anchor,
        f"{release_hashes}{anchor}",
        "trust release hashes",
    )


def _add_stability_explainer(
    document: str,
    stability_summary: Mapping[str, Any] | None,
) -> str:
    if stability_summary is None:
        return document
    anchor = '<div class="stability-method">'
    start = document.find(anchor)
    if start == -1:
        raise ValidationError("stability method anchor is missing")
    end = document.find("</div>", start)
    if end == -1:
        raise ValidationError("stability method is not closed")
    end += len("</div>")
    explanation = (
        '<p class="stability-explainer"><strong>How to read the deltas:</strong> '
        "positive means improvement after applying each metric’s declared direction; "
        "negative means regression. “Consistent” means the two observed passes had the "
        "same comparison direction—it is not a statistical-significance threshold.</p>"
    )
    return document[:end] + explanation + document[end:]


def _tolerance_visual(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
) -> str:
    objective = _mapping(recommendation.get("objective"), "recommendation.objective")
    selection = _mapping(recommendation.get("selection"), "recommendation.selection")
    metric = str(objective.get("metric"))
    direction = str(objective.get("direction"))
    if direction not in {"min", "max"}:
        raise ValidationError("recommendation objective direction must be min or max")

    rows: list[tuple[Candidate, float]] = []
    for candidate in benchmarks.candidates:
        value = candidate.metrics.get(metric)
        if value is not None:
            rows.append((candidate, float(value)))
    if not rows:
        raise ValidationError("recommendation objective is missing from every candidate")

    numeric_best = float(selection.get("numeric_best_value"))
    tolerance = float(selection.get("objective_tolerance_percent"))
    boundary = (
        numeric_best + abs(numeric_best) * tolerance / 100.0
        if direction == "min"
        else numeric_best - abs(numeric_best) * tolerance / 100.0
    )
    values = [value for _, value in rows]
    domain_low = min(*values, boundary)
    domain_high = max(*values, boundary)
    span = domain_high - domain_low
    padding = max(span * 0.07, abs(domain_low) * 0.0025, 1e-9)
    domain_low -= padding
    domain_high += padding
    domain_span = domain_high - domain_low

    def position(value: float) -> float:
        return min(100.0, max(0.0, ((value - domain_low) / domain_span) * 100.0))

    shortlist = {str(item) for item in selection.get("shortlist_ids", ())}
    selected_id = str(recommendation.get("selected_id"))
    ordered_rows = sorted(rows, key=lambda item: item[1], reverse=direction == "max")
    row_markup: list[str] = []
    for candidate, value in ordered_rows:
        inside = candidate.candidate_id in shortlist
        classes = ["tolerance-row"]
        if inside:
            classes.append("is-inside")
        if candidate.candidate_id == selected_id:
            classes.append("is-selected")
        role = "Inside cutoff" if inside else "Outside cutoff"
        if candidate.candidate_id == selected_id:
            role = f"{role} · selected"
        row_markup.append(
            f'<li class="{" ".join(classes)}">'
            '<div class="tolerance-name">'
            f"<strong>{_escape(candidate.label)}</strong>"
            f"<span>{_escape(role)}</span></div>"
            '<div class="tolerance-scale" aria-hidden="true">'
            f'<span class="tolerance-cutoff" style="--position: {position(boundary):.4f}%"></span>'
            f'<span class="tolerance-marker" style="--position: {position(value):.4f}%"></span>'
            "</div>"
            f'<span class="tolerance-value">{_escape(_metric_value(metric, value))}</span>'
            "</li>"
        )

    direction_copy = "Lower is better" if direction == "min" else "Higher is better"
    value_kind = "fixture" if benchmarks.synthetic else "measured"
    return (
        '<figure class="tolerance-visual" aria-labelledby="tolerance-visual-title">'
        '<div class="tolerance-visual-heading">'
        '<div><p class="visual-kicker">Decision track</p>'
        '<h3 id="tolerance-visual-title">'
        f"{_escape(f'{tolerance:.2f}% objective tolerance')}</h3></div>"
        f"<p><strong>{_escape(direction_copy)}.</strong> The predeclared cutoff was "
        f"<strong>{_escape(_metric_value(metric, boundary))}</strong> for "
        f"{_escape(_metric_label(metric))}.</p></div>"
        '<div class="tolerance-direction" aria-hidden="true">'
        f"<span>{'Faster' if direction == 'min' else 'Lower'}</span>"
        f"<span>Cutoff · {_escape(_metric_value(metric, boundary))}</span>"
        f"<span>{'Slower' if direction == 'min' else 'Higher'}</span></div>"
        f'<ol class="tolerance-list">{"".join(row_markup)}</ol>'
        f"<figcaption>Marker positions show the {value_kind} objective values on one shared scale. "
        "Exact values and decision roles remain in the evidence table below.</figcaption>"
        "</figure>"
    )


_ATTRIBUTION_STAGE_LABELS = {
    "reference": "Reference",
    "quantization": "Quantization",
    "arm-kernel": "KleidiAI build",
    "runtime-tuning": "Runtime tuning",
}
_MEASURED_EFFECT_LABELS = {
    "improved": "Measured improvement",
    "tradeoff": "Measured tradeoff",
    "changed": "Measured change",
    "held": "Held",
}
_SYNTHETIC_EFFECT_LABELS = {
    "improved": "Fixture improvement",
    "tradeoff": "Fixture tradeoff",
    "changed": "Fixture change",
    "held": "Held",
}


def _decision_passport(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
) -> Mapping[str, Any]:
    constraints = Constraints.from_mapping(
        _mapping(recommendation.get("constraints"), "recommendation.constraints")
    )
    passport = _mapping(
        build_decision_passport(benchmarks, constraints),
        "decision passport",
    )
    selected = _mapping(passport.get("selected_decision"), "decision passport selected_decision")
    if selected.get("candidate_id") != recommendation.get("selected_id"):
        raise ValidationError(
            "decision passport selected candidate does not match the supplied recommendation"
        )
    return passport


def _stage_label(stage: Mapping[str, Any], *, synthetic: bool) -> str:
    attribution_stage = stage.get("attribution_stage")
    if stage.get("recognized_attribution_stage") is True:
        label = _ATTRIBUTION_STAGE_LABELS.get(str(attribution_stage))
        if label is not None:
            return label
    stage_kind = "Synthetic fixture" if synthetic else "Measured"
    return f"{stage_kind} stage {stage.get('stage')}"


def _change_phrase(metric: str, change: Mapping[str, Any]) -> str:
    raw_percent = change.get("percent")
    if isinstance(raw_percent, (int, float)) and not isinstance(raw_percent, bool):
        percent = float(raw_percent)
        if not math.isfinite(percent):
            raise ValidationError("decision passport change percent must be finite")
        if math.isclose(percent, 0.0, rel_tol=1e-9, abs_tol=1e-12):
            return "held"
        direction = "lower" if percent < 0 else "higher"
        return f"{_format_number(abs(percent), digits=2)}% {direction}"

    previous = change.get("previous")
    current = change.get("current")
    if (
        isinstance(previous, (int, float))
        and not isinstance(previous, bool)
        and isinstance(current, (int, float))
        and not isinstance(current, bool)
    ):
        return (
            f"{_metric_value(metric, float(previous))} to {_metric_value(metric, float(current))}"
        )
    raise ValidationError("decision passport change is missing comparable metric values")


def _representative_changes(
    stage: Mapping[str, Any],
    *,
    objective_metric: str,
) -> tuple[Mapping[str, Any], ...]:
    raw_delta = stage.get("delta_from_previous")
    if raw_delta is None:
        return ()
    delta = _mapping(raw_delta, "decision passport ladder delta")
    raw_metrics = delta.get("metrics")
    if not isinstance(raw_metrics, Sequence) or isinstance(raw_metrics, (str, bytes)):
        raise ValidationError("decision passport ladder metrics must be an array")

    changes: list[Mapping[str, Any]] = []
    for raw_change in raw_metrics:
        change = _mapping(raw_change, "decision passport ladder metric")
        if change.get("effect") == "held":
            continue
        raw_percent = change.get("percent")
        if raw_percent is not None and (
            isinstance(raw_percent, bool)
            or not isinstance(raw_percent, (int, float))
            or not math.isfinite(float(raw_percent))
        ):
            raise ValidationError("decision passport ladder metric percent must be finite")
        changes.append(change)

    def magnitude(change: Mapping[str, Any]) -> tuple[float, str]:
        raw_percent = change.get("percent")
        percent = abs(float(raw_percent)) if raw_percent is not None else -1.0
        return (-percent, str(change.get("metric")))

    objective_changes = [
        change for change in changes if str(change.get("metric")) == objective_metric
    ]
    other_changes = [change for change in changes if str(change.get("metric")) != objective_metric]
    ordered = objective_changes[:1] + sorted(other_changes, key=magnitude)
    return tuple(ordered[:3])


def _optimization_ladder_markup(passport: Mapping[str, Any]) -> str:
    objective = _mapping(passport.get("objective"), "decision passport objective")
    selected = _mapping(
        passport.get("selected_decision"),
        "decision passport selected_decision",
    )
    raw_ladder = passport.get("ladder")
    if not isinstance(raw_ladder, Sequence) or isinstance(raw_ladder, (str, bytes)):
        raise ValidationError("decision passport ladder must be an array")
    if not raw_ladder:
        raise ValidationError("decision passport ladder must contain at least one stage")

    evidence_grade = str(passport.get("evidence_grade"))
    synthetic_evidence = evidence_grade == "synthetic"
    stage_count = len(raw_ladder)
    stage_count_label = {
        1: "One",
        2: "Two",
        3: "Three",
        4: "Four",
        5: "Five",
        6: "Six",
    }.get(stage_count, str(stage_count))
    stage_noun = "stage" if stage_count == 1 else "stages"
    stage_heading = f"{stage_count_label} {stage_noun}. One honest runway."
    path_label = "Synthetic fixture path" if synthetic_evidence else "Measured optimization path"
    candidate_kind = "synthetic" if synthetic_evidence else "measured"
    effect_labels = _SYNTHETIC_EFFECT_LABELS if synthetic_evidence else _MEASURED_EFFECT_LABELS
    stages_label = (
        "Synthetic fixture optimization stages"
        if synthetic_evidence
        else "Measured optimization stages"
    )
    metric = str(objective.get("metric"))
    direction = str(objective.get("direction"))
    boundary = float(objective.get("shortlist_boundary"))
    selected_runway = _mapping(
        objective.get("selected_runway"),
        "decision passport selected runway",
    )
    runway_value = float(selected_runway.get("absolute"))
    cutoff_symbol = "≤" if direction == "min" else "≥"
    runway_text = (
        "on the current cutoff"
        if math.isclose(runway_value, 0.0, rel_tol=1e-9, abs_tol=1e-12)
        else f"{_metric_value(metric, runway_value)} inside the current cutoff"
    )

    raw_closest = passport.get("closest_outside_shortlist")
    closest = (
        _mapping(raw_closest, "decision passport closest outside shortlist")
        if raw_closest is not None
        else None
    )
    closest_id = str(closest.get("candidate_id")) if closest is not None else None
    if closest is None:
        closest_markup = (
            "<div><dt>Closest outside-shortlist stage</dt>"
            "<dd>Every frontier stage is inside the current shortlist.</dd></div>"
        )
    else:
        shortfall = _mapping(
            closest.get("shortfall_to_shortlist"),
            "decision passport shortlist shortfall",
        )
        closest_markup = (
            "<div><dt>Closest outside-shortlist stage</dt>"
            f"<dd>{_escape(closest.get('label'))}"
            f"<span>{_escape(_metric_value(metric, float(shortfall.get('absolute'))))} "
            "outside the current cutoff</span></dd></div>"
        )

    stage_markup: list[str] = []
    for raw_stage in raw_ladder:
        stage = _mapping(raw_stage, "decision passport ladder stage")
        candidate_id = str(stage.get("candidate_id"))
        stage_number = int(stage.get("stage"))
        stage_classes = ["optimization-stage"]
        if stage.get("selected") is True:
            stage_classes.append("is-selected")
            decision_label = "Canonical selected stage"
        elif candidate_id == closest_id:
            stage_classes.append("is-closest")
            decision_label = "Closest outside shortlist"
        elif stage.get("shortlisted") is True:
            decision_label = "Inside current shortlist"
        elif stage.get("eligible") is True and stage.get("frontier") is True:
            decision_label = "Eligible frontier stage"
        elif stage.get("eligible") is True:
            decision_label = f"Eligible {candidate_kind} stage"
        else:
            decision_label = "Outside declared constraints"

        objective_value = stage.get("objective_value")
        objective_text = (
            _metric_value(metric, float(objective_value))
            if isinstance(objective_value, (int, float)) and not isinstance(objective_value, bool)
            else "Unavailable"
        )
        changes = _representative_changes(stage, objective_metric=metric)
        if changes:
            change_items = []
            for change in changes:
                change_metric = str(change.get("metric"))
                effect = str(change.get("effect"))
                effect_label = effect_labels.get(
                    effect,
                    "Fixture change" if synthetic_evidence else "Measured change",
                )
                change_items.append(
                    f'<li class="is-{_escape(effect)}">'
                    f"<span>{_escape(_metric_label(change_metric))}</span>"
                    f"<strong>{_escape(_change_phrase(change_metric, change))}</strong>"
                    f"<em>{_escape(effect_label)}</em></li>"
                )
            changes_markup = (
                f'<ul class="stage-changes" aria-label="Largest {candidate_kind} changes from the '
                f'previous stage">{"".join(change_items)}</ul>'
            )
        else:
            reference_kind = "fixture" if synthetic_evidence else "measurement"
            changes_markup = (
                f'<p class="stage-reference-note">Reference {reference_kind} for every later '
                "stage-to-stage comparison.</p>"
            )

        stage_markup.append(
            f'<li class="{" ".join(stage_classes)}">'
            f'<span class="stage-marker" aria-hidden="true">{stage_number:02d}</span>'
            '<div class="stage-body">'
            f'<p class="stage-role">{_escape(_stage_label(stage, synthetic=synthetic_evidence))}</p>'
            f"<h3>{_escape(stage.get('label'))}</h3>"
            f'<code class="stage-id">{_escape(candidate_id)}</code>'
            f'<span class="stage-decision-label">{_escape(decision_label)}</span>'
            '<p class="stage-objective">'
            f"<span>{_escape(_metric_label(metric))}</span>"
            f"<strong>{_escape(objective_text)}"
            "</strong></p>"
            f"{changes_markup}</div></li>"
        )

    method = _mapping(passport.get("method"), "decision passport method")
    boundary_caveat = str(method.get("current_boundary_caveat"))
    return (
        '<section id="optimization-ladder" class="optimization-ladder" '
        'aria-labelledby="optimization-ladder-heading">'
        '<div class="optimization-ladder-inner">'
        '<header class="optimization-ladder-heading">'
        f'<div class="section-title"><p class="section-kicker">00 · {_escape(path_label)}</p>'
        f'<h2 id="optimization-ladder-heading">{_escape(stage_heading)}</h2>'
        '</div><div class="ladder-intro-copy">'
        "<p>Each stop comes from the deterministic decision passport. Its highlighted changes "
        f"compare that {candidate_kind} candidate with the previous stage; the objective is "
        "always shown."
        '</p><p class="ladder-evidence-grade"><span>Evidence grade</span>'
        f"<strong>{_escape(evidence_grade)}</strong></p></div></header>"
        '<div class="ladder-runway" role="group" aria-label="Honest runway to the current cutoff">'
        '<p class="runway-call-sign">Current objective runway</p><dl>'
        "<div><dt>Current shortlist cutoff</dt>"
        f"<dd>{_escape(cutoff_symbol)} {_escape(_metric_value(metric, boundary))}"
        f"<span>{_escape(_metric_label(metric))}</span></dd></div>"
        "<div><dt>Canonical selected stage</dt>"
        f"<dd>{_escape(selected.get('label'))}"
        f"<span>{_escape(runway_text)}</span></dd></div>"
        f"{closest_markup}</dl></div>"
        f'<ol class="optimization-stages" style="--stage-count: {stage_count}" '
        f'aria-label="{_escape(stages_label)}">'
        f"{''.join(stage_markup)}</ol>"
        '<p class="optimization-ladder-caveat"><strong>Decision boundary:</strong> '
        "This derived attribution view does not recalculate or replace the canonical decision. "
        f"{_escape(boundary_caveat)}</p>"
        "</div></section>\n"
    )


def _locked_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_true_flags(
    source: Mapping[str, Any],
    names: Sequence[str],
    *,
    label: str,
) -> None:
    for name in names:
        if source.get(name) is not True:
            raise ValidationError(f"{label} flag is not true: {name}")


def _capacity_result_from_study(study: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recompute the compact publication result from validated study data."""

    plan = _mapping(study.get("plan"), "capacity study plan")
    load_contract = _mapping(study.get("load_contract"), "capacity load contract")
    methodology = _mapping(
        load_contract.get("methodology"),
        "capacity load methodology",
    )
    candidates = [
        _mapping(candidate, "capacity candidate") for candidate in plan.get("candidates", ())
    ]
    cells = [_mapping(cell, "capacity cell") for cell in study.get("cells", ())]
    selections = [
        _mapping(selection, "capacity selection") for selection in study.get("selections", ())
    ]
    quality_checks = [
        _mapping(check, "capacity quality check") for check in study.get("quality_checks", ())
    ]
    passes = list(plan.get("passes", ()))

    cell_index = {
        (
            str(cell.get("candidate_id")),
            int(cell.get("server_parallel")),
            int(cell.get("client_concurrency")),
        ): cell
        for cell in cells
    }
    selection_index = {str(selection.get("candidate_id")): selection for selection in selections}
    selected_points: dict[str, Any] = {}
    selected_summaries: dict[str, Mapping[str, Any]] = {}

    for candidate in candidates:
        candidate_id = str(candidate.get("id"))
        role = str(candidate.get("role"))
        selection = _mapping(
            selection_index.get(candidate_id),
            f"capacity selection for {candidate_id}",
        )
        selected = _mapping(
            selection.get("selected_cell"),
            f"capacity selected cell for {candidate_id}",
        )
        selected_key = (
            candidate_id,
            int(selected.get("server_parallel")),
            int(selected.get("client_concurrency")),
        )
        selected_cell = cell_index.get(selected_key)
        if selected_cell is None:
            raise ValidationError(f"capacity selected cell is missing: {candidate_id}")
        summary = _mapping(
            selected_cell.get("summary"),
            f"capacity selected summary for {candidate_id}",
        )
        selected_summaries[role] = summary

        selected_quality = [
            check
            for check in quality_checks
            if str(check.get("candidate_id")) == candidate_id
            and int(check.get("server_parallel")) == selected_key[1]
        ]
        if len(selected_quality) != 1:
            raise ValidationError(f"capacity selected quality check must be unique: {candidate_id}")
        quality = selected_quality[0]
        candidate_quality = [
            check for check in quality_checks if str(check.get("candidate_id")) == candidate_id
        ]
        outcomes_consistent = (
            bool(candidate_quality)
            and len({str(check.get("outcomes_sha256")) for check in candidate_quality}) == 1
        )
        comparison = _mapping(
            selection.get("comparison_to_reference_percent"),
            f"capacity reference comparison for {candidate_id}",
        )

        selected_points[candidate_id] = {
            "label": candidate.get("label"),
            "role": candidate.get("role"),
            "server_parallel": selected_key[1],
            "client_concurrency": selected_key[2],
            "eligible_cell_count": int(selection.get("eligible_cell_count")),
            "within_tolerance_cell_count": int(selection.get("within_tolerance_cell_count")),
            "generated_tokens_per_second_median": summary.get("generated_tokens_per_second_median"),
            "ttft_ms_p95_median": summary.get("ttft_ms_p95_median"),
            "e2e_latency_ms_p95_median": summary.get("e2e_latency_ms_p95_median"),
            "server_peak_rss_mib_max": summary.get("server_peak_rss_mib_max"),
            "gain_vs_own_p1c1_percent": comparison.get("generated_tokens_per_second_median"),
            "quality": {
                "passed": int(quality.get("passed")),
                "total": int(quality.get("total")),
                "score": quality.get("score"),
                "retention_vs_reference": quality.get("retention_vs_reference"),
                "outcomes_consistent_across_parallel": outcomes_consistent,
                "gate_met": quality.get("gate_met"),
            },
        }

    canonical = selected_summaries.get("canonical-reference")
    alternative = selected_summaries.get("resource-alternative")
    if canonical is None or alternative is None:
        raise ValidationError("capacity result requires canonical and alternative roles")
    comparison_metrics = (
        "generated_tokens_per_second_median",
        "ttft_ms_p95_median",
        "e2e_latency_ms_p95_median",
        "server_peak_rss_mib_max",
    )
    selected_comparison: dict[str, float] = {}
    for metric in comparison_metrics:
        delta = _percent_delta(
            float(alternative.get(metric)),
            float(canonical.get(metric)),
        )
        if delta is None:
            raise ValidationError(f"capacity selected comparison has zero baseline: {metric}")
        selected_comparison[metric] = delta

    measured_requests = sum(
        int(_mapping(metric, "capacity pass metric").get("request_count"))
        for cell in cells
        for metric in cell.get("pass_metrics", ())
    )
    warmups_per_level = int(methodology.get("warmup_requests_per_level"))
    return {
        "pass_count": len(passes),
        "candidate_count": len(candidates),
        "cell_count": len(cells),
        "eligible_cell_count": sum(
            int(selection.get("eligible_cell_count")) for selection in selections
        ),
        "measured_request_count": measured_requests,
        "warmup_request_count": len(cells) * len(passes) * warmups_per_level,
        "output_tokens_per_request": int(methodology.get("output_tokens")),
        "selected_operating_points": selected_points,
        "q4_vs_q8_at_selected_points_percent": selected_comparison,
    }


def _capacity_values_match(locked: object, expected: object) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(locked, Mapping) or set(locked) != set(expected):
            return False
        return all(_capacity_values_match(locked[name], expected[name]) for name in expected)
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if (
            not isinstance(locked, Sequence)
            or isinstance(locked, (str, bytes))
            or len(locked) != len(expected)
        ):
            return False
        return all(
            _capacity_values_match(locked_value, expected_value)
            for locked_value, expected_value in zip(locked, expected, strict=True)
        )
    if isinstance(expected, bool):
        return locked is expected
    if isinstance(expected, float):
        return (
            isinstance(locked, (int, float))
            and not isinstance(locked, bool)
            and math.isfinite(float(locked))
            and math.isclose(float(locked), expected, rel_tol=1e-12, abs_tol=1e-12)
        )
    if isinstance(expected, int):
        return isinstance(locked, int) and not isinstance(locked, bool) and locked == expected
    return locked == expected


def _capacity_proof_context(
    study: Mapping[str, Any],
    capacity_lock: Mapping[str, Any],
    canonical_lock: Mapping[str, Any],
    *,
    study_sha256: str,
    canonical_lock_sha256: str,
) -> Mapping[str, str]:
    """Bind a validated capacity study to both reviewed release locks."""

    validate_capacity_study(study)
    study_digest = _locked_sha256(study_sha256, "capacity_study_sha256")
    canonical_lock_digest = _locked_sha256(
        canonical_lock_sha256,
        "evidence_lock_sha256",
    )

    lock = _mapping(capacity_lock, "capacity_evidence_lock")
    if (
        lock.get("schema_version") != "1.4"
        or lock.get("classification") != "supplementary-capacity"
    ):
        raise ValidationError("capacity_evidence_lock must be supplementary-capacity schema 1.4")

    study_provenance = _mapping(study.get("provenance"), "capacity study provenance")
    study_source = _mapping(study_provenance.get("source"), "capacity study source")
    lock_source = _mapping(lock.get("source"), "capacity_evidence_lock.source")
    source_pairs = {
        "run_id": "run_id",
        "run_attempt": "run_attempt",
        "revision": "head_sha",
        "workflow": "workflow",
    }
    for study_name, lock_name in source_pairs.items():
        if study_source.get(study_name) != lock_source.get(lock_name):
            raise ValidationError(f"capacity evidence source does not match study: {lock_name}")
    if lock_source.get("runner") != study_provenance.get("runner"):
        raise ValidationError("capacity evidence runner does not match study")

    review = _mapping(lock.get("review"), "capacity_evidence_lock.review")
    _require_true_flags(
        review,
        (
            "all_checksums_verified",
            "archive_digest_matches_actions_digest",
            "exact_file_coverage",
            "status_complete",
            "measurement_valid",
            "valid_evidence",
        ),
        label="capacity_evidence_lock.review",
    )
    if review.get("synthetic") is not False or study.get("synthetic") is not False:
        raise ValidationError("capacity evidence must be measured, not synthetic")
    if (
        review.get("canonical_outputs_modified") is not False
        or study.get("canonical_outputs_modified") is not False
    ):
        raise ValidationError("capacity evidence must not modify canonical outputs")

    checksum_entries = review.get("checksum_entries")
    if (
        not isinstance(checksum_entries, int)
        or isinstance(checksum_entries, bool)
        or checksum_entries <= 0
    ):
        raise ValidationError("capacity checksum_entries must be a positive integer")
    checksum_digest = _locked_sha256(
        review.get("checksum_manifest_sha256"),
        "capacity checksum manifest",
    )
    artifacts = _mapping(
        review.get("artifacts_sha256"),
        "capacity_evidence_lock.review.artifacts_sha256",
    )
    for name, digest in artifacts.items():
        _locked_sha256(digest, f"capacity artifact {name}")
    if artifacts.get("capacity_study") != study_digest:
        raise ValidationError("capacity study digest does not match its evidence lock")

    recomputation = _mapping(
        review.get("recomputation"),
        "capacity_evidence_lock.review.recomputation",
    )
    _require_true_flags(
        recomputation,
        (
            "raw_inputs_reassembled",
            "capacity_study_exact_match",
            "capacity_receipt_regenerated",
            "capacity_receipt_exact_match",
        ),
        label="capacity_evidence_lock.review.recomputation",
    )
    if recomputation.get("mismatched_cell_count") != 0:
        raise ValidationError("capacity recomputation contains mismatched cells")
    if recomputation.get("failed_request_count") != 0:
        raise ValidationError("capacity recomputation contains request errors")

    locked_result = _mapping(lock.get("result"), "capacity_evidence_lock.result")
    expected_result = _capacity_result_from_study(study)
    if not _capacity_values_match(locked_result, expected_result):
        raise ValidationError("capacity locked result does not match the validated study")
    if recomputation.get("recomputed_cell_count") != expected_result["cell_count"]:
        raise ValidationError("capacity recomputation cell count does not match the study")
    if recomputation.get("measured_request_count") != expected_result["measured_request_count"]:
        raise ValidationError("capacity recomputation request count does not match the study")
    if recomputation.get("completed_request_count") != expected_result["measured_request_count"]:
        raise ValidationError("capacity recomputation did not complete every request")

    replay = _mapping(review.get("replay"), "capacity_evidence_lock.review.replay")
    _require_true_flags(
        replay,
        (
            "valid",
            "decision_reproduced",
            "fully_reproduced",
            "authoritative_outputs_match",
            "report_matches_archive",
        ),
        label="capacity_evidence_lock.review.replay",
    )
    if replay.get("differences") != [] or replay.get("warnings") != []:
        raise ValidationError("capacity embedded canonical replay is not clean")

    capacity_canonical = _mapping(
        lock.get("canonical_evidence"),
        "capacity_evidence_lock.canonical_evidence",
    )
    study_canonical = _mapping(
        study_provenance.get("canonical_evidence"),
        "capacity study canonical evidence",
    )
    if capacity_canonical.get("outputs_modified") is not False:
        raise ValidationError("capacity lock says canonical outputs were modified")
    for field in ("run_id", "release_tag", "release_sha256", "lock_sha256"):
        if capacity_canonical.get(field) != study_canonical.get(field):
            raise ValidationError(f"capacity lock canonical linkage does not match study: {field}")

    canonical_source = _mapping(canonical_lock.get("source"), "evidence_lock.source")
    canonical_archive = _mapping(canonical_lock.get("archive"), "evidence_lock.archive")
    canonical_expectations = {
        "run_id": str(canonical_source.get("run_id")),
        "release_tag": canonical_archive.get("release_tag"),
        "release_sha256": canonical_archive.get("sha256"),
        "lock_sha256": canonical_lock_digest,
    }
    for field, expected in canonical_expectations.items():
        if capacity_canonical.get(field) != expected:
            raise ValidationError(f"capacity evidence does not match canonical lock: {field}")

    archive = _mapping(lock.get("archive"), "capacity_evidence_lock.archive")
    archive_digest = _locked_sha256(
        archive.get("sha256"),
        "capacity release archive",
    )
    if archive.get("actions_digest") != f"sha256:{archive_digest}":
        raise ValidationError("capacity release archive does not match the Actions digest")
    archive_size = archive.get("size_bytes")
    if not isinstance(archive_size, int) or isinstance(archive_size, bool) or archive_size <= 0:
        raise ValidationError("capacity release archive size must be a positive integer")
    release_tag = archive.get("release_tag")
    if not isinstance(release_tag, str) or not release_tag:
        raise ValidationError("capacity release tag must be non-empty")
    asset_name = archive.get("release_asset_name")
    if not isinstance(asset_name, str) or not asset_name or "/" in asset_name or "\\" in asset_name:
        raise ValidationError("capacity release asset name must be a safe filename")
    raw_archive_url = archive.get("release_asset_url")
    if not isinstance(raw_archive_url, str):
        raise ValidationError("capacity release asset URL must be a non-empty string")
    archive_url = _validated_href(
        raw_archive_url,
        label="capacity release asset URL",
    )
    repository = study_source.get("repository")
    if not isinstance(repository, str) or repository.count("/") != 1:
        raise ValidationError("capacity study repository must be owner/name")
    expected_archive_url = (
        f"https://github.com/{repository}/releases/download/{release_tag}/{asset_name}"
    )
    if archive_url != expected_archive_url:
        raise ValidationError("capacity release asset URL does not match its repository lock")

    return {
        "archive_sha256": archive_digest,
        "archive_url": archive_url,
        "checksum_entries": str(checksum_entries),
        "checksum_manifest_sha256": checksum_digest,
        "release_tag": release_tag,
        "study_sha256": study_digest,
    }


def _relative_measure_phrase(
    delta_percent: float,
    metric: str,
) -> str:
    if delta_percent == 0:
        return f"no change in {metric}"
    direction = "more" if delta_percent > 0 else "less"
    return f"{_format_number(abs(delta_percent), digits=2)}% {direction} {metric}"


def _capacity_failure_label(reasons: object) -> tuple[str, str]:
    if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes)):
        raise ValidationError("capacity failure reasons must be an array")
    normalized = [str(reason) for reason in reasons]
    labels: list[str] = []
    if any("ttft_ms_p95_above_maximum" in reason for reason in normalized):
        labels.append("TTFT")
    if any("e2e_latency_ms_p95_above_maximum" in reason for reason in normalized):
        labels.append("E2E")
    if not labels:
        labels.append("Gate")
    short = " + ".join(labels)
    expanded = " and ".join(
        {
            "TTFT": "time to first token",
            "E2E": "end-to-end latency",
            "Gate": "one or more declared gates",
        }[label]
        for label in labels
    )
    return short, expanded


def _capacity_envelope_markup(
    study: Mapping[str, Any],
    proof: Mapping[str, str],
    *,
    study_href: str,
    receipt_href: str,
) -> str:
    """Render the measured 3 by 3 operating envelope without browser-side math."""

    study_href = _validated_evidence_href(
        study_href,
        label="capacity_study_href",
        filename="capacity-study.json",
    )
    receipt_href = _validated_evidence_href(
        receipt_href,
        label="capacity_receipt_href",
        filename="capacity-receipt.md",
    )
    plan = _mapping(study.get("plan"), "capacity study plan")
    raw_candidates = plan.get("candidates")
    raw_cells = study.get("cells")
    raw_selections = study.get("selections")
    raw_quality = study.get("quality_checks")
    raw_passes = plan.get("passes")
    if any(
        not isinstance(value, Sequence) or isinstance(value, (str, bytes))
        for value in (raw_candidates, raw_cells, raw_selections, raw_quality, raw_passes)
    ):
        raise ValidationError("capacity study presentation arrays are invalid")

    candidates = [_mapping(item, "capacity candidate") for item in raw_candidates]
    cells = [_mapping(item, "capacity cell") for item in raw_cells]
    selections = {
        str(item.get("candidate_id")): item
        for item in (_mapping(raw_item, "capacity selection") for raw_item in raw_selections)
    }
    quality_checks = [_mapping(raw_item, "capacity quality check") for raw_item in raw_quality]
    levels_parallel = [int(value) for value in plan.get("server_parallel_levels", ())]
    levels_concurrency = [int(value) for value in plan.get("client_concurrency_levels", ())]
    if levels_parallel != [1, 2, 4] or levels_concurrency != [1, 2, 4]:
        raise ValidationError("capacity presentation requires the reviewed 1, 2, 4 matrix")

    cell_index = {
        (
            str(cell.get("candidate_id")),
            int(cell.get("server_parallel")),
            int(cell.get("client_concurrency")),
        ): cell
        for cell in cells
    }
    measured_request_count = sum(
        int(pass_metric.get("request_count"))
        for cell in cells
        for pass_metric in (
            _mapping(item, "capacity pass metric") for item in cell.get("pass_metrics", ())
        )
    )
    blocked_count = sum(
        _mapping(cell.get("summary"), "capacity cell summary").get("capacity_gate_met") is not True
        for cell in cells
    )

    candidate_boards: list[str] = []
    selected_summaries: dict[str, Mapping[str, Any]] = {}
    selected_points_by_role: dict[str, tuple[int, int]] = {}
    candidate_by_role = {str(candidate.get("role")): candidate for candidate in candidates}
    for candidate in candidates:
        candidate_id = str(candidate.get("id"))
        selection = _mapping(
            selections.get(candidate_id),
            f"capacity selection for {candidate_id}",
        )
        selected = _mapping(selection.get("selected_cell"), "capacity selected cell")
        reference = _mapping(selection.get("reference_cell"), "capacity reference cell")
        selected_key = (
            candidate_id,
            int(selected.get("server_parallel")),
            int(selected.get("client_concurrency")),
        )
        reference_key = (
            candidate_id,
            int(reference.get("server_parallel")),
            int(reference.get("client_concurrency")),
        )
        selected_cell = cell_index.get(selected_key)
        if selected_cell is None:
            raise ValidationError(f"capacity selected cell is missing: {candidate_id}")
        selected_summary = _mapping(
            selected_cell.get("summary"),
            "capacity selected cell summary",
        )
        selected_summaries[str(candidate.get("role"))] = selected_summary
        selected_points_by_role[str(candidate.get("role"))] = (
            selected_key[1],
            selected_key[2],
        )

        quality = next(
            (
                check
                for check in quality_checks
                if str(check.get("candidate_id")) == candidate_id
                and int(check.get("server_parallel")) == selected_key[1]
            ),
            None,
        )
        if quality is None:
            raise ValidationError(f"capacity selected quality check is missing: {candidate_id}")

        rows: list[str] = []
        blocked_items: list[str] = []
        for server_parallel in levels_parallel:
            row_cells: list[str] = []
            for client_concurrency in levels_concurrency:
                key = (candidate_id, server_parallel, client_concurrency)
                cell = cell_index.get(key)
                if cell is None:
                    raise ValidationError(f"capacity matrix cell is missing: {key!r}")
                summary = _mapping(cell.get("summary"), "capacity matrix summary")
                gate_met = summary.get("capacity_gate_met") is True
                is_selected = key == selected_key
                is_reference = key == reference_key
                if is_selected:
                    state = "Selected"
                    state_class = "is-selected"
                    reason_markup = ""
                elif gate_met:
                    state = "Pass"
                    state_class = "is-pass"
                    reason_markup = ""
                else:
                    state = "Blocked"
                    state_class = "is-blocked"
                    reason, expanded_reason = _capacity_failure_label(
                        summary.get("failure_reasons")
                    )
                    reason_markup = (
                        f'<span class="capacity-failure" aria-label="Blocked by '
                        f'{_escape(expanded_reason)}">{_escape(reason)}</span>'
                    )
                    blocked_items.append(
                        f"<li><strong>P{server_parallel} / C{client_concurrency}</strong>"
                        f"<span>{_escape(expanded_reason.capitalize())} exceeded the "
                        "predeclared limit.</span></li>"
                    )

                reference_markup = (
                    '<span class="capacity-reference">Reference</span>' if is_reference else ""
                )
                row_cells.append(
                    f'<td class="capacity-cell {state_class}" '
                    f'data-capacity-state="{state.lower()}">'
                    '<div class="capacity-cell-top">'
                    f'<span class="capacity-state">{_escape(state)}</span>{reference_markup}'
                    "</div>"
                    '<p class="capacity-rate">'
                    f"<strong>{_format_number(float(summary.get('generated_tokens_per_second_median')), digits=2)}</strong>"
                    "<span>tok/s</span></p>"
                    '<dl class="capacity-cell-metrics">'
                    "<div><dt>E2E</dt>"
                    f"<dd>{_format_number(float(summary.get('e2e_latency_ms_p95_median')), digits=0)} ms</dd></div>"
                    "<div><dt>TTFT</dt>"
                    f"<dd>{_format_number(float(summary.get('ttft_ms_p95_median')), digits=0)} ms</dd></div>"
                    "</dl>"
                    f"{reason_markup}</td>"
                )
            rows.append(
                f'<tr><th scope="row"><span>Server slots</span>P{server_parallel}</th>'
                f"{''.join(row_cells)}</tr>"
            )

        role = str(candidate.get("role"))
        role_label = {
            "canonical-reference": "Canonical reference",
            "resource-alternative": "Resource alternative",
        }.get(role, role.replace("-", " ").title())
        accent_class = "is-q8" if role == "canonical-reference" else "is-q4"
        candidate_boards.append(
            f'<article class="capacity-board {accent_class}">'
            '<header class="capacity-board-heading">'
            f"<p>{_escape(role_label)}</p><h3>{_escape(candidate.get('label'))}</h3>"
            '<dl class="capacity-selected-summary">'
            f"<div><dt>Selected point</dt><dd>P{selected_key[1]} / C{selected_key[2]}</dd></div>"
            "<div><dt>Peak RSS</dt>"
            f"<dd>{_format_number(float(selected_summary.get('server_peak_rss_mib_max')), digits=1)} MiB</dd></div>"
            "<div><dt>Quality</dt>"
            f"<dd>{int(quality.get('passed'))}/{int(quality.get('total'))}</dd></div>"
            f"<div><dt>Eligible</dt><dd>{int(selection.get('eligible_cell_count'))}/9</dd></div>"
            "</dl></header>"
            '<div class="capacity-table-wrap">'
            f'<table class="capacity-matrix"><caption>{_escape(candidate.get("label"))} '
            "serving-capacity matrix</caption>"
            '<thead><tr><th scope="col">P × C</th>'
            + "".join(
                f'<th scope="col"><span>Clients</span>C{level}</th>' for level in levels_concurrency
            )
            + f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            '<details class="capacity-blocked-details"><summary>Why cells were blocked</summary>'
            f"<ul>{''.join(blocked_items)}</ul></details></article>"
        )

    canonical_candidate = candidate_by_role.get("canonical-reference")
    alternative_candidate = candidate_by_role.get("resource-alternative")
    canonical_summary = selected_summaries.get("canonical-reference")
    alternative_summary = selected_summaries.get("resource-alternative")
    canonical_point = selected_points_by_role.get("canonical-reference")
    alternative_point = selected_points_by_role.get("resource-alternative")
    if (
        canonical_candidate is None
        or alternative_candidate is None
        or canonical_summary is None
        or alternative_summary is None
        or canonical_point is None
        or alternative_point is None
    ):
        raise ValidationError("capacity study must include canonical and alternative roles")
    alternative_label = str(alternative_candidate.get("label"))
    throughput_delta = _percent_delta(
        float(alternative_summary.get("generated_tokens_per_second_median")),
        float(canonical_summary.get("generated_tokens_per_second_median")),
    )
    rss_delta = _percent_delta(
        float(alternative_summary.get("server_peak_rss_mib_max")),
        float(canonical_summary.get("server_peak_rss_mib_max")),
    )
    if throughput_delta is None or rss_delta is None:
        raise ValidationError("capacity selected-point comparison cannot use a zero baseline")
    throughput_phrase = _relative_measure_phrase(
        throughput_delta,
        "generation throughput",
    )
    rss_phrase = _relative_measure_phrase(rss_delta, "peak RSS")
    if canonical_point == alternative_point:
        parallel, concurrency = canonical_point
        heading = f"The envelope opens at {parallel} × {concurrency}."
        selection_summary = (
            "The predeclared gate-and-tiebreaker policy selected "
            f"P{parallel}/C{concurrency} for both candidates."
        )
        comparison_label = f"At separately selected P{parallel}/C{concurrency} points:"
    else:
        heading = "Two candidates. Two selected envelopes."
        selection_summary = (
            "The predeclared gate-and-tiebreaker policy selected "
            f"Q8 P{canonical_point[0]}/C{canonical_point[1]} and "
            f"Q4 P{alternative_point[0]}/C{alternative_point[1]}."
        )
        comparison_label = (
            f"At Q8 P{canonical_point[0]}/C{canonical_point[1]} and "
            f"Q4 P{alternative_point[0]}/C{alternative_point[1]}:"
        )

    load_slo = _mapping(plan.get("load_slo"), "capacity load SLO")
    capacity_gate = _mapping(plan.get("capacity_gate"), "capacity gate")
    return (
        '<section id="capacity-envelope" class="capacity-envelope" '
        'aria-labelledby="capacity-heading">'
        '<div class="capacity-envelope-inner">'
        '<header class="capacity-heading"><div class="section-title">'
        '<p class="section-kicker">S1 · Supplementary capacity</p>'
        f'<h2 id="capacity-heading">{_escape(heading)}</h2></div>'
        f"<div><p>{_escape(selection_summary)} This sizes each candidate; it does "
        "not replace the canonical Q8 model decision.</p>"
        f'<p class="capacity-compare"><strong>{_escape(comparison_label)}</strong> '
        f"compared with Q8, {_escape(alternative_label)} measured {throughput_phrase} "
        f"and {rss_phrase}.</p></div></header>"
        '<dl class="capacity-facts" aria-label="Capacity study at a glance">'
        f"<div><dt>Operating points</dt><dd>{len(cells)}</dd></div>"
        f"<div><dt>Measured requests</dt><dd>{measured_request_count}</dd></div>"
        f"<div><dt>Exact-reversal passes</dt><dd>{len(raw_passes)}</dd></div>"
        f"<div><dt>Latency-blocked cells</dt><dd>{blocked_count}</dd></div>"
        "</dl>"
        f'<div class="capacity-boards">{"".join(candidate_boards)}</div>'
        '<dl class="capacity-slo-strip" aria-label="Predeclared capacity limits">'
        "<div><dt>TTFT p95</dt>"
        f"<dd>≤ {_format_number(float(load_slo.get('max_ttft_ms_p95')), digits=0)} ms</dd></div>"
        "<div><dt>E2E p95</dt>"
        f"<dd>≤ {_format_number(float(load_slo.get('max_e2e_latency_ms_p95')), digits=0)} ms</dd></div>"
        "<div><dt>Peak RSS</dt>"
        f"<dd>≤ {_format_number(float(capacity_gate.get('max_server_peak_rss_mib')), digits=0)} MiB</dd></div>"
        "<div><dt>Completion</dt>"
        f"<dd>{float(load_slo.get('min_completion_rate')) * 100:.0f}%</dd></div></dl>"
        '<nav class="capacity-actions" aria-label="Capacity evidence links">'
        f'<a href="{_escape(receipt_href)}">Open capacity receipt</a>'
        f'<a href="{_escape(study_href)}">Inspect validated JSON</a>'
        f'<a href="{_escape(proof["archive_url"])}">Download {_escape(proof["release_tag"])} evidence</a>'
        "</nav>"
        '<p class="capacity-proof-line"><strong>Locked proof:</strong> '
        f"{_escape(proof['checksum_entries'])} payload checksums · study "
        f"<code>{_escape(proof['study_sha256'][:12])}…</code> · archive "
        f"<code>{_escape(proof['archive_sha256'][:12])}…</code></p>"
        '<p class="capacity-boundary"><strong>Boundary:</strong> This is a bounded, closed-loop '
        "study on one native Arm64 runner. Each displayed p95 is the median of two pass-level "
        "p95 values; within each pass, eight measured requests make p95 the observed maximum. "
        "The KleidiAI marker confirms the enabled model-buffer path, not universal microkernel "
        "execution.</p>"
        "</div></section>\n"
    )


_COCKPIT_PROFILES = (
    ("canonical-latency", "Latency first", "Canonical"),
    ("memory-first", "Memory first", "Derived"),
    ("first-token-first", "First token first", "Derived"),
)
_COCKPIT_METRIC_DIRECTIONS = {
    "e2e_latency_ms_p95": "min",
    "ttft_ms_p95": "min",
    "prompt_tokens_per_second": "max",
    "prompt_tps": "max",
    "generation_tokens_per_second": "max",
    "generation_tps": "max",
    "peak_rss_mib": "min",
    "model_size_mib": "min",
    "quality_score": "max",
}


def _cockpit_profile_entries(
    policy_profiles: Mapping[str, Any] | None,
) -> Mapping[str, Mapping[str, Any]]:
    """Return the three presentation profiles only when all are supplied."""

    if not isinstance(policy_profiles, Mapping):
        return {}
    raw_profiles = policy_profiles.get("profiles")
    entries: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw_profiles, Sequence) and not isinstance(raw_profiles, (str, bytes)):
        for raw_entry in raw_profiles:
            if not isinstance(raw_entry, Mapping):
                continue
            profile_id = raw_entry.get("id")
            if isinstance(profile_id, str):
                entries[profile_id] = raw_entry
    elif isinstance(raw_profiles, Mapping):
        for profile_id, raw_entry in raw_profiles.items():
            if isinstance(profile_id, str) and isinstance(raw_entry, Mapping):
                entries[profile_id] = raw_entry
    else:
        for profile_id, raw_entry in policy_profiles.items():
            if isinstance(profile_id, str) and isinstance(raw_entry, Mapping):
                entries[profile_id] = raw_entry

    required_ids = {profile_id for profile_id, _, _ in _COCKPIT_PROFILES}
    if not required_ids <= set(entries):
        return {}
    return {profile_id: entries[profile_id] for profile_id, _, _ in _COCKPIT_PROFILES}


def _cockpit_recommendation(
    profile: Mapping[str, Any],
    *,
    profile_id: str,
) -> Mapping[str, Any]:
    raw_recommendation = profile.get("recommendation", profile.get("decision", profile))
    return _mapping(raw_recommendation, f"policy_profiles.{profile_id}.recommendation")


def _cockpit_delta(
    recommendation: Mapping[str, Any],
    *,
    improvement: bool,
) -> tuple[str, float] | None:
    raw_deltas = recommendation.get("deltas_vs_baseline")
    if not isinstance(raw_deltas, Mapping):
        return None
    ranked: list[tuple[float, str, float]] = []
    for metric, raw_delta in raw_deltas.items():
        if not isinstance(metric, str) or metric not in _COCKPIT_METRIC_DIRECTIONS:
            continue
        if not isinstance(raw_delta, Mapping):
            continue
        raw_percent = raw_delta.get("percent")
        if (
            isinstance(raw_percent, bool)
            or not isinstance(raw_percent, (int, float))
            or not math.isfinite(float(raw_percent))
        ):
            continue
        percent = float(raw_percent)
        if math.isclose(percent, 0.0, rel_tol=0.0, abs_tol=1e-12):
            continue
        direction = _COCKPIT_METRIC_DIRECTIONS[metric]
        is_improvement = percent < 0.0 if direction == "min" else percent > 0.0
        if is_improvement is improvement:
            ranked.append((abs(percent), metric, percent))
    if not ranked:
        return None
    _, metric, percent = sorted(ranked, key=lambda item: (-item[0], item[1]))[0]
    return metric, percent


def _cockpit_delta_markup(
    recommendation: Mapping[str, Any],
    *,
    improvement: bool,
    synthetic: bool,
) -> str:
    selected_delta = _cockpit_delta(recommendation, improvement=improvement)
    label = (
        ("Fixture improvement" if synthetic else "Strongest improvement")
        if improvement
        else ("Fixture tradeoff" if synthetic else "Main tradeoff")
    )
    class_name = "is-improvement" if improvement else "is-tradeoff"
    if selected_delta is None:
        baseline_copy = (
            "No beneficial profile delta versus the fixture baseline."
            if improvement and synthetic
            else (
                "No adverse profile delta versus the fixture baseline."
                if synthetic
                else (
                    "No beneficial profile delta versus the measured baseline."
                    if improvement
                    else "No adverse profile delta versus the measured baseline."
                )
            )
        )
        return (
            f'<div class="{class_name}"><dt>{_escape(label)}</dt>'
            f"<dd><strong>No change</strong><span>{_escape(baseline_copy)}</span></dd></div>"
        )

    metric, percent = selected_delta
    direction = _COCKPIT_METRIC_DIRECTIONS[metric]
    change = (
        ("lower" if percent < 0.0 else "higher")
        if direction == "min"
        else ("higher" if percent > 0.0 else "lower")
    )
    return (
        f'<div class="{class_name}"><dt>{_escape(label)}</dt>'
        f"<dd><strong>{abs(percent):,.1f}% {change}</strong>"
        f"<span>{_escape(_metric_label(metric))} vs baseline</span></dd></div>"
    )


def _policy_cockpit_markup(
    benchmarks: BenchmarkSet,
    policy_profiles: Mapping[str, Any] | None,
) -> str:
    entries = _cockpit_profile_entries(policy_profiles)
    if not entries:
        return ""

    tabs: list[str] = []
    panels: list[str] = []
    for index, (profile_id, control_label, classification) in enumerate(_COCKPIT_PROFILES):
        profile = entries[profile_id]
        recommendation = _cockpit_recommendation(profile, profile_id=profile_id)
        selected_id = str(recommendation.get("selected_id"))
        selected = benchmarks.by_id(selected_id)
        objective = _mapping(
            recommendation.get("objective"),
            f"policy_profiles.{profile_id}.objective",
        )
        objective_metric = str(objective.get("metric"))
        if objective_metric not in selected.metrics:
            raise ValidationError(
                f"policy_profiles.{profile_id} objective metric is missing from its selection"
            )

        selected_state = "true" if index == 0 else "false"
        tab_index = "0" if index == 0 else "-1"
        hidden = "" if index == 0 else " hidden"
        evidence_label = (
            ("Primary fixture" if index == 0 else "Derived fixture")
            if benchmarks.synthetic
            else classification
        )
        tabs.append(
            f'<button id="cockpit-tab-{index}" type="button" role="tab" '
            f'aria-selected="{selected_state}" aria-controls="cockpit-panel-{index}" '
            f'data-cockpit-target="{index}" tabindex="{tab_index}">'
            f"<strong>{_escape(control_label)}</strong><span>{_escape(evidence_label)}</span>"
            "</button>"
        )
        panels.append(
            f'<section id="cockpit-panel-{index}" class="cockpit-panel" role="tabpanel" '
            f'aria-labelledby="cockpit-tab-{index}" data-cockpit-panel="{index}"{hidden}>'
            '<div class="cockpit-decision">'
            f'<p class="cockpit-classification">{_escape(evidence_label)} result</p>'
            f"<h3>{_escape(selected.label)}</h3>"
            '<dl class="cockpit-objective">'
            f"<div><dt>Objective</dt><dd>{_escape(_metric_label(objective_metric))}</dd></div>"
            f"<div><dt>Result</dt><dd>{_escape(_metric_value(objective_metric, float(selected.metrics[objective_metric])))}</dd></div>"
            "</dl></div>"
            '<dl class="cockpit-deltas">'
            f"{_cockpit_delta_markup(recommendation, improvement=True, synthetic=benchmarks.synthetic)}"
            f"{_cockpit_delta_markup(recommendation, improvement=False, synthetic=benchmarks.synthetic)}"
            "</dl></section>"
        )

    source_copy = (
        "Three synthetic fixture decisions are already computed; the browser only switches views."
        if benchmarks.synthetic
        else "Three validated Arm64 decisions are already computed; the browser only switches views."
    )
    initial_status = (
        "Latency first selected. Primary fixture result shown."
        if benchmarks.synthetic
        else "Latency first selected. Canonical result shown."
    )
    return (
        '<section class="policy-cockpit" aria-labelledby="policy-cockpit-heading">'
        '<div class="cockpit-heading"><div>'
        '<p class="cockpit-call-sign">Precomputed policy cockpit</p>'
        '<h2 id="policy-cockpit-heading">Try ParetoPilot</h2></div>'
        f"<p>{_escape(source_copy)} No browser-side ranking or benchmark calculation occurs.</p>"
        "</div>"
        '<div class="cockpit-tabs" role="tablist" aria-label="Choose a deployment priority">'
        f"{''.join(tabs)}</div>"
        f'<p class="sr-only" data-cockpit-status aria-live="polite" aria-atomic="true">{_escape(initial_status)}</p>'
        f'<div class="cockpit-panels">{"".join(panels)}</div>'
        '<div class="cockpit-proof"><span aria-label="Decision pipeline">'
        "Verify provenance → Apply gates → Compute frontier → Select policy</span>"
        '<a href="evidence/optimization-receipt.md">Open this decision’s evidence receipt</a>'
        "</div>"
        '<noscript><p class="cockpit-noscript">JavaScript is unavailable, so the canonical '
        "latency result remains shown.</p></noscript>"
        "</section>\n"
    )


def _hero_markup(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
    proof: Mapping[str, str],
    *,
    canonical_report_href: str,
    policy_cockpit: str = "",
    capacity_available: bool = False,
) -> tuple[str, str, str]:
    source = _source_context(benchmarks)
    selected = benchmarks.by_id(str(recommendation["selected_id"]))
    selection = _mapping(recommendation.get("selection"), "recommendation.selection")
    objective = _mapping(recommendation.get("objective"), "recommendation.objective")
    shortlist = [str(item) for item in selection.get("shortlist_ids", ())]
    tolerance = float(selection.get("objective_tolerance_percent"))
    metric = str(objective.get("metric"))
    selected_value = float(selected.metrics[metric])

    candidate_kind = "synthetic fixture" if benchmarks.synthetic else "measured"
    provenance_items = [
        f"{'Canonical' if proof else 'Source'} run {source['run_id']}",
        source["cpu"],
        f"{source['cpu_count']} {source['architecture']} vCPUs",
        f"{len(benchmarks.candidates)} {candidate_kind} candidates",
    ]
    if proof:
        provenance_items.extend(
            (
                f"{proof['checksum_entries']} files verified",
                f"{proof['comparison_count']} authoritative comparisons",
            )
        )
    provenance = (
        '<aside class="provenance-strip" aria-label="Evidence provenance"><ul>'
        + "".join(f"<li>{_escape(item)}</li>" for item in provenance_items)
        + "</ul></aside>\n"
    )

    headline = (
        (
            "<h1>Explore the deployment decision that "
            '<span class="hero-selection">survives the fixture.</span></h1>\n'
        )
        if benchmarks.synthetic
        else (
            "<h1>Choose the Arm64 deployment that "
            '<span class="hero-selection">survives the evidence.</span></h1>\n'
        )
    )
    evidence_copy = (
        "This presentation view is derived from the byte-verified, locked v1.1 evidence."
        if proof
        else (
            "This is an unverified presentation preview; no release lock or byte-verified "
            "canonical report was supplied."
        )
    )
    study_verb = "evaluated"
    study_kind = "synthetic fixture" if benchmarks.synthetic else "Arm64"
    if not benchmarks.synthetic:
        study_verb = "measured"
    lede = (
        f'<p class="report-lede">ParetoPilot {study_verb} {len(benchmarks.candidates)} '
        f"{study_kind} configurations and retained {_escape(selected.label)} after quality, "
        f"Pareto-frontier, and {_escape(_metric_label(metric))} checks. "
        f"{_escape(evidence_copy)}</p>\n"
    )
    decision_rail = policy_cockpit or (
        '<dl class="decision-rail" aria-label="Decision at a glance">'
        f"<div><dt>Selected objective</dt><dd>{_escape(_metric_value(metric, selected_value))}</dd></div>"
        f"<div><dt>Inside cutoff</dt><dd>{len(shortlist)} of {len(benchmarks.candidates)}</dd></div>"
        f"<div><dt>Predeclared window</dt><dd>{tolerance:.2f}%</dd></div>"
        f"<div><dt>Evidence class</dt><dd>{'Locked canonical' if proof else 'Unverified preview'}</dd></div>"
        "</dl>\n"
    )
    actions = [
        (
            "#optimization-ladder",
            "Trace the optimization ladder",
            "secondary",
        ),
        (
            "https://github.com/agrovr/ParetoPilot/blob/main/docs/github-action.md",
            "Use the GitHub Action",
            "secondary",
        ),
        (
            "https://github.com/agrovr/ParetoPilot",
            "View source on GitHub",
            "secondary",
        ),
    ]
    if proof:
        actions[0:0] = (
            (
                canonical_report_href,
                "Open exact canonical report",
                "primary",
            ),
            (
                proof["release_url"],
                f"Open {proof['release_tag']} evidence release",
                "secondary",
            ),
        )
    action_markup = (
        '<nav class="hero-actions" aria-label="Project links">'
        + "".join(
            f'<a class="action-{kind}" href="{_escape(href)}">{_escape(label)}</a>'
            for href, label, kind in actions
        )
        + "</nav>\n"
    )
    section_links = [
        ("optimization-ladder", "00", "Optimize"),
        ("why-heading", "01", "Decision"),
        ("tradeoffs-heading", "02", "Tradeoffs"),
        ("policies-heading", "03", "Policies"),
        ("load-heading", "04", "Load"),
        ("repeat-heading", "05", "Repeat"),
        ("scatter-heading", "06", "Two-metric"),
        ("evidence-heading", "07", "Evidence"),
        ("trust-heading", "08", "Reproduce"),
    ]
    if capacity_available:
        section_links.insert(1, ("capacity-envelope", "S1", "Capacity"))
    flight_log = (
        '<nav class="flight-log" aria-label="Report sections">'
        '<span class="flight-log-label">Flight log</span><ol>'
        + "".join(
            f'<li><a href="#{heading_id}"><strong>{number}</strong>{label}</a></li>'
            for heading_id, number, label in section_links
        )
        + "</ol></nav>\n"
    )
    hero = (
        '<div class="hero-layout"><div class="hero-headline">'
        f"{headline}</div>"
        f'<div class="hero-proof">{lede}{decision_rail}{action_markup}</div></div>\n'
    )
    return provenance, hero, flight_log


_SHOWCASE_CSS = r"""

/* Judge-facing presentation layer. The canonical report remains byte-frozen. */
.showcase {
  --flight-ink: #13233d;
  --flight-ink-soft: #203653;
  --flight-panel: #13233d;
  --flight-cobalt: #2866d7;
  --flight-cobalt-solid: #2866d7;
  --flight-cobalt-soft: #e6eefc;
  --flight-teal: #116e6a;
  --flight-teal-soft: #dff3f1;
  --flight-amber: #8a5713;
  --flight-amber-soft: #f7ecd6;
  --flight-danger: #9f2d24;
  --flight-danger-soft: #f9e7e4;
  --flight-slate: #57708f;
  --flight-canvas: #f6f8fc;
  --flight-paper: #fffdf7;
  --flight-paper-blue: #f0f4fb;
  --flight-white: #ffffff;
  --flight-on-light: #13233d;
  --flight-on-dark: #f6f8fc;
  --flight-on-dark-muted: #c7d4e5;
  --flight-text-muted: #475f7c;
  --flight-text-subtle: #57708f;
  --flight-line: #cbd6e5;
  --flight-line-strong: #9fb0c4;
  --flight-panel-line: #40536e;
  --flight-control-border: #607491;
  --flight-focus: #2866d7;
  --flight-focus-inverse: #f0cf86;
  --flight-chart-axis: #758aa4;
  --flight-command-bg: #08111f;
  --flight-command-text: #f6f8fc;
  --flight-held-bg: #e7ebf1;
  --flight-purple: #6b4fa1;
  --flight-cyan: #246f91;
  --flight-hero-accent: #8edbd6;
  --flight-link-inverse: #9ac0ff;
  --flight-provenance-divider: #bcd0fa;
  --flight-primary-hover: #dce7f8;
  --bg: var(--flight-canvas);
  --surface: var(--flight-paper-blue);
  --surface-strong: var(--flight-held-bg);
  --ink: var(--flight-ink);
  --muted: var(--flight-text-muted);
  --line: var(--flight-line);
  --line-strong: var(--flight-line-strong);
  --accent: var(--flight-cobalt);
  --accent-dark: var(--flight-cobalt);
  --accent-soft: var(--flight-cobalt-soft);
  --success: var(--flight-teal);
  --success-soft: var(--flight-teal-soft);
  --warning: var(--flight-amber);
  --warning-soft: var(--flight-amber-soft);
  --danger: var(--flight-danger);
  --danger-soft: var(--flight-danger-soft);
  --focus: var(--flight-focus);
  background: var(--flight-canvas);
  color: var(--flight-ink);
  font-size: 1rem;
}
html[data-theme="light"] { color-scheme: light; }
html[data-theme="dark"] { color-scheme: dark; background: #0b1220; }
html[data-theme="dark"] .showcase {
  --flight-ink: #eef4ff;
  --flight-ink-soft: #1b2b44;
  --flight-panel: #0c1728;
  --flight-cobalt: #8db5ff;
  --flight-cobalt-solid: #2866d7;
  --flight-cobalt-soft: #172a49;
  --flight-teal: #54d2c7;
  --flight-teal-soft: #123a3a;
  --flight-amber: #f2bd68;
  --flight-amber-soft: #3a2b17;
  --flight-danger: #ff9a8b;
  --flight-danger-soft: #3a1f24;
  --flight-slate: #a8b8cd;
  --flight-canvas: #0b1220;
  --flight-paper: #111b2e;
  --flight-paper-blue: #17243a;
  --flight-text-muted: #b7c4d8;
  --flight-text-subtle: #a8b8cd;
  --flight-line: #33455f;
  --flight-line-strong: #526984;
  --flight-panel-line: #40536e;
  --flight-control-border: #607491;
  --flight-focus: #8db5ff;
  --flight-chart-axis: #91a4bd;
  --flight-command-bg: #050912;
  --flight-command-text: #e8f0fb;
  --flight-held-bg: #24334a;
  --flight-purple: #c4a7ff;
  --flight-cyan: #74c7ec;
  --flight-hero-accent: #8edbd6;
  --flight-link-inverse: #9ac0ff;
}
.showcase h1, .showcase h2, .showcase h3 {
  color: inherit;
  letter-spacing: -.025em;
}
.showcase p { max-width: 72ch; }
.showcase a { color: var(--flight-cobalt); text-underline-offset: .18em; }
.showcase .skip-link {
  background: var(--flight-panel);
  color: var(--flight-on-dark);
}
.showcase button:focus-visible,
.showcase a:focus-visible,
.showcase summary:focus-visible,
.showcase [tabindex]:focus-visible {
  outline: 3px solid var(--flight-focus);
  outline-offset: 4px;
}
.showcase .report-header button:focus-visible,
.showcase .report-header a:focus-visible,
.showcase .report-header [tabindex]:focus-visible,
.showcase .trust-section button:focus-visible,
.showcase .trust-section a:focus-visible,
.showcase .trust-section summary:focus-visible,
.showcase .trust-section [tabindex]:focus-visible,
.showcase .report-footer a:focus-visible {
  outline-color: var(--flight-focus-inverse);
}
.showcase .report-header {
  width: 100%;
  max-width: none;
  padding: 0 max(1rem, calc((100vw - 78rem) / 2)) 3.75rem;
  background: var(--flight-panel);
  color: var(--flight-on-dark);
}
.showcase .provenance-strip {
  margin-inline: min(-1rem, calc((78rem - 100vw) / 2));
  padding: .72rem max(1rem, calc((100vw - 78rem) / 2));
  background: var(--flight-cobalt-solid);
  color: var(--flight-white);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: .75rem;
  font-weight: 700;
  letter-spacing: .035em;
  text-transform: uppercase;
}
.showcase .provenance-strip ul {
  display: flex;
  flex-wrap: wrap;
  gap: .35rem 1.35rem;
  margin: 0;
  padding: 0;
  list-style: none;
}
.showcase .provenance-strip li + li::before {
  content: "/";
  margin-right: 1.35rem;
  color: var(--flight-provenance-divider);
}
.showcase .brand-line {
  margin: 2.1rem 0 3.35rem;
  padding-bottom: .9rem;
  border-color: var(--flight-panel-line);
}
.showcase .brand { color: var(--flight-white); font-size: 1.18rem; }
.showcase .source-type { color: var(--flight-on-dark-muted); }
.showcase .brand-controls {
  display: flex;
  flex-wrap: wrap;
  gap: .65rem 1rem;
  align-items: center;
  justify-content: flex-end;
}
.showcase .theme-toggle {
  display: inline-flex;
  min-height: 2.75rem;
  align-items: center;
  gap: .55rem;
  padding: .55rem .7rem;
  border: 1px solid var(--flight-control-border);
  border-radius: 0;
  background: transparent;
  color: var(--flight-on-dark);
  cursor: pointer;
  font: inherit;
  font-size: .82rem;
  font-weight: 760;
}
.showcase .theme-toggle:hover { background: var(--flight-ink-soft); }
.showcase .theme-toggle-state {
  min-width: 2.25rem;
  padding: .18rem .35rem;
  background: var(--flight-on-dark);
  color: var(--flight-on-light);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: .68rem;
  text-align: center;
  text-transform: uppercase;
}
.showcase .theme-toggle[aria-pressed="true"] .theme-toggle-state {
  background: var(--flight-hero-accent);
  color: var(--flight-on-light);
}
.showcase h1 {
  max-width: 13ch;
  font-size: clamp(3.25rem, 7vw, 6rem);
  line-height: .95;
  letter-spacing: -.035em;
}
.showcase .hero-layout { width: 100%; }
.showcase .hero-headline,
.showcase .hero-proof { min-width: 0; }
.showcase .hero-selection { color: var(--flight-hero-accent); }
.showcase .report-lede {
  max-width: 58ch;
  margin-top: 1.35rem;
  color: var(--flight-on-dark-muted);
  font-size: clamp(1.1rem, 2vw, 1.35rem);
}
.showcase .decision-rail {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0;
  margin: 2rem 0 0;
  border-block: 1px solid var(--flight-panel-line);
}
.showcase .decision-rail div { padding: .85rem 1rem .9rem 0; }
.showcase .decision-rail div:nth-child(even) {
  padding-left: 1rem;
  border-left: 1px solid var(--flight-panel-line);
}
.showcase .decision-rail dt {
  color: var(--flight-on-dark-muted);
  font-size: .75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .04em;
}
.showcase .decision-rail dd {
  margin: .22rem 0 0;
  color: var(--flight-white);
  font-size: 1.15rem;
  font-weight: 800;
  font-variant-numeric: tabular-nums;
}
.showcase .policy-cockpit {
  min-width: 0;
  margin-top: 1.35rem;
  border-block: 1px solid var(--flight-panel-line);
}
.showcase .cockpit-heading {
  display: grid;
  grid-template-columns: minmax(0, .72fr) minmax(0, 1.28fr);
  gap: .75rem;
  align-items: end;
  padding: .7rem 0 .75rem;
}
.showcase .cockpit-heading h2 {
  margin: .12rem 0 0;
  color: var(--flight-white);
  font-size: 1.12rem;
  line-height: 1.05;
}
.showcase .cockpit-heading > p {
  margin: 0;
  color: var(--flight-on-dark-muted);
  font-size: .76rem;
  line-height: 1.4;
}
.showcase .cockpit-call-sign {
  margin: 0;
  color: var(--flight-hero-accent);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: .66rem;
  font-weight: 800;
  letter-spacing: .045em;
  text-transform: uppercase;
}
.showcase .cockpit-tabs {
  display: grid;
  grid-auto-columns: minmax(7.25rem, 1fr);
  grid-auto-flow: column;
  max-width: 100%;
  overflow-x: auto;
  border-block: 1px solid var(--flight-panel-line);
  contain: inline-size paint;
  overscroll-behavior-inline: contain;
  scrollbar-width: none;
}
.showcase .cockpit-tabs::-webkit-scrollbar { display: none; }
.showcase .cockpit-tabs button {
  min-width: 0;
  min-height: 2.75rem;
  padding: .52rem .62rem;
  border: 0;
  border-right: 1px solid var(--flight-panel-line);
  border-radius: 0;
  background: transparent;
  color: var(--flight-on-dark);
  cursor: pointer;
  text-align: left;
}
.showcase .cockpit-tabs button:last-child { border-right: 0; }
.showcase .cockpit-tabs button:hover { background: var(--flight-ink-soft); }
.showcase .cockpit-tabs button strong,
.showcase .cockpit-tabs button span {
  display: block;
  min-width: 0;
  overflow-wrap: anywhere;
}
.showcase .cockpit-tabs button strong { font-size: .78rem; }
.showcase .cockpit-tabs button span {
  margin-top: .08rem;
  color: var(--flight-on-dark-muted);
  font-size: .63rem;
}
.showcase .cockpit-tabs button[aria-selected="true"] {
  background: var(--flight-hero-accent);
  color: var(--flight-on-light);
}
.showcase .cockpit-tabs button[aria-selected="true"] span {
  color: var(--flight-on-light);
}
.showcase .cockpit-tabs button:focus-visible {
  position: relative;
  z-index: 1;
  outline-color: var(--flight-focus-inverse);
  outline-offset: -4px;
}
.showcase .cockpit-panel {
  min-width: 0;
  padding: .8rem 0 .85rem;
}
.showcase .cockpit-panel[hidden] { display: none; }
.showcase .cockpit-decision {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.25fr);
  gap: .25rem 1rem;
  align-items: end;
}
.showcase .cockpit-classification {
  margin: 0;
  color: var(--flight-hero-accent);
  font-size: .67rem;
  font-weight: 800;
  letter-spacing: .035em;
  text-transform: uppercase;
}
.showcase .cockpit-decision h3 {
  grid-column: 1;
  margin: 0;
  color: var(--flight-white);
  font-size: clamp(1rem, 2vw, 1.35rem);
  line-height: 1.08;
  overflow-wrap: anywhere;
}
.showcase .cockpit-objective {
  display: grid;
  grid-column: 2;
  grid-row: 1 / span 2;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: .25rem .8rem;
  margin: 0;
}
.showcase .cockpit-objective div { min-width: 0; }
.showcase .cockpit-objective dt,
.showcase .cockpit-deltas dt {
  color: var(--flight-on-dark-muted);
  font-size: .62rem;
  font-weight: 700;
  letter-spacing: .035em;
  text-transform: uppercase;
}
.showcase .cockpit-objective dd {
  margin: .12rem 0 0;
  color: var(--flight-white);
  font-size: .78rem;
  font-weight: 780;
  font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere;
}
.showcase .cockpit-deltas {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin: .72rem 0 0;
  border-top: 1px solid var(--flight-panel-line);
}
.showcase .cockpit-deltas > div {
  min-width: 0;
  padding: .58rem .65rem 0 0;
}
.showcase .cockpit-deltas > div + div {
  padding-left: .65rem;
  border-left: 1px solid var(--flight-panel-line);
}
.showcase .cockpit-deltas dd { margin: .12rem 0 0; }
.showcase .cockpit-deltas strong,
.showcase .cockpit-deltas span {
  display: block;
  min-width: 0;
  overflow-wrap: anywhere;
}
.showcase .cockpit-deltas strong {
  color: var(--flight-white);
  font-size: .78rem;
  font-variant-numeric: tabular-nums;
}
.showcase .cockpit-deltas span {
  margin-top: .08rem;
  color: var(--flight-on-dark-muted);
  font-size: .68rem;
  line-height: 1.35;
}
.showcase .cockpit-deltas .is-improvement dt { color: var(--flight-hero-accent); }
.showcase .cockpit-deltas .is-tradeoff dt { color: var(--flight-focus-inverse); }
.showcase .cockpit-proof {
  display: flex;
  flex-wrap: wrap;
  gap: .35rem .9rem;
  align-items: baseline;
  justify-content: space-between;
  padding: .62rem 0 .1rem;
  border-top: 1px solid var(--flight-panel-line);
  font-size: .67rem;
}
.showcase .cockpit-proof span {
  color: var(--flight-on-dark-muted);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}
.showcase .cockpit-proof a {
  color: var(--flight-link-inverse);
  font-weight: 800;
}
.showcase .cockpit-noscript {
  margin: .65rem 0 0;
  color: var(--flight-on-dark-muted);
  font-size: .72rem;
}
.showcase .hero-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: .7rem 1.2rem;
  margin-top: 1.4rem;
}
.showcase .hero-actions a {
  display: inline-flex;
  min-height: 2.75rem;
  align-items: center;
  padding: .65rem .85rem;
  font-weight: 760;
}
.showcase .hero-actions .action-primary {
  background: var(--flight-white);
  color: var(--flight-on-light);
  text-decoration: none;
}
.showcase .hero-actions .action-primary:hover { background: var(--flight-primary-hover); }
.showcase .hero-actions .action-secondary { color: var(--flight-on-dark); }
.showcase .flight-log {
  display: grid;
  gap: .55rem;
  margin-top: 1.4rem;
  padding-top: 1rem;
  border-top: 1px solid var(--flight-panel-line);
}
.showcase .flight-log-label {
  color: var(--flight-on-dark-muted);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: .72rem;
  font-weight: 800;
  letter-spacing: .055em;
  text-transform: uppercase;
}
.showcase .flight-log ol {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0;
  margin: 0;
  padding: 0;
  list-style: none;
  border-block: 1px solid var(--flight-panel-line);
}
.showcase .flight-log li { min-width: 0; }
.showcase .flight-log a {
  display: flex;
  gap: .5rem;
  align-items: baseline;
  min-height: 2.65rem;
  padding: .65rem .55rem;
  color: var(--flight-on-dark);
  font-size: .82rem;
  font-weight: 700;
  text-decoration: none;
}
.showcase .flight-log a:hover { background: var(--flight-ink-soft); }
.showcase .flight-log strong {
  color: var(--flight-hero-accent);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: .7rem;
}
.showcase .verdict-layout {
  overflow: hidden;
  margin-top: 2.4rem;
  border: 0;
  background: var(--flight-paper);
  color: var(--flight-ink);
}
.showcase .verdict-column { padding: 1.45rem; }
.showcase .canonical-column {
  background: var(--flight-cobalt-solid);
  color: var(--flight-white);
}
.showcase .canonical-column .context-label,
.showcase .canonical-column .column-note,
.showcase .canonical-column dt { color: var(--flight-white); }
.showcase .alternative-column { background: var(--flight-paper); }
.showcase .report-main {
  width: 100%;
  max-width: none;
  margin: 0;
}
.showcase .optimization-ladder {
  width: 100%;
  padding: clamp(3.6rem, 7vw, 6.4rem) 0;
  border-block: 1px solid var(--flight-panel-line);
  background: var(--flight-panel);
  color: var(--flight-on-dark);
}
.showcase .optimization-ladder-inner {
  width: min(calc(100% - 2rem), 78rem);
  margin-inline: auto;
}
.showcase .optimization-ladder-heading {
  display: grid;
  gap: .8rem 2.5rem;
  align-items: end;
  margin-bottom: 2.2rem;
}
.showcase .optimization-ladder .section-kicker {
  color: var(--flight-hero-accent);
}
.showcase .optimization-ladder-heading h2 {
  max-width: 15ch;
  color: var(--flight-white);
  font-size: clamp(2.2rem, 5vw, 3.8rem);
  line-height: .98;
}
.showcase .ladder-intro-copy > p {
  max-width: 62ch;
  margin: 0;
  color: var(--flight-on-dark-muted);
  font-size: 1.03rem;
}
.showcase .ladder-intro-copy > .ladder-evidence-grade {
  display: flex;
  flex-wrap: wrap;
  gap: .35rem .65rem;
  align-items: baseline;
  margin-top: .85rem;
  font-size: .76rem;
}
.showcase .ladder-evidence-grade span {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: .7rem;
  font-weight: 800;
  letter-spacing: .04em;
  text-transform: uppercase;
}
.showcase .ladder-evidence-grade strong {
  color: var(--flight-hero-accent);
  font-weight: 780;
}
.showcase .ladder-runway {
  margin-bottom: 2.6rem;
  border-block: 1px solid var(--flight-panel-line);
}
.showcase .runway-call-sign {
  margin: 0;
  padding: .72rem 0;
  color: var(--flight-hero-accent);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: .72rem;
  font-weight: 800;
  letter-spacing: .055em;
  text-transform: uppercase;
}
.showcase .ladder-runway dl {
  display: grid;
  margin: 0;
  border-top: 1px solid var(--flight-panel-line);
}
.showcase .ladder-runway dl > div {
  min-width: 0;
  padding: 1rem 0;
}
.showcase .ladder-runway dl > div + div {
  border-top: 1px solid var(--flight-panel-line);
}
.showcase .ladder-runway dt {
  color: var(--flight-on-dark-muted);
  font-size: .75rem;
  font-weight: 700;
  letter-spacing: .035em;
  text-transform: uppercase;
}
.showcase .ladder-runway dd {
  margin: .3rem 0 0;
  color: var(--flight-white);
  font-size: 1.02rem;
  font-weight: 780;
}
.showcase .ladder-runway dd span {
  display: block;
  margin-top: .3rem;
  color: var(--flight-on-dark-muted);
  font-size: .82rem;
  font-weight: 600;
}
.showcase .optimization-stages {
  position: relative;
  display: grid;
  gap: 2rem;
  margin: 0;
  padding: 0;
  list-style: none;
}
.showcase .optimization-stages::before {
  position: absolute;
  z-index: 0;
  top: 1.5rem;
  bottom: 1.5rem;
  left: 1.45rem;
  border-left: 2px dashed var(--flight-control-border);
  content: "";
}
.showcase .optimization-stage {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: 3rem minmax(0, 1fr);
  gap: 1rem;
  min-width: 0;
}
.showcase .stage-marker {
  display: grid;
  width: 3rem;
  height: 3rem;
  place-items: center;
  border: 2px solid var(--flight-control-border);
  background: var(--flight-panel);
  color: var(--flight-on-dark-muted);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: .75rem;
  font-weight: 800;
  font-variant-numeric: tabular-nums;
}
.showcase .optimization-stage.is-selected .stage-marker {
  border-color: var(--flight-hero-accent);
  background: var(--flight-hero-accent);
  color: var(--flight-on-light);
}
.showcase .optimization-stage.is-closest .stage-marker {
  border-color: var(--flight-focus-inverse);
  color: var(--flight-focus-inverse);
}
.showcase .stage-body {
  min-width: 0;
  padding-bottom: .2rem;
}
.showcase .stage-role {
  margin: 0 0 .35rem;
  color: var(--flight-hero-accent);
  font-size: .78rem;
  font-weight: 780;
}
.showcase .stage-body h3 {
  max-width: 20ch;
  margin: 0;
  color: var(--flight-white);
  font-size: clamp(1.18rem, 2vw, 1.45rem);
  line-height: 1.08;
}
.showcase .stage-id {
  display: block;
  margin-top: .45rem;
  color: var(--flight-on-dark-muted);
  font-size: .72rem;
  overflow-wrap: anywhere;
}
.showcase .stage-decision-label {
  display: inline-block;
  margin-top: .75rem;
  padding: .3rem .45rem;
  border: 1px solid var(--flight-control-border);
  color: var(--flight-on-dark-muted);
  font-size: .72rem;
  font-weight: 760;
}
.showcase .is-selected .stage-decision-label {
  border-color: var(--flight-hero-accent);
  background: var(--flight-hero-accent);
  color: var(--flight-on-light);
}
.showcase .is-closest .stage-decision-label {
  border-color: var(--flight-focus-inverse);
  color: var(--flight-focus-inverse);
}
.showcase .stage-objective {
  display: grid;
  gap: .2rem;
  margin: 1rem 0 0;
  padding-block: .75rem;
  border-block: 1px solid var(--flight-panel-line);
}
.showcase .stage-objective span {
  color: var(--flight-on-dark-muted);
  font-size: .75rem;
}
.showcase .stage-objective strong {
  color: var(--flight-white);
  font-size: 1.05rem;
  font-variant-numeric: tabular-nums;
}
.showcase .stage-changes {
  margin: 0;
  padding: 0;
  list-style: none;
}
.showcase .stage-changes li {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: .22rem .6rem;
  padding: .68rem 0;
  border-bottom: 1px solid var(--flight-panel-line);
}
.showcase .stage-changes span {
  color: var(--flight-on-dark-muted);
  font-size: .76rem;
}
.showcase .stage-changes strong {
  color: var(--flight-white);
  font-size: .82rem;
  font-variant-numeric: tabular-nums;
}
.showcase .stage-changes em {
  grid-column: 1 / -1;
  color: var(--flight-on-dark-muted);
  font-size: .7rem;
  font-style: normal;
}
.showcase .stage-changes .is-improved em { color: var(--flight-hero-accent); }
.showcase .stage-changes .is-tradeoff em { color: var(--flight-focus-inverse); }
.showcase .stage-reference-note {
  margin: .8rem 0 0;
  color: var(--flight-on-dark-muted);
  font-size: .8rem;
}
.showcase .optimization-ladder-caveat {
  margin: 2.25rem 0 0;
  padding-top: 1rem;
  border-top: 1px solid var(--flight-panel-line);
  color: var(--flight-on-dark-muted);
  font-size: .85rem;
}
.showcase .optimization-ladder-caveat strong { color: var(--flight-white); }
.showcase .capacity-envelope {
  width: 100%;
  padding: clamp(3.8rem, 7vw, 6.5rem) 0;
  border-bottom: 1px solid var(--flight-line-strong);
  background: var(--flight-canvas);
  color: var(--flight-ink);
}
.showcase .capacity-envelope-inner {
  width: min(calc(100% - 2rem), 78rem);
  margin-inline: auto;
}
.showcase .capacity-heading {
  display: grid;
  gap: 1rem 2.5rem;
  align-items: end;
}
.showcase .capacity-heading h2 {
  max-width: 13ch;
  font-size: clamp(2.3rem, 5.8vw, 4.5rem);
  line-height: .92;
}
.showcase .capacity-heading > div:last-child > p {
  margin: 0;
  color: var(--flight-text-muted);
  font-size: 1.03rem;
}
.showcase .capacity-heading .capacity-compare {
  margin-top: 1rem;
  padding-left: .9rem;
  border-left: 4px solid var(--flight-teal);
}
.showcase .capacity-compare strong { color: var(--flight-ink); }
.showcase .capacity-facts {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin: 2.4rem 0;
  border-block: 2px solid var(--flight-ink);
}
.showcase .capacity-facts > div {
  min-width: 0;
  padding: .9rem .75rem;
}
.showcase .capacity-facts > div:nth-child(even) {
  border-left: 1px solid var(--flight-line-strong);
}
.showcase .capacity-facts > div:nth-child(n + 3) {
  border-top: 1px solid var(--flight-line-strong);
}
.showcase .capacity-facts dt,
.showcase .capacity-selected-summary dt,
.showcase .capacity-slo-strip dt {
  color: var(--flight-text-subtle);
  font-size: .67rem;
  font-weight: 800;
  letter-spacing: .04em;
  text-transform: uppercase;
}
.showcase .capacity-facts dd {
  margin: .2rem 0 0;
  color: var(--flight-ink);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: clamp(1.25rem, 3vw, 1.75rem);
  font-weight: 850;
  font-variant-numeric: tabular-nums;
}
.showcase .capacity-boards {
  display: grid;
  gap: 2rem;
}
.showcase .capacity-board {
  --capacity-accent: var(--flight-cobalt);
  --capacity-accent-soft: var(--flight-cobalt-soft);
  min-width: 0;
  border: 2px solid var(--flight-ink);
  border-top: 8px solid var(--capacity-accent);
  background: var(--flight-paper);
  box-shadow: .55rem .55rem 0 var(--capacity-accent-soft);
}
.showcase .capacity-board.is-q4 {
  --capacity-accent: var(--flight-teal);
  --capacity-accent-soft: var(--flight-teal-soft);
}
.showcase .capacity-board-heading {
  padding: 1rem;
  border-bottom: 1px solid var(--flight-line-strong);
}
.showcase .capacity-board-heading > p {
  margin: 0 0 .35rem;
  color: var(--capacity-accent);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: .7rem;
  font-weight: 850;
  letter-spacing: .055em;
  text-transform: uppercase;
}
.showcase .capacity-board-heading h3 {
  max-width: 24ch;
  margin: 0;
  font-size: clamp(1.2rem, 2.7vw, 1.65rem);
  line-height: 1.05;
}
.showcase .capacity-selected-summary {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: .65rem 1rem;
  margin: 1rem 0 0;
}
.showcase .capacity-selected-summary dd,
.showcase .capacity-slo-strip dd {
  margin: .18rem 0 0;
  color: var(--flight-ink);
  font-size: .86rem;
  font-weight: 780;
  font-variant-numeric: tabular-nums;
}
.showcase .capacity-table-wrap {
  width: 100%;
  padding: .75rem;
}
.showcase .capacity-matrix {
  width: 100%;
  table-layout: fixed;
  border-collapse: separate;
  border-spacing: .25rem;
  font-variant-numeric: tabular-nums;
}
.showcase .capacity-matrix caption {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
  border: 0;
}
.showcase .capacity-matrix th,
.showcase .capacity-matrix td {
  border: 0;
  text-align: left;
}
.showcase .capacity-matrix thead th {
  padding: .35rem .2rem;
  background: transparent;
  color: var(--flight-text-subtle);
  font-size: .72rem;
  text-align: center;
}
.showcase .capacity-matrix thead th:first-child { width: 2.65rem; }
.showcase .capacity-matrix thead th span {
  display: block;
  font-size: .56rem;
  font-weight: 650;
}
.showcase .capacity-matrix tbody th {
  padding: .45rem .1rem;
  background: transparent;
  color: var(--flight-text-subtle);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: .72rem;
  text-align: center;
  vertical-align: middle;
}
.showcase .capacity-matrix tbody th span {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
  border: 0;
}
.showcase .capacity-cell {
  position: relative;
  min-width: 0;
  padding: clamp(.35rem, 1.3vw, .68rem);
  border: 1px solid var(--flight-line-strong) !important;
  background: var(--flight-paper-blue);
  vertical-align: top;
}
.showcase .capacity-cell.is-blocked {
  border-color: var(--flight-danger) !important;
  background: var(--flight-danger-soft);
}
.showcase .capacity-cell.is-selected {
  border: 3px solid var(--capacity-accent) !important;
  background: var(--capacity-accent-soft);
  box-shadow: inset 0 -.32rem 0 var(--capacity-accent);
}
.showcase .capacity-cell-top {
  display: flex;
  flex-wrap: wrap;
  gap: .2rem .35rem;
  align-items: baseline;
}
.showcase .capacity-state,
.showcase .capacity-reference,
.showcase .capacity-failure {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: clamp(.55rem, 1.45vw, .68rem);
  font-weight: 850;
  letter-spacing: .02em;
  text-transform: uppercase;
}
.showcase .capacity-state { color: var(--flight-text-subtle); }
.showcase .is-blocked .capacity-state,
.showcase .capacity-failure { color: var(--flight-danger); }
.showcase .is-selected .capacity-state { color: var(--capacity-accent); }
.showcase .capacity-reference {
  color: var(--flight-cobalt);
  font-size: .56rem;
}
.showcase .capacity-rate {
  display: flex;
  flex-wrap: wrap;
  gap: .05rem .28rem;
  align-items: baseline;
  margin: .45rem 0;
}
.showcase .capacity-rate strong {
  color: var(--flight-ink);
  font-size: clamp(.9rem, 2.5vw, 1.22rem);
  line-height: 1;
}
.showcase .capacity-rate span {
  color: var(--flight-text-subtle);
  font-size: clamp(.54rem, 1.35vw, .66rem);
  font-weight: 700;
}
.showcase .capacity-cell-metrics {
  margin: 0;
  color: var(--flight-text-subtle);
  font-size: clamp(.56rem, 1.35vw, .68rem);
}
.showcase .capacity-cell-metrics > div {
  display: flex;
  flex-wrap: wrap;
  gap: .15rem .3rem;
  justify-content: space-between;
}
.showcase .capacity-cell-metrics dt { font-weight: 750; }
.showcase .capacity-cell-metrics dd {
  margin: 0;
  color: var(--flight-ink);
  font-weight: 680;
}
.showcase .capacity-failure {
  display: block;
  margin-top: .4rem;
  overflow-wrap: anywhere;
}
.showcase .capacity-blocked-details {
  margin: 0 1rem 1rem;
  padding-top: .75rem;
  border-top: 1px solid var(--flight-line-strong);
}
.showcase .capacity-blocked-details summary {
  color: var(--capacity-accent);
  font-size: .82rem;
  font-weight: 780;
  cursor: pointer;
}
.showcase .capacity-blocked-details ul {
  display: grid;
  gap: .45rem;
  margin: .7rem 0 0;
  padding: 0;
  list-style: none;
}
.showcase .capacity-blocked-details li {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: .55rem;
  color: var(--flight-text-muted);
  font-size: .75rem;
}
.showcase .capacity-blocked-details li strong {
  color: var(--flight-danger);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}
.showcase .capacity-slo-strip {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin: 2.5rem 0 0;
  border-block: 1px solid var(--flight-line-strong);
}
.showcase .capacity-slo-strip > div { padding: .8rem .75rem; }
.showcase .capacity-slo-strip > div:nth-child(even) {
  border-left: 1px solid var(--flight-line-strong);
}
.showcase .capacity-slo-strip > div:nth-child(n + 3) {
  border-top: 1px solid var(--flight-line-strong);
}
.showcase .capacity-actions {
  display: flex;
  flex-wrap: wrap;
  gap: .65rem 1.25rem;
  margin-top: 1.4rem;
}
.showcase .capacity-actions a {
  font-size: .82rem;
  font-weight: 780;
}
.showcase .capacity-proof-line,
.showcase .capacity-boundary {
  max-width: 88ch;
  color: var(--flight-text-muted);
  font-size: .78rem;
}
.showcase .capacity-proof-line { margin: 1rem 0 0; }
.showcase .capacity-boundary {
  margin: .75rem 0 0;
  padding-top: .75rem;
  border-top: 1px solid var(--flight-line);
}
.showcase .capacity-proof-line strong,
.showcase .capacity-boundary strong { color: var(--flight-ink); }
.showcase .capacity-proof-line code {
  color: var(--flight-ink);
  overflow-wrap: anywhere;
}
.showcase .report-section {
  width: min(calc(100% - 2rem), 78rem);
  margin-inline: auto;
  padding: clamp(3.4rem, 6vw, 5.4rem) 0;
  border-color: var(--flight-line);
}
.showcase .section-heading {
  gap: .65rem 2.5rem;
  margin-bottom: 2rem;
}
.showcase .section-title { min-width: 0; }
.showcase .section-kicker {
  margin: 0 0 .55rem;
  color: var(--flight-cobalt);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: .75rem;
  font-weight: 800;
  letter-spacing: .055em;
  text-transform: uppercase;
}
.showcase .section-heading h2 {
  max-width: 18ch;
  font-size: clamp(1.9rem, 4vw, 2.75rem);
  line-height: 1.02;
}
.showcase .section-heading p {
  max-width: 62ch;
  color: var(--flight-text-muted);
  font-size: 1.03rem;
}
.showcase .visual-kicker {
  margin: 0 0 .3rem;
  color: var(--flight-cobalt);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: .76rem;
  font-weight: 800;
  letter-spacing: .04em;
  text-transform: uppercase;
}
.showcase .tolerance-visual {
  margin: 0 0 2.4rem;
  padding: 1.35rem 0 1.2rem;
  border-block: 2px solid var(--flight-ink);
}
.showcase .tolerance-visual-heading {
  display: grid;
  gap: .75rem 2rem;
  align-items: end;
}
.showcase .tolerance-visual-heading h3 {
  margin: 0;
  font-size: clamp(1.55rem, 3vw, 2.15rem);
}
.showcase .tolerance-visual-heading p {
  margin: 0;
  color: var(--flight-text-muted);
}
.showcase .tolerance-direction {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  margin: 1.25rem 0 .55rem;
  color: var(--flight-text-subtle);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: .72rem;
  font-weight: 700;
  text-transform: uppercase;
}
.showcase .tolerance-list { margin: 0; padding: 0; list-style: none; }
.showcase .tolerance-row {
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(0, 2fr) minmax(0, .65fr);
  gap: 1rem;
  align-items: center;
  min-height: 4.15rem;
  padding: .75rem 0;
  border-top: 1px solid var(--flight-line);
}
.showcase .tolerance-row.is-selected {
  margin-inline: -.75rem;
  padding-inline: .75rem;
  background: var(--flight-cobalt-soft);
}
.showcase .tolerance-name { min-width: 0; }
.showcase .tolerance-name strong,
.showcase .tolerance-name span { display: block; }
.showcase .tolerance-name strong { overflow-wrap: anywhere; }
.showcase .tolerance-name span {
  margin-top: .12rem;
  color: var(--flight-text-muted);
  font-size: .78rem;
}
.showcase .tolerance-scale {
  position: relative;
  height: 1.35rem;
}
.showcase .tolerance-scale::before {
  content: "";
  position: absolute;
  inset: calc(50% - 1px) 0 auto;
  height: 2px;
  background: var(--flight-line-strong);
}
.showcase .tolerance-cutoff {
  position: absolute;
  left: var(--position);
  top: 0;
  width: 2px;
  height: 100%;
  background: var(--flight-amber);
}
.showcase .tolerance-marker {
  position: absolute;
  left: var(--position);
  top: 50%;
  width: .9rem;
  height: .9rem;
  border: 3px solid var(--flight-paper);
  border-radius: 50%;
  background: var(--flight-slate);
  transform: translate(-50%, -50%);
}
.showcase .is-inside .tolerance-marker { background: var(--flight-cobalt); }
.showcase .is-selected .tolerance-marker {
  width: 1.15rem;
  height: 1.15rem;
  border-color: var(--flight-white);
  background: var(--flight-cobalt);
}
.showcase .tolerance-value {
  justify-self: end;
  font-weight: 800;
  font-variant-numeric: tabular-nums;
}
.showcase .tolerance-visual figcaption {
  margin-top: .8rem;
  color: var(--flight-text-subtle);
  font-size: .83rem;
}
.showcase .why-layout { gap: 2rem; }
.showcase .reason-block,
.showcase .no-data {
  border-color: var(--flight-line-strong);
  border-radius: 0;
  background: var(--flight-paper-blue);
}
.showcase .reason-number {
  color: var(--flight-cobalt);
  font-size: clamp(2.6rem, 5vw, 4.5rem);
  line-height: 1;
}
.showcase table { font-size: .92rem; }
.showcase table.sr-only {
  width: 1px !important;
  min-width: 0 !important;
  max-width: 1px !important;
  table-layout: fixed;
  contain: strict;
}
.showcase caption { color: var(--flight-text-subtle); }
.showcase th, .showcase td {
  padding: .82rem .72rem;
  border-color: var(--flight-line);
  word-break: normal;
}
.showcase th { overflow-wrap: normal; }
.showcase td { overflow-wrap: anywhere; }
.showcase thead th {
  background: var(--flight-panel);
  color: var(--flight-white);
}
.showcase tbody tr:nth-child(even) { background: var(--flight-paper-blue); }
.showcase .table-scroll,
.showcase .command {
  width: 100%;
  max-width: 100%;
  min-width: 0;
  contain: inline-size;
}
.showcase .table-scroll { overflow-x: auto; }
.showcase .load-table,
.showcase .stability-table {
  width: 72rem;
  max-width: none;
  table-layout: fixed;
}
.showcase .candidate-table {
  width: 92rem;
  max-width: none;
  table-layout: auto;
}
.showcase .candidate-table caption {
  width: min(calc(100vw - 2rem), 78rem);
  max-width: calc(100vw - 2rem);
  white-space: normal;
}
.showcase .candidate-table th:first-child { width: 12rem; }
.showcase .candidate-table th:nth-child(3) { width: 10rem; }
.showcase .metadata-table {
  width: 100%;
  min-width: 48rem;
  table-layout: fixed;
}
.showcase .metadata-table th:first-child { width: 13rem; }
.showcase .metadata-table code {
  display: block;
  width: 100%;
  max-width: 100%;
  max-height: 18rem;
  padding: .75rem .85rem;
  overflow: auto;
  border: 1px solid var(--flight-panel-line);
  background: var(--flight-command-bg);
  color: var(--flight-command-text);
  line-height: 1.55;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: normal;
}
.showcase .table-scroll-hint {
  display: none;
  margin: .2rem 0 .65rem;
  color: var(--flight-text-muted);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: .7rem;
  font-weight: 800;
  letter-spacing: .04em;
  text-transform: uppercase;
}
.showcase .data-disclosure {
  margin-top: 1.3rem;
  border-block: 1px solid var(--flight-line-strong);
}
.showcase details:not([open]) > :not(summary) { display: none; }
.showcase .data-disclosure > summary {
  width: 100%;
  padding: .85rem 0;
  color: var(--flight-cobalt);
  font-weight: 800;
}
.showcase .data-disclosure[open] > summary {
  margin: 0 0 .85rem;
  border-bottom: 1px solid var(--flight-line);
}
.showcase .tradeoff-board {
  border-color: var(--flight-ink);
  border-width: 2px 0;
}
.showcase .tradeoff-row { border-color: var(--flight-line); }
.showcase .tradeoff-metric { font-size: 1rem; }
.showcase .tradeoff-value {
  font-size: 1.02rem;
  font-weight: 700;
}
.showcase .value-label { color: var(--flight-text-subtle); }
.showcase .effect-better {
  border-radius: 3px;
  background: var(--flight-teal-soft);
  color: var(--flight-teal);
}
.showcase .effect-tradeoff {
  border-radius: 3px;
  background: var(--flight-amber-soft);
  color: var(--flight-amber);
}
.showcase .effect-held {
  border-radius: 3px;
  background: var(--flight-held-bg);
  color: var(--flight-text-muted);
}
.showcase .profile-tabs {
  display: grid;
  grid-auto-columns: minmax(10.5rem, 1fr);
  grid-auto-flow: column;
  gap: 0;
  padding: 0;
  max-width: 100%;
  overflow-x: clip;
  overflow-y: hidden;
  border: 2px solid var(--flight-ink);
  contain: inline-size paint;
  overscroll-behavior-inline: contain;
  scrollbar-width: none;
}
.showcase .profile-tabs::-webkit-scrollbar {
  width: 0;
  height: 0;
  display: none;
}
.showcase .profile-tabs button {
  min-height: 4.15rem;
  padding: .75rem .9rem;
  border: 0;
  border-right: 1px solid var(--flight-line-strong);
  border-radius: 0;
  background: var(--flight-paper);
}
.showcase .profile-tabs button:last-child { border-right: 0; }
.showcase .profile-tabs button:hover { background: var(--flight-cobalt-soft); }
.showcase .profile-tabs button span { color: var(--flight-text-subtle); }
.showcase .profile-tabs button[aria-selected="true"] {
  background: var(--flight-panel);
  color: var(--flight-white);
}
.showcase .profile-tabs button[aria-selected="true"] span { color: #c7d4e5; }
.showcase .profile-tabs button:focus-visible {
  position: relative;
  z-index: 1;
  outline-offset: -4px;
}
.showcase .profile-tabs button[aria-selected="true"]:focus-visible {
  outline-color: var(--flight-focus-inverse);
}
.showcase .profile-tabs[data-overflow="scroll"] { overflow-x: auto; }
.showcase .profile-panel {
  padding-top: 2rem;
  border-bottom: 2px solid var(--flight-ink);
}
.showcase .profile-metrics {
  min-width: 0;
  border-color: var(--flight-line-strong);
}
.showcase .profile-metrics li { border-color: var(--flight-line); }
.showcase .profile-metrics span,
.showcase .profile-metrics strong {
  min-width: 0;
  overflow-wrap: anywhere;
}
.showcase .profile-metrics span { color: var(--flight-text-subtle); }
.showcase .series-key-wrap {
  margin: 1.2rem 0;
  padding: .9rem 0;
  border-block: 1px solid var(--flight-line-strong);
}
.showcase .series-key-wrap > p {
  margin: 0 0 .65rem;
  color: var(--flight-text-subtle);
  font-size: .8rem;
  font-weight: 760;
}
.showcase .series-key {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
  gap: .55rem 1.25rem;
  margin: 0;
  padding: 0;
  list-style: none;
}
.showcase .series-key li {
  display: grid;
  grid-template-columns: 2.6rem minmax(0, 1fr);
  gap: .15rem .65rem;
  align-items: center;
}
.showcase .series-swatch {
  display: block;
  width: 2.6rem;
  height: 1rem;
  overflow: visible;
}
.showcase .series-swatch .legend-line {
  stroke: var(--series-color);
  stroke-width: 3;
  stroke-dasharray: var(--series-dash);
}
.showcase .series-swatch .legend-marker {
  fill: var(--series-color);
  stroke: var(--flight-paper);
  stroke-width: 1.5;
}
.showcase .series-swatch .legend-marker-selected {
  fill: var(--flight-paper);
  stroke: var(--series-color);
  stroke-width: 2.5;
}
.showcase .series-name { min-width: 0; font-weight: 720; overflow-wrap: anywhere; }
.showcase .series-role {
  grid-column: 2;
  color: var(--flight-text-subtle);
  font-size: .75rem;
}
.showcase [data-series-style="0"] {
  --series-color: var(--flight-cobalt);
  --series-dash: none;
  --series-line-style: solid;
}
.showcase [data-series-style="1"] {
  --series-color: var(--flight-slate);
  --series-dash: 7 4;
  --series-line-style: dashed;
}
.showcase [data-series-style="2"] {
  --series-color: var(--flight-teal);
  --series-dash: 2 3;
  --series-line-style: dotted;
}
.showcase [data-series-style="3"] {
  --series-color: var(--flight-amber);
  --series-dash: 10 4 2 4;
  --series-line-style: double;
}
.showcase [data-series-style="4"] {
  --series-color: var(--flight-purple);
  --series-dash: 5 4;
  --series-line-style: dashed;
}
.showcase [data-series-style="5"] {
  --series-color: var(--flight-cyan);
  --series-dash: 11 4;
  --series-line-style: solid;
}
.showcase .load-context-grid {
  grid-template-columns: minmax(0, 1fr);
  gap: 1.6rem;
}
.showcase .load-context {
  min-width: 0;
  padding-top: 1rem;
  border-top: 2px solid var(--flight-ink);
}
.showcase .load-context:not(.load-binding) .compact-table {
  width: 100%;
  min-width: 0;
  table-layout: fixed;
}
.showcase .load-context:not(.load-binding) .compact-table th:first-child {
  width: 64%;
}
.showcase .load-binding .compact-table {
  width: 64rem;
  min-width: 64rem;
  table-layout: fixed;
}
.showcase .chart-grid-layout {
  grid-template-columns: minmax(0, 1fr);
  gap: 1.5rem;
}
.showcase .chart-figure {
  max-width: 100%;
  padding: 1rem;
  overflow-x: auto;
  overflow-y: hidden;
  overscroll-behavior-inline: contain;
  contain: inline-size paint;
  border: 1px solid var(--flight-ink);
  background: var(--flight-paper);
}
.showcase .chart-scroll-hint {
  display: none;
  margin: 0 0 .75rem;
  color: var(--flight-text-muted);
  font-size: .74rem;
  font-weight: 760;
  letter-spacing: .04em;
  text-transform: uppercase;
}
.showcase .command {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.showcase .chart-figure svg {
  width: 100%;
  min-width: 36rem;
  height: auto;
  overflow: visible;
}
.showcase .scatter-figure svg {
  display: block;
  width: min(100%, 40rem);
  margin-inline: auto;
}
.showcase [data-series-style] .chart-line {
  fill: none;
  stroke: var(--series-color);
  stroke-dasharray: var(--series-dash);
}
.showcase [data-series-style] .label-leader,
.showcase [data-series-style] .scatter-leader {
  fill: none;
  stroke: var(--series-color);
}
.showcase [data-series-style] .chart-marker {
  fill: var(--series-color);
  stroke: var(--flight-paper);
  stroke-dasharray: none;
}
.showcase [data-series-style="0"] .chart-marker {
  fill: var(--flight-paper);
  stroke: var(--series-color);
  stroke-width: 4;
}
.showcase .direct-label,
.showcase .label-leader,
.showcase .scatter-leader { display: none; }
.showcase .chart-axis { stroke: var(--flight-chart-axis); }
.showcase .chart-grid { stroke: var(--flight-line); }
.showcase .chart-tick,
.showcase .chart-axis-label {
  fill: var(--flight-text-muted);
  font-size: 13px;
}
.showcase .chart-axis-label { font-size: 14px; font-weight: 700; }
.showcase .slo-reference-line {
  stroke: var(--flight-amber);
  stroke-width: 2;
  stroke-dasharray: 8 5;
}
.showcase .slo-reference-label {
  fill: var(--flight-amber);
  font-size: 13px;
  font-weight: 800;
  paint-order: stroke;
  stroke: var(--flight-paper);
  stroke-width: 4px;
}
.showcase .chart-tick[x="66"][y="34"] {
  text-anchor: start;
  transform: translateX(14px);
  font-weight: 750;
  paint-order: stroke;
  stroke: var(--flight-paper);
  stroke-width: 4px;
}
.showcase .scatter-figure .chart-tick[x="68"] {
  text-anchor: end;
}
.showcase .chart-figure figcaption {
  color: var(--flight-text-subtle);
  font-size: .84rem;
}
.showcase .stability-explainer {
  margin: .8rem 0 0;
  padding: .8rem 0;
  border-block: 1px solid var(--flight-line-strong);
  color: var(--flight-text-muted);
}
.showcase .evidence-limit {
  border-color: var(--flight-line-strong);
  color: var(--flight-text-muted);
}
.showcase .stability-method { color: var(--flight-text-muted); }
.showcase summary { color: var(--flight-cobalt); }
.showcase .trust-section {
  padding: clamp(2rem, 4vw, 3.25rem);
  background: var(--flight-panel);
  color: var(--flight-on-dark);
}
.showcase .trust-section .section-heading p,
.showcase .trust-section .hash-list dt,
.showcase .trust-section .column-note { color: var(--flight-on-dark-muted); }
.showcase .trust-section .hash-list > div,
.showcase .trust-section .reproduction-note {
  border-color: var(--flight-panel-line);
}
.showcase .trust-section summary,
.showcase .trust-section a { color: var(--flight-link-inverse); }
.showcase .trust-section table { color: var(--flight-on-dark); }
.showcase .trust-section caption { color: var(--flight-on-dark-muted); }
.showcase .trust-section th,
.showcase .trust-section td { border-color: var(--flight-panel-line); }
.showcase .trust-section tbody tr:nth-child(even) {
  background: var(--flight-ink-soft);
}
.showcase .command {
  border-radius: 0;
  background: var(--flight-command-bg);
  color: var(--flight-command-text);
}
.showcase .report-footer {
  width: 100%;
  max-width: none;
  padding: 1.6rem max(1rem, calc((100vw - 78rem) / 2)) 2.2rem;
  border: 0;
  background: var(--flight-panel);
  color: var(--flight-on-dark-muted);
}
@media (min-width: 48rem) {
  .showcase .optimization-ladder-heading {
    grid-template-columns: minmax(0, .8fr) minmax(0, 1.2fr);
  }
  .showcase .capacity-heading {
    grid-template-columns: minmax(0, .82fr) minmax(0, 1.18fr);
  }
  .showcase .capacity-facts,
  .showcase .capacity-slo-strip {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }
  .showcase .capacity-facts > div + div,
  .showcase .capacity-slo-strip > div + div {
    border-top: 0;
    border-left: 1px solid var(--flight-line-strong);
  }
  .showcase .ladder-runway dl {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
  .showcase .ladder-runway dl > div {
    padding: 1rem;
  }
  .showcase .ladder-runway dl > div:first-child { padding-left: 0; }
  .showcase .ladder-runway dl > div + div {
    border-top: 0;
    border-left: 1px solid var(--flight-panel-line);
  }
  .showcase .optimization-stages {
    grid-template-columns: repeat(var(--stage-count, 4), minmax(0, 1fr));
    gap: 1.25rem;
  }
  .showcase .optimization-stages::before {
    top: 1.45rem;
    right: 1.5rem;
    bottom: auto;
    left: 1.5rem;
    border-top: 2px dashed var(--flight-control-border);
    border-left: 0;
  }
  .showcase .optimization-stage {
    display: block;
  }
  .showcase .stage-body { margin-top: 1rem; }
  .showcase .decision-rail { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .showcase .decision-rail div + div {
    padding-left: 1rem;
    border-left: 1px solid var(--flight-panel-line);
  }
  .showcase .flight-log ol { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .showcase .verdict-column { padding: 1.75rem; }
  .showcase .section-heading {
    grid-template-columns: minmax(0, .65fr) minmax(0, 1.35fr);
    align-items: end;
  }
  .showcase .tolerance-visual-heading {
    grid-template-columns: minmax(0, .8fr) minmax(0, 1.2fr);
  }
  .showcase .why-layout {
    grid-template-columns: minmax(0, .72fr) minmax(0, 1.28fr);
  }
  .showcase .profile-panel {
    grid-template-columns: minmax(0, 1.2fr) minmax(0, .8fr);
  }
}
@media (min-width: 52rem) {
  .showcase .tradeoff-row {
    grid-template-columns:
      minmax(0, 1.1fr)
      minmax(0, .8fr)
      4.5rem
      minmax(0, .8fr)
      minmax(0, 1fr);
  }
}
@media (min-width: 64rem) {
  .showcase .hero-layout {
    display: grid;
    width: 100%;
    max-width: 78rem;
    margin-inline: auto;
    grid-template-columns: minmax(0, 1.08fr) minmax(0, .92fr);
    gap: clamp(2.5rem, 5vw, 5.5rem);
    align-items: start;
  }
  .showcase .hero-proof { padding-top: .35rem; }
  .showcase .hero-proof .report-lede { margin-top: 0; }
  .showcase .hero-proof .decision-rail {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    margin-top: 1.5rem;
  }
  .showcase .hero-proof .decision-rail div:nth-child(3) {
    padding-left: 0;
    border-left: 0;
  }
  .showcase .hero-proof .decision-rail div:nth-child(n + 3) {
    border-top: 1px solid var(--flight-panel-line);
  }
  .showcase .flight-log { margin-top: 2rem; }
}
@media (min-width: 68rem) {
  .showcase .capacity-boards {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 2.2rem;
  }
  .showcase .load-context-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .showcase .load-binding,
  .showcase .load-context:last-child {
    grid-column: 1 / -1;
  }
  .showcase .chart-grid-layout {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .showcase .chart-grid-layout > :last-child:nth-child(odd):not(:first-child) {
    grid-column: 1 / -1;
    width: calc((100% - 1.5rem) / 2);
    justify-self: center;
  }
}
@media (max-width: 47.99rem) {
  .showcase .report-header {
    padding-inline: 1rem;
    padding-bottom: 2.75rem;
  }
  .showcase .provenance-strip { margin-inline: -1rem; padding-inline: 1rem; }
  .showcase .provenance-strip ul { display: grid; grid-template-columns: 1fr 1fr; gap: .35rem 1rem; }
  .showcase .provenance-strip li + li::before { content: none; }
  .showcase .brand-line { margin: 1.6rem 0 2.4rem; }
  .showcase h1 { font-size: clamp(3rem, 15vw, 4.5rem); }
  .showcase .cockpit-heading,
  .showcase .cockpit-decision {
    grid-template-columns: minmax(0, 1fr);
  }
  .showcase .cockpit-objective {
    grid-column: 1;
    grid-row: auto;
    margin-top: .5rem;
  }
  .showcase .cockpit-deltas { grid-template-columns: minmax(0, 1fr); }
  .showcase .cockpit-deltas > div + div {
    padding-left: 0;
    border-top: 1px solid var(--flight-panel-line);
    border-left: 0;
  }
  .showcase .optimization-ladder-inner {
    width: min(calc(100% - 2rem), 78rem);
  }
  .showcase .capacity-envelope-inner {
    width: min(calc(100% - 2rem), 78rem);
  }
  .showcase .optimization-stages {
    grid-template-columns: 1fr;
  }
  .showcase .decision-rail div:nth-child(odd) { padding-left: 0; border-left: 0; }
  .showcase .decision-rail div:nth-child(even) { padding-left: .85rem; }
  .showcase .verdict-layout { margin-top: 2rem; }
  .showcase .report-section { width: min(calc(100% - 2rem), 78rem); }
  .showcase .tolerance-direction { font-size: .64rem; }
  .showcase .chart-scroll-hint { display: block; }
  .showcase .table-scroll-hint {
    position: sticky;
    left: 0;
    display: block;
    width: fit-content;
  }
  .showcase .tolerance-row {
    grid-template-columns: minmax(0, 1fr) auto;
    gap: .55rem .8rem;
    padding: .85rem 0;
  }
  .showcase .tolerance-scale { grid-column: 1 / -1; grid-row: 2; }
  .showcase .tolerance-value { grid-column: 2; grid-row: 1; }
  .showcase .tolerance-row.is-selected { margin-inline: -.5rem; padding-inline: .5rem; }
  .showcase .series-key { grid-template-columns: 1fr; }
  .showcase .chart-figure { padding: .6rem; }
  .showcase .chart-tick,
  .showcase .chart-axis-label { font-size: 14px; }
  .showcase .trust-section { width: 100%; padding-inline: 1rem; }
}
@media print {
  html[data-theme] {
    color-scheme: light;
    background: #ffffff;
  }
  .showcase,
  html[data-theme] .showcase {
    --flight-ink: #13233d;
    --flight-ink-soft: #203653;
    --flight-panel: #13233d;
    --flight-cobalt: #2866d7;
    --flight-cobalt-solid: #2866d7;
    --flight-cobalt-soft: #e6eefc;
    --flight-teal: #116e6a;
    --flight-teal-soft: #dff3f1;
    --flight-amber: #8a5713;
    --flight-amber-soft: #f7ecd6;
    --flight-danger: #9f2d24;
    --flight-danger-soft: #f9e7e4;
    --flight-slate: #57708f;
    --flight-canvas: #ffffff;
    --flight-paper: #ffffff;
    --flight-paper-blue: #ffffff;
    --flight-text-muted: #475f7c;
    --flight-text-subtle: #57708f;
    --flight-line: #cbd6e5;
    --flight-line-strong: #9fb0c4;
    --flight-control-border: #607491;
    --flight-chart-axis: #758aa4;
    --flight-command-bg: #ffffff;
    --flight-command-text: #13233d;
    --flight-held-bg: #e7ebf1;
    --flight-purple: #6b4fa1;
    --flight-cyan: #246f91;
  }
  .showcase .report-header,
  .showcase .optimization-ladder,
  .showcase .trust-section,
  .showcase .report-footer {
    background: var(--flight-white);
    color: var(--flight-ink);
  }
  .showcase .provenance-strip,
  .showcase .canonical-column {
    background: var(--flight-white);
    color: var(--flight-ink);
    border: 1px solid var(--flight-ink);
  }
  .showcase .hero-actions { display: none; }
  .showcase .theme-toggle { display: none; }
  .showcase .report-header *,
  .showcase .optimization-ladder *,
  .showcase .trust-section *,
  .showcase .report-footer *,
  .showcase .canonical-column * {
    color: var(--flight-ink) !important;
  }
  .showcase .optimization-ladder {
    padding-block: 1.5rem;
    border-color: var(--flight-ink);
  }
  .showcase .capacity-envelope { padding-block: 1.5rem; }
  .showcase .capacity-envelope,
  .showcase .capacity-envelope * {
    background-color: var(--flight-white);
    color: var(--flight-ink) !important;
  }
  .showcase .capacity-board {
    break-inside: avoid;
    box-shadow: none;
  }
  .showcase .optimization-ladder-heading {
    grid-template-columns: minmax(0, 1fr);
  }
  .showcase .optimization-stages {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .showcase .optimization-stages::before { display: none; }
  .showcase .stage-marker,
  .showcase .optimization-stage.is-selected .stage-marker {
    border-color: var(--flight-ink);
    background: var(--flight-white);
  }
  .showcase .stage-decision-label,
  .showcase .is-selected .stage-decision-label {
    border-color: var(--flight-ink);
    background: var(--flight-white);
  }
  .showcase .flight-log { display: none; }
  .showcase .data-disclosure:not([open]) > :not(summary),
  .showcase details:not([open]) > :not(summary) {
    display: block;
  }
  .showcase .table-scroll,
  .showcase .chart-figure,
  .showcase .json-block,
  .showcase .metadata-table code,
  .showcase .command {
    max-height: none !important;
    overflow: visible !important;
    contain: none !important;
  }
  .showcase .compact-table,
  .showcase .metadata-table,
  .showcase .load-binding .compact-table,
  .showcase .candidate-table,
  .showcase .load-table,
  .showcase .stability-table,
  .showcase .scatter-table {
    width: auto !important;
    min-width: 0 !important;
    table-layout: auto !important;
  }
  .showcase .json-block,
  .showcase .command {
    white-space: pre-wrap !important;
    overflow-wrap: anywhere;
  }
}
"""

_SHOWCASE_THEME_BOOTSTRAP = r"""
<script>
(() => {
  "use strict";
  const storageKey = "paretopilot.theme.v1";
  let savedTheme = null;
  try {
    savedTheme = window.localStorage.getItem(storageKey);
  } catch (_error) {
    savedTheme = null;
  }
  const savedThemeIsValid = savedTheme === "light" || savedTheme === "dark";
  const systemPrefersDark =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = savedThemeIsValid ? savedTheme : (systemPrefersDark ? "dark" : "light");
  document.documentElement.dataset.theme = theme;
  const themeColor = document.querySelector('meta[name="theme-color"]');
  if (themeColor) themeColor.content = theme === "dark" ? "#0b1220" : "#13233d";
})();
</script>
"""

_SHOWCASE_NOSCRIPT_HEAD = r"""
  <noscript>
    <style>
      .showcase .theme-toggle { display: none; }
      .showcase .cockpit-tabs { display: none; }
      .showcase .profile-tabs { display: none; }
      .showcase .profile-panel[hidden] { display: grid !important; }
    </style>
  </noscript>
"""

_SHOWCASE_SCRIPT = r"""
  <noscript>
    <p class="sr-only">JavaScript is unavailable, so every precomputed policy panel is shown.</p>
  </noscript>
  <script>
  (() => {
    "use strict";
    const root = document.documentElement;
    const storageKey = "paretopilot.theme.v1";
    const themeToggle = document.querySelector("[data-theme-toggle]");
    const themeState = themeToggle
      ? themeToggle.querySelector("[data-theme-state]")
      : null;
    const themeColor = document.querySelector('meta[name="theme-color"]');
    const applyTheme = (theme, persist) => {
      const resolvedTheme = theme === "dark" ? "dark" : "light";
      const isDark = resolvedTheme === "dark";
      root.dataset.theme = resolvedTheme;
      if (themeToggle) themeToggle.setAttribute("aria-pressed", String(isDark));
      if (themeState) themeState.textContent = isDark ? "On" : "Off";
      if (themeColor) themeColor.content = isDark ? "#0b1220" : "#13233d";
      if (persist) {
        try {
          window.localStorage.setItem(storageKey, resolvedTheme);
        } catch (_error) {
          // The theme still changes when storage is unavailable.
        }
      }
    };
    applyTheme(root.dataset.theme, false);
    if (themeToggle) {
      themeToggle.hidden = false;
      themeToggle.addEventListener("click", () => {
        applyTheme(root.dataset.theme === "dark" ? "light" : "dark", true);
      });
    }
    const cockpitTabs = Array.from(document.querySelectorAll("[data-cockpit-target]"));
    const cockpitPanels = Array.from(document.querySelectorAll("[data-cockpit-panel]"));
    const cockpitStatus = document.querySelector("[data-cockpit-status]");
    const activateCockpitTab = (tab, moveFocus) => {
      const target = tab.getAttribute("data-cockpit-target");
      for (const item of cockpitTabs) {
        const active = item === tab;
        item.setAttribute("aria-selected", active ? "true" : "false");
        item.setAttribute("tabindex", active ? "0" : "-1");
      }
      for (const panel of cockpitPanels) {
        panel.hidden = panel.getAttribute("data-cockpit-panel") !== target;
      }
      if (cockpitStatus) {
        const label = tab.querySelector("strong");
        const evidenceClass = tab.querySelector("span");
        cockpitStatus.textContent =
          `${label ? label.textContent : "Policy"} selected. ` +
          `${evidenceClass ? evidenceClass.textContent : "Policy"} result shown.`;
      }
      if (moveFocus) tab.focus();
      window.requestAnimationFrame(() => {
        tab.scrollIntoView({ block: "nearest", inline: "nearest" });
      });
    };
    for (const tab of cockpitTabs) {
      tab.addEventListener("click", () => activateCockpitTab(tab, true));
      tab.addEventListener("keydown", (event) => {
        const supportedKeys = ["ArrowLeft", "ArrowRight", "Home", "End"];
        if (!supportedKeys.includes(event.key)) return;
        event.preventDefault();
        let next = 0;
        if (event.key === "Home") {
          next = 0;
        } else if (event.key === "End") {
          next = cockpitTabs.length - 1;
        } else {
          const step = event.key === "ArrowRight" ? 1 : -1;
          next = (cockpitTabs.indexOf(tab) + step + cockpitTabs.length) % cockpitTabs.length;
        }
        const target = cockpitTabs[next];
        if (target) activateCockpitTab(target, true);
      });
    }
    const tabs = Array.from(document.querySelectorAll("[data-profile-target]"));
    const tablist = document.querySelector(".profile-tabs");
    const syncTabOverflow = () => {
      if (!tablist) return;
      const overflows = tablist.scrollWidth > tablist.clientWidth + 1;
      tablist.dataset.overflow = overflows ? "scroll" : "fit";
    };
    syncTabOverflow();
    if (tablist && typeof window.ResizeObserver === "function") {
      const overflowObserver = new window.ResizeObserver(syncTabOverflow);
      overflowObserver.observe(tablist);
      for (const tab of tabs) overflowObserver.observe(tab);
    } else {
      window.addEventListener("resize", syncTabOverflow);
    }
    for (const tab of tabs) {
      tab.addEventListener("click", () => {
        window.requestAnimationFrame(() => {
          tab.scrollIntoView({ block: "nearest", inline: "nearest" });
          syncTabOverflow();
        });
      });
      tab.addEventListener("keydown", (event) => {
        if (event.key !== "Home" && event.key !== "End") return;
        event.preventDefault();
        const target = event.key === "Home" ? tabs[0] : tabs[tabs.length - 1];
        if (target) target.click();
      });
    }
  })();
  </script>
"""


def render_showcase_v11(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
    *,
    policy_profiles: Mapping[str, Any] | None = None,
    load_sweep: Mapping[str, Any] | None = None,
    stability_summary: Mapping[str, Any] | None = None,
    evidence_lock: Mapping[str, Any] | None = None,
    evidence_lock_sha256: str = "",
    canonical_html: str | None = None,
    canonical_report_href: str = "evidence/report-v1.1.html",
    capacity_study: Mapping[str, Any] | None = None,
    capacity_evidence_lock: Mapping[str, Any] | None = None,
    capacity_study_sha256: str = "",
    capacity_study_href: str = "evidence/capacity-study.json",
    capacity_receipt_href: str = "evidence/capacity-receipt.md",
    benchmarks_sha256: str = "",
    recommendation_sha256: str = "",
    profiles_sha256: str = "",
    load_sha256: str = "",
    stability_sha256: str = "",
) -> str:
    """Render a responsive presentation of already validated v1.1 evidence.

    ``evidence_lock`` and ``canonical_html`` are a pair: supplying one without
    the other fails closed.  The optional capacity study and its review lock are
    also a pair, and are accepted only when the canonical proof is locked.
    When no canonical proof is supplied, the page is explicitly labelled as an
    unverified preview.
    """

    canonical_report_href = _validated_report_href(canonical_report_href)
    canonical = render_report_v11(
        benchmarks,
        recommendation,
        policy_profiles=policy_profiles,
        load_sweep=load_sweep,
        stability_summary=stability_summary,
        benchmarks_sha256=benchmarks_sha256,
        recommendation_sha256=recommendation_sha256,
        profiles_sha256=profiles_sha256,
        load_sha256=load_sha256,
        stability_sha256=stability_sha256,
    )
    if (evidence_lock is None) != (canonical_html is None):
        raise ValidationError("evidence_lock and canonical_html must be supplied together")
    if canonical_html is not None and canonical_html != canonical:
        raise ValidationError(
            "supplied canonical_html does not match the validated v1.1 renderer output"
        )
    if (capacity_study is None) != (capacity_evidence_lock is None):
        raise ValidationError("capacity_study and capacity_evidence_lock must be supplied together")
    if capacity_study is not None and evidence_lock is None:
        raise ValidationError("capacity evidence requires locked canonical proof")

    canonical_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    proof = _proof_context(
        evidence_lock,
        benchmarks,
        recommendation,
        canonical_sha256=canonical_sha256,
        benchmarks_sha256=benchmarks_sha256,
        recommendation_sha256=recommendation_sha256,
        profiles_sha256=profiles_sha256,
        load_sha256=load_sha256,
        stability_sha256=stability_sha256,
    )
    passport = _decision_passport(benchmarks, recommendation)
    optimization_ladder = _optimization_ladder_markup(passport)
    capacity_envelope = ""
    if capacity_study is not None and capacity_evidence_lock is not None:
        capacity_proof = _capacity_proof_context(
            capacity_study,
            capacity_evidence_lock,
            _mapping(evidence_lock, "evidence_lock"),
            study_sha256=capacity_study_sha256,
            canonical_lock_sha256=evidence_lock_sha256,
        )
        capacity_envelope = _capacity_envelope_markup(
            capacity_study,
            capacity_proof,
            study_href=capacity_study_href,
            receipt_href=capacity_receipt_href,
        )
    policy_cockpit = _policy_cockpit_markup(benchmarks, policy_profiles)
    provenance, hero, hero_tail = _hero_markup(
        benchmarks,
        recommendation,
        proof,
        canonical_report_href=canonical_report_href,
        policy_cockpit=policy_cockpit,
        capacity_available=bool(capacity_envelope),
    )
    legend, style_by_id = _series_key(
        benchmarks,
        recommendation,
        label="Candidate encoding used in every chart",
    )
    tolerance = _tolerance_visual(benchmarks, recommendation)

    document = canonical
    if not proof:
        document, source_badge_count = re.subn(
            r'(<span class="source-type">).*?(</span>)',
            r"\1Unverified presentation preview · v1.1 view\2",
            document,
            count=1,
        )
        if source_badge_count != 1:
            raise ValidationError("canonical source badge is missing")
    theme_toggle = (
        '<button type="button" class="theme-toggle" data-theme-toggle '
        'aria-pressed="false" hidden>'
        '<span class="theme-toggle-label">Dark mode</span>'
        '<span class="theme-toggle-state" data-theme-state aria-hidden="true">Off</span>'
        "</button>"
    )
    document, brand_control_count = re.subn(
        (
            r'(<div class="brand-line"><span class="brand">ParetoPilot</span>)'
            r'(<span class="source-type">.*?</span>)(</div>)'
        ),
        lambda match: (
            f'{match.group(1)}<div class="brand-controls">{match.group(2)}'
            f"{theme_toggle}</div>{match.group(3)}"
        ),
        document,
        count=1,
    )
    if brand_control_count != 1:
        raise ValidationError("canonical brand line is missing")
    document_title = (
        "<title>ParetoPilot | synthetic decision preview</title>"
        if benchmarks.synthetic
        else "<title>ParetoPilot | Arm64 measured flight log</title>"
    )
    document = _replace_once(
        document,
        "<title>ParetoPilot v1.1 deployment decision report</title>",
        document_title,
        "document title",
    )
    meta_description = (
        "Byte-verified Arm64 deployment decision from ParetoPilot canonical run data."
        if proof
        else "Unverified preview of a ParetoPilot Arm64 deployment decision."
    )
    document = _replace_once(
        document,
        '<meta name="color-scheme" content="light">\n',
        (
            '<meta name="color-scheme" content="light dark">\n'
            '<meta name="theme-color" content="#13233d">\n'
            f'<meta name="description" content="{_escape(meta_description)}">\n'
            f"{_SHOWCASE_THEME_BOOTSTRAP}"
        ),
        "head metadata",
    )
    document = _replace_once(
        document,
        "</style>\n</head>",
        f"{_SHOWCASE_CSS}</style>\n{_SHOWCASE_NOSCRIPT_HEAD}</head>",
        "style close",
    )
    body_class = "showcase is-verified" if proof else "showcase is-preview"
    document = _replace_once(
        document,
        "<body>\n",
        f'<body class="{body_class}">\n',
        "body",
    )
    document = _replace_once(
        document,
        '<header class="report-header">\n',
        f'<header class="report-header">\n{provenance}',
        "report header",
    )
    document = _replace_once(
        document,
        "<h1>Arm64 deployment decision evidence</h1>\n"
        '<p class="report-lede">One measured study can support different deployment '
        "priorities without pretending there is one universal winner.</p>\n",
        hero,
        "hero copy",
    )
    document = _replace_once(
        document,
        '<section class="verdict-layout" aria-label="Decision summary">',
        f'{hero_tail}<section class="verdict-layout" aria-label="Decision summary">',
        "decision summary",
    )
    document = _replace_once(
        document,
        '<main id="main-content" class="report-main">\n',
        (f'<main id="main-content" class="report-main">\n{optimization_ladder}{capacity_envelope}'),
        "optimization ladder",
    )
    document = _replace_once(
        document,
        '<div class="why-layout">',
        f'{tolerance}<div class="why-layout">',
        "objective tolerance layout",
    )
    if '<div class="chart-grid-layout">' in document:
        document = _replace_once(
            document,
            '<div class="chart-grid-layout">',
            f'{legend}<div class="chart-grid-layout">',
            "load chart grid",
        )
    if '<figure class="chart-figure scatter-figure">' in document:
        document = _replace_once(
            document,
            '<figure class="chart-figure scatter-figure">',
            f'{legend}<figure class="chart-figure scatter-figure">',
            "scatter chart",
        )
    document = _tag_chart_series(document, benchmarks, style_by_id)
    document = _add_section_kickers(document)
    document = document.replace(
        '<h2 id="scatter-heading">p95 end-to-end latency vs generation throughput</h2>',
        '<h2 id="scatter-heading">Latency versus generation throughput</h2>',
    )
    if load_sweep is not None:
        document = _wrap_table_region(
            document,
            aria_label="Scrollable load command bindings",
            summary="Inspect validated command bindings",
        )
        document = _wrap_table_region(
            document,
            aria_label="Scrollable load sweep evidence",
            summary=(
                "Inspect every synthetic fixture load row"
                if benchmarks.synthetic
                else "Inspect every measured load row"
            ),
        )
    if stability_summary is not None:
        document = _wrap_table_region(
            document,
            aria_label="Scrollable pass-level stability evidence",
            summary="Inspect pass-level values and deltas",
        )
    document = _correct_load_axis_ceilings(document, load_sweep)
    document = _add_load_slo_reference(
        document,
        load_sweep,
        synthetic=benchmarks.synthetic,
    )
    document = _add_stability_explainer(document, stability_summary)
    document = _label_interactive_regions(document, benchmarks, load_sweep)
    document = _replace_once(
        document,
        (
            '<div class="table-scroll" tabindex="0" role="region" '
            'aria-label="Scrollable full candidate evidence">'
        ),
        (
            '<div class="table-scroll" tabindex="0" role="region" '
            'aria-label="Scrollable full candidate evidence">'
            '<p class="table-scroll-hint">Scroll the table horizontally.</p>'
        ),
        "candidate evidence table",
    )
    document = _add_release_hashes(document, proof)
    scatter_kind = "synthetic fixture" if benchmarks.synthetic else "measured"
    document = document.replace(
        (
            "Each labeled point is one measured candidate. Left is lower p95 end-to-end "
            "latency; higher is greater generation throughput."
        ),
        (
            f"Each point is one {scatter_kind} candidate; the candidate legend carries the full "
            "names. Left is lower p95 end-to-end latency; higher is greater generation "
            "throughput."
        ),
    )
    document = document.replace('viewBox="0 0 800 310"', 'viewBox="0 0 650 310"')
    document = _replace_once(
        document,
        "</body>\n",
        f"{_SHOWCASE_SCRIPT}</body>\n",
        "body close",
    )
    return document
