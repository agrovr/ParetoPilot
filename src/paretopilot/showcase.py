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

from paretopilot.domain import BenchmarkSet, Candidate, ValidationError
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
    if threshold > y_domain_max:
        new_description = (
            f"{old_description} The declared {_format_number(threshold, digits=0)} ms "
            "latency ceiling is above the plotted measured range; a passing level must "
            "also satisfy the TTFT and completion-rate gates."
        )
    else:
        new_description = (
            f"{old_description} The amber line marks the declared "
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


def _validated_report_href(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError("canonical_report_href must be a non-empty string")
    if "\\" in value:
        raise ValidationError("canonical_report_href must use URL path separators")
    parsed = urlsplit(value)
    if parsed.scheme:
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValidationError("canonical_report_href must be a relative path or an HTTPS URL")
        return value
    if parsed.netloc or value.startswith(("/", "#")):
        raise ValidationError("canonical_report_href must be a relative path or an HTTPS URL")
    path_parts = tuple(part for part in parsed.path.split("/") if part)
    if not path_parts or ".." in path_parts:
        raise ValidationError("canonical_report_href must be a safe relative path")
    return value


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
        "<figcaption>Marker positions show the measured objective values on one shared scale. "
        "Exact values and decision roles remain in the evidence table below.</figcaption>"
        "</figure>"
    )


def _hero_markup(
    benchmarks: BenchmarkSet,
    recommendation: Mapping[str, Any],
    proof: Mapping[str, str],
    *,
    canonical_report_href: str,
) -> tuple[str, str, str]:
    source = _source_context(benchmarks)
    selected = benchmarks.by_id(str(recommendation["selected_id"]))
    selection = _mapping(recommendation.get("selection"), "recommendation.selection")
    objective = _mapping(recommendation.get("objective"), "recommendation.objective")
    shortlist = [str(item) for item in selection.get("shortlist_ids", ())]
    tolerance = float(selection.get("objective_tolerance_percent"))
    metric = str(objective.get("metric"))
    selected_value = float(selected.metrics[metric])

    provenance_items = [
        f"{'Canonical' if proof else 'Source'} run {source['run_id']}",
        source["cpu"],
        f"{source['cpu_count']} {source['architecture']} vCPUs",
        f"{len(benchmarks.candidates)} measured candidates",
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
        f'<h1><span class="hero-selection">{_escape(selected.label)}</span> '
        "stays the deployment choice.</h1>\n"
    )
    evidence_copy = (
        "This presentation view is derived from the byte-verified, locked v1.1 evidence."
        if proof
        else (
            "This is an unverified presentation preview; no release lock or byte-verified "
            "canonical report was supplied."
        )
    )
    lede = (
        '<p class="report-lede">One measured Arm64 study, '
        f"{len(benchmarks.candidates)} candidates, and the tradeoffs behind the decision. "
        f"{_escape(evidence_copy)}</p>\n"
    )
    decision_rail = (
        '<dl class="decision-rail" aria-label="Decision at a glance">'
        f"<div><dt>Selected objective</dt><dd>{_escape(_metric_value(metric, selected_value))}</dd></div>"
        f"<div><dt>Inside cutoff</dt><dd>{len(shortlist)} of {len(benchmarks.candidates)}</dd></div>"
        f"<div><dt>Predeclared window</dt><dd>{tolerance:.2f}%</dd></div>"
        f"<div><dt>Evidence class</dt><dd>{'Locked canonical' if proof else 'Unverified preview'}</dd></div>"
        "</dl>\n"
    )
    actions = [
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
                f"Download {proof['release_tag']} evidence",
                "secondary",
            ),
        )
    action_markup = (
        '<nav class="hero-actions" aria-label="Evidence links">'
        + "".join(
            f'<a class="action-{kind}" href="{_escape(href)}">{_escape(label)}</a>'
            for href, label, kind in actions
        )
        + "</nav>\n"
    )
    section_links = (
        ("why-heading", "01", "Decision"),
        ("tradeoffs-heading", "02", "Tradeoffs"),
        ("policies-heading", "03", "Policies"),
        ("load-heading", "04", "Load"),
        ("repeat-heading", "05", "Repeat"),
        ("scatter-heading", "06", "Two-metric"),
        ("evidence-heading", "07", "Evidence"),
        ("trust-heading", "08", "Reproduce"),
    )
    flight_log = (
        '<nav class="flight-log" aria-label="Report sections">'
        '<span class="flight-log-label">Flight log</span><ol>'
        + "".join(
            f'<li><a href="#{heading_id}"><strong>{number}</strong>{label}</a></li>'
            for heading_id, number, label in section_links
        )
        + "</ol></nav>\n"
    )
    return provenance, headline + lede, decision_rail + action_markup + flight_log


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
  grid-template-columns: minmax(13rem, 1.05fr) minmax(15rem, 2fr) minmax(7rem, .65fr);
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
  overflow-wrap: anywhere;
}
.showcase thead th {
  background: var(--flight-panel);
  color: var(--flight-white);
}
.showcase tbody tr:nth-child(even) { background: var(--flight-paper-blue); }
.showcase .table-scroll,
.showcase .command {
  width: 100%;
  max-width: 100%;
  contain: inline-size paint;
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
  max-width: 100%;
  overflow-x: auto;
  border: 2px solid var(--flight-ink);
  contain: inline-size paint;
  overscroll-behavior-inline: contain;
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
.showcase .profile-panel {
  padding-top: 2rem;
  border-bottom: 2px solid var(--flight-ink);
}
.showcase .profile-metrics { border-color: var(--flight-line-strong); }
.showcase .profile-metrics li { border-color: var(--flight-line); }
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
.showcase .trust-section table { color: var(--flight-ink); }
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
  .showcase .decision-rail { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .showcase .decision-rail div + div {
    padding-left: 1rem;
    border-left: 1px solid var(--flight-panel-line);
  }
  .showcase .flight-log ol { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .showcase .verdict-column { padding: 1.75rem; }
  .showcase .section-heading {
    grid-template-columns: minmax(15rem, .65fr) minmax(22rem, 1.35fr);
    align-items: end;
  }
  .showcase .tolerance-visual-heading {
    grid-template-columns: minmax(15rem, .8fr) minmax(22rem, 1.2fr);
  }
}
@media (min-width: 68rem) {
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
  .showcase .profile-tabs { margin-inline: -1rem; border-inline: 0; }
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
  .showcase .trust-section *,
  .showcase .report-footer *,
  .showcase .canonical-column * {
    color: var(--flight-ink) !important;
  }
  .showcase .flight-log { display: none; }
  .showcase .data-disclosure:not([open]) > :not(summary),
  .showcase details:not([open]) > :not(summary) {
    display: block;
  }
  .showcase .table-scroll,
  .showcase .chart-figure,
  .showcase .json-block,
  .showcase .command {
    max-height: none !important;
    overflow: visible !important;
    contain: none !important;
  }
  .showcase .compact-table,
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
    const tabs = Array.from(document.querySelectorAll("[data-profile-target]"));
    for (const tab of tabs) {
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
    canonical_html: str | None = None,
    canonical_report_href: str = "evidence/report-v1.1.html",
    benchmarks_sha256: str = "",
    recommendation_sha256: str = "",
    profiles_sha256: str = "",
    load_sha256: str = "",
    stability_sha256: str = "",
) -> str:
    """Render a responsive presentation of already validated v1.1 evidence.

    ``evidence_lock`` and ``canonical_html`` are a pair: supplying one without
    the other fails closed.  When neither is supplied, the page is explicitly
    labelled as an unverified preview.
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
    provenance, hero, hero_tail = _hero_markup(
        benchmarks,
        recommendation,
        proof,
        canonical_report_href=canonical_report_href,
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
    document = _replace_once(
        document,
        "<title>ParetoPilot v1.1 deployment decision report</title>",
        "<title>ParetoPilot | Arm64 measured flight log</title>",
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
            summary="Inspect every measured load row",
        )
    if stability_summary is not None:
        document = _wrap_table_region(
            document,
            aria_label="Scrollable pass-level stability evidence",
            summary="Inspect pass-level values and deltas",
        )
    document = _correct_load_axis_ceilings(document, load_sweep)
    document = _add_load_slo_reference(document, load_sweep)
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
    document = document.replace(
        (
            "Each labeled point is one measured candidate. Left is lower p95 end-to-end "
            "latency; higher is greater generation throughput."
        ),
        (
            "Each point is one measured candidate; the candidate legend carries the full "
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
