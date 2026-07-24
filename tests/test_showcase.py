"""Tests for the judge-facing presentation layer."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import re
import unittest

from paretopilot.domain import ValidationError
from paretopilot.showcase import render_showcase_v11

from test_report_v11 import (
    canonical_benchmarks,
    canonical_recommendation,
    derived_profiles,
    measured_load_sweep,
    measured_stability,
    rendered_v11,
)


def contrast_ratio(foreground: str, background: str) -> float:
    def luminance(value: str) -> float:
        channels = [int(value[index : index + 2], 16) / 255 for index in range(1, len(value), 2)]
        linear = [
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    foreground_luminance = luminance(foreground)
    background_luminance = luminance(background)
    lighter = max(foreground_luminance, background_luminance)
    darker = min(foreground_luminance, background_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def css_rule_body(document: str, selector: str) -> str:
    marker = f"{selector} {{"
    selector_start = document.find(marker)
    if selector_start == -1:
        raise AssertionError(f"CSS rule is missing: {selector}")

    opening_brace = selector_start + len(marker) - 1
    depth = 0
    for index in range(opening_brace, len(document)):
        if document[index] == "{":
            depth += 1
        elif document[index] == "}":
            depth -= 1
            if depth == 0:
                return document[opening_brace + 1 : index]
    raise AssertionError(f"CSS rule is not closed: {selector}")


def css_hex_tokens(document: str, selector: str) -> dict[str, str]:
    tokens = {
        name: value.lower()
        for name, value in re.findall(
            r"(--[\w-]+)\s*:\s*(#[0-9a-fA-F]{6})\s*;",
            css_rule_body(document, selector),
        )
    }
    if not tokens:
        raise AssertionError(f"CSS rule has no hexadecimal tokens: {selector}")
    return tokens


def evidence_lock(
    *,
    artifacts_sha256: dict[str, str] | None = None,
) -> dict[str, object]:
    if artifacts_sha256 is None:
        artifacts_sha256 = {
            "benchmark_set": "a" * 64,
            "recommendation": "b" * 64,
            "policy_profiles": "c" * 64,
            "load_evaluation": "d" * 64,
            "repeat_stability": "e" * 64,
            "report_v1_1": ("23347e60894f6563978ae530bb70604b17448f191e6fc51c85860ec8f9f97f35"),
        }
    return {
        "schema_version": "1.1",
        "classification": "canonical",
        "source": {"run_id": "not supplied"},
        "archive": {
            "release_tag": "v1.1.0",
            "release_url": "https://github.com/agrovr/ParetoPilot/releases/tag/v1.1.0",
            "sha256": "f" * 64,
        },
        "review": {
            "checksum_entries": 150,
            "checksum_manifest_sha256": "a" * 64,
            "artifacts_sha256": artifacts_sha256,
            "all_checksums_verified": True,
            "exact_file_coverage": True,
            "status_complete": True,
            "measurement_valid": True,
            "valid_evidence": True,
            "replay": {
                "valid": True,
                "decision_reproduced": True,
                "fully_reproduced": True,
                "report_matches_archive": True,
                "differences": [],
                "warnings": [],
                "checksum_entry_count": 150,
                "checksum_manifest_sha256": "a" * 64,
                "selected_id": "q8-generic",
                "authoritative_comparisons": [
                    "benchmark-set",
                    "benchmark-set-pass-1",
                    "benchmark-set-pass-2",
                    "load-evaluation",
                    "policy-profiles",
                    "recommendation",
                    "repeat-stability",
                    "report",
                    "report-v1.1",
                ],
            },
        },
    }


def rendered_showcase(
    *,
    lock: bool = True,
    canonical_html: str | None = None,
    load_sweep: dict[str, object] | None = None,
) -> str:
    benchmarks = canonical_benchmarks()
    recommendation = canonical_recommendation(benchmarks)
    load = measured_load_sweep() if load_sweep is None else load_sweep
    canonical = rendered_v11(data=benchmarks, load=load)
    supplied_canonical = (canonical if canonical_html is None else canonical_html) if lock else None
    return render_showcase_v11(
        benchmarks,
        recommendation,
        policy_profiles=derived_profiles(benchmarks),
        load_sweep=load,
        stability_summary=measured_stability(benchmarks),
        evidence_lock=evidence_lock() if lock else None,
        canonical_html=supplied_canonical,
        canonical_report_href="evidence/report-v1.1.html",
        benchmarks_sha256="a" * 64,
        recommendation_sha256="b" * 64,
        profiles_sha256="c" * 64,
        load_sha256="d" * 64,
        stability_sha256="e" * 64,
    )


class ShowcaseV11Tests(unittest.TestCase):
    def test_archived_renderer_fixture_digest_remains_frozen(self) -> None:
        canonical = rendered_v11()

        self.assertEqual(
            hashlib.sha256(canonical.encode()).hexdigest(),
            "23347e60894f6563978ae530bb70604b17448f191e6fc51c85860ec8f9f97f35",
        )
        self.assertNotIn('class="showcase', canonical)
        self.assertNotIn("Measured Flight Log", canonical)

    def test_showcase_is_deterministic_and_keeps_the_evidence_story(self) -> None:
        first = rendered_showcase()
        second = rendered_showcase()

        self.assertEqual(first.encode(), second.encode())
        self.assertIn("<title>ParetoPilot | Arm64 measured flight log</title>", first)
        self.assertIn('class="showcase is-verified"', first)
        self.assertIn(
            '<span class="hero-selection">Q8 generic reference</span> stays the deployment choice.',
            first,
        )
        self.assertIn("1.00% objective tolerance", first)
        self.assertIn("150 files verified", first)
        self.assertIn("9 authoritative comparisons", first)
        self.assertIn("arm64 vCPUs", first)
        self.assertIn("Open exact canonical report", first)
        self.assertIn('href="evidence/report-v1.1.html"', first)
        self.assertIn("Canonical recommendation", first)
        self.assertIn("Derived policy scenario", first)
        self.assertIn("Trust and reproduction", first)
        self.assertIn("Benchmark SHA-256", first)
        self.assertIn("Canonical report SHA-256", first)
        self.assertIn("Evidence archive SHA-256", first)
        self.assertIn("Checksum manifest SHA-256", first)

    def test_showcase_preserves_canonical_sections_and_evidence_tables(self) -> None:
        canonical = rendered_v11()
        showcase = rendered_showcase()
        canonical_section_ids = set(re.findall(r'<section[^>]+id="([^"]+)"', canonical))
        showcase_section_ids = set(re.findall(r'<section[^>]+id="([^"]+)"', showcase))
        canonical_measurements = set(
            re.findall(
                r">(-?\d[\d,.]*(?:%| ms| MiB| tok/s)?)<",
                canonical,
            )
        )
        canonical_captions = re.findall(r"<caption>.*?</caption>", canonical, re.DOTALL)
        canonical_tab_controls = set(re.findall(r'aria-controls="([^"]+)"', canonical))
        showcase_tab_controls = set(re.findall(r'aria-controls="([^"]+)"', showcase))
        showcase_ids = set(re.findall(r'\sid="([^"]+)"', showcase))

        self.assertGreaterEqual(len(canonical_section_ids), 3)
        self.assertLessEqual(canonical_section_ids, showcase_section_ids)
        self.assertEqual(canonical.count("<tbody>"), showcase.count("<tbody>"))
        self.assertEqual(canonical.count("<tr>"), showcase.count("<tr>"))
        self.assertEqual(canonical.count("<td"), showcase.count("<td"))
        self.assertGreaterEqual(len(canonical_measurements), 20)
        self.assertGreaterEqual(len(canonical_captions), 5)
        self.assertGreaterEqual(len(canonical_tab_controls), 3)
        for measurement in canonical_measurements:
            self.assertIn(f">{measurement}<", showcase)
        for caption in canonical_captions:
            self.assertIn(caption, showcase)
        self.assertIn(
            '<main id="main-content" class="report-main" tabindex="-1">',
            showcase,
        )
        self.assertIn('role="tablist"', showcase)
        self.assertLessEqual(canonical_tab_controls, showcase_tab_controls)
        self.assertLessEqual(canonical_tab_controls, showcase_ids)
        self.assertNotEqual(canonical.encode(), showcase.encode())

    def test_charts_use_stable_series_tags_and_responsive_html_legends(self) -> None:
        report = rendered_showcase()
        expected_shapes = {
            "Q8 generic reference": "circle",
            "Q4 generic": "circle",
            "Q4 + KleidiAI": "square",
            "Q4 + KleidiAI tuned": "triangle",
        }

        self.assertGreaterEqual(report.count('data-series-style="0"'), 6)
        self.assertIn("Candidate encoding used in every chart", report)
        self.assertIn('<span class="series-name">Q4 + KleidiAI tuned</span>', report)
        self.assertIn(".showcase .chart-figure svg", report)
        self.assertIn("min-width: 0;", report)
        self.assertIn(".showcase .direct-label,", report)
        self.assertIn("display: none;", report)
        self.assertIn(".showcase [data-series-style] .chart-line", report)
        self.assertIn("fill: none;", report)
        self.assertIn('class="series-swatch"', report)
        self.assertIn('class="legend-marker"', report)
        self.assertIn('class="legend-marker legend-marker-selected"', report)
        self.assertIn("stroke-dasharray: var(--series-dash);", report)
        self.assertIn(".showcase .scatter-figure svg", report)
        self.assertIn("width: min(100%, 40rem);", report)
        self.assertNotIn('viewBox="0 0 800 310"', report)
        self.assertIn('viewBox="0 0 650 310"', report)
        self.assertNotIn(
            'role="region" aria-label="Scrollable JSON details"',
            report,
        )
        self.assertIn("max-height: none !important;", report)
        self.assertIn("table-layout: auto !important;", report)
        self.assertIn("white-space: pre-wrap !important;", report)
        self.assertIn('<p class="table-scroll-hint">Scroll the table horizontally.</p>', report)
        self.assertIn(".showcase .candidate-table {\n  width: 92rem;", report)
        self.assertIn(".showcase .candidate-table caption {", report)
        self.assertIn(
            '.showcase [data-series-style="0"] .chart-marker {',
            report,
        )
        self.assertNotIn(
            '.showcase .scatter-figure [data-series-style="0"] .chart-marker {',
            report,
        )
        self.assertIn(
            '.showcase .scatter-figure .chart-tick[x="68"] {\n  text-anchor: end;\n}',
            report,
        )
        self.assertNotIn(
            '.showcase .chart-tick[x="68"][y="34"]',
            report,
        )
        for label, shape in expected_shapes.items():
            groups = re.findall(
                rf'<g aria-label="{re.escape(label)}"[^>]*'
                rf'data-marker-shape="{shape}"[^>]*>(.*?)</g>',
                report,
                re.DOTALL,
            )
            self.assertGreaterEqual(len(groups), 4, label)
            expected_tag = {
                "circle": "circle",
                "square": "rect",
                "triangle": "path",
            }[shape]
            for group in groups:
                self.assertIn(f'<{expected_tag} class="chart-marker', group)
            self.assertRegex(
                report,
                rf'<li[^>]*data-marker-shape="{shape}"[^>]*>.*?'
                rf'<span class="series-name">{re.escape(label)}</span>',
            )

    def test_load_chart_axes_and_slo_label_use_the_actual_plot_domain(self) -> None:
        load = measured_load_sweep()
        report = rendered_showcase()
        chart_metrics = (
            ("requests_per_second", "req/s"),
            ("generated_tokens_per_second", "tok/s"),
            ("e2e_latency_ms_p95", "ms"),
        )

        for metric, unit in chart_metrics:
            maximum = max(float(row[metric]) for row in load["rows"])
            ceiling = f"{maximum * 1.08:,.4f}".rstrip("0").rstrip(".")
            self.assertIn(f">{ceiling} {unit}</text>", report)
        self.assertIn('class="slo-reference-label"', report)
        self.assertIn('text-anchor="start">SLO', report)

    def test_out_of_range_slo_is_annotated_without_a_false_reference_line(self) -> None:
        load = deepcopy(measured_load_sweep())
        maximum = max(float(row["e2e_latency_ms_p95"]) for row in load["rows"])
        threshold = maximum * 2
        load["slo"]["max_e2e_latency_ms_p95"] = threshold

        report = rendered_showcase(lock=False, load_sweep=load)

        self.assertIn(
            f"SLO · {threshold:,.0f} ms · above plotted range",
            report,
        )
        self.assertNotIn('<line class="slo-reference-line"', report)
        self.assertIn("latency ceiling is above the plotted measured range", report)

    def test_no_javascript_fallback_css_is_in_the_document_head(self) -> None:
        report = rendered_showcase()
        head, body = report.split("</head>", 1)

        self.assertIn("<noscript>", head)
        self.assertIn(".showcase .theme-toggle { display: none; }", head)
        self.assertIn(".showcase .profile-tabs { display: none; }", head)
        self.assertIn(".showcase .profile-panel[hidden]", head)
        self.assertNotRegex(body, r"<noscript>\s*<style>")
        self.assertIn("JavaScript is unavailable", body)

    def test_theme_toggle_is_accessible_persistent_and_applied_before_styles(self) -> None:
        report = rendered_showcase()
        head, body = report.split("</head>", 1)

        self.assertEqual(report.count("data-theme-toggle"), 2)
        self.assertEqual(
            report.count(
                '<button type="button" class="theme-toggle" data-theme-toggle '
                'aria-pressed="false" hidden>'
            ),
            1,
        )
        self.assertIn('<span class="theme-toggle-label">Dark mode</span>', report)
        self.assertIn("data-theme-state", report)
        self.assertIn('<meta name="color-scheme" content="light dark">', head)
        self.assertLess(head.index("paretopilot.theme.v1"), head.index("<style>"))
        self.assertIn('savedTheme === "light" || savedTheme === "dark"', head)
        self.assertIn('window.matchMedia("(prefers-color-scheme: dark)")', head)
        self.assertIn("window.localStorage.getItem(storageKey)", head)
        self.assertIn("window.localStorage.setItem(storageKey, resolvedTheme)", body)
        self.assertIn("try {", head)
        self.assertIn("try {", body)
        self.assertIn("root.dataset.theme = resolvedTheme;", body)
        self.assertNotIn(".style.colorScheme", report)
        self.assertIn(
            'themeToggle.setAttribute("aria-pressed", String(isDark))',
            body,
        )
        self.assertIn(
            'themeColor.content = isDark ? "#0b1220" : "#13233d"',
            body,
        )
        self.assertIn('html[data-theme="dark"] .showcase {', report)
        self.assertIn(".showcase .theme-toggle { display: none; }", report)
        state_sync = body.index("applyTheme(root.dataset.theme, false);")
        reveal = body.index("themeToggle.hidden = false;")
        listener = body.index('themeToggle.addEventListener("click"')
        self.assertLess(state_sync, reveal)
        self.assertLess(reveal, listener)

        print_theme_selector = ".showcase,\n  html[data-theme] .showcase"
        print_rule_start = report.index(f"{print_theme_selector} {{")
        print_media_start = report.rfind("@media print {", 0, print_rule_start)
        self.assertNotEqual(print_media_start, -1)
        print_css = css_rule_body(report[print_media_start:], "@media print")
        self.assertIn(
            f"{print_theme_selector} {{",
            print_css,
        )
        print_tokens = css_hex_tokens(print_css, print_theme_selector)
        light_tokens = css_hex_tokens(report, ".showcase")
        for token in (
            "--flight-ink",
            "--flight-cobalt",
            "--flight-teal",
            "--flight-amber",
            "--flight-danger",
            "--flight-text-muted",
            "--flight-control-border",
            "--flight-chart-axis",
            "--flight-purple",
            "--flight-cyan",
        ):
            with self.subTest(print_token=token):
                self.assertEqual(print_tokens[token], light_tokens[token])
        self.assertEqual(print_tokens["--flight-command-text"], light_tokens["--flight-ink"])
        for token in (
            "--flight-canvas",
            "--flight-paper",
            "--flight-paper-blue",
            "--flight-command-bg",
        ):
            with self.subTest(print_surface=token):
                self.assertEqual(print_tokens[token], "#ffffff")
        self.assertIn(".showcase .theme-toggle { display: none; }", print_css)

    def test_light_and_dark_theme_tokens_meet_contrast_requirements(self) -> None:
        report = rendered_showcase()
        light_tokens = css_hex_tokens(report, ".showcase")
        dark_tokens = dict(light_tokens)
        dark_tokens.update(css_hex_tokens(report, 'html[data-theme="dark"] .showcase'))
        text_pairs = (
            ("body", "--flight-ink", "--flight-canvas"),
            ("muted", "--flight-text-muted", "--flight-canvas"),
            ("subtle surface text", "--flight-text-subtle", "--flight-paper-blue"),
            ("link", "--flight-cobalt", "--flight-canvas"),
            ("inverse", "--flight-on-dark", "--flight-panel"),
            ("solid cobalt", "--flight-white", "--flight-cobalt-solid"),
            ("table header", "--flight-white", "--flight-panel"),
            ("striped table row", "--flight-ink", "--flight-paper-blue"),
            ("code", "--flight-command-text", "--flight-command-bg"),
            ("success", "--flight-teal", "--flight-teal-soft"),
            ("warning", "--flight-amber", "--flight-amber-soft"),
            ("danger", "--flight-danger", "--flight-danger-soft"),
        )
        non_text_pairs = (
            ("control border", "--flight-control-border", "--flight-panel"),
            ("focus", "--flight-focus", "--flight-canvas"),
            ("inverse focus", "--flight-focus-inverse", "--flight-panel"),
            ("chart axis", "--flight-chart-axis", "--flight-paper"),
            ("cobalt chart", "--flight-cobalt", "--flight-paper"),
            ("slate chart", "--flight-slate", "--flight-paper"),
            ("teal chart", "--flight-teal", "--flight-paper"),
            ("amber chart", "--flight-amber", "--flight-paper"),
            ("purple chart", "--flight-purple", "--flight-paper"),
            ("cyan chart", "--flight-cyan", "--flight-paper"),
        )

        for theme, tokens in (("light", light_tokens), ("dark", dark_tokens)):
            for role, foreground_token, background_token in text_pairs:
                with self.subTest(theme=theme, role=role):
                    self.assertGreaterEqual(
                        contrast_ratio(
                            tokens[foreground_token],
                            tokens[background_token],
                        ),
                        4.5,
                    )
            for role, foreground_token, background_token in non_text_pairs:
                with self.subTest(theme=theme, role=role):
                    self.assertGreaterEqual(
                        contrast_ratio(
                            tokens[foreground_token],
                            tokens[background_token],
                        ),
                        3.0,
                    )

    def test_tolerance_track_keeps_full_names_values_and_roles_in_text(self) -> None:
        report = rendered_showcase()

        self.assertIn('<figure class="tolerance-visual"', report)
        self.assertIn("Q8 generic reference", report)
        self.assertIn("Q4 + KleidiAI tuned", report)
        self.assertIn("Inside cutoff", report)
        self.assertIn("Outside cutoff", report)
        self.assertIn("2,330.9 ms", report)
        self.assertIn("2,354.2 ms", report)
        self.assertIn("Exact values and decision roles remain in the evidence table", report)

    def test_showcase_rejects_a_different_canonical_report(self) -> None:
        with self.assertRaisesRegex(ValidationError, "does not match"):
            rendered_showcase(canonical_html="<!doctype html><title>different</title>")

    def test_lock_and_canonical_report_are_paired_and_hash_bound(self) -> None:
        benchmarks = canonical_benchmarks()
        recommendation = canonical_recommendation(benchmarks)
        canonical = rendered_v11(data=benchmarks)
        kwargs = {
            "policy_profiles": derived_profiles(benchmarks),
            "load_sweep": measured_load_sweep(),
            "stability_summary": measured_stability(benchmarks),
            "benchmarks_sha256": "a" * 64,
            "recommendation_sha256": "b" * 64,
            "profiles_sha256": "c" * 64,
            "load_sha256": "d" * 64,
            "stability_sha256": "e" * 64,
        }

        with self.assertRaisesRegex(ValidationError, "must be supplied together"):
            render_showcase_v11(
                benchmarks,
                recommendation,
                evidence_lock=evidence_lock(),
                **kwargs,
            )
        with self.assertRaisesRegex(ValidationError, "must be supplied together"):
            render_showcase_v11(
                benchmarks,
                recommendation,
                canonical_html=canonical,
                **kwargs,
            )

        tampered_lock = deepcopy(evidence_lock())
        tampered_lock["review"]["artifacts_sha256"]["benchmark_set"] = "0" * 64
        with self.assertRaisesRegex(ValidationError, "artifact digest does not match"):
            render_showcase_v11(
                benchmarks,
                recommendation,
                evidence_lock=tampered_lock,
                canonical_html=canonical,
                **kwargs,
            )

        preview = render_showcase_v11(benchmarks, recommendation, **kwargs)
        self.assertIn("Unverified preview", preview)
        self.assertIn("Unverified presentation preview · v1.1 view", preview)
        self.assertIn("Source run", preview)
        self.assertNotIn("Open exact canonical report", preview)

    def test_showcase_rejects_unsafe_canonical_report_links(self) -> None:
        benchmarks = canonical_benchmarks()
        recommendation = canonical_recommendation(benchmarks)

        for href in (
            "javascript:alert(1)",
            "http://example.com/report.html",
            "//example.com/report.html",
            "/absolute/report.html",
            "../outside/report.html",
            r"evidence\report.html",
        ):
            with (
                self.subTest(href=href),
                self.assertRaisesRegex(
                    ValidationError,
                    "canonical_report_href",
                ),
            ):
                render_showcase_v11(
                    benchmarks,
                    recommendation,
                    canonical_report_href=href,
                )

    def test_showcase_rejects_unverified_or_cross_run_evidence_locks(self) -> None:
        benchmarks = canonical_benchmarks()
        recommendation = canonical_recommendation(benchmarks)
        canonical = rendered_v11(
            data=benchmarks,
            profiles=False,
            load=None,
            stability=None,
        )
        kwargs = {
            "benchmarks_sha256": "a" * 64,
            "recommendation_sha256": "b" * 64,
        }

        unverified = evidence_lock()
        unverified["review"]["all_checksums_verified"] = False
        with self.assertRaisesRegex(ValidationError, "all_checksums_verified"):
            render_showcase_v11(
                benchmarks,
                recommendation,
                evidence_lock=unverified,
                canonical_html=canonical,
                **kwargs,
            )

        cross_run = deepcopy(evidence_lock())
        cross_run["source"]["run_id"] = "another-run"
        with self.assertRaisesRegex(ValidationError, "run id"):
            render_showcase_v11(
                benchmarks,
                recommendation,
                evidence_lock=cross_run,
                canonical_html=canonical,
                **kwargs,
            )


if __name__ == "__main__":
    unittest.main()
