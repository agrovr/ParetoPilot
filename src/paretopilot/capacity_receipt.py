"""Human-readable proof receipt for a validated supplementary capacity study."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from paretopilot.capacity_eval import validate_capacity_study


def render_capacity_receipt(study: Mapping[str, Any]) -> str:
    """Render deterministic Markdown from one validated capacity artifact."""

    validate_capacity_study(study)
    plan = study["plan"]
    assert isinstance(plan, Mapping)
    candidates = plan["candidates"]
    assert isinstance(candidates, list)
    candidate_by_id = {
        str(candidate["id"]): candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
    }
    selections = study["selections"]
    cells = study["cells"]
    quality_checks = study["quality_checks"]
    configurations = study["server_configurations"]
    assert isinstance(selections, list)
    assert isinstance(cells, list)
    assert isinstance(quality_checks, list)
    assert isinstance(configurations, list)

    lines = [
        "# ParetoPilot Arm64 Capacity Receipt",
        "",
        "> **SUPPLEMENTARY EVIDENCE · CANONICAL v1.1 UNCHANGED**",
        "",
        (
            "This receipt asks a separate deployment question: after the frozen v1.1 "
            "candidate decision, which observed server-slot and client-concurrency point "
            "best satisfies the predeclared capacity gates?"
        ),
        "",
        f"**Scope:** {_markdown(str(plan['scope']))}",
        "",
        (
            "`P` = llama-server slots · `C` = simultaneous clients · `TTFT` = time to "
            "first token · `E2E` = end-to-end latency · `RSS` = peak resident memory · "
            "`SLO` = the predeclared service-level limit."
        ),
        "",
        (
            "**Decision boundary:** this receipt does not choose between Q8 and Q4. Choose "
            "the model candidate first, then use its row here to select a serving point."
        ),
        "",
        "## Selected operating points",
        "",
        (
            "| Candidate | Role | Selected point | Median generation rate | "
            "Generation rate vs P1/C1 |"
        ),
        "| --- | --- | --- | ---: | ---: |",
    ]
    for selection in selections:
        assert isinstance(selection, Mapping)
        candidate = candidate_by_id[str(selection["candidate_id"])]
        selected = selection["selected_cell"]
        if selected is None:
            point = "No eligible point"
            throughput = "Not applicable"
            change = "Not applicable"
        else:
            assert isinstance(selected, Mapping)
            point = f"P{selected['server_parallel']} / C{selected['client_concurrency']}"
            throughput = f"{float(selected['generated_tokens_per_second_median']):.2f} tok/s"
            comparison = selection["comparison_to_reference_percent"]
            assert isinstance(comparison, Mapping)
            delta = float(comparison["generated_tokens_per_second_median"])
            change = f"{delta:+.1f}%"
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown(str(candidate["label"])),
                    _human_role(str(candidate["role"])),
                    point,
                    throughput,
                    change,
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            (
                "Selection maximizes the median generated-token rate only among cells where "
                "both counterbalanced passes meet the load SLO and stability limits, the "
                "quality gate passes, and peak server RSS stays within the predeclared limit. "
                "Points within the 1% objective tolerance use lower observed-p95 E2E, lower "
                "observed-p95 TTFT, lower RSS, fewer server slots, then fewer clients."
            ),
        )
    )

    for candidate_id, candidate in candidate_by_id.items():
        lines.extend(
            (
                "",
                f"## Capacity envelope — {_markdown(str(candidate['label']))}",
                "",
                "| Server slots | C1 | C2 | C4 |",
                "| ---: | --- | --- | --- |",
            )
        )
        for server_parallel in plan["server_parallel_levels"]:
            row = [f"P{server_parallel}"]
            for client_concurrency in plan["client_concurrency_levels"]:
                cell = _find_cell(
                    cells,
                    candidate_id=candidate_id,
                    server_parallel=int(server_parallel),
                    client_concurrency=int(client_concurrency),
                )
                summary = cell["summary"]
                assert isinstance(summary, Mapping)
                status = "PASS" if summary["capacity_gate_met"] else "FAIL"
                throughput = float(summary["generated_tokens_per_second_median"])
                e2e = summary["e2e_latency_ms_p95_median"]
                ttft = summary["ttft_ms_p95_median"]
                row.append(
                    "<br>".join(
                        (
                            f"**{status} · {throughput:.2f} tok/s**",
                            f"E2E p95 {_milliseconds(e2e)}",
                            f"TTFT p95 {_milliseconds(ttft)}",
                        )
                    )
                )
            lines.append("| " + " | ".join(row) + " |")

        selection = next(item for item in selections if item["candidate_id"] == candidate_id)
        assert isinstance(selection, Mapping)
        selected = selection["selected_cell"]
        if selected is None:
            lines.extend(
                (
                    "",
                    "**Envelope result:** no observed point cleared every predeclared gate.",
                )
            )
        else:
            assert isinstance(selected, Mapping)
            selected_cell = _find_cell(
                cells,
                candidate_id=candidate_id,
                server_parallel=int(selected["server_parallel"]),
                client_concurrency=int(selected["client_concurrency"]),
            )
            summary = selected_cell["summary"]
            assert isinstance(summary, Mapping)
            lines.extend(
                (
                    "",
                    (
                        f"**Envelope result:** P{selected['server_parallel']} / "
                        f"C{selected['client_concurrency']} is the best observed passing "
                        "point under the fixed objective."
                    ),
                    "",
                    "| Gate | Observed | Result |",
                    "| --- | ---: | --- |",
                    (
                        "| Both counterbalanced passes meet load SLO | "
                        f"{'Yes' if summary['every_pass_slo_met'] else 'No'} | "
                        f"{_status(bool(summary['every_pass_slo_met']))} |"
                    ),
                    (
                        "| Quality floor and retention | "
                        f"{'Yes' if summary['quality_gate_met'] else 'No'} | "
                        f"{_status(bool(summary['quality_gate_met']))} |"
                    ),
                    (
                        "| Peak server RSS | "
                        f"{float(summary['server_peak_rss_mib_max']):.1f} MiB / "
                        f"{float(plan['capacity_gate']['max_server_peak_rss_mib']):.1f} MiB | "
                        f"{_status(float(summary['server_peak_rss_mib_max']) <= float(plan['capacity_gate']['max_server_peak_rss_mib']))} |"
                    ),
                    (
                        "| Completion rate, worst pass | "
                        f"{float(summary['completion_rate_min']) * 100:.1f}% | "
                        f"{_status(float(summary['completion_rate_min']) >= float(study['load_contract']['slo']['min_completion_rate']))} |"
                    ),
                    (
                        "| Generation-rate spread | "
                        f"{float(summary['throughput_relative_spread_percent']):.1f}% / "
                        f"{float(plan['capacity_gate']['max_throughput_relative_spread_percent']):.1f}% | "
                        f"{_status(float(summary['throughput_relative_spread_percent']) <= float(plan['capacity_gate']['max_throughput_relative_spread_percent']))} |"
                    ),
                    (
                        "| Observed-p95 E2E spread | "
                        f"{float(summary['e2e_relative_spread_percent']):.1f}% / "
                        f"{float(plan['capacity_gate']['max_e2e_relative_spread_percent']):.1f}% | "
                        f"{_status(float(summary['e2e_relative_spread_percent']) <= float(plan['capacity_gate']['max_e2e_relative_spread_percent']))} |"
                    ),
                )
            )

        rejected = [
            cell
            for cell in cells
            if cell["candidate_id"] == candidate_id and not cell["summary"]["capacity_gate_met"]
        ]
        lines.extend(("", "### Rejected points", ""))
        if not rejected:
            lines.append("No point was rejected.")
        else:
            lines.extend(
                (
                    "| Point | Exact gate failures |",
                    "| --- | --- |",
                )
            )
            for cell in rejected:
                summary = cell["summary"]
                reasons = "<br>".join(
                    f"{_human_failure(str(reason))} (`{_markdown(str(reason))}`)"
                    for reason in summary["failure_reasons"]
                )
                lines.append(
                    f"| P{cell['server_parallel']} / C{cell['client_concurrency']} | {reasons} |"
                )

    lines.extend(
        (
            "",
            "## Quality guard",
            "",
            "| Candidate | Server slots | Score | Reference retention | Outcome stability | Result |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        )
    )
    for check in quality_checks:
        assert isinstance(check, Mapping)
        candidate = candidate_by_id[str(check["candidate_id"])]
        retention = check["retention_vs_reference"]
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown(str(candidate["label"])),
                    str(check["server_parallel"]),
                    f"{int(check['passed'])}/{int(check['total'])}",
                    ("Not applicable" if retention is None else f"{float(retention) * 100:.1f}%"),
                    ("Matched" if check["outcomes_match_candidate_reference"] else "Changed"),
                    _status(bool(check["gate_met"])),
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "## Methodology",
            "",
            f"- Server slots: `{', '.join(map(str, plan['server_parallel_levels']))}`.",
            f"- Simultaneous clients: `{', '.join(map(str, plan['client_concurrency_levels']))}`.",
            (
                f"- Context is held at `{int(plan['per_slot_context_tokens'])}` tokens per "
                "server slot; total context therefore scales with `--parallel`."
            ),
            (f"- Two mirrored forward/reverse passes: {_pass_description(plan['passes'])}."),
            (
                "- Every load artifact uses the checksummed fixed prompt plan, 64 output "
                "tokens, prompt caching disabled, four warmups, and eight measured requests "
                "at each client level."
            ),
            (
                "- Performance servers are restarted between configurations. GNU `time -v` "
                "captures peak RSS for each load-only 1/2/4-client sweep."
            ),
            (
                "- Six additional fresh servers run the sequential 24-case deployment guard "
                "after the performance matrix, one per candidate and slot level."
            ),
            (
                "- Capacity commands are compared with the frozen v1.1 deployment commands. "
                "Only `--parallel`, formula-bound `--ctx-size`, and the local port may differ."
            ),
            (
                "- KleidiAI-enabled runs require the observed `CPU_KLEIDIAI model buffer` "
                "marker; generic runs must not contain it. This is not a microkernel trace."
            ),
        )
    )

    provenance = study["provenance"]
    assert isinstance(provenance, Mapping)
    source = provenance["source"]
    runner = provenance["runner"]
    runtime = provenance["runtime"]
    canonical = provenance["canonical_evidence"]
    assert isinstance(source, Mapping)
    assert isinstance(runner, Mapping)
    assert isinstance(runtime, Mapping)
    assert isinstance(canonical, Mapping)
    fingerprints = study["input_fingerprints"]
    assert isinstance(fingerprints, Mapping)
    lines.extend(
        (
            "",
            "## Provenance and fingerprints",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Classification | `{study['classification']}` |",
            f"| Source repository | `{_markdown(str(source['repository']))}` |",
            f"| Source revision | `{source['revision']}` |",
            f"| Workflow run | `{source['run_id']}` attempt `{source['run_attempt']}` |",
            f"| Runner | `{_markdown(str(runner['os']))}` · `{runner['architecture']}` · `{_markdown(str(runner['cpu']))}` · `{runner['cpu_count']}` CPUs |",
            f"| Runtime | `{_markdown(str(runtime['name']))}` at `{runtime['revision']}` |",
            f"| Frozen canonical run | `{canonical['run_id']}` · `{canonical['release_tag']}` |",
            f"| Capacity plan SHA-256 | `{fingerprints['capacity_plan_sha256']}` |",
            f"| Load plan SHA-256 | `{fingerprints['load_plan_sha256']}` |",
            f"| Capacity manifest SHA-256 | `{fingerprints['manifest_sha256']}` |",
            f"| Capacity source artifacts | `{_source_artifact_count(fingerprints)}` files |",
            "| Canonical outputs modified | **No** |",
        )
    )

    lines.extend(
        (
            "",
            "## Reproduce and compare",
            "",
            "Run from the extracted supplementary capacity bundle. This first verifies every "
            "archived file, installs the exact source revision, and then requires byte-for-byte "
            "matches for both derived outputs.",
            "",
            "<details>",
            "<summary>Show exact commands</summary>",
            "",
            "```bash",
            *_reproduction_command(plan, str(source["revision"])),
            "```",
            "",
            "</details>",
            "",
            "## Boundary",
            "",
            _markdown(str(study["boundary_caveat"])),
            "",
            "**Canonical outputs modified: No**",
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def _find_cell(
    cells: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
    server_parallel: int,
    client_concurrency: int,
) -> Mapping[str, Any]:
    return next(
        cell
        for cell in cells
        if cell["candidate_id"] == candidate_id
        and cell["server_parallel"] == server_parallel
        and cell["client_concurrency"] == client_concurrency
    )


def _milliseconds(value: Any) -> str:
    return "not measured" if value is None else f"{float(value):.1f} ms"


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _markdown(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`")


def _human_role(value: str) -> str:
    return {
        "canonical-reference": "Canonical latency choice",
        "resource-alternative": "Measured resource alternative",
    }.get(value, _markdown(value))


def _human_failure(value: str) -> str:
    direct = {
        "one_or_more_passes_failed_load_slo": "At least one pass missed the load SLO",
        "server_peak_rss_above_maximum": "Peak memory exceeded the service budget",
        "quality_gate_failed": "The task-specific quality guard failed",
        "throughput_relative_spread_above_maximum": (
            "Generation rate varied too much between passes"
        ),
        "e2e_relative_spread_above_maximum": (
            "Observed-p95 end-to-end latency varied too much between passes"
        ),
    }
    if value in direct:
        return direct[value]
    pass_id, separator, reason = value.partition(":")
    reason_labels = {
        "completion_rate_below_minimum": "completion rate was below the minimum",
        "no_completed_request_ttft": "no completed request produced TTFT evidence",
        "ttft_ms_p95_above_maximum": "observed-p95 TTFT exceeded the limit",
        "no_completed_request_e2e": "no completed request produced E2E evidence",
        "e2e_latency_ms_p95_above_maximum": "observed-p95 E2E exceeded the limit",
    }
    if separator and reason in reason_labels:
        return f"{pass_id.capitalize()} pass: {reason_labels[reason]}"
    return value.replace("_", " ").capitalize()


def _pass_description(values: Any) -> str:
    assert isinstance(values, list)
    descriptions = []
    for value in values:
        assert isinstance(value, Mapping)
        candidates = " → ".join(map(str, value["candidate_order"]))
        server_levels = " → ".join(f"P{level}" for level in value["server_parallel_order"])
        client_levels = " → ".join(f"C{level}" for level in value["client_concurrency_order"])
        descriptions.append(f"`{value['id']}` ({candidates}; {server_levels}; {client_levels})")
    return "; ".join(descriptions)


def _source_artifact_count(fingerprints: Mapping[str, Any]) -> int:
    return sum(
        len(fingerprints[name])
        for name in ("load_artifacts", "rss_artifacts", "server_logs", "quality_artifacts")
    )


def _reproduction_command(plan: Mapping[str, Any], source_revision: str) -> list[str]:
    lines = [
        "sha256sum --check SHA256SUMS",
        (
            "python -m pip install --no-deps --force-reinstall "
            f'"git+https://github.com/agrovr/ParetoPilot.git@{source_revision}"'
        ),
        "paretopilot assemble-capacity \\",
        "  --plan capacity-plan.json \\",
        "  --load-plan load-plan.json \\",
        "  --manifest manifest.json \\",
    ]
    arguments: list[str] = []
    for pass_spec in plan["passes"]:
        for candidate_id in pass_spec["candidate_order"]:
            for parallel in pass_spec["server_parallel_order"]:
                label = f"{pass_spec['id']}/{candidate_id}/p{parallel}"
                root = f"runs/{pass_spec['id']}/{candidate_id}/p{parallel}"
                arguments.extend(
                    (
                        f"  --load {label}={root}/load-evaluation.json \\",
                        f"  --rss {label}={root}/server-time.txt \\",
                        f"  --server-log {label}={root}/server.stderr.log \\",
                    )
                )
    for candidate in plan["candidates"]:
        for parallel in plan["server_parallel_levels"]:
            label = f"{candidate['id']}/p{parallel}"
            arguments.append(
                "  --quality "
                f"{label}=quality/{candidate['id']}/p{parallel}/quality-evidence.json \\"
            )
    arguments.extend(
        (
            "  --output capacity-study.reproduced.json",
            (
                "paretopilot capacity-receipt capacity-study.reproduced.json "
                "--output capacity-receipt.reproduced.md"
            ),
            "cmp --silent capacity-study.json capacity-study.reproduced.json",
            "cmp --silent capacity-receipt.md capacity-receipt.reproduced.md",
        )
    )
    return lines + arguments
