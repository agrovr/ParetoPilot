"""Command-line interface for validating and selecting benchmark candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from paretopilot import __version__
from paretopilot.analysis import recommend
from paretopilot.capacity_eval import assemble_capacity_study, bind_capacity_quality
from paretopilot.capacity_receipt import render_capacity_receipt
from paretopilot.capacity_replay import replay_capacity_bundle
from paretopilot.decision_passport import build_decision_passport
from paretopilot.doctor import inspect_environment
from paretopilot.domain import BenchmarkSet, Constraints, ValidationError
from paretopilot.experiment import assemble_experiment
from paretopilot.io import (
    load_benchmarks,
    load_benchmarks_snapshot,
    load_constraints,
    load_constraints_snapshot,
    load_json_object,
    sha256_file,
    write_json,
    write_text,
)
from paretopilot.launch_kit import create_launch_kit
from paretopilot.llama_bench import LlamaBenchRecord, load_llama_bench_jsonl
from paretopilot.llama_compare import compare_llama_bench_summaries
from paretopilot.llama_summary import summarize_llama_bench_paths
from paretopilot.load_eval import (
    build_load_evidence_binding,
    combine_load_evaluations,
    evaluate_llama_server_load,
    load_load_plan,
)
from paretopilot.optimization_receipt import render_optimization_receipt
from paretopilot.pass_eval import assemble_repeat_pass
from paretopilot.profiles import evaluate_policy_profiles, load_policy_set
from paretopilot.published_proof import verify_published_evidence
from paretopilot.replay import replay_evidence
from paretopilot.report import render_report
from paretopilot.report_v11 import render_report_v11
from paretopilot.server_eval import (
    evaluate_server,
    parse_gnu_time_peak_rss,
    pool_server_evaluations,
)
from paretopilot.showcase import render_showcase_v11
from paretopilot.stability import summarize_stability
from paretopilot.study import assemble_study


PINNED_LLAMA_CPP_COMMIT = "67b9b0e7f6ce45d929a4411907d3c48ec719e81c"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paretopilot",
        description="Quality-aware recommendation for reproducible inference benchmarks.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init",
        help="create a new synthetic Launch Kit for the ParetoPilot decision gate",
    )
    init_parser.add_argument(
        "directory",
        type=Path,
        help="new directory to create; existing paths are never overwritten",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="inspect whether this host can produce Arm64 benchmark evidence",
    )
    doctor_parser.add_argument(
        "--require-evidence-host",
        "--require-arm64",
        dest="require_evidence_host",
        action="store_true",
        help="return a nonzero status unless this is a native Arm64 Linux host",
    )
    doctor_parser.add_argument("--output", type=Path)

    ci_gate_parser = subparsers.add_parser(
        "ci-gate",
        help="generate a decision receipt and enforce it in continuous integration",
    )
    ci_gate_parser.add_argument("results", type=Path)
    ci_gate_parser.add_argument("--constraints", required=True, type=Path)
    ci_gate_parser.add_argument("--output-dir", required=True, type=Path)
    ci_gate_parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="allow explicitly synthetic benchmark inputs for smoke testing",
    )
    ci_gate_parser.add_argument(
        "--expect-selected-id",
        help="return a nonzero status unless this candidate is selected",
    )
    ci_gate_parser.add_argument(
        "--require-arm64-provenance",
        action="store_true",
        help=(
            "return a nonzero status unless the passport has complete source-declared "
            "Arm64 attribution metadata"
        ),
    )

    passport_parser = subparsers.add_parser(
        "passport",
        help="export supplementary Arm64 provenance and optimization-stage context",
    )
    passport_parser.add_argument("results", type=Path)
    passport_parser.add_argument("--constraints", required=True, type=Path)
    passport_parser.add_argument("--output", required=True, type=Path)
    passport_parser.add_argument(
        "--require-arm64-provenance",
        action="store_true",
        help=(
            "return a nonzero status unless the passport has complete source-declared "
            "Arm64 attribution metadata"
        ),
    )

    optimization_receipt_parser = subparsers.add_parser(
        "optimization-receipt",
        help="export a deterministic Markdown optimization proof from the decision passport",
    )
    optimization_receipt_parser.add_argument("results", type=Path)
    optimization_receipt_parser.add_argument("--constraints", required=True, type=Path)
    optimization_receipt_parser.add_argument("--output", required=True, type=Path)
    optimization_receipt_parser.add_argument(
        "--require-arm64-provenance",
        action="store_true",
        help=(
            "return a nonzero status unless the receipt has complete source-declared "
            "Arm64 attribution metadata"
        ),
    )

    validate_parser = subparsers.add_parser("validate", help="validate a benchmark result file")
    validate_parser.add_argument("results", type=Path)

    llama_parser = subparsers.add_parser(
        "validate-llama-bench",
        help="validate upstream llama-bench JSONL and summarize its records",
    )
    llama_parser.add_argument("input", type=Path)
    llama_parser.add_argument("--output", type=Path)
    llama_parser.add_argument(
        "--evidence",
        action="store_true",
        help="return nonzero unless the file meets the current evidence gates",
    )
    llama_parser.add_argument(
        "--expected-commit",
        default=PINNED_LLAMA_CPP_COMMIT,
        help="expected pinned llama.cpp commit for evidence validation",
    )
    llama_parser.add_argument(
        "--expected-threads",
        type=_positive_cli_int,
        help="expected n_threads value for evidence validation",
    )
    llama_parser.add_argument(
        "--expected-batch",
        type=_positive_cli_int,
        help="expected n_batch value for evidence validation",
    )
    llama_parser.add_argument(
        "--expected-ubatch",
        type=_positive_cli_int,
        help="expected n_ubatch value for evidence validation",
    )

    summarize_parser = subparsers.add_parser(
        "summarize-llama-bench",
        help="pool compatible labeled llama-bench artifacts into one variant summary",
    )
    summarize_parser.add_argument("--label", required=True)
    summarize_parser.add_argument(
        "--artifact",
        dest="artifacts",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="labeled JSONL artifact; repeat for each benchmark pass",
    )
    summarize_parser.add_argument("--settings", required=True, type=Path)
    summarize_parser.add_argument("--output", required=True, type=Path)

    compare_parser = subparsers.add_parser(
        "compare-llama-bench",
        help="compare compatible generic and KleidiAI variant summaries",
    )
    compare_parser.add_argument("--generic", required=True, type=Path)
    compare_parser.add_argument("--kleidiai", required=True, type=Path)
    compare_parser.add_argument("--output", required=True, type=Path)

    evaluate_parser = subparsers.add_parser(
        "evaluate-server",
        help="run the fixed quality and latency suite against llama-server",
    )
    evaluate_parser.add_argument("--base-url", required=True)
    evaluate_parser.add_argument("--suite", required=True, type=Path)
    evaluate_parser.add_argument("--candidate-id", required=True)
    evaluate_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    evaluate_parser.add_argument("--output", required=True, type=Path)

    bind_quality_parser = subparsers.add_parser(
        "bind-capacity-quality",
        help="bind one quality evaluation to its exact capacity server and workflow run",
    )
    bind_quality_parser.add_argument("--evaluation", required=True, type=Path)
    bind_quality_parser.add_argument("--server-command", required=True, type=Path)
    bind_quality_parser.add_argument("--base-url", required=True)
    bind_quality_parser.add_argument("--pass-id", required=True)
    bind_quality_parser.add_argument("--candidate-id", required=True)
    bind_quality_parser.add_argument(
        "--server-parallel",
        required=True,
        type=_positive_cli_int,
    )
    bind_quality_parser.add_argument("--run-id", required=True)
    bind_quality_parser.add_argument(
        "--run-attempt",
        required=True,
        type=_positive_cli_int,
    )
    bind_quality_parser.add_argument("--output", required=True, type=Path)

    pool_server_parser = subparsers.add_parser(
        "pool-server-evaluations",
        help="pool compatible mirrored-pass llama-server evaluation artifacts",
    )
    pool_server_parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        type=Path,
        help="raw server-evaluation JSON artifact; repeat from 2 to 8 times",
    )
    pool_server_parser.add_argument("--output", required=True, type=Path)

    evaluate_load_parser = subparsers.add_parser(
        "evaluate-load",
        help="run a bounded 1/2/4-client load plan against llama-server",
    )
    evaluate_load_parser.add_argument("--base-url", required=True)
    evaluate_load_parser.add_argument("--candidate-id", required=True)
    evaluate_load_parser.add_argument("--plan", required=True, type=Path)
    evaluate_load_parser.add_argument(
        "--server-command",
        required=True,
        type=Path,
        help="exact server command used for the load sweep",
    )
    evaluate_load_parser.add_argument(
        "--canonical-server-command",
        required=True,
        type=Path,
        help="canonical deployment command used for material-equivalence validation",
    )
    evaluate_load_parser.add_argument(
        "--execution-order",
        type=_concurrency_execution_order,
        help=(
            "optional comma-separated permutation of the plan's client levels; "
            "rows remain serialized in canonical order"
        ),
    )
    evaluate_load_parser.add_argument("--output", required=True, type=Path)

    combine_load_parser = subparsers.add_parser(
        "combine-load",
        help="combine compatible candidate load sweeps for report rendering",
    )
    combine_load_parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        type=Path,
        help="candidate load-evaluation JSON; repeat from 2 to 32 times",
    )
    combine_load_parser.add_argument("--output", required=True, type=Path)

    capacity_parser = subparsers.add_parser(
        "assemble-capacity",
        help="assemble a strict supplementary server-slots by client-concurrency study",
    )
    capacity_parser.add_argument("--plan", required=True, type=Path)
    capacity_parser.add_argument("--load-plan", required=True, type=Path)
    capacity_parser.add_argument("--manifest", required=True, type=Path)
    capacity_parser.add_argument(
        "--load",
        dest="capacity_loads",
        action="append",
        required=True,
        metavar="PASS/CANDIDATE/PARALLEL=PATH",
        help="labeled measured load artifact; repeat for every pass and server level",
    )
    capacity_parser.add_argument(
        "--rss",
        dest="capacity_rss",
        action="append",
        required=True,
        metavar="PASS/CANDIDATE/PARALLEL=PATH",
        help="labeled GNU time -v server output; repeat for every pass and server level",
    )
    capacity_parser.add_argument(
        "--server-log",
        dest="capacity_logs",
        action="append",
        required=True,
        metavar="PASS/CANDIDATE/PARALLEL=PATH",
        help="labeled server log used to verify the KleidiAI model-buffer marker",
    )
    capacity_parser.add_argument(
        "--quality",
        dest="capacity_quality",
        action="append",
        required=True,
        metavar="CANDIDATE/PARALLEL=PATH",
        help="labeled source-bound quality wrapper; repeat for every candidate and server level",
    )
    capacity_parser.add_argument("--output", required=True, type=Path)

    capacity_receipt_parser = subparsers.add_parser(
        "capacity-receipt",
        help="render a deterministic Markdown receipt from a validated capacity study",
    )
    capacity_receipt_parser.add_argument("study", type=Path)
    capacity_receipt_parser.add_argument("--output", required=True, type=Path)

    stability_parser = subparsers.add_parser(
        "summarize-stability",
        help="summarize observed directions across validated benchmark passes",
    )
    stability_parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="labeled benchmark-set JSON; repeat from 2 to 8 times",
    )
    stability_parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        required=True,
        metavar="NAME=min|max",
        help="metric direction; repeat for each summarized metric",
    )
    stability_parser.add_argument("--output", required=True, type=Path)

    repeat_pass_parser = subparsers.add_parser(
        "assemble-repeat-pass",
        help="rebuild one pass in a mirrored pair from raw experiment evidence",
    )
    repeat_pass_parser.add_argument("--experiment", required=True, type=Path)
    repeat_pass_parser.add_argument(
        "--pass-number",
        required=True,
        type=int,
        choices=(1, 2),
    )
    repeat_pass_parser.add_argument("--output", required=True, type=Path)

    rss_parser = subparsers.add_parser(
        "parse-peak-rss",
        help="convert GNU time -v maximum RSS output into a strict JSON artifact",
    )
    rss_parser.add_argument("input", type=Path)
    rss_parser.add_argument("--candidate-id", required=True)
    rss_parser.add_argument("--output", required=True, type=Path)

    study_parser = subparsers.add_parser(
        "assemble-study",
        help="verify a published paired Arm64 bundle and create real selection inputs",
    )
    study_parser.add_argument("bundle", type=Path)
    study_parser.add_argument("--practical-effect-threshold-percent", type=float, default=1.0)
    study_parser.add_argument("--benchmarks-output", required=True, type=Path)
    study_parser.add_argument("--constraints-output", required=True, type=Path)
    study_parser.add_argument("--assembly-output", type=Path)

    verify_study_parser = subparsers.add_parser(
        "verify-study",
        help="verify a published paired Arm64 bundle without writing files",
    )
    verify_study_parser.add_argument("bundle", type=Path)
    verify_study_parser.add_argument(
        "--practical-effect-threshold-percent", type=float, default=1.0
    )

    report_parser = subparsers.add_parser(
        "report",
        help="generate a deterministic self-contained HTML decision report",
    )
    report_parser.add_argument("results", type=Path)
    report_parser.add_argument("--constraints", required=True, type=Path)
    report_parser.add_argument("--output", required=True, type=Path)
    report_parser.add_argument("--recommendation-output", type=Path)

    report_v11_parser = subparsers.add_parser(
        "report-v11",
        help="render the additive v1.1 decision, policy, load, and stability report",
    )
    report_v11_parser.add_argument("results", type=Path)
    report_v11_parser.add_argument("--recommendation", required=True, type=Path)
    report_v11_parser.add_argument("--profiles", type=Path)
    report_v11_parser.add_argument("--load", type=Path)
    report_v11_parser.add_argument("--stability", type=Path)
    report_v11_parser.add_argument("--output", required=True, type=Path)

    showcase_v11_parser = subparsers.add_parser(
        "showcase-v11",
        help="render a judge-facing v1.1 presentation, locked when proof inputs are supplied",
    )
    showcase_v11_parser.add_argument("results", type=Path)
    showcase_v11_parser.add_argument("--recommendation", required=True, type=Path)
    showcase_v11_parser.add_argument("--profiles", type=Path)
    showcase_v11_parser.add_argument("--load", type=Path)
    showcase_v11_parser.add_argument("--stability", type=Path)
    showcase_v11_parser.add_argument("--evidence-lock", type=Path)
    showcase_v11_parser.add_argument("--canonical-report", type=Path)
    showcase_v11_parser.add_argument(
        "--canonical-report-href",
        default="evidence/report-v1.1.html",
    )
    showcase_v11_parser.add_argument("--capacity-study", type=Path)
    showcase_v11_parser.add_argument("--capacity-evidence-lock", type=Path)
    showcase_v11_parser.add_argument(
        "--capacity-study-href",
        default="evidence/capacity-study.json",
    )
    showcase_v11_parser.add_argument(
        "--capacity-receipt-href",
        default="evidence/capacity-receipt.md",
    )
    showcase_v11_parser.add_argument("--output", required=True, type=Path)

    profiles_parser = subparsers.add_parser(
        "profiles",
        help="precompute canonical and derived deployment-policy recommendations",
    )
    profiles_parser.add_argument("results", type=Path)
    profiles_parser.add_argument("--constraints", required=True, type=Path)
    profiles_parser.add_argument("--policies", required=True, type=Path)
    profiles_parser.add_argument("--output", required=True, type=Path)

    replay_parser = subparsers.add_parser(
        "replay",
        help="verify and regenerate outputs from an extracted evidence directory",
    )
    replay_parser.add_argument("evidence", type=Path)
    replay_parser.add_argument("--output-dir", required=True, type=Path)
    replay_parser.add_argument(
        "--policies",
        type=Path,
        help="optionally precompute policy profiles from this configuration",
    )

    replay_capacity_parser = subparsers.add_parser(
        "replay-capacity",
        help="verify and reproduce an extracted supplementary capacity bundle",
    )
    replay_capacity_parser.add_argument("bundle", type=Path)
    replay_capacity_parser.add_argument("--output-dir", required=True, type=Path)

    verify_published_parser = subparsers.add_parser(
        "verify-published",
        help="verify and replay the pinned canonical and capacity release archives",
        description=(
            "Verify both pinned public Arm64 evidence releases on any supported host. "
            "By default, ParetoPilot downloads both exact archives, checks their size and "
            "SHA-256 pins, safely extracts them, and replays the archived decisions."
        ),
        epilog=(
            "This does not require an Arm64 machine and does not rerun inference. "
            "Supply both archive options for a fully offline check. The output directory "
            "must not already exist."
        ),
    )
    verify_published_parser.add_argument(
        "--canonical-archive",
        type=Path,
        help="exact pinned v1.1 archive to use instead of downloading it",
    )
    verify_published_parser.add_argument(
        "--capacity-archive",
        type=Path,
        help="exact pinned v1.4 archive to use instead of downloading it",
    )
    verify_published_parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new directory for the human-readable Markdown and machine-readable JSON proofs",
    )

    experiment_parser = subparsers.add_parser(
        "assemble-experiment",
        help="verify a multi-candidate Arm64 manifest and create a real benchmark set",
    )
    experiment_parser.add_argument("manifest", type=Path)
    experiment_parser.add_argument("--output", required=True, type=Path)

    recommend_parser = subparsers.add_parser(
        "recommend",
        help="filter, compute a Pareto frontier, and recommend a candidate",
    )
    recommend_parser.add_argument("results", type=Path)
    recommend_parser.add_argument("--constraints", required=True, type=Path)
    recommend_parser.add_argument("--output", type=Path)
    return parser


def _build_decision_artifacts(
    results_path: Path,
    constraints_path: Path,
) -> tuple[BenchmarkSet, Constraints, dict[str, Any], str]:
    """Build one recommendation and report from the same validated inputs."""

    benchmarks, benchmarks_sha256 = load_benchmarks_snapshot(results_path)
    constraints, constraints_sha256 = load_constraints_snapshot(constraints_path)
    recommendation = dict(recommend(benchmarks, constraints))
    recommendation["input_fingerprints"] = {
        "benchmarks_sha256": benchmarks_sha256,
        "constraints_sha256": constraints_sha256,
    }
    report_html = render_report(
        benchmarks,
        constraints,
        recommendation,
        benchmarks_sha256=recommendation["input_fingerprints"]["benchmarks_sha256"],
        constraints_sha256=recommendation["input_fingerprints"]["constraints_sha256"],
    )
    return benchmarks, constraints, recommendation, report_html


def _build_passport_artifact(
    benchmarks: BenchmarkSet,
    constraints: Constraints,
    *,
    input_fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    """Bind a deterministic passport to the exact serialized CI inputs."""

    passport = dict(build_decision_passport(benchmarks, constraints))
    passport["input_fingerprints"] = dict(input_fingerprints)
    return passport


def _require_arm64_attribution(passport: Mapping[str, Any]) -> None:
    if passport.get("evidence_grade") == "arm64-attributed":
        return
    provenance = passport.get("provenance")
    issues: list[str] = []
    if isinstance(provenance, Mapping):
        raw_issues = provenance.get("issues")
        if isinstance(raw_issues, Sequence) and not isinstance(raw_issues, (str, bytes, bytearray)):
            issues = [str(issue) for issue in raw_issues]
    detail = f": {'; '.join(issues)}" if issues else ""
    raise ValidationError(
        "complete source-declared Arm64 attribution metadata is required; "
        f"decision passport grade is {passport.get('evidence_grade')!r}{detail}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            payload = create_launch_kit(args.directory)
            exit_code = 0
        elif args.command == "doctor":
            report = inspect_environment()
            payload = report.to_mapping()
            if args.output:
                write_json(args.output, payload)
            exit_code = 3 if args.require_evidence_host and not report.evidence_eligible else 0
        elif args.command == "ci-gate":
            benchmarks, constraints, recommendation, report_html = _build_decision_artifacts(
                args.results,
                args.constraints,
            )
            passport = _build_passport_artifact(
                benchmarks,
                constraints,
                input_fingerprints=recommendation["input_fingerprints"],
            )
            if benchmarks.synthetic and not args.allow_synthetic:
                raise ValidationError(
                    "ci-gate requires measured evidence; use --allow-synthetic only "
                    "for an explicit smoke test"
                )
            if args.require_arm64_provenance:
                _require_arm64_attribution(passport)
            expected_selected_id = args.expect_selected_id
            if expected_selected_id is not None:
                if not expected_selected_id.strip():
                    raise ValidationError("expect-selected-id must be a non-empty candidate id")
                if recommendation["selected_id"] != expected_selected_id:
                    raise ValidationError(
                        "selected candidate does not match the CI expectation: "
                        f"expected {expected_selected_id!r}, "
                        f"got {recommendation['selected_id']!r}"
                    )

            output_dir = args.output_dir
            recommendation_path = output_dir / "recommendation.json"
            report_path = output_dir / "report.html"
            passport_path = output_dir / "decision-passport.json"
            optimization_receipt_path = output_dir / "optimization-receipt.md"
            gate_receipt_path = output_dir / "gate.json"
            destinations = [
                recommendation_path,
                report_path,
                passport_path,
                optimization_receipt_path,
                gate_receipt_path,
            ]
            _require_new_distinct_outputs(destinations)
            _require_new_or_empty_directory(output_dir)

            optimization_receipt = render_optimization_receipt(passport)
            write_json(recommendation_path, recommendation)
            write_text(report_path, report_html)
            write_json(passport_path, passport)
            write_text(optimization_receipt_path, optimization_receipt)
            receipt = {
                "schema_version": "1.2",
                "valid": True,
                "selected_id": recommendation["selected_id"],
                "baseline_id": recommendation["baseline_id"],
                "expected_selected_id": expected_selected_id,
                "expectation_matched": (
                    expected_selected_id is None
                    or recommendation["selected_id"] == expected_selected_id
                ),
                "synthetic_source": benchmarks.synthetic,
                "candidate_count": len(benchmarks.candidates),
                "eligible_count": len(recommendation["eligible_ids"]),
                "frontier_count": len(recommendation["frontier_ids"]),
                "recommendation": str(recommendation_path),
                "recommendation_sha256": sha256_file(recommendation_path),
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "decision_passport": str(passport_path),
                "decision_passport_sha256": sha256_file(passport_path),
                "optimization_receipt": str(optimization_receipt_path),
                "optimization_receipt_sha256": sha256_file(optimization_receipt_path),
                "evidence_grade": passport["evidence_grade"],
            }
            write_json(gate_receipt_path, receipt)
            payload = {
                **receipt,
                "receipt": str(gate_receipt_path),
                "receipt_sha256": sha256_file(gate_receipt_path),
            }
            exit_code = 0
        elif args.command == "passport":
            _require_new_distinct_outputs([args.output])
            benchmarks, benchmarks_sha256 = load_benchmarks_snapshot(args.results)
            constraints, constraints_sha256 = load_constraints_snapshot(args.constraints)
            passport = _build_passport_artifact(
                benchmarks,
                constraints,
                input_fingerprints={
                    "benchmarks_sha256": benchmarks_sha256,
                    "constraints_sha256": constraints_sha256,
                },
            )
            if args.require_arm64_provenance:
                _require_arm64_attribution(passport)
            write_json(args.output, passport)
            payload = {
                "valid": True,
                "evidence_grade": passport["evidence_grade"],
                "selected_id": passport["selected_decision"]["candidate_id"],
                "passport": str(args.output),
                "passport_sha256": sha256_file(args.output),
            }
            exit_code = 0
        elif args.command == "optimization-receipt":
            _require_new_distinct_outputs([args.output])
            benchmarks, benchmarks_sha256 = load_benchmarks_snapshot(args.results)
            constraints, constraints_sha256 = load_constraints_snapshot(args.constraints)
            passport = _build_passport_artifact(
                benchmarks,
                constraints,
                input_fingerprints={
                    "benchmarks_sha256": benchmarks_sha256,
                    "constraints_sha256": constraints_sha256,
                },
            )
            if args.require_arm64_provenance:
                _require_arm64_attribution(passport)
            write_text(args.output, render_optimization_receipt(passport))
            payload = {
                "valid": True,
                "evidence_grade": passport["evidence_grade"],
                "selected_id": passport["selected_decision"]["candidate_id"],
                "optimization_receipt": str(args.output),
                "optimization_receipt_sha256": sha256_file(args.output),
            }
            exit_code = 0
        elif args.command == "validate":
            benchmarks = load_benchmarks(args.results)
            payload = {
                "valid": True,
                "schema_version": benchmarks.schema_version,
                "baseline_id": benchmarks.baseline_id,
                "candidate_count": len(benchmarks.candidates),
                "synthetic": benchmarks.synthetic,
                "input_sha256": sha256_file(args.results),
            }
            exit_code = 0
        elif args.command == "validate-llama-bench":
            records = load_llama_bench_jsonl(args.input)
            evidence_issues = _llama_evidence_issues(
                records,
                expected_commit=args.expected_commit,
                expected_threads=args.expected_threads,
                expected_batch=args.expected_batch,
                expected_ubatch=args.expected_ubatch,
            )
            payload = {
                "valid": True,
                "evidence_valid": not evidence_issues,
                "evidence_issues": evidence_issues,
                "record_count": len(records),
                "input_sha256": sha256_file(args.input),
                "build_commits": sorted({record.build_commit for record in records}),
                "model_filenames": sorted({record.model_filename for record in records}),
                "test_counts": dict(
                    sorted(Counter(record.test_kind for record in records).items())
                ),
                "repetition_counts": sorted({record.repetition_count for record in records}),
                "execution_settings": {
                    "n_threads": _declared_values(records, "n_threads"),
                    "n_batch": _declared_values(records, "n_batch"),
                    "n_ubatch": _declared_values(records, "n_ubatch"),
                    "n_gpu_layers": _declared_values(records, "n_gpu_layers"),
                    "devices": _declared_values(records, "devices"),
                    "no_op_offload": _declared_values(records, "no_op_offload"),
                },
                "synthetic_fixture": any(record.synthetic_fixture for record in records),
            }
            if args.output:
                write_json(args.output, payload)
            exit_code = 4 if args.evidence and evidence_issues else 0
        elif args.command == "summarize-llama-bench":
            settings = load_json_object(args.settings)
            artifacts = _parse_labeled_artifacts(args.artifacts)
            summary = summarize_llama_bench_paths(
                args.label,
                artifacts,
                settings=settings,
            )
            payload = summary.to_mapping()
            payload["input_fingerprints"] = {
                "settings_sha256": sha256_file(args.settings),
                "artifacts_sha256": {label: sha256_file(path) for label, path in artifacts},
            }
            write_json(args.output, payload)
            exit_code = 0
        elif args.command == "compare-llama-bench":
            generic = load_json_object(args.generic)
            kleidiai = load_json_object(args.kleidiai)
            comparison = compare_llama_bench_summaries(generic, kleidiai)
            payload = comparison.to_mapping()
            payload["input_fingerprints"] = {
                "generic_summary_sha256": sha256_file(args.generic),
                "kleidiai_summary_sha256": sha256_file(args.kleidiai),
            }
            write_json(args.output, payload)
            exit_code = 0
        elif args.command == "evaluate-server":
            payload = evaluate_server(
                args.base_url,
                args.suite,
                candidate_id=args.candidate_id,
                timeout_seconds=args.timeout_seconds,
            )
            write_json(args.output, payload)
            exit_code = 0
        elif args.command == "bind-capacity-quality":
            _require_new_distinct_outputs([args.output])
            payload = bind_capacity_quality(
                evaluation_path=args.evaluation,
                server_command_path=args.server_command,
                base_url=args.base_url,
                pass_id=args.pass_id,
                candidate_id=args.candidate_id,
                server_parallel=args.server_parallel,
                run_id=args.run_id,
                run_attempt=args.run_attempt,
            )
            write_json(args.output, payload)
            exit_code = 0
        elif args.command == "pool-server-evaluations":
            payload = pool_server_evaluations(args.inputs)
            write_json(args.output, payload)
            exit_code = 0
        elif args.command == "evaluate-load":
            _require_new_distinct_outputs([args.output])
            plan = load_load_plan(args.plan)
            evidence_binding = build_load_evidence_binding(
                base_url=args.base_url,
                plan_path=args.plan,
                server_command_path=args.server_command,
                canonical_server_command_path=args.canonical_server_command,
            )
            payload = evaluate_llama_server_load(
                args.base_url,
                plan,
                candidate_id=args.candidate_id,
                evidence_binding=evidence_binding,
                execution_order=args.execution_order,
            )
            write_json(args.output, payload)
            exit_code = 0
        elif args.command == "combine-load":
            _require_new_distinct_outputs([args.output])
            evaluations = [load_json_object(path) for path in args.inputs]
            payload = combine_load_evaluations(
                evaluations,
                require_evidence_bindings=True,
            )
            write_json(args.output, payload)
            exit_code = 0
        elif args.command == "assemble-capacity":
            _require_new_distinct_outputs([args.output])
            payload = assemble_capacity_study(
                plan_path=args.plan,
                load_plan_path=args.load_plan,
                manifest_path=args.manifest,
                load_artifacts=_parse_labeled_artifacts(args.capacity_loads),
                rss_artifacts=_parse_labeled_artifacts(args.capacity_rss),
                server_logs=_parse_labeled_artifacts(args.capacity_logs),
                quality_artifacts=_parse_labeled_artifacts(args.capacity_quality),
            )
            write_json(args.output, payload)
            exit_code = 0
        elif args.command == "capacity-receipt":
            _require_new_distinct_outputs([args.output])
            study = load_json_object(args.study)
            write_text(args.output, render_capacity_receipt(study))
            payload = {
                "classification": study.get("classification"),
                "capacity_receipt": str(args.output),
                "capacity_receipt_sha256": sha256_file(args.output),
            }
            exit_code = 0
        elif args.command == "summarize-stability":
            _require_new_distinct_outputs([args.output])
            artifacts = _parse_labeled_artifacts(args.inputs)
            directions = _parse_metric_directions(args.metrics)
            payload = summarize_stability(
                [load_benchmarks(path) for _, path in artifacts],
                metric_directions=directions,
                pass_labels=[label for label, _ in artifacts],
                input_fingerprints={label: sha256_file(path) for label, path in artifacts},
            )
            write_json(args.output, payload)
            exit_code = 0
        elif args.command == "assemble-repeat-pass":
            _require_new_distinct_outputs([args.output])
            payload = assemble_repeat_pass(
                args.experiment,
                pass_number=args.pass_number,
            )
            write_json(args.output, payload)
            exit_code = 0
        elif args.command == "parse-peak-rss":
            if not args.candidate_id.strip():
                raise ValidationError("candidate-id must be a non-empty string")
            payload = {
                "schema_version": "1.0",
                "candidate_id": args.candidate_id,
                "synthetic": False,
                "peak_rss_mib": parse_gnu_time_peak_rss(args.input),
                "source_sha256": sha256_file(args.input),
                "method": "GNU time -v maximum resident set size",
            }
            write_json(args.output, payload)
            exit_code = 0
        elif args.command == "assemble-study":
            destinations = [args.benchmarks_output, args.constraints_output]
            if args.assembly_output is not None:
                destinations.append(args.assembly_output)
            _require_new_distinct_outputs(destinations)
            assembly = assemble_study(
                args.bundle,
                practical_effect_threshold_percent=(args.practical_effect_threshold_percent),
            )
            payload = assembly.to_mapping()
            write_json(args.benchmarks_output, assembly.benchmark_set)
            write_json(args.constraints_output, assembly.constraints)
            if args.assembly_output is not None:
                write_json(args.assembly_output, payload)
            exit_code = 0
        elif args.command == "verify-study":
            assembly = assemble_study(
                args.bundle,
                practical_effect_threshold_percent=(args.practical_effect_threshold_percent),
            )
            benchmarks = BenchmarkSet.from_mapping(assembly.benchmark_set)
            constraints = Constraints.from_mapping(assembly.constraints)
            decision = recommend(benchmarks, constraints)
            payload = {
                "valid": True,
                "bundle": str(args.bundle),
                "candidate_count": len(benchmarks.candidates),
                "selected_id": decision["selected_id"],
                "adoption_gate": dict(assembly.adoption_gate),
            }
            exit_code = 0
        elif args.command == "report":
            destinations = [args.output]
            if args.recommendation_output is not None:
                destinations.append(args.recommendation_output)
            _require_new_distinct_outputs(destinations)
            benchmarks, _, recommendation, report_html = _build_decision_artifacts(
                args.results,
                args.constraints,
            )
            write_text(args.output, report_html)
            if args.recommendation_output is not None:
                write_json(args.recommendation_output, recommendation)
            payload = {
                "valid": True,
                "selected_id": recommendation["selected_id"],
                "baseline_id": recommendation["baseline_id"],
                "synthetic_source": recommendation["synthetic_source"],
                "report": str(args.output),
                "report_sha256": sha256_file(args.output),
                "recommendation": (
                    str(args.recommendation_output)
                    if args.recommendation_output is not None
                    else None
                ),
            }
            exit_code = 0
        elif args.command == "report-v11":
            _require_new_distinct_outputs([args.output])
            benchmarks = load_benchmarks(args.results)
            recommendation = load_json_object(args.recommendation)
            policy_profiles = load_json_object(args.profiles) if args.profiles is not None else None
            load_sweep = load_json_object(args.load) if args.load is not None else None
            stability_summary = (
                load_json_object(args.stability) if args.stability is not None else None
            )
            report_html = render_report_v11(
                benchmarks,
                recommendation,
                policy_profiles=policy_profiles,
                load_sweep=load_sweep,
                stability_summary=stability_summary,
                benchmarks_sha256=sha256_file(args.results),
                recommendation_sha256=sha256_file(args.recommendation),
                profiles_sha256=sha256_file(args.profiles) if args.profiles is not None else "",
                load_sha256=sha256_file(args.load) if args.load is not None else "",
                stability_sha256=(
                    sha256_file(args.stability) if args.stability is not None else ""
                ),
            )
            write_text(args.output, report_html)
            payload = {
                "valid": True,
                "selected_id": recommendation.get("selected_id"),
                "baseline_id": benchmarks.baseline_id,
                "synthetic_source": benchmarks.synthetic,
                "policy_profiles_supplied": policy_profiles is not None,
                "load_sweep_supplied": load_sweep is not None,
                "stability_summary_supplied": stability_summary is not None,
                "report": str(args.output),
                "report_sha256": sha256_file(args.output),
            }
            exit_code = 0
        elif args.command == "showcase-v11":
            _require_new_distinct_outputs([args.output])
            benchmarks = load_benchmarks(args.results)
            recommendation = load_json_object(args.recommendation)
            policy_profiles = load_json_object(args.profiles) if args.profiles is not None else None
            load_sweep = load_json_object(args.load) if args.load is not None else None
            stability_summary = (
                load_json_object(args.stability) if args.stability is not None else None
            )
            evidence_lock = (
                load_json_object(args.evidence_lock) if args.evidence_lock is not None else None
            )
            capacity_study = (
                load_json_object(args.capacity_study) if args.capacity_study is not None else None
            )
            capacity_evidence_lock = (
                load_json_object(args.capacity_evidence_lock)
                if args.capacity_evidence_lock is not None
                else None
            )
            canonical_html = None
            if args.canonical_report is not None:
                try:
                    canonical_bytes = args.canonical_report.read_bytes()
                    canonical_html = canonical_bytes.decode("utf-8")
                except OSError as exc:
                    raise ValidationError(
                        f"could not read canonical report: {args.canonical_report}"
                    ) from exc
                except UnicodeDecodeError as exc:
                    raise ValidationError("canonical report must be valid UTF-8") from exc
            report_html = render_showcase_v11(
                benchmarks,
                recommendation,
                policy_profiles=policy_profiles,
                load_sweep=load_sweep,
                stability_summary=stability_summary,
                evidence_lock=evidence_lock,
                evidence_lock_sha256=(
                    sha256_file(args.evidence_lock) if args.evidence_lock is not None else ""
                ),
                canonical_html=canonical_html,
                canonical_report_href=args.canonical_report_href,
                capacity_study=capacity_study,
                capacity_evidence_lock=capacity_evidence_lock,
                capacity_study_sha256=(
                    sha256_file(args.capacity_study) if args.capacity_study is not None else ""
                ),
                capacity_study_href=args.capacity_study_href,
                capacity_receipt_href=args.capacity_receipt_href,
                benchmarks_sha256=sha256_file(args.results),
                recommendation_sha256=sha256_file(args.recommendation),
                profiles_sha256=sha256_file(args.profiles) if args.profiles is not None else "",
                load_sha256=sha256_file(args.load) if args.load is not None else "",
                stability_sha256=(
                    sha256_file(args.stability) if args.stability is not None else ""
                ),
            )
            write_text(args.output, report_html)
            payload = {
                "valid": True,
                "presentation_view": True,
                "selected_id": recommendation.get("selected_id"),
                "baseline_id": benchmarks.baseline_id,
                "canonical_report_verified": canonical_html is not None,
                "evidence_lock_supplied": evidence_lock is not None,
                "capacity_study_supplied": capacity_study is not None,
                "capacity_evidence_lock_supplied": capacity_evidence_lock is not None,
                "report": str(args.output),
                "report_sha256": sha256_file(args.output),
            }
            exit_code = 0
        elif args.command == "profiles":
            _require_new_distinct_outputs([args.output])
            benchmarks = load_benchmarks(args.results)
            constraints = load_constraints(args.constraints)
            policy_set = load_policy_set(args.policies)
            payload = dict(evaluate_policy_profiles(benchmarks, constraints, policy_set))
            payload["input_fingerprints"] = {
                "benchmarks_sha256": sha256_file(args.results),
                "constraints_sha256": sha256_file(args.constraints),
                "policies_sha256": sha256_file(args.policies),
            }
            write_json(args.output, payload)
            exit_code = 0
        elif args.command == "replay":
            replay_result = replay_evidence(
                args.evidence,
                args.output_dir,
                policies_path=args.policies,
            )
            payload = {
                "schema_version": replay_result["schema_version"],
                "valid": replay_result["valid"],
                "verdict": replay_result["verdict"],
                "decision_reproduced": replay_result["decision_reproduced"],
                "fully_reproduced": replay_result["fully_reproduced"],
                "report_matches_archive": replay_result["report_matches_archive"],
                "selected_id": replay_result["selected_id"],
                "checksum_entries": replay_result["checksums"]["entry_count"],
                "differences": replay_result["differences"],
                "warnings": replay_result["warnings"],
                "policy_selected_ids": replay_result["policy_selected_ids"],
                "output_directory": replay_result["output_directory"],
                "details": str(Path(replay_result["output_directory"]) / "replay.json"),
            }
            exit_code = 0 if replay_result["valid"] else 5
        elif args.command == "replay-capacity":
            replay_result = replay_capacity_bundle(args.bundle, args.output_dir)
            payload = {
                "schema_version": replay_result["schema_version"],
                "classification": replay_result["classification"],
                "valid": replay_result["valid"],
                "verdict": replay_result["verdict"],
                "status_complete": replay_result["status_complete"],
                "capacity_study_reproduced": replay_result["capacity_study_reproduced"],
                "capacity_receipt_reproduced": replay_result["capacity_receipt_reproduced"],
                "checksum_entries": replay_result["checksums"]["entry_count"],
                "checksum_manifest_sha256": replay_result["checksums"]["manifest_sha256"],
                "canonical_replay": replay_result["canonical_replay"],
                "selected_operating_points": replay_result["selected_operating_points"],
                "output_directory": str(args.output_dir.resolve()),
                "details": str(args.output_dir.resolve() / "capacity-replay.json"),
            }
            exit_code = 0
        elif args.command == "verify-published":
            proof = verify_published_evidence(
                args.output_dir,
                canonical_archive=args.canonical_archive,
                capacity_archive=args.capacity_archive,
            )
            print(proof["verdict"])
            print(f"Proof: {args.output_dir.resolve() / 'published-proof.md'}")
            return 0
        elif args.command == "assemble-experiment":
            payload = assemble_experiment(args.manifest)
            write_json(args.output, payload)
            exit_code = 0
        elif args.command == "recommend":
            benchmarks, benchmarks_sha256 = load_benchmarks_snapshot(args.results)
            constraints, constraints_sha256 = load_constraints_snapshot(args.constraints)
            payload = dict(recommend(benchmarks, constraints))
            payload["input_fingerprints"] = {
                "benchmarks_sha256": benchmarks_sha256,
                "constraints_sha256": constraints_sha256,
            }
            if args.output:
                write_json(args.output, payload)
            exit_code = 0

        print(json.dumps(payload, indent=2, sort_keys=True))
        return exit_code
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _parse_labeled_artifacts(values: Sequence[str]) -> list[tuple[str, Path]]:
    """Parse repeated ``LABEL=PATH`` CLI values while preserving their order."""

    artifacts: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        label, separator, path_text = value.partition("=")
        label = label.strip()
        path_text = path_text.strip()
        if not separator or not label or not path_text:
            raise ValidationError(
                f"artifact {value!r} must use the form LABEL=PATH with non-empty values"
            )
        if label in labels:
            raise ValidationError(f"artifact label {label!r} must be unique")
        labels.add(label)
        artifacts.append((label, Path(path_text)))
    return artifacts


def _parse_metric_directions(values: Sequence[str]) -> Mapping[str, str]:
    """Parse repeated ``NAME=min|max`` values into one strict direction map."""

    directions: dict[str, str] = {}
    for value in values:
        metric, separator, direction = value.partition("=")
        metric = metric.strip()
        direction = direction.strip()
        if not separator or not metric or direction not in {"min", "max"}:
            raise ValidationError(f"metric {value!r} must use the form NAME=min or NAME=max")
        if metric in directions:
            raise ValidationError(f"metric {metric!r} must be unique")
        directions[metric] = direction
    return directions


def _require_new_distinct_outputs(paths: Sequence[Path]) -> None:
    """Fail before multi-file commands can leave a partial output set."""

    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValidationError("output paths must be distinct")
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ValidationError(
            "refusing to overwrite existing output file(s): " + ", ".join(existing)
        )


def _require_new_or_empty_directory(path: Path) -> None:
    """Require a new or empty real directory for one atomic-looking output set."""

    try:
        if not path.exists():
            return
        if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
            raise ValidationError("ci-gate output directory must be new or empty")
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(f"could not inspect ci-gate output directory {path}: {exc}") from exc


def _positive_cli_int(value: str) -> int:
    """Parse a strictly positive benchmark setting for argparse."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _concurrency_execution_order(value: str) -> tuple[int, ...]:
    """Parse a comma-separated client-concurrency execution order."""

    parts = value.split(",")
    if not parts or any(not part or part.strip() != part for part in parts):
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list without empty values or surrounding spaces"
        )
    try:
        parsed = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must contain only integers") from exc
    if any(level not in {1, 2, 4} for level in parsed):
        raise argparse.ArgumentTypeError("may contain only 1, 2, and 4")
    if len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("must not repeat a concurrency level")
    return parsed


def _declared_values(
    records: Sequence[LlamaBenchRecord], field_name: str
) -> list[int | str | None]:
    """Return deterministic distinct values, retaining missing declarations."""

    values = {getattr(record, field_name) for record in records}
    return sorted(values, key=lambda value: (value is None, str(value)))


def _commits_match(actual: str, expected: str) -> bool:
    """Accept the full pinned SHA or the short prefix emitted by some builds."""

    if len(actual) < 7 or len(expected) < 7:
        return False
    return actual == expected or actual.startswith(expected) or expected.startswith(actual)


def _llama_evidence_issues(
    records: Sequence[LlamaBenchRecord],
    *,
    expected_commit: str,
    expected_threads: int | None = None,
    expected_batch: int | None = None,
    expected_ubatch: int | None = None,
) -> list[str]:
    issues: list[str] = []
    commits = sorted({record.build_commit for record in records})
    models = sorted({record.model_filename for record in records})
    repetition_counts = sorted({record.repetition_count for record in records})

    if any(record.synthetic_fixture for record in records):
        issues.append("synthetic fixtures cannot be used as benchmark evidence")
    if len(commits) != 1:
        issues.append("all records must use one llama.cpp build commit")
    if any(not _commits_match(commit, expected_commit) for commit in commits):
        issues.append(f"records must use pinned llama.cpp commit {expected_commit}")
    if len(models) != 1:
        issues.append("all records in one artifact must use one model file")
    if len(repetition_counts) != 1:
        issues.append("all records must use the same repetition count")
    if any(count < 10 for count in repetition_counts):
        issues.append("final benchmark evidence requires at least 10 repetitions")
    if not any(record.test_kind in {"pp", "pg"} for record in records):
        issues.append("benchmark evidence must include prompt processing")
    if not any(record.test_kind in {"tg", "pg"} for record in records):
        issues.append("benchmark evidence must include token generation")

    required_settings: tuple[tuple[str, int | str | None], ...] = (
        ("n_threads", expected_threads),
        ("n_batch", expected_batch),
        ("n_ubatch", expected_ubatch),
        ("n_gpu_layers", 0),
        ("devices", "none"),
        ("no_op_offload", 1),
    )
    for field_name, expected_value in required_settings:
        values = [getattr(record, field_name) for record in records]
        if any(value is None for value in values):
            issues.append(f"all records must declare {field_name}")
            continue
        distinct_values = set(values)
        if len(distinct_values) != 1:
            issues.append(f"all records must use one {field_name} value")
        if expected_value is not None and distinct_values != {expected_value}:
            issues.append(f"records must use {field_name}={expected_value}")
    return issues


if __name__ == "__main__":
    raise SystemExit(main())
