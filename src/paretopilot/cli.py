"""Command-line interface for validating and selecting benchmark candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Sequence

from paretopilot.analysis import recommend
from paretopilot.doctor import inspect_environment
from paretopilot.domain import ValidationError
from paretopilot.io import (
    load_benchmarks,
    load_constraints,
    load_json_object,
    sha256_file,
    write_json,
)
from paretopilot.llama_bench import LlamaBenchRecord, load_llama_bench_jsonl
from paretopilot.llama_compare import compare_llama_bench_summaries
from paretopilot.llama_summary import summarize_llama_bench_paths


PINNED_LLAMA_CPP_COMMIT = "67b9b0e7f6ce45d929a4411907d3c48ec719e81c"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paretopilot",
        description="Quality-aware recommendation for reproducible inference benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    recommend_parser = subparsers.add_parser(
        "recommend",
        help="filter, compute a Pareto frontier, and recommend a candidate",
    )
    recommend_parser.add_argument("results", type=Path)
    recommend_parser.add_argument("--constraints", required=True, type=Path)
    recommend_parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            report = inspect_environment()
            payload = report.to_mapping()
            if args.output:
                write_json(args.output, payload)
            exit_code = (
                3
                if args.require_evidence_host and not report.evidence_eligible
                else 0
            )
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
                "model_filenames": sorted(
                    {record.model_filename for record in records}
                ),
                "test_counts": dict(
                    sorted(Counter(record.test_kind for record in records).items())
                ),
                "repetition_counts": sorted(
                    {record.repetition_count for record in records}
                ),
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
                "artifacts_sha256": {
                    label: sha256_file(path) for label, path in artifacts
                },
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
        elif args.command == "recommend":
            benchmarks = load_benchmarks(args.results)
            constraints = load_constraints(args.constraints)
            payload = dict(recommend(benchmarks, constraints))
            payload["input_fingerprints"] = {
                "benchmarks_sha256": sha256_file(args.results),
                "constraints_sha256": sha256_file(args.constraints),
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


def _positive_cli_int(value: str) -> int:
    """Parse a strictly positive benchmark setting for argparse."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
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
