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
from paretopilot.io import load_benchmarks, load_constraints, sha256_file, write_json
from paretopilot.llama_bench import LlamaBenchRecord, load_llama_bench_jsonl


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
        elif args.command == "validate-llama-bench":
            records = load_llama_bench_jsonl(args.input)
            evidence_issues = _llama_evidence_issues(
                records,
                expected_commit=args.expected_commit,
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
                "synthetic_fixture": any(record.synthetic_fixture for record in records),
            }
            if args.output:
                write_json(args.output, payload)
            exit_code = 4 if args.evidence and evidence_issues else 0
        else:
            benchmarks = load_benchmarks(args.results)
        if args.command == "validate":
            payload = {
                "valid": True,
                "schema_version": benchmarks.schema_version,
                "baseline_id": benchmarks.baseline_id,
                "candidate_count": len(benchmarks.candidates),
                "synthetic": benchmarks.synthetic,
                "input_sha256": sha256_file(args.results),
            }
            exit_code = 0
        elif args.command == "recommend":
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


def _commits_match(actual: str, expected: str) -> bool:
    """Accept the full pinned SHA or the short prefix emitted by some builds."""

    if len(actual) < 7 or len(expected) < 7:
        return False
    return actual == expected or actual.startswith(expected) or expected.startswith(actual)


def _llama_evidence_issues(
    records: Sequence[LlamaBenchRecord], *, expected_commit: str
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
    return issues


if __name__ == "__main__":
    raise SystemExit(main())
