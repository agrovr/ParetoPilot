"""Rebuild one balanced benchmark pass from checksummed raw experiment evidence."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from paretopilot.domain import BenchmarkSet, ValidationError
from paretopilot.io import load_json_object, sha256_file
from paretopilot.llama_summary import summarize_llama_bench_paths
from paretopilot.server_eval import (
    EvaluationSuite,
    load_evaluation_suite,
    parse_gnu_time_peak_rss,
    validate_server_evaluation,
)


def assemble_repeat_pass(
    experiment_dir: Path,
    *,
    pass_number: int,
    benchmark_mapping: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Rebuild one supplementary pass from the canonical experiment's raw files."""

    if isinstance(pass_number, bool) or pass_number not in {1, 2}:
        raise ValidationError("repeat pass number must be 1 or 2")
    root = _resolved_directory(experiment_dir)
    benchmark_path = root / "benchmark-set.json"
    if benchmark_mapping is None:
        benchmark_mapping = load_json_object(benchmark_path)
    benchmarks = BenchmarkSet.from_mapping(benchmark_mapping)
    metadata = benchmark_mapping.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValidationError("repeat-pass source benchmark metadata must be an object")
    evidence = metadata.get("candidate_evidence")
    if not isinstance(evidence, Mapping):
        raise ValidationError(
            "repeat-pass source benchmark must include candidate_evidence metadata"
        )
    if list(evidence) != sorted(evidence):
        raise ValidationError(
            "repeat-pass candidate_evidence metadata must use canonical candidate order"
        )

    suite_identity = metadata.get("evaluation_suite")
    if not isinstance(suite_identity, Mapping):
        raise ValidationError("repeat-pass source benchmark must include evaluation_suite metadata")
    suite_path = root / "evaluation-suite.json"
    suite = load_evaluation_suite(suite_path)
    suite_sha256 = sha256_file(suite_path)
    if suite_identity.get("id") != suite.suite_id or suite_identity.get("sha256") != suite_sha256:
        raise ValidationError(
            "repeat-pass evaluation-suite.json does not match source benchmark metadata"
        )

    pass_label = f"pass-{pass_number}"
    candidate_rows: list[Mapping[str, Any]] = []
    fingerprints: dict[str, Mapping[str, str]] = {}
    for candidate in benchmarks.candidates:
        candidate_id = candidate.candidate_id
        candidate_evidence = evidence.get(candidate_id)
        if not isinstance(candidate_evidence, Mapping):
            raise ValidationError(f"repeat-pass evidence is missing candidate {candidate_id!r}")
        refs = candidate_evidence.get("artifacts")
        if not isinstance(refs, Mapping):
            raise ValidationError(
                f"repeat-pass candidate {candidate_id!r} artifacts must be an object"
            )
        required_refs = {
            "throughput_settings",
            f"throughput_pass_{pass_number}",
            f"server_evaluation_pass_{pass_number}",
            f"server_time_pass_{pass_number}",
        }
        missing = sorted(required_refs - set(refs))
        if missing:
            raise ValidationError(
                f"repeat-pass candidate {candidate_id!r} is missing source artifacts: "
                + ", ".join(missing)
            )

        settings_path = _verified_ref(
            root,
            refs["throughput_settings"],
            context=f"{candidate_id} throughput settings",
        )
        throughput_path = _verified_ref(
            root,
            refs[f"throughput_pass_{pass_number}"],
            context=f"{candidate_id} {pass_label} throughput",
        )
        server_path = _verified_ref(
            root,
            refs[f"server_evaluation_pass_{pass_number}"],
            context=f"{candidate_id} {pass_label} server evaluation",
        )
        server_time_path = _verified_ref(
            root,
            refs[f"server_time_pass_{pass_number}"],
            context=f"{candidate_id} {pass_label} GNU time",
        )

        settings = load_json_object(settings_path)
        throughput = summarize_llama_bench_paths(
            candidate_id,
            [(pass_label, throughput_path)],
            settings=settings,
        ).to_mapping()
        server = load_json_object(server_path)
        validate_server_evaluation(server)
        _validate_server_suite(
            server,
            suite=suite,
            suite_sha256=suite_sha256,
            candidate_id=candidate_id,
        )
        peak_rss_mib = parse_gnu_time_peak_rss(server_time_path)
        tests = throughput.get("tests")
        if not isinstance(tests, Mapping):
            raise ValidationError(
                f"repeat-pass candidate {candidate_id!r} throughput tests are missing"
            )
        pp = tests.get("pp")
        tg = tests.get("tg")
        if not isinstance(pp, Mapping) or not isinstance(tg, Mapping):
            raise ValidationError(
                f"repeat-pass candidate {candidate_id!r} requires pp and tg tests"
            )
        pp_stats = pp.get("tokens_per_second")
        tg_stats = tg.get("tokens_per_second")
        if not isinstance(pp_stats, Mapping) or not isinstance(tg_stats, Mapping):
            raise ValidationError(
                f"repeat-pass candidate {candidate_id!r} token statistics are missing"
            )
        quality = server.get("quality")
        latency = server.get("latency")
        if not isinstance(quality, Mapping) or not isinstance(latency, Mapping):
            raise ValidationError(
                f"repeat-pass candidate {candidate_id!r} server metrics are missing"
            )

        candidate_rows.append(
            {
                "id": candidate_id,
                "label": candidate.label,
                "parameters": deepcopy(dict(candidate.parameters)),
                "metrics": {
                    "prompt_tps": pp_stats["median"],
                    "generation_tps": tg_stats["median"],
                    "quality_score": quality["score"],
                    "ttft_ms_p95": latency["ttft_ms_p95"],
                    "e2e_latency_ms_p95": latency["e2e_latency_ms_p95"],
                    "peak_rss_mib": peak_rss_mib,
                    "model_size_mib": candidate.metrics["model_size_mib"],
                },
            }
        )
        fingerprints[candidate_id] = {
            "throughput_settings_sha256": sha256_file(settings_path),
            "throughput_sha256": sha256_file(throughput_path),
            "server_evaluation_sha256": sha256_file(server_path),
            "server_time_sha256": sha256_file(server_time_path),
        }

    candidate_ids = [candidate.candidate_id for candidate in benchmarks.candidates]
    if [row["id"] for row in candidate_rows] != candidate_ids:
        raise ValidationError("repeat-pass candidate coverage is incomplete")
    if set(evidence) != set(candidate_ids):
        raise ValidationError(
            "repeat-pass candidate_evidence contains unknown or missing candidates"
        )
    result: Mapping[str, Any] = {
        "schema_version": benchmarks.schema_version,
        "baseline_id": benchmarks.baseline_id,
        "synthetic": benchmarks.synthetic,
        "metadata": {
            "classification": "supplementary-repeat-pass",
            "pass_label": pass_label,
            "source_benchmark_sha256": sha256_file(benchmark_path),
            "source_artifacts": fingerprints,
        },
        "candidates": candidate_rows,
    }
    BenchmarkSet.from_mapping(result)
    return result


def _resolved_directory(path: Path) -> Path:
    candidate = Path(path)
    try:
        if candidate.is_symlink():
            raise ValidationError("experiment directory must not be a symbolic link")
        resolved = candidate.resolve(strict=True)
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(f"could not resolve experiment directory {candidate}: {exc}") from exc
    if not resolved.is_dir():
        raise ValidationError(f"experiment path is not a directory: {candidate}")
    return resolved


def _verified_ref(root: Path, value: Any, *, context: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValidationError(f"{context} reference must contain only path and sha256")
    encoded = value.get("path")
    digest = value.get("sha256")
    if not isinstance(encoded, str) or not encoded or "\\" in encoded or ":" in encoded:
        raise ValidationError(f"{context} path must be a portable relative POSIX path")
    relative = PurePosixPath(encoded)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != encoded
    ):
        raise ValidationError(f"{context} path must be a portable relative POSIX path")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValidationError(f"{context} sha256 must be 64 lowercase hexadecimal characters")
    candidate = root.joinpath(*relative.parts)
    current = root
    try:
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValidationError(f"{context} path must not contain symbolic links")
        resolved = candidate.resolve(strict=True)
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(f"{context} file is missing: {encoded}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValidationError(f"{context} must resolve to a regular file inside the experiment")
    actual = sha256_file(resolved)
    if actual != digest:
        raise ValidationError(f"{context} SHA-256 mismatch: expected {digest}, found {actual}")
    return resolved


def _validate_server_suite(
    server: Mapping[str, Any],
    *,
    suite: EvaluationSuite,
    suite_sha256: str,
    candidate_id: str,
) -> None:
    if server.get("candidate_id") != candidate_id:
        raise ValidationError(f"repeat-pass server candidate id does not match {candidate_id!r}")
    if server.get("synthetic") is not False:
        raise ValidationError(
            f"repeat-pass server evaluation for {candidate_id!r} must be measured"
        )
    recorded_suite = server.get("suite")
    if not isinstance(recorded_suite, Mapping):
        raise ValidationError(f"repeat-pass server suite for {candidate_id!r} must be an object")
    expected_suite = {
        "id": suite.suite_id,
        "license": suite.license,
        "sha256": suite_sha256,
        "quality_case_count": len(suite.quality_cases),
        "performance_repetitions": suite.repetitions,
        "performance_warmups": suite.warmups,
        "generation_tokens": suite.generation_tokens,
    }
    for field, expected in expected_suite.items():
        if recorded_suite.get(field) != expected:
            raise ValidationError(
                f"repeat-pass server suite {field} for {candidate_id!r} does not "
                "match evaluation-suite.json"
            )
    quality = server.get("quality")
    cases = quality.get("cases") if isinstance(quality, Mapping) else None
    if not isinstance(cases, list) or len(cases) != len(suite.quality_cases):
        raise ValidationError(
            f"repeat-pass quality cases for {candidate_id!r} do not match the suite"
        )
    for index, (recorded, expected) in enumerate(zip(cases, suite.quality_cases, strict=True)):
        if not isinstance(recorded, Mapping):
            raise ValidationError(
                f"repeat-pass quality case {index} for {candidate_id!r} must be an object"
            )
        if (
            recorded.get("id") != expected.case_id
            or recorded.get("prompt") != expected.prompt
            or recorded.get("accepted_answers") != list(expected.accepted_answers)
            or recorded.get("match_mode", "normalized-text") != expected.match_mode
        ):
            raise ValidationError(
                f"repeat-pass quality case {index} for {candidate_id!r} does not "
                "match evaluation-suite.json"
            )
