"""Replay an extracted ParetoPilot evidence directory without rerunning inference."""

from __future__ import annotations

from contextlib import suppress
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Mapping

from paretopilot.analysis import recommend
from paretopilot.domain import BenchmarkSet, ValidationError
from paretopilot.experiment import assemble_experiment
from paretopilot.io import (
    load_constraints,
    load_json_object,
    sha256_file,
    write_json,
    write_text,
)
from paretopilot.load_eval import (
    combine_load_evaluations,
    load_load_plan,
    validate_load_artifact_against_plan,
    validate_combined_load_evaluation,
)
from paretopilot.pass_eval import assemble_repeat_pass
from paretopilot.profiles import evaluate_policy_profiles, load_policy_set
from paretopilot.report import render_report
from paretopilot.report_v11 import render_report_v11
from paretopilot.stability import summarize_stability


_SHA256_LINE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_REQUIRED_STATUS = {
    "schema_version": "1.0",
    "classification": "canonical",
    "status": "complete",
    "measurement_valid": True,
    "valid_evidence": True,
}
_REQUIRED_PATHS = {
    "status.json",
    "experiment/manifest.json",
    "experiment/constraints.json",
    "experiment/benchmark-set.json",
    "experiment/recommendation.json",
}
_V11_REQUIRED_PATHS = {
    "experiment/evaluation-suite.json",
    "extensions/benchmark-set-pass-1.json",
    "extensions/benchmark-set-pass-2.json",
    "extensions/evaluation-suite.json",
    "extensions/load-evaluation.json",
    "extensions/load-plan.json",
    "extensions/policy-config.json",
    "extensions/policy-profiles.json",
    "extensions/repeat-stability.json",
    "report-v1.1.html",
}


def _safe_relative_path(encoded: str, *, line_number: int) -> PurePosixPath:
    if "\\" in encoded or ":" in encoded:
        raise ValidationError(f"SHA256SUMS line {line_number} contains an unsafe path {encoded!r}")
    normalized = encoded[2:] if encoded.startswith("./") else encoded
    path = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("./")
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != normalized
        or normalized == "SHA256SUMS"
    ):
        raise ValidationError(f"SHA256SUMS line {line_number} contains an unsafe path {encoded!r}")
    return path


def _evidence_files(root: Path) -> set[str]:
    files: set[str] = set()
    try:
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValidationError(
                    "evidence directory contains a symbolic link: "
                    + path.relative_to(root).as_posix()
                )
            if path.is_file():
                files.add(path.relative_to(root).as_posix())
            elif not path.is_dir():
                raise ValidationError(
                    "evidence directory contains a non-regular entry: "
                    + path.relative_to(root).as_posix()
                )
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(f"could not enumerate evidence directory {root}: {exc}") from exc
    files.discard("SHA256SUMS")
    return files


def _verify_checksums(root: Path) -> tuple[Mapping[str, str], str]:
    checksum_path = root / "SHA256SUMS"
    try:
        raw = checksum_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("SHA256SUMS must contain UTF-8 text") from exc
    except OSError as exc:
        raise ValidationError(f"could not read {checksum_path}: {exc}") from exc
    if not raw or not raw.endswith("\n"):
        raise ValidationError("SHA256SUMS must be non-empty and end with a newline")

    entries: dict[str, str] = {}
    casefold_paths: set[str] = set()
    root_resolved = root.resolve(strict=True)
    for line_number, line in enumerate(raw.splitlines(), start=1):
        match = _SHA256_LINE.fullmatch(line)
        if match is None:
            raise ValidationError(f"SHA256SUMS line {line_number} is malformed")
        expected_digest, encoded = match.groups()
        relative = _safe_relative_path(encoded, line_number=line_number)
        relative_text = relative.as_posix()
        if relative_text in entries or relative_text.casefold() in casefold_paths:
            raise ValidationError(f"SHA256SUMS contains duplicate entry {relative_text!r}")
        casefold_paths.add(relative_text.casefold())

        candidate = root.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValidationError(f"SHA256SUMS entry is missing: {relative_text}") from exc
        if (
            not resolved.is_relative_to(root_resolved)
            or candidate.is_symlink()
            or not candidate.is_file()
        ):
            raise ValidationError(f"SHA256SUMS entry is not a safe regular file: {relative_text}")
        actual_digest = sha256_file(candidate)
        if actual_digest != expected_digest:
            raise ValidationError(
                f"SHA256 mismatch for {relative_text}: expected {expected_digest}, "
                f"found {actual_digest}"
            )
        entries[relative_text] = expected_digest

    actual_files = _evidence_files(root)
    listed_files = set(entries)
    unlisted = sorted(actual_files - listed_files)
    missing = sorted(listed_files - actual_files)
    if unlisted:
        raise ValidationError(
            "evidence directory contains files missing from SHA256SUMS: " + ", ".join(unlisted)
        )
    if missing:
        raise ValidationError(
            "SHA256SUMS contains files missing from the evidence directory: " + ", ".join(missing)
        )
    absent_required = sorted(_REQUIRED_PATHS - listed_files)
    if absent_required:
        raise ValidationError(
            "evidence directory is missing required checksummed files: "
            + ", ".join(absent_required)
        )
    return dict(sorted(entries.items())), sha256_file(checksum_path)


def _validate_status(root: Path) -> Mapping[str, Any]:
    status = load_json_object(root / "status.json")
    expected_fields = {*_REQUIRED_STATUS, "reason"}
    actual_fields = set(status)
    if actual_fields != expected_fields:
        details: list[str] = []
        missing = sorted(expected_fields - actual_fields)
        unknown = sorted(actual_fields - expected_fields)
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValidationError("status.json has an invalid canonical schema: " + "; ".join(details))
    for field, expected in _REQUIRED_STATUS.items():
        if status.get(field) != expected:
            raise ValidationError(f"status.json {field} must be {expected!r}")
    reason = status.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValidationError("status.json reason must be a non-empty string")
    return status


def _resolve_evidence_directory(path: Path) -> Path:
    candidate = Path(path)
    try:
        if candidate.is_symlink():
            raise ValidationError("evidence directory must not be a symbolic link")
        resolved = candidate.resolve(strict=True)
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(f"could not resolve evidence directory {candidate}: {exc}") from exc
    if not resolved.is_dir():
        raise ValidationError(f"evidence path is not a directory: {candidate}")
    return resolved


def _validate_output_path(evidence_root: Path, output_dir: Path) -> Path:
    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise ValidationError(f"refusing to overwrite existing output directory: {target}")
    try:
        resolved = target.resolve(strict=False)
    except OSError as exc:
        raise ValidationError(f"could not resolve output directory {target}: {exc}") from exc
    if (
        resolved == evidence_root
        or resolved.is_relative_to(evidence_root)
        or evidence_root.is_relative_to(resolved)
    ):
        raise ValidationError(
            "output directory must not contain or be contained by the evidence directory"
        )
    return resolved


def _comparison(
    authoritative: Path,
    regenerated: Path,
    *,
    evidence_root: Path,
) -> Mapping[str, Any]:
    regenerated_digest = sha256_file(regenerated)
    if not authoritative.exists():
        return {
            "present": False,
            "matches": None,
            "authoritative_path": authoritative.relative_to(evidence_root).as_posix(),
            "authoritative_sha256": None,
            "regenerated_sha256": regenerated_digest,
        }
    if authoritative.is_symlink() or not authoritative.is_file():
        raise ValidationError(f"authoritative output is not a regular file: {authoritative}")
    authoritative_digest = sha256_file(authoritative)
    return {
        "present": True,
        "matches": authoritative_digest == regenerated_digest,
        "authoritative_path": authoritative.relative_to(evidence_root).as_posix(),
        "authoritative_sha256": authoritative_digest,
        "regenerated_sha256": regenerated_digest,
    }


def _preserve_archived_profile_versions(
    regenerated: Mapping[str, Any],
    archived: Mapping[str, Any],
) -> None:
    """Retain historical generator versions without weakening profile identity."""

    regenerated_profiles = regenerated.get("profiles")
    archived_profiles = archived.get("profiles")
    if not isinstance(regenerated_profiles, list) or not isinstance(
        archived_profiles,
        list,
    ):
        raise ValidationError(
            "v1.1 policy profiles must contain generated and archived profile arrays"
        )
    if len(regenerated_profiles) != len(archived_profiles):
        raise ValidationError("v1.1 archived policy profile count does not match the policy config")
    for index, (fresh_value, archived_value) in enumerate(
        zip(regenerated_profiles, archived_profiles, strict=True)
    ):
        if not isinstance(fresh_value, dict) or not isinstance(
            archived_value,
            Mapping,
        ):
            raise ValidationError(f"v1.1 policy profile {index} must be an object")
        if fresh_value.get("id") != archived_value.get("id"):
            raise ValidationError("v1.1 archived policy profile ids do not match the policy config")
        fresh_recommendation = fresh_value.get("recommendation")
        archived_recommendation = archived_value.get("recommendation")
        if not isinstance(fresh_recommendation, dict) or not isinstance(
            archived_recommendation,
            Mapping,
        ):
            raise ValidationError(f"v1.1 policy profile {index} recommendation must be an object")
        archived_version = archived_recommendation.get("paretopilot_version")
        if not isinstance(archived_version, str) or not archived_version.strip():
            raise ValidationError(
                f"v1.1 policy profile {index} paretopilot_version must be non-empty"
            )
        fresh_recommendation["paretopilot_version"] = archived_version


def replay_evidence(
    evidence_dir: Path,
    output_dir: Path,
    *,
    policies_path: Path | None = None,
) -> Mapping[str, Any]:
    """Verify and replay one already-extracted canonical evidence directory.

    The inference workload is not rerun.  Instead, the command verifies every
    checksummed file and the canonical completion status, reassembles the
    BenchmarkSet from raw aggregate artifacts, and regenerates deterministic
    decision outputs in a new directory.
    """

    root = _resolve_evidence_directory(evidence_dir)
    destination = _validate_output_path(root, output_dir)
    checksum_entries, checksum_manifest_sha256 = _verify_checksums(root)
    status = _validate_status(root)
    v11_contract = any(
        path.startswith("extensions/") or path == "report-v1.1.html" for path in checksum_entries
    )
    if v11_contract:
        missing_v11 = sorted(_V11_REQUIRED_PATHS - set(checksum_entries))
        if missing_v11:
            raise ValidationError(
                "v1.1 evidence is missing required checksummed files: " + ", ".join(missing_v11)
            )

    manifest_path = root / "experiment" / "manifest.json"
    constraints_path = root / "experiment" / "constraints.json"
    manifest = load_json_object(manifest_path)
    if v11_contract and manifest.get("evaluation_suite_path") != "evaluation-suite.json":
        raise ValidationError(
            "v1.1 experiment manifest evaluation_suite_path must be 'evaluation-suite.json'"
        )
    if manifest.get("classification") != status["classification"]:
        raise ValidationError(
            "experiment manifest classification does not match canonical status.json"
        )
    benchmark_mapping = assemble_experiment(manifest_path)
    metadata = benchmark_mapping.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("classification") != "canonical":
        raise ValidationError("reassembled benchmark metadata must retain canonical classification")
    benchmarks = BenchmarkSet.from_mapping(benchmark_mapping)
    if v11_contract:
        unsafe_ids = [
            candidate.candidate_id
            for candidate in benchmarks.candidates
            if _CANDIDATE_ID.fullmatch(candidate.candidate_id) is None
        ]
        if unsafe_ids:
            raise ValidationError(
                "v1.1 benchmark candidate ids are unsafe for evidence paths: "
                + ", ".join(repr(candidate_id) for candidate_id in unsafe_ids)
            )
    constraints = load_constraints(constraints_path)
    if v11_contract:
        experiment_suite = root / "experiment" / "evaluation-suite.json"
        extension_suite = root / "extensions" / "evaluation-suite.json"
        if sha256_file(experiment_suite) != sha256_file(extension_suite):
            raise ValidationError(
                "v1.1 extension evaluation suite does not match the experiment suite"
            )

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                dir=destination.parent,
                prefix=f".{destination.name}.replay-",
            )
        )
    except OSError as exc:
        raise ValidationError(
            f"could not create replay staging directory for {destination}: {exc}"
        ) from exc

    try:
        benchmark_output = staging / "benchmark-set.json"
        constraints_output = staging / "constraints.json"
        recommendation_output = staging / "recommendation.json"
        report_output = staging / "report.html"
        write_json(benchmark_output, benchmark_mapping)
        write_text(
            constraints_output,
            constraints_path.read_text(encoding="utf-8"),
        )

        recommendation = dict(recommend(benchmarks, constraints))
        archived_recommendation_path = root / "experiment" / "recommendation.json"
        if archived_recommendation_path.exists():
            archived_recommendation = load_json_object(archived_recommendation_path)
            archived_version = archived_recommendation.get("paretopilot_version")
            if not isinstance(archived_version, str) or not archived_version.strip():
                raise ValidationError(
                    "authoritative recommendation paretopilot_version must be a non-empty string"
                )
            # A replay may use a newer ParetoPilot release.  Retaining the
            # archived generator version keeps unchanged historical decisions
            # byte-comparable while every other field is freshly recomputed.
            recommendation["paretopilot_version"] = archived_version
        recommendation["input_fingerprints"] = {
            "benchmarks_sha256": sha256_file(benchmark_output),
            "constraints_sha256": sha256_file(constraints_path),
        }
        write_json(recommendation_output, recommendation)
        report_html = render_report(
            benchmarks,
            constraints,
            recommendation,
            benchmarks_sha256=recommendation["input_fingerprints"]["benchmarks_sha256"],
            constraints_sha256=recommendation["input_fingerprints"]["constraints_sha256"],
        )
        write_text(report_output, report_html)

        generated_files = [
            "benchmark-set.json",
            "constraints.json",
            "recommendation.json",
            "report.html",
        ]
        profile_count = 0
        profile_selected_ids: dict[str, str] = {}
        profile_output: Path | None = None
        effective_policies_path = policies_path
        if v11_contract:
            archived_policies_path = root / "extensions" / "policy-config.json"
            if policies_path is not None and sha256_file(policies_path) != sha256_file(
                archived_policies_path
            ):
                raise ValidationError(
                    "supplied policy configuration does not match v1.1 archived policy config"
                )
            effective_policies_path = archived_policies_path
        if effective_policies_path is not None:
            policy_set = load_policy_set(effective_policies_path)
            profile_mapping = dict(evaluate_policy_profiles(benchmarks, constraints, policy_set))
            if v11_contract:
                archived_profiles = load_json_object(root / "extensions" / "policy-profiles.json")
                _preserve_archived_profile_versions(
                    profile_mapping,
                    archived_profiles,
                )
            profile_mapping["input_fingerprints"] = {
                "benchmarks_sha256": sha256_file(benchmark_output),
                "constraints_sha256": sha256_file(constraints_path),
                "policies_sha256": sha256_file(effective_policies_path),
            }
            profile_output = staging / "policy-profiles.json"
            write_json(profile_output, profile_mapping)
            generated_files.append("policy-profiles.json")
            profile_count = len(profile_mapping["profiles"])
            profile_selected_ids = {
                profile["id"]: profile["recommendation"]["selected_id"]
                for profile in profile_mapping["profiles"]
            }

        load_output: Path | None = None
        stability_output: Path | None = None
        report_v11_output: Path | None = None
        if v11_contract:
            assert profile_output is not None
            load_plan_path = root / "extensions" / "load-plan.json"
            load_plan = load_load_plan(load_plan_path)
            archived_load = load_json_object(root / "extensions" / "load-evaluation.json")
            validate_combined_load_evaluation(
                archived_load,
                require_evidence_bindings=True,
            )
            validate_load_artifact_against_plan(
                archived_load,
                load_plan,
                context="v1.1 combined load evaluation",
            )
            raw_highest = archived_load.get("highest_slo_concurrency")
            if not isinstance(raw_highest, Mapping):
                raise ValidationError("v1.1 load highest_slo_concurrency must be an object")
            load_candidate_ids = sorted(
                candidate.candidate_id for candidate in benchmarks.candidates
            )
            if list(raw_highest) != load_candidate_ids:
                raise ValidationError(
                    "v1.1 load candidate ids must exactly match the benchmark candidates"
                )
            single_loads = [
                load_json_object(
                    root / "extensions" / "load" / candidate_id / "load-evaluation.json"
                )
                for candidate_id in load_candidate_ids
            ]
            for candidate_id, single_load in zip(
                load_candidate_ids,
                single_loads,
                strict=True,
            ):
                validate_load_artifact_against_plan(
                    single_load,
                    load_plan,
                    context=f"v1.1 load evaluation for {candidate_id!r}",
                )
            rebuilt_load = combine_load_evaluations(
                single_loads,
                require_evidence_bindings=True,
            )
            bindings = rebuilt_load["evidence_bindings"]
            assert isinstance(bindings, Mapping)
            if bindings["plan_sha256"] != sha256_file(load_plan_path):
                raise ValidationError(
                    "v1.1 load evidence plan SHA-256 does not match load-plan.json"
                )
            configurations = bindings["candidate_server_configurations"]
            assert isinstance(configurations, Mapping)
            for candidate_id in load_candidate_ids:
                configuration = configurations[candidate_id]
                assert isinstance(configuration, Mapping)
                load_command = root / "extensions" / "load" / candidate_id / "server-command.json"
                canonical_command = (
                    root / "experiment" / "candidates" / candidate_id / "server-command.json"
                )
                if configuration["load_server_command_sha256"] != sha256_file(load_command):
                    raise ValidationError(
                        f"v1.1 load command SHA-256 does not match for {candidate_id}"
                    )
                if configuration["canonical_server_command_sha256"] != sha256_file(
                    canonical_command
                ):
                    raise ValidationError(
                        f"v1.1 canonical server command SHA-256 does not match for {candidate_id}"
                    )
            load_output = staging / "load-evaluation.json"
            write_json(load_output, rebuilt_load)

            archived_stability = load_json_object(root / "extensions" / "repeat-stability.json")
            pass_labels = archived_stability.get("pass_labels")
            if pass_labels != ["pass-1", "pass-2"]:
                raise ValidationError("v1.1 stability pass_labels must be ['pass-1', 'pass-2']")
            pass_outputs: dict[str, Path] = {}
            rebuilt_passes: list[BenchmarkSet] = []
            for pass_number, label in enumerate(pass_labels, start=1):
                mapping = assemble_repeat_pass(
                    root / "experiment",
                    pass_number=pass_number,
                    benchmark_mapping=benchmark_mapping,
                )
                output_path = staging / f"benchmark-set-{label}.json"
                write_json(output_path, mapping)
                pass_outputs[label] = output_path
                rebuilt_passes.append(BenchmarkSet.from_mapping(mapping))
            rebuilt_stability = summarize_stability(
                rebuilt_passes,
                metric_directions=archived_stability.get("metric_directions"),
                pass_labels=pass_labels,
                input_fingerprints={
                    label: sha256_file(pass_outputs[label]) for label in pass_labels
                },
            )
            stability_output = staging / "repeat-stability.json"
            write_json(stability_output, rebuilt_stability)

            report_v11_output = staging / "report-v1.1.html"
            write_text(
                report_v11_output,
                render_report_v11(
                    benchmarks,
                    recommendation,
                    policy_profiles=load_json_object(profile_output),
                    load_sweep=rebuilt_load,
                    stability_summary=rebuilt_stability,
                    benchmarks_sha256=sha256_file(benchmark_output),
                    recommendation_sha256=sha256_file(recommendation_output),
                    profiles_sha256=sha256_file(profile_output),
                    load_sha256=sha256_file(load_output),
                    stability_sha256=sha256_file(stability_output),
                ),
            )
            generated_files.extend(
                [
                    "load-evaluation.json",
                    "benchmark-set-pass-1.json",
                    "benchmark-set-pass-2.json",
                    "repeat-stability.json",
                    "report-v1.1.html",
                ]
            )

        comparisons: dict[str, Mapping[str, Any]] = {
            "benchmark-set": _comparison(
                root / "experiment" / "benchmark-set.json",
                benchmark_output,
                evidence_root=root,
            ),
            "recommendation": _comparison(
                root / "experiment" / "recommendation.json",
                recommendation_output,
                evidence_root=root,
            ),
            "report": _comparison(
                root / "experiment" / "report.html",
                report_output,
                evidence_root=root,
            ),
        }
        if profile_output is not None:
            comparisons["policy-profiles"] = _comparison(
                (
                    root / "extensions" / "policy-profiles.json"
                    if v11_contract
                    else root / "experiment" / "policy-profiles.json"
                ),
                profile_output,
                evidence_root=root,
            )
        if v11_contract:
            assert load_output is not None
            assert stability_output is not None
            assert report_v11_output is not None
            comparisons["load-evaluation"] = _comparison(
                root / "extensions" / "load-evaluation.json",
                load_output,
                evidence_root=root,
            )
            for label in ("pass-1", "pass-2"):
                comparisons[f"benchmark-set-{label}"] = _comparison(
                    root / "extensions" / f"benchmark-set-{label}.json",
                    pass_outputs[label],
                    evidence_root=root,
                )
            comparisons["repeat-stability"] = _comparison(
                root / "extensions" / "repeat-stability.json",
                stability_output,
                evidence_root=root,
            )
            comparisons["report-v1.1"] = _comparison(
                root / "report-v1.1.html",
                report_v11_output,
                evidence_root=root,
            )

        present_comparisons = [
            comparison for comparison in comparisons.values() if comparison["present"]
        ]
        fully_reproduced = all(comparison["matches"] for comparison in present_comparisons)
        core_names = (
            (
                "benchmark-set",
                "recommendation",
                "policy-profiles",
                "load-evaluation",
                "benchmark-set-pass-1",
                "benchmark-set-pass-2",
                "repeat-stability",
            )
            if v11_contract
            else ("benchmark-set", "recommendation")
        )
        decision_reproduced = all(
            not comparisons[name]["present"] or comparisons[name]["matches"] for name in core_names
        )
        differences = [
            name
            for name, comparison in comparisons.items()
            if comparison["present"] and not comparison["matches"]
        ]
        warnings = [
            (
                f"{name} differs from the checksummed archive; this does not "
                "invalidate measured evidence or a reproduced core decision."
            )
            for name in differences
            if name not in core_names
        ]
        valid = decision_reproduced
        if not valid:
            verdict = "FAIL: evidence verified, but the core decision did not reproduce."
        elif fully_reproduced:
            verdict = "PASS: evidence verified and all archived outputs reproduced."
        else:
            verdict = (
                "PASS: evidence and core decision reproduced; presentation or "
                "optional output differs."
            )
        report_comparison = comparisons["report"]
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "replay_contract": "1.1" if v11_contract else "1.0",
            "valid": valid,
            "verdict": verdict,
            "evidence_directory": str(root),
            "output_directory": str(destination),
            "checksums": {
                "verified": True,
                "entry_count": len(checksum_entries),
                "manifest_sha256": checksum_manifest_sha256,
            },
            "status_complete": True,
            "candidate_count": len(benchmarks.candidates),
            "selected_id": recommendation["selected_id"],
            "generated_files": generated_files,
            "decision_reproduced": decision_reproduced,
            "fully_reproduced": fully_reproduced,
            "report_matches_archive": (
                report_comparison["matches"] if report_comparison["present"] else None
            ),
            "differences": differences,
            "warnings": warnings,
            "authoritative_outputs_match": fully_reproduced,
            "authoritative_comparisons": comparisons,
            "policy_profile_count": profile_count,
            "policy_selected_ids": profile_selected_ids,
        }
        write_json(staging / "replay.json", payload)

        if destination.exists() or destination.is_symlink():
            raise ValidationError(f"refusing to overwrite existing output directory: {destination}")
        try:
            os.rename(staging, destination)
        except OSError as exc:
            raise ValidationError(
                f"could not publish replay output directory {destination}: {exc}"
            ) from exc
        return payload
    except UnicodeDecodeError as exc:
        raise ValidationError(
            f"constraints file must contain UTF-8 text: {constraints_path}"
        ) from exc
    finally:
        if "staging" in locals() and staging.exists():
            with suppress(OSError):
                shutil.rmtree(staging)
