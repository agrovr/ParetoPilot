"""Assemble a verified paired Arm64 evidence bundle into a ParetoPilot study.

The assembler is intentionally conservative.  It only turns a bundle into
selection inputs after checking the bundle-wide SHA-256 manifest, canonical
completion status, source summaries, generated comparisons, and the identity
of the model and runtime used by both variants.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from paretopilot.domain import ValidationError
from paretopilot.io import load_json_object, sha256_file
from paretopilot.llama_compare import compare_llama_bench_summaries


_SHA256_LINE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_STATUS = {
    "classification": "canonical",
    "measurement_valid": True,
    "schema_version": "1.0",
    "status": "complete",
    "valid_evidence": True,
}
_REQUIRED_COMPATIBILITY = {
    "validated",
    "same_build_commit",
    "same_model_filename",
    "same_benchmark_settings_except_kleidiai",
    "same_workload_shapes",
    "same_sample_counts",
    "same_synthetic_status",
}
_COMPARISON_INPUTS = {
    "summary/comparison-pooled.json": (
        "summary/generic.json",
        "summary/kleidiai.json",
    ),
    "summary/comparison-pair-1.json": (
        "summary/generic-pass-1.json",
        "summary/kleidiai-pass-1.json",
    ),
    "summary/comparison-pair-2.json": (
        "summary/generic-pass-2.json",
        "summary/kleidiai-pass-2.json",
    ),
}
_REQUIRED_FILES = {
    "status.json",
    "manifest.json",
    "provenance.json",
    "environment/runner.json",
    "settings/generic.json",
    "settings/kleidiai.json",
    *_COMPARISON_INPUTS,
    *(path for pair in _COMPARISON_INPUTS.values() for path in pair),
}


class StudyAssemblyError(ValidationError):
    """Raised when a published study bundle cannot be trusted or assembled."""


@dataclass(frozen=True, slots=True)
class StudyAssembly:
    """Deterministic, JSON-compatible inputs for ParetoPilot recommendation.

    ``benchmark_set`` can be passed directly to
    :meth:`paretopilot.domain.BenchmarkSet.from_mapping`, and ``constraints``
    can be passed to :meth:`paretopilot.domain.Constraints.from_mapping`.
    ``adoption_gate`` retains the human-readable confidence decision that is
    represented numerically in the generated constraint set.
    """

    benchmark_set: Mapping[str, Any]
    constraints: Mapping[str, Any]
    adoption_gate: Mapping[str, Any]

    def to_mapping(self) -> dict[str, Any]:
        """Return an independent JSON-compatible representation."""

        return {
            "benchmark_set": deepcopy(dict(self.benchmark_set)),
            "constraints": deepcopy(dict(self.constraints)),
            "adoption_gate": deepcopy(dict(self.adoption_gate)),
        }


def _error(message: str) -> StudyAssemblyError:
    return StudyAssemblyError(message)


def _object(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{context} must be an object")
    return value


def _string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(f"{context} must be a non-empty string")
    return value


def _boolean(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise _error(f"{context} must be a boolean")
    return value


def _positive_integer(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _error(f"{context} must be a positive integer")
    return value


def _positive_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise _error(f"{context} must be finite and positive")
    return result


def _same_number(actual: Any, expected: float, *, context: str) -> None:
    number = _positive_number(actual, context=context)
    if not math.isclose(number, expected, rel_tol=1e-12, abs_tol=1e-9):
        raise _error(f"{context} is inconsistent with its source medians")


def _safe_checksum_path(encoded: str, *, line_number: int) -> PurePosixPath:
    if "\\" in encoded or ":" in encoded:
        raise _error(f"SHA256SUMS line {line_number} contains an unsafe relative path {encoded!r}")
    path = PurePosixPath(encoded)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != encoded
        or encoded == "SHA256SUMS"
    ):
        raise _error(f"SHA256SUMS line {line_number} contains an unsafe relative path {encoded!r}")
    return path


def _bundle_files(root: Path) -> set[str]:
    files: set[str] = set()
    try:
        for path in root.rglob("*"):
            if path.is_symlink():
                raise _error(f"bundle contains a symbolic link: {path.relative_to(root)}")
            if path.is_file():
                files.add(path.relative_to(root).as_posix())
            elif not path.is_dir():
                raise _error(f"bundle contains a non-regular filesystem entry: {path}")
    except StudyAssemblyError:
        raise
    except OSError as exc:
        raise _error(f"could not enumerate evidence bundle {root}: {exc}") from exc
    files.discard("SHA256SUMS")
    return files


def _verify_checksums(root: Path) -> tuple[dict[str, str], str]:
    checksum_file = root / "SHA256SUMS"
    try:
        raw = checksum_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise _error(f"could not read checksum manifest {checksum_file}: {exc}") from exc
    except UnicodeError as exc:
        raise _error("SHA256SUMS must be valid UTF-8") from exc

    if not raw or not raw.endswith("\n"):
        raise _error("SHA256SUMS must be non-empty and end with a newline")

    root_resolved = root.resolve(strict=True)
    entries: dict[str, str] = {}
    casefold_paths: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        match = _SHA256_LINE.fullmatch(line)
        if match is None:
            raise _error(f"SHA256SUMS line {line_number} is malformed")
        expected_digest, encoded_path = match.groups()
        relative = _safe_checksum_path(encoded_path, line_number=line_number)
        if encoded_path in entries or encoded_path.casefold() in casefold_paths:
            raise _error(f"SHA256SUMS contains duplicate entry {encoded_path!r}")
        casefold_paths.add(encoded_path.casefold())

        candidate = root.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise _error(f"SHA256SUMS entry is missing: {encoded_path}") from exc
        if not resolved.is_relative_to(root_resolved) or candidate.is_symlink():
            raise _error(f"SHA256SUMS entry escapes the bundle: {encoded_path}")
        if not candidate.is_file():
            raise _error(f"SHA256SUMS entry is not a regular file: {encoded_path}")

        actual_digest = sha256_file(candidate)
        if actual_digest != expected_digest:
            raise _error(
                f"SHA256 mismatch for {encoded_path}: expected {expected_digest}, "
                f"found {actual_digest}"
            )
        entries[encoded_path] = expected_digest

    actual_files = _bundle_files(root)
    listed_files = set(entries)
    missing_entries = sorted(actual_files - listed_files)
    extra_entries = sorted(listed_files - actual_files)
    if missing_entries:
        raise _error("bundle contains files missing from SHA256SUMS: " + ", ".join(missing_entries))
    if extra_entries:
        raise _error(
            "SHA256SUMS contains entries missing from the bundle: " + ", ".join(extra_entries)
        )
    absent_required = sorted(_REQUIRED_FILES - listed_files)
    if absent_required:
        raise _error(
            "evidence bundle is missing required checksummed files: " + ", ".join(absent_required)
        )
    return dict(sorted(entries.items())), sha256_file(checksum_file)


def _load(root: Path, relative_path: str) -> Mapping[str, Any]:
    try:
        return load_json_object(root / relative_path)
    except ValidationError as exc:
        raise _error(f"invalid {relative_path}: {exc}") from exc


def _validate_status(status: Mapping[str, Any]) -> None:
    expected_fields = {*_REQUIRED_STATUS, "reason"}
    if set(status) != expected_fields:
        unknown = sorted(set(status) - expected_fields)
        missing = sorted(expected_fields - set(status))
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise _error("status.json has an invalid canonical schema: " + "; ".join(details))
    for field, expected in _REQUIRED_STATUS.items():
        if status.get(field) != expected:
            raise _error(f"status.json {field} must be {expected!r}")
    _string(status.get("reason"), context="status.json reason")


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != "1.0":
        raise _error("manifest.json schema_version must currently be '1.0'")
    if manifest.get("classification") != "canonical":
        raise _error("manifest.json classification must be 'canonical'")
    if manifest.get("synthetic") is not False:
        raise _error("manifest.json synthetic must be false")
    _string(manifest.get("evidence_scope"), context="manifest.json evidence_scope")

    benchmark = _object(manifest.get("benchmark"), context="manifest.json benchmark")
    benchmark_values = {
        "threads": _positive_integer(
            benchmark.get("threads"), context="manifest.json benchmark.threads"
        ),
        "batch_size": _positive_integer(
            benchmark.get("batch_size"), context="manifest.json benchmark.batch_size"
        ),
        "ubatch_size": _positive_integer(
            benchmark.get("ubatch_size"), context="manifest.json benchmark.ubatch_size"
        ),
        "prompt_tokens": _positive_integer(
            benchmark.get("prompt_tokens"), context="manifest.json benchmark.prompt_tokens"
        ),
        "generation_tokens": _positive_integer(
            benchmark.get("generation_tokens"),
            context="manifest.json benchmark.generation_tokens",
        ),
        "repetitions_per_pass": _positive_integer(
            benchmark.get("repetitions_per_pass"),
            context="manifest.json benchmark.repetitions_per_pass",
        ),
        "cpu_only": _boolean(benchmark.get("cpu_only"), context="manifest.json benchmark.cpu_only"),
        "warmup": _boolean(benchmark.get("warmup"), context="manifest.json benchmark.warmup"),
    }
    if not benchmark_values["cpu_only"]:
        raise _error("manifest.json benchmark.cpu_only must be true")
    if benchmark_values["ubatch_size"] > benchmark_values["batch_size"]:
        raise _error("manifest.json ubatch_size cannot exceed batch_size")

    model = _object(manifest.get("model"), context="manifest.json model")
    model_sha = _string(model.get("sha256"), context="manifest.json model.sha256")
    if _SHA256.fullmatch(model_sha) is None:
        raise _error("manifest.json model.sha256 must be a lowercase SHA-256 digest")
    for field in ("filename", "quantization", "repository", "revision"):
        _string(model.get(field), context=f"manifest.json model.{field}")
    _positive_integer(model.get("size_bytes"), context="manifest.json model.size_bytes")

    llama_cpp = _object(manifest.get("llama_cpp"), context="manifest.json llama_cpp")
    commit = _string(llama_cpp.get("commit"), context="manifest.json llama_cpp.commit")
    if _FULL_COMMIT.fullmatch(commit) is None:
        raise _error("manifest.json llama_cpp.commit must be a 40-character commit hash")
    if llama_cpp.get("generic_kleidiai") is not False:
        raise _error("manifest.json llama_cpp.generic_kleidiai must be false")
    if llama_cpp.get("optimized_kleidiai") is not True:
        raise _error("manifest.json llama_cpp.optimized_kleidiai must be true")

    kleidiai = _object(manifest.get("kleidiai"), context="manifest.json kleidiai")
    _string(kleidiai.get("version"), context="manifest.json kleidiai.version")
    source_sha = _string(
        kleidiai.get("source_sha256"), context="manifest.json kleidiai.source_sha256"
    )
    if _SHA256.fullmatch(source_sha) is None:
        raise _error("manifest.json kleidiai.source_sha256 must be a SHA-256 digest")
    _positive_integer(
        kleidiai.get("source_size_bytes"),
        context="manifest.json kleidiai.source_size_bytes",
    )

    quality = _object(manifest.get("quality"), context="manifest.json quality")
    if quality.get("evaluated") is not False:
        raise _error(
            "this runtime-kernel study requires manifest.json quality.evaluated to be false"
        )
    _string(quality.get("reason"), context="manifest.json quality.reason")

    repository = _object(manifest.get("repository"), context="manifest.json repository")
    for field in (
        "name",
        "run_id",
        "run_attempt",
        "commit",
        "workflow_commit",
        "workflow_ref",
    ):
        _string(repository.get(field), context=f"manifest.json repository.{field}")
    if repository["commit"] != repository["workflow_commit"]:
        raise _error("manifest.json repository and workflow commits must match")

    execution_order = manifest.get("execution_order")
    if execution_order != [
        "generic-pass-1",
        "kleidiai-pass-1",
        "kleidiai-pass-2",
        "generic-pass-2",
    ]:
        raise _error("manifest.json execution_order is not the canonical paired order")

    return {
        "benchmark": benchmark_values,
        "model": model,
        "model_sha256": model_sha,
        "llama_cpp": llama_cpp,
        "llama_cpp_commit": commit,
        "kleidiai": kleidiai,
        "quality": quality,
        "repository": repository,
    }


def _validate_settings(
    generic: Mapping[str, Any],
    kleidiai: Mapping[str, Any],
    *,
    manifest_values: Mapping[str, Any],
) -> None:
    generic_copy = deepcopy(dict(generic))
    kleidiai_copy = deepcopy(dict(kleidiai))
    generic_build = _object(generic_copy.get("build"), context="settings/generic.json build")
    kleidiai_build = _object(kleidiai_copy.get("build"), context="settings/kleidiai.json build")
    if generic_build != {"kleidiai": False}:
        raise _error("settings/generic.json build must select only generic kernels")
    if kleidiai_build != {"kleidiai": True}:
        raise _error("settings/kleidiai.json build must select only KleidiAI kernels")
    del generic_copy["build"]
    del kleidiai_copy["build"]
    if generic_copy != kleidiai_copy:
        raise _error("generic and KleidiAI settings differ beyond build.kleidiai")

    expected = dict(manifest_values["benchmark"])
    expected["llama_cpp_commit"] = manifest_values["llama_cpp_commit"]
    expected["model_sha256"] = manifest_values["model_sha256"]
    if generic_copy != expected:
        raise _error("settings files do not exactly match the canonical manifest")


def _validate_provenance(
    provenance: Mapping[str, Any],
    *,
    manifest_values: Mapping[str, Any],
) -> None:
    if provenance.get("schema_version") != "1.0":
        raise _error("provenance.json schema_version must currently be '1.0'")
    repository = manifest_values["repository"]
    if provenance.get("repository") != repository["name"]:
        raise _error("provenance.json repository does not match manifest.json")
    if provenance.get("repository_commit") != repository["commit"]:
        raise _error("provenance.json repository_commit does not match manifest.json")
    if provenance.get("workflow_commit") != repository["workflow_commit"]:
        raise _error("provenance.json workflow_commit does not match manifest.json")

    run = _object(provenance.get("run"), context="provenance.json run")
    if str(run.get("id")) != repository["run_id"]:
        raise _error("provenance.json run.id does not match manifest.json")
    if str(run.get("attempt")) != repository["run_attempt"]:
        raise _error("provenance.json run.attempt does not match manifest.json")
    _positive_integer(run.get("job_id"), context="provenance.json run.job_id")
    _string(run.get("url"), context="provenance.json run.url")

    verification = _object(provenance.get("verification"), context="provenance.json verification")
    required = {
        "compact_bundle_sha256_verified": True,
        "measurement_valid": True,
        "original_artifact_checksums_verified": True,
        "status": "complete",
        "valid_evidence": True,
    }
    for field, expected in required.items():
        if verification.get(field) != expected:
            raise _error(f"provenance.json verification.{field} must be {expected!r}")
    if verification.get("local_raw_artifacts_validated") != 4:
        raise _error("provenance.json must record four validated raw artifacts")

    canonical_inputs = _object(
        provenance.get("canonical_inputs"), context="provenance.json canonical_inputs"
    )
    for field in (
        "threads",
        "batch_size",
        "ubatch_size",
        "prompt_tokens",
        "generation_tokens",
        "repetitions_per_pass",
    ):
        if canonical_inputs.get(field) != manifest_values["benchmark"][field]:
            raise _error(f"provenance.json canonical_inputs.{field} does not match manifest")


def _validate_runner(
    runner: Mapping[str, Any],
    *,
    manifest_values: Mapping[str, Any],
) -> None:
    if runner.get("machine") != "aarch64":
        raise _error("environment/runner.json machine must be 'aarch64'")
    platform = _string(runner.get("platform"), context="environment/runner.json platform")
    if "aarch64" not in platform.lower():
        raise _error("environment/runner.json platform must identify aarch64")
    _string(runner.get("python_version"), context="environment/runner.json python_version")
    environment = _object(
        runner.get("captured_environment"),
        context="environment/runner.json captured_environment",
    )
    repository = manifest_values["repository"]
    expected = {
        "RUNNER_ARCH": "ARM64",
        "RUNNER_OS": "Linux",
        "GITHUB_REPOSITORY": repository["name"],
        "GITHUB_RUN_ID": repository["run_id"],
        "GITHUB_RUN_ATTEMPT": repository["run_attempt"],
        "GITHUB_SHA": repository["commit"],
        "GITHUB_WORKFLOW_SHA": repository["workflow_commit"],
        "GITHUB_WORKFLOW_REF": repository["workflow_ref"],
    }
    for field, value in expected.items():
        if environment.get(field) != value:
            raise _error(
                f"environment/runner.json captured_environment.{field} does not match the manifest"
            )


def _validate_comparison(
    root: Path,
    checksums: Mapping[str, str],
    comparison_path: str,
    generic_path: str,
    kleidiai_path: str,
) -> dict[str, Any]:
    generic = _load(root, generic_path)
    kleidiai = _load(root, kleidiai_path)
    try:
        expected = compare_llama_bench_summaries(generic, kleidiai).to_mapping()
    except ValidationError as exc:
        raise _error(f"{comparison_path} source summaries are incompatible: {exc}") from exc
    expected["input_fingerprints"] = {
        "generic_summary_sha256": checksums[generic_path],
        "kleidiai_summary_sha256": checksums[kleidiai_path],
    }
    stored = dict(_load(root, comparison_path))
    if stored != expected:
        raise _error(f"{comparison_path} does not match its checksummed source summaries")
    compatibility = _object(stored.get("compatibility"), context=f"{comparison_path} compatibility")
    if set(compatibility) != _REQUIRED_COMPATIBILITY or any(
        compatibility.get(field) is not True for field in _REQUIRED_COMPATIBILITY
    ):
        raise _error(f"{comparison_path} does not contain all validated compatibility gates")
    if stored.get("synthetic_fixture") is not False:
        raise _error(f"{comparison_path} synthetic_fixture must be false")
    if set(_object(stored.get("tests"), context=f"{comparison_path} tests")) != {
        "pp",
        "tg",
    }:
        raise _error(f"{comparison_path} must contain exactly pp and tg workloads")
    return stored


def _validate_comparison_set(
    comparisons: Mapping[str, Mapping[str, Any]],
    *,
    manifest_values: Mapping[str, Any],
) -> None:
    pooled = comparisons["summary/comparison-pooled.json"]
    model_filename = pooled.get("model_filename")
    build_commit = pooled.get("build_commit")
    if not isinstance(model_filename, str):
        raise _error("pooled comparison model_filename must be a string")
    if (
        PurePosixPath(model_filename.replace("\\", "/")).name
        != manifest_values["model"]["filename"]
    ):
        raise _error("comparison model filename does not match the canonical model")
    if not isinstance(build_commit, str) or len(build_commit) < 7:
        raise _error("comparison build_commit must be an unambiguous commit prefix")
    if not manifest_values["llama_cpp_commit"].startswith(build_commit):
        raise _error("comparison build_commit does not match the canonical runtime")

    for path, comparison in comparisons.items():
        if comparison.get("model_filename") != model_filename:
            raise _error(f"{path} does not use the same exact model as the pooled comparison")
        if comparison.get("build_commit") != build_commit:
            raise _error(f"{path} does not use the same runtime as the pooled comparison")
        tests = _object(comparison.get("tests"), context=f"{path} tests")
        pp = _object(tests.get("pp"), context=f"{path} tests.pp")
        tg = _object(tests.get("tg"), context=f"{path} tests.tg")
        if (pp.get("n_prompt"), pp.get("n_gen")) != (
            manifest_values["benchmark"]["prompt_tokens"],
            0,
        ):
            raise _error(f"{path} prompt workload does not match the manifest")
        if (tg.get("n_prompt"), tg.get("n_gen")) != (
            0,
            manifest_values["benchmark"]["generation_tokens"],
        ):
            raise _error(f"{path} generation workload does not match the manifest")

    repetitions = manifest_values["benchmark"]["repetitions_per_pass"]
    pair_paths = (
        "summary/comparison-pair-1.json",
        "summary/comparison-pair-2.json",
    )
    for workload in ("pp", "tg"):
        pooled_test = pooled["tests"][workload]
        for role in ("generic", "kleidiai"):
            pooled_samples = pooled_test[role]["tokens_per_second_sample_count"]
            pair_samples = [
                comparisons[path]["tests"][workload][role]["tokens_per_second_sample_count"]
                for path in pair_paths
            ]
            if pair_samples != [repetitions, repetitions]:
                raise _error(f"paired {workload} sample counts do not match repetitions_per_pass")
            if pooled_samples != sum(pair_samples):
                raise _error(f"pooled {workload} sample count is not the sum of paired samples")


def _metric_snapshot(comparison: Mapping[str, Any], role: str) -> dict[str, float]:
    tests = comparison["tests"]
    pp = tests["pp"][role]
    tg = tests["tg"][role]
    return {
        "prompt_tokens_per_second": float(pp["tokens_per_second_median"]),
        "generation_tokens_per_second": float(tg["tokens_per_second_median"]),
        "prompt_duration_ms": float(pp["duration_ns_median"]) / 1_000_000.0,
        "generation_duration_ms": float(tg["duration_ns_median"]) / 1_000_000.0,
        "model_identity_quality_retention": 1.0,
    }


def _adoption_gate(
    comparisons: Mapping[str, Mapping[str, Any]],
    *,
    threshold_percent: float,
) -> dict[str, Any]:
    pair_paths = (
        "summary/comparison-pair-1.json",
        "summary/comparison-pair-2.json",
    )
    pair_changes = [
        float(comparisons[path]["tests"]["tg"]["median_throughput_percent_change"])
        for path in pair_paths
    ]
    pooled_change = float(
        comparisons["summary/comparison-pooled.json"]["tests"]["tg"][
            "median_throughput_percent_change"
        ]
    )
    direction_consistent = all(change > 0.0 for change in pair_changes)
    meets_threshold = pooled_change > threshold_percent or math.isclose(
        pooled_change,
        threshold_percent,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    eligible = direction_consistent and meets_threshold

    reasons: list[str] = []
    if not direction_consistent:
        reasons.append(
            "paired generation-throughput directions are inconsistent: "
            + ", ".join(f"{change:+.6f}%" for change in pair_changes)
        )
    if not meets_threshold:
        reasons.append(
            f"pooled generation-throughput change {pooled_change:+.6f}% is below "
            f"the predeclared {threshold_percent:.6f}% practical-effect threshold"
        )
    if eligible:
        reasons.append(
            "both paired generation-throughput changes are positive and the pooled "
            "change meets the predeclared practical-effect threshold"
        )

    status = (
        "eligible-for-adoption"
        if eligible
        else "inconclusive"
        if not direction_consistent
        else "below-practical-effect-threshold"
    )
    return {
        "metric": "generation_tokens_per_second",
        "status": status,
        "eligible": eligible,
        "paired_direction_consistent": direction_consistent,
        "pair_percent_changes": pair_changes,
        "pooled_percent_change": pooled_change,
        "practical_effect_threshold_percent": threshold_percent,
        "meets_practical_effect_threshold": meets_threshold,
        "policy": (
            "adopt only when both paired generation-throughput changes are positive "
            "and the pooled change meets the predeclared threshold"
        ),
        "reasons": reasons,
    }


def assemble_study(
    bundle_directory: str | Path,
    *,
    practical_effect_threshold_percent: float = 1.0,
) -> StudyAssembly:
    """Verify and assemble one canonical paired Arm64 evidence bundle.

    The practical-effect threshold is expressed as a percentage-point change
    in pooled generation throughput.  It defaults to a predeclared 1.0%.  A
    KleidiAI candidate must improve in the same direction in both pairs *and*
    meet that pooled threshold; otherwise the generated constraints retain the
    generic baseline.
    """

    if isinstance(practical_effect_threshold_percent, bool) or not isinstance(
        practical_effect_threshold_percent, (int, float)
    ):
        raise _error("practical_effect_threshold_percent must be a number")
    threshold = float(practical_effect_threshold_percent)
    if not math.isfinite(threshold) or threshold < 0:
        raise _error("practical_effect_threshold_percent must be finite and non-negative")

    root = Path(bundle_directory)
    try:
        if root.is_symlink():
            raise _error("evidence bundle directory cannot be a symbolic link")
        if not root.is_dir():
            raise _error(f"evidence bundle directory does not exist: {root}")
    except OSError as exc:
        raise _error(f"could not inspect evidence bundle {root}: {exc}") from exc

    checksums, checksum_manifest_sha256 = _verify_checksums(root)
    status = _load(root, "status.json")
    _validate_status(status)
    manifest = _load(root, "manifest.json")
    manifest_values = _validate_manifest(manifest)
    provenance = _load(root, "provenance.json")
    _validate_provenance(provenance, manifest_values=manifest_values)
    runner = _load(root, "environment/runner.json")
    _validate_runner(runner, manifest_values=manifest_values)
    generic_settings = _load(root, "settings/generic.json")
    kleidiai_settings = _load(root, "settings/kleidiai.json")
    _validate_settings(
        generic_settings,
        kleidiai_settings,
        manifest_values=manifest_values,
    )

    comparisons = {
        comparison_path: _validate_comparison(
            root,
            checksums,
            comparison_path,
            generic_path,
            kleidiai_path,
        )
        for comparison_path, (generic_path, kleidiai_path) in _COMPARISON_INPUTS.items()
    }
    _validate_comparison_set(comparisons, manifest_values=manifest_values)
    pooled = comparisons["summary/comparison-pooled.json"]
    adoption_gate = _adoption_gate(comparisons, threshold_percent=threshold)

    generic_metrics = _metric_snapshot(pooled, "generic")
    generic_metrics["confidence_and_practical_effect_gate"] = 1.0
    kleidiai_metrics = _metric_snapshot(pooled, "kleidiai")
    kleidiai_metrics["confidence_and_practical_effect_gate"] = (
        1.0 if adoption_gate["eligible"] else 0.0
    )

    common_parameters = {
        "model_filename": manifest_values["model"]["filename"],
        "model_sha256": manifest_values["model_sha256"],
        "llama_cpp_commit": manifest_values["llama_cpp_commit"],
        "benchmark": deepcopy(dict(manifest_values["benchmark"])),
        "quality_evidence": {
            "kind": "model-identity",
            "direct_quality_evaluation": False,
            "statement": (
                "Both variants use the identical model artifact; this is model-identity "
                "retention, not a directly measured task-quality score."
            ),
        },
    }
    generic_parameters = deepcopy(common_parameters)
    generic_parameters.update(
        {
            "runtime_variant": "generic",
            "kleidiai_enabled": False,
            "adoption_gate": {
                "status": "reference-baseline",
                "eligible": True,
                "reasons": ["canonical reference baseline"],
            },
        }
    )
    kleidiai_parameters = deepcopy(common_parameters)
    kleidiai_parameters.update(
        {
            "runtime_variant": "kleidiai",
            "kleidiai_enabled": True,
            "kleidiai_version": manifest_values["kleidiai"]["version"],
            "adoption_gate": deepcopy(adoption_gate),
        }
    )

    benchmark_set = {
        "schema_version": "1.0",
        "baseline_id": "generic-baseline",
        "synthetic": False,
        "candidates": [
            {
                "id": "generic-baseline",
                "label": "Generic Arm64 baseline",
                "parameters": generic_parameters,
                "metrics": generic_metrics,
            },
            {
                "id": "kleidiai-optimized",
                "label": "KleidiAI Arm64 runtime",
                "parameters": kleidiai_parameters,
                "metrics": kleidiai_metrics,
            },
        ],
        "metadata": {
            "study_type": "paired-arm64-runtime-kernel",
            "classification": "canonical",
            "evidence_scope": manifest["evidence_scope"],
            "status": deepcopy(dict(status)),
            "repository": deepcopy(dict(manifest_values["repository"])),
            "provenance": deepcopy(dict(provenance)),
            "model": deepcopy(dict(manifest_values["model"])),
            "runtime": {
                "llama_cpp": deepcopy(dict(manifest_values["llama_cpp"])),
                "kleidiai": deepcopy(dict(manifest_values["kleidiai"])),
            },
            "runner": deepcopy(dict(runner)),
            "benchmark": deepcopy(dict(manifest_values["benchmark"])),
            "quality_evidence": {
                "metric": "model_identity_quality_retention",
                "retention": 1.0,
                "basis": "identical-model-sha256",
                "model_sha256": manifest_values["model_sha256"],
                "direct_quality_evaluation": False,
                "manifest_statement": deepcopy(dict(manifest_values["quality"])),
            },
            "adoption_gate": deepcopy(adoption_gate),
            "comparisons": {
                "pooled": deepcopy(pooled),
                "pair_1": deepcopy(comparisons["summary/comparison-pair-1.json"]),
                "pair_2": deepcopy(comparisons["summary/comparison-pair-2.json"]),
            },
            "evidence_integrity": {
                "algorithm": "sha256",
                "checksum_manifest": "SHA256SUMS",
                "checksum_manifest_sha256": checksum_manifest_sha256,
                "verified_entry_count": len(checksums),
                "verified_files": dict(checksums),
            },
        },
    }
    constraints = {
        "min_quality_retention": 1.0,
        "quality_metric": "model_identity_quality_retention",
        "max_values": {},
        "min_values": {"confidence_and_practical_effect_gate": 1.0},
        "objective": {
            "metric": "generation_tokens_per_second",
            "direction": "max",
        },
        "frontier_metrics": {
            "generation_tokens_per_second": "max",
            "prompt_tokens_per_second": "max",
            "generation_duration_ms": "min",
            "prompt_duration_ms": "min",
            "model_identity_quality_retention": "max",
        },
    }
    return StudyAssembly(
        benchmark_set=benchmark_set,
        constraints=constraints,
        adoption_gate=adoption_gate,
    )


# Descriptive alias for callers that want the evidence type in the function name.
assemble_arm64_study = assemble_study
