"""Strictly assemble a multi-candidate Arm64 experiment into a BenchmarkSet.

The manifest is deliberately small and closed: unknown fields are rejected at
every schema-controlled level.  Candidate artifacts are content-addressed by
SHA-256 and must be portable, regular files below the manifest directory.  The
assembler validates and cross-checks aggregate artifacts against the recorded
settings, exact command vector, and balanced raw measurement passes:

* a :mod:`paretopilot.llama_summary` mapping for prompt/generation throughput;
* a :mod:`paretopilot.server_eval` mapping for quality and request latency; and
* a GNU ``time -v`` measurement mapping for peak RSS and deployment identity.

The exact manifest contract is documented by ``assemble_experiment``.  The
returned object is a JSON-compatible mapping accepted directly by
``BenchmarkSet.from_mapping``.  No metric is estimated or filled with a
default: every emitted value is derived from a validated source artifact or
from declared model byte size using an explicit binary-unit conversion.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path, PurePosixPath
import re
import statistics
from typing import Any, Mapping, Sequence

from paretopilot.domain import BenchmarkSet, ValidationError
from paretopilot.io import load_json_object, sha256_file
from paretopilot.llama_summary import summarize_llama_bench_paths
from paretopilot.server_eval import parse_gnu_time_peak_rss, pool_server_evaluations


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
_METRIC_SOURCE_POINTERS = {
    "prompt_tps": "/tests/pp/tokens_per_second/median",
    "generation_tps": "/tests/tg/tokens_per_second/median",
    "quality_score": "/quality/score",
    "ttft_ms_p95": "/latency/ttft_ms_p95",
    "e2e_latency_ms_p95": "/latency/e2e_latency_ms_p95",
    "peak_rss_mib": "/maximum_resident_set_kbytes (divided by 1024)",
    "model_size_mib": "/candidates/*/model/size_bytes (divided by 1048576)",
}


class ExperimentAssemblyError(ValidationError):
    """Raised when a multi-candidate experiment cannot be trusted."""


def assemble_experiment(manifest_path: Path) -> Mapping[str, Any]:
    """Validate an experiment manifest and return a real BenchmarkSet mapping.

    Manifest schema ``1.0`` (all listed fields are required)::

        {
          "schema_version": "1.0",
          "experiment_id": "arm64-candidate-study",
          "baseline_id": "q8-generic",
          "classification": "canonical",
          "synthetic": false,
          "source": {
            "repository": "...", "revision": "<40 hex>",
            "workflow": "...", "run_id": "...", "run_attempt": 1,
            "generated_at_utc": "2026-07-22T12:00:00Z",
            "runner": {"os": "...", "architecture": "arm64",
                       "cpu": "...", "cpu_count": 4}
          },
          "model_family": {"name": "...", "repository": "...",
                           "revision": "..."},
          "runtime": {"name": "llama.cpp", "repository": "...",
                      "revision": "<40 hex>"},
          "optimization_library": {"name": "KleidiAI", "repository": "...",
                                   "version": "v1.24.0",
                                   "source_archive_sha256": "<64 hex>",
                                   "source_archive_size_bytes": 123},
          "evaluation_suite": {"id": "...", "sha256": "<64 hex>"},
          "candidates": [{
            "id": "...", "label": "...", "parameters": {},
            "model": {"family": "...", "repository": "...",
                      "revision": "...", "filename": "...",
                      "sha256": "<64 hex>", "size_bytes": 123,
                      "quantization": "Q4_0"},
            "deployment_argv": ["./llama-server", "--model", "..."],
            "artifacts": {
              "llama_summary": {"path": "...", "sha256": "<64 hex>"},
              "server_evaluation": {"path": "...", "sha256": "<64 hex>"},
              "resource_measurement": {"path": "...", "sha256": "<64 hex>"},
              "throughput_settings": {"path": "...", "sha256": "<64 hex>"},
              "throughput_command": {"path": "...", "sha256": "<64 hex>"},
              "throughput_pass_1": {"path": "...", "sha256": "<64 hex>"},
              "throughput_pass_2": {"path": "...", "sha256": "<64 hex>"},
              "throughput_stderr_pass_1": {"path": "...", "sha256": "<64 hex>"},
              "throughput_stderr_pass_2": {"path": "...", "sha256": "<64 hex>"},
              "server_evaluation_pass_1": {"path": "...", "sha256": "<64 hex>"},
              "server_evaluation_pass_2": {"path": "...", "sha256": "<64 hex>"},
              "server_time_pass_1": {"path": "...", "sha256": "<64 hex>"},
              "server_time_pass_2": {"path": "...", "sha256": "<64 hex>"},
              "server_stderr_pass_1": {"path": "...", "sha256": "<64 hex>"},
              "server_stderr_pass_2": {"path": "...", "sha256": "<64 hex>"}
            }
          }]
        }

    At least three candidates are required.  Different quantizations and model
    hashes are allowed, but every candidate must stay in the declared model
    family/repository/revision and provide its own server evaluation.  Artifact
    paths are relative POSIX paths resolved below the manifest directory.
    """

    path = Path(manifest_path)
    try:
        if path.is_symlink():
            raise _error("experiment manifest must not be a symbolic link")
        resolved_manifest = path.resolve(strict=True)
    except ExperimentAssemblyError:
        raise
    except OSError as exc:
        raise _error(f"could not resolve experiment manifest {path}: {exc}") from exc
    if not resolved_manifest.is_file():
        raise _error(f"experiment manifest is not a regular file: {path}")

    try:
        manifest = load_json_object(resolved_manifest)
    except ValidationError as exc:
        raise _error(f"invalid experiment manifest: {exc}") from exc

    _exact_fields(
        manifest,
        {
            "schema_version",
            "experiment_id",
            "baseline_id",
            "classification",
            "synthetic",
            "source",
            "model_family",
            "runtime",
            "optimization_library",
            "evaluation_suite",
            "candidates",
        },
        "experiment manifest",
    )
    if manifest.get("schema_version") != "1.0":
        raise _error("experiment manifest schema_version must currently be '1.0'")
    experiment_id = _string(manifest.get("experiment_id"), "experiment_id")
    baseline_id = _candidate_id(manifest.get("baseline_id"), "baseline_id")
    classification = _string(manifest.get("classification"), "classification")
    if classification not in {"canonical", "exploratory"}:
        raise _error("classification must be 'canonical' or 'exploratory'")
    if manifest.get("synthetic") is not False:
        raise _error("experiment manifest synthetic must be false")

    source = _validate_source(_object(manifest.get("source"), "source"))
    model_family = _validate_model_family(_object(manifest.get("model_family"), "model_family"))
    runtime = _validate_runtime(_object(manifest.get("runtime"), "runtime"))
    optimization_library = _validate_optimization_library(
        _object(manifest.get("optimization_library"), "optimization_library")
    )
    evaluation_suite = _validate_evaluation_identity(
        _object(manifest.get("evaluation_suite"), "evaluation_suite"),
        "evaluation_suite",
    )

    raw_candidates = manifest.get("candidates")
    if not isinstance(raw_candidates, list):
        raise _error("candidates must be an array")
    if len(raw_candidates) < 3:
        raise _error("candidates must contain at least three candidates")

    root = resolved_manifest.parent
    seen_ids: set[str] = set()
    seen_ids_casefold: set[str] = set()
    seen_paths: set[str] = set()
    candidates: list[Mapping[str, Any]] = []
    evidence: dict[str, Any] = {}
    for index, raw_candidate in enumerate(raw_candidates):
        context = f"candidates[{index}]"
        candidate = _object(raw_candidate, context)
        _exact_fields(
            candidate,
            {
                "id",
                "label",
                "parameters",
                "model",
                "deployment_argv",
                "artifacts",
            },
            context,
        )
        candidate_id = _candidate_id(candidate.get("id"), f"{context}.id")
        if candidate_id in seen_ids or candidate_id.casefold() in seen_ids_casefold:
            raise _error(f"candidate ids must be unique: {candidate_id!r}")
        seen_ids.add(candidate_id)
        seen_ids_casefold.add(candidate_id.casefold())
        label = _string(candidate.get("label"), f"candidate {candidate_id!r} label")
        parameters = _object(candidate.get("parameters"), f"candidate {candidate_id!r} parameters")
        model = _validate_candidate_model(
            _object(candidate.get("model"), f"candidate {candidate_id!r} model"),
            model_family,
            candidate_id,
        )
        argv = _argv(candidate.get("deployment_argv"), candidate_id)
        refs = _validate_artifact_refs(
            _object(candidate.get("artifacts"), f"candidate {candidate_id!r} artifacts"),
            candidate_id,
        )

        artifact_paths: dict[str, Path] = {}
        for artifact_name in sorted(refs):
            ref = refs[artifact_name]
            folded = ref["path"].casefold()
            if folded in seen_paths:
                raise _error(f"artifact paths must be unique across candidates: {ref['path']!r}")
            seen_paths.add(folded)
            artifact_paths[artifact_name] = _resolve_verified_artifact(
                root,
                path_text=ref["path"],
                expected_sha256=ref["sha256"],
                context=f"candidate {candidate_id!r} {artifact_name}",
            )

        loaded = {
            name: _load_json_artifact(
                artifact_paths[name],
                context=f"candidate {candidate_id!r} {name}",
            )
            for name in (
                "llama_summary",
                "server_evaluation",
                "resource_measurement",
                "throughput_settings",
                "throughput_command",
            )
        }
        settings = _validate_throughput_settings(
            loaded["throughput_settings"],
            candidate_id=candidate_id,
            parameters=parameters,
            model=model,
            runtime=runtime,
            deployment_argv=argv,
        )
        _validate_throughput_command(
            loaded["throughput_command"],
            candidate_id=candidate_id,
            settings=settings,
            model=model,
        )

        prompt_tps, generation_tps = _validate_llama_summary(
            loaded["llama_summary"],
            candidate_id=candidate_id,
            model=model,
            runtime=runtime,
        )
        quality_score, ttft_p95, e2e_p95 = _validate_server_evaluation(
            loaded["server_evaluation"],
            candidate_id=candidate_id,
            evaluation_suite=evaluation_suite,
        )
        peak_rss_mib = _validate_resource_measurement(
            loaded["resource_measurement"],
            candidate_id=candidate_id,
            source=source,
            model=model,
            runtime=runtime,
            evaluation_suite=evaluation_suite,
            deployment_argv=argv,
        )
        _verify_llama_summary_from_raw(
            loaded["llama_summary"],
            candidate_id=candidate_id,
            settings=settings,
            refs=refs,
            artifact_paths=artifact_paths,
        )
        _verify_server_evaluation_from_raw(
            loaded["server_evaluation"],
            candidate_id=candidate_id,
            artifact_paths=artifact_paths,
        )
        _verify_resource_measurement_from_raw(
            peak_rss_mib,
            candidate_id=candidate_id,
            artifact_paths=artifact_paths,
        )
        _verify_kleidiai_dispatch_from_raw(
            candidate_id=candidate_id,
            settings=settings,
            artifact_paths=artifact_paths,
        )

        metrics = {
            "prompt_tps": prompt_tps,
            "generation_tps": generation_tps,
            "quality_score": quality_score,
            "ttft_ms_p95": ttft_p95,
            "e2e_latency_ms_p95": e2e_p95,
            "peak_rss_mib": peak_rss_mib,
            "model_size_mib": model["size_bytes"] / (1024.0 * 1024.0),
        }
        candidates.append(
            {
                "id": candidate_id,
                "label": label,
                "parameters": {
                    "configuration": deepcopy(dict(parameters)),
                    "model": deepcopy(model),
                    "runtime": deepcopy(runtime),
                    "deployment_argv": list(argv),
                },
                "metrics": metrics,
            }
        )
        evidence[candidate_id] = {
            "model": deepcopy(model),
            "artifacts": deepcopy(refs),
            "metric_sources": {
                name: {
                    "artifact": (
                        "llama_summary"
                        if name in {"prompt_tps", "generation_tps"}
                        else "server_evaluation"
                        if name in {"quality_score", "ttft_ms_p95", "e2e_latency_ms_p95"}
                        else "resource_measurement"
                        if name == "peak_rss_mib"
                        else "manifest"
                    ),
                    "json_pointer": pointer,
                }
                for name, pointer in _METRIC_SOURCE_POINTERS.items()
            },
        }

    if baseline_id not in seen_ids:
        raise _error("baseline_id must refer to one candidate")

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "baseline_id": baseline_id,
        "synthetic": False,
        "metadata": {
            "experiment_id": experiment_id,
            "classification": classification,
            "source": source,
            "model_family": model_family,
            "runtime": runtime,
            "optimization_library": optimization_library,
            "evaluation_suite": evaluation_suite,
            "manifest": {
                "filename": resolved_manifest.name,
                "sha256": sha256_file(resolved_manifest),
            },
            "candidate_evidence": {key: evidence[key] for key in sorted(evidence)},
        },
        "candidates": sorted(candidates, key=lambda item: str(item["id"])),
    }
    try:
        BenchmarkSet.from_mapping(result)
    except ValidationError as exc:
        raise _error(f"assembled benchmark set is invalid: {exc}") from exc
    return result


def _error(message: str) -> ExperimentAssemblyError:
    return ExperimentAssemblyError(message)


def _exact_fields(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(raw)
    missing = sorted(expected - actual)
    unknown = sorted(str(key) for key in actual - expected)
    details: list[str] = []
    if missing:
        details.append("missing fields: " + ", ".join(missing))
    if unknown:
        details.append("unknown fields: " + ", ".join(unknown))
    if details:
        raise _error(f"{context} has " + "; ".join(details))


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{context} must be an object")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(f"{context} must be a non-empty string")
    if "\x00" in value:
        raise _error(f"{context} must not contain NUL")
    return value.strip()


def _candidate_id(value: Any, context: str) -> str:
    candidate_id = _string(value, context)
    if _CANDIDATE_ID.fullmatch(candidate_id) is None:
        raise _error(
            f"{context} must use 1-64 lowercase letters, digits, dots, underscores, or hyphens"
        )
    return candidate_id


def _sha256(value: Any, context: str) -> str:
    digest = _string(value, context)
    if _SHA256.fullmatch(digest) is None:
        raise _error(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _git_commit(value: Any, context: str) -> str:
    commit = _string(value, context)
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise _error(f"{context} must be a 40-character lowercase Git commit")
    return commit


def _integer(
    value: Any,
    context: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        upper = f" and at most {maximum}" if maximum is not None else ""
        raise _error(f"{context} must be a {qualifier} integer{upper}")
    return value


def _number(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise _error(f"{context} must be finite")
    if strictly_positive and result <= 0:
        raise _error(f"{context} must be greater than zero")
    if minimum is not None and result < minimum:
        raise _error(f"{context} must be at least {minimum:g}")
    if maximum is not None and result > maximum:
        raise _error(f"{context} must be at most {maximum:g}")
    return result


def _string_list(value: Any, context: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        suffix = " a non-empty" if nonempty else " an"
        raise _error(f"{context} must be{suffix} array")
    return [_string(item, f"{context}[{index}]") for index, item in enumerate(value)]


def _validate_source(raw: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(
        raw,
        {
            "repository",
            "revision",
            "workflow",
            "run_id",
            "run_attempt",
            "generated_at_utc",
            "runner",
        },
        "source",
    )
    runner = _object(raw.get("runner"), "source.runner")
    _exact_fields(runner, {"os", "architecture", "cpu", "cpu_count"}, "source.runner")
    architecture = _string(runner.get("architecture"), "source.runner.architecture")
    if architecture != "arm64":
        raise _error("source.runner.architecture must be 'arm64'")
    timestamp = _string(raw.get("generated_at_utc"), "source.generated_at_utc")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _error("source.generated_at_utc must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise _error("source.generated_at_utc must identify UTC")
    return {
        "repository": _string(raw.get("repository"), "source.repository"),
        "revision": _git_commit(raw.get("revision"), "source.revision"),
        "workflow": _string(raw.get("workflow"), "source.workflow"),
        "run_id": _string(raw.get("run_id"), "source.run_id"),
        "run_attempt": _integer(raw.get("run_attempt"), "source.run_attempt", minimum=1),
        "generated_at_utc": timestamp,
        "runner": {
            "os": _string(runner.get("os"), "source.runner.os"),
            "architecture": architecture,
            "cpu": _string(runner.get("cpu"), "source.runner.cpu"),
            "cpu_count": _integer(runner.get("cpu_count"), "source.runner.cpu_count", minimum=1),
        },
    }


def _validate_model_family(raw: Mapping[str, Any]) -> dict[str, str]:
    _exact_fields(raw, {"name", "repository", "revision"}, "model_family")
    return {
        key: _string(raw.get(key), f"model_family.{key}")
        for key in ("name", "repository", "revision")
    }


def _validate_runtime(raw: Mapping[str, Any]) -> dict[str, str]:
    _exact_fields(raw, {"name", "repository", "revision"}, "runtime")
    return {
        "name": _string(raw.get("name"), "runtime.name"),
        "repository": _string(raw.get("repository"), "runtime.repository"),
        "revision": _git_commit(raw.get("revision"), "runtime.revision"),
    }


def _validate_optimization_library(raw: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(
        raw,
        {
            "name",
            "repository",
            "version",
            "source_archive_sha256",
            "source_archive_size_bytes",
        },
        "optimization_library",
    )
    return {
        "name": _string(raw.get("name"), "optimization_library.name"),
        "repository": _string(raw.get("repository"), "optimization_library.repository"),
        "version": _string(raw.get("version"), "optimization_library.version"),
        "source_archive_sha256": _sha256(
            raw.get("source_archive_sha256"),
            "optimization_library.source_archive_sha256",
        ),
        "source_archive_size_bytes": _integer(
            raw.get("source_archive_size_bytes"),
            "optimization_library.source_archive_size_bytes",
            minimum=1,
            maximum=2**63 - 1,
        ),
    }


def _validate_evaluation_identity(raw: Mapping[str, Any], context: str) -> dict[str, str]:
    _exact_fields(raw, {"id", "sha256"}, context)
    return {
        "id": _string(raw.get("id"), f"{context}.id"),
        "sha256": _sha256(raw.get("sha256"), f"{context}.sha256"),
    }


def _validate_candidate_model(
    raw: Mapping[str, Any], model_family: Mapping[str, str], candidate_id: str
) -> dict[str, Any]:
    context = f"candidate {candidate_id!r} model"
    _exact_fields(
        raw,
        {
            "family",
            "repository",
            "revision",
            "filename",
            "sha256",
            "size_bytes",
            "quantization",
        },
        context,
    )
    model = {
        "family": _string(raw.get("family"), f"{context}.family"),
        "repository": _string(raw.get("repository"), f"{context}.repository"),
        "revision": _string(raw.get("revision"), f"{context}.revision"),
        "filename": _string(raw.get("filename"), f"{context}.filename"),
        "sha256": _sha256(raw.get("sha256"), f"{context}.sha256"),
        "size_bytes": _integer(
            raw.get("size_bytes"),
            f"{context}.size_bytes",
            minimum=1,
            maximum=2**63 - 1,
        ),
        "quantization": _string(raw.get("quantization"), f"{context}.quantization"),
    }
    if PurePosixPath(model["filename"]).name != model["filename"] or "\\" in model["filename"]:
        raise _error(f"{context}.filename must be a basename")
    expected = {
        "family": model_family["name"],
        "repository": model_family["repository"],
        "revision": model_family["revision"],
    }
    for key, expected_value in expected.items():
        if model[key] != expected_value:
            raise _error(f"{context}.{key} does not match declared model_family {key}")
    return model


def _argv(value: Any, candidate_id: str) -> list[str]:
    return _argv_vector(value, f"candidate {candidate_id!r} deployment_argv")


def _argv_vector(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 256:
        raise _error(f"{context} must be a non-empty array of at most 256 strings")
    argv: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise _error(f"{context}[{index}] must be a non-empty string")
        if len(item) > 4096 or "\x00" in item or any(character in item for character in "\r\n"):
            raise _error(
                f"{context}[{index}] must be at most 4096 characters without NUL or newlines"
            )
        # This is an argv vector, not a shell command.  Retain each argument
        # exactly instead of trimming or splitting it later.
        argv.append(item)
    return argv


def _validate_artifact_refs(raw: Mapping[str, Any], candidate_id: str) -> dict[str, dict[str, str]]:
    names = {
        "llama_summary",
        "server_evaluation",
        "resource_measurement",
        "throughput_settings",
        "throughput_command",
        "throughput_pass_1",
        "throughput_pass_2",
        "throughput_stderr_pass_1",
        "throughput_stderr_pass_2",
        "server_evaluation_pass_1",
        "server_evaluation_pass_2",
        "server_time_pass_1",
        "server_time_pass_2",
        "server_stderr_pass_1",
        "server_stderr_pass_2",
    }
    _exact_fields(raw, names, f"candidate {candidate_id!r} artifacts")
    refs: dict[str, dict[str, str]] = {}
    for name in sorted(names):
        context = f"candidate {candidate_id!r} artifacts.{name}"
        ref = _object(raw.get(name), context)
        _exact_fields(ref, {"path", "sha256"}, context)
        path = _safe_relative_path(ref.get("path"), f"{context}.path")
        refs[name] = {
            "path": path,
            "sha256": _sha256(ref.get("sha256"), f"{context}.sha256"),
        }
    return refs


def _safe_relative_path(value: Any, context: str) -> str:
    text = _string(value, context)
    if "\\" in text or ":" in text:
        raise _error(f"{context} must be a portable relative POSIX path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != text
    ):
        raise _error(f"{context} must be a portable relative POSIX path")
    return text


def _resolve_verified_artifact(
    root: Path, *, path_text: str, expected_sha256: str, context: str
) -> Path:
    relative = PurePosixPath(path_text)
    candidate = root.joinpath(*relative.parts)
    current = root
    try:
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise _error(f"{context} path must not contain symbolic links")
        resolved = candidate.resolve(strict=True)
    except ExperimentAssemblyError:
        raise
    except OSError as exc:
        raise _error(f"{context} file is missing: {path_text}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise _error(f"{context} must resolve to a regular file below the manifest")
    try:
        if resolved.stat().st_size > _MAX_ARTIFACT_BYTES:
            raise _error(f"{context} exceeds the 32 MiB input limit")
    except OSError as exc:
        raise _error(f"could not inspect {context}: {exc}") from exc
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise _error(f"SHA-256 mismatch for {context}: expected {expected_sha256}, found {actual}")
    return resolved


def _load_json_artifact(path: Path, *, context: str) -> Mapping[str, Any]:
    try:
        return load_json_object(path)
    except ValidationError as exc:
        raise _error(f"invalid {context}: {exc}") from exc


def _validate_throughput_settings(
    raw: Mapping[str, Any],
    *,
    candidate_id: str,
    parameters: Mapping[str, Any],
    model: Mapping[str, Any],
    runtime: Mapping[str, str],
    deployment_argv: Sequence[str],
) -> dict[str, Any]:
    context = f"candidate {candidate_id!r} throughput settings"
    _exact_fields(
        raw,
        {
            "threads",
            "batch_size",
            "ubatch_size",
            "prompt_tokens",
            "generation_tokens",
            "repetitions_per_pass",
            "warmup",
            "cpu_only",
            "model_sha256",
            "llama_cpp_commit",
            "build",
        },
        context,
    )
    settings: dict[str, Any] = {
        name: _integer(raw.get(name), f"{context}.{name}", minimum=1)
        for name in (
            "threads",
            "batch_size",
            "ubatch_size",
            "prompt_tokens",
            "generation_tokens",
            "repetitions_per_pass",
        )
    }
    if raw.get("warmup") is not True:
        raise _error(f"{context}.warmup must be true")
    if raw.get("cpu_only") is not True:
        raise _error(f"{context}.cpu_only must be true")
    settings["warmup"] = True
    settings["cpu_only"] = True
    settings["model_sha256"] = _sha256(raw.get("model_sha256"), f"{context}.model_sha256")
    if settings["model_sha256"] != model["sha256"]:
        raise _error(f"{context}.model_sha256 does not match candidate model")
    settings["llama_cpp_commit"] = _git_commit(
        raw.get("llama_cpp_commit"), f"{context}.llama_cpp_commit"
    )
    if settings["llama_cpp_commit"] != runtime["revision"]:
        raise _error(f"{context}.llama_cpp_commit does not match declared runtime revision")
    build = _object(raw.get("build"), f"{context}.build")
    _exact_fields(build, {"kleidiai"}, f"{context}.build")
    if not isinstance(build.get("kleidiai"), bool):
        raise _error(f"{context}.build.kleidiai must be a boolean")
    settings["build"] = {"kleidiai": build["kleidiai"]}

    parameter_values = {
        name: _integer(
            parameters.get(name),
            f"candidate {candidate_id!r} parameters.{name}",
            minimum=1,
        )
        for name in ("threads", "batch_size", "ubatch_size")
    }
    context_size = _integer(
        parameters.get("context_size"),
        f"candidate {candidate_id!r} parameters.context_size",
        minimum=1,
    )
    if not isinstance(parameters.get("kleidiai"), bool):
        raise _error(f"candidate {candidate_id!r} parameters.kleidiai must be a boolean")
    for name, value in parameter_values.items():
        if value != settings[name]:
            raise _error(
                f"candidate {candidate_id!r} parameters.{name} does not match throughput settings"
            )
    if parameters["kleidiai"] != settings["build"]["kleidiai"]:
        raise _error(
            f"candidate {candidate_id!r} parameters.kleidiai does not match throughput settings"
        )

    integer_options = {
        "--threads": settings["threads"],
        "--threads-batch": settings["threads"],
        "--batch-size": settings["batch_size"],
        "--ubatch-size": settings["ubatch_size"],
        "--ctx-size": context_size,
    }
    for option, expected in integer_options.items():
        actual_text = _argv_option(deployment_argv, option, candidate_id)
        if re.fullmatch(r"[1-9][0-9]*", actual_text) is None or int(actual_text) != expected:
            raise _error(
                f"candidate {candidate_id!r} deployment_argv {option} does not match "
                "throughput settings"
            )
    fixed_options = {
        "--parallel": "1",
        "--n-gpu-layers": "0",
        "-lv": "4",
        "--host": "127.0.0.1",
    }
    for option, expected in fixed_options.items():
        if _argv_option(deployment_argv, option, candidate_id) != expected:
            raise _error(
                f"candidate {candidate_id!r} deployment_argv {option} must be {expected!r}"
            )
    port_text = _argv_option(deployment_argv, "--port", candidate_id)
    if re.fullmatch(r"[1-9][0-9]*", port_text) is None or not 1 <= int(port_text) <= 65535:
        raise _error(f"candidate {candidate_id!r} deployment_argv --port must be from 1 to 65535")
    model_argument = _argv_option(deployment_argv, "--model", candidate_id)
    model_basename = PurePosixPath(model_argument.replace("\\", "/")).name
    if model_basename != model["filename"]:
        raise _error(
            f"candidate {candidate_id!r} deployment_argv --model does not match candidate model"
        )

    executable_parts = {
        part.casefold() for part in PurePosixPath(str(deployment_argv[0]).replace("\\", "/")).parts
    }
    build_markers = {marker for marker in ("generic", "kleidiai") if marker in executable_parts}
    if len(build_markers) != 1:
        raise _error(
            f"candidate {candidate_id!r} deployment_argv executable must identify exactly one "
            "generic or kleidiai build"
        )
    argv_kleidiai = "kleidiai" in build_markers
    if argv_kleidiai != settings["build"]["kleidiai"]:
        raise _error(
            f"candidate {candidate_id!r} deployment_argv build does not match kleidiai setting"
        )
    return settings


def _validate_throughput_command(
    raw: Mapping[str, Any],
    *,
    candidate_id: str,
    settings: Mapping[str, Any],
    model: Mapping[str, Any],
) -> None:
    context = f"candidate {candidate_id!r} throughput command"
    _exact_fields(raw, {"schema_version", "working_directory", "argv"}, context)
    if raw.get("schema_version") != "1.0":
        raise _error(f"{context}.schema_version must currently be '1.0'")
    if raw.get("working_directory") != ".candidate-models":
        raise _error(f"{context}.working_directory must be '.candidate-models'")
    actual = _argv_vector(raw.get("argv"), f"{context}.argv")
    build = "kleidiai" if settings["build"]["kleidiai"] else "generic"
    expected = [
        f"../.candidate-build/{build}/bin/llama-bench",
        "-m",
        str(model["filename"]),
        "-p",
        str(settings["prompt_tokens"]),
        "-n",
        str(settings["generation_tokens"]),
        "-t",
        str(settings["threads"]),
        "-b",
        str(settings["batch_size"]),
        "-ub",
        str(settings["ubatch_size"]),
        "-r",
        str(settings["repetitions_per_pass"]),
        "-dev",
        "none",
        "-ngl",
        "0",
        "-nopo",
        "1",
        "-v",
        "-o",
        "jsonl",
    ]
    if actual != expected:
        raise _error(f"{context}.argv does not match the recorded settings and model")


def _argv_option(argv: Sequence[str], option: str, candidate_id: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise _error(
            f"candidate {candidate_id!r} deployment_argv must contain exactly one {option} value"
        )
    value = argv[positions[0] + 1]
    if value.startswith("--"):
        raise _error(f"candidate {candidate_id!r} deployment_argv {option} is missing its value")
    return value


def _verify_llama_summary_from_raw(
    aggregate: Mapping[str, Any],
    *,
    candidate_id: str,
    settings: Mapping[str, Any],
    refs: Mapping[str, Mapping[str, str]],
    artifact_paths: Mapping[str, Path],
) -> None:
    labeled_paths = [
        ("pass-1", artifact_paths["throughput_pass_1"]),
        ("pass-2", artifact_paths["throughput_pass_2"]),
    ]
    try:
        summary = summarize_llama_bench_paths(
            candidate_id,
            labeled_paths,
            settings=settings,
        )
    except ValidationError as exc:
        raise _error(
            f"candidate {candidate_id!r} raw throughput evidence is invalid: {exc}"
        ) from exc
    rebuilt = summary.to_mapping()
    rebuilt["input_fingerprints"] = {
        "settings_sha256": refs["throughput_settings"]["sha256"],
        "artifacts_sha256": {
            "pass-1": refs["throughput_pass_1"]["sha256"],
            "pass-2": refs["throughput_pass_2"]["sha256"],
        },
    }
    tests = _object(rebuilt.get("tests"), f"candidate {candidate_id!r} rebuilt throughput tests")
    expected_shapes = {
        "pp": (settings["prompt_tokens"], 0),
        "tg": (0, settings["generation_tokens"]),
    }
    expected_sample_count = settings["repetitions_per_pass"] * len(labeled_paths)
    for test_kind, expected_shape in expected_shapes.items():
        test = _object(tests.get(test_kind), f"candidate {candidate_id!r} rebuilt {test_kind}")
        if (test.get("n_prompt"), test.get("n_gen")) != expected_shape:
            raise _error(
                f"candidate {candidate_id!r} raw {test_kind} shape does not match throughput settings"
            )
        for metric in ("tokens_per_second", "duration_ns"):
            statistics_value = _object(
                test.get(metric), f"candidate {candidate_id!r} rebuilt {test_kind}.{metric}"
            )
            if statistics_value.get("sample_count") != expected_sample_count:
                raise _error(
                    f"candidate {candidate_id!r} raw {test_kind} sample count does not match "
                    "two throughput passes and repetitions_per_pass"
                )

    canonical_aggregate = _canonicalize_llama_summary_paths(
        aggregate,
        refs=refs,
        candidate_id=candidate_id,
    )
    canonical_rebuilt = _canonicalize_llama_summary_paths(
        rebuilt,
        refs=refs,
        candidate_id=candidate_id,
    )
    if canonical_aggregate != canonical_rebuilt:
        raise _error(
            f"candidate {candidate_id!r} llama summary does not exactly match recomputed raw evidence"
        )


def _canonicalize_llama_summary_paths(
    raw: Mapping[str, Any],
    *,
    refs: Mapping[str, Mapping[str, str]],
    candidate_id: str,
) -> dict[str, Any]:
    canonical = deepcopy(dict(raw))
    label_paths = {
        "pass-1": refs["throughput_pass_1"]["path"],
        "pass-2": refs["throughput_pass_2"]["path"],
    }

    def canonicalize(container: Mapping[str, Any], context: str) -> None:
        labels = _string_list(container.get("source_labels"), f"{context}.source_labels")
        files = _string_list(container.get("source_files"), f"{context}.source_files")
        if len(files) != len(labels) or any(label not in label_paths for label in labels):
            raise _error(f"{context} source paths cannot be matched to pass labels")
        assert isinstance(container, dict)
        container["source_files"] = [label_paths[label] for label in labels]

    canonicalize(canonical, f"candidate {candidate_id!r} llama summary")
    tests = _object(canonical.get("tests"), f"candidate {candidate_id!r} llama summary.tests")
    for test_kind, test_value in tests.items():
        test = _object(test_value, f"candidate {candidate_id!r} llama summary.tests.{test_kind}")
        canonicalize(test, f"candidate {candidate_id!r} llama summary.tests.{test_kind}")
    return canonical


def _verify_server_evaluation_from_raw(
    aggregate: Mapping[str, Any],
    *,
    candidate_id: str,
    artifact_paths: Mapping[str, Path],
) -> None:
    try:
        rebuilt = pool_server_evaluations(
            [
                artifact_paths["server_evaluation_pass_1"],
                artifact_paths["server_evaluation_pass_2"],
            ]
        )
    except ValidationError as exc:
        raise _error(f"candidate {candidate_id!r} raw server evidence is invalid: {exc}") from exc
    if aggregate != rebuilt:
        raise _error(
            f"candidate {candidate_id!r} server evaluation does not exactly match pooled raw evidence"
        )


def _verify_resource_measurement_from_raw(
    aggregate_peak_rss_mib: float,
    *,
    candidate_id: str,
    artifact_paths: Mapping[str, Path],
) -> None:
    try:
        pass_peaks = [
            parse_gnu_time_peak_rss(artifact_paths["server_time_pass_1"]),
            parse_gnu_time_peak_rss(artifact_paths["server_time_pass_2"]),
        ]
    except ValidationError as exc:
        raise _error(f"candidate {candidate_id!r} raw GNU time evidence is invalid: {exc}") from exc
    if aggregate_peak_rss_mib != max(pass_peaks):
        raise _error(
            f"candidate {candidate_id!r} resource measurement maximum does not match raw GNU time"
        )


def _verify_kleidiai_dispatch_from_raw(
    *,
    candidate_id: str,
    settings: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
) -> None:
    marker = b"CPU_KLEIDIAI model buffer"
    names = (
        "throughput_stderr_pass_1",
        "throughput_stderr_pass_2",
        "server_stderr_pass_1",
        "server_stderr_pass_2",
    )
    observed: dict[str, bool] = {}
    for name in names:
        try:
            observed[name] = marker in artifact_paths[name].read_bytes()
        except OSError as exc:
            raise _error(
                f"candidate {candidate_id!r} could not read {name} dispatch evidence: {exc}"
            ) from exc
    if settings["build"]["kleidiai"]:
        missing = [name for name, present in observed.items() if not present]
        if missing:
            raise _error(
                f"candidate {candidate_id!r} is missing KleidiAI dispatch marker in: "
                + ", ".join(missing)
            )
    else:
        unexpected = [name for name, present in observed.items() if present]
        if unexpected:
            raise _error(
                f"candidate {candidate_id!r} unexpectedly dispatched KleidiAI in: "
                + ", ".join(unexpected)
            )


def _validate_llama_summary(
    raw: Mapping[str, Any],
    *,
    candidate_id: str,
    model: Mapping[str, Any],
    runtime: Mapping[str, str],
) -> tuple[float, float]:
    context = f"candidate {candidate_id!r} llama summary"
    required_fields = {
        "schema_version",
        "label",
        "build_commit",
        "model_filename",
        "settings",
        "synthetic_fixture",
        "source_files",
        "source_labels",
        "tests",
    }
    actual_fields = set(raw)
    missing = sorted(required_fields - actual_fields)
    unknown = sorted(actual_fields - required_fields - {"input_fingerprints"})
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        if unknown:
            details.append("unknown fields: " + ", ".join(unknown))
        raise _error(f"{context} has " + "; ".join(details))
    if raw.get("schema_version") != "1.0":
        raise _error(f"{context} schema_version must be '1.0'")
    if _string(raw.get("label"), f"{context}.label") != candidate_id:
        raise _error(f"{context} label does not match candidate id")
    build_commit = _string(raw.get("build_commit"), f"{context}.build_commit")
    if (
        len(build_commit) < 7
        or re.fullmatch(r"[0-9a-f]+", build_commit) is None
        or not runtime["revision"].startswith(build_commit)
    ):
        raise _error(f"{context} build_commit does not match declared runtime revision")
    if _string(raw.get("model_filename"), f"{context}.model_filename") != model["filename"]:
        raise _error(f"{context} model_filename does not match candidate model")
    _object(raw.get("settings"), f"{context}.settings")
    if raw.get("synthetic_fixture") is not False:
        raise _error(f"{context} synthetic_fixture must be false")
    _string_list(raw.get("source_files"), f"{context}.source_files")
    source_labels = _string_list(raw.get("source_labels"), f"{context}.source_labels")
    if "input_fingerprints" in raw:
        fingerprints = _object(raw.get("input_fingerprints"), f"{context}.input_fingerprints")
        _exact_fields(
            fingerprints,
            {"settings_sha256", "artifacts_sha256"},
            f"{context}.input_fingerprints",
        )
        _sha256(
            fingerprints.get("settings_sha256"),
            f"{context}.input_fingerprints.settings_sha256",
        )
        artifact_hashes = _object(
            fingerprints.get("artifacts_sha256"),
            f"{context}.input_fingerprints.artifacts_sha256",
        )
        if set(artifact_hashes) != set(source_labels):
            raise _error(f"{context}.input_fingerprints artifact labels must match source_labels")
        for label, digest in artifact_hashes.items():
            _sha256(
                digest,
                f"{context}.input_fingerprints.artifacts_sha256.{label}",
            )

    tests = _object(raw.get("tests"), f"{context}.tests")
    unknown_tests = sorted(set(tests) - {"pp", "tg", "pg"})
    if unknown_tests:
        raise _error(f"{context}.tests contains unknown fields: {', '.join(unknown_tests)}")
    missing_tests = sorted({"pp", "tg"} - set(tests))
    if missing_tests:
        raise _error(f"{context}.tests is missing metrics: {', '.join(missing_tests)}")
    throughput: dict[str, float] = {}
    for test_kind in sorted(tests):
        test = _object(tests[test_kind], f"{context}.tests.{test_kind}")
        throughput[test_kind] = _validate_llama_test(test, test_kind, context)
    return throughput["pp"], throughput["tg"]


def _validate_llama_test(raw: Mapping[str, Any], test_kind: str, parent_context: str) -> float:
    context = f"{parent_context}.tests.{test_kind}"
    _exact_fields(
        raw,
        {
            "test_kind",
            "n_prompt",
            "n_gen",
            "tokens_per_second",
            "duration_ns",
            "source_files",
            "source_labels",
        },
        context,
    )
    if raw.get("test_kind") != test_kind:
        raise _error(f"{context}.test_kind must be {test_kind!r}")
    n_prompt = _integer(raw.get("n_prompt"), f"{context}.n_prompt")
    n_gen = _integer(raw.get("n_gen"), f"{context}.n_gen")
    if test_kind == "pp" and not (n_prompt > 0 and n_gen == 0):
        raise _error(f"{context} must describe a prompt-only workload")
    if test_kind == "tg" and not (n_prompt == 0 and n_gen > 0):
        raise _error(f"{context} must describe a generation-only workload")
    if test_kind == "pg" and not (n_prompt > 0 and n_gen > 0):
        raise _error(f"{context} must describe a prompt-plus-generation workload")
    tps = _validate_statistics(
        _object(raw.get("tokens_per_second"), f"{context}.tokens_per_second"),
        f"{context}.tokens_per_second",
        strictly_positive=True,
    )
    _validate_statistics(
        _object(raw.get("duration_ns"), f"{context}.duration_ns"),
        f"{context}.duration_ns",
        strictly_positive=True,
    )
    _string_list(raw.get("source_files"), f"{context}.source_files")
    _string_list(raw.get("source_labels"), f"{context}.source_labels")
    return tps["median"]


def _validate_statistics(
    raw: Mapping[str, Any], context: str, *, strictly_positive: bool
) -> dict[str, float]:
    _exact_fields(raw, {"sample_count", "mean", "median", "sample_stdev", "min", "max"}, context)
    _integer(raw.get("sample_count"), f"{context}.sample_count", minimum=1)
    values = {
        name: _number(
            raw.get(name),
            f"{context}.{name}",
            strictly_positive=strictly_positive and name != "sample_stdev",
            minimum=0.0 if name == "sample_stdev" else None,
        )
        for name in ("mean", "median", "sample_stdev", "min", "max")
    }
    if not values["min"] <= values["median"] <= values["max"]:
        raise _error(f"{context} median must fall between min and max")
    if not values["min"] <= values["mean"] <= values["max"]:
        raise _error(f"{context} mean must fall between min and max")
    return values


def _validate_server_evaluation(
    raw: Mapping[str, Any],
    *,
    candidate_id: str,
    evaluation_suite: Mapping[str, str],
) -> tuple[float, float, float]:
    context = f"candidate {candidate_id!r} server evaluation"
    _exact_fields(
        raw,
        {"schema_version", "candidate_id", "synthetic", "suite", "quality", "latency"},
        context,
    )
    if raw.get("schema_version") != "1.0":
        raise _error(f"{context} schema_version must be '1.0'")
    if _string(raw.get("candidate_id"), f"{context}.candidate_id") != candidate_id:
        raise _error(f"{context} candidate_id does not match candidate id")
    if raw.get("synthetic") is not False:
        raise _error(f"{context} synthetic must be false")

    suite = _object(raw.get("suite"), f"{context}.suite")
    _exact_fields(
        suite,
        {
            "id",
            "license",
            "sha256",
            "quality_case_count",
            "performance_repetitions",
            "performance_warmups",
            "generation_tokens",
            "cache_prompt",
            "seed",
            "temperature",
        },
        f"{context}.suite",
    )
    if _string(suite.get("id"), f"{context}.suite.id") != evaluation_suite["id"]:
        raise _error(f"{context} suite id does not match declared evaluation suite")
    if _sha256(suite.get("sha256"), f"{context}.suite.sha256") != evaluation_suite["sha256"]:
        raise _error(f"{context} suite sha256 does not match declared evaluation suite")
    _string(suite.get("license"), f"{context}.suite.license")
    case_count = _integer(
        suite.get("quality_case_count"), f"{context}.suite.quality_case_count", minimum=1
    )
    repetitions = _integer(
        suite.get("performance_repetitions"),
        f"{context}.suite.performance_repetitions",
        minimum=1,
    )
    _integer(
        suite.get("performance_warmups"),
        f"{context}.suite.performance_warmups",
        minimum=1,
    )
    _integer(suite.get("generation_tokens"), f"{context}.suite.generation_tokens", minimum=1)
    if suite.get("cache_prompt") is not False:
        raise _error(f"{context}.suite.cache_prompt must be false")
    _integer(suite.get("seed"), f"{context}.suite.seed")
    if _number(suite.get("temperature"), f"{context}.suite.temperature") != 0.0:
        raise _error(f"{context}.suite.temperature must be zero")

    quality = _object(raw.get("quality"), f"{context}.quality")
    _exact_fields(quality, {"method", "score", "passed", "total", "cases"}, f"{context}.quality")
    _string(quality.get("method"), f"{context}.quality.method")
    score = _number(quality.get("score"), f"{context}.quality.score", minimum=0.0, maximum=1.0)
    passed = _integer(quality.get("passed"), f"{context}.quality.passed")
    total = _integer(quality.get("total"), f"{context}.quality.total", minimum=1)
    if total != case_count or passed > total:
        raise _error(f"{context} quality counts do not match the evaluation suite")
    if not math.isclose(score, passed / total, rel_tol=1e-12, abs_tol=1e-12):
        raise _error(f"{context} quality score does not match passed/total")
    cases = quality.get("cases")
    if not isinstance(cases, list) or len(cases) != total:
        raise _error(f"{context}.quality.cases must contain exactly {total} cases")
    matched_count = 0
    case_ids: set[str] = set()
    for index, case_value in enumerate(cases):
        case = _object(case_value, f"{context}.quality.cases[{index}]")
        _exact_fields(
            case,
            {"id", "prompt", "accepted_answers", "response", "matched", "matched_answer"},
            f"{context}.quality.cases[{index}]",
        )
        case_id = _string(case.get("id"), f"{context}.quality.cases[{index}].id")
        if case_id in case_ids:
            raise _error(f"{context}.quality.cases contains duplicate id {case_id!r}")
        case_ids.add(case_id)
        _string(case.get("prompt"), f"{context}.quality.cases[{index}].prompt")
        accepted_answers = _string_list(
            case.get("accepted_answers"),
            f"{context}.quality.cases[{index}].accepted_answers",
        )
        response = case.get("response")
        if not isinstance(response, str):
            raise _error(f"{context}.quality.cases[{index}].response must be a string")
        matched = case.get("matched")
        if not isinstance(matched, bool):
            raise _error(f"{context}.quality.cases[{index}].matched must be a boolean")
        matched_answer = case.get("matched_answer")
        if matched and not isinstance(matched_answer, str):
            raise _error(
                f"{context}.quality.cases[{index}].matched_answer must be a string when matched"
            )
        if not matched and matched_answer is not None:
            raise _error(
                f"{context}.quality.cases[{index}].matched_answer must be null when unmatched"
            )
        normalized_response = _normalize_quality_answer(response)
        computed_answer = next(
            (
                answer
                for answer in accepted_answers
                if normalized_response == _normalize_quality_answer(answer)
            ),
            None,
        )
        if matched != (computed_answer is not None):
            raise _error(
                f"{context}.quality.cases[{index}].matched does not agree with the response"
            )
        if matched_answer != computed_answer:
            raise _error(
                f"{context}.quality.cases[{index}].matched_answer does not agree with the response"
            )
        matched_count += int(matched)
    if matched_count != passed:
        raise _error(f"{context} matched case count does not equal quality.passed")

    latency = _object(raw.get("latency"), f"{context}.latency")
    _exact_fields(
        latency,
        {
            "method",
            "ttft_ms_p50",
            "ttft_ms_p95",
            "e2e_latency_ms_p50",
            "e2e_latency_ms_p95",
            "samples",
        },
        f"{context}.latency",
    )
    _string(latency.get("method"), f"{context}.latency.method")
    samples = latency.get("samples")
    if not isinstance(samples, list) or len(samples) != repetitions:
        raise _error(f"{context}.latency.samples must contain exactly {repetitions} samples")
    ttft_values: list[float] = []
    e2e_values: list[float] = []
    for index, sample_value in enumerate(samples, start=1):
        sample = _object(sample_value, f"{context}.latency.samples[{index - 1}]")
        _exact_fields(
            sample,
            {
                "index",
                "ttft_ms",
                "e2e_latency_ms",
                "event_count",
                "predicted_tokens",
                "content",
            },
            f"{context}.latency.samples[{index - 1}]",
        )
        if (
            _integer(
                sample.get("index"), f"{context}.latency.samples[{index - 1}].index", minimum=1
            )
            != index
        ):
            raise _error(f"{context}.latency sample indexes must be consecutive from one")
        ttft = _number(
            sample.get("ttft_ms"),
            f"{context}.latency.samples[{index - 1}].ttft_ms",
            strictly_positive=True,
        )
        e2e = _number(
            sample.get("e2e_latency_ms"),
            f"{context}.latency.samples[{index - 1}].e2e_latency_ms",
            strictly_positive=True,
        )
        if e2e < ttft:
            raise _error(f"{context}.latency sample e2e latency cannot be below TTFT")
        _integer(
            sample.get("event_count"),
            f"{context}.latency.samples[{index - 1}].event_count",
            minimum=1,
        )
        if (
            _integer(
                sample.get("predicted_tokens"),
                f"{context}.latency.samples[{index - 1}].predicted_tokens",
                minimum=1,
            )
            != suite["generation_tokens"]
        ):
            raise _error(
                f"{context}.latency.samples[{index - 1}].predicted_tokens must equal "
                "suite generation_tokens"
            )
        _string(sample.get("content"), f"{context}.latency.samples[{index - 1}].content")
        ttft_values.append(ttft)
        e2e_values.append(e2e)

    expected = {
        "ttft_ms_p50": float(statistics.median(ttft_values)),
        "ttft_ms_p95": _nearest_rank(ttft_values, 95),
        "e2e_latency_ms_p50": float(statistics.median(e2e_values)),
        "e2e_latency_ms_p95": _nearest_rank(e2e_values, 95),
    }
    actual: dict[str, float] = {}
    for name, expected_value in expected.items():
        value = _number(latency.get(name), f"{context}.latency.{name}", strictly_positive=True)
        if not math.isclose(value, expected_value, rel_tol=1e-9, abs_tol=1e-9):
            raise _error(f"{context}.latency.{name} does not match raw samples")
        actual[name] = value
    return score, actual["ttft_ms_p95"], actual["e2e_latency_ms_p95"]


def _nearest_rank(values: Sequence[float], percentile: int) -> float:
    ordered = sorted(values)
    rank = math.ceil((percentile / 100.0) * len(ordered))
    return float(ordered[max(0, rank - 1)])


def _validate_resource_measurement(
    raw: Mapping[str, Any],
    *,
    candidate_id: str,
    source: Mapping[str, Any],
    model: Mapping[str, Any],
    runtime: Mapping[str, str],
    evaluation_suite: Mapping[str, str],
    deployment_argv: Sequence[str],
) -> float:
    context = f"candidate {candidate_id!r} resource measurement"
    _exact_fields(
        raw,
        {
            "schema_version",
            "candidate_id",
            "synthetic",
            "run_id",
            "model",
            "runtime",
            "evaluation_suite",
            "deployment_argv",
            "measurement_tool",
            "maximum_resident_set_kbytes",
        },
        context,
    )
    if raw.get("schema_version") != "1.0":
        raise _error(f"{context} schema_version must be '1.0'")
    if _string(raw.get("candidate_id"), f"{context}.candidate_id") != candidate_id:
        raise _error(f"{context} candidate_id does not match candidate id")
    if raw.get("synthetic") is not False:
        raise _error(f"{context} synthetic must be false")
    if _string(raw.get("run_id"), f"{context}.run_id") != source["run_id"]:
        raise _error(f"{context} run_id does not match source run")

    measured_model = _object(raw.get("model"), f"{context}.model")
    expected_model_fields = {"family", "repository", "revision", "filename", "sha256"}
    _exact_fields(measured_model, expected_model_fields, f"{context}.model")
    for field in sorted(expected_model_fields):
        if _string(measured_model.get(field), f"{context}.model.{field}") != model[field]:
            raise _error(f"{context} model {field} does not match candidate model")

    measured_runtime = _object(raw.get("runtime"), f"{context}.runtime")
    _exact_fields(measured_runtime, {"name", "repository", "revision"}, f"{context}.runtime")
    for field in ("name", "repository", "revision"):
        if _string(measured_runtime.get(field), f"{context}.runtime.{field}") != runtime[field]:
            raise _error(f"{context} runtime {field} does not match declared runtime")

    measured_suite = _validate_evaluation_identity(
        _object(raw.get("evaluation_suite"), f"{context}.evaluation_suite"),
        f"{context}.evaluation_suite",
    )
    if measured_suite != evaluation_suite:
        raise _error(f"{context} evaluation_suite does not match declared suite")
    measured_argv = _argv(raw.get("deployment_argv"), candidate_id)
    if measured_argv != list(deployment_argv):
        raise _error(f"{context} deployment_argv does not match candidate command")
    tool = _string(raw.get("measurement_tool"), f"{context}.measurement_tool")
    if tool not in {"GNU time -v", "/usr/bin/time -v"}:
        raise _error(f"{context}.measurement_tool must identify GNU time -v")
    maximum_kib = _integer(
        raw.get("maximum_resident_set_kbytes"),
        f"{context}.maximum_resident_set_kbytes",
        minimum=1,
        maximum=2**63 - 1,
    )
    return maximum_kib / 1024.0


def _normalize_quality_answer(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())
