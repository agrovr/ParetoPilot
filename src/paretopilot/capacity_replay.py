"""Replay an extracted supplementary capacity bundle without rerunning inference."""

from __future__ import annotations

from contextlib import suppress
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

from paretopilot.capacity_eval import assemble_capacity_study, load_capacity_plan
from paretopilot.capacity_receipt import render_capacity_receipt
from paretopilot.domain import ValidationError
from paretopilot.io import load_json_object, sha256_file, write_json, write_text
from paretopilot.replay import replay_evidence


_SHA256_LINE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_REQUIRED_PATHS = {
    "capacity-plan.json",
    "capacity-receipt.md",
    "capacity-study.json",
    "canonical/frozen-SHA256SUMS",
    "canonical/release-archive.sha256",
    "canonical/replay.json",
    "evaluation-suite.json",
    "load-plan.json",
    "manifest.json",
    "status.json",
}
_STATUS_FIELDS = {
    "canonical_outputs_modified",
    "classification",
    "eligible_cell_counts",
    "measurement_valid",
    "reason",
    "schema_version",
    "selected_operating_points",
    "status",
}
_CANONICAL_TRUE_FIELDS = (
    "authoritative_outputs_match",
    "decision_reproduced",
    "fully_reproduced",
    "report_matches_archive",
    "status_complete",
    "valid",
)
_CANONICAL_STABLE_FIELDS = (
    "authoritative_comparisons",
    "authoritative_outputs_match",
    "candidate_count",
    "checksums",
    "decision_reproduced",
    "differences",
    "fully_reproduced",
    "generated_files",
    "policy_profile_count",
    "policy_selected_ids",
    "replay_contract",
    "report_matches_archive",
    "schema_version",
    "selected_id",
    "status_complete",
    "valid",
    "warnings",
)
_MAX_ARCHIVE_ENTRIES = 10_000
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024


def replay_capacity_bundle(
    bundle_dir: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Verify and reproduce one extracted supplementary capacity bundle.

    The measured workload is not rerun. Every archived regular file must have
    exact ``SHA256SUMS`` coverage. The capacity study is rebuilt from the raw
    load, RSS, server-log, and quality artifacts; its Markdown receipt is then
    regenerated. Both derived files must be byte-identical to the archive.

    The embedded frozen canonical archive is also safely extracted and replayed
    with :func:`paretopilot.replay.replay_evidence`. Only a new output directory
    is published, so the source bundle remains unchanged.
    """

    root = _resolve_bundle_directory(bundle_dir)
    destination = _validate_output_path(root, output_dir)
    checksums, checksum_manifest_sha256 = _verify_checksums(root)
    status = _load_capacity_status(root)

    plan_path = _required_file(root, "capacity-plan.json")
    load_plan_path = _required_file(root, "load-plan.json")
    manifest_path = _required_file(root, "manifest.json")
    authoritative_study_path = _required_file(root, "capacity-study.json")
    authoritative_receipt_path = _required_file(root, "capacity-receipt.md")
    plan = load_capacity_plan(plan_path)
    load_artifacts, rss_artifacts, server_logs, quality_artifacts = _capacity_sources(
        root,
        plan,
    )

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                dir=destination.parent,
                prefix=f".{destination.name}.capacity-replay-",
            )
        )
    except OSError as exc:
        raise ValidationError(
            f"could not create capacity replay staging directory for {destination}: {exc}"
        ) from exc

    try:
        study = assemble_capacity_study(
            plan_path=plan_path,
            load_plan_path=load_plan_path,
            manifest_path=manifest_path,
            load_artifacts=load_artifacts,
            rss_artifacts=rss_artifacts,
            server_logs=server_logs,
            quality_artifacts=quality_artifacts,
        )
        _validate_status_against_study(status, study)

        reproduced_study = staging / "capacity-study.reproduced.json"
        reproduced_receipt = staging / "capacity-receipt.reproduced.md"
        write_json(reproduced_study, study)
        write_text(reproduced_receipt, render_capacity_receipt(study))

        study_comparison = _require_byte_identical(
            authoritative_study_path,
            reproduced_study,
            authoritative_name="capacity-study.json",
        )
        receipt_comparison = _require_byte_identical(
            authoritative_receipt_path,
            reproduced_receipt,
            authoritative_name="capacity-receipt.md",
        )

        manifest = load_json_object(manifest_path)
        canonical_replay = _verify_canonical_replay(
            root,
            manifest=manifest,
            study=study,
            work_parent=destination.parent,
        )
        selected_points = _selected_operating_points(study)
        payload: Mapping[str, Any] = {
            "schema_version": "1.0",
            "classification": "supplementary-capacity-replay",
            "valid": True,
            "status_complete": True,
            "checksums": {
                "verified": True,
                "entry_count": len(checksums),
                "manifest_sha256": checksum_manifest_sha256,
            },
            "authoritative_comparisons": {
                "capacity-study": study_comparison,
                "capacity-receipt": receipt_comparison,
            },
            "capacity_study_reproduced": True,
            "capacity_receipt_reproduced": True,
            "canonical_replay": canonical_replay,
            "selected_operating_points": selected_points,
            "generated_files": [
                "capacity-study.reproduced.json",
                "capacity-receipt.reproduced.md",
            ],
            "verdict": (
                "PASS: bundle verified, capacity outputs reproduced byte-for-byte, "
                "and the published v1.1 result replayed."
            ),
        }
        write_json(staging / "capacity-replay.json", payload)
        if destination.exists() or destination.is_symlink():
            raise ValidationError(f"refusing to overwrite existing output directory: {destination}")
        os.replace(staging, destination)
        return payload
    except ValidationError:
        raise
    except (OSError, OverflowError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        raise ValidationError(f"capacity replay failed: {exc}") from exc
    finally:
        if staging.exists():
            with suppress(OSError):
                shutil.rmtree(staging)


def _resolve_bundle_directory(path: Path) -> Path:
    candidate = Path(path)
    try:
        if candidate.is_symlink():
            raise ValidationError("capacity bundle directory must not be a symbolic link")
        resolved = candidate.resolve(strict=True)
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(
            f"could not resolve capacity bundle directory {candidate}: {exc}"
        ) from exc
    if not resolved.is_dir():
        raise ValidationError(f"capacity bundle path is not a directory: {candidate}")
    return resolved


def _validate_output_path(bundle_root: Path, output_dir: Path) -> Path:
    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise ValidationError(f"refusing to overwrite existing output directory: {target}")
    try:
        resolved = target.resolve(strict=False)
    except OSError as exc:
        raise ValidationError(f"could not resolve output directory {target}: {exc}") from exc
    if (
        resolved == bundle_root
        or resolved.is_relative_to(bundle_root)
        or bundle_root.is_relative_to(resolved)
    ):
        raise ValidationError(
            "output directory must not contain or be contained by the capacity bundle"
        )
    return resolved


def _safe_relative_path(encoded: str, *, context: str) -> PurePosixPath:
    if "\\" in encoded or ":" in encoded:
        raise ValidationError(f"{context} contains an unsafe path {encoded!r}")
    normalized = encoded[2:] if encoded.startswith("./") else encoded
    path = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("./")
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != normalized
    ):
        raise ValidationError(f"{context} contains an unsafe path {encoded!r}")
    return path


def _bundle_files(root: Path) -> set[str]:
    files: set[str] = set()
    try:
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValidationError(f"capacity bundle contains a symbolic link: {relative}")
            if path.is_file():
                files.add(relative)
            elif not path.is_dir():
                raise ValidationError(f"capacity bundle contains a non-regular entry: {relative}")
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(f"could not enumerate capacity bundle {root}: {exc}") from exc
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
    for line_number, line in enumerate(raw.splitlines(), start=1):
        match = _SHA256_LINE.fullmatch(line)
        if match is None:
            raise ValidationError(f"SHA256SUMS line {line_number} is malformed")
        expected_digest, encoded = match.groups()
        relative = _safe_relative_path(
            encoded,
            context=f"SHA256SUMS line {line_number}",
        )
        relative_text = relative.as_posix()
        if relative_text == "SHA256SUMS":
            raise ValidationError("SHA256SUMS must not contain an entry for itself")
        if relative_text in entries or relative_text.casefold() in casefold_paths:
            raise ValidationError(f"SHA256SUMS contains duplicate entry {relative_text!r}")
        casefold_paths.add(relative_text.casefold())
        file_path = _required_file(root, relative_text)
        actual_digest = sha256_file(file_path)
        if actual_digest != expected_digest:
            raise ValidationError(
                f"SHA256 mismatch for {relative_text}: expected {expected_digest}, "
                f"found {actual_digest}"
            )
        entries[relative_text] = expected_digest

    actual_files = _bundle_files(root)
    listed_files = set(entries)
    unlisted = sorted(actual_files - listed_files)
    missing = sorted(listed_files - actual_files)
    if unlisted:
        raise ValidationError(
            "capacity bundle contains files missing from SHA256SUMS: " + ", ".join(unlisted)
        )
    if missing:
        raise ValidationError(
            "SHA256SUMS contains files missing from the capacity bundle: " + ", ".join(missing)
        )
    absent_required = sorted(_REQUIRED_PATHS - listed_files)
    if absent_required:
        raise ValidationError(
            "capacity bundle is missing required checksummed files: " + ", ".join(absent_required)
        )
    return dict(sorted(entries.items())), sha256_file(checksum_path)


def _required_file(root: Path, relative_text: str) -> Path:
    relative = _safe_relative_path(relative_text, context="capacity bundle path")
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValidationError(f"capacity bundle file is missing: {relative_text}") from exc
    if not resolved.is_relative_to(root) or candidate.is_symlink() or not candidate.is_file():
        raise ValidationError(f"capacity bundle path is not a safe regular file: {relative_text}")
    return candidate


def _safe_id(value: str, *, context: str) -> str:
    if _SAFE_ID.fullmatch(value) is None:
        raise ValidationError(f"{context} is unsafe for a capacity bundle path: {value!r}")
    return value


def _capacity_sources(
    root: Path,
    plan: Any,
) -> tuple[
    list[tuple[str, Path]],
    list[tuple[str, Path]],
    list[tuple[str, Path]],
    list[tuple[str, Path]],
]:
    load_artifacts: list[tuple[str, Path]] = []
    rss_artifacts: list[tuple[str, Path]] = []
    server_logs: list[tuple[str, Path]] = []
    quality_artifacts: list[tuple[str, Path]] = []
    for pass_spec in plan.passes:
        pass_id = _safe_id(str(pass_spec.id), context="capacity pass id")
        for candidate_value in pass_spec.candidate_order:
            candidate_id = _safe_id(str(candidate_value), context="capacity candidate id")
            for parallel in pass_spec.server_parallel_order:
                label = f"{pass_id}/{candidate_id}/p{int(parallel)}"
                base = f"runs/{pass_id}/{candidate_id}/p{int(parallel)}"
                load_artifacts.append((label, _required_file(root, f"{base}/load-evaluation.json")))
                rss_artifacts.append((label, _required_file(root, f"{base}/server-time.txt")))
                server_logs.append((label, _required_file(root, f"{base}/server.stderr.log")))
    for candidate in plan.candidates:
        candidate_id = _safe_id(str(candidate.id), context="capacity candidate id")
        for parallel in plan.server_parallel_levels:
            label = f"{candidate_id}/p{int(parallel)}"
            quality_artifacts.append(
                (
                    label,
                    _required_file(
                        root,
                        (f"quality/{candidate_id}/p{int(parallel)}/quality-evidence.json"),
                    ),
                )
            )
    return load_artifacts, rss_artifacts, server_logs, quality_artifacts


def _load_capacity_status(root: Path) -> Mapping[str, Any]:
    status = load_json_object(_required_file(root, "status.json"))
    if set(status) != _STATUS_FIELDS:
        missing = sorted(_STATUS_FIELDS - set(status))
        unknown = sorted(set(status) - _STATUS_FIELDS)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unknown:
            detail.append("unknown " + ", ".join(unknown))
        raise ValidationError(
            "status.json has an invalid supplementary capacity schema: " + "; ".join(detail)
        )
    expected = {
        "schema_version": "1.0",
        "classification": "supplementary-capacity",
        "status": "complete",
        "measurement_valid": True,
        "canonical_outputs_modified": False,
    }
    for field, expected_value in expected.items():
        if status.get(field) != expected_value:
            raise ValidationError(f"status.json {field} must be {expected_value!r}")
    reason = status.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValidationError("status.json reason must be a non-empty string")
    return status


def _validate_status_against_study(
    status: Mapping[str, Any],
    study: Mapping[str, Any],
) -> None:
    eligible_counts: dict[str, int] = {}
    for selection in study["selections"]:
        if not isinstance(selection, Mapping):
            raise ValidationError("capacity study selection must be an object")
        eligible_counts[str(selection["candidate_id"])] = int(selection["eligible_cell_count"])
    if status.get("eligible_cell_counts") != eligible_counts:
        raise ValidationError(
            "status.json eligible_cell_counts do not match the reassembled capacity study"
        )
    if status.get("selected_operating_points") != _selected_operating_points(study):
        raise ValidationError(
            "status.json selected_operating_points do not match the reassembled capacity study"
        )


def _selected_operating_points(study: Mapping[str, Any]) -> Mapping[str, Any]:
    selected_points: dict[str, Any] = {}
    selections = study.get("selections")
    if not isinstance(selections, Sequence) or isinstance(selections, (str, bytes)):
        raise ValidationError("capacity study selections must be an array")
    for selection in selections:
        if not isinstance(selection, Mapping):
            raise ValidationError("capacity study selection must be an object")
        candidate_id = str(selection["candidate_id"])
        selected = selection["selected_cell"]
        if selected is None:
            selected_points[candidate_id] = None
        elif isinstance(selected, Mapping):
            selected_points[candidate_id] = {
                "server_parallel": int(selected["server_parallel"]),
                "client_concurrency": int(selected["client_concurrency"]),
            }
        else:
            raise ValidationError("capacity study selected_cell must be an object or null")
    return selected_points


def _require_byte_identical(
    authoritative: Path,
    reproduced: Path,
    *,
    authoritative_name: str,
) -> Mapping[str, Any]:
    try:
        authoritative_bytes = authoritative.read_bytes()
        reproduced_bytes = reproduced.read_bytes()
    except OSError as exc:
        raise ValidationError(f"could not compare reproduced {authoritative_name}: {exc}") from exc
    authoritative_sha256 = sha256_file(authoritative)
    reproduced_sha256 = sha256_file(reproduced)
    if authoritative_bytes != reproduced_bytes:
        raise ValidationError(
            f"reproduced {authoritative_name} is not byte-identical to the archive "
            f"(archived {authoritative_sha256}, reproduced {reproduced_sha256})"
        )
    return {
        "present": True,
        "matches": True,
        "authoritative_path": authoritative_name,
        "authoritative_sha256": authoritative_sha256,
        "regenerated_sha256": reproduced_sha256,
    }


def _verify_canonical_replay(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    study: Mapping[str, Any],
    work_parent: Path,
) -> Mapping[str, Any]:
    release_path = _required_file(root, "canonical/release-archive.sha256")
    try:
        release_text = release_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("canonical release-archive.sha256 must be UTF-8") from exc
    except OSError as exc:
        raise ValidationError(f"could not read canonical release archive pin: {exc}") from exc
    if not release_text.endswith("\n") or len(release_text.splitlines()) != 1:
        raise ValidationError(
            "canonical release-archive.sha256 must contain exactly one newline-terminated entry"
        )
    match = _SHA256_LINE.fullmatch(release_text.rstrip("\n"))
    if match is None:
        raise ValidationError("canonical release-archive.sha256 is malformed")
    expected_archive_sha256, encoded_archive_path = match.groups()
    archive_relative = _safe_relative_path(
        encoded_archive_path,
        context="canonical release-archive.sha256",
    )
    if (
        len(archive_relative.parts) != 2
        or archive_relative.parts[0] != "canonical"
        or not archive_relative.name.endswith(".zip")
    ):
        raise ValidationError("canonical release archive must be one ZIP directly below canonical/")
    archive_path = _required_file(root, archive_relative.as_posix())
    if sha256_file(archive_path) != expected_archive_sha256:
        raise ValidationError("canonical release archive does not match its SHA-256 pin")

    manifest_canonical = _mapping(
        manifest.get("canonical_evidence"),
        "manifest canonical_evidence",
    )
    study_provenance = _mapping(study.get("provenance"), "capacity study provenance")
    study_canonical = _mapping(
        study_provenance.get("canonical_evidence"),
        "capacity study canonical_evidence",
    )
    for context, canonical in (
        ("manifest", manifest_canonical),
        ("capacity study", study_canonical),
    ):
        if canonical.get("release_sha256") != expected_archive_sha256:
            raise ValidationError(
                f"{context} canonical release SHA-256 does not match the embedded archive"
            )

    archived_replay = load_json_object(_required_file(root, "canonical/replay.json"))
    _validate_canonical_replay_payload(archived_replay, context="archived canonical replay")
    try:
        with tempfile.TemporaryDirectory(
            dir=work_parent,
            prefix=".paretopilot-canonical-replay-",
        ) as temporary_name:
            temporary = Path(temporary_name)
            extracted = temporary / "evidence"
            extracted.mkdir()
            _extract_zip_safely(archive_path, extracted)
            archived_frozen_checksums = _required_file(
                root,
                "canonical/frozen-SHA256SUMS",
            )
            extracted_checksums = _required_file(extracted, "SHA256SUMS")
            if archived_frozen_checksums.read_bytes() != extracted_checksums.read_bytes():
                raise ValidationError(
                    "embedded canonical SHA256SUMS does not match canonical/frozen-SHA256SUMS"
                )
            replayed = replay_evidence(extracted, temporary / "replayed")
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(f"could not replay embedded canonical evidence: {exc}") from exc

    _validate_canonical_replay_payload(replayed, context="reproduced canonical replay")
    for field in _CANONICAL_STABLE_FIELDS:
        if replayed.get(field) != archived_replay.get(field):
            raise ValidationError(
                f"reproduced canonical replay field {field!r} does not match the archive"
            )
    return {
        "verified": True,
        "archive_path": archive_relative.as_posix(),
        "archive_sha256": expected_archive_sha256,
        "replay_sha256": sha256_file(_required_file(root, "canonical/replay.json")),
        "replay_contract": replayed["replay_contract"],
        "selected_id": replayed["selected_id"],
        "fully_reproduced": True,
    }


def _validate_canonical_replay_payload(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> None:
    for field in _CANONICAL_TRUE_FIELDS:
        if payload.get(field) is not True:
            raise ValidationError(f"{context} field {field!r} must be true")
    if payload.get("replay_contract") != "1.1":
        raise ValidationError(f"{context} replay_contract must be '1.1'")
    if payload.get("differences") != [] or payload.get("warnings") != []:
        raise ValidationError(f"{context} must not contain differences or warnings")
    checksums = _mapping(payload.get("checksums"), f"{context} checksums")
    if checksums.get("verified") is not True:
        raise ValidationError(f"{context} checksums must be verified")
    comparisons = _mapping(
        payload.get("authoritative_comparisons"),
        f"{context} authoritative_comparisons",
    )
    if not comparisons:
        raise ValidationError(f"{context} authoritative comparisons must not be empty")
    for name, value in comparisons.items():
        comparison = _mapping(value, f"{context} comparison {name!r}")
        if comparison.get("present") is not True or comparison.get("matches") is not True:
            raise ValidationError(f"{context} authoritative comparison {name!r} must match")


def _extract_zip_safely(archive: Path, destination: Path) -> None:
    try:
        with zipfile.ZipFile(archive) as handle:
            members = handle.infolist()
            if not members or len(members) > _MAX_ARCHIVE_ENTRIES:
                raise ValidationError("canonical ZIP has an invalid entry count")
            total_size = sum(member.file_size for member in members)
            if total_size > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ValidationError("canonical ZIP exceeds the extraction size limit")
            seen: set[str] = set()
            destination_resolved = destination.resolve(strict=True)
            for index, member in enumerate(members, start=1):
                if member.flag_bits & 0x1:
                    raise ValidationError("canonical ZIP must not contain encrypted entries")
                relative, is_directory = _safe_zip_member(member, index=index)
                relative_text = relative.as_posix()
                folded = relative_text.casefold()
                if folded in seen:
                    raise ValidationError(
                        f"canonical ZIP contains duplicate entry {relative_text!r}"
                    )
                seen.add(folded)
                target = destination.joinpath(*relative.parts)
                target_resolved = target.resolve(strict=False)
                if not target_resolved.is_relative_to(destination_resolved):
                    raise ValidationError(
                        f"canonical ZIP entry escapes extraction root: {relative_text!r}"
                    )
                if is_directory:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    raise ValidationError(
                        f"canonical ZIP entry collides with an existing path: {relative_text!r}"
                    )
                with handle.open(member, "r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                if target.stat().st_size != member.file_size:
                    raise ValidationError(
                        f"canonical ZIP entry size changed during extraction: {relative_text!r}"
                    )
    except ValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValidationError(f"could not safely extract canonical ZIP: {exc}") from exc


def _safe_zip_member(
    member: zipfile.ZipInfo,
    *,
    index: int,
) -> tuple[PurePosixPath, bool]:
    encoded = member.filename
    is_directory = member.is_dir()
    normalized = encoded[:-1] if is_directory and encoded.endswith("/") else encoded
    relative = _safe_relative_path(
        normalized,
        context=f"canonical ZIP entry {index}",
    )
    file_type = stat.S_IFMT((member.external_attr >> 16) & 0xFFFF)
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ValidationError(f"canonical ZIP entry {index} is not a regular file or directory")
    if is_directory and file_type == stat.S_IFREG:
        raise ValidationError(f"canonical ZIP entry {index} has conflicting file metadata")
    if not is_directory and file_type == stat.S_IFDIR:
        raise ValidationError(f"canonical ZIP entry {index} has conflicting directory metadata")
    return relative, is_directory


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{context} must be an object")
    return value
