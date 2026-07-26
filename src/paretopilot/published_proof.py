"""Verify ParetoPilot's pinned public evidence releases without rerunning inference."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import hashlib
import http.client
from importlib import resources
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, BinaryIO, Mapping, Sequence
import unicodedata
import urllib.error
import urllib.request
from urllib.parse import urlparse
import zipfile

from paretopilot import __version__
from paretopilot.capacity_replay import replay_capacity_bundle
from paretopilot.domain import ValidationError
from paretopilot.io import write_json, write_text
from paretopilot.replay import replay_evidence


_LOCK_RESOURCE = "data/published-evidence-lock.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_TAG = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")
_RUN_ID = re.compile(r"^[1-9][0-9]{0,19}$")
_CANONICAL_COMPARISONS = (
    "benchmark-set",
    "benchmark-set-pass-1",
    "benchmark-set-pass-2",
    "load-evaluation",
    "policy-profiles",
    "recommendation",
    "repeat-stability",
    "report",
    "report-v1.1",
)
_CAPACITY_COMPARISONS = ("capacity-study", "capacity-receipt")
_EXPECTED_CAPACITY_SELECTIONS = {
    "q8-generic": {
        "server_parallel": 4,
        "client_concurrency": 4,
    },
    "q4-kleidiai-tuned": {
        "server_parallel": 4,
        "client_concurrency": 4,
    },
}
_MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 512
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 16 * 1024 * 1024
_MAX_ARCHIVE_MEMBER_BYTES = 4 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 64 * 1024
_DOWNLOAD_TIMEOUT_SECONDS = 30
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


@dataclass(frozen=True)
class ArchiveLock:
    """One immutable official release-asset pin."""

    filename: str
    release_tag: str
    run_id: str
    sha256: str
    size_bytes: int
    url: str


@dataclass(frozen=True)
class PublishedEvidenceLock:
    """The two public evidence pins and their interpretation limits."""

    canonical: ArchiveLock
    capacity: ArchiveLock
    evidence_limits: tuple[str, ...]


def verify_published_evidence(
    output_dir: Path,
    *,
    canonical_archive: Path | None = None,
    capacity_archive: Path | None = None,
) -> Mapping[str, Any]:
    """Download or read, verify, safely extract, and replay both official archives.

    Supplying both archive paths makes the command fully offline. Local archives
    are still required to match the packaged official byte size and SHA-256.
    The measured inference workload is never executed.
    """

    destination = _new_output_directory(output_dir)
    lock = _load_published_lock()
    workspace: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        workspace = Path(
            tempfile.mkdtemp(
                dir=destination.parent,
                prefix=f".{destination.name}.published-proof-",
            )
        )
        acquired = workspace / "archives"
        acquired.mkdir()
        canonical_path = _acquire_archive(
            lock.canonical,
            local_path=canonical_archive,
            download_dir=acquired,
            label="canonical",
        )
        capacity_path = _acquire_archive(
            lock.capacity,
            local_path=capacity_archive,
            download_dir=acquired,
            label="capacity",
        )

        canonical_evidence = workspace / "canonical-evidence"
        capacity_evidence = workspace / "capacity-evidence"
        canonical_evidence.mkdir()
        capacity_evidence.mkdir()
        _extract_zip_safely(canonical_path, canonical_evidence, label="canonical")
        _extract_zip_safely(capacity_path, capacity_evidence, label="capacity")

        canonical_replay = replay_evidence(
            canonical_evidence,
            workspace / "canonical-replay",
        )
        capacity_replay = replay_capacity_bundle(
            capacity_evidence,
            workspace / "capacity-replay",
        )
        proof = _build_proof(lock, canonical_replay, capacity_replay)

        staged_output = workspace / "published-output"
        staged_output.mkdir()
        write_json(staged_output / "published-proof.json", proof)
        write_text(staged_output / "published-proof.md", _render_markdown(proof))
        _publish_output(staged_output, destination)
        return proof
    except ValidationError:
        raise
    except (OSError, OverflowError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        raise ValidationError(f"published evidence verification failed: {exc}") from exc
    finally:
        if workspace is not None and workspace.exists():
            with suppress(OSError):
                shutil.rmtree(workspace)


def _new_output_directory(path: Path) -> Path:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise ValidationError(f"refusing to overwrite existing output directory: {target}")
    try:
        return target.resolve(strict=False)
    except OSError as exc:
        raise ValidationError(f"could not resolve output directory {target}: {exc}") from exc


def _publish_output(staged_output: Path, destination: Path) -> None:
    """Atomically publish the completed proof directory.

    All child files are written inside the private workspace first. Renaming
    the completed directory as one filesystem operation prevents a raced
    destination symlink from redirecting either proof file outside the
    requested parent.
    """

    if destination.exists() or destination.is_symlink():
        raise ValidationError(f"refusing to overwrite existing output directory: {destination}")
    try:
        staged_output.rename(destination)
    except OSError as exc:
        if destination.exists() or destination.is_symlink():
            raise ValidationError(
                f"refusing to overwrite existing output directory: {destination}"
            ) from exc
        raise ValidationError(
            f"could not atomically publish proof directory {destination}: {exc}"
        ) from exc


def _load_published_lock() -> PublishedEvidenceLock:
    try:
        raw = resources.files("paretopilot").joinpath(_LOCK_RESOURCE).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"could not read packaged published evidence lock: {exc}") from exc
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_mapping_without_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite value {value}")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValidationError(f"packaged published evidence lock is invalid JSON: {exc}") from exc
    root = _strict_mapping(
        payload,
        context="published evidence lock",
        fields={"schema_version", "archives", "evidence_limits"},
    )
    if root["schema_version"] != "1.0":
        raise ValidationError("published evidence lock schema_version must be '1.0'")
    archives = _strict_mapping(
        root["archives"],
        context="published evidence lock archives",
        fields={"canonical", "capacity"},
    )
    evidence_limits = root["evidence_limits"]
    if (
        not isinstance(evidence_limits, Sequence)
        or isinstance(evidence_limits, (str, bytes, bytearray))
        or not 1 <= len(evidence_limits) <= 16
        or any(not isinstance(item, str) or not item.strip() for item in evidence_limits)
    ):
        raise ValidationError("published evidence lock evidence_limits must be non-empty strings")
    return PublishedEvidenceLock(
        canonical=_archive_lock(archives["canonical"], label="canonical"),
        capacity=_archive_lock(archives["capacity"], label="capacity"),
        evidence_limits=tuple(evidence_limits),
    )


def _archive_lock(value: Any, *, label: str) -> ArchiveLock:
    payload = _strict_mapping(
        value,
        context=f"{label} archive lock",
        fields={"filename", "release_tag", "run_id", "sha256", "size_bytes", "url"},
    )
    filename = payload["filename"]
    release_tag = payload["release_tag"]
    run_id = payload["run_id"]
    digest = payload["sha256"]
    size_bytes = payload["size_bytes"]
    url = payload["url"]
    if (
        not isinstance(filename, str)
        or not filename
        or PurePosixPath(filename).name != filename
        or "\\" in filename
        or ":" in filename
        or not filename.endswith(".zip")
    ):
        raise ValidationError(f"{label} archive filename is unsafe")
    if not isinstance(release_tag, str) or _RELEASE_TAG.fullmatch(release_tag) is None:
        raise ValidationError(f"{label} archive release_tag is invalid")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValidationError(f"{label} archive run_id must be a positive decimal string")
    if not filename.removesuffix(".zip").endswith(f"-{run_id}"):
        raise ValidationError(f"{label} archive filename must end with its run_id")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValidationError(f"{label} archive sha256 is invalid")
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or not 1 <= size_bytes <= _MAX_DOWNLOAD_BYTES
    ):
        raise ValidationError(f"{label} archive size_bytes is outside the download limit")
    if not isinstance(url, str):
        raise ValidationError(f"{label} archive url must be a string")
    parsed = urlparse(url)
    expected_path = f"/agrovr/ParetoPilot/releases/download/{release_tag}/{filename}"
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != expected_path
    ):
        raise ValidationError(f"{label} archive url must be a direct HTTPS GitHub release URL")
    return ArchiveLock(
        filename=filename,
        release_tag=release_tag,
        run_id=run_id,
        sha256=digest,
        size_bytes=size_bytes,
        url=url,
    )


def _mapping_without_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _strict_mapping(value: Any, *, context: str, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{context} must be an object")
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValidationError(f"{context} has invalid fields: {'; '.join(details)}")
    return value


def _acquire_archive(
    lock: ArchiveLock,
    *,
    local_path: Path | None,
    download_dir: Path,
    label: str,
) -> Path:
    destination = download_dir / lock.filename
    if local_path is not None:
        _copy_local_archive(local_path, destination, lock, label=label)
        return destination
    _download_archive(lock, destination, label=label)
    return destination


def _copy_local_archive(
    path: Path,
    destination: Path,
    lock: ArchiveLock,
    *,
    label: str,
) -> None:
    """Copy and pin a local archive from one descriptor into private storage."""

    candidate = Path(path)
    descriptor: int | None = None
    created = False
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        if candidate.is_symlink():
            raise ValidationError(f"{label} archive must not be a symbolic link")
        descriptor = os.open(candidate, flags)
        mode = os.fstat(descriptor).st_mode
        if not stat.S_ISREG(mode):
            raise ValidationError(f"{label} archive is not a regular file: {candidate}")

        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as source:
            descriptor = None
            actual_size = os.fstat(source.fileno()).st_size
            if actual_size != lock.size_bytes:
                raise ValidationError(
                    f"{label} archive size mismatch: "
                    f"expected {lock.size_bytes}, found {actual_size}"
                )
            with destination.open("xb") as output:
                created = True
                copied_size = _copy_download(
                    source,
                    output,
                    digest=digest,
                    size_limit=lock.size_bytes,
                    label=label,
                )

        if copied_size != lock.size_bytes:
            raise ValidationError(
                f"{label} archive size mismatch: expected {lock.size_bytes}, found {copied_size}"
            )
        actual_digest = digest.hexdigest()
        if actual_digest != lock.sha256:
            raise ValidationError(
                f"{label} archive SHA-256 mismatch: expected {lock.sha256}, found {actual_digest}"
            )
    except ValidationError:
        if created:
            with suppress(OSError):
                destination.unlink()
        raise
    except OSError as exc:
        if created:
            with suppress(OSError):
                destination.unlink()
        raise ValidationError(f"could not copy {label} archive {candidate}: {exc}") from exc
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)


def _download_archive(lock: ArchiveLock, destination: Path, *, label: str) -> None:
    request = urllib.request.Request(
        lock.url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": f"ParetoPilot/{__version__} published-evidence-verifier",
        },
    )
    digest = hashlib.sha256()
    downloaded = 0
    created = False
    try:
        with urllib.request.urlopen(  # noqa: S310 - URL is a validated packaged HTTPS pin.
            request,
            timeout=_DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:
            status = response.getcode()
            if status not in {None, 200}:
                raise ValidationError(f"{label} archive download returned HTTP {status}")
            encoding = response.headers.get("Content-Encoding")
            if encoding not in {None, "", "identity"}:
                raise ValidationError(f"{label} archive download used unexpected content encoding")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError as exc:
                    raise ValidationError(
                        f"{label} archive download has an invalid Content-Length"
                    ) from exc
                if declared_size != lock.size_bytes:
                    raise ValidationError(
                        f"{label} archive download size mismatch: "
                        f"expected {lock.size_bytes}, server declared {declared_size}"
                    )
            with destination.open("xb") as output:
                created = True
                downloaded = _copy_download(
                    response,
                    output,
                    digest=digest,
                    size_limit=lock.size_bytes,
                    label=label,
                )
    except ValidationError:
        if created:
            with suppress(OSError):
                destination.unlink()
        raise
    except (
        OSError,
        http.client.HTTPException,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as exc:
        if created:
            with suppress(OSError):
                destination.unlink()
        raise ValidationError(f"could not download {label} archive: {exc}") from exc
    if downloaded != lock.size_bytes:
        with suppress(OSError):
            destination.unlink()
        raise ValidationError(
            f"{label} archive download size mismatch: "
            f"expected {lock.size_bytes}, found {downloaded}"
        )
    actual_digest = digest.hexdigest()
    if actual_digest != lock.sha256:
        with suppress(OSError):
            destination.unlink()
        raise ValidationError(
            f"{label} archive download SHA-256 mismatch: "
            f"expected {lock.sha256}, found {actual_digest}"
        )


def _copy_download(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    digest: Any,
    size_limit: int,
    label: str,
) -> int:
    total = 0
    while True:
        remaining = size_limit - total
        chunk = source.read(min(_DOWNLOAD_CHUNK_BYTES, remaining + 1))
        if not chunk:
            return total
        total += len(chunk)
        if total > size_limit:
            raise ValidationError(f"{label} archive download exceeds its pinned size")
        destination.write(chunk)
        digest.update(chunk)


def _extract_zip_safely(archive: Path, destination: Path, *, label: str) -> None:
    try:
        if not destination.is_dir() or destination.is_symlink():
            raise ValidationError(f"{label} extraction destination must be a new directory")
        if any(destination.iterdir()):
            raise ValidationError(f"{label} extraction destination must be empty")
        with zipfile.ZipFile(archive) as handle:
            members = handle.infolist()
            if not members:
                raise ValidationError(f"{label} archive is empty")
            if len(members) > _MAX_ARCHIVE_MEMBERS:
                raise ValidationError(
                    f"{label} archive has too many members: {len(members)} > {_MAX_ARCHIVE_MEMBERS}"
                )
            total_declared = sum(member.file_size for member in members)
            if total_declared > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ValidationError(f"{label} archive exceeds the uncompressed size limit")

            explicit_entries: set[str] = set()
            node_types: dict[str, str] = {}
            canonical_names: dict[str, str] = {}
            destination_resolved = destination.resolve(strict=True)
            total_written = 0
            for index, member in enumerate(members, start=1):
                relative, is_directory = _safe_zip_member(member, label=label, index=index)
                relative_text = relative.as_posix()
                path_key = _path_key(relative_text)
                if path_key in explicit_entries:
                    raise ValidationError(
                        f"{label} archive contains a duplicate or case-colliding "
                        f"member: {relative_text!r}"
                    )
                explicit_entries.add(path_key)
                _register_archive_path(
                    relative,
                    is_directory=is_directory,
                    node_types=node_types,
                    canonical_names=canonical_names,
                    label=label,
                )
                target = destination.joinpath(*relative.parts)
                if not target.resolve(strict=False).is_relative_to(destination_resolved):
                    raise ValidationError(
                        f"{label} archive member escapes extraction root: {relative_text!r}"
                    )
                if is_directory:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    raise ValidationError(
                        f"{label} archive member collides with an existing path: {relative_text!r}"
                    )
                with handle.open(member, "r") as source, target.open("xb") as output:
                    written = _copy_archive_member(
                        source,
                        output,
                        expected_size=member.file_size,
                        label=label,
                        relative_text=relative_text,
                    )
                total_written += written
                if total_written > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ValidationError(f"{label} archive exceeds the extraction size limit")
    except ValidationError:
        raise
    except (
        OSError,
        RuntimeError,
        NotImplementedError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        raise ValidationError(f"could not safely extract {label} archive: {exc}") from exc


def _safe_zip_member(
    member: zipfile.ZipInfo,
    *,
    label: str,
    index: int,
) -> tuple[PurePosixPath, bool]:
    if member.flag_bits & 0x1:
        raise ValidationError(f"{label} archive member {index} is encrypted")
    if member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        raise ValidationError(f"{label} archive member {index} uses unsupported compression")
    if member.file_size > _MAX_ARCHIVE_MEMBER_BYTES:
        raise ValidationError(f"{label} archive member {index} exceeds the member size limit")

    encoded = member.filename
    is_directory = member.is_dir()
    normalized = encoded[:-1] if is_directory and encoded.endswith("/") else encoded
    if (
        "\\" in normalized
        or ":" in normalized
        or any(ord(character) < 32 or character in '<>"|?*' for character in normalized)
    ):
        raise ValidationError(f"{label} archive member {index} has an unsafe path")
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("./")
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != normalized
    ):
        raise ValidationError(f"{label} archive member {index} has an unsafe path")
    for part in relative.parts:
        if part != part.rstrip(" ."):
            raise ValidationError(f"{label} archive member {index} has an unsafe path")
        stem = part.split(".", maxsplit=1)[0].casefold()
        if stem in _WINDOWS_RESERVED_NAMES:
            raise ValidationError(f"{label} archive member {index} has an unsafe path")

    file_type = stat.S_IFMT((member.external_attr >> 16) & 0xFFFF)
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ValidationError(f"{label} archive member {index} is a symbolic link or special entry")
    if is_directory and file_type == stat.S_IFREG:
        raise ValidationError(f"{label} archive member {index} has conflicting file metadata")
    if not is_directory and file_type == stat.S_IFDIR:
        raise ValidationError(f"{label} archive member {index} has conflicting directory metadata")
    return relative, is_directory


def _register_archive_path(
    path: PurePosixPath,
    *,
    is_directory: bool,
    node_types: dict[str, str],
    canonical_names: dict[str, str],
    label: str,
) -> None:
    prefixes: list[str] = []
    for part in path.parts:
        prefixes.append(part)
        display = "/".join(prefixes)
        key = _path_key(display)
        previous_name = canonical_names.get(key)
        if previous_name is not None and previous_name != display:
            raise ValidationError(
                f"{label} archive contains a duplicate or case-colliding path: {display!r}"
            )
        canonical_names.setdefault(key, display)

    for index in range(1, len(path.parts)):
        prefix = "/".join(path.parts[:index])
        key = _path_key(prefix)
        if node_types.get(key) == "file":
            raise ValidationError(
                f"{label} archive path is nested below a file: {path.as_posix()!r}"
            )
        node_types.setdefault(key, "directory")

    final_key = _path_key(path.as_posix())
    existing = node_types.get(final_key)
    if is_directory:
        if existing == "file":
            raise ValidationError(
                f"{label} archive member collides with a file: {path.as_posix()!r}"
            )
        node_types[final_key] = "directory"
    else:
        if existing == "directory":
            raise ValidationError(
                f"{label} archive member collides with a directory: {path.as_posix()!r}"
            )
        node_types[final_key] = "file"


def _path_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _copy_archive_member(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    expected_size: int,
    label: str,
    relative_text: str,
) -> int:
    total = 0
    while True:
        remaining = expected_size - total
        chunk = source.read(min(_DOWNLOAD_CHUNK_BYTES, remaining + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > expected_size:
            raise ValidationError(
                f"{label} archive member expanded beyond its declared size: {relative_text!r}"
            )
        destination.write(chunk)
    if total != expected_size:
        raise ValidationError(
            f"{label} archive member size mismatch during extraction: {relative_text!r}"
        )
    return total


def _build_proof(
    lock: PublishedEvidenceLock,
    canonical: Mapping[str, Any],
    capacity: Mapping[str, Any],
) -> Mapping[str, Any]:
    canonical_comparisons = _validated_canonical_replay(canonical)
    capacity_comparisons = _validated_capacity_replay(capacity, canonical_lock=lock.canonical)
    proof: Mapping[str, Any] = {
        "schema_version": "1.0",
        "classification": "published-evidence-proof",
        "valid": True,
        "measurements_rerun": False,
        "archives": {
            "canonical": _archive_proof(lock.canonical),
            "capacity": _archive_proof(lock.capacity),
        },
        "canonical": {
            "replay_contract": "1.1",
            "valid": True,
            "selected_id": "q8-generic",
            "status_complete": True,
            "checksum_entries": int(
                _mapping(canonical["checksums"], "canonical checksums")["entry_count"]
            ),
            "authoritative_comparison_count": len(canonical_comparisons),
            "authoritative_comparisons": canonical_comparisons,
            "differences": [],
            "warnings": [],
        },
        "capacity": {
            "valid": True,
            "status_complete": True,
            "selected_operating_points": {
                candidate_id: {
                    **selection,
                    "operating_point": (
                        f"P{selection['server_parallel']}/C{selection['client_concurrency']}"
                    ),
                }
                for candidate_id, selection in _EXPECTED_CAPACITY_SELECTIONS.items()
            },
            "artifact_matches": capacity_comparisons,
            "embedded_canonical_replay": {
                "verified": True,
                "replay_contract": "1.1",
                "selected_id": "q8-generic",
                "fully_reproduced": True,
                "archive_sha256": lock.canonical.sha256,
            },
        },
        "evidence_limits": list(lock.evidence_limits),
        "verdict": (
            "PASS: pinned canonical v1.1 and capacity v1.4 archives verified and replayed."
        ),
    }
    return proof


def _archive_proof(lock: ArchiveLock) -> Mapping[str, Any]:
    return {
        "release_tag": lock.release_tag,
        "run_id": lock.run_id,
        "actions_url": f"https://github.com/agrovr/ParetoPilot/actions/runs/{lock.run_id}",
        "filename": lock.filename,
        "url": lock.url,
        "size_bytes": lock.size_bytes,
        "sha256": lock.sha256,
        "verified": True,
    }


def _validated_canonical_replay(
    replay: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    required_true = (
        "valid",
        "status_complete",
        "decision_reproduced",
        "fully_reproduced",
        "report_matches_archive",
        "authoritative_outputs_match",
    )
    for field in required_true:
        if replay.get(field) is not True:
            raise ValidationError(f"canonical replay field {field!r} must be true")
    if replay.get("replay_contract") != "1.1":
        raise ValidationError("canonical replay_contract must be '1.1'")
    if replay.get("selected_id") != "q8-generic":
        raise ValidationError("canonical replay must select 'q8-generic'")
    if replay.get("differences") != [] or replay.get("warnings") != []:
        raise ValidationError("canonical replay must have zero differences and zero warnings")
    checksums = _mapping(replay.get("checksums"), "canonical replay checksums")
    if checksums.get("verified") is not True:
        raise ValidationError("canonical replay checksums must be verified")
    if not isinstance(checksums.get("entry_count"), int) or checksums["entry_count"] <= 0:
        raise ValidationError("canonical replay checksum entry_count must be positive")

    raw_comparisons = _mapping(
        replay.get("authoritative_comparisons"),
        "canonical authoritative comparisons",
    )
    if set(raw_comparisons) != set(_CANONICAL_COMPARISONS):
        raise ValidationError(
            "canonical replay must contain the nine pinned authoritative comparisons"
        )
    comparisons: dict[str, Mapping[str, Any]] = {}
    for name in _CANONICAL_COMPARISONS:
        comparisons[name] = _matching_comparison(
            raw_comparisons[name],
            context=f"canonical comparison {name!r}",
        )
    return comparisons


def _validated_capacity_replay(
    replay: Mapping[str, Any],
    *,
    canonical_lock: ArchiveLock,
) -> Mapping[str, Mapping[str, Any]]:
    for field in (
        "valid",
        "status_complete",
        "capacity_study_reproduced",
        "capacity_receipt_reproduced",
    ):
        if replay.get(field) is not True:
            raise ValidationError(f"capacity replay field {field!r} must be true")
    selected = _mapping(
        replay.get("selected_operating_points"),
        "capacity selected operating points",
    )
    if selected != _EXPECTED_CAPACITY_SELECTIONS:
        raise ValidationError("capacity replay must select P4/C4 for both published candidates")

    raw_comparisons = _mapping(
        replay.get("authoritative_comparisons"),
        "capacity authoritative comparisons",
    )
    if set(raw_comparisons) != set(_CAPACITY_COMPARISONS):
        raise ValidationError("capacity replay has an unexpected authoritative comparison set")
    comparisons = {
        name: _matching_comparison(
            raw_comparisons[name],
            context=f"capacity comparison {name!r}",
        )
        for name in _CAPACITY_COMPARISONS
    }

    embedded = _mapping(replay.get("canonical_replay"), "embedded canonical replay")
    for field in ("verified", "fully_reproduced"):
        if embedded.get(field) is not True:
            raise ValidationError(f"embedded canonical replay field {field!r} must be true")
    if embedded.get("replay_contract") != "1.1":
        raise ValidationError("embedded canonical replay_contract must be '1.1'")
    if embedded.get("selected_id") != "q8-generic":
        raise ValidationError("embedded canonical replay must select 'q8-generic'")
    if embedded.get("archive_sha256") != canonical_lock.sha256:
        raise ValidationError("embedded canonical replay archive SHA-256 does not match the pin")
    return comparisons


def _matching_comparison(value: Any, *, context: str) -> Mapping[str, Any]:
    comparison = _mapping(value, context)
    if comparison.get("present") is not True or comparison.get("matches") is not True:
        raise ValidationError(f"{context} must be present and byte-identical")
    authoritative_path = comparison.get("authoritative_path")
    if not isinstance(authoritative_path, str) or not authoritative_path:
        raise ValidationError(f"{context} authoritative_path must be non-empty")
    authoritative_sha256 = comparison.get("authoritative_sha256")
    regenerated_sha256 = comparison.get("regenerated_sha256")
    for label, digest in (
        ("authoritative_sha256", authoritative_sha256),
        ("regenerated_sha256", regenerated_sha256),
    ):
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValidationError(f"{context} {label} must be a SHA-256 digest")
    if authoritative_sha256 != regenerated_sha256:
        raise ValidationError(f"{context} digests must match")
    return {
        "present": True,
        "matches": True,
        "authoritative_path": authoritative_path,
        "authoritative_sha256": authoritative_sha256,
        "regenerated_sha256": regenerated_sha256,
    }


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{context} must be an object")
    return value


def _render_markdown(proof: Mapping[str, Any]) -> str:
    archives = _mapping(proof["archives"], "proof archives")
    canonical_archive = _mapping(archives["canonical"], "canonical archive proof")
    capacity_archive = _mapping(archives["capacity"], "capacity archive proof")
    canonical = _mapping(proof["canonical"], "canonical proof")
    capacity = _mapping(proof["capacity"], "capacity proof")
    comparisons = _mapping(
        canonical["authoritative_comparisons"],
        "canonical proof comparisons",
    )
    capacity_matches = _mapping(capacity["artifact_matches"], "capacity artifact matches")
    embedded = _mapping(capacity["embedded_canonical_replay"], "embedded canonical proof")
    selections = _mapping(capacity["selected_operating_points"], "capacity selections")

    lines = [
        "# ParetoPilot published evidence proof",
        "",
        f"**{proof['verdict']}**",
        "",
        "This check used the exact public release archives below. It verified their pinned bytes, "
        "safely extracted them, and rebuilt the published decisions from archived measurements. "
        "It did not run a new inference benchmark.",
        "",
        "## Verified release archives",
        "",
        "| Evidence | Release | Actions run | Official archive | Exact size | SHA-256 |",
        "| --- | --- | ---: | --- | ---: | --- |",
        (
            f"| Canonical | `{canonical_archive['release_tag']}` | "
            f"[`{canonical_archive['run_id']}`]({canonical_archive['actions_url']}) | "
            f"[{canonical_archive['filename']}]({canonical_archive['url']}) | "
            f"{int(canonical_archive['size_bytes']):,} bytes | "
            f"`{canonical_archive['sha256']}` |"
        ),
        (
            f"| Capacity | `{capacity_archive['release_tag']}` | "
            f"[`{capacity_archive['run_id']}`]({capacity_archive['actions_url']}) | "
            f"[{capacity_archive['filename']}]({capacity_archive['url']}) | "
            f"{int(capacity_archive['size_bytes']):,} bytes | "
            f"`{capacity_archive['sha256']}` |"
        ),
        "",
        "## Canonical v1.1 result",
        "",
        f"- Selected candidate: `{canonical['selected_id']}`.",
        f"- Replay contract: `{canonical['replay_contract']}`.",
        f"- Checksummed files: {int(canonical['checksum_entries'])}.",
        "- Differences: 0.",
        "- Warnings: 0.",
        "",
        "All nine authoritative outputs matched byte for byte:",
        "",
        "| Output | Archived SHA-256 | Fresh replay SHA-256 | Result |",
        "| --- | --- | --- | --- |",
    ]
    for name in _CANONICAL_COMPARISONS:
        comparison = _mapping(comparisons[name], f"proof comparison {name!r}")
        lines.append(
            f"| `{name}` | `{comparison['authoritative_sha256']}` | "
            f"`{comparison['regenerated_sha256']}` | PASS |"
        )

    lines.extend(
        [
            "",
            "## Supplementary v1.4 capacity result",
            "",
        ]
    )
    for candidate_id in ("q8-generic", "q4-kleidiai-tuned"):
        selection = _mapping(selections[candidate_id], f"capacity selection {candidate_id!r}")
        lines.append(
            f"- `{candidate_id}` selected **{selection['operating_point']}** "
            f"(server parallelism 4, client concurrency 4)."
        )
    for name in _CAPACITY_COMPARISONS:
        match = _mapping(capacity_matches[name], f"capacity match {name!r}")
        lines.append(f"- `{name}`: byte-for-byte PASS (`{match['authoritative_sha256']}`).")
    lines.extend(
        [
            (
                "- Embedded canonical replay: PASS "
                f"(`{embedded['replay_contract']}`, selected `{embedded['selected_id']}`, "
                "fully reproduced)."
            ),
            "",
            "## Evidence limits",
            "",
        ]
    )
    for limit in proof["evidence_limits"]:
        lines.append(f"- {limit}")
    lines.extend(["", "No archived measurement was changed by this verification.", ""])
    return "\n".join(lines)
