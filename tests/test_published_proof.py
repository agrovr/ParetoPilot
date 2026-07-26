from __future__ import annotations

import copy
import hashlib
import http.client
import io
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import warnings
import zipfile

from paretopilot import cli
from paretopilot.domain import ValidationError
from paretopilot.published_proof import (
    ArchiveLock,
    PublishedEvidenceLock,
    _CANONICAL_COMPARISONS,
    _archive_lock as _parse_archive_lock,
    _build_proof,
    _copy_local_archive,
    _download_archive,
    _extract_zip_safely,
    _load_published_lock,
    _publish_output,
    _safe_zip_member,
    verify_published_evidence,
)


CANONICAL_SHA256 = "b5586878ccd214667911390f417db0417111ac2c31d163a2f5f55c4469aefeb2"
CAPACITY_SHA256 = "a73d801bc3997f1c0b0158e92c8305987da8638501b74c5ecd2af3aaca57aaa7"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_zip(path: Path, entries: list[tuple[str | zipfile.ZipInfo, bytes]]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w") as archive:
            for name, payload in entries:
                archive.writestr(name, payload)


def _archive_lock(path: Path, *, release_tag: str, label: str) -> ArchiveLock:
    filename = path.name
    return ArchiveLock(
        filename=filename,
        release_tag=release_tag,
        run_id=("30055662526" if label == "canonical" else "30144901854"),
        sha256=_digest(path),
        size_bytes=path.stat().st_size,
        url=f"https://github.com/agrovr/ParetoPilot/releases/download/{release_tag}/{filename}",
    )


def _comparison(name: str) -> dict[str, object]:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return {
        "present": True,
        "matches": True,
        "authoritative_path": f"published/{name}",
        "authoritative_sha256": digest,
        "regenerated_sha256": digest,
    }


def _canonical_replay() -> dict[str, object]:
    return {
        "valid": True,
        "status_complete": True,
        "decision_reproduced": True,
        "fully_reproduced": True,
        "report_matches_archive": True,
        "authoritative_outputs_match": True,
        "replay_contract": "1.1",
        "selected_id": "q8-generic",
        "differences": [],
        "warnings": [],
        "checksums": {
            "verified": True,
            "entry_count": 150,
        },
        "authoritative_comparisons": {name: _comparison(name) for name in _CANONICAL_COMPARISONS},
    }


def _capacity_replay(canonical_sha256: str) -> dict[str, object]:
    return {
        "valid": True,
        "status_complete": True,
        "capacity_study_reproduced": True,
        "capacity_receipt_reproduced": True,
        "selected_operating_points": {
            "q8-generic": {
                "server_parallel": 4,
                "client_concurrency": 4,
            },
            "q4-kleidiai-tuned": {
                "server_parallel": 4,
                "client_concurrency": 4,
            },
        },
        "authoritative_comparisons": {
            name: _comparison(name) for name in ("capacity-study", "capacity-receipt")
        },
        "canonical_replay": {
            "verified": True,
            "fully_reproduced": True,
            "replay_contract": "1.1",
            "selected_id": "q8-generic",
            "archive_sha256": canonical_sha256,
        },
    }


class _DownloadResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, content_length: str | None = None) -> None:
        super().__init__(payload)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def getcode(self) -> int:
        return 200


class _InterruptedDownloadResponse(_DownloadResponse):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self._read_count = 0

    def read(self, size: int = -1) -> bytes:
        self._read_count += 1
        if self._read_count == 1:
            return super().read(min(size, 4))
        raise http.client.IncompleteRead(b"", 1)


class PublishedProofTests(unittest.TestCase):
    def test_packaged_lock_has_the_exact_official_release_pins(self) -> None:
        lock = _load_published_lock()

        self.assertEqual(lock.canonical.release_tag, "v1.1.0")
        self.assertEqual(lock.canonical.run_id, "30055662526")
        self.assertEqual(lock.canonical.size_bytes, 402899)
        self.assertEqual(lock.canonical.sha256, CANONICAL_SHA256)
        self.assertEqual(lock.capacity.release_tag, "v1.4.0")
        self.assertEqual(lock.capacity.run_id, "30144901854")
        self.assertEqual(lock.capacity.size_bytes, 794681)
        self.assertEqual(lock.capacity.sha256, CAPACITY_SHA256)
        self.assertEqual(len(lock.evidence_limits), 4)
        self.assertTrue(all(limit.endswith(".") for limit in lock.evidence_limits))

    def test_offline_verification_is_deterministic_and_publishes_only_two_proofs(self) -> None:
        with TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            canonical_archive = root / "canonical.zip"
            capacity_archive = root / "capacity.zip"
            _write_zip(canonical_archive, [("canonical.txt", b"canonical\n")])
            _write_zip(capacity_archive, [("capacity.txt", b"capacity\n")])
            lock = PublishedEvidenceLock(
                canonical=_archive_lock(
                    canonical_archive,
                    release_tag="v1.1.0",
                    label="canonical",
                ),
                capacity=_archive_lock(
                    capacity_archive,
                    release_tag="v1.4.0",
                    label="capacity",
                ),
                evidence_limits=("This is a test limit.",),
            )
            capacity_replay = _capacity_replay(lock.canonical.sha256)
            outputs = (root / "proof-one", root / "proof-two")
            snapshots: list[dict[str, bytes]] = []

            for output in outputs:
                extracted: dict[str, bytes] = {}

                def replay_canonical(evidence: Path, _output: Path) -> dict[str, object]:
                    extracted["canonical"] = (evidence / "canonical.txt").read_bytes()
                    return _canonical_replay()

                def replay_capacity(evidence: Path, _output: Path) -> dict[str, object]:
                    extracted["capacity"] = (evidence / "capacity.txt").read_bytes()
                    return capacity_replay

                with (
                    patch(
                        "paretopilot.published_proof._load_published_lock",
                        return_value=lock,
                    ),
                    patch(
                        "paretopilot.published_proof.replay_evidence",
                        side_effect=replay_canonical,
                    ),
                    patch(
                        "paretopilot.published_proof.replay_capacity_bundle",
                        side_effect=replay_capacity,
                    ),
                ):
                    result = verify_published_evidence(
                        output,
                        canonical_archive=canonical_archive,
                        capacity_archive=capacity_archive,
                    )

                self.assertTrue(result["valid"])
                self.assertFalse(result["measurements_rerun"])
                self.assertEqual(result["canonical"]["selected_id"], "q8-generic")
                self.assertEqual(result["canonical"]["authoritative_comparison_count"], 9)
                self.assertEqual(result["canonical"]["differences"], [])
                self.assertEqual(result["canonical"]["warnings"], [])
                self.assertEqual(
                    result["archives"]["canonical"]["run_id"],
                    "30055662526",
                )
                self.assertEqual(
                    result["archives"]["capacity"]["run_id"],
                    "30144901854",
                )
                self.assertEqual(
                    result["capacity"]["selected_operating_points"]["q8-generic"][
                        "operating_point"
                    ],
                    "P4/C4",
                )
                self.assertEqual(
                    {path.name for path in output.iterdir()},
                    {"published-proof.json", "published-proof.md"},
                )
                self.assertEqual(extracted["canonical"], b"canonical\n")
                self.assertEqual(extracted["capacity"], b"capacity\n")
                markdown = (output / "published-proof.md").read_text(encoding="utf-8")
                self.assertIn("All nine archived outputs matched byte for byte", markdown)
                self.assertIn("`30055662526`", markdown)
                self.assertIn("`30144901854`", markdown)
                self.assertIn(
                    "[`30055662526`](https://github.com/agrovr/ParetoPilot/actions/runs/"
                    "30055662526)",
                    markdown,
                )
                self.assertIn(
                    "[`30144901854`](https://github.com/agrovr/ParetoPilot/actions/runs/"
                    "30144901854)",
                    markdown,
                )
                self.assertIn("**P4/C4**", markdown)
                self.assertIn("It did not run a new inference benchmark", markdown)
                snapshots.append(
                    {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
                )

            self.assertEqual(snapshots[0], snapshots[1])

    def test_existing_output_is_rejected_before_acquisition_or_replay(self) -> None:
        with TemporaryDirectory() as temporary_name:
            output = Path(temporary_name) / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep\n", encoding="utf-8")
            with (
                patch("paretopilot.published_proof._load_published_lock") as load_lock,
                patch("paretopilot.published_proof.replay_evidence") as replay,
                self.assertRaisesRegex(ValidationError, "refusing to overwrite"),
            ):
                verify_published_evidence(output)

            load_lock.assert_not_called()
            replay.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")

    def test_archive_lock_rejects_non_decimal_or_filename_mismatched_run_ids(self) -> None:
        base = {
            "filename": "paretopilot-v1.1.0-evidence-30055662526.zip",
            "release_tag": "v1.1.0",
            "run_id": "30055662526",
            "sha256": "0" * 64,
            "size_bytes": 100,
            "url": (
                "https://github.com/agrovr/ParetoPilot/releases/download/v1.1.0/"
                "paretopilot-v1.1.0-evidence-30055662526.zip"
            ),
        }
        for run_id, message in (
            ("030055662526", "positive decimal"),
            ("run-30055662526", "positive decimal"),
            ("30144901854", "filename must end"),
        ):
            with (
                self.subTest(run_id=run_id),
                self.assertRaisesRegex(
                    ValidationError,
                    message,
                ),
            ):
                _parse_archive_lock({**base, "run_id": run_id}, label="canonical")

    def test_local_archive_must_match_both_pinned_size_and_sha256(self) -> None:
        with TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            canonical_archive = root / "canonical.zip"
            capacity_archive = root / "capacity.zip"
            _write_zip(canonical_archive, [("canonical.txt", b"canonical\n")])
            _write_zip(capacity_archive, [("capacity.txt", b"capacity\n")])
            canonical_lock = _archive_lock(
                canonical_archive,
                release_tag="v1.1.0",
                label="canonical",
            )
            capacity_lock = _archive_lock(
                capacity_archive,
                release_tag="v1.4.0",
                label="capacity",
            )

            wrong_size = PublishedEvidenceLock(
                canonical=ArchiveLock(
                    **{
                        **canonical_lock.__dict__,
                        "size_bytes": canonical_lock.size_bytes + 1,
                    }
                ),
                capacity=capacity_lock,
                evidence_limits=("Limit.",),
            )
            with (
                patch(
                    "paretopilot.published_proof._load_published_lock",
                    return_value=wrong_size,
                ),
                self.assertRaisesRegex(ValidationError, "size mismatch"),
            ):
                verify_published_evidence(
                    root / "size-output",
                    canonical_archive=canonical_archive,
                    capacity_archive=capacity_archive,
                )

            wrong_hash = PublishedEvidenceLock(
                canonical=ArchiveLock(
                    **{
                        **canonical_lock.__dict__,
                        "sha256": "0" * 64,
                    }
                ),
                capacity=capacity_lock,
                evidence_limits=("Limit.",),
            )
            with (
                patch(
                    "paretopilot.published_proof._load_published_lock",
                    return_value=wrong_hash,
                ),
                self.assertRaisesRegex(ValidationError, "SHA-256 mismatch"),
            ):
                verify_published_evidence(
                    root / "hash-output",
                    canonical_archive=canonical_archive,
                    capacity_archive=capacity_archive,
                )

    def test_local_archive_is_private_copied_before_extraction(self) -> None:
        with TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            canonical_archive = root / "canonical.zip"
            capacity_archive = root / "capacity.zip"
            _write_zip(canonical_archive, [("canonical.txt", b"canonical\n")])
            _write_zip(capacity_archive, [("capacity.txt", b"capacity\n")])
            lock = PublishedEvidenceLock(
                canonical=_archive_lock(
                    canonical_archive,
                    release_tag="v1.1.0",
                    label="canonical",
                ),
                capacity=_archive_lock(
                    capacity_archive,
                    release_tag="v1.4.0",
                    label="capacity",
                ),
                evidence_limits=("Limit.",),
            )
            extracted_paths: list[Path] = []

            def extract(archive: Path, destination: Path, *, label: str) -> None:
                extracted_paths.append(archive)
                if label == "canonical":
                    canonical_archive.write_bytes(b"replaced after private acquisition")
                _extract_zip_safely(archive, destination, label=label)

            with (
                patch(
                    "paretopilot.published_proof._load_published_lock",
                    return_value=lock,
                ),
                patch(
                    "paretopilot.published_proof._extract_zip_safely",
                    side_effect=extract,
                ),
                patch(
                    "paretopilot.published_proof.replay_evidence",
                    return_value=_canonical_replay(),
                ),
                patch(
                    "paretopilot.published_proof.replay_capacity_bundle",
                    return_value=_capacity_replay(lock.canonical.sha256),
                ),
            ):
                result = verify_published_evidence(
                    root / "proof",
                    canonical_archive=canonical_archive,
                    capacity_archive=capacity_archive,
                )

            self.assertTrue(result["valid"])
            self.assertEqual(len(extracted_paths), 2)
            self.assertNotIn(canonical_archive, extracted_paths)
            self.assertNotIn(capacity_archive, extracted_paths)

    def test_local_archive_open_is_nonblocking_when_the_platform_supports_it(self) -> None:
        nonblocking = getattr(os, "O_NONBLOCK", 0)
        if not nonblocking:
            self.skipTest("O_NONBLOCK is unavailable on this platform")
        lock = ArchiveLock(
            filename="release.zip",
            release_tag="v1.1.0",
            run_id="30055662526",
            sha256="0" * 64,
            size_bytes=1,
            url="https://github.com/agrovr/ParetoPilot/releases/download/v1.1.0/release.zip",
        )
        with (
            TemporaryDirectory() as temporary_name,
            patch(
                "paretopilot.published_proof.os.open",
                side_effect=OSError("stop after flags are captured"),
            ) as opened,
            self.assertRaisesRegex(ValidationError, "could not copy"),
        ):
            root = Path(temporary_name)
            _copy_local_archive(
                root / "source.zip",
                root / "private.zip",
                lock,
                label="canonical",
            )

        flags = opened.call_args.args[1]
        self.assertEqual(flags & nonblocking, nonblocking)

    def test_download_is_streamed_and_bounded_by_the_exact_pin(self) -> None:
        payload = b"bounded release bytes"
        lock = ArchiveLock(
            filename="release.zip",
            release_tag="v1.1.0",
            run_id="30055662526",
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            url="https://github.com/agrovr/ParetoPilot/releases/download/v1.1.0/release.zip",
        )
        with TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            destination = root / lock.filename
            with patch(
                "paretopilot.published_proof.urllib.request.urlopen",
                return_value=_DownloadResponse(payload, content_length=str(len(payload))),
            ) as urlopen:
                _download_archive(lock, destination, label="canonical")

            self.assertEqual(destination.read_bytes(), payload)
            request = urlopen.call_args.args[0]
            self.assertEqual(request.get_header("Accept-encoding"), "identity")
            self.assertIn("ParetoPilot/", request.get_header("User-agent"))

            oversized = root / "oversized.zip"
            with (
                patch(
                    "paretopilot.published_proof.urllib.request.urlopen",
                    return_value=_DownloadResponse(payload + b"!"),
                ),
                self.assertRaisesRegex(ValidationError, "exceeds its pinned size"),
            ):
                _download_archive(lock, oversized, label="canonical")
            self.assertFalse(oversized.exists())

            existing = root / "existing.zip"
            existing.write_bytes(b"preserve me")
            with (
                patch(
                    "paretopilot.published_proof.urllib.request.urlopen",
                    return_value=_DownloadResponse(payload),
                ),
                self.assertRaisesRegex(ValidationError, "could not download"),
            ):
                _download_archive(lock, existing, label="canonical")
            self.assertEqual(existing.read_bytes(), b"preserve me")

            interrupted = root / "interrupted.zip"
            with (
                patch(
                    "paretopilot.published_proof.urllib.request.urlopen",
                    return_value=_InterruptedDownloadResponse(payload),
                ),
                self.assertRaisesRegex(ValidationError, "could not download"),
            ):
                _download_archive(lock, interrupted, label="canonical")
            self.assertFalse(interrupted.exists())

    def test_publish_race_preserves_an_existing_destination(self) -> None:
        with TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            staged = root / "staged"
            staged.mkdir()
            (staged / "published-proof.json").write_text("{}\n", encoding="utf-8")
            (staged / "published-proof.md").write_text("# Proof\n", encoding="utf-8")
            destination = root / "raced-output"
            destination.mkdir()
            marker = destination / "keep.txt"
            marker.write_text("keep\n", encoding="utf-8")

            with self.assertRaisesRegex(ValidationError, "refusing to overwrite"):
                _publish_output(staged, destination)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")

    def test_publish_rejects_a_destination_symlink_without_writing_through_it(self) -> None:
        with TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            staged = root / "staged"
            staged.mkdir()
            (staged / "published-proof.json").write_text("{}\n", encoding="utf-8")
            (staged / "published-proof.md").write_text("# Proof\n", encoding="utf-8")
            outside = root / "outside"
            outside.mkdir()
            marker = outside / "keep.txt"
            marker.write_text("keep\n", encoding="utf-8")
            destination = root / "raced-output"
            try:
                os.symlink(outside, destination, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symbolic links are unavailable: {exc}")

            with self.assertRaisesRegex(ValidationError, "refusing to overwrite"):
                _publish_output(staged, destination)

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")
            self.assertFalse((outside / "published-proof.json").exists())
            self.assertFalse((outside / "published-proof.md").exists())

    def test_publish_swap_race_never_writes_through_a_directory_symlink(self) -> None:
        with TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            staged = root / "staged"
            staged.mkdir()
            (staged / "published-proof.json").write_text("{}\n", encoding="utf-8")
            (staged / "published-proof.md").write_text("# Proof\n", encoding="utf-8")
            outside = root / "outside"
            outside.mkdir()
            marker = outside / "keep.txt"
            marker.write_text("keep\n", encoding="utf-8")
            destination = root / "raced-output"
            parked = root / "parked-output"
            original_mkdir = Path.mkdir
            original_rename = Path.rename

            def racing_mkdir(path: Path, *args: object, **kwargs: object) -> None:
                original_mkdir(path, *args, **kwargs)
                if path == destination:
                    original_rename(path, parked)
                    os.symlink(outside, destination, target_is_directory=True)

            def racing_rename(path: Path, target: Path) -> Path:
                if path == staged and Path(target) == destination and not destination.exists():
                    os.symlink(outside, destination, target_is_directory=True)
                return original_rename(path, target)

            try:
                os.symlink(outside, root / "symlink-check", target_is_directory=True)
                (root / "symlink-check").unlink()
            except OSError as exc:
                self.skipTest(f"directory symbolic links are unavailable: {exc}")

            with (
                patch.object(Path, "mkdir", new=racing_mkdir),
                patch.object(Path, "rename", new=racing_rename),
            ):
                try:
                    _publish_output(staged, destination)
                except ValidationError:
                    pass

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")
            self.assertFalse((outside / "published-proof.json").exists())
            self.assertFalse((outside / "published-proof.md").exists())

    def test_packaged_lock_rejects_duplicate_keys_at_any_depth(self) -> None:
        duplicate_documents = (
            '{"schema_version":"1.0","schema_version":"1.0"}',
            '{"archives":{"canonical":{"sha256":"a","sha256":"b"}}}',
        )
        for document in duplicate_documents:
            with (
                self.subTest(document=document),
                patch("paretopilot.published_proof.resources.files") as files,
                self.assertRaisesRegex(ValidationError, "duplicate object key"),
            ):
                files.return_value.joinpath.return_value.read_text.return_value = document
                _load_published_lock()

    def test_proof_contract_fails_closed_on_replay_drift(self) -> None:
        with TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            canonical_archive = root / "canonical.zip"
            capacity_archive = root / "capacity.zip"
            _write_zip(canonical_archive, [("canonical.txt", b"canonical\n")])
            _write_zip(capacity_archive, [("capacity.txt", b"capacity\n")])
            lock = PublishedEvidenceLock(
                canonical=_archive_lock(
                    canonical_archive,
                    release_tag="v1.1.0",
                    label="canonical",
                ),
                capacity=_archive_lock(
                    capacity_archive,
                    release_tag="v1.4.0",
                    label="capacity",
                ),
                evidence_limits=("Limit.",),
            )
            cases = (
                (
                    "wrong selection",
                    lambda canonical, _capacity: canonical.update(selected_id="q4-generic"),
                    "must select",
                ),
                (
                    "warning",
                    lambda canonical, _capacity: canonical["warnings"].append("drift"),
                    "zero differences and zero warnings",
                ),
                (
                    "missing comparison",
                    lambda canonical, _capacity: canonical["authoritative_comparisons"].pop(
                        "report-v1.1"
                    ),
                    "nine pinned authoritative comparisons",
                ),
                (
                    "wrong capacity point",
                    lambda _canonical, capacity: capacity["selected_operating_points"][
                        "q8-generic"
                    ].update(client_concurrency=2),
                    "select P4/C4",
                ),
            )
            for name, mutate, message in cases:
                canonical = copy.deepcopy(_canonical_replay())
                capacity = copy.deepcopy(_capacity_replay(lock.canonical.sha256))
                mutate(canonical, capacity)
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValidationError, message),
                ):
                    _build_proof(lock, canonical, capacity)

    def test_safe_extraction_rejects_unsafe_duplicate_and_colliding_paths(self) -> None:
        cases: tuple[tuple[str, list[tuple[str | zipfile.ZipInfo, bytes]], str], ...] = (
            ("traversal", [("../escape.txt", b"bad")], "unsafe path"),
            ("absolute", [("/absolute.txt", b"bad")], "unsafe path"),
            ("drive", [("C:/drive.txt", b"bad")], "unsafe path"),
            (
                "duplicate",
                [("same.txt", b"one"), ("same.txt", b"two")],
                "duplicate or case-colliding",
            ),
            (
                "case-collision",
                [("Case.txt", b"one"), ("case.txt", b"two")],
                "duplicate or case-colliding",
            ),
            (
                "file-directory",
                [("parent", b"file"), ("parent/child.txt", b"child")],
                "nested below a file",
            ),
            ("reserved", [("CON.txt", b"bad")], "unsafe path"),
            ("trailing-dot", [("file.", b"bad")], "unsafe path"),
        )
        with TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            for name, entries, message in cases:
                with self.subTest(name=name):
                    archive = root / f"{name}.zip"
                    destination = root / f"{name}-output"
                    _write_zip(archive, entries)
                    destination.mkdir()
                    with self.assertRaisesRegex(ValidationError, message):
                        _extract_zip_safely(archive, destination, label="test")
            self.assertFalse((root / "escape.txt").exists())

        backslash = zipfile.ZipInfo("placeholder")
        backslash.filename = "folder\\file.txt"
        with self.assertRaisesRegex(ValidationError, "unsafe path"):
            _safe_zip_member(backslash, label="test", index=1)

    def test_safe_extraction_rejects_symlinks_special_entries_and_limits(self) -> None:
        symlink = zipfile.ZipInfo("link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        fifo = zipfile.ZipInfo("pipe")
        fifo.create_system = 3
        fifo.external_attr = (stat.S_IFIFO | 0o644) << 16

        with TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            for name, info in (("symlink", symlink), ("fifo", fifo)):
                with self.subTest(name=name):
                    archive = root / f"{name}.zip"
                    destination = root / f"{name}-output"
                    _write_zip(archive, [(info, b"target")])
                    destination.mkdir()
                    with self.assertRaisesRegex(
                        ValidationError,
                        "symbolic link or special entry",
                    ):
                        _extract_zip_safely(archive, destination, label="test")

            member_archive = root / "member-limit.zip"
            _write_zip(member_archive, [("one", b"1"), ("two", b"2")])
            member_output = root / "member-limit-output"
            member_output.mkdir()
            with (
                patch("paretopilot.published_proof._MAX_ARCHIVE_MEMBERS", 1),
                self.assertRaisesRegex(ValidationError, "too many members"),
            ):
                _extract_zip_safely(member_archive, member_output, label="test")

            size_archive = root / "size-limit.zip"
            _write_zip(size_archive, [("large.txt", b"1234")])
            size_output = root / "size-limit-output"
            size_output.mkdir()
            with (
                patch("paretopilot.published_proof._MAX_ARCHIVE_UNCOMPRESSED_BYTES", 3),
                self.assertRaisesRegex(ValidationError, "uncompressed size limit"),
            ):
                _extract_zip_safely(size_archive, size_output, label="test")

    def test_cli_prints_a_short_pass_and_supports_offline_archive_flags(self) -> None:
        proof = {
            "verdict": "PASS: test proof",
        }
        with TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            output = root / "proof"
            stdout = io.StringIO()
            with (
                patch(
                    "paretopilot.cli.verify_published_evidence",
                    return_value=proof,
                ) as verify,
                patch("sys.stdout", stdout),
            ):
                exit_code = cli.main(
                    [
                        "verify-published",
                        "--canonical-archive",
                        str(root / "canonical.zip"),
                        "--capacity-archive",
                        str(root / "capacity.zip"),
                        "--output-dir",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                stdout.getvalue().splitlines(),
                [
                    "PASS: test proof",
                    f"Report: {output.resolve() / 'published-proof.md'}",
                ],
            )
            verify.assert_called_once_with(
                output,
                canonical_archive=root / "canonical.zip",
                capacity_archive=root / "capacity.zip",
            )

    def test_cli_help_explains_online_offline_and_output_behavior(self) -> None:
        stdout = io.StringIO()
        with (
            patch("sys.stdout", stdout),
            self.assertRaises(SystemExit) as exit_context,
        ):
            cli.main(["verify-published", "--help"])

        self.assertEqual(exit_context.exception.code, 0)
        help_text = stdout.getvalue()
        normalized_help = " ".join(help_text.split())
        self.assertIn("does not require an Arm64 machine", normalized_help)
        self.assertIn(
            "Supply both archive options for a fully offline check",
            normalized_help,
        )
        self.assertIn("must not already exist", normalized_help)
        self.assertIn(
            "new directory for the Markdown and JSON verification results",
            normalized_help,
        )

    def test_local_symbolic_link_archive_is_rejected_when_supported(self) -> None:
        with TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            target = root / "target.zip"
            _write_zip(target, [("safe.txt", b"safe")])
            link = root / "link.zip"
            try:
                os.symlink(target, link)
            except OSError as exc:
                self.skipTest(f"symbolic links are unavailable: {exc}")
            lock = PublishedEvidenceLock(
                canonical=_archive_lock(target, release_tag="v1.1.0", label="canonical"),
                capacity=_archive_lock(target, release_tag="v1.4.0", label="capacity"),
                evidence_limits=("Limit.",),
            )
            with (
                patch(
                    "paretopilot.published_proof._load_published_lock",
                    return_value=lock,
                ),
                self.assertRaisesRegex(ValidationError, "symbolic link"),
            ):
                verify_published_evidence(
                    root / "proof",
                    canonical_archive=link,
                    capacity_archive=target,
                )


if __name__ == "__main__":
    unittest.main()
