from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from paretopilot.capacity_replay import (
    _extract_zip_safely,
    replay_capacity_bundle,
)
from paretopilot.domain import ValidationError


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write(path: Path, content: bytes = b"fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_checksums(root: Path) -> None:
    entries = []
    for path in sorted(
        (path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"),
        key=lambda path: path.relative_to(root).as_posix(),
    ):
        entries.append(f"{_digest(path)}  ./{path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _study() -> dict[str, object]:
    return {
        "selections": [
            {
                "candidate_id": "candidate-a",
                "eligible_cell_count": 1,
                "selected_cell": {
                    "server_parallel": 1,
                    "client_concurrency": 1,
                },
            }
        ],
        "provenance": {
            "canonical_evidence": {
                "release_sha256": "0" * 64,
            }
        },
    }


def _status(study: dict[str, object]) -> dict[str, object]:
    selection = study["selections"][0]  # type: ignore[index]
    return {
        "schema_version": "1.0",
        "status": "complete",
        "classification": "supplementary-capacity",
        "measurement_valid": True,
        "canonical_outputs_modified": False,
        "eligible_cell_counts": {
            selection["candidate_id"]: selection["eligible_cell_count"],  # type: ignore[index]
        },
        "selected_operating_points": {
            selection["candidate_id"]: selection["selected_cell"],  # type: ignore[index]
        },
        "reason": "fixture capacity study completed",
    }


def _plan() -> SimpleNamespace:
    candidate = SimpleNamespace(id="candidate-a")
    pass_spec = SimpleNamespace(
        id="forward",
        candidate_order=("candidate-a",),
        server_parallel_order=(1,),
    )
    return SimpleNamespace(
        candidates=(candidate,),
        passes=(pass_spec,),
        server_parallel_levels=(1,),
    )


def _bundle(root: Path, *, authoritative_study: dict[str, object] | None = None) -> Path:
    root.mkdir()
    study = authoritative_study or _study()
    for name in (
        "capacity-plan.json",
        "evaluation-suite.json",
        "load-plan.json",
    ):
        _write_json(root / name, {})
    _write_json(
        root / "manifest.json",
        {
            "canonical_evidence": {
                "release_sha256": "0" * 64,
            }
        },
    )
    _write_json(root / "capacity-study.json", study)
    (root / "capacity-receipt.md").write_text(
        "fixture receipt\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_json(root / "status.json", _status(study))

    _write(root / "runs/forward/candidate-a/p1/load-evaluation.json")
    _write(root / "runs/forward/candidate-a/p1/server-time.txt")
    _write(root / "runs/forward/candidate-a/p1/server.stderr.log")
    _write(root / "quality/candidate-a/p1/quality-evidence.json")

    frozen_checksums = b"fixture canonical checksums\n"
    _write(root / "canonical/frozen-SHA256SUMS", frozen_checksums)
    archive_path = root / "canonical/frozen.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("SHA256SUMS", frozen_checksums)
    archive_digest = _digest(archive_path)
    (root / "canonical/release-archive.sha256").write_text(
        f"{archive_digest}  canonical/frozen.zip\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_json(root / "canonical/replay.json", {"valid": True})
    _rewrite_checksums(root)
    return root


class CapacityReplayTests(unittest.TestCase):
    def test_replay_rebuilds_exact_outputs_and_publishes_only_new_directory(self) -> None:
        from tempfile import TemporaryDirectory

        study = _study()
        canonical_result = {
            "verified": True,
            "archive_path": "canonical/frozen.zip",
            "archive_sha256": "a" * 64,
            "replay_sha256": "b" * 64,
            "replay_contract": "1.1",
            "selected_id": "candidate-a",
            "fully_reproduced": True,
        }
        with TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            bundle = _bundle(temporary / "bundle", authoritative_study=study)
            before = {
                path.relative_to(bundle).as_posix(): path.read_bytes()
                for path in bundle.rglob("*")
                if path.is_file()
            }
            output = temporary / "replayed"
            with (
                patch(
                    "paretopilot.capacity_replay.load_capacity_plan",
                    return_value=_plan(),
                ),
                patch(
                    "paretopilot.capacity_replay.assemble_capacity_study",
                    return_value=study,
                ) as assemble,
                patch(
                    "paretopilot.capacity_replay.render_capacity_receipt",
                    return_value="fixture receipt\n",
                ),
                patch(
                    "paretopilot.capacity_replay._verify_canonical_replay",
                    return_value=canonical_result,
                ),
            ):
                result = replay_capacity_bundle(bundle, output)

            self.assertTrue(result["valid"])
            self.assertTrue(result["capacity_study_reproduced"])
            self.assertTrue(result["capacity_receipt_reproduced"])
            self.assertEqual(result["canonical_replay"], canonical_result)
            self.assertEqual(
                (output / "capacity-study.reproduced.json").read_bytes(),
                (bundle / "capacity-study.json").read_bytes(),
            )
            self.assertEqual(
                (output / "capacity-receipt.reproduced.md").read_bytes(),
                (bundle / "capacity-receipt.md").read_bytes(),
            )
            self.assertTrue((output / "capacity-replay.json").is_file())
            after = {
                path.relative_to(bundle).as_posix(): path.read_bytes()
                for path in bundle.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)

            arguments = assemble.call_args.kwargs
            self.assertEqual(
                [label for label, _path in arguments["load_artifacts"]],
                ["forward/candidate-a/p1"],
            )
            self.assertEqual(
                [label for label, _path in arguments["rss_artifacts"]],
                ["forward/candidate-a/p1"],
            )
            self.assertEqual(
                [label for label, _path in arguments["server_logs"]],
                ["forward/candidate-a/p1"],
            )
            self.assertEqual(
                [label for label, _path in arguments["quality_artifacts"]],
                ["candidate-a/p1"],
            )

    def test_replay_fails_closed_on_unlisted_or_unsafe_checksum_paths(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            unlisted = _bundle(temporary / "unlisted")
            _write(unlisted / "not-checksummed.txt")
            with self.assertRaisesRegex(
                ValidationError,
                "files missing from SHA256SUMS",
            ):
                replay_capacity_bundle(unlisted, temporary / "unlisted-output")

            unsafe = _bundle(temporary / "unsafe")
            with (unsafe / "SHA256SUMS").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(f"{'0' * 64}  ../escape\n")
            with self.assertRaisesRegex(ValidationError, "unsafe path"):
                replay_capacity_bundle(unsafe, temporary / "unsafe-output")

    def test_replay_fails_closed_on_derived_output_or_canonical_mismatch(self) -> None:
        from tempfile import TemporaryDirectory

        archived = _study()
        rebuilt = {**archived, "unexpected_difference": True}
        with TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            bundle = _bundle(temporary / "bundle", authoritative_study=archived)
            output = temporary / "mismatch-output"
            with (
                patch(
                    "paretopilot.capacity_replay.load_capacity_plan",
                    return_value=_plan(),
                ),
                patch(
                    "paretopilot.capacity_replay.assemble_capacity_study",
                    return_value=rebuilt,
                ),
                patch(
                    "paretopilot.capacity_replay.render_capacity_receipt",
                    return_value="fixture receipt\n",
                ),
            ):
                with self.assertRaisesRegex(ValidationError, "not byte-identical"):
                    replay_capacity_bundle(bundle, output)
            self.assertFalse(output.exists())

            canonical_output = temporary / "canonical-output"
            with (
                patch(
                    "paretopilot.capacity_replay.load_capacity_plan",
                    return_value=_plan(),
                ),
                patch(
                    "paretopilot.capacity_replay.assemble_capacity_study",
                    return_value=archived,
                ),
                patch(
                    "paretopilot.capacity_replay.render_capacity_receipt",
                    return_value="fixture receipt\n",
                ),
                patch(
                    "paretopilot.capacity_replay._verify_canonical_replay",
                    side_effect=ValidationError("canonical replay mismatch"),
                ),
            ):
                with self.assertRaisesRegex(ValidationError, "canonical replay mismatch"):
                    replay_capacity_bundle(bundle, canonical_output)
            self.assertFalse(canonical_output.exists())

    def test_safe_zip_extraction_rejects_path_traversal(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            archive_path = temporary / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.txt", "bad")
            destination = temporary / "destination"
            destination.mkdir()
            with self.assertRaisesRegex(ValidationError, "unsafe path"):
                _extract_zip_safely(archive_path, destination)
            self.assertFalse((temporary / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
