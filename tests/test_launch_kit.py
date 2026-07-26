from __future__ import annotations

import io
import json
import os
from pathlib import Path, PurePosixPath
import shlex
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from paretopilot import cli
from paretopilot.analysis import recommend
from paretopilot.domain import ValidationError
from paretopilot.io import load_benchmarks, load_constraints, write_text
from paretopilot import launch_kit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILES = {
    ".github/workflows/paretopilot.yml",
    ".gitignore",
    "README.md",
    "benchmarks/benchmark-set.json",
    "constraints/deployment.json",
}
ACTION_SHA = "db9ccaf37e3c7e807832652e237de813675ed807"


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


class LaunchKitTests(unittest.TestCase):
    def test_launch_kit_writes_exact_expected_tree(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"

            first_payload = launch_kit.create_launch_kit(first)
            launch_kit.create_launch_kit(second)

            self.assertEqual(set(first_payload["files"]), EXPECTED_FILES)
            self.assertEqual(_tree_bytes(first), _tree_bytes(second))
            self.assertEqual(set(_tree_bytes(first)), EXPECTED_FILES)
            self.assertEqual(first_payload["classification"], "synthetic-launch-kit")
            self.assertTrue(first_payload["synthetic_source"])
            self.assertEqual(first_payload["selected_id"], "q4-kleidiai")
            self.assertEqual(first_payload["output_directory"], str(first.resolve()))
            self.assertEqual(first_payload["working_directory"], str(first.resolve()))

            command = shlex.split(str(first_payload["next_command"]))
            with patch("sys.stdout", io.StringIO()):
                self.assertEqual(cli.main(command[1:]), 0)
            self.assertTrue((first / "paretopilot-output" / "gate.json").is_file())

    def test_template_inputs_match_public_examples_byte_for_byte(self) -> None:
        templates = launch_kit._load_templates()

        self.assertEqual(
            templates[PurePosixPath("benchmarks/benchmark-set.json")],
            (REPOSITORY_ROOT / "examples" / "synthetic-results.json").read_bytes(),
        )
        self.assertEqual(
            templates[PurePosixPath("constraints/deployment.json")],
            (REPOSITORY_ROOT / "configs" / "constraints.example.json").read_bytes(),
        )

    def test_generated_inputs_validate_and_select_q4_kleidiai(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "kit"
            launch_kit.create_launch_kit(root)

            benchmarks = load_benchmarks(root / "benchmarks" / "benchmark-set.json")
            constraints = load_constraints(root / "constraints" / "deployment.json")
            decision = recommend(benchmarks, constraints)

            self.assertTrue(benchmarks.synthetic)
            self.assertEqual(decision["selected_id"], "q4-kleidiai")

    def test_generated_workflow_is_read_only_pinned_and_explicitly_synthetic(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "kit"
            launch_kit.create_launch_kit(root)
            workflow = (root / ".github" / "workflows" / "paretopilot.yml").read_text(
                encoding="utf-8"
            )

            self.assertIn("permissions:\n  contents: read", workflow)
            self.assertIn(f"uses: agrovr/ParetoPilot@{ACTION_SHA} # v1.4.0", workflow)
            self.assertIn('require-measured: "false"', workflow)
            self.assertIn("expected-selected-id: q4-kleidiai", workflow)
            self.assertNotIn("require-arm64-provenance", workflow)
            self.assertNotIn("pull_request_target", workflow)

    def test_every_generated_text_file_is_utf8_lf_with_trailing_newline(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "kit"
            launch_kit.create_launch_kit(root)

            for relative, data in _tree_bytes(root).items():
                with self.subTest(path=relative):
                    data.decode("utf-8")
                    self.assertNotIn(b"\r", data)
                    self.assertTrue(data.endswith(b"\n"))

    def test_existing_file_directory_and_symlink_are_preserved(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            existing_file = root / "file"
            existing_file.write_text("keep me\n", encoding="utf-8")
            existing_directory = root / "directory"
            existing_directory.mkdir()
            marker = existing_directory / "marker.txt"
            marker.write_text("keep me too\n", encoding="utf-8")

            for destination in (existing_file, existing_directory):
                with self.subTest(destination=destination):
                    with self.assertRaisesRegex(ValidationError, "refusing to overwrite"):
                        launch_kit.create_launch_kit(destination)

            self.assertEqual(existing_file.read_text(encoding="utf-8"), "keep me\n")
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep me too\n")

            symlink = root / "symlink"
            try:
                os.symlink(existing_directory, symlink, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symbolic links are unavailable: {exc}")
            with self.assertRaisesRegex(ValidationError, "refusing to overwrite"):
                launch_kit.create_launch_kit(symlink)
            self.assertTrue(symlink.is_symlink())

    def test_broken_symlink_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "broken-link"
            try:
                os.symlink(root / "missing-target", destination, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symbolic links are unavailable: {exc}")

            with self.assertRaisesRegex(ValidationError, "refusing to overwrite"):
                launch_kit.create_launch_kit(destination)
            self.assertTrue(destination.is_symlink())

    def test_missing_or_non_directory_parent_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            file_parent = root / "file-parent"
            file_parent.write_text("not a directory\n", encoding="utf-8")

            for destination in (root / "missing" / "kit", file_parent / "kit"):
                with self.subTest(destination=destination):
                    with self.assertRaisesRegex(ValidationError, "parent|resolve"):
                        launch_kit.create_launch_kit(destination)
                    self.assertFalse(os.path.lexists(destination))

    def test_template_validation_failure_creates_no_destination(self) -> None:
        templates = dict(launch_kit._load_templates())
        benchmark_path = PurePosixPath("benchmarks/benchmark-set.json")
        templates[benchmark_path] = templates[benchmark_path].replace(
            b'"synthetic": true',
            b'"synthetic": false',
            1,
        )

        with TemporaryDirectory() as directory:
            destination = Path(directory) / "kit"
            with (
                patch.object(launch_kit, "_load_templates", return_value=templates),
                self.assertRaisesRegex(ValidationError, "explicitly synthetic"),
            ):
                launch_kit.create_launch_kit(destination)
            self.assertFalse(os.path.lexists(destination))

    def test_write_failure_preserves_the_incomplete_destination(self) -> None:
        original_write_text = write_text
        calls = 0

        def fail_after_one_write(path: Path, text: str, *, overwrite: bool = False) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                original_write_text(path, text, overwrite=overwrite)
                return
            raise ValidationError("injected write failure")

        with TemporaryDirectory() as directory:
            destination = Path(directory) / "kit"
            with (
                patch.object(launch_kit, "write_text", side_effect=fail_after_one_write),
                self.assertRaisesRegex(ValidationError, "incomplete starter project remains"),
            ):
                launch_kit.create_launch_kit(destination)
            first_file = destination / ".github" / "workflows" / "paretopilot.yml"
            self.assertTrue(first_file.is_file())
            self.assertIn("ParetoPilot decision", first_file.read_text(encoding="utf-8"))

    def test_changed_file_is_preserved_after_a_write_failure(self) -> None:
        original_write_text = write_text
        calls = 0

        def change_then_fail(path: Path, text: str, *, overwrite: bool = False) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                original_write_text(path, text, overwrite=overwrite)
                path.write_text("unexpected external content\n", encoding="utf-8")
                return
            raise ValidationError("injected write failure")

        with TemporaryDirectory() as directory:
            destination = Path(directory) / "kit"
            with (
                patch.object(launch_kit, "write_text", side_effect=change_then_fail),
                self.assertRaisesRegex(ValidationError, "incomplete starter project remains"),
            ):
                launch_kit.create_launch_kit(destination)
            changed = destination / ".github" / "workflows" / "paretopilot.yml"
            self.assertEqual(
                changed.read_text(encoding="utf-8"),
                "unexpected external content\n",
            )

    def test_preflight_filesystem_error_is_normalized_without_a_destination(self) -> None:
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "kit"
            with (
                patch.object(
                    launch_kit,
                    "TemporaryDirectory",
                    side_effect=PermissionError("injected preflight denial"),
                ),
                self.assertRaisesRegex(ValidationError, "could not validate.*preflight denial"),
            ):
                launch_kit.create_launch_kit(destination)
            self.assertFalse(os.path.lexists(destination))


class LaunchKitCliTests(unittest.TestCase):
    def test_cli_creates_kit_then_refuses_to_overwrite_it(self) -> None:
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "kit"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                exit_code = cli.main(["init", str(destination)])

            payload = json.loads(stdout.getvalue())
            before = _tree_bytes(destination)
            self.assertEqual(exit_code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(set(payload["files"]), EXPECTED_FILES)
            self.assertEqual(payload["working_directory"], str(destination.resolve()))

            second_stdout = io.StringIO()
            second_stderr = io.StringIO()
            with patch("sys.stdout", second_stdout), patch("sys.stderr", second_stderr):
                second_exit = cli.main(["init", str(destination)])

            self.assertEqual(second_exit, 2)
            self.assertEqual(second_stdout.getvalue(), "")
            self.assertTrue(second_stderr.getvalue().startswith("error: "))
            self.assertEqual(_tree_bytes(destination), before)

    def test_generated_files_run_through_ci_gate_and_keep_synthetic_guard(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            kit = root / "kit"
            output = root / "output"
            with patch("sys.stdout", io.StringIO()):
                self.assertEqual(cli.main(["init", str(kit)]), 0)

            gate_stdout = io.StringIO()
            with patch("sys.stdout", gate_stdout):
                gate_exit = cli.main(
                    [
                        "ci-gate",
                        str(kit / "benchmarks" / "benchmark-set.json"),
                        "--constraints",
                        str(kit / "constraints" / "deployment.json"),
                        "--output-dir",
                        str(output),
                        "--allow-synthetic",
                        "--expect-selected-id",
                        "q4-kleidiai",
                    ]
                )

            self.assertEqual(gate_exit, 0)
            self.assertEqual(json.loads(gate_stdout.getvalue())["selected_id"], "q4-kleidiai")
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "recommendation.json",
                    "decision-passport.json",
                    "optimization-receipt.md",
                    "report.html",
                    "gate.json",
                },
            )

            guarded_output = root / "guarded-output"
            with patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
                guarded_exit = cli.main(
                    [
                        "ci-gate",
                        str(kit / "benchmarks" / "benchmark-set.json"),
                        "--constraints",
                        str(kit / "constraints" / "deployment.json"),
                        "--output-dir",
                        str(guarded_output),
                    ]
                )
            self.assertEqual(guarded_exit, 2)
            self.assertFalse(guarded_output.exists())


if __name__ == "__main__":
    unittest.main()
