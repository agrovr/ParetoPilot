from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from paretopilot import cli
from paretopilot.doctor import EnvironmentReport


class CliTests(unittest.TestCase):
    def test_commit_match_accepts_normal_sha_prefixes_only(self) -> None:
        self.assertTrue(
            cli._commits_match("67b9b0e7", cli.PINNED_LLAMA_CPP_COMMIT)
        )
        self.assertFalse(cli._commits_match("6", cli.PINNED_LLAMA_CPP_COMMIT))

    def test_doctor_prints_report(self) -> None:
        report = EnvironmentReport(
            machine_architecture="AMD64",
            processor="test-processor",
            platform="test-platform",
            operating_system="Windows",
            os_release="11",
            python_version="3.12.0",
            is_arm64=False,
            evidence_eligible=False,
            warnings=("smoke-test-only",),
        )
        output = io.StringIO()
        with patch.object(cli, "inspect_environment", return_value=report), patch(
            "sys.stdout", output
        ):
            exit_code = cli.main(["doctor"])

        self.assertEqual(exit_code, 0)
        self.assertFalse(json.loads(output.getvalue())["evidence_eligible"])

    def test_doctor_can_require_arm64(self) -> None:
        report = EnvironmentReport(
            machine_architecture="AMD64",
            processor="unknown",
            platform="test-platform",
            operating_system="Windows",
            os_release="11",
            python_version="3.12.0",
            is_arm64=False,
            evidence_eligible=False,
            warnings=("smoke-test-only",),
        )
        with patch.object(cli, "inspect_environment", return_value=report), patch(
            "sys.stdout", io.StringIO()
        ):
            exit_code = cli.main(["doctor", "--require-evidence-host"])

        self.assertEqual(exit_code, 3)

    def test_validate_llama_bench_summarizes_fixture(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "llama_bench.synthetic.jsonl"
        output = io.StringIO()
        with patch("sys.stdout", output):
            exit_code = cli.main(["validate-llama-bench", str(fixture)])

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["record_count"], 3)
        self.assertEqual(payload["test_counts"], {"pg": 1, "pp": 1, "tg": 1})
        self.assertEqual(payload["repetition_counts"], [2])
        self.assertFalse(payload["evidence_valid"])
        self.assertTrue(payload["synthetic_fixture"])

    def test_validate_llama_bench_evidence_gate_is_nonzero_for_fixture(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "llama_bench.synthetic.jsonl"
        output = io.StringIO()
        with patch("sys.stdout", output):
            exit_code = cli.main(
                ["validate-llama-bench", str(fixture), "--evidence"]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 4)
        self.assertFalse(payload["evidence_valid"])
        self.assertTrue(
            any("synthetic" in issue for issue in payload["evidence_issues"])
        )

    def test_validate_llama_bench_refuses_to_overwrite_input(self) -> None:
        source = Path(__file__).parent / "fixtures" / "llama_bench.synthetic.jsonl"
        with TemporaryDirectory() as directory:
            copied = Path(directory) / "raw.jsonl"
            copied.write_bytes(source.read_bytes())
            original = copied.read_bytes()
            with patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
                exit_code = cli.main(
                    ["validate-llama-bench", str(copied), "--output", str(copied)]
                )

            self.assertEqual(exit_code, 2)
            self.assertEqual(copied.read_bytes(), original)

    def test_recommendation_includes_input_fingerprints(self) -> None:
        root = Path(__file__).parents[1]
        output = io.StringIO()
        with patch("sys.stdout", output):
            exit_code = cli.main(
                [
                    "recommend",
                    str(root / "examples" / "synthetic-results.json"),
                    "--constraints",
                    str(root / "configs" / "constraints.example.json"),
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(payload["input_fingerprints"]["benchmarks_sha256"]), 64)
        self.assertEqual(len(payload["input_fingerprints"]["constraints_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
