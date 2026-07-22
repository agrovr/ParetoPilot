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

    def test_validate_llama_bench_evidence_checks_reported_runtime_settings(self) -> None:
        common = {
            "build_commit": cli.PINNED_LLAMA_CPP_COMMIT[:7],
            "model_filename": "model.gguf",
            "n_threads": 4,
            "n_batch": 512,
            "n_ubatch": 128,
            "n_gpu_layers": 0,
            "devices": "none",
            "no_op_offload": 1,
            "avg_ns": 100.0,
            "avg_ts": 10.0,
            "samples_ns": [100.0] * 10,
            "samples_ts": [10.0] * 10,
        }
        rows = [
            {**common, "n_prompt": 512, "n_gen": 0},
            {**common, "n_prompt": 0, "n_gen": 128},
        ]
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "evidence.jsonl"
            artifact.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch("sys.stdout", output):
                exit_code = cli.main(
                    [
                        "validate-llama-bench",
                        str(artifact),
                        "--evidence",
                        "--expected-threads",
                        "4",
                        "--expected-batch",
                        "512",
                        "--expected-ubatch",
                        "128",
                    ]
                )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["evidence_valid"])
        self.assertEqual(payload["execution_settings"]["devices"], ["none"])

    def test_validate_llama_bench_evidence_rejects_reported_setting_mismatch(self) -> None:
        common = {
            "build_commit": cli.PINNED_LLAMA_CPP_COMMIT[:7],
            "model_filename": "model.gguf",
            "n_threads": 4,
            "n_batch": 256,
            "n_ubatch": 128,
            "n_gpu_layers": 0,
            "devices": "none",
            "no_op_offload": 1,
            "avg_ns": 100.0,
            "avg_ts": 10.0,
            "samples_ns": [100.0] * 10,
            "samples_ts": [10.0] * 10,
        }
        rows = [
            {**common, "n_prompt": 512, "n_gen": 0},
            {**common, "n_prompt": 0, "n_gen": 128},
        ]
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "evidence.jsonl"
            artifact.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch("sys.stdout", output):
                exit_code = cli.main(
                    [
                        "validate-llama-bench",
                        str(artifact),
                        "--evidence",
                        "--expected-batch",
                        "512",
                    ]
                )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 4)
        self.assertIn("records must use n_batch=512", payload["evidence_issues"])

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

    def test_summarize_llama_bench_writes_pooled_labeled_artifacts(self) -> None:
        source = Path(__file__).parent / "fixtures" / "llama_bench.synthetic.jsonl"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_bytes(source.read_bytes())
            second.write_bytes(source.read_bytes())
            settings = root / "settings.json"
            settings.write_text(
                json.dumps({"threads": 4, "build": {"kleidiai": False}}),
                encoding="utf-8",
            )
            destination = root / "summary.json"
            stdout = io.StringIO()

            with patch("sys.stdout", stdout):
                exit_code = cli.main(
                    [
                        "summarize-llama-bench",
                        "--label",
                        "generic",
                        "--artifact",
                        f"pass-1={first}",
                        "--artifact",
                        f"pass-2={second}",
                        "--settings",
                        str(settings),
                        "--output",
                        str(destination),
                    ]
                )

            printed = json.loads(stdout.getvalue())
            written = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed, written)
            self.assertEqual(written["label"], "generic")
            self.assertEqual(written["source_labels"], ["pass-1", "pass-2"])
            self.assertEqual(
                written["tests"]["pp"]["tokens_per_second"]["sample_count"],
                4,
            )
            self.assertEqual(
                set(written["input_fingerprints"]["artifacts_sha256"]),
                {"pass-1", "pass-2"},
            )
            self.assertEqual(
                len(written["input_fingerprints"]["settings_sha256"]),
                64,
            )

    def test_summarize_rejects_duplicate_labels_and_existing_output(self) -> None:
        source = Path(__file__).parent / "fixtures" / "llama_bench.synthetic.jsonl"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = root / "settings.json"
            settings.write_text('{"threads": 4}', encoding="utf-8")
            destination = root / "summary.json"
            destination.write_text('{"preserve": true}\n', encoding="utf-8")
            original = destination.read_bytes()
            stderr = io.StringIO()

            with patch("sys.stdout", io.StringIO()), patch("sys.stderr", stderr):
                duplicate_exit = cli.main(
                    [
                        "summarize-llama-bench",
                        "--label",
                        "generic",
                        "--artifact",
                        f"pass={source}",
                        "--artifact",
                        f"pass={source}",
                        "--settings",
                        str(settings),
                        "--output",
                        str(root / "unused.json"),
                    ]
                )
                overwrite_exit = cli.main(
                    [
                        "summarize-llama-bench",
                        "--label",
                        "generic",
                        "--artifact",
                        f"pass={source}",
                        "--settings",
                        str(settings),
                        "--output",
                        str(destination),
                    ]
                )

            self.assertEqual(duplicate_exit, 2)
            self.assertEqual(overwrite_exit, 2)
            self.assertIn("must be unique", stderr.getvalue())
            self.assertIn("refusing to overwrite", stderr.getvalue())
            self.assertEqual(destination.read_bytes(), original)

    def test_summarize_uses_strict_json_for_settings(self) -> None:
        source = Path(__file__).parent / "fixtures" / "llama_bench.synthetic.jsonl"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = root / "settings.json"
            settings.write_text('{"threads": 4, "threads": 8}', encoding="utf-8")
            stderr = io.StringIO()

            with patch("sys.stdout", io.StringIO()), patch("sys.stderr", stderr):
                exit_code = cli.main(
                    [
                        "summarize-llama-bench",
                        "--label",
                        "generic",
                        "--artifact",
                        f"pass={source}",
                        "--settings",
                        str(settings),
                        "--output",
                        str(root / "summary.json"),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertIn("duplicate JSON object key", stderr.getvalue())

    def test_compare_llama_bench_writes_compatible_report(self) -> None:
        source = Path(__file__).parent / "fixtures" / "llama_bench.synthetic.jsonl"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            summaries: dict[str, Path] = {}
            for label, kleidiai in (("generic", False), ("optimized", True)):
                settings = root / f"{label}-settings.json"
                settings.write_text(
                    json.dumps({"threads": 4, "build": {"kleidiai": kleidiai}}),
                    encoding="utf-8",
                )
                summary = root / f"{label}.json"
                with patch("sys.stdout", io.StringIO()):
                    exit_code = cli.main(
                        [
                            "summarize-llama-bench",
                            "--label",
                            label,
                            "--artifact",
                            f"pass={source}",
                            "--settings",
                            str(settings),
                            "--output",
                            str(summary),
                        ]
                    )
                self.assertEqual(exit_code, 0)
                summaries[label] = summary

            destination = root / "comparison.json"
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                exit_code = cli.main(
                    [
                        "compare-llama-bench",
                        "--generic",
                        str(summaries["generic"]),
                        "--kleidiai",
                        str(summaries["optimized"]),
                        "--output",
                        str(destination),
                    ]
                )

            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, json.loads(stdout.getvalue()))
            self.assertTrue(payload["compatibility"]["validated"])
            self.assertEqual(payload["variants"]["generic"]["label"], "generic")
            self.assertEqual(payload["variants"]["kleidiai"]["label"], "optimized")
            self.assertEqual(payload["tests"]["pp"]["median_throughput_speedup"], 1.0)
            self.assertEqual(
                len(payload["input_fingerprints"]["generic_summary_sha256"]),
                64,
            )

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
