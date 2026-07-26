from __future__ import annotations

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from paretopilot import cli
from paretopilot.doctor import EnvironmentReport
from paretopilot.domain import ValidationError
from paretopilot.report_v11 import render_report_v11
from test_report_v11 import (
    canonical_benchmarks,
    canonical_recommendation,
    derived_profiles,
    measured_load_sweep,
    measured_stability,
)
from test_showcase import evidence_lock


def _server_evaluation_payload(latency_ms: float) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "candidate_id": "candidate-a",
        "synthetic": False,
        "suite": {
            "id": "suite-v1",
            "license": "CC0-1.0",
            "sha256": "a" * 64,
            "quality_case_count": 1,
            "performance_repetitions": 1,
            "performance_warmups": 1,
            "generation_tokens": 64,
            "cache_prompt": False,
            "seed": 4242,
            "temperature": 0,
        },
        "quality": {
            "method": "fixed exact-answer smoke evaluation",
            "score": 1.0,
            "passed": 1,
            "total": 1,
            "cases": [
                {
                    "id": "identity",
                    "prompt": "Reply YES.",
                    "accepted_answers": ["YES"],
                    "response": "YES",
                    "matched": True,
                    "matched_answer": "YES",
                }
            ],
        },
        "latency": {
            "method": "single-client streamed HTTP requests",
            "ttft_ms_p50": latency_ms / 2,
            "ttft_ms_p95": latency_ms / 2,
            "e2e_latency_ms_p50": latency_ms,
            "e2e_latency_ms_p95": latency_ms,
            "samples": [
                {
                    "index": 1,
                    "ttft_ms": latency_ms / 2,
                    "e2e_latency_ms": latency_ms,
                    "event_count": 64,
                    "predicted_tokens": 64,
                    "content": "fixed output",
                }
            ],
        },
    }


def _benchmark_payload() -> dict[str, object]:
    benchmarks = canonical_benchmarks()
    return {
        "schema_version": benchmarks.schema_version,
        "baseline_id": benchmarks.baseline_id,
        "synthetic": benchmarks.synthetic,
        "metadata": dict(benchmarks.metadata),
        "candidates": [
            {
                "id": candidate.candidate_id,
                "label": candidate.label,
                "parameters": dict(candidate.parameters),
                "metrics": dict(candidate.metrics),
            }
            for candidate in benchmarks.candidates
        ],
    }


def _write_json_fixture(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_showcase_fixture_bundle(root: Path) -> dict[str, Path]:
    paths = {
        "results": root / "benchmark-set.json",
        "recommendation": root / "recommendation.json",
        "profiles": root / "policy-profiles.json",
        "load": root / "load-evaluation.json",
        "stability": root / "repeat-stability.json",
        "evidence_lock": root / "evidence-lock.json",
        "canonical": root / "report-v1.1.html",
    }
    benchmarks = canonical_benchmarks()
    _write_json_fixture(paths["results"], _benchmark_payload())
    benchmarks_sha256 = cli.sha256_file(paths["results"])

    recommendation = canonical_recommendation(benchmarks)
    recommendation["input_fingerprints"]["benchmarks_sha256"] = benchmarks_sha256
    profiles = derived_profiles(benchmarks)
    profiles["input_fingerprints"]["benchmarks_sha256"] = benchmarks_sha256
    for profile in profiles["profiles"]:
        fingerprints = profile["recommendation"].get("input_fingerprints")
        if isinstance(fingerprints, dict):
            fingerprints["benchmarks_sha256"] = benchmarks_sha256
    fixtures = {
        "recommendation": recommendation,
        "profiles": profiles,
        "load": measured_load_sweep(),
        "stability": measured_stability(benchmarks),
    }
    for name, payload in fixtures.items():
        _write_json_fixture(paths[name], payload)

    loaded_benchmarks = cli.load_benchmarks(paths["results"])
    loaded_recommendation = cli.load_json_object(paths["recommendation"])
    loaded_profiles = cli.load_json_object(paths["profiles"])
    loaded_load = cli.load_json_object(paths["load"])
    loaded_stability = cli.load_json_object(paths["stability"])
    canonical = render_report_v11(
        loaded_benchmarks,
        loaded_recommendation,
        policy_profiles=loaded_profiles,
        load_sweep=loaded_load,
        stability_summary=loaded_stability,
        benchmarks_sha256=cli.sha256_file(paths["results"]),
        recommendation_sha256=cli.sha256_file(paths["recommendation"]),
        profiles_sha256=cli.sha256_file(paths["profiles"]),
        load_sha256=cli.sha256_file(paths["load"]),
        stability_sha256=cli.sha256_file(paths["stability"]),
    )
    paths["canonical"].write_text(canonical, encoding="utf-8", newline="\n")
    _write_json_fixture(
        paths["evidence_lock"],
        evidence_lock(
            artifacts_sha256={
                "benchmark_set": cli.sha256_file(paths["results"]),
                "recommendation": cli.sha256_file(paths["recommendation"]),
                "policy_profiles": cli.sha256_file(paths["profiles"]),
                "load_evaluation": cli.sha256_file(paths["load"]),
                "repeat_stability": cli.sha256_file(paths["stability"]),
                "report_v1_1": cli.sha256_file(paths["canonical"]),
            }
        ),
    )
    return paths


class CliTests(unittest.TestCase):
    def test_version_is_available_without_a_subcommand(self) -> None:
        output = io.StringIO()
        with patch("sys.stdout", output), self.assertRaises(SystemExit) as raised:
            cli.main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertRegex(output.getvalue(), r"paretopilot \d+\.\d+\.\d+")

    def test_public_help_explains_decision_and_published_artifacts(self) -> None:
        expected_phrases = {
            "passport": ("machine-readable decision details", "source-declared Arm64 metadata"),
            "optimization-receipt": (
                "Markdown decision summary",
                "source-declared Arm64 metadata",
            ),
            "verify-published": (
                "published v1.1 model and latency study",
                "separate v1.4 capacity study",
            ),
        }

        for command, phrases in expected_phrases.items():
            with self.subTest(command=command):
                output = io.StringIO()
                with patch("sys.stdout", output), self.assertRaises(SystemExit) as raised:
                    cli.main([command, "--help"])
                self.assertEqual(raised.exception.code, 0)
                normalized = " ".join(output.getvalue().split())
                for phrase in phrases:
                    self.assertIn(phrase, normalized)

    def test_commit_match_accepts_normal_sha_prefixes_only(self) -> None:
        self.assertTrue(cli._commits_match("67b9b0e7", cli.PINNED_LLAMA_CPP_COMMIT))
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
        with (
            patch.object(cli, "inspect_environment", return_value=report),
            patch("sys.stdout", output),
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
        with (
            patch.object(cli, "inspect_environment", return_value=report),
            patch("sys.stdout", io.StringIO()),
        ):
            exit_code = cli.main(["doctor", "--require-evidence-host"])

        self.assertEqual(exit_code, 3)

    def test_ci_gate_generates_a_hashed_receipt_for_an_explicit_smoke_test(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "gate"
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                exit_code = cli.main(
                    [
                        "ci-gate",
                        "examples/synthetic-results.json",
                        "--constraints",
                        "configs/constraints.example.json",
                        "--output-dir",
                        str(output_dir),
                        "--allow-synthetic",
                        "--expect-selected-id",
                        "q4-kleidiai",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            recommendation_path = output_dir / "recommendation.json"
            report_path = output_dir / "report.html"
            passport_path = output_dir / "decision-passport.json"
            optimization_receipt_path = output_dir / "optimization-receipt.md"
            receipt_path = output_dir / "gate.json"
            recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
            passport = json.loads(passport_path.read_text(encoding="utf-8"))
            optimization_receipt = optimization_receipt_path.read_text(encoding="utf-8")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["selected_id"], "q4-kleidiai")
            self.assertTrue(payload["expectation_matched"])
            self.assertTrue(payload["synthetic_source"])
            self.assertEqual(recommendation["selected_id"], "q4-kleidiai")
            self.assertEqual(receipt["selected_id"], "q4-kleidiai")
            self.assertEqual(receipt["schema_version"], "1.2")
            self.assertEqual(receipt["evidence_grade"], "synthetic")
            self.assertEqual(passport["evidence_grade"], "synthetic")
            self.assertIn("Synthetic", optimization_receipt)
            self.assertNotIn("Measured improvement", optimization_receipt)
            self.assertEqual(
                receipt["recommendation_sha256"],
                cli.sha256_file(recommendation_path),
            )
            self.assertEqual(receipt["report_sha256"], cli.sha256_file(report_path))
            self.assertEqual(
                receipt["decision_passport_sha256"],
                cli.sha256_file(passport_path),
            )
            self.assertEqual(
                receipt["optimization_receipt_sha256"],
                cli.sha256_file(optimization_receipt_path),
            )
            self.assertEqual(
                receipt["optimization_receipt"],
                str(optimization_receipt_path),
            )
            self.assertEqual(payload["receipt_sha256"], cli.sha256_file(receipt_path))

    def test_ci_gate_binds_every_artifact_to_one_input_byte_snapshot(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            constraints_path = root / "constraints.json"
            constraints_path.write_bytes(Path("configs/constraints.example.json").read_bytes())
            original_constraints_sha256 = cli.sha256_file(constraints_path)
            output_dir = root / "gate"
            original_loader = cli.load_constraints_snapshot

            def load_then_mutate(path: Path):
                constraints, digest = original_loader(path)
                path.write_text("{not valid JSON", encoding="utf-8")
                return constraints, digest

            stdout = io.StringIO()
            with (
                patch.object(
                    cli,
                    "load_constraints_snapshot",
                    side_effect=load_then_mutate,
                ) as snapshot_loader,
                patch("sys.stdout", stdout),
            ):
                exit_code = cli.main(
                    [
                        "ci-gate",
                        "examples/synthetic-results.json",
                        "--constraints",
                        str(constraints_path),
                        "--output-dir",
                        str(output_dir),
                        "--allow-synthetic",
                        "--expect-selected-id",
                        "q4-kleidiai",
                    ]
                )

            recommendation = json.loads(
                (output_dir / "recommendation.json").read_text(encoding="utf-8")
            )
            passport = json.loads(
                (output_dir / "decision-passport.json").read_text(encoding="utf-8")
            )
            optimization_receipt = (output_dir / "optimization-receipt.md").read_text(
                encoding="utf-8"
            )
            self.assertEqual(exit_code, 0)
            snapshot_loader.assert_called_once_with(constraints_path)
            self.assertEqual(
                recommendation["input_fingerprints"],
                passport["input_fingerprints"],
            )
            self.assertEqual(
                recommendation["input_fingerprints"]["constraints_sha256"],
                original_constraints_sha256,
            )
            self.assertIn(original_constraints_sha256, optimization_receipt)

    def test_optimization_receipt_cli_exports_a_deterministic_human_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "optimization-receipt.md"
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                exit_code = cli.main(
                    [
                        "optimization-receipt",
                        "examples/synthetic-results.json",
                        "--constraints",
                        "configs/constraints.example.json",
                        "--output",
                        str(output_path),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            receipt = output_path.read_text(encoding="utf-8")
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["selected_id"], "q4-kleidiai")
            self.assertEqual(payload["evidence_grade"], "synthetic")
            self.assertEqual(payload["optimization_receipt"], str(output_path))
            self.assertEqual(
                payload["optimization_receipt_sha256"],
                cli.sha256_file(output_path),
            )
            self.assertIn("# ParetoPilot decision summary", receipt)
            self.assertIn("Synthetic", receipt)
            self.assertTrue(receipt.endswith("\n"))

            with patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
                overwrite_exit = cli.main(
                    [
                        "optimization-receipt",
                        "examples/synthetic-results.json",
                        "--constraints",
                        "configs/constraints.example.json",
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertEqual(overwrite_exit, 2)

    def test_passport_cli_exports_strict_arm64_attribution_for_canonical_fixture(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            benchmarks_path = root / "benchmark-set.json"
            passport_path = root / "decision-passport.json"
            benchmark_payload = _benchmark_payload()
            benchmark_payload["metadata"] = {
                "classification": "canonical",
                "source": {
                    "repository": "agrovr/ParetoPilot",
                    "revision": "8a9ddce0afa2272c4a4097fe87ef6f06cb7689a9",
                    "workflow": ".github/workflows/candidate-study-arm64.yml",
                    "run_id": "30055662526",
                    "run_attempt": 1,
                    "runner": {
                        "architecture": "arm64",
                        "cpu": "Neoverse-N2",
                        "cpu_count": 4,
                        "os": "Ubuntu 24.04",
                    },
                },
                "runtime": {
                    "name": "llama.cpp",
                    "repository": "https://github.com/ggml-org/llama.cpp",
                    "revision": "67b9b0e7f6ce45d929a4411907d3c48ec719e81c",
                },
                "model_family": {
                    "name": "Qwen2.5-1.5B-Instruct",
                    "repository": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                    "revision": "91cad51170dc346986eccefdc2dd33a9da36ead9",
                },
                "evaluation_suite": {
                    "id": "paretopilot-qwen-behavior-v2",
                    "sha256": ("e49c16fba32fd65c947264aef4141026ab68b1fd415ef09eeea6e8ade9a545c7"),
                },
            }
            _write_json_fixture(benchmarks_path, benchmark_payload)
            stdout = io.StringIO()

            with patch("sys.stdout", stdout):
                exit_code = cli.main(
                    [
                        "passport",
                        str(benchmarks_path),
                        "--constraints",
                        "configs/constraints.candidate-study.json",
                        "--output",
                        str(passport_path),
                        "--require-arm64-provenance",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            passport = json.loads(passport_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["evidence_grade"], "arm64-attributed")
            self.assertEqual(payload["selected_id"], "q8-generic")
            self.assertEqual(passport["evidence_grade"], "arm64-attributed")
            self.assertEqual(
                [stage["candidate_id"] for stage in passport["ladder"]],
                [
                    "q8-generic",
                    "q4-generic",
                    "q4-kleidiai",
                    "q4-kleidiai-tuned",
                ],
            )
            self.assertEqual(
                passport["input_fingerprints"]["benchmarks_sha256"],
                cli.sha256_file(benchmarks_path),
            )
            self.assertEqual(payload["passport_sha256"], cli.sha256_file(passport_path))

    def test_passport_arm64_provenance_gate_rejects_synthetic_without_output(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist.json"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                exit_code = cli.main(
                    [
                        "passport",
                        "examples/synthetic-results.json",
                        "--constraints",
                        "configs/constraints.example.json",
                        "--output",
                        str(output),
                        "--require-arm64-provenance",
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn(
                "complete source-declared Arm64 attribution metadata is required",
                stderr.getvalue(),
            )
            self.assertFalse(output.exists())

    def test_ci_gate_rejects_synthetic_evidence_by_default_without_outputs(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "must-not-exist"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                exit_code = cli.main(
                    [
                        "ci-gate",
                        "examples/synthetic-results.json",
                        "--constraints",
                        "configs/constraints.example.json",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("requires measured evidence", stderr.getvalue())
            self.assertFalse(output_dir.exists())

    def test_ci_gate_can_require_fully_attributed_arm64_evidence(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "must-not-exist"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                exit_code = cli.main(
                    [
                        "ci-gate",
                        "examples/synthetic-results.json",
                        "--constraints",
                        "configs/constraints.example.json",
                        "--output-dir",
                        str(output_dir),
                        "--allow-synthetic",
                        "--require-arm64-provenance",
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn(
                "complete source-declared Arm64 attribution metadata is required",
                stderr.getvalue(),
            )
            self.assertFalse(output_dir.exists())

    def test_ci_gate_rejects_an_unexpected_selection_without_outputs(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "must-not-exist"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                exit_code = cli.main(
                    [
                        "ci-gate",
                        "examples/synthetic-results.json",
                        "--constraints",
                        "configs/constraints.example.json",
                        "--output-dir",
                        str(output_dir),
                        "--allow-synthetic",
                        "--expect-selected-id",
                        "baseline-q8",
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("does not match the CI expectation", stderr.getvalue())
            self.assertFalse(output_dir.exists())

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
            exit_code = cli.main(["validate-llama-bench", str(fixture), "--evidence"])

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 4)
        self.assertFalse(payload["evidence_valid"])
        self.assertTrue(any("synthetic" in issue for issue in payload["evidence_issues"]))

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
                exit_code = cli.main(["validate-llama-bench", str(copied), "--output", str(copied)])

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

    def test_parse_peak_rss_writes_strict_resource_value(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "time.txt"
            source.write_text(
                "\tMaximum resident set size (kbytes): 2097152\n",
                encoding="utf-8",
            )
            output_path = root / "rss.json"
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                exit_code = cli.main(
                    [
                        "parse-peak-rss",
                        str(source),
                        "--candidate-id",
                        "candidate-a",
                        "--output",
                        str(output_path),
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["peak_rss_mib"], 2048.0)
            self.assertEqual(payload, json.loads(output_path.read_text(encoding="utf-8")))

    def test_bind_capacity_quality_cli_writes_source_bound_wrapper(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "quality.json"
            command = root / "server-command.json"
            destination = root / "bound-quality.json"
            payload = {
                "schema_version": "1.0",
                "classification": "capacity-quality-binding",
            }
            stdout = io.StringIO()

            with (
                patch.object(
                    cli,
                    "bind_capacity_quality",
                    return_value=payload,
                ) as bind,
                patch("sys.stdout", stdout),
            ):
                exit_code = cli.main(
                    [
                        "bind-capacity-quality",
                        "--evaluation",
                        str(evaluation),
                        "--server-command",
                        str(command),
                        "--base-url",
                        "http://127.0.0.1:19404",
                        "--pass-id",
                        "quality",
                        "--candidate-id",
                        "q8-generic",
                        "--server-parallel",
                        "4",
                        "--run-id",
                        "123456",
                        "--run-attempt",
                        "2",
                        "--output",
                        str(destination),
                    ]
                )

            self.assertEqual(exit_code, 0)
            bind.assert_called_once_with(
                evaluation_path=evaluation,
                server_command_path=command,
                base_url="http://127.0.0.1:19404",
                pass_id="quality",
                candidate_id="q8-generic",
                server_parallel=4,
                run_id="123456",
                run_attempt=2,
            )
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), payload)
            self.assertEqual(json.loads(stdout.getvalue()), payload)

    def test_pool_server_evaluations_cli_writes_once_and_preserves_inputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "pass-a.json"
            second = root / "pass-b.json"
            first.write_text(json.dumps(_server_evaluation_payload(100.0)), encoding="utf-8")
            second.write_text(json.dumps(_server_evaluation_payload(200.0)), encoding="utf-8")
            originals = (first.read_bytes(), second.read_bytes())
            destination = root / "pooled.json"
            stdout = io.StringIO()

            with patch("sys.stdout", stdout):
                exit_code = cli.main(
                    [
                        "pool-server-evaluations",
                        "--input",
                        str(first),
                        "--input",
                        str(second),
                        "--output",
                        str(destination),
                    ]
                )

            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, json.loads(stdout.getvalue()))
            self.assertEqual(payload["suite"]["performance_repetitions"], 2)
            self.assertEqual(payload["suite"]["performance_warmups"], 2)
            self.assertEqual(payload["latency"]["e2e_latency_ms_p50"], 150.0)
            self.assertEqual((first.read_bytes(), second.read_bytes()), originals)

            stderr = io.StringIO()
            with patch("sys.stdout", io.StringIO()), patch("sys.stderr", stderr):
                overwrite_exit = cli.main(
                    [
                        "pool-server-evaluations",
                        "--input",
                        str(first),
                        "--input",
                        str(second),
                        "--output",
                        str(destination),
                    ]
                )
            self.assertEqual(overwrite_exit, 2)
            self.assertIn("refusing to overwrite", stderr.getvalue())

    def test_evaluate_load_cli_uses_strict_plan_and_live_runner(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text('{"schema_version":"1.0"}', encoding="utf-8")
            server_command = root / "server-command.json"
            canonical_server_command = root / "canonical-server-command.json"
            server_command.write_text('{"schema_version":"1.0"}', encoding="utf-8")
            canonical_server_command.write_text('{"schema_version":"1.0"}', encoding="utf-8")
            output_path = root / "load.json"
            artifact = {
                "schema_version": "1.0",
                "candidate_id": "candidate-a",
                "rows": [],
            }
            sentinel_plan = object()
            sentinel_binding = {"plan_sha256": "a" * 64}
            stdout = io.StringIO()
            with (
                patch.object(cli, "load_load_plan", return_value=sentinel_plan) as load_plan,
                patch.object(
                    cli,
                    "build_load_evidence_binding",
                    return_value=sentinel_binding,
                ) as build_binding,
                patch.object(
                    cli,
                    "evaluate_llama_server_load",
                    return_value=artifact,
                ) as evaluate,
                patch("sys.stdout", stdout),
            ):
                exit_code = cli.main(
                    [
                        "evaluate-load",
                        "--base-url",
                        "http://127.0.0.1:8080",
                        "--candidate-id",
                        "candidate-a",
                        "--plan",
                        str(plan_path),
                        "--server-command",
                        str(server_command),
                        "--canonical-server-command",
                        str(canonical_server_command),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            load_plan.assert_called_once_with(plan_path)
            build_binding.assert_called_once_with(
                base_url="http://127.0.0.1:8080",
                plan_path=plan_path,
                server_command_path=server_command,
                canonical_server_command_path=canonical_server_command,
            )
            evaluate.assert_called_once_with(
                "http://127.0.0.1:8080",
                sentinel_plan,
                candidate_id="candidate-a",
                evidence_binding=sentinel_binding,
                execution_order=None,
            )
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                artifact,
            )
            self.assertEqual(json.loads(stdout.getvalue()), artifact)

    def test_combine_load_cli_reads_inputs_and_writes_combined_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.json"
            second = root / "b.json"
            first.write_text('{"candidate_id":"a"}', encoding="utf-8")
            second.write_text('{"candidate_id":"b"}', encoding="utf-8")
            output_path = root / "combined.json"
            combined = {
                "schema_version": "1.0",
                "rows": [{"candidate_id": "a"}, {"candidate_id": "b"}],
            }
            with (
                patch.object(
                    cli,
                    "combine_load_evaluations",
                    return_value=combined,
                ) as combine,
                patch("sys.stdout", io.StringIO()),
            ):
                exit_code = cli.main(
                    [
                        "combine-load",
                        "--input",
                        str(first),
                        "--input",
                        str(second),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            combine.assert_called_once_with(
                [{"candidate_id": "a"}, {"candidate_id": "b"}],
                require_evidence_bindings=True,
            )
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                combined,
            )

    def test_capacity_cli_preserves_every_labeled_source_contract(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "capacity.json"
            load_plan = root / "load.json"
            manifest = root / "manifest.json"
            for path in (plan, load_plan, manifest):
                path.write_text("{}\n", encoding="utf-8")
            output = root / "capacity-study.json"
            artifact = {
                "schema_version": "1.1",
                "classification": "supplementary-capacity",
            }
            stdout = io.StringIO()
            with (
                patch.object(
                    cli,
                    "assemble_capacity_study",
                    return_value=artifact,
                ) as assemble,
                patch("sys.stdout", stdout),
            ):
                exit_code = cli.main(
                    [
                        "assemble-capacity",
                        "--plan",
                        str(plan),
                        "--load-plan",
                        str(load_plan),
                        "--manifest",
                        str(manifest),
                        "--load",
                        "forward/q8-generic/p1=load-artifact.json",
                        "--rss",
                        "forward/q8-generic/p1=server-time.txt",
                        "--server-log",
                        "forward/q8-generic/p1=server.stderr.log",
                        "--quality",
                        "q8-generic/p1=quality.json",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            assemble.assert_called_once_with(
                plan_path=plan,
                load_plan_path=load_plan,
                manifest_path=manifest,
                load_artifacts=[("forward/q8-generic/p1", Path("load-artifact.json"))],
                rss_artifacts=[("forward/q8-generic/p1", Path("server-time.txt"))],
                server_logs=[("forward/q8-generic/p1", Path("server.stderr.log"))],
                quality_artifacts=[("q8-generic/p1", Path("quality.json"))],
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), artifact)
            self.assertEqual(json.loads(stdout.getvalue()), artifact)

    def test_capacity_receipt_cli_writes_once(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "capacity-study.json"
            study.write_text(
                '{"classification":"supplementary-capacity"}\n',
                encoding="utf-8",
            )
            output = root / "capacity-receipt.md"
            stdout = io.StringIO()
            with (
                patch.object(
                    cli,
                    "render_capacity_receipt",
                    return_value="# Capacity receipt\n",
                ) as render,
                patch("sys.stdout", stdout),
            ):
                exit_code = cli.main(
                    [
                        "capacity-receipt",
                        str(study),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            render.assert_called_once_with({"classification": "supplementary-capacity"})
            self.assertEqual(output.read_text(encoding="utf-8"), "# Capacity receipt\n")
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["classification"], "supplementary-capacity")
            self.assertEqual(payload["capacity_receipt"], str(output))
            self.assertEqual(len(payload["capacity_receipt_sha256"]), 64)

    def test_capacity_replay_cli_reports_the_verified_bundle(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            output = root / "replayed"
            bundle.mkdir()
            replay = {
                "schema_version": "1.0",
                "classification": "supplementary-capacity-replay",
                "valid": True,
                "status_complete": True,
                "capacity_study_reproduced": True,
                "capacity_receipt_reproduced": True,
                "checksums": {
                    "entry_count": 121,
                    "manifest_sha256": "a" * 64,
                },
                "canonical_replay": {
                    "verified": True,
                    "selected_id": "q8-generic",
                },
                "selected_operating_points": {
                    "q8-generic": {
                        "server_parallel": 4,
                        "client_concurrency": 4,
                    }
                },
                "verdict": "PASS",
            }
            stdout = io.StringIO()
            with (
                patch.object(
                    cli,
                    "replay_capacity_bundle",
                    return_value=replay,
                ) as replay_bundle,
                patch("sys.stdout", stdout),
            ):
                exit_code = cli.main(
                    [
                        "replay-capacity",
                        str(bundle),
                        "--output-dir",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            replay_bundle.assert_called_once_with(bundle, output)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["valid"])
            self.assertTrue(payload["capacity_study_reproduced"])
            self.assertTrue(payload["capacity_receipt_reproduced"])
            self.assertEqual(payload["checksum_entries"], 121)
            self.assertEqual(payload["output_directory"], str(output.resolve()))
            self.assertEqual(payload["details"], str(output.resolve() / "capacity-replay.json"))

    def test_stability_cli_parses_labeled_inputs_and_metric_directions(self) -> None:
        repository = Path(__file__).parents[1]
        source = repository / "examples" / "synthetic-results.json"
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "stability.json"
            summary = {
                "schema_version": "1.0",
                "baseline_id": "baseline",
                "rows": [],
            }
            stdout = io.StringIO()
            with (
                patch.object(
                    cli,
                    "summarize_stability",
                    return_value=summary,
                ) as summarize,
                patch("sys.stdout", stdout),
            ):
                exit_code = cli.main(
                    [
                        "summarize-stability",
                        "--input",
                        f"A={source}",
                        "--input",
                        f"B={source}",
                        "--metric",
                        "latency_ms=min",
                        "--metric",
                        "throughput=max",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(summarize.call_args.kwargs["pass_labels"], ["A", "B"])
            self.assertEqual(
                summarize.call_args.kwargs["metric_directions"],
                {"latency_ms": "min", "throughput": "max"},
            )
            self.assertEqual(json.loads(stdout.getvalue()), summary)

        with self.assertRaisesRegex(ValidationError, "must be unique"):
            cli._parse_metric_directions(["latency=min", "latency=max"])

    def test_report_v11_cli_loads_extensions_and_writes_once(self) -> None:
        repository = Path(__file__).parents[1]
        results = repository / "examples" / "synthetic-results.json"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            recommendation = root / "recommendation.json"
            profiles = root / "profiles.json"
            load = root / "load.json"
            stability = root / "stability.json"
            output = root / "report-v1.1.html"
            recommendation.write_text(
                json.dumps({"selected_id": "balanced"}),
                encoding="utf-8",
            )
            profiles.write_text(json.dumps({"profiles": []}), encoding="utf-8")
            load.write_text(json.dumps({"rows": []}), encoding="utf-8")
            stability.write_text(json.dumps({"rows": []}), encoding="utf-8")
            stdout = io.StringIO()

            with (
                patch.object(
                    cli,
                    "render_report_v11",
                    return_value="<!doctype html><title>ParetoPilot v1.1</title>",
                ) as render,
                patch("sys.stdout", stdout),
            ):
                exit_code = cli.main(
                    [
                        "report-v11",
                        str(results),
                        "--recommendation",
                        str(recommendation),
                        "--profiles",
                        str(profiles),
                        "--load",
                        str(load),
                        "--stability",
                        str(stability),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("ParetoPilot v1.1", output.read_text(encoding="utf-8"))
            kwargs = render.call_args.kwargs
            self.assertEqual(kwargs["policy_profiles"], {"profiles": []})
            self.assertEqual(kwargs["load_sweep"], {"rows": []})
            self.assertEqual(kwargs["stability_summary"], {"rows": []})
            self.assertEqual(len(kwargs["benchmarks_sha256"]), 64)
            self.assertEqual(len(kwargs["recommendation_sha256"]), 64)
            self.assertEqual(len(kwargs["profiles_sha256"]), 64)
            self.assertEqual(len(kwargs["load_sha256"]), 64)
            self.assertEqual(len(kwargs["stability_sha256"]), 64)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["valid"])
            self.assertTrue(payload["policy_profiles_supplied"])
            self.assertTrue(payload["load_sweep_supplied"])
            self.assertTrue(payload["stability_summary_supplied"])

            stderr = io.StringIO()
            with patch("sys.stdout", io.StringIO()), patch("sys.stderr", stderr):
                overwrite_exit = cli.main(
                    [
                        "report-v11",
                        str(results),
                        "--recommendation",
                        str(recommendation),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(overwrite_exit, 2)
            self.assertIn("refusing to overwrite", stderr.getvalue())

    def test_showcase_v11_cli_renders_locked_fixtures_deterministically(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _write_showcase_fixture_bundle(root)
            canonical_before = paths["canonical"].read_bytes()
            base_args = [
                "showcase-v11",
                str(paths["results"]),
                "--recommendation",
                str(paths["recommendation"]),
                "--profiles",
                str(paths["profiles"]),
                "--load",
                str(paths["load"]),
                "--stability",
                str(paths["stability"]),
            ]
            locked_outputs = [root / "showcase-a.html", root / "showcase-b.html"]
            locked_payloads: list[dict[str, object]] = []

            for output in locked_outputs:
                stdout = io.StringIO()
                with patch("sys.stdout", stdout):
                    exit_code = cli.main(
                        [
                            *base_args,
                            "--evidence-lock",
                            str(paths["evidence_lock"]),
                            "--canonical-report",
                            str(paths["canonical"]),
                            "--canonical-report-href",
                            "proof/report-v1.1.html",
                            "--output",
                            str(output),
                        ]
                    )
                self.assertEqual(exit_code, 0)
                locked_payloads.append(json.loads(stdout.getvalue()))

            first_html = locked_outputs[0].read_text(encoding="utf-8")
            self.assertEqual(locked_outputs[0].read_bytes(), locked_outputs[1].read_bytes())
            self.assertEqual(
                locked_payloads[0]["report_sha256"],
                locked_payloads[1]["report_sha256"],
            )
            for payload, output in zip(locked_payloads, locked_outputs, strict=True):
                self.assertTrue(payload["valid"])
                self.assertTrue(payload["presentation_view"])
                self.assertTrue(payload["canonical_report_verified"])
                self.assertTrue(payload["evidence_lock_supplied"])
                self.assertFalse(payload["capacity_study_supplied"])
                self.assertFalse(payload["capacity_evidence_lock_supplied"])
                self.assertEqual(payload["selected_id"], "q8-generic")
                self.assertEqual(payload["baseline_id"], "q8-generic")
                self.assertEqual(payload["report"], str(output))

            self.assertIn('class="showcase is-verified"', first_html)
            self.assertIn("<title>ParetoPilot | Arm64 inference results</title>", first_html)
            self.assertIn('href="proof/report-v1.1.html"', first_html)
            self.assertIn("150 files verified", first_html)
            self.assertIn("Published and reproduced", first_html)
            self.assertIn("View archived v1.1 report", first_html)
            self.assertNotEqual(first_html.encode(), canonical_before)
            self.assertNotIn(b'class="showcase', canonical_before)
            self.assertIn(
                b"<title>ParetoPilot v1.1 deployment decision report</title>",
                canonical_before,
            )
            self.assertEqual(paths["canonical"].read_bytes(), canonical_before)

            unlocked_output = root / "showcase-without-lock.html"
            unlocked_stdout = io.StringIO()
            with patch("sys.stdout", unlocked_stdout):
                unlocked_exit = cli.main(
                    [
                        *base_args,
                        "--output",
                        str(unlocked_output),
                    ]
                )
            unlocked_payload = json.loads(unlocked_stdout.getvalue())
            self.assertEqual(unlocked_exit, 0)
            self.assertTrue(unlocked_payload["presentation_view"])
            self.assertFalse(unlocked_payload["canonical_report_verified"])
            self.assertFalse(unlocked_payload["evidence_lock_supplied"])
            self.assertFalse(unlocked_payload["capacity_study_supplied"])
            self.assertFalse(unlocked_payload["capacity_evidence_lock_supplied"])
            unlocked_html = unlocked_output.read_text(encoding="utf-8")
            self.assertIn('class="showcase is-preview"', unlocked_html)
            self.assertIn("Unverified preview", unlocked_html)
            self.assertNotIn("View archived v1.1 report", unlocked_html)

    def test_showcase_v11_cli_forwards_locked_capacity_evidence(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _write_showcase_fixture_bundle(root)
            capacity_study = root / "capacity-study.json"
            capacity_lock = root / "capacity-evidence.json"
            _write_json_fixture(
                capacity_study,
                {"classification": "supplementary-capacity"},
            )
            _write_json_fixture(
                capacity_lock,
                {
                    "schema_version": "1.4",
                    "classification": "supplementary-capacity",
                },
            )
            output = root / "showcase-capacity.html"
            stdout = io.StringIO()
            with (
                patch.object(
                    cli,
                    "render_showcase_v11",
                    return_value="<!doctype html><title>Capacity</title>",
                ) as render,
                patch("sys.stdout", stdout),
            ):
                exit_code = cli.main(
                    [
                        "showcase-v11",
                        str(paths["results"]),
                        "--recommendation",
                        str(paths["recommendation"]),
                        "--evidence-lock",
                        str(paths["evidence_lock"]),
                        "--canonical-report",
                        str(paths["canonical"]),
                        "--capacity-study",
                        str(capacity_study),
                        "--capacity-evidence-lock",
                        str(capacity_lock),
                        "--capacity-study-href",
                        "proof/capacity-study.json",
                        "--capacity-receipt-href",
                        "proof/capacity-receipt.md",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            kwargs = render.call_args.kwargs
            self.assertEqual(
                kwargs["capacity_study"],
                {"classification": "supplementary-capacity"},
            )
            self.assertEqual(
                kwargs["capacity_evidence_lock"],
                {
                    "schema_version": "1.4",
                    "classification": "supplementary-capacity",
                },
            )
            self.assertEqual(kwargs["capacity_study_sha256"], cli.sha256_file(capacity_study))
            self.assertEqual(
                kwargs["evidence_lock_sha256"], cli.sha256_file(paths["evidence_lock"])
            )
            self.assertEqual(kwargs["capacity_study_href"], "proof/capacity-study.json")
            self.assertEqual(
                kwargs["capacity_receipt_href"],
                "proof/capacity-receipt.md",
            )
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["capacity_study_supplied"])
            self.assertTrue(payload["capacity_evidence_lock_supplied"])

    def test_showcase_v11_cli_rejects_canonical_report_drift_without_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _write_showcase_fixture_bundle(root)
            paths["canonical"].write_text(
                paths["canonical"].read_text(encoding="utf-8") + "\n<!-- drift -->\n",
                encoding="utf-8",
            )
            output = root / "must-not-exist.html"
            stdout = io.StringIO()
            stderr = io.StringIO()

            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                exit_code = cli.main(
                    [
                        "showcase-v11",
                        str(paths["results"]),
                        "--recommendation",
                        str(paths["recommendation"]),
                        "--profiles",
                        str(paths["profiles"]),
                        "--load",
                        str(paths["load"]),
                        "--stability",
                        str(paths["stability"]),
                        "--evidence-lock",
                        str(paths["evidence_lock"]),
                        "--canonical-report",
                        str(paths["canonical"]),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("canonical_html does not match", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_showcase_v11_cli_rejects_line_ending_byte_drift(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _write_showcase_fixture_bundle(root)
            original = paths["canonical"].read_bytes()
            self.assertIn(b"\n", original)
            paths["canonical"].write_bytes(original.replace(b"\n", b"\r\n"))
            output = root / "must-not-exist.html"
            stderr = io.StringIO()

            with patch("sys.stdout", io.StringIO()), patch("sys.stderr", stderr):
                exit_code = cli.main(
                    [
                        "showcase-v11",
                        str(paths["results"]),
                        "--recommendation",
                        str(paths["recommendation"]),
                        "--profiles",
                        str(paths["profiles"]),
                        "--load",
                        str(paths["load"]),
                        "--stability",
                        str(paths["stability"]),
                        "--evidence-lock",
                        str(paths["evidence_lock"]),
                        "--canonical-report",
                        str(paths["canonical"]),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertIn("canonical_html does not match", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_showcase_v11_cli_rejects_missing_canonical_report_without_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _write_showcase_fixture_bundle(root)
            output = root / "must-not-exist.html"
            stdout = io.StringIO()
            stderr = io.StringIO()

            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                exit_code = cli.main(
                    [
                        "showcase-v11",
                        str(paths["results"]),
                        "--recommendation",
                        str(paths["recommendation"]),
                        "--evidence-lock",
                        str(paths["evidence_lock"]),
                        "--canonical-report",
                        str(root / "missing-report.html"),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("could not read canonical report", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_showcase_v11_cli_rejects_invalid_utf8_canonical_report_without_output(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _write_showcase_fixture_bundle(root)
            paths["canonical"].write_bytes(b"\xff\xfe\xfa")
            output = root / "must-not-exist.html"
            stdout = io.StringIO()
            stderr = io.StringIO()

            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                exit_code = cli.main(
                    [
                        "showcase-v11",
                        str(paths["results"]),
                        "--recommendation",
                        str(paths["recommendation"]),
                        "--evidence-lock",
                        str(paths["evidence_lock"]),
                        "--canonical-report",
                        str(paths["canonical"]),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("canonical report must be valid UTF-8", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_paired_fixture_to_report_is_one_verified_workflow(self) -> None:
        repository = Path(__file__).parents[1]
        bundle = repository / "tests" / "fixtures" / "paired-study"
        with TemporaryDirectory() as directory:
            output_root = Path(directory)
            benchmarks = output_root / "benchmarks.json"
            constraints = output_root / "constraints.json"
            assembly = output_root / "assembly.json"
            with patch("sys.stdout", io.StringIO()):
                assemble_exit = cli.main(
                    [
                        "assemble-study",
                        str(bundle),
                        "--benchmarks-output",
                        str(benchmarks),
                        "--constraints-output",
                        str(constraints),
                        "--assembly-output",
                        str(assembly),
                    ]
                )

            report = output_root / "report.html"
            recommendation = output_root / "recommendation.json"
            report_stdout = io.StringIO()
            with patch("sys.stdout", report_stdout):
                report_exit = cli.main(
                    [
                        "report",
                        str(benchmarks),
                        "--constraints",
                        str(constraints),
                        "--output",
                        str(report),
                        "--recommendation-output",
                        str(recommendation),
                    ]
                )

            self.assertEqual(assemble_exit, 0)
            self.assertEqual(report_exit, 0)
            self.assertEqual(
                json.loads(recommendation.read_text(encoding="utf-8"))["selected_id"],
                "generic-baseline",
            )
            html = report.read_text(encoding="utf-8")
            self.assertIn("Baseline retained", html)
            self.assertIn("inconclusive", html.lower())
            self.assertFalse(json.loads(report_stdout.getvalue())["synthetic_source"])

    def test_verify_study_does_not_write_files(self) -> None:
        repository = Path(__file__).parents[1]
        bundle = repository / "tests" / "fixtures" / "paired-study"
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            exit_code = cli.main(["verify-study", str(bundle)])
        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["selected_id"], "generic-baseline")


if __name__ == "__main__":
    unittest.main()
