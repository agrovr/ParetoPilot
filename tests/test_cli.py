from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from paretopilot import cli
from paretopilot.doctor import EnvironmentReport
from paretopilot.domain import ValidationError


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


class CliTests(unittest.TestCase):
    def test_version_is_available_without_a_subcommand(self) -> None:
        output = io.StringIO()
        with patch("sys.stdout", output), self.assertRaises(SystemExit) as raised:
            cli.main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertRegex(output.getvalue(), r"paretopilot \d+\.\d+\.\d+")

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
