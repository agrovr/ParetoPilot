from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paretopilot import cli
from paretopilot.domain import BenchmarkSet, ValidationError
from paretopilot.pass_eval import assemble_repeat_pass


SUITE_ID = "paretopilot-test-suite-v2"
RUNTIME_COMMIT = "a" * 40


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluation_suite() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "id": SUITE_ID,
        "license": "CC0-1.0",
        "quality_cases": [
            {
                "id": "identity",
                "prompt": "Reply with exactly YES.",
                "accepted_answers": ["YES"],
                "match_mode": "trimmed-exact",
            }
        ],
        "performance": {
            "prompt": "Write a short deterministic test response.",
            "generation_tokens": 64,
            "repetitions": 5,
            "warmups": 1,
        },
    }


def _server_evaluation(
    candidate_id: str,
    *,
    suite_sha256: str,
    latency_offset: float,
) -> dict[str, object]:
    ttft_values = [20.0 + latency_offset + index for index in range(5)]
    e2e_values = [200.0 + latency_offset + (index * 10) for index in range(5)]
    return {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "synthetic": False,
        "suite": {
            "id": SUITE_ID,
            "license": "CC0-1.0",
            "sha256": suite_sha256,
            "quality_case_count": 1,
            "performance_repetitions": 5,
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
                    "prompt": "Reply with exactly YES.",
                    "accepted_answers": ["YES"],
                    "match_mode": "trimmed-exact",
                    "response": "YES",
                    "matched": True,
                    "matched_answer": "YES",
                }
            ],
        },
        "latency": {
            "method": "single-client streamed HTTP requests",
            "ttft_ms_p50": ttft_values[2],
            "ttft_ms_p95": ttft_values[4],
            "e2e_latency_ms_p50": e2e_values[2],
            "e2e_latency_ms_p95": e2e_values[4],
            "samples": [
                {
                    "index": index + 1,
                    "ttft_ms": ttft_values[index],
                    "e2e_latency_ms": e2e_values[index],
                    "event_count": 64,
                    "predicted_tokens": 64,
                    "content": f"measured completion {index + 1}",
                }
                for index in range(5)
            ],
        },
    }


def _llama_rows(
    filename: str,
    *,
    prompt_tps: float,
    generation_tps: float,
) -> list[dict[str, object]]:
    def row(n_prompt: int, n_gen: int, tokens_per_second: float) -> dict[str, object]:
        samples = [
            tokens_per_second - 2.0,
            tokens_per_second - 1.0,
            tokens_per_second,
            tokens_per_second + 1.0,
            tokens_per_second + 2.0,
        ]
        durations = [1_000_000_000.0] * 5
        return {
            "build_commit": RUNTIME_COMMIT,
            "model_filename": filename,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "avg_ns": 1_000_000_000.0,
            "avg_ts": tokens_per_second,
            "samples_ns": durations,
            "samples_ts": samples,
            "n_threads": 4,
            "n_batch": 512,
            "n_ubatch": 128,
            "n_gpu_layers": 0,
            "devices": "none",
            "no_op_offload": 1,
        }

    return [
        row(512, 0, prompt_tps),
        row(0, 128, generation_tps),
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    _write_text(
        path,
        "".join(json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in rows),
    )


def _artifact_ref(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
    }


def _build_repeat_experiment(root: Path) -> dict[str, object]:
    suite_path = root / "evaluation-suite.json"
    _write_json(suite_path, _evaluation_suite())
    suite_sha256 = _sha256(suite_path)

    candidates: list[dict[str, object]] = []
    evidence: dict[str, object] = {}
    specifications = (
        ("baseline", "Baseline", "baseline.gguf", 100.0, 25.0, 0.0, 2_048_000),
        ("tuned", "Tuned", "tuned.gguf", 140.0, 35.0, -3.0, 1_536_000),
    )
    for (
        candidate_id,
        label,
        filename,
        prompt_tps,
        generation_tps,
        latency_offset,
        peak_rss_kib,
    ) in specifications:
        candidate_root = root / "candidates" / candidate_id / "raw"
        settings_path = candidate_root / "throughput-settings.json"
        throughput_path = candidate_root / "throughput-pass-1.jsonl"
        server_path = candidate_root / "server-evaluation-pass-1.json"
        server_time_path = candidate_root / "server-time-pass-1.txt"

        _write_json(
            settings_path,
            {
                "threads": 4,
                "batch_size": 512,
                "ubatch_size": 128,
                "prompt_tokens": 512,
                "generation_tokens": 128,
                "repetitions_per_pass": 5,
                "cpu_only": True,
                "n_gpu_layers": 0,
                "devices": "none",
                "no_op_offload": 1,
            },
        )
        _write_jsonl(
            throughput_path,
            _llama_rows(
                filename,
                prompt_tps=prompt_tps,
                generation_tps=generation_tps,
            ),
        )
        _write_json(
            server_path,
            _server_evaluation(
                candidate_id,
                suite_sha256=suite_sha256,
                latency_offset=latency_offset,
            ),
        )
        _write_text(
            server_time_path,
            "Command being timed: llama-server\n"
            f"Maximum resident set size (kbytes): {peak_rss_kib}\n"
            "Exit status: 0\n",
        )

        candidates.append(
            {
                "id": candidate_id,
                "label": label,
                "parameters": {
                    "threads": 4,
                    "batch_size": 512,
                    "ubatch_size": 128,
                },
                "metrics": {
                    "prompt_tps": prompt_tps,
                    "generation_tps": generation_tps,
                    "quality_score": 1.0,
                    "ttft_ms_p95": 24.0 + latency_offset,
                    "e2e_latency_ms_p95": 240.0 + latency_offset,
                    "peak_rss_mib": peak_rss_kib / 1024.0,
                    "model_size_mib": 900.0 if candidate_id == "baseline" else 500.0,
                },
            }
        )
        evidence[candidate_id] = {
            "artifacts": {
                "throughput_settings": _artifact_ref(root, settings_path),
                "throughput_pass_1": _artifact_ref(root, throughput_path),
                "server_evaluation_pass_1": _artifact_ref(root, server_path),
                "server_time_pass_1": _artifact_ref(root, server_time_path),
            }
        }

    benchmark: dict[str, object] = {
        "schema_version": "1.0",
        "baseline_id": "baseline",
        "synthetic": False,
        "metadata": {
            "classification": "canonical",
            "evaluation_suite": {
                "id": SUITE_ID,
                "sha256": suite_sha256,
            },
            "candidate_evidence": evidence,
        },
        "candidates": candidates,
    }
    _write_json(root / "benchmark-set.json", benchmark)
    return benchmark


def _load_benchmark(root: Path) -> dict[str, object]:
    return json.loads((root / "benchmark-set.json").read_text(encoding="utf-8"))


def _rewrite_benchmark(root: Path, benchmark: dict[str, object]) -> None:
    _write_json(root / "benchmark-set.json", benchmark)


class RepeatPassAssemblyTests(unittest.TestCase):
    def test_reconstructs_valid_pass_from_raw_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_repeat_experiment(root)

            first = assemble_repeat_pass(root, pass_number=1)
            second = assemble_repeat_pass(root, pass_number=1)
            validated = BenchmarkSet.from_mapping(first)

            self.assertEqual(first, second)
            self.assertEqual(validated.baseline_id, "baseline")
            self.assertFalse(validated.synthetic)
            self.assertEqual(
                first["metadata"]["classification"],
                "supplementary-repeat-pass",
            )
            self.assertEqual(first["metadata"]["pass_label"], "pass-1")
            self.assertEqual(
                first["metadata"]["source_benchmark_sha256"],
                _sha256(root / "benchmark-set.json"),
            )
            self.assertEqual(validated.by_id("tuned").metrics["prompt_tps"], 140.0)
            self.assertEqual(validated.by_id("tuned").metrics["generation_tps"], 35.0)
            self.assertEqual(validated.by_id("tuned").metrics["peak_rss_mib"], 1500.0)
            self.assertEqual(validated.by_id("tuned").metrics["model_size_mib"], 500.0)
            self.assertEqual(
                set(first["metadata"]["source_artifacts"]["baseline"]),
                {
                    "throughput_settings_sha256",
                    "throughput_sha256",
                    "server_evaluation_sha256",
                    "server_time_sha256",
                },
            )

            first["candidates"][0]["metrics"]["prompt_tps"] = 0
            self.assertEqual(
                assemble_repeat_pass(root, pass_number=1)["candidates"][0]["metrics"]["prompt_tps"],
                100.0,
            )

    def test_raw_tampering_is_rejected_even_if_derived_output_is_rehashed(self) -> None:
        artifact_names = (
            "throughput_pass_1",
            "server_evaluation_pass_1",
            "server_time_pass_1",
        )
        for artifact_name in artifact_names:
            with self.subTest(artifact=artifact_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                benchmark = _build_repeat_experiment(root)
                derived_path = root / "derived-pass-1.json"
                _write_json(
                    derived_path,
                    assemble_repeat_pass(root, pass_number=1),
                )

                ref = benchmark["metadata"]["candidate_evidence"]["baseline"]["artifacts"][
                    artifact_name
                ]
                raw_path = root / Path(ref["path"])
                raw_path.write_bytes(raw_path.read_bytes() + b"\n")

                # A checksum over a derived file cannot authorize changed source
                # evidence. The immutable source reference still has to match.
                _write_text(
                    root / "derived-pass-1.sha256",
                    f"{_sha256(derived_path)}  {derived_path.name}\n",
                )
                with self.assertRaisesRegex(ValidationError, "SHA-256 mismatch"):
                    assemble_repeat_pass(root, pass_number=1)

    def test_rehashed_server_case_mismatch_is_rejected_against_suite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = _build_repeat_experiment(root)
            ref = benchmark["metadata"]["candidate_evidence"]["baseline"]["artifacts"][
                "server_evaluation_pass_1"
            ]
            server_path = root / Path(ref["path"])
            server = json.loads(server_path.read_text(encoding="utf-8"))
            server["quality"]["cases"][0]["prompt"] = "Reply with exactly NO."
            _write_json(server_path, server)
            ref["sha256"] = _sha256(server_path)
            _rewrite_benchmark(root, benchmark)

            with self.assertRaisesRegex(ValidationError, "does not match evaluation-suite"):
                assemble_repeat_pass(root, pass_number=1)

    def test_suite_identity_and_file_digest_must_match_source_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = _build_repeat_experiment(root)
            benchmark["metadata"]["evaluation_suite"]["id"] = "different-suite"
            _rewrite_benchmark(root, benchmark)

            with self.assertRaisesRegex(
                ValidationError,
                "evaluation-suite.json does not match source benchmark metadata",
            ):
                assemble_repeat_pass(root, pass_number=1)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_repeat_experiment(root)
            suite = json.loads((root / "evaluation-suite.json").read_text(encoding="utf-8"))
            suite["performance"]["prompt"] = "Changed after measurement."
            _write_json(root / "evaluation-suite.json", suite)

            with self.assertRaisesRegex(
                ValidationError,
                "evaluation-suite.json does not match source benchmark metadata",
            ):
                assemble_repeat_pass(root, pass_number=1)

    def test_missing_candidate_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = _build_repeat_experiment(root)
            del benchmark["metadata"]["candidate_evidence"]["tuned"]
            _rewrite_benchmark(root, benchmark)

            with self.assertRaisesRegex(ValidationError, "missing candidate 'tuned'"):
                assemble_repeat_pass(root, pass_number=1)

    def test_unsafe_artifact_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = _build_repeat_experiment(root)
            ref = benchmark["metadata"]["candidate_evidence"]["baseline"]["artifacts"][
                "throughput_pass_1"
            ]
            ref["path"] = "../outside.jsonl"
            _rewrite_benchmark(root, benchmark)

            with self.assertRaisesRegex(ValidationError, "portable relative POSIX path"):
                assemble_repeat_pass(root, pass_number=1)

    def test_artifact_symlink_is_rejected_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = _build_repeat_experiment(root)
            ref = benchmark["metadata"]["candidate_evidence"]["baseline"]["artifacts"][
                "throughput_pass_1"
            ]
            original = root / Path(ref["path"])
            target = root / "real-throughput.jsonl"
            original.replace(target)
            try:
                original.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symbolic links are unavailable on this host: {exc}")
            ref["sha256"] = _sha256(target)
            _rewrite_benchmark(root, benchmark)

            with self.assertRaisesRegex(ValidationError, "symbolic links"):
                assemble_repeat_pass(root, pass_number=1)

    def test_cli_assemble_repeat_pass_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_repeat_experiment(root)
            output_path = root / "output" / "pass-1.json"
            stdout = io.StringIO()
            stderr = io.StringIO()

            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                exit_code = cli.main(
                    [
                        "assemble-repeat-pass",
                        "--experiment",
                        str(root),
                        "--pass-number",
                        "1",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0, stderr.getvalue())
            self.assertTrue(output_path.is_file())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["pass_label"], "pass-1")
            self.assertEqual(
                json.loads(stdout.getvalue())["metadata"]["classification"],
                "supplementary-repeat-pass",
            )

    def test_mapping_argument_is_copied_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = _build_repeat_experiment(root)
            supplied = deepcopy(benchmark)

            output = assemble_repeat_pass(
                root,
                pass_number=1,
                benchmark_mapping=supplied,
            )

            supplied["candidates"][0]["metrics"]["model_size_mib"] = 1.0
            self.assertEqual(output["candidates"][0]["metrics"]["model_size_mib"], 900.0)


if __name__ == "__main__":
    unittest.main()
