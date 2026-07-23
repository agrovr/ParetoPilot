from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from paretopilot.analysis import recommend
from paretopilot.domain import BenchmarkSet, Constraints
from paretopilot.experiment import ExperimentAssemblyError, assemble_experiment
from paretopilot.llama_summary import summarize_llama_bench_paths
from paretopilot.server_eval import pool_server_evaluations


RUNTIME_COMMIT = "a" * 40
SOURCE_COMMIT = "b" * 40
SUITE_SHA256 = "e" * 64
MODEL_REPOSITORY = "example/ParetoPilot-1.7B-GGUF"
MODEL_REVISION = "pinned-model-revision"
MODEL_FAMILY = "ParetoPilot-1.7B-Instruct"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _benchmark_rows(
    filename: str,
    prompt_tps: float,
    generation_tps: float,
    *,
    batch_size: int,
    ubatch_size: int,
) -> list[dict[str, object]]:
    def row(n_prompt: int, n_gen: int, tps: float, duration_ns: float) -> dict[str, object]:
        samples_tps = [tps - 1.0, tps - 0.5, tps, tps + 0.5, tps + 1.0]
        samples_ns = [duration_ns] * 5
        return {
            "build_commit": RUNTIME_COMMIT,
            "model_filename": filename,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "avg_ns": duration_ns,
            "avg_ts": tps,
            "samples_ns": samples_ns,
            "samples_ts": samples_tps,
            "n_threads": 4,
            "n_batch": batch_size,
            "n_ubatch": ubatch_size,
            "n_gpu_layers": 0,
            "devices": "none",
            "no_op_offload": 1,
        }

    return [
        row(512, 0, prompt_tps, 1_000_000_000.0),
        row(0, 128, generation_tps, 4_000_000_000.0),
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    _write_text(
        path,
        "".join(json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in rows),
    )


def _throughput_settings(model: dict, *, batch_size: int, ubatch_size: int, kleidiai: bool) -> dict:
    return {
        "threads": 4,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "prompt_tokens": 512,
        "generation_tokens": 128,
        "repetitions_per_pass": 5,
        "warmup": True,
        "cpu_only": True,
        "model_sha256": model["sha256"],
        "llama_cpp_commit": RUNTIME_COMMIT,
        "build": {"kleidiai": kleidiai},
    }


def _server_evaluation(candidate_id: str, latency_offset: float) -> dict:
    ttft = [
        50.0 + latency_offset,
        51.0 + latency_offset,
        52.0 + latency_offset,
        53.0 + latency_offset,
        54.0 + latency_offset,
    ]
    e2e = [
        500.0 + latency_offset,
        510.0 + latency_offset,
        520.0 + latency_offset,
        530.0 + latency_offset,
        540.0 + latency_offset,
    ]
    return {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "synthetic": False,
        "suite": {
            "id": "paretopilot-arm64-eval-v1",
            "license": "CC0-1.0",
            "sha256": SUITE_SHA256,
            "quality_case_count": 2,
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
            "passed": 2,
            "total": 2,
            "cases": [
                {
                    "id": "arithmetic",
                    "prompt": "Return only the result of 2 + 2.",
                    "accepted_answers": ["4"],
                    "response": "4",
                    "matched": True,
                    "matched_answer": "4",
                },
                {
                    "id": "capital",
                    "prompt": "Return only the capital of France.",
                    "accepted_answers": ["Paris"],
                    "response": "Paris",
                    "matched": True,
                    "matched_answer": "Paris",
                },
            ],
        },
        "latency": {
            "method": "single-client streamed HTTP requests",
            "ttft_ms_p50": ttft[2],
            "ttft_ms_p95": ttft[4],
            "e2e_latency_ms_p50": e2e[2],
            "e2e_latency_ms_p95": e2e[4],
            "samples": [
                {
                    "index": index,
                    "ttft_ms": ttft[index - 1],
                    "e2e_latency_ms": e2e[index - 1],
                    "event_count": 64,
                    "predicted_tokens": 64,
                    "content": f"measured completion {index}",
                }
                for index in range(1, 6)
            ],
        },
    }


def _resource_measurement(
    candidate_id: str,
    model: dict,
    argv: list[str],
    peak_kib: int,
) -> dict:
    return {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "synthetic": False,
        "run_id": "987654321",
        "model": {
            key: model[key] for key in ("family", "repository", "revision", "filename", "sha256")
        },
        "runtime": {
            "name": "llama.cpp",
            "repository": "https://github.com/ggml-org/llama.cpp",
            "revision": RUNTIME_COMMIT,
        },
        "evaluation_suite": {
            "id": "paretopilot-arm64-eval-v1",
            "sha256": SUITE_SHA256,
        },
        "deployment_argv": argv,
        "measurement_tool": "/usr/bin/time -v",
        "maximum_resident_set_kbytes": peak_kib,
    }


def _model(filename: str, digest_digit: str, size_bytes: int, quantization: str) -> dict:
    return {
        "family": MODEL_FAMILY,
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "filename": filename,
        "sha256": digest_digit * 64,
        "size_bytes": size_bytes,
        "quantization": quantization,
    }


def _build_experiment(root: Path) -> tuple[Path, dict]:
    specifications = [
        (
            "q8-generic",
            "Q8_0 generic baseline",
            _model("model-q8_0.gguf", "1", 2_000_000_000, "Q8_0"),
            False,
            110.0,
            28.0,
            0.0,
            2_300_000,
        ),
        (
            "q4-generic",
            "Q4_0 generic",
            _model("model-q4_0.gguf", "2", 1_100_000_000, "Q4_0"),
            False,
            130.0,
            33.0,
            -2.0,
            1_450_000,
        ),
        (
            "q4-kleidiai",
            "Q4_0 KleidiAI",
            _model("model-q4_0.gguf", "2", 1_100_000_000, "Q4_0"),
            True,
            145.0,
            36.0,
            -4.0,
            1_430_000,
        ),
        (
            "q4-kleidiai-tuned",
            "Q4_0 KleidiAI tuned",
            _model("model-q4_0.gguf", "2", 1_100_000_000, "Q4_0"),
            True,
            150.0,
            39.0,
            -6.0,
            1_420_000,
        ),
    ]
    candidates: list[dict] = []
    for (
        candidate_id,
        label,
        model,
        kleidiai,
        prompt_tps,
        generation_tps,
        latency_offset,
        peak_kib,
    ) in specifications:
        batch_size = 512
        ubatch_size = 512 if candidate_id.endswith("tuned") else 128
        build_name = "kleidiai" if kleidiai else "generic"
        argv = [
            f"./.candidate-build/{build_name}/bin/llama-server",
            "--model",
            f"./.candidate-models/{model['filename']}",
            "--threads",
            "4",
            "--threads-batch",
            "4",
            "--batch-size",
            str(batch_size),
            "--ubatch-size",
            str(ubatch_size),
            "--ctx-size",
            "2048",
            "--parallel",
            "1",
            "--n-gpu-layers",
            "0",
            "-lv",
            "4",
            "--host",
            "127.0.0.1",
            "--port",
            str(18081 + len(candidates)),
        ]
        refs: dict[str, dict[str, str]] = {}

        def add_ref(artifact_name: str, filename: str) -> Path:
            relative = f"artifacts/{candidate_id}/{filename}"
            artifact_path = root / Path(relative)
            refs[artifact_name] = {"path": relative, "sha256": _sha256(artifact_path)}
            return artifact_path

        settings = _throughput_settings(
            model,
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            kleidiai=kleidiai,
        )
        settings_path = root / f"artifacts/{candidate_id}/throughput-settings.json"
        _write_json(settings_path, settings)
        add_ref("throughput_settings", "throughput-settings.json")

        throughput_command_path = root / f"artifacts/{candidate_id}/throughput-command.json"
        _write_json(
            throughput_command_path,
            {
                "schema_version": "1.0",
                "working_directory": ".candidate-models",
                "argv": [
                    f"../.candidate-build/{build_name}/bin/llama-bench",
                    "-m",
                    model["filename"],
                    "-p",
                    str(settings["prompt_tokens"]),
                    "-n",
                    str(settings["generation_tokens"]),
                    "-t",
                    str(settings["threads"]),
                    "-b",
                    str(settings["batch_size"]),
                    "-ub",
                    str(settings["ubatch_size"]),
                    "-r",
                    str(settings["repetitions_per_pass"]),
                    "-dev",
                    "none",
                    "-ngl",
                    "0",
                    "-nopo",
                    "1",
                    "-v",
                    "-o",
                    "jsonl",
                ],
            },
        )
        add_ref("throughput_command", "throughput-command.json")

        throughput_paths: list[Path] = []
        for pass_number in (1, 2):
            path = root / f"artifacts/{candidate_id}/throughput-pass-{pass_number}.jsonl"
            _write_jsonl(
                path,
                _benchmark_rows(
                    model["filename"],
                    prompt_tps,
                    generation_tps,
                    batch_size=batch_size,
                    ubatch_size=ubatch_size,
                ),
            )
            throughput_paths.append(path)
            add_ref(f"throughput_pass_{pass_number}", f"throughput-pass-{pass_number}.jsonl")
            stderr_path = (
                root / f"artifacts/{candidate_id}/throughput-pass-{pass_number}.stderr.log"
            )
            _write_text(
                stderr_path,
                "CPU_KLEIDIAI model buffer enabled\n" if kleidiai else "generic CPU buffer\n",
            )
            add_ref(
                f"throughput_stderr_pass_{pass_number}",
                f"throughput-pass-{pass_number}.stderr.log",
            )

        summary = summarize_llama_bench_paths(
            candidate_id,
            [("pass-1", throughput_paths[0]), ("pass-2", throughput_paths[1])],
            settings=settings,
        ).to_mapping()
        summary["input_fingerprints"] = {
            "settings_sha256": _sha256(settings_path),
            "artifacts_sha256": {
                "pass-1": _sha256(throughput_paths[0]),
                "pass-2": _sha256(throughput_paths[1]),
            },
        }
        summary_path = root / f"artifacts/{candidate_id}/llama-summary.json"
        _write_json(summary_path, summary)
        add_ref("llama_summary", "llama-summary.json")

        server_paths: list[Path] = []
        for pass_number, pass_offset in ((1, 0.0), (2, 0.25)):
            path = root / f"artifacts/{candidate_id}/server-evaluation-pass-{pass_number}.json"
            _write_json(path, _server_evaluation(candidate_id, latency_offset + pass_offset))
            server_paths.append(path)
            add_ref(
                f"server_evaluation_pass_{pass_number}",
                f"server-evaluation-pass-{pass_number}.json",
            )
            stderr_path = root / f"artifacts/{candidate_id}/server-pass-{pass_number}.stderr.log"
            _write_text(
                stderr_path,
                "CPU_KLEIDIAI model buffer enabled\n" if kleidiai else "generic CPU buffer\n",
            )
            add_ref(
                f"server_stderr_pass_{pass_number}",
                f"server-pass-{pass_number}.stderr.log",
            )
        aggregate_server_path = root / f"artifacts/{candidate_id}/server-evaluation.json"
        _write_json(aggregate_server_path, pool_server_evaluations(server_paths))
        add_ref("server_evaluation", "server-evaluation.json")

        for pass_number, pass_peak_kib in ((1, peak_kib - 1_000), (2, peak_kib)):
            time_path = root / f"artifacts/{candidate_id}/server-time-pass-{pass_number}.txt"
            _write_text(
                time_path,
                "Command being timed: llama-server\n"
                f"Maximum resident set size (kbytes): {pass_peak_kib}\n"
                "Exit status: 0\n",
            )
            add_ref(f"server_time_pass_{pass_number}", f"server-time-pass-{pass_number}.txt")

        resource_path = root / f"artifacts/{candidate_id}/resource-measurement.json"
        _write_json(
            resource_path,
            _resource_measurement(candidate_id, model, argv, peak_kib),
        )
        add_ref("resource_measurement", "resource-measurement.json")

        for artifact_name, ref in refs.items():
            artifact_path = root / Path(ref["path"])
            # Recalculate after all aggregate files have been finalized.
            ref["sha256"] = _sha256(artifact_path)
        candidates.append(
            {
                "id": candidate_id,
                "label": label,
                "parameters": {
                    "cpu_only": True,
                    "threads": 4,
                    "batch_size": batch_size,
                    "ubatch_size": ubatch_size,
                    "context_size": 2048,
                    "kleidiai": kleidiai,
                    "tuned": candidate_id.endswith("tuned"),
                },
                "model": model,
                "deployment_argv": argv,
                "artifacts": refs,
            }
        )

    manifest = {
        "schema_version": "1.0",
        "experiment_id": "arm64-four-candidate-study",
        "baseline_id": "q8-generic",
        "classification": "exploratory",
        "synthetic": False,
        "source": {
            "repository": "https://github.com/agrovr/ParetoPilot",
            "revision": SOURCE_COMMIT,
            "workflow": ".github/workflows/experiment-arm64.yml",
            "run_id": "987654321",
            "run_attempt": 1,
            "generated_at_utc": "2026-07-22T18:30:00Z",
            "runner": {
                "os": "Ubuntu 24.04",
                "architecture": "arm64",
                "cpu": "Arm Neoverse-N2",
                "cpu_count": 4,
            },
        },
        "model_family": {
            "name": MODEL_FAMILY,
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
        },
        "runtime": {
            "name": "llama.cpp",
            "repository": "https://github.com/ggml-org/llama.cpp",
            "revision": RUNTIME_COMMIT,
        },
        "optimization_library": {
            "name": "KleidiAI",
            "repository": "https://github.com/ARM-software/kleidiai",
            "version": "v1.24.0",
            "source_archive_sha256": "f" * 64,
            "source_archive_size_bytes": 2_466_038,
        },
        "evaluation_suite": {
            "id": "paretopilot-arm64-eval-v1",
            "sha256": SUITE_SHA256,
        },
        "candidates": candidates,
    }
    path = root / "experiment.json"
    _write_json(path, manifest)
    return path, manifest


def _rewrite_manifest(path: Path, manifest: dict) -> None:
    _write_json(path, manifest)


def _bind_evaluation_suite_file(
    root: Path,
    manifest_path: Path,
    manifest: dict,
) -> Path:
    suite = {
        "schema_version": "1.0",
        "id": "paretopilot-qwen-behavior-v2",
        "license": "CC0-1.0",
        "quality_cases": [
            {
                "id": "arithmetic",
                "prompt": "Return only the result of 2 + 2.",
                "accepted_answers": ["4"],
                "match_mode": "normalized-text",
            },
            {
                "id": "capital",
                "prompt": "Return only the capital of France.",
                "accepted_answers": ["Paris"],
                "match_mode": "normalized-text",
            },
        ],
        "performance": {
            "prompt": "Explain why reproducible measurements matter.",
            "generation_tokens": 64,
            "repetitions": 5,
            "warmups": 1,
        },
    }
    suite_path = root / "evaluation-suite.json"
    _write_json(suite_path, suite)
    suite_sha256 = _sha256(suite_path)
    manifest["evaluation_suite"] = {
        "id": suite["id"],
        "sha256": suite_sha256,
    }
    manifest["evaluation_suite_path"] = "evaluation-suite.json"

    for candidate in manifest["candidates"]:
        refs = candidate["artifacts"]
        pass_paths = []
        for pass_number in (1, 2):
            ref = refs[f"server_evaluation_pass_{pass_number}"]
            path = root / Path(ref["path"])
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["suite"].update(
                {
                    "id": suite["id"],
                    "license": suite["license"],
                    "sha256": suite_sha256,
                    "quality_case_count": len(suite["quality_cases"]),
                    "performance_repetitions": suite["performance"]["repetitions"],
                    "performance_warmups": suite["performance"]["warmups"],
                    "generation_tokens": suite["performance"]["generation_tokens"],
                }
            )
            for case, expected in zip(
                payload["quality"]["cases"],
                suite["quality_cases"],
                strict=True,
            ):
                case["match_mode"] = expected["match_mode"]
            _write_json(path, payload)
            ref["sha256"] = _sha256(path)
            pass_paths.append(path)

        aggregate_ref = refs["server_evaluation"]
        aggregate_path = root / Path(aggregate_ref["path"])
        _write_json(aggregate_path, pool_server_evaluations(pass_paths))
        aggregate_ref["sha256"] = _sha256(aggregate_path)

        resource_ref = refs["resource_measurement"]
        resource_path = root / Path(resource_ref["path"])
        resource = json.loads(resource_path.read_text(encoding="utf-8"))
        resource["evaluation_suite"] = {
            "id": suite["id"],
            "sha256": suite_sha256,
        }
        _write_json(resource_path, resource)
        resource_ref["sha256"] = _sha256(resource_path)

    _rewrite_manifest(manifest_path, manifest)
    return suite_path


def _set_server_run_counts(
    payload: dict,
    *,
    repetitions: int | None = None,
    warmups: int | None = None,
) -> None:
    if repetitions is not None:
        payload["suite"]["performance_repetitions"] = repetitions
        samples = payload["latency"]["samples"][:repetitions]
        payload["latency"]["samples"] = samples
        for field, source in (
            ("ttft_ms_p50", "ttft_ms"),
            ("ttft_ms_p95", "ttft_ms"),
            ("e2e_latency_ms_p50", "e2e_latency_ms"),
            ("e2e_latency_ms_p95", "e2e_latency_ms"),
        ):
            values = sorted(float(sample[source]) for sample in samples)
            percentile = 50 if field.endswith("p50") else 95
            rank = max(1, (percentile * len(values) + 99) // 100)
            payload["latency"][field] = values[rank - 1]
    if warmups is not None:
        payload["suite"]["performance_warmups"] = warmups


def _mutate_artifact(
    root: Path,
    manifest_path: Path,
    manifest: dict,
    candidate_index: int,
    artifact_name: str,
    mutate,
    *,
    refresh_hash: bool = True,
) -> None:
    ref = manifest["candidates"][candidate_index]["artifacts"][artifact_name]
    artifact_path = root / Path(ref["path"])
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(artifact_path, payload)
    if refresh_hash:
        ref["sha256"] = _sha256(artifact_path)
    _rewrite_manifest(manifest_path, manifest)


def _mutate_text_artifact(
    root: Path,
    manifest_path: Path,
    manifest: dict,
    candidate_index: int,
    artifact_name: str,
    mutate,
    *,
    refresh_hash: bool = True,
) -> None:
    ref = manifest["candidates"][candidate_index]["artifacts"][artifact_name]
    artifact_path = root / Path(ref["path"])
    _write_text(artifact_path, mutate(artifact_path.read_text(encoding="utf-8")))
    if refresh_hash:
        ref["sha256"] = _sha256(artifact_path)
    _rewrite_manifest(manifest_path, manifest)


class ExperimentAssemblyTests(unittest.TestCase):
    def test_checksummed_v2_evaluation_suite_is_bound_to_every_server_artifact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _build_experiment(root)
            _bind_evaluation_suite_file(root, manifest_path, manifest)

            mapping = assemble_experiment(manifest_path)

            self.assertEqual(
                mapping["metadata"]["evaluation_suite"]["id"],
                "paretopilot-qwen-behavior-v2",
            )
            self.assertEqual(
                mapping["metadata"]["evaluation_suite"]["sha256"],
                _sha256(root / "evaluation-suite.json"),
            )

    def test_v2_suite_path_hash_contract_and_run_settings_fail_closed(self) -> None:
        scenarios = (
            (
                "missing path",
                lambda root, manifest, suite: manifest.pop("evaluation_suite_path"),
                "requires a checksummed evaluation_suite_path",
            ),
            (
                "unsafe path",
                lambda root, manifest, suite: manifest.update(
                    {"evaluation_suite_path": "../evaluation-suite.json"}
                ),
                "portable relative POSIX path",
            ),
            (
                "suite hash",
                lambda root, manifest, suite: _write_text(suite, "{}\n"),
                "SHA-256 mismatch",
            ),
            (
                "duplicate artifact path",
                lambda root, manifest, suite: manifest["candidates"][0]["artifacts"][
                    "throughput_settings"
                ].update(
                    {
                        "path": "evaluation-suite.json",
                        "sha256": _sha256(suite),
                    }
                ),
                "artifact paths must be unique",
            ),
            (
                "license",
                lambda root, manifest, suite: _mutate_artifact(
                    root,
                    root / "experiment.json",
                    manifest,
                    0,
                    "server_evaluation",
                    lambda payload: payload["suite"].update({"license": "unknown"}),
                ),
                "license does not match",
            ),
            (
                "case prompt",
                lambda root, manifest, suite: _mutate_artifact(
                    root,
                    root / "experiment.json",
                    manifest,
                    0,
                    "server_evaluation",
                    lambda payload: payload["quality"]["cases"][0].update(
                        {"prompt": "Different prompt"}
                    ),
                ),
                "does not match the checksummed evaluation suite",
            ),
            (
                "case answers",
                lambda root, manifest, suite: _mutate_artifact(
                    root,
                    root / "experiment.json",
                    manifest,
                    0,
                    "server_evaluation",
                    lambda payload: payload["quality"]["cases"][0].update(
                        {"accepted_answers": ["4", "four"]}
                    ),
                ),
                "does not match the checksummed evaluation suite",
            ),
            (
                "case mode",
                lambda root, manifest, suite: _mutate_artifact(
                    root,
                    root / "experiment.json",
                    manifest,
                    0,
                    "server_evaluation",
                    lambda payload: payload["quality"]["cases"][0].update(
                        {"match_mode": "trimmed-exact"}
                    ),
                ),
                "does not match the checksummed evaluation suite",
            ),
            (
                "repetitions",
                lambda root, manifest, suite: _mutate_artifact(
                    root,
                    root / "experiment.json",
                    manifest,
                    0,
                    "server_evaluation",
                    lambda payload: _set_server_run_counts(
                        payload,
                        repetitions=9,
                    ),
                ),
                "performance_repetitions does not match",
            ),
            (
                "warmups",
                lambda root, manifest, suite: _mutate_artifact(
                    root,
                    root / "experiment.json",
                    manifest,
                    0,
                    "server_evaluation",
                    lambda payload: _set_server_run_counts(
                        payload,
                        warmups=3,
                    ),
                ),
                "performance_warmups does not match",
            ),
        )
        for label, mutate, message in scenarios:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, manifest = _build_experiment(root)
                suite_path = _bind_evaluation_suite_file(root, manifest_path, manifest)
                mutate(root, manifest, suite_path)
                _rewrite_manifest(manifest_path, manifest)
                with self.assertRaisesRegex(ExperimentAssemblyError, message):
                    assemble_experiment(manifest_path)

    def test_intended_four_candidate_study_is_domain_and_recommend_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = _build_experiment(root)

            mapping = assemble_experiment(manifest_path)
            benchmarks = BenchmarkSet.from_mapping(mapping)

            self.assertFalse(benchmarks.synthetic)
            self.assertEqual(benchmarks.baseline_id, "q8-generic")
            self.assertEqual(benchmarks.metadata["classification"], "exploratory")
            self.assertEqual(
                benchmarks.metadata["optimization_library"]["version"],
                "v1.24.0",
            )
            self.assertEqual(len(benchmarks.candidates), 4)
            self.assertEqual(
                set(benchmarks.by_id("q4-kleidiai-tuned").metrics),
                {
                    "prompt_tps",
                    "generation_tps",
                    "quality_score",
                    "ttft_ms_p95",
                    "e2e_latency_ms_p95",
                    "peak_rss_mib",
                    "model_size_mib",
                },
            )
            self.assertEqual(
                benchmarks.by_id("q4-kleidiai").parameters["deployment_argv"],
                [
                    "./.candidate-build/kleidiai/bin/llama-server",
                    "--model",
                    "./.candidate-models/model-q4_0.gguf",
                    "--threads",
                    "4",
                    "--threads-batch",
                    "4",
                    "--batch-size",
                    "512",
                    "--ubatch-size",
                    "128",
                    "--ctx-size",
                    "2048",
                    "--parallel",
                    "1",
                    "--n-gpu-layers",
                    "0",
                    "-lv",
                    "4",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "18083",
                ],
            )
            self.assertEqual(
                mapping["metadata"]["candidate_evidence"]["q4-kleidiai"]["artifacts"][
                    "server_evaluation"
                ]["sha256"],
                _sha256(root / "artifacts/q4-kleidiai/server-evaluation.json"),
            )

            constraints = Constraints.from_mapping(
                {
                    "min_quality_retention": 0.95,
                    "quality_metric": "quality_score",
                    "objective": {"metric": "generation_tps", "direction": "max"},
                    "frontier_metrics": {
                        "generation_tps": "max",
                        "prompt_tps": "max",
                        "quality_score": "max",
                        "ttft_ms_p95": "min",
                        "e2e_latency_ms_p95": "min",
                        "peak_rss_mib": "min",
                        "model_size_mib": "min",
                    },
                }
            )
            decision = recommend(benchmarks, constraints)
            self.assertEqual(decision["selected_id"], "q4-kleidiai-tuned")

    def test_mapping_is_deterministic_and_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = _build_experiment(Path(temporary))
            first = assemble_experiment(manifest_path)
            second = assemble_experiment(manifest_path)
            self.assertEqual(first, second)
            first["candidates"][0]["parameters"]["deployment_argv"].append("changed")
            self.assertNotEqual(first, assemble_experiment(manifest_path))

    def test_tampered_artifact_is_rejected_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _build_experiment(root)
            _mutate_artifact(
                root,
                manifest_path,
                manifest,
                0,
                "llama_summary",
                lambda payload: payload.update({"model_filename": "tampered.gguf"}),
                refresh_hash=False,
            )
            with self.assertRaisesRegex(ExperimentAssemblyError, "SHA-256 mismatch"):
                assemble_experiment(manifest_path)

    def test_rehashed_raw_throughput_tamper_is_rejected_by_recomputation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _build_experiment(root)

            def alter_first_row(text: str) -> str:
                lines = text.splitlines()
                row = json.loads(lines[0])
                row["samples_ts"] = [value + 10.0 for value in row["samples_ts"]]
                row["avg_ts"] += 10.0
                lines[0] = json.dumps(row, allow_nan=False, sort_keys=True)
                return "\n".join(lines) + "\n"

            _mutate_text_artifact(
                root,
                manifest_path,
                manifest,
                0,
                "throughput_pass_1",
                alter_first_row,
            )
            with self.assertRaisesRegex(
                ExperimentAssemblyError, "llama summary does not exactly match recomputed"
            ):
                assemble_experiment(manifest_path)

    def test_rehashed_throughput_command_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _build_experiment(root)

            def change_repetitions(payload: dict) -> None:
                argv = payload["argv"]
                argv[argv.index("-r") + 1] = "99"

            _mutate_artifact(
                root,
                manifest_path,
                manifest,
                0,
                "throughput_command",
                change_repetitions,
            )
            with self.assertRaisesRegex(
                ExperimentAssemblyError,
                "throughput command.*does not match the recorded settings and model",
            ):
                assemble_experiment(manifest_path)

    def test_rehashed_kleidiai_dispatch_log_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _build_experiment(root)
            _mutate_text_artifact(
                root,
                manifest_path,
                manifest,
                2,
                "server_stderr_pass_2",
                lambda _: "generic CPU buffer\n",
            )
            with self.assertRaisesRegex(
                ExperimentAssemblyError,
                "missing KleidiAI dispatch marker.*server_stderr_pass_2",
            ):
                assemble_experiment(manifest_path)

    def test_rehashed_raw_server_tamper_is_rejected_by_repooling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _build_experiment(root)

            def shift_latency(payload: dict) -> None:
                latency = payload["latency"]
                for sample in latency["samples"]:
                    sample["ttft_ms"] += 10.0
                    sample["e2e_latency_ms"] += 10.0
                latency["ttft_ms_p50"] += 10.0
                latency["ttft_ms_p95"] += 10.0
                latency["e2e_latency_ms_p50"] += 10.0
                latency["e2e_latency_ms_p95"] += 10.0

            _mutate_artifact(
                root,
                manifest_path,
                manifest,
                0,
                "server_evaluation_pass_2",
                shift_latency,
            )
            with self.assertRaisesRegex(
                ExperimentAssemblyError, "server evaluation does not exactly match pooled"
            ):
                assemble_experiment(manifest_path)

    def test_rehashed_raw_gnu_time_tamper_is_rejected_by_maximum_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _build_experiment(root)

            def increase_peak(text: str) -> str:
                return text.replace(
                    "Maximum resident set size (kbytes): 2300000",
                    "Maximum resident set size (kbytes): 2400000",
                )

            _mutate_text_artifact(
                root,
                manifest_path,
                manifest,
                0,
                "server_time_pass_2",
                increase_peak,
            )
            with self.assertRaisesRegex(
                ExperimentAssemblyError, "resource measurement maximum does not match raw"
            ):
                assemble_experiment(manifest_path)

    def test_manifest_parameters_and_deployment_argv_must_match_settings(self) -> None:
        mutations = {
            "parameters": (
                lambda candidate: candidate["parameters"].update({"threads": 8}),
                r"parameters\.threads does not match throughput settings",
            ),
            "argv": (
                lambda candidate: candidate["deployment_argv"].__setitem__(
                    candidate["deployment_argv"].index("--batch-size") + 1,
                    "256",
                ),
                "deployment_argv --batch-size does not match throughput settings",
            ),
            "kleidiai": (
                lambda candidate: candidate["parameters"].update({"kleidiai": True}),
                r"parameters\.kleidiai does not match throughput settings",
            ),
            "context_size": (
                lambda candidate: candidate["parameters"].update({"context_size": 4096}),
                "deployment_argv --ctx-size does not match throughput settings",
            ),
            "gpu_layers": (
                lambda candidate: candidate["deployment_argv"].__setitem__(
                    candidate["deployment_argv"].index("--n-gpu-layers") + 1,
                    "1",
                ),
                "deployment_argv --n-gpu-layers must be '0'",
            ),
            "parallel": (
                lambda candidate: candidate["deployment_argv"].__setitem__(
                    candidate["deployment_argv"].index("--parallel") + 1,
                    "2",
                ),
                "deployment_argv --parallel must be '1'",
            ),
            "log_verbosity": (
                lambda candidate: candidate["deployment_argv"].__setitem__(
                    candidate["deployment_argv"].index("-lv") + 1,
                    "3",
                ),
                "deployment_argv -lv must be '4'",
            ),
            "host": (
                lambda candidate: candidate["deployment_argv"].__setitem__(
                    candidate["deployment_argv"].index("--host") + 1,
                    "0.0.0.0",
                ),
                "deployment_argv --host must be '127.0.0.1'",
            ),
            "port": (
                lambda candidate: candidate["deployment_argv"].__setitem__(
                    candidate["deployment_argv"].index("--port") + 1,
                    "70000",
                ),
                "deployment_argv --port must be from 1 to 65535",
            ),
        }
        for scenario, (mutation, pattern) in mutations.items():
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, manifest = _build_experiment(root)
                mutation(manifest["candidates"][0])
                _rewrite_manifest(manifest_path, manifest)
                with self.assertRaisesRegex(ExperimentAssemblyError, pattern):
                    assemble_experiment(manifest_path)

    def test_llama_summary_input_fingerprints_must_match_raw_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _build_experiment(root)
            _mutate_artifact(
                root,
                manifest_path,
                manifest,
                0,
                "llama_summary",
                lambda payload: payload["input_fingerprints"].update({"settings_sha256": "9" * 64}),
            )
            with self.assertRaisesRegex(
                ExperimentAssemblyError, "llama summary does not exactly match recomputed"
            ):
                assemble_experiment(manifest_path)

    def test_unsafe_and_absolute_artifact_paths_are_rejected(self) -> None:
        attacks = ["../outside.json", "/tmp/outside.json", "C:/outside.json", "a\\b.json"]
        for attack in attacks:
            with self.subTest(path=attack), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, manifest = _build_experiment(root)
                manifest["candidates"][0]["artifacts"]["llama_summary"]["path"] = attack
                _rewrite_manifest(manifest_path, manifest)
                with self.assertRaisesRegex(ExperimentAssemblyError, "relative POSIX path"):
                    assemble_experiment(manifest_path)

    def test_candidate_id_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _build_experiment(root)
            _mutate_artifact(
                root,
                manifest_path,
                manifest,
                0,
                "server_evaluation",
                lambda payload: payload.update({"candidate_id": "wrong-candidate"}),
            )
            with self.assertRaisesRegex(ExperimentAssemblyError, "candidate_id does not match"):
                assemble_experiment(manifest_path)

    def test_run_model_runtime_and_eval_identity_mismatches_are_rejected(self) -> None:
        mutations = {
            "run": lambda payload: payload.update({"run_id": "other-run"}),
            "model": lambda payload: payload["model"].update({"sha256": "9" * 64}),
            "runtime": lambda payload: payload["runtime"].update({"revision": "9" * 40}),
            "evaluation": lambda payload: payload["evaluation_suite"].update({"sha256": "9" * 64}),
        }
        patterns = {
            "run": "run_id does not match",
            "model": "model sha256 does not match",
            "runtime": "runtime revision does not match",
            "evaluation": "evaluation_suite does not match",
        }
        for identity, mutation in mutations.items():
            with self.subTest(identity=identity), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, manifest = _build_experiment(root)
                _mutate_artifact(
                    root,
                    manifest_path,
                    manifest,
                    0,
                    "resource_measurement",
                    mutation,
                )
                with self.assertRaisesRegex(ExperimentAssemblyError, patterns[identity]):
                    assemble_experiment(manifest_path)

    def test_summary_runtime_and_model_mismatches_are_rejected(self) -> None:
        mutations = {
            "runtime": lambda payload: payload.update({"build_commit": "9" * 12}),
            "model": lambda payload: payload.update({"model_filename": "other.gguf"}),
        }
        for identity, mutation in mutations.items():
            with self.subTest(identity=identity), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, manifest = _build_experiment(root)
                _mutate_artifact(root, manifest_path, manifest, 0, "llama_summary", mutation)
                with self.assertRaisesRegex(ExperimentAssemblyError, "does not match"):
                    assemble_experiment(manifest_path)

    def test_server_suite_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _build_experiment(root)
            _mutate_artifact(
                root,
                manifest_path,
                manifest,
                0,
                "server_evaluation",
                lambda payload: payload["suite"].update({"id": "wrong-suite"}),
            )
            with self.assertRaisesRegex(ExperimentAssemblyError, "suite id does not match"):
                assemble_experiment(manifest_path)

    def test_quality_match_is_recomputed_from_response(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _build_experiment(root)
            _mutate_artifact(
                root,
                manifest_path,
                manifest,
                0,
                "server_evaluation",
                lambda payload: payload["quality"]["cases"][0].update(
                    {"response": "5", "matched": True, "matched_answer": "4"}
                ),
            )
            with self.assertRaisesRegex(
                ExperimentAssemblyError,
                "matched outcome does not match response|does not agree with the response",
            ):
                assemble_experiment(manifest_path)

    def test_missing_required_metrics_are_rejected(self) -> None:
        mutations = {
            "throughput": (
                "llama_summary",
                lambda payload: payload["tests"].pop("tg"),
                "missing metrics: tg",
            ),
            "quality": (
                "server_evaluation",
                lambda payload: payload["quality"].pop("score"),
                "missing fields: score",
            ),
            "rss": (
                "resource_measurement",
                lambda payload: payload.pop("maximum_resident_set_kbytes"),
                "missing fields: maximum_resident_set_kbytes",
            ),
        }
        for metric, (artifact, mutation, pattern) in mutations.items():
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, manifest = _build_experiment(root)
                _mutate_artifact(root, manifest_path, manifest, 0, artifact, mutation)
                with self.assertRaisesRegex(ExperimentAssemblyError, pattern):
                    assemble_experiment(manifest_path)

    def test_synthetic_artifacts_are_rejected(self) -> None:
        cases = {
            "summary": (
                "llama_summary",
                lambda payload: payload.update({"synthetic_fixture": True}),
            ),
            "server": ("server_evaluation", lambda payload: payload.update({"synthetic": True})),
            "resource": (
                "resource_measurement",
                lambda payload: payload.update({"synthetic": True}),
            ),
        }
        for name, (artifact, mutation) in cases.items():
            with self.subTest(artifact=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, manifest = _build_experiment(root)
                _mutate_artifact(root, manifest_path, manifest, 0, artifact, mutation)
                with self.assertRaisesRegex(ExperimentAssemblyError, "synthetic"):
                    assemble_experiment(manifest_path)

    def test_duplicate_candidates_baseline_omission_and_small_study_are_rejected(self) -> None:
        mutations = {
            "duplicate": (
                lambda manifest: manifest["candidates"].append(deepcopy(manifest["candidates"][0])),
                "candidate ids must be unique",
            ),
            "baseline": (
                lambda manifest: manifest.update({"baseline_id": "missing-baseline"}),
                "baseline_id must refer",
            ),
            "small": (
                lambda manifest: manifest.update({"candidates": manifest["candidates"][:2]}),
                "at least three",
            ),
        }
        for scenario, (mutation, pattern) in mutations.items():
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, manifest = _build_experiment(root)
                mutation(manifest)
                _rewrite_manifest(manifest_path, manifest)
                with self.assertRaisesRegex(ExperimentAssemblyError, pattern):
                    assemble_experiment(manifest_path)

    def test_unknown_fields_are_rejected_in_manifest_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _build_experiment(root)
            manifest["surprise"] = True
            _rewrite_manifest(manifest_path, manifest)
            with self.assertRaisesRegex(ExperimentAssemblyError, "unknown fields: surprise"):
                assemble_experiment(manifest_path)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _build_experiment(root)
            _mutate_artifact(
                root,
                manifest_path,
                manifest,
                0,
                "server_evaluation",
                lambda payload: payload["quality"].update({"surprise": True}),
            )
            with self.assertRaisesRegex(ExperimentAssemblyError, "unknown fields: surprise"):
                assemble_experiment(manifest_path)

    def test_deployment_argv_mismatch_and_newline_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _build_experiment(root)
            _mutate_artifact(
                root,
                manifest_path,
                manifest,
                0,
                "resource_measurement",
                lambda payload: payload["deployment_argv"].append("--wrong"),
            )
            with self.assertRaisesRegex(ExperimentAssemblyError, "deployment_argv does not match"):
                assemble_experiment(manifest_path)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _build_experiment(root)
            manifest["candidates"][0]["deployment_argv"][0] = "llama-server\nmalicious"
            _rewrite_manifest(manifest_path, manifest)
            with self.assertRaisesRegex(ExperimentAssemblyError, "without NUL or newlines"):
                assemble_experiment(manifest_path)

    def test_different_model_hashes_are_allowed_only_with_pinned_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _build_experiment(root)
            mapping = assemble_experiment(manifest_path)
            q8_hash = mapping["metadata"]["candidate_evidence"]["q8-generic"]["model"]["sha256"]
            q4_hash = mapping["metadata"]["candidate_evidence"]["q4-generic"]["model"]["sha256"]
            self.assertNotEqual(q8_hash, q4_hash)

            manifest["candidates"][1]["model"]["revision"] = "unapproved-revision"
            _rewrite_manifest(manifest_path, manifest)
            with self.assertRaisesRegex(ExperimentAssemblyError, "declared model_family revision"):
                assemble_experiment(manifest_path)


if __name__ == "__main__":
    unittest.main()
