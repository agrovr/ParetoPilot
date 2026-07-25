from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from paretopilot.capacity_eval import (
    assemble_capacity_study,
    bind_capacity_quality,
    load_capacity_plan,
    validate_capacity_study,
)
from paretopilot.domain import ValidationError
from paretopilot.io import sha256_json
from paretopilot.load_eval import (
    LoadRequest,
    build_load_evidence_binding,
    evaluate_load,
    load_load_plan,
)


ROOT = Path(__file__).parents[1]
CAPACITY_PLAN = ROOT / "configs" / "capacity.arm64.json"
LOAD_PLAN = ROOT / "configs" / "load.arm64.json"
SUITE_SHA256 = "e49c16fba32fd65c947264aef4141026ab68b1fd415ef09eeea6e8ade9a545c7"
RUN_ID = "123456"
RUN_ATTEMPT = 1


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _path_for(items: list[tuple[str, Path]], label: str) -> Path:
    return next(path for item_label, path in items if item_label == label)


def _refresh_embedded_load_source(artifact: dict[str, object], label: str) -> None:
    source = artifact["source_artifacts"]["load_evaluations"][label]
    source["content_sha256"] = sha256_json(source["evaluation"])


def _refresh_embedded_quality_source(artifact: dict[str, object], label: str) -> None:
    source = artifact["source_artifacts"]["quality_wrappers"][label]
    wrapper = source["wrapper"]
    wrapper["evaluation_content_sha256"] = sha256_json(wrapper["evaluation"])
    source["content_sha256"] = sha256_json(wrapper)


def _canonical_argv(candidate_id: str) -> list[str]:
    if candidate_id == "q8-generic":
        build = "generic"
        model = "qwen2.5-1.5b-instruct-q8_0.gguf"
        ubatch = "128"
        port = "18081"
    else:
        build = "kleidiai"
        model = "qwen2.5-1.5b-instruct-q4_0.gguf"
        ubatch = "512"
        port = "18084"
    return [
        f"./.candidate-build/{build}/bin/llama-server",
        "--model",
        f"./.candidate-models/{model}",
        "--threads",
        "4",
        "--threads-batch",
        "4",
        "--batch-size",
        "512",
        "--ubatch-size",
        ubatch,
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
        port,
    ]


def _capacity_argv(candidate_id: str, parallel: int, port: int) -> list[str]:
    argv = _canonical_argv(candidate_id)
    argv[argv.index("--ctx-size") + 1] = str(2048 * parallel)
    argv[argv.index("--parallel") + 1] = str(parallel)
    argv[argv.index("--port") + 1] = str(port)
    return argv


def _quality_payload(candidate_id: str) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "synthetic": False,
        "suite": {
            "id": "paretopilot-qwen-behavior-v2",
            "license": "CC0-1.0",
            "sha256": SUITE_SHA256,
            "quality_case_count": 1,
            "performance_repetitions": 2,
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
                    "id": "ready",
                    "prompt": "Reply READY.",
                    "accepted_answers": ["READY"],
                    "response": "READY",
                    "matched": True,
                    "matched_answer": "READY",
                }
            ],
        },
        "latency": {
            "method": "single-client streamed HTTP requests",
            "ttft_ms_p50": 100.0,
            "ttft_ms_p95": 110.0,
            "e2e_latency_ms_p50": 500.0,
            "e2e_latency_ms_p95": 520.0,
            "samples": [
                {
                    "index": 1,
                    "ttft_ms": 90.0,
                    "e2e_latency_ms": 480.0,
                    "event_count": 64,
                    "predicted_tokens": 64,
                    "content": "sample 1",
                },
                {
                    "index": 2,
                    "ttft_ms": 110.0,
                    "e2e_latency_ms": 520.0,
                    "event_count": 64,
                    "predicted_tokens": 64,
                    "content": "sample 2",
                },
            ],
        },
    }


def _manifest() -> dict[str, object]:
    return {
        "schema_version": "1.1",
        "classification": "supplementary-capacity",
        "synthetic": False,
        "source": {
            "repository": "agrovr/ParetoPilot",
            "revision": "1" * 40,
            "workflow": ".github/workflows/capacity-study-arm64.yml",
            "run_id": RUN_ID,
            "run_attempt": RUN_ATTEMPT,
            "generated_at_utc": "2026-07-25T00:00:00Z",
        },
        "runner": {
            "os": "Ubuntu 24.04",
            "architecture": "arm64",
            "cpu": "Neoverse-N2",
            "cpu_count": 4,
        },
        "runtime": {
            "name": "llama.cpp",
            "repository": "https://github.com/ggml-org/llama.cpp",
            "revision": "2" * 40,
        },
        "optimization_library": {
            "name": "KleidiAI",
            "repository": "https://github.com/ARM-software/kleidiai",
            "version": "v1.24.0",
            "source_archive_sha256": "8" * 64,
            "size_bytes": 2_048_000,
        },
        "toolchain": {
            "gcc_version_sha256": "9" * 64,
            "gxx_version_sha256": "a" * 64,
            "cmake_version_sha256": "b" * 64,
            "ninja_version_sha256": "c" * 64,
        },
        "candidates": {
            "q8-generic": {
                "model": {
                    "family": "Qwen2.5-1.5B-Instruct",
                    "repository": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                    "revision": "3" * 40,
                    "filename": "qwen2.5-1.5b-instruct-q8_0.gguf",
                    "sha256": "4" * 64,
                    "size_bytes": 1_894_532_128,
                },
                "build": {
                    "label": "generic",
                    "server_binary_sha256": "5" * 64,
                    "kleidiai_enabled": False,
                    "cmake_cache_sha256": "d" * 64,
                    "configure_log_sha256": "e" * 64,
                    "compile_log_sha256": "f" * 64,
                },
            },
            "q4-kleidiai-tuned": {
                "model": {
                    "family": "Qwen2.5-1.5B-Instruct",
                    "repository": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                    "revision": "3" * 40,
                    "filename": "qwen2.5-1.5b-instruct-q4_0.gguf",
                    "sha256": "6" * 64,
                    "size_bytes": 1_066_227_232,
                },
                "build": {
                    "label": "kleidiai",
                    "server_binary_sha256": "7" * 64,
                    "kleidiai_enabled": True,
                    "cmake_cache_sha256": "0" * 64,
                    "configure_log_sha256": "1a" * 32,
                    "compile_log_sha256": "2b" * 32,
                },
            },
        },
        "canonical_commands": {
            "q8-generic": {
                "sha256": ("8b5040317a38b52930ab87494bb4d02aaccf696ebe7d0826df33816a3045d74f"),
                "argv": _canonical_argv("q8-generic"),
            },
            "q4-kleidiai-tuned": {
                "sha256": ("4e09666b2e6542694e0a96f3e20f27697a7f9202155f7b9c80658fd51ccc3487"),
                "argv": _canonical_argv("q4-kleidiai-tuned"),
            },
        },
        "canonical_evidence": {
            "lock_sha256": ("9a00187cb4619daec3596139c97de49127841ceb3c2c7edd85092df2474c578d"),
            "release_sha256": ("b5586878ccd214667911390f417db0417111ac2c31d163a2f5f55c4469aefeb2"),
            "run_id": "30055662526",
            "release_tag": "v1.1.0",
        },
    }


class CapacityFixture:
    def __init__(
        self,
        root: Path,
        *,
        command_mutation=None,
        reverse_scale: float = 1.05,
        duration_overrides: dict[tuple[int, int], float] | None = None,
    ) -> None:
        self.root = root
        self.plan = load_capacity_plan(CAPACITY_PLAN)
        self.load_plan = load_load_plan(LOAD_PLAN)
        self.manifest = _manifest()
        self.manifest_path = root / "manifest.json"
        _write_json(self.manifest_path, self.manifest)
        self.loads: list[tuple[str, Path]] = []
        self.rss: list[tuple[str, Path]] = []
        self.logs: list[tuple[str, Path]] = []
        self.quality: list[tuple[str, Path]] = []
        self.quality_evaluations: dict[str, Path] = {}
        self.quality_commands: dict[str, Path] = {}
        self.quality_base_urls: dict[str, str] = {}
        self.quality_wrappers: dict[str, Path] = {}

        port = 19000
        for pass_spec in self.plan.passes:
            for candidate in self.plan.candidates:
                for parallel in self.plan.server_parallel_levels:
                    port += 1
                    label = f"{pass_spec.id}/{candidate.id}/p{parallel}"
                    directory = root / "runs" / pass_spec.id / candidate.id / f"p{parallel}"
                    command_path = directory / "server-command.json"
                    argv = _capacity_argv(candidate.id, parallel, port)
                    if command_mutation is not None:
                        argv = command_mutation(label, argv)
                    _write_json(
                        command_path,
                        {"schema_version": "1.0", "argv": argv},
                    )
                    binding = build_load_evidence_binding(
                        base_url=f"http://127.0.0.1:{port}",
                        plan_path=LOAD_PLAN,
                        server_command_path=command_path,
                        canonical_server_command_path=command_path,
                    )
                    pass_scale = 1.0 if pass_spec.id == "forward" else reverse_scale

                    def runner(
                        request: LoadRequest,
                        *,
                        server_parallel=parallel,
                        pass_scale=pass_scale,
                        candidate_id=candidate.id,
                    ) -> dict[str, object]:
                        concurrency = request.concurrency
                        base_duration = (duration_overrides or {}).get(
                            (server_parallel, concurrency),
                            {
                                (1, 1): 0.9,
                                (1, 2): 2.2,
                                (1, 4): 7.0,
                                (2, 1): 0.95,
                                (2, 2): 1.35,
                                (2, 4): 4.0,
                                (4, 1): 1.0,
                                (4, 2): 1.2,
                                (4, 4): 2.6,
                            }[(server_parallel, concurrency)],
                        )
                        candidate_scale = 0.92 if candidate_id == "q4-kleidiai-tuned" else 1.0
                        duration = base_duration * pass_scale * candidate_scale
                        wave = (request.request_index - 1) // concurrency
                        started = wave * (duration + 0.01)
                        return {
                            "completed": True,
                            "ttft_ms": 350.0 + concurrency * 100 + server_parallel * 25,
                            "e2e_latency_ms": duration * 1000.0,
                            "generated_tokens": request.output_tokens,
                            "error": None,
                            "started_at_seconds": started,
                            "finished_at_seconds": started + duration,
                        }

                    load_payload = evaluate_load(
                        candidate_id=candidate.id,
                        prompts=self.load_plan.prompts,
                        output_tokens=self.load_plan.output_tokens,
                        warmup_requests_per_level=self.load_plan.warmup_requests_per_level,
                        measured_requests_per_level=self.load_plan.measured_requests_per_level,
                        request_runner=runner,
                        slo=self.load_plan.slo,
                        concurrency_levels=self.load_plan.concurrency_levels,
                        execution_order=pass_spec.client_concurrency_order,
                        synthetic=False,
                        evidence_binding=binding,
                    )
                    load_path = directory / "load-evaluation.json"
                    _write_json(load_path, load_payload)
                    self.loads.append((label, load_path))

                    rss_mib = (
                        (2200.0 if candidate.id == "q8-generic" else 1500.0)
                        + (parallel - 1) * 240.0
                        + (25.0 if pass_spec.id == "reverse" else 0.0)
                    )
                    rss_path = directory / "server-time.txt"
                    rss_path.write_text(
                        "Command being timed: llama-server\n"
                        f"Maximum resident set size (kbytes): {int(rss_mib * 1024)}\n",
                        encoding="utf-8",
                    )
                    self.rss.append((label, rss_path))

                    log_path = directory / "server.stderr.log"
                    log_path.write_text(
                        (
                            "CPU_KLEIDIAI model buffer\n"
                            if candidate.kleidiai_expected
                            else "generic CPU backend\n"
                        ),
                        encoding="utf-8",
                    )
                    self.logs.append((label, log_path))

        quality_root = root / "quality"
        for candidate in self.plan.candidates:
            for parallel in self.plan.server_parallel_levels:
                port += 1
                label = f"{candidate.id}/p{parallel}"
                directory = quality_root / candidate.id / f"p{parallel}"
                evaluation_path = directory / "server-evaluation.json"
                command_path = directory / "server-command.json"
                wrapper_path = directory / "quality-evidence.json"
                base_url = f"http://127.0.0.1:{port}"
                _write_json(evaluation_path, _quality_payload(candidate.id))
                _write_json(
                    command_path,
                    {
                        "schema_version": "1.0",
                        "argv": _capacity_argv(candidate.id, parallel, port),
                    },
                )
                wrapper = bind_capacity_quality(
                    evaluation_path=evaluation_path,
                    server_command_path=command_path,
                    base_url=base_url,
                    pass_id="quality",
                    candidate_id=candidate.id,
                    server_parallel=parallel,
                    run_id=RUN_ID,
                    run_attempt=RUN_ATTEMPT,
                )
                _write_json(wrapper_path, wrapper)
                self.quality.append((label, wrapper_path))
                self.quality_evaluations[label] = evaluation_path
                self.quality_commands[label] = command_path
                self.quality_base_urls[label] = base_url
                self.quality_wrappers[label] = wrapper_path

    def rebind_quality(
        self,
        label: str,
        *,
        base_url: str | None = None,
        run_id: str = RUN_ID,
        run_attempt: int = RUN_ATTEMPT,
    ) -> None:
        candidate_id, parallel_text = label.split("/p", maxsplit=1)
        wrapper = bind_capacity_quality(
            evaluation_path=self.quality_evaluations[label],
            server_command_path=self.quality_commands[label],
            base_url=base_url or self.quality_base_urls[label],
            pass_id="quality",
            candidate_id=candidate_id,
            server_parallel=int(parallel_text),
            run_id=run_id,
            run_attempt=run_attempt,
        )
        _write_json(self.quality_wrappers[label], wrapper)

    def assemble(self) -> dict[str, object]:
        return dict(
            assemble_capacity_study(
                plan_path=CAPACITY_PLAN,
                load_plan_path=LOAD_PLAN,
                manifest_path=self.manifest_path,
                load_artifacts=self.loads,
                rss_artifacts=self.rss,
                server_logs=self.logs,
                quality_artifacts=self.quality,
            )
        )


class CapacityEvaluationTests(unittest.TestCase):
    def test_bundled_plan_is_strict_bounded_and_counterbalanced(self) -> None:
        plan = load_capacity_plan(CAPACITY_PLAN)

        self.assertEqual(plan.server_parallel_levels, (1, 2, 4))
        self.assertEqual(plan.client_concurrency_levels, (1, 2, 4))
        self.assertEqual(plan.per_slot_context_tokens, 2048)
        self.assertEqual(len(plan.candidates), 2)
        self.assertEqual(
            plan.passes[1].candidate_order, tuple(reversed(plan.passes[0].candidate_order))
        )
        self.assertEqual(
            plan.passes[1].server_parallel_order,
            tuple(reversed(plan.passes[0].server_parallel_order)),
        )
        self.assertEqual(
            plan.passes[1].client_concurrency_order,
            tuple(reversed(plan.passes[0].client_concurrency_order)),
        )

        with TemporaryDirectory() as directory:
            raw = json.loads(CAPACITY_PLAN.read_text(encoding="utf-8"))
            raw["surprise"] = True
            path = Path(directory) / "plan.json"
            _write_json(path, raw)
            with self.assertRaisesRegex(ValidationError, "unknown fields"):
                load_capacity_plan(path)

    def test_assembles_two_candidate_capacity_envelopes_and_recomputes_selection(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = CapacityFixture(Path(directory))
            artifact = fixture.assemble()

            validate_capacity_study(artifact)
            self.assertEqual(len(artifact["cells"]), 18)
            self.assertEqual(len(artifact["server_configurations"]), 12)
            self.assertEqual(len(artifact["quality_checks"]), 6)
            self.assertEqual(
                len(artifact["source_artifacts"]["load_evaluations"]),
                12,
            )
            self.assertEqual(
                len(artifact["source_artifacts"]["quality_wrappers"]),
                6,
            )
            reverse = artifact["source_artifacts"]["load_evaluations"]["reverse/q8-generic/p4"][
                "evaluation"
            ]
            self.assertEqual(reverse["schema_version"], "1.1")
            self.assertEqual(reverse["execution_order"], [4, 2, 1])
            self.assertFalse(artifact["canonical_outputs_modified"])
            for selection in artifact["selections"]:
                self.assertEqual(
                    selection["selected_cell"],
                    {
                        "server_parallel": 4,
                        "client_concurrency": 2,
                        "generated_tokens_per_second_median": (
                            selection["selected_cell"]["generated_tokens_per_second_median"]
                        ),
                    },
                )
                self.assertGreater(
                    selection["comparison_to_reference_percent"][
                        "generated_tokens_per_second_median"
                    ],
                    0.0,
                )

            tampered = deepcopy(artifact)
            tampered["cells"][0]["summary"]["generated_tokens_per_second_median"] += 1.0
            with self.assertRaisesRegex(ValidationError, "cells do not match embedded load"):
                validate_capacity_study(tampered)

    def test_rejects_material_command_drift_and_context_confounds(self) -> None:
        scenarios = {
            "threads": (
                lambda label, argv: [
                    "8" if index == argv.index("--threads") + 1 and label.endswith("/p2") else value
                    for index, value in enumerate(argv)
                ],
                "materially differs",
            ),
            "context": (
                lambda label, argv: [
                    (
                        "2048"
                        if index == argv.index("--ctx-size") + 1 and label.endswith("/p4")
                        else value
                    )
                    for index, value in enumerate(argv)
                ],
                "preserve the planned per-slot context",
            ),
        }
        for name, (mutation, message) in scenarios.items():
            with self.subTest(name=name), TemporaryDirectory() as directory:
                fixture = CapacityFixture(Path(directory), command_mutation=mutation)
                with self.assertRaisesRegex(ValidationError, message):
                    fixture.assemble()

    def test_tampered_pass_aggregate_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = CapacityFixture(Path(directory))
            load_label, load_path = next(
                item for item in fixture.loads if item[0] == "reverse/q8-generic/p4"
            )
            raw = json.loads(load_path.read_text(encoding="utf-8"))
            row = next(item for item in raw["rows"] if item["concurrency"] == 4)
            row["slo_met"] = False
            row["slo_failures"] = ["e2e_latency_ms_p95_above_maximum"]
            _write_json(load_path, raw)
            self.assertEqual(load_label, "reverse/q8-generic/p4")

            with self.assertRaisesRegex(ValidationError, "aggregate fields"):
                fixture.assemble()

    def test_embedded_load_slo_and_quality_are_revalidated_and_recomputed(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = CapacityFixture(Path(directory)).assemble()

            stale_load = deepcopy(artifact)
            load_label = "forward/q8-generic/p4"
            load_evaluation = stale_load["source_artifacts"]["load_evaluations"][load_label][
                "evaluation"
            ]
            sample = load_evaluation["rows"][0]["samples"][0]
            sample["e2e_latency_ms"] += 250.0
            sample["finished_at_seconds"] += 0.25
            _refresh_embedded_load_source(stale_load, load_label)
            with self.assertRaisesRegex(ValidationError, "aggregate fields"):
                validate_capacity_study(stale_load)

            stale_slo = deepcopy(artifact)
            load_evaluation = stale_slo["source_artifacts"]["load_evaluations"][load_label][
                "evaluation"
            ]
            load_evaluation["slo"]["max_e2e_latency_ms_p95"] = 6400.0
            _refresh_embedded_load_source(stale_slo, load_label)
            with self.assertRaisesRegex(ValidationError, "SLO does not match the plan"):
                validate_capacity_study(stale_slo)

            stale_quality = deepcopy(artifact)
            quality_label = "q4-kleidiai-tuned/p2"
            evaluation = stale_quality["source_artifacts"]["quality_wrappers"][quality_label][
                "wrapper"
            ]["evaluation"]
            evaluation["quality"]["score"] = 0.0
            evaluation["quality"]["passed"] = 0
            evaluation["quality"]["cases"][0]["response"] = "NO"
            evaluation["quality"]["cases"][0]["matched"] = False
            evaluation["quality"]["cases"][0]["matched_answer"] = None
            _refresh_embedded_quality_source(stale_quality, quality_label)
            with self.assertRaisesRegex(
                ValidationError,
                "quality_checks do not match embedded quality sources",
            ):
                validate_capacity_study(stale_quality)

    def test_concurrency_rows_cannot_be_swapped_under_stale_cells(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = CapacityFixture(Path(directory)).assemble()
            label = "forward/q8-generic/p4"
            evaluation = artifact["source_artifacts"]["load_evaluations"][label]["evaluation"]
            first = deepcopy(evaluation["rows"][0])
            second = deepcopy(evaluation["rows"][1])
            evaluation["rows"][0] = second
            evaluation["rows"][0]["concurrency"] = 1
            evaluation["rows"][1] = first
            evaluation["rows"][1]["concurrency"] = 2
            _refresh_embedded_load_source(artifact, label)
            configuration = next(
                value
                for value in artifact["server_configurations"]
                if value["source_load_label"] == label
            )
            configuration["load_evaluation_content_sha256"] = artifact["source_artifacts"][
                "load_evaluations"
            ][label]["content_sha256"]

            with self.assertRaisesRegex(
                ValidationError,
                "cells do not match embedded load sources",
            ):
                validate_capacity_study(artifact)

    def test_canonical_command_argv_and_content_hash_drift_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = CapacityFixture(Path(directory))

            argv_drift = deepcopy(fixture.manifest)
            argv = argv_drift["canonical_commands"]["q8-generic"]["argv"]
            argv[argv.index("--threads") + 1] = "8"
            _write_json(fixture.manifest_path, argv_drift)
            with self.assertRaisesRegex(
                ValidationError,
                "does not match the canonical JSON command",
            ):
                fixture.assemble()

            hash_drift = deepcopy(fixture.manifest)
            hash_drift["canonical_commands"]["q8-generic"]["sha256"] = "0" * 64
            _write_json(fixture.manifest_path, hash_drift)
            with self.assertRaisesRegex(ValidationError, "sha256 does not match the plan"):
                fixture.assemble()

    def test_fingerprint_source_configuration_and_provenance_crosslinks(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = CapacityFixture(Path(directory)).assemble()
            load_label = "forward/q8-generic/p1"
            quality_label = "q8-generic/p1"

            scenarios: list[tuple[str, dict[str, object], str]] = []

            load_fingerprint = deepcopy(artifact)
            load_fingerprint["input_fingerprints"]["load_artifacts"][load_label] = "0" * 64
            scenarios.append(
                ("load fingerprint", load_fingerprint, "input_sha256 is not cross-linked")
            )

            quality_fingerprint = deepcopy(artifact)
            quality_fingerprint["input_fingerprints"]["quality_artifacts"][quality_label] = "0" * 64
            scenarios.append(
                (
                    "quality fingerprint",
                    quality_fingerprint,
                    "input_sha256 is not cross-linked",
                )
            )

            rss_fingerprint = deepcopy(artifact)
            rss_fingerprint["input_fingerprints"]["rss_artifacts"][load_label] = "0" * 64
            scenarios.append(("RSS fingerprint", rss_fingerprint, "rss_sha256 is not cross-linked"))

            log_fingerprint = deepcopy(artifact)
            log_fingerprint["input_fingerprints"]["server_logs"][load_label] = "0" * 64
            scenarios.append(
                (
                    "log fingerprint",
                    log_fingerprint,
                    "server_log_sha256 is not cross-linked",
                )
            )

            source_drift = deepcopy(artifact)
            source_drift["source_artifacts"]["load_evaluations"][load_label]["input_sha256"] = (
                "0" * 64
            )
            source_drift["input_fingerprints"]["load_artifacts"][load_label] = "0" * 64
            scenarios.append(
                (
                    "source to configuration",
                    source_drift,
                    "load_evaluation_sha256 is not cross-linked",
                )
            )

            configuration_drift = deepcopy(artifact)
            configuration = next(
                value
                for value in configuration_drift["server_configurations"]
                if value["source_load_label"] == load_label
            )
            configuration["load_evaluation_sha256"] = "0" * 64
            scenarios.append(
                (
                    "configuration",
                    configuration_drift,
                    "load_evaluation_sha256 is not cross-linked",
                )
            )

            provenance_drift = deepcopy(artifact)
            provenance_drift["provenance"]["toolchain"]["gcc_version_sha256"] = "0" * 64
            scenarios.append(
                (
                    "provenance",
                    provenance_drift,
                    "provenance content fingerprint is not recomputable",
                )
            )

            for name, tampered, message in scenarios:
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(
                        ValidationError,
                        message,
                    ),
                ):
                    validate_capacity_study(tampered)

    def test_duplicate_resolved_paths_and_load_identities_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = CapacityFixture(Path(directory))
            duplicate_label = fixture.loads[1][0]
            fixture.loads[1] = (duplicate_label, fixture.loads[0][1])
            with self.assertRaisesRegex(ValidationError, "resolve to the same path"):
                fixture.assemble()

        with TemporaryDirectory() as directory:
            fixture = CapacityFixture(Path(directory))
            forward_path = _path_for(fixture.loads, "forward/q8-generic/p1")
            reverse_path = _path_for(fixture.loads, "reverse/q8-generic/p1")
            forward = json.loads(forward_path.read_text(encoding="utf-8"))
            reverse = json.loads(reverse_path.read_text(encoding="utf-8"))
            reverse["evidence_binding"] = deepcopy(forward["evidence_binding"])
            _write_json(reverse_path, reverse)
            with self.assertRaisesRegex(ValidationError, "duplicate capacity load identity"):
                fixture.assemble()

    def test_quality_binding_rejects_url_port_and_workflow_identity_mismatches(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = CapacityFixture(Path(directory))
            label = "q8-generic/p1"

            with self.assertRaisesRegex(ValidationError, "port must match"):
                fixture.rebind_quality(
                    label,
                    base_url="http://127.0.0.1:65535",
                )

            for name, run_id, run_attempt in (
                ("run id", "999999", RUN_ATTEMPT),
                ("run attempt", RUN_ID, RUN_ATTEMPT + 1),
            ):
                with self.subTest(name=name):
                    fixture.rebind_quality(
                        label,
                        run_id=run_id,
                        run_attempt=run_attempt,
                    )
                    with self.assertRaisesRegex(
                        ValidationError,
                        "workflow run identity does not match provenance",
                    ):
                        fixture.assemble()

    def test_reverse_pass_requires_its_predeclared_execution_order(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = CapacityFixture(Path(directory))
            path = _path_for(fixture.loads, "reverse/q8-generic/p4")
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["schema_version"], "1.1")
            self.assertEqual(raw["execution_order"], [4, 2, 1])
            raw["execution_order"] = [2, 4, 1]
            _write_json(path, raw)

            with self.assertRaisesRegex(ValidationError, "execution_order does not match"):
                fixture.assemble()

    def test_counterbalanced_spread_gates_reject_unstable_cells(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = CapacityFixture(Path(directory), reverse_scale=1.35).assemble()
            cell = next(
                value
                for value in artifact["cells"]
                if value["candidate_id"] == "q8-generic"
                and value["server_parallel"] == 4
                and value["client_concurrency"] == 2
            )
            failures = cell["summary"]["failure_reasons"]
            self.assertIn("throughput_relative_spread_above_maximum", failures)
            self.assertIn("e2e_relative_spread_above_maximum", failures)
            self.assertFalse(cell["summary"]["capacity_gate_met"])

    def test_one_percent_objective_tolerance_uses_latency_tie_breaker(self) -> None:
        durations = {
            (1, 1): 2.5,
            (1, 2): 5.5,
            (1, 4): 6.0,
            (2, 1): 2.5,
            (2, 2): 5.5,
            (2, 4): 6.0,
            (4, 1): 1.0,
            (4, 2): 2.0,
            (4, 4): 5.5,
        }
        with TemporaryDirectory() as directory:
            artifact = CapacityFixture(
                Path(directory),
                duration_overrides=durations,
            ).assemble()

            for selection in artifact["selections"]:
                self.assertEqual(selection["objective_tolerance_percent"], 1.0)
                self.assertEqual(selection["within_tolerance_cell_count"], 2)
                self.assertEqual(
                    {
                        "server_parallel": selection["selected_cell"]["server_parallel"],
                        "client_concurrency": selection["selected_cell"]["client_concurrency"],
                    },
                    {"server_parallel": 4, "client_concurrency": 1},
                )
                self.assertGreater(
                    selection["numeric_best_generated_tokens_per_second_median"],
                    selection["selected_cell"]["generated_tokens_per_second_median"],
                )

    def test_rss_limit_removes_a_parallel_level_from_the_envelope(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = CapacityFixture(Path(directory))
            for label, path in fixture.rss:
                if label.endswith("q4-kleidiai-tuned/p4"):
                    path.write_text(
                        "Maximum resident set size (kbytes): 5242880\n",
                        encoding="utf-8",
                    )

            artifact = fixture.assemble()
            affected = [
                cell
                for cell in artifact["cells"]
                if cell["candidate_id"] == "q4-kleidiai-tuned" and cell["server_parallel"] == 4
            ]
            self.assertTrue(affected)
            self.assertTrue(
                all(
                    "server_peak_rss_above_maximum" in cell["summary"]["failure_reasons"]
                    for cell in affected
                )
            )

    def test_quality_regression_fails_every_cell_for_that_server_parallel(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = CapacityFixture(Path(directory))
            label = "q4-kleidiai-tuned/p2"
            path = fixture.quality_evaluations[label]
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["quality"]["score"] = 0.0
            raw["quality"]["passed"] = 0
            raw["quality"]["cases"][0]["response"] = "NO"
            raw["quality"]["cases"][0]["matched"] = False
            raw["quality"]["cases"][0]["matched_answer"] = None
            _write_json(path, raw)
            fixture.rebind_quality(label)

            artifact = fixture.assemble()
            affected = [
                cell
                for cell in artifact["cells"]
                if cell["candidate_id"] == "q4-kleidiai-tuned" and cell["server_parallel"] == 2
            ]
            self.assertTrue(affected)
            self.assertTrue(
                all(
                    "quality_gate_failed" in cell["summary"]["failure_reasons"] for cell in affected
                )
            )


if __name__ == "__main__":
    unittest.main()
