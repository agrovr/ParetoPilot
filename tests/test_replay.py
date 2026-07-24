from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from paretopilot import cli
from paretopilot.analysis import recommend
from paretopilot.domain import BenchmarkSet, Constraints, ValidationError
from paretopilot.io import sha256_file
from paretopilot.load_eval import (
    LoadRequest,
    build_load_evidence_binding,
    combine_load_evaluations,
    evaluate_load,
)
from paretopilot.profiles import evaluate_policy_profiles, load_policy_set
from paretopilot.replay import replay_evidence
from paretopilot.report_v11 import render_report_v11
from paretopilot.stability import summarize_stability


def _benchmark_mapping() -> dict[str, object]:
    values = (
        ("q8-generic", 100.0, 20.0, 4000.0, 50.0, 100.0),
        ("q4-generic", 99.8, 18.0, 2000.0, 44.0, 110.0),
        ("q4-kleidiai", 102.0, 17.8, 2001.0, 40.0, 111.0),
        ("q4-kleidiai-tuned", 100.2, 18.2, 2001.0, 40.2, 130.0),
    )
    return {
        "schema_version": "1.0",
        "baseline_id": "q8-generic",
        "synthetic": False,
        "metadata": {"classification": "canonical"},
        "candidates": [
            {
                "id": candidate_id,
                "label": candidate_id,
                "parameters": {},
                "metrics": {
                    "quality_score": 1.0,
                    "e2e_latency_ms_p95": e2e,
                    "generation_tps": generation,
                    "peak_rss_mib": memory,
                    "model_size_mib": memory / 2.0,
                    "ttft_ms_p95": ttft,
                    "prompt_tps": prompt,
                },
            }
            for candidate_id, e2e, generation, memory, ttft, prompt in values
        ],
    }


def _constraints_mapping() -> dict[str, object]:
    return {
        "min_quality_retention": 1.0,
        "quality_metric": "quality_score",
        "max_values": {
            "e2e_latency_ms_p95": 150.0,
            "peak_rss_mib": 5000.0,
        },
        "min_values": {},
        "objective": {
            "metric": "e2e_latency_ms_p95",
            "direction": "min",
        },
        "objective_tolerance_percent": 1.0,
        "preference_order": [
            "q8-generic",
            "q4-generic",
            "q4-kleidiai",
            "q4-kleidiai-tuned",
        ],
        "frontier_metrics": {
            "e2e_latency_ms_p95": "min",
            "generation_tps": "max",
            "model_size_mib": "min",
            "peak_rss_mib": "min",
            "quality_score": "max",
            "ttft_ms_p95": "min",
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _rewrite_checksums(root: Path) -> None:
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{path.relative_to(root).as_posix()}"
        for path in paths
    ]
    (root / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _evidence(root: Path) -> Path:
    evidence = root / "evidence"
    _write_json(
        evidence / "status.json",
        {
            "schema_version": "1.0",
            "classification": "canonical",
            "status": "complete",
            "measurement_valid": True,
            "valid_evidence": True,
            "reason": "all replay gates passed",
        },
    )
    _write_json(
        evidence / "experiment" / "manifest.json",
        {
            "schema_version": "1.0",
            "classification": "canonical",
            "test_fixture": True,
        },
    )
    _write_json(
        evidence / "experiment" / "constraints.json",
        _constraints_mapping(),
    )
    benchmark_path = evidence / "experiment" / "benchmark-set.json"
    constraints_path = evidence / "experiment" / "constraints.json"
    _write_json(benchmark_path, _benchmark_mapping())
    recommendation = dict(
        recommend(
            BenchmarkSet.from_mapping(_benchmark_mapping()),
            Constraints.from_mapping(_constraints_mapping()),
        )
    )
    recommendation["input_fingerprints"] = {
        "benchmarks_sha256": sha256_file(benchmark_path),
        "constraints_sha256": sha256_file(constraints_path),
    }
    _write_json(
        evidence / "experiment" / "recommendation.json",
        recommendation,
    )
    _rewrite_checksums(evidence)
    return evidence


def _deployment_argv(candidate_id: str, *, port: int = 8080) -> list[str]:
    return [
        "./build/llama-server",
        "--model",
        f"./models/{candidate_id}.gguf",
        "--threads",
        "4",
        "--parallel",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def _v11_benchmark_mapping() -> dict[str, object]:
    mapping = deepcopy(_benchmark_mapping())
    for candidate in mapping["candidates"]:
        candidate_id = str(candidate["id"])
        candidate["parameters"] = {
            "configuration": {"candidate": candidate_id},
            "deployment_argv": _deployment_argv(candidate_id),
        }
    return mapping


def _policy_mapping() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "canonical_profile_id": "canonical-latency",
        "profiles": [
            {
                "id": "canonical-latency",
                "label": "Canonical latency",
                "description": "Use the predeclared latency objective.",
                "classification": "canonical",
                "objective": {
                    "metric": "e2e_latency_ms_p95",
                    "direction": "min",
                },
                "objective_tolerance_percent": 1.0,
                "preference_policy": "canonical",
            },
            {
                "id": "memory-first",
                "label": "Memory first",
                "description": "Prefer the lowest measured peak memory.",
                "classification": "derived-non-canonical",
                "objective": {
                    "metric": "peak_rss_mib",
                    "direction": "min",
                },
                "objective_tolerance_percent": 0.0,
                "preference_policy": "none",
            },
        ],
    }


def _load_plan_mapping() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "prompts": ["Explain one Arm64 deployment tradeoff."],
        "output_tokens": 8,
        "warmup_requests_per_level": 0,
        "measured_requests_per_level": 1,
        "concurrency_levels": [1],
        "timeout_seconds": 30.0,
        "slo": {
            "min_completion_rate": 1.0,
            "max_ttft_ms_p95": 50.0,
            "max_e2e_latency_ms_p95": 200.0,
        },
    }


def _load_runner(request: LoadRequest, *, duration_seconds: float = 0.1) -> dict[str, object]:
    started = 100.0 + request.request_index
    return {
        "completed": True,
        "ttft_ms": 10.0,
        "e2e_latency_ms": duration_seconds * 1000.0,
        "generated_tokens": request.output_tokens,
        "error": None,
        "started_at_seconds": started,
        "finished_at_seconds": started + duration_seconds,
    }


def _single_load_evaluation(
    evidence: Path,
    candidate_id: str,
    *,
    duration_seconds: float = 0.1,
) -> dict[str, object]:
    candidate_ids = [str(candidate["id"]) for candidate in _v11_benchmark_mapping()["candidates"]]
    port = 18081 + candidate_ids.index(candidate_id)
    plan_path = evidence / "extensions" / "load-plan.json"
    load_command = evidence / "extensions" / "load" / candidate_id / "server-command.json"
    canonical_command = (
        evidence / "experiment" / "candidates" / candidate_id / "server-command.json"
    )
    binding = build_load_evidence_binding(
        base_url=f"http://127.0.0.1:{port}",
        plan_path=plan_path,
        server_command_path=load_command,
        canonical_server_command_path=canonical_command,
    )
    return dict(
        evaluate_load(
            candidate_id=candidate_id,
            prompts=("Explain one Arm64 deployment tradeoff.",),
            output_tokens=8,
            warmup_requests_per_level=0,
            measured_requests_per_level=1,
            request_runner=lambda request: _load_runner(
                request,
                duration_seconds=duration_seconds,
            ),
            slo={
                "min_completion_rate": 1.0,
                "max_ttft_ms_p95": 50.0,
                "max_e2e_latency_ms_p95": 200.0,
            },
            concurrency_levels=(1,),
            peak_rss_mib_by_concurrency={1: 1000.0 + candidate_ids.index(candidate_id)},
            synthetic=False,
            evidence_binding=binding,
        )
    )


def _pass_mapping(multiplier: float) -> dict[str, object]:
    mapping = _v11_benchmark_mapping()
    mapping["metadata"] = {"classification": "supplementary-repeat-pass"}
    for candidate in mapping["candidates"]:
        candidate["metrics"]["e2e_latency_ms_p95"] *= multiplier
        candidate["metrics"]["generation_tps"] *= multiplier
    return mapping


def _v11_evidence(root: Path) -> Path:
    evidence = root / "evidence"
    benchmark_mapping = _v11_benchmark_mapping()
    constraints_mapping = _constraints_mapping()
    _write_json(
        evidence / "status.json",
        {
            "schema_version": "1.0",
            "classification": "canonical",
            "status": "complete",
            "measurement_valid": True,
            "valid_evidence": True,
            "reason": "all replay gates passed",
        },
    )
    _write_json(
        evidence / "experiment" / "manifest.json",
        {
            "schema_version": "1.0",
            "classification": "canonical",
            "test_fixture": True,
            "evaluation_suite_path": "evaluation-suite.json",
        },
    )
    benchmark_path = evidence / "experiment" / "benchmark-set.json"
    constraints_path = evidence / "experiment" / "constraints.json"
    _write_json(benchmark_path, benchmark_mapping)
    _write_json(constraints_path, constraints_mapping)
    evaluation_suite = {
        "schema_version": "1.0",
        "suite_id": "test-replay-suite",
    }
    _write_json(
        evidence / "experiment" / "evaluation-suite.json",
        evaluation_suite,
    )
    _write_json(
        evidence / "extensions" / "evaluation-suite.json",
        evaluation_suite,
    )

    recommendation = dict(
        recommend(
            BenchmarkSet.from_mapping(benchmark_mapping),
            Constraints.from_mapping(constraints_mapping),
        )
    )
    recommendation["input_fingerprints"] = {
        "benchmarks_sha256": sha256_file(benchmark_path),
        "constraints_sha256": sha256_file(constraints_path),
    }
    recommendation_path = evidence / "experiment" / "recommendation.json"
    _write_json(recommendation_path, recommendation)

    policy_path = evidence / "extensions" / "policy-config.json"
    _write_json(policy_path, _policy_mapping())
    profiles = dict(
        evaluate_policy_profiles(
            BenchmarkSet.from_mapping(benchmark_mapping),
            Constraints.from_mapping(constraints_mapping),
            load_policy_set(policy_path),
        )
    )
    profiles["input_fingerprints"] = {
        "benchmarks_sha256": sha256_file(benchmark_path),
        "constraints_sha256": sha256_file(constraints_path),
        "policies_sha256": sha256_file(policy_path),
    }
    profiles_path = evidence / "extensions" / "policy-profiles.json"
    _write_json(profiles_path, profiles)

    load_plan_path = evidence / "extensions" / "load-plan.json"
    _write_json(load_plan_path, _load_plan_mapping())
    candidate_ids = [str(candidate["id"]) for candidate in benchmark_mapping["candidates"]]
    for index, candidate_id in enumerate(candidate_ids):
        _write_json(
            evidence / "experiment" / "candidates" / candidate_id / "server-command.json",
            {
                "schema_version": "1.0",
                "argv": _deployment_argv(candidate_id),
            },
        )
        _write_json(
            evidence / "extensions" / "load" / candidate_id / "server-command.json",
            {
                "schema_version": "1.0",
                "argv": _deployment_argv(candidate_id, port=18081 + index),
            },
        )
    single_loads: list[dict[str, object]] = []
    for candidate_id in candidate_ids:
        single = _single_load_evaluation(evidence, candidate_id)
        single_loads.append(single)
        _write_json(
            evidence / "extensions" / "load" / candidate_id / "load-evaluation.json",
            single,
        )
    combined_load = dict(
        combine_load_evaluations(
            single_loads,
            require_evidence_bindings=True,
        )
    )
    combined_load_path = evidence / "extensions" / "load-evaluation.json"
    _write_json(combined_load_path, combined_load)

    pass_paths: list[Path] = []
    pass_sets: list[BenchmarkSet] = []
    for index, multiplier in enumerate((1.0, 1.01), start=1):
        pass_path = evidence / "extensions" / f"benchmark-set-pass-{index}.json"
        pass_mapping = _pass_mapping(multiplier)
        _write_json(pass_path, pass_mapping)
        pass_paths.append(pass_path)
        pass_sets.append(BenchmarkSet.from_mapping(pass_mapping))
    stability = dict(
        summarize_stability(
            pass_sets,
            metric_directions={
                "e2e_latency_ms_p95": "min",
                "generation_tps": "max",
            },
            pass_labels=("pass-1", "pass-2"),
            input_fingerprints={
                "pass-1": sha256_file(pass_paths[0]),
                "pass-2": sha256_file(pass_paths[1]),
            },
        )
    )
    stability_path = evidence / "extensions" / "repeat-stability.json"
    _write_json(stability_path, stability)

    report_v11 = render_report_v11(
        BenchmarkSet.from_mapping(benchmark_mapping),
        recommendation,
        policy_profiles=profiles,
        load_sweep=combined_load,
        stability_summary=stability,
        benchmarks_sha256=sha256_file(benchmark_path),
        recommendation_sha256=sha256_file(recommendation_path),
        profiles_sha256=sha256_file(profiles_path),
        load_sha256=sha256_file(combined_load_path),
        stability_sha256=sha256_file(stability_path),
    )
    (evidence / "report-v1.1.html").write_text(
        report_v11,
        encoding="utf-8",
        newline="\n",
    )
    _rewrite_checksums(evidence)
    return evidence


def _replay_v11(evidence: Path, output: Path) -> dict[str, object]:
    with (
        patch(
            "paretopilot.replay.assemble_experiment",
            return_value=_v11_benchmark_mapping(),
        ),
        patch(
            "paretopilot.replay.assemble_repeat_pass",
            side_effect=lambda _experiment, *, pass_number, benchmark_mapping: _pass_mapping(
                1.0 if pass_number == 1 else 1.01
            ),
        ),
    ):
        return dict(replay_evidence(evidence, output))


class ReplayTests(unittest.TestCase):
    def test_v11_replay_is_complete_deterministic_and_fully_reproduced(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _v11_evidence(root)

            first = _replay_v11(evidence, root / "first")
            _replay_v11(evidence, root / "second")

            self.assertEqual(first["replay_contract"], "1.1")
            self.assertTrue(first["valid"])
            self.assertTrue(first["decision_reproduced"])
            self.assertTrue(first["fully_reproduced"])
            self.assertEqual(first["differences"], [])
            self.assertEqual(first["warnings"], [])
            self.assertEqual(first["policy_profile_count"], 2)
            self.assertEqual(
                first["generated_files"],
                [
                    "benchmark-set.json",
                    "constraints.json",
                    "recommendation.json",
                    "report.html",
                    "policy-profiles.json",
                    "load-evaluation.json",
                    "benchmark-set-pass-1.json",
                    "benchmark-set-pass-2.json",
                    "repeat-stability.json",
                    "report-v1.1.html",
                ],
            )
            self.assertTrue(
                all(
                    comparison["matches"]
                    for comparison in first["authoritative_comparisons"].values()
                    if comparison["present"]
                )
            )
            for filename in first["generated_files"]:
                self.assertEqual(
                    (root / "first" / filename).read_bytes(),
                    (root / "second" / filename).read_bytes(),
                    filename,
                )

    def test_v11_missing_and_partial_extensions_fail_before_replay(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _evidence(root / "partial")
            _write_json(
                evidence / "extensions" / "load-plan.json",
                _load_plan_mapping(),
            )
            _rewrite_checksums(evidence)

            with self.assertRaisesRegex(
                ValidationError,
                "v1.1 evidence is missing required checksummed files:.*"
                "extensions/policy-config.json.*report-v1.1.html",
            ):
                _replay_v11(evidence, root / "partial-output")
            self.assertFalse((root / "partial-output").exists())

            evidence = _v11_evidence(root / "missing")
            (evidence / "extensions" / "policy-profiles.json").unlink()
            _rewrite_checksums(evidence)

            with self.assertRaisesRegex(
                ValidationError,
                "v1.1 evidence is missing required checksummed files: "
                "extensions/policy-profiles.json",
            ):
                _replay_v11(evidence, root / "missing-output")
            self.assertFalse((root / "missing-output").exists())

    def test_v11_rejects_a_manifest_that_splits_the_evaluation_suite_path(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _v11_evidence(root)
            manifest_path = evidence / "experiment" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["evaluation_suite_path"] = "../extensions/evaluation-suite.json"
            _write_json(manifest_path, manifest)
            _rewrite_checksums(evidence)

            with self.assertRaisesRegex(
                ValidationError,
                "v1.1 experiment manifest evaluation_suite_path must be 'evaluation-suite.json'",
            ):
                _replay_v11(evidence, root / "output")
            self.assertFalse((root / "output").exists())

    def test_v11_detects_tampering_between_single_and_combined_load_artifacts(
        self,
    ) -> None:
        candidate_id = "q4-generic"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _v11_evidence(root / "single")
            alternate = _single_load_evaluation(
                evidence,
                candidate_id,
                duration_seconds=0.12,
            )
            _write_json(
                evidence / "extensions" / "load" / candidate_id / "load-evaluation.json",
                alternate,
            )
            _rewrite_checksums(evidence)

            single_tamper = _replay_v11(evidence, root / "single-output")

            self.assertFalse(single_tamper["valid"])
            self.assertFalse(single_tamper["decision_reproduced"])
            self.assertIn("load-evaluation", single_tamper["differences"])
            self.assertFalse(
                single_tamper["authoritative_comparisons"]["load-evaluation"]["matches"]
            )

            evidence = _v11_evidence(root / "combined")
            single_paths = {
                str(candidate["id"]): (
                    evidence / "extensions" / "load" / str(candidate["id"]) / "load-evaluation.json"
                )
                for candidate in _v11_benchmark_mapping()["candidates"]
            }
            singles = [
                (
                    alternate
                    if current_id == candidate_id
                    else json.loads(path.read_text(encoding="utf-8"))
                )
                for current_id, path in single_paths.items()
            ]
            tampered_combined = combine_load_evaluations(
                singles,
                require_evidence_bindings=True,
            )
            _write_json(
                evidence / "extensions" / "load-evaluation.json",
                tampered_combined,
            )
            _rewrite_checksums(evidence)

            combined_tamper = _replay_v11(evidence, root / "combined-output")

            self.assertFalse(combined_tamper["valid"])
            self.assertIn("load-evaluation", combined_tamper["differences"])
            self.assertFalse(
                combined_tamper["authoritative_comparisons"]["load-evaluation"]["matches"]
            )

    def test_v11_rejects_plan_and_server_command_hash_mismatches(self) -> None:
        cases = (
            (
                "plan",
                "v1.1 load evidence plan SHA-256 does not match load-plan.json",
            ),
            (
                "load-command",
                "v1.1 load command SHA-256 does not match for q4-generic",
            ),
            (
                "canonical-command",
                "v1.1 canonical server command SHA-256 does not match for q4-generic",
            ),
        )
        for label, message in cases:
            with self.subTest(label=label), TemporaryDirectory() as directory:
                root = Path(directory)
                evidence = _v11_evidence(root)
                if label == "plan":
                    plan = _load_plan_mapping()
                    plan["timeout_seconds"] = 31.0
                    _write_json(evidence / "extensions" / "load-plan.json", plan)
                else:
                    command = evidence / (
                        "extensions/load/q4-generic/server-command.json"
                        if label == "load-command"
                        else "experiment/candidates/q4-generic/server-command.json"
                    )
                    command.write_text(
                        command.read_text(encoding="utf-8") + "\n",
                        encoding="utf-8",
                        newline="\n",
                    )
                _rewrite_checksums(evidence)

                with self.assertRaisesRegex(ValidationError, message):
                    _replay_v11(evidence, root / "output")
                self.assertFalse((root / "output").exists())

    def test_v11_stability_pass_drift_is_a_core_replay_failure(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _v11_evidence(root)
            pass_path = evidence / "extensions" / "benchmark-set-pass-2.json"
            pass_mapping = json.loads(pass_path.read_text(encoding="utf-8"))
            pass_mapping["candidates"][0]["metrics"]["generation_tps"] += 0.5
            _write_json(pass_path, pass_mapping)
            _rewrite_checksums(evidence)

            payload = _replay_v11(evidence, root / "output")

            self.assertFalse(payload["valid"])
            self.assertFalse(payload["decision_reproduced"])
            self.assertIn("benchmark-set-pass-2", payload["differences"])
            self.assertFalse(
                payload["authoritative_comparisons"]["benchmark-set-pass-2"]["matches"]
            )

    def test_v11_rejects_a_supplied_policy_config_that_differs_from_archive(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _v11_evidence(root)
            external = root / "different-policies.json"
            policies = _policy_mapping()
            policies["profiles"][1]["description"] = "A different policy description."
            _write_json(external, policies)

            with (
                patch(
                    "paretopilot.replay.assemble_experiment",
                    return_value=_v11_benchmark_mapping(),
                ),
                self.assertRaisesRegex(
                    ValidationError,
                    "supplied policy configuration does not match v1.1 archived policy config",
                ),
            ):
                replay_evidence(
                    evidence,
                    root / "output",
                    policies_path=external,
                )
            self.assertFalse((root / "output").exists())

    def test_v11_report_only_drift_is_a_non_core_warning(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _v11_evidence(root)
            (evidence / "report-v1.1.html").write_text(
                "<!doctype html><title>Archived presentation drift</title>\n",
                encoding="utf-8",
                newline="\n",
            )
            _rewrite_checksums(evidence)

            payload = _replay_v11(evidence, root / "output")

            self.assertTrue(payload["valid"])
            self.assertTrue(payload["decision_reproduced"])
            self.assertFalse(payload["fully_reproduced"])
            self.assertEqual(payload["differences"], ["report-v1.1"])
            self.assertEqual(len(payload["warnings"]), 1)
            self.assertIn(
                "does not invalidate measured evidence",
                payload["warnings"][0],
            )
            self.assertFalse(payload["authoritative_comparisons"]["report-v1.1"]["matches"])

    def test_v11_rejects_candidate_ids_that_can_traverse_evidence_paths(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _v11_evidence(root)
            combined_path = evidence / "extensions" / "load-evaluation.json"
            combined = json.loads(combined_path.read_text(encoding="utf-8"))
            original_id = "q4-generic"
            traversal_id = "../escape"
            malicious_benchmark = _v11_benchmark_mapping()
            for candidate in malicious_benchmark["candidates"]:
                if candidate["id"] == original_id:
                    candidate["id"] = traversal_id
                    candidate["label"] = "Traversal candidate"
            _write_json(
                evidence / "experiment" / "benchmark-set.json",
                malicious_benchmark,
            )

            highest = dict(combined["highest_slo_concurrency"])
            highest[traversal_id] = highest.pop(original_id)
            combined["highest_slo_concurrency"] = dict(sorted(highest.items()))
            for row in combined["rows"]:
                if row["candidate_id"] == original_id:
                    row["candidate_id"] = traversal_id
            combined["rows"].sort(key=lambda row: (row["candidate_id"], row["concurrency"]))
            bindings = combined["evidence_bindings"]
            configurations = dict(bindings["candidate_server_configurations"])
            configurations[traversal_id] = configurations.pop(original_id)
            bindings["candidate_server_configurations"] = dict(sorted(configurations.items()))
            request_urls = dict(bindings["candidate_request_base_urls"])
            request_urls[traversal_id] = request_urls.pop(original_id)
            bindings["candidate_request_base_urls"] = dict(sorted(request_urls.items()))
            _write_json(combined_path, combined)

            original_single = json.loads(
                (evidence / "extensions" / "load" / original_id / "load-evaluation.json").read_text(
                    encoding="utf-8"
                )
            )
            original_single["candidate_id"] = traversal_id
            for row in original_single["rows"]:
                row["candidate_id"] = traversal_id
            _write_json(
                evidence / "extensions" / "escape" / "load-evaluation.json",
                original_single,
            )
            for source, target in (
                (
                    evidence / "extensions" / "load" / original_id / "server-command.json",
                    evidence / "extensions" / "escape" / "server-command.json",
                ),
                (
                    evidence / "experiment" / "candidates" / original_id / "server-command.json",
                    evidence / "experiment" / "escape" / "server-command.json",
                ),
            ):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            _rewrite_checksums(evidence)

            with (
                patch(
                    "paretopilot.replay.assemble_experiment",
                    return_value=malicious_benchmark,
                ),
                patch(
                    "paretopilot.replay.assemble_repeat_pass",
                    side_effect=lambda _experiment, *, pass_number, benchmark_mapping: (
                        _pass_mapping(1.0 if pass_number == 1 else 1.01)
                    ),
                ),
                self.assertRaisesRegex(
                    ValidationError,
                    "v1.1 benchmark candidate ids are unsafe for evidence paths: "
                    r".*\.\./escape",
                ),
            ):
                replay_evidence(evidence, root / "output")
            self.assertFalse((root / "output").exists())

    def test_replay_verifies_regenerates_and_compares_authoritative_outputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _evidence(root)
            first_output = root / "first"
            with (
                patch(
                    "paretopilot.replay.assemble_experiment",
                    return_value=_benchmark_mapping(),
                ),
                patch(
                    "paretopilot.replay.render_report",
                    return_value="<!doctype html>\n<title>Replay</title>\n",
                ),
            ):
                first = replay_evidence(evidence, first_output)

            self.assertTrue(first["valid"])
            self.assertEqual(first["selected_id"], "q8-generic")
            self.assertTrue(first["checksums"]["verified"])
            self.assertEqual(first["checksums"]["entry_count"], 5)
            self.assertEqual(
                first["generated_files"],
                [
                    "benchmark-set.json",
                    "constraints.json",
                    "recommendation.json",
                    "report.html",
                ],
            )
            self.assertTrue((first_output / "replay.json").is_file())
            self.assertTrue(first["authoritative_comparisons"]["benchmark-set"]["matches"])
            self.assertTrue(first["authoritative_comparisons"]["recommendation"]["matches"])
            self.assertFalse(first["authoritative_comparisons"]["report"]["present"])

            for filename in ("report.html",):
                source = first_output / filename
                target = evidence / "experiment" / filename
                target.write_bytes(source.read_bytes())
            _rewrite_checksums(evidence)

            second_output = root / "second"
            with (
                patch(
                    "paretopilot.replay.assemble_experiment",
                    return_value=_benchmark_mapping(),
                ),
                patch(
                    "paretopilot.replay.render_report",
                    return_value="<!doctype html>\n<title>Replay</title>\n",
                ),
            ):
                second = replay_evidence(evidence, second_output)

            self.assertTrue(second["valid"])
            self.assertTrue(second["decision_reproduced"])
            self.assertTrue(second["fully_reproduced"])
            self.assertTrue(second["report_matches_archive"])
            self.assertTrue(second["authoritative_outputs_match"])
            self.assertTrue(
                all(
                    comparison["matches"]
                    for comparison in second["authoritative_comparisons"].values()
                )
            )

    def test_replay_preserves_archived_generator_version_for_exact_comparison(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _evidence(root)
            seed_output = root / "seed"
            with (
                patch(
                    "paretopilot.replay.assemble_experiment",
                    return_value=_benchmark_mapping(),
                ),
                patch("paretopilot.replay.render_report", return_value="<p>stable</p>\n"),
            ):
                replay_evidence(evidence, seed_output)

            for filename in ("benchmark-set.json", "recommendation.json", "report.html"):
                (evidence / "experiment" / filename).write_bytes(
                    (seed_output / filename).read_bytes()
                )
            recommendation_path = evidence / "experiment" / "recommendation.json"
            archived = json.loads(recommendation_path.read_text(encoding="utf-8"))
            archived["paretopilot_version"] = "0.9.0"
            _write_json(recommendation_path, archived)
            _rewrite_checksums(evidence)

            output = root / "replayed"
            with (
                patch(
                    "paretopilot.replay.assemble_experiment",
                    return_value=_benchmark_mapping(),
                ),
                patch("paretopilot.replay.render_report", return_value="<p>stable</p>\n"),
            ):
                payload = replay_evidence(evidence, output)

            regenerated = json.loads((output / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(regenerated["paretopilot_version"], "0.9.0")
            self.assertTrue(payload["authoritative_comparisons"]["recommendation"]["matches"])

    def test_archived_report_drift_is_a_warning_not_evidence_failure(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _evidence(root)
            seed_output = root / "seed"
            with (
                patch(
                    "paretopilot.replay.assemble_experiment",
                    return_value=_benchmark_mapping(),
                ),
                patch("paretopilot.replay.render_report", return_value="<p>old</p>\n"),
            ):
                replay_evidence(evidence, seed_output)

            for filename in ("benchmark-set.json", "recommendation.json", "report.html"):
                (evidence / "experiment" / filename).write_bytes(
                    (seed_output / filename).read_bytes()
                )
            _rewrite_checksums(evidence)

            output = root / "replayed"
            stdout = io.StringIO()
            with (
                patch(
                    "paretopilot.replay.assemble_experiment",
                    return_value=_benchmark_mapping(),
                ),
                patch("paretopilot.replay.render_report", return_value="<p>new</p>\n"),
                patch("sys.stdout", stdout),
            ):
                exit_code = cli.main(["replay", str(evidence), "--output-dir", str(output)])

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["valid"])
            self.assertTrue(payload["decision_reproduced"])
            self.assertFalse(payload["fully_reproduced"])
            self.assertFalse(payload["report_matches_archive"])
            self.assertEqual(payload["differences"], ["report"])
            self.assertEqual(len(payload["warnings"]), 1)
            self.assertIn("does not invalidate measured evidence", payload["warnings"][0])
            self.assertTrue(payload["verdict"].startswith("PASS:"))

    def test_replay_returns_failed_verdict_and_nonzero_cli_for_output_mismatch(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _evidence(root)
            _write_json(
                evidence / "experiment" / "recommendation.json",
                {
                    "paretopilot_version": "1.0.0",
                    "selected_id": "not-the-regenerated-result",
                },
            )
            _rewrite_checksums(evidence)
            output = root / "output"
            stdout = io.StringIO()
            with (
                patch(
                    "paretopilot.replay.assemble_experiment",
                    return_value=_benchmark_mapping(),
                ),
                patch("paretopilot.replay.render_report", return_value="<p>report</p>\n"),
                patch("sys.stdout", stdout),
            ):
                exit_code = cli.main(["replay", str(evidence), "--output-dir", str(output)])

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 5)
            self.assertFalse(payload["valid"])
            self.assertFalse(payload["decision_reproduced"])
            self.assertTrue(payload["verdict"].startswith("FAIL:"))
            details = json.loads((output / "replay.json").read_text(encoding="utf-8"))
            self.assertFalse(details["authoritative_outputs_match"])
            self.assertFalse(details["authoritative_comparisons"]["recommendation"]["matches"])
            self.assertTrue((output / "replay.json").is_file())

    def test_replay_can_precompute_all_policy_profiles_in_one_command(self) -> None:
        repository = Path(__file__).parents[1]
        policies = repository / "configs" / "policies.arm64.json"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _evidence(root)
            output = root / "output"
            with (
                patch(
                    "paretopilot.replay.assemble_experiment",
                    return_value=_benchmark_mapping(),
                ),
                patch("paretopilot.replay.render_report", return_value="<p>report</p>\n"),
            ):
                payload = replay_evidence(
                    evidence,
                    output,
                    policies_path=policies,
                )

            self.assertTrue(payload["valid"])
            self.assertEqual(payload["policy_profile_count"], 5)
            self.assertEqual(
                payload["policy_selected_ids"],
                {
                    "canonical-latency": "q8-generic",
                    "memory-first": "q4-generic",
                    "first-token-first": "q4-kleidiai",
                    "prompt-ingest-first": "q4-kleidiai-tuned",
                    "decode-first": "q8-generic",
                },
            )
            profiles = json.loads((output / "policy-profiles.json").read_text(encoding="utf-8"))
            self.assertEqual(
                set(profiles["input_fingerprints"]),
                {"benchmarks_sha256", "constraints_sha256", "policies_sha256"},
            )

    def test_checksum_tampering_and_unlisted_files_fail_before_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _evidence(root)
            (evidence / "status.json").write_text("{}\n", encoding="utf-8")
            output = root / "tampered-output"
            with self.assertRaisesRegex(ValidationError, "SHA256 mismatch for status.json"):
                replay_evidence(evidence, output)
            self.assertFalse(output.exists())

            evidence = _evidence(root / "unlisted-case")
            (evidence / "unlisted.txt").write_text("not listed\n", encoding="utf-8")
            output = root / "unlisted-output"
            with self.assertRaisesRegex(ValidationError, "missing from SHA256SUMS"):
                replay_evidence(evidence, output)
            self.assertFalse(output.exists())

    def test_core_authoritative_outputs_are_required_for_reproduction(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _evidence(root)
            (evidence / "experiment" / "recommendation.json").unlink()
            _rewrite_checksums(evidence)

            with self.assertRaisesRegex(
                ValidationError,
                "missing required checksummed files: experiment/recommendation.json",
            ):
                replay_evidence(evidence, root / "output")

    def test_checksum_manifest_rejects_unsafe_duplicate_and_malformed_entries(self) -> None:
        cases = (
            (
                "unsafe",
                f"{'0' * 64}  ./../outside\n",
                "unsafe path",
            ),
            (
                "duplicate",
                None,
                "duplicate entry",
            ),
            (
                "malformed",
                f"{'0' * 64} *status.json\n",
                "is malformed",
            ),
        )
        for label, replacement, message in cases:
            with self.subTest(label=label), TemporaryDirectory() as directory:
                root = Path(directory)
                evidence = _evidence(root)
                checksum_path = evidence / "SHA256SUMS"
                if label == "duplicate":
                    first_line = checksum_path.read_text(encoding="utf-8").splitlines()[0]
                    checksum_path.write_text(
                        checksum_path.read_text(encoding="utf-8") + first_line + "\n",
                        encoding="utf-8",
                    )
                else:
                    assert replacement is not None
                    checksum_path.write_text(replacement, encoding="utf-8")
                with self.assertRaisesRegex(ValidationError, message):
                    replay_evidence(evidence, root / "output")

    def test_status_must_be_exactly_complete_canonical_and_valid(self) -> None:
        cases = (
            ("status", "partial", "status.json status must be 'complete'"),
            ("measurement_valid", False, "measurement_valid must be True"),
            ("extra", True, "unknown extra"),
        )
        for field, value, message in cases:
            with self.subTest(field=field), TemporaryDirectory() as directory:
                root = Path(directory)
                evidence = _evidence(root)
                status_path = evidence / "status.json"
                status = json.loads(status_path.read_text(encoding="utf-8"))
                status[field] = value
                _write_json(status_path, status)
                _rewrite_checksums(evidence)
                with self.assertRaisesRegex(ValidationError, message):
                    replay_evidence(evidence, root / "output")

    def test_replay_rejects_existing_overlapping_and_non_directory_inputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _evidence(root)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ValidationError, "refusing to overwrite"):
                replay_evidence(evidence, existing)

            with self.assertRaisesRegex(ValidationError, "must not contain"):
                replay_evidence(evidence, evidence / "output")

            archive = root / "evidence.zip"
            archive.write_bytes(b"not an extracted directory")
            with self.assertRaisesRegex(ValidationError, "is not a directory"):
                replay_evidence(archive, root / "archive-output")


if __name__ == "__main__":
    unittest.main()
