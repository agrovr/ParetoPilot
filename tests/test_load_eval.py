from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from paretopilot.domain import ValidationError
from paretopilot.load_eval import (
    LoadRequest,
    build_load_evidence_binding,
    combine_load_evaluations,
    evaluate_load,
    evaluate_llama_server_load,
    llama_server_request_runner,
    load_load_plan,
    load_plan_evaluation_contract,
    validate_load_artifact_against_plan,
    validate_combined_load_evaluation,
    validate_load_evaluation,
)


SLO = {
    "min_completion_rate": 1.0,
    "max_ttft_ms_p95": 50.0,
    "max_e2e_latency_ms_p95": 130.0,
}


def _result(
    request: LoadRequest,
    *,
    completed: bool = True,
    generated_tokens: int | None = None,
    error: str | None = None,
) -> dict[str, object]:
    duration_seconds = 0.1 + request.concurrency * 0.01
    wave = (request.request_index - 1) // request.concurrency
    start = request.concurrency * 100.0 + wave * 0.2
    return {
        "completed": completed,
        "ttft_ms": 20.0 + request.concurrency if completed else None,
        "e2e_latency_ms": duration_seconds * 1000.0,
        "generated_tokens": (
            request.output_tokens
            if generated_tokens is None and completed
            else generated_tokens or 0
        ),
        "error": error,
        "started_at_seconds": start,
        "finished_at_seconds": start + duration_seconds,
    }


def _evaluate(
    candidate_id: str = "candidate-a",
    *,
    request_runner=None,
    slo: dict[str, object] | None = None,
    peak_rss: dict[int, float] | None = None,
):
    runner = request_runner or _result
    return evaluate_load(
        candidate_id=candidate_id,
        prompts=("Explain Arm efficiency briefly.", "Name one deployment tradeoff."),
        output_tokens=16,
        warmup_requests_per_level=2,
        measured_requests_per_level=8,
        request_runner=runner,
        slo=slo or SLO,
        peak_rss_mib_by_concurrency=peak_rss,
        synthetic=True,
    )


class _StreamResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines

    def __enter__(self) -> "_StreamResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):
        return iter(self.lines)


class LoadEvaluationTests(unittest.TestCase):
    def test_load_artifacts_must_match_their_checksummed_plan_contract(self) -> None:
        plan = load_load_plan(Path(__file__).parents[1] / "configs" / "load.arm64.json")
        artifact = evaluate_load(
            candidate_id="candidate-a",
            prompts=plan.prompts,
            output_tokens=plan.output_tokens,
            warmup_requests_per_level=plan.warmup_requests_per_level,
            measured_requests_per_level=plan.measured_requests_per_level,
            request_runner=_result,
            slo=plan.slo,
            concurrency_levels=plan.concurrency_levels,
            synthetic=True,
        )

        validate_load_artifact_against_plan(artifact, plan)
        contract = load_plan_evaluation_contract(plan)
        self.assertEqual(artifact["methodology"], contract["methodology"])
        self.assertEqual(artifact["slo"], contract["slo"])
        self.assertNotIn("timeout_seconds", contract["methodology"])

        methodology_tamper = deepcopy(artifact)
        methodology_tamper["methodology"]["prompts"][0]["text"] = "different"
        with self.assertRaisesRegex(ValidationError, "methodology does not match"):
            validate_load_artifact_against_plan(methodology_tamper, plan)

        slo_tamper = deepcopy(artifact)
        slo_tamper["slo"]["max_ttft_ms_p95"] += 1.0
        with self.assertRaisesRegex(ValidationError, "SLO does not match"):
            validate_load_artifact_against_plan(slo_tamper, plan)

    def test_bundled_load_plan_is_strict_and_bounded(self) -> None:
        plan = load_load_plan(Path(__file__).parents[1] / "configs" / "load.arm64.json")

        self.assertEqual(plan.concurrency_levels, (1, 2, 4))
        self.assertEqual(plan.output_tokens, 64)
        self.assertEqual(plan.warmup_requests_per_level, 4)
        self.assertEqual(plan.measured_requests_per_level, 8)
        self.assertEqual(len(plan.prompts), 3)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            raw = {
                "schema_version": "1.0",
                "prompts": ["Prompt"],
                "output_tokens": 16,
                "warmup_requests_per_level": 0,
                "measured_requests_per_level": 4,
                "concurrency_levels": [1, 2, 4],
                "timeout_seconds": 30.0,
                "slo": SLO,
                "surprise": True,
            }
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "unknown fields"):
                load_load_plan(path)

    def test_llama_server_runner_captures_strict_sse_measurement(self) -> None:
        request = LoadRequest(
            prompt="Prompt",
            prompt_index=1,
            output_tokens=8,
            concurrency=1,
            request_index=1,
            warmup=False,
        )
        response = _StreamResponse(
            [
                b'data: {"content":"Arm","tokens":[1,2,3]}\n',
                b'data: {"content":" result","tokens":[4,5,6,7,8],"timings":{"predicted_n":8}}\n',
            ]
        )
        clock_values = iter((10.0, 10.02, 10.10))
        runner = llama_server_request_runner(
            "http://127.0.0.1:8080",
            clock=lambda: next(clock_values),
        )

        with patch("paretopilot.load_eval.urlopen", return_value=response) as mocked:
            result = runner(request)

        self.assertTrue(result["completed"])
        self.assertEqual(result["generated_tokens"], 8)
        self.assertAlmostEqual(result["ttft_ms"], 20.0)
        self.assertAlmostEqual(result["e2e_latency_ms"], 100.0)
        sent = mocked.call_args.args[0]
        sent_payload = json.loads(sent.data)
        self.assertEqual(sent_payload["n_predict"], 8)
        self.assertTrue(sent_payload["ignore_eos"])
        self.assertFalse(sent_payload["cache_prompt"])
        self.assertEqual(sent_payload["temperature"], 0)

    def test_llama_server_runner_measures_http_and_network_failures(self) -> None:
        request = LoadRequest("Prompt", 1, 8, 1, 1, False)
        http_error = HTTPError(
            "http://127.0.0.1:8080/completion",
            503,
            "busy",
            {},
            BytesIO(b"server busy"),
        )
        http_clock = iter((1.0, 1.05))
        http_runner = llama_server_request_runner(
            "http://127.0.0.1:8080",
            clock=lambda: next(http_clock),
        )
        with patch("paretopilot.load_eval.urlopen", side_effect=http_error):
            result = http_runner(request)
        self.assertFalse(result["completed"])
        self.assertEqual(result["error"], "HTTP 503: server busy")
        self.assertAlmostEqual(result["e2e_latency_ms"], 50.0)

        network_clock = iter((2.0, 2.08))
        network_runner = llama_server_request_runner(
            "http://127.0.0.1:8080",
            clock=lambda: next(network_clock),
        )
        with patch(
            "paretopilot.load_eval.urlopen",
            side_effect=URLError("connection refused"),
        ):
            result = network_runner(request)
        self.assertFalse(result["completed"])
        self.assertIn("network error", result["error"])
        self.assertAlmostEqual(result["e2e_latency_ms"], 80.0)

    def test_llama_server_runner_rejects_malformed_or_contradictory_sse(self) -> None:
        request = LoadRequest("Prompt", 1, 8, 1, 1, False)
        scenarios = [
            (
                [b"data: not-json\n"],
                "invalid SSE JSON",
            ),
            (
                [b'data: {"content":"x","tokens":[1],"timings":{"predicted_n":8}}\n'],
                "token arrays do not match",
            ),
            (
                [b'data: {"content":7,"timings":{"predicted_n":8}}\n'],
                "content must be a string",
            ),
        ]
        for lines, message in scenarios:
            with self.subTest(message=message):
                clock_values = iter((1.0, 1.01, 1.1))
                runner = llama_server_request_runner(
                    "http://127.0.0.1:8080",
                    clock=lambda: next(clock_values),
                )
                with patch(
                    "paretopilot.load_eval.urlopen",
                    return_value=_StreamResponse(lines),
                ):
                    with self.assertRaisesRegex(ValidationError, message):
                        runner(request)

    def test_evaluate_load_uses_fixed_inputs_and_computes_slo_capacity(self) -> None:
        calls: list[LoadRequest] = []
        lock = threading.Lock()

        def runner(request: LoadRequest):
            with lock:
                calls.append(request)
            return _result(request)

        artifact = _evaluate(
            request_runner=runner,
            peak_rss={1: 1000.0, 2: 1010.0, 4: 1030.0},
        )

        validate_load_evaluation(artifact)
        self.assertEqual(artifact["highest_slo_concurrency"], 2)
        self.assertEqual([row["concurrency"] for row in artifact["rows"]], [1, 2, 4])
        self.assertEqual([row["slo_met"] for row in artifact["rows"]], [True, True, False])
        self.assertEqual(
            artifact["rows"][2]["slo_failures"],
            ["e2e_latency_ms_p95_above_maximum"],
        )
        self.assertEqual(artifact["rows"][0]["completion_rate"], 1.0)
        self.assertEqual(artifact["rows"][0]["error_rate"], 0.0)
        self.assertAlmostEqual(artifact["rows"][0]["ttft_ms_p50"], 21.0)
        self.assertAlmostEqual(artifact["rows"][1]["e2e_latency_ms_p95"], 120.0)
        self.assertEqual(artifact["rows"][2]["peak_rss_mib"], 1030.0)
        self.assertEqual(len(calls), 30)
        self.assertEqual(sum(call.warmup for call in calls), 6)
        self.assertTrue(all(call.output_tokens == 16 for call in calls))
        self.assertEqual(
            [sample["prompt_index"] for sample in artifact["rows"][0]["samples"]],
            [1, 2, 1, 2, 1, 2, 1, 2],
        )
        self.assertAlmostEqual(
            artifact["rows"][0]["requests_per_second"],
            8 / 1.51,
        )
        self.assertAlmostEqual(
            artifact["rows"][0]["generated_tokens_per_second"],
            128 / 1.51,
        )

    def test_expected_request_failures_are_measured_not_hidden(self) -> None:
        def runner(request: LoadRequest):
            if request.concurrency == 4 and not request.warmup:
                return _result(
                    request,
                    completed=False,
                    generated_tokens=0,
                    error="request timed out",
                )
            return _result(request)

        artifact = _evaluate(request_runner=runner)
        failed_row = artifact["rows"][2]

        self.assertEqual(failed_row["completed_requests"], 0)
        self.assertEqual(failed_row["failed_requests"], 8)
        self.assertEqual(failed_row["completion_rate"], 0.0)
        self.assertEqual(failed_row["error_rate"], 1.0)
        self.assertEqual(failed_row["requests_per_second"], 0.0)
        self.assertIsNone(failed_row["ttft_ms_p50"])
        self.assertIsNone(failed_row["e2e_latency_ms_p95"])
        self.assertEqual(
            failed_row["slo_failures"],
            [
                "completion_rate_below_minimum",
                "no_completed_request_ttft",
                "no_completed_request_e2e",
            ],
        )
        self.assertEqual(artifact["highest_slo_concurrency"], 2)

    def test_runner_results_fail_closed_on_malformed_evidence(self) -> None:
        scenarios = {
            "unknown": (
                lambda request: {**_result(request), "surprise": True},
                "unknown fields",
            ),
            "short-success": (
                lambda request: _result(request, generated_tokens=15),
                "fixed output token count",
            ),
            "missing-error": (
                lambda request: _result(request, completed=False, generated_tokens=0),
                "error must be a non-empty string",
            ),
            "mismatched-time": (
                lambda request: {
                    **_result(request),
                    "e2e_latency_ms": 999.0,
                },
                "must match its start/finish",
            ),
            "non-finite": (
                lambda request: {
                    **_result(request),
                    "ttft_ms": math.nan,
                },
                "must be a finite number",
            ),
        }
        for name, (runner, message) in scenarios.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValidationError, message):
                    evaluate_load(
                        candidate_id="candidate-a",
                        prompts=("Prompt",),
                        output_tokens=16,
                        warmup_requests_per_level=0,
                        measured_requests_per_level=1,
                        request_runner=runner,
                        slo=SLO,
                        concurrency_levels=(1,),
                        synthetic=True,
                    )

    def test_unexpected_runner_exception_fails_the_evaluation(self) -> None:
        def runner(_request: LoadRequest):
            raise RuntimeError("connection pool broke")

        with self.assertRaisesRegex(
            ValidationError,
            "request runner raised RuntimeError.*connection pool broke",
        ):
            evaluate_load(
                candidate_id="candidate-a",
                prompts=("Prompt",),
                output_tokens=16,
                warmup_requests_per_level=0,
                measured_requests_per_level=1,
                request_runner=runner,
                slo=SLO,
                concurrency_levels=(1,),
                synthetic=True,
            )

    def test_configuration_bounds_fail_closed(self) -> None:
        base = {
            "candidate_id": "candidate-a",
            "prompts": ("Prompt",),
            "output_tokens": 16,
            "warmup_requests_per_level": 0,
            "measured_requests_per_level": 8,
            "request_runner": _result,
            "slo": SLO,
            "synthetic": True,
        }
        scenarios = [
            ({"concurrency_levels": (1, 3)}, "only 1, 2, and 4"),
            ({"concurrency_levels": (2, 1)}, "unique and increasing"),
            ({"measured_requests_per_level": 6}, "divisible"),
            ({"prompts": ("same", "same")}, "must be unique"),
            (
                {
                    "slo": {
                        **SLO,
                        "max_ttft_ms_p95": 200.0,
                        "max_e2e_latency_ms_p95": 100.0,
                    }
                },
                "cannot exceed max E2E",
            ),
            (
                {"peak_rss_mib_by_concurrency": {1: 1000.0}},
                "exactly every concurrency level",
            ),
        ]
        for changes, message in scenarios:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValidationError, message):
                    evaluate_load(**{**base, **changes})

    def test_artifact_validator_recomputes_aggregates_and_prompt_schedule(self) -> None:
        artifact = _evaluate()
        scenarios = []

        throughput = deepcopy(artifact)
        throughput["rows"][0]["requests_per_second"] += 0.1
        scenarios.append((throughput, "aggregate fields"))

        digest = deepcopy(artifact)
        digest["rows"][0]["samples"][0]["prompt_sha256"] = "0" * 64
        scenarios.append((digest, "does not match"))

        schedule = deepcopy(artifact)
        schedule["rows"][0]["samples"][0]["prompt_index"] = 2
        schedule["rows"][0]["samples"][0]["prompt_sha256"] = schedule["methodology"]["prompts"][1][
            "sha256"
        ]
        scenarios.append((schedule, "round-robin schedule"))

        highest = deepcopy(artifact)
        highest["highest_slo_concurrency"] = 4
        scenarios.append((highest, "highest_slo_concurrency"))

        unknown = deepcopy(artifact)
        unknown["surprise"] = True
        scenarios.append((unknown, "unknown fields"))

        for payload, message in scenarios:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValidationError, message):
                    validate_load_evaluation(payload)

    def test_combine_load_evaluations_is_canonical_and_strict(self) -> None:
        candidate_b = _evaluate("candidate-b")
        candidate_a = _evaluate("candidate-a")

        combined = combine_load_evaluations([candidate_b, candidate_a])

        validate_combined_load_evaluation(combined)
        self.assertEqual(
            [row["candidate_id"] for row in combined["rows"]],
            ["candidate-a"] * 3 + ["candidate-b"] * 3,
        )
        self.assertEqual(
            combined["highest_slo_concurrency"],
            {"candidate-a": 2, "candidate-b": 2},
        )

        with self.assertRaisesRegex(ValidationError, "duplicate"):
            combine_load_evaluations([candidate_a, candidate_a])

        mismatched = deepcopy(candidate_b)
        mismatched["slo"]["max_e2e_latency_ms_p95"] = 140.0
        # Keep the standalone artifact internally valid before checking compatibility.
        for row in mismatched["rows"]:
            row["slo_met"] = True
            row["slo_failures"] = []
        mismatched["highest_slo_concurrency"] = 4
        validate_load_evaluation(mismatched)
        with self.assertRaisesRegex(ValidationError, "SLO does not match"):
            combine_load_evaluations([candidate_a, mismatched])

    def test_measured_load_binding_locks_plan_and_material_server_configuration(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "prompts": ["Prompt"],
                        "output_tokens": 16,
                        "warmup_requests_per_level": 0,
                        "measured_requests_per_level": 1,
                        "concurrency_levels": [1],
                        "timeout_seconds": 60.0,
                        "slo": SLO,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def write_command(path: Path, *, port: int, parallel: int = 1, threads: int = 4):
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": "1.0",
                            "argv": [
                                "./build/llama-server",
                                "--model",
                                "./models/q4.gguf",
                                "--threads",
                                str(threads),
                                "--parallel",
                                str(parallel),
                                "--host",
                                "127.0.0.1",
                                "--port",
                                str(port),
                            ],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

            canonical = root / "canonical.json"
            load_command = root / "load.json"
            write_command(canonical, port=8080)
            write_command(load_command, port=18181)
            binding = build_load_evidence_binding(
                base_url="http://127.0.0.1:18181",
                plan_path=plan,
                server_command_path=load_command,
                canonical_server_command_path=canonical,
            )
            self.assertEqual(binding["server_configuration"]["canonical_parallel"], 1)
            self.assertEqual(
                binding["server_configuration"]["differing_binding_options"],
                ["--port"],
            )
            self.assertEqual(
                binding["request_base_url"],
                "http://127.0.0.1:18181",
            )

            artifact = evaluate_load(
                candidate_id="candidate-a",
                prompts=("Prompt",),
                output_tokens=16,
                warmup_requests_per_level=0,
                measured_requests_per_level=1,
                request_runner=_result,
                slo=SLO,
                concurrency_levels=(1,),
                synthetic=False,
                evidence_binding=binding,
            )
            validate_load_evaluation(artifact, require_evidence_binding=True)
            with self.assertRaisesRegex(ValidationError, "base_url does not match"):
                evaluate_llama_server_load(
                    "http://127.0.0.1:18182",
                    load_load_plan(plan),
                    candidate_id="candidate-a",
                    evidence_binding=binding,
                )

            second_command = root / "second-load.json"
            write_command(second_command, port=18182)
            second_binding = build_load_evidence_binding(
                base_url="http://127.0.0.1:18182",
                plan_path=plan,
                server_command_path=second_command,
                canonical_server_command_path=canonical,
            )
            second_artifact = evaluate_load(
                candidate_id="candidate-b",
                prompts=("Prompt",),
                output_tokens=16,
                warmup_requests_per_level=0,
                measured_requests_per_level=1,
                request_runner=_result,
                slo=SLO,
                concurrency_levels=(1,),
                synthetic=False,
                evidence_binding=second_binding,
            )
            combined = combine_load_evaluations(
                [second_artifact, artifact],
                require_evidence_bindings=True,
            )
            self.assertEqual(
                list(combined["evidence_bindings"]["candidate_request_base_urls"]),
                ["candidate-a", "candidate-b"],
            )
            tampered_combined = deepcopy(combined)
            tampered_combined["evidence_bindings"]["candidate_request_base_urls"]["candidate-b"] = (
                "http://127.0.0.1:18181"
            )
            with self.assertRaisesRegex(ValidationError, "port must match"):
                validate_combined_load_evaluation(
                    tampered_combined,
                    require_evidence_bindings=True,
                )

            with self.assertRaisesRegex(ValidationError, "port must match"):
                build_load_evidence_binding(
                    base_url="http://127.0.0.1:19191",
                    plan_path=plan,
                    server_command_path=load_command,
                    canonical_server_command_path=canonical,
                )

            write_command(load_command, port=18181, parallel=4)
            with self.assertRaisesRegex(ValidationError, "--parallel must match"):
                build_load_evidence_binding(
                    base_url="http://127.0.0.1:18181",
                    plan_path=plan,
                    server_command_path=load_command,
                    canonical_server_command_path=canonical,
                )

            write_command(load_command, port=18181, threads=8)
            with self.assertRaisesRegex(ValidationError, "materially differs"):
                build_load_evidence_binding(
                    base_url="http://127.0.0.1:18181",
                    plan_path=plan,
                    server_command_path=load_command,
                    canonical_server_command_path=canonical,
                )


if __name__ == "__main__":
    unittest.main()
