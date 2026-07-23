from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from paretopilot.domain import ValidationError
from paretopilot.server_eval import (
    _answer_matches,
    evaluate_server,
    load_evaluation_suite,
    parse_gnu_time_peak_rss,
    pool_server_evaluations,
    validate_server_evaluation,
)


ROOT = Path(__file__).parents[1]
SUITE = ROOT / "evals" / "qwen-smoke-v1.json"
BEHAVIOR_SUITE = ROOT / "evals" / "qwen-behavior-v2.json"


def _evaluation_payload(
    *,
    candidate_id: str = "candidate-a",
    response: str = "YES",
    matched: bool = True,
    ttft_values: tuple[float, ...] = (10.0, 20.0),
    e2e_values: tuple[float, ...] = (100.0, 200.0),
    warmups: int = 1,
) -> dict[str, object]:
    samples = [
        {
            "index": index,
            "ttft_ms": ttft,
            "e2e_latency_ms": e2e,
            "event_count": 64,
            "predicted_tokens": 64,
            "content": f"sample {index}",
        }
        for index, (ttft, e2e) in enumerate(zip(ttft_values, e2e_values, strict=True), start=1)
    ]
    ordered_ttft = sorted(ttft_values)
    ordered_e2e = sorted(e2e_values)
    return {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "synthetic": False,
        "suite": {
            "id": "suite-v1",
            "license": "CC0-1.0",
            "sha256": "a" * 64,
            "quality_case_count": 1,
            "performance_repetitions": len(samples),
            "performance_warmups": warmups,
            "generation_tokens": 64,
            "cache_prompt": False,
            "seed": 4242,
            "temperature": 0,
        },
        "quality": {
            "method": "fixed exact-answer smoke evaluation",
            "score": float(matched),
            "passed": int(matched),
            "total": 1,
            "cases": [
                {
                    "id": "classification",
                    "prompt": "Reply YES.",
                    "accepted_answers": ["YES"],
                    "response": response,
                    "matched": matched,
                    "matched_answer": "YES" if matched else None,
                }
            ],
        },
        "latency": {
            "method": "single-client streamed HTTP requests",
            "ttft_ms_p50": (ordered_ttft[(len(samples) - 1) // 2] + ordered_ttft[len(samples) // 2])
            / 2,
            "ttft_ms_p95": ordered_ttft[-1],
            "e2e_latency_ms_p50": (
                ordered_e2e[(len(samples) - 1) // 2] + ordered_e2e[len(samples) // 2]
            )
            / 2,
            "e2e_latency_ms_p95": ordered_e2e[-1],
            "samples": samples,
        },
    }


class _Response:
    def __init__(self, payload: bytes | list[bytes]) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        assert isinstance(self.payload, bytes)
        return self.payload

    def __iter__(self):
        assert isinstance(self.payload, list)
        return iter(self.payload)


class ServerEvalTests(unittest.TestCase):
    def test_public_validator_checks_one_serialized_evaluation(self) -> None:
        payload = _evaluation_payload()
        validate_server_evaluation(payload)

        payload["candidate_id"] = ""
        with self.assertRaises(ValidationError):
            validate_server_evaluation(payload)

    def test_bundled_suite_is_valid_and_versioned(self) -> None:
        suite = load_evaluation_suite(SUITE)
        self.assertEqual(suite.suite_id, "paretopilot-qwen-smoke-v1")
        self.assertEqual(len(suite.quality_cases), 5)
        self.assertEqual(suite.repetitions, 10)
        self.assertEqual(suite.warmups, 1)

    def test_declared_quality_match_modes_preserve_format_and_json_structure(self) -> None:
        self.assertTrue(_answer_matches("  ARM READY\n", "ARM READY", "trimmed-exact"))
        self.assertFalse(_answer_matches("arm ready", "ARM READY", "trimmed-exact"))
        self.assertFalse(_answer_matches("ARM READY.", "ARM READY", "trimmed-exact"))
        self.assertTrue(
            _answer_matches(
                '{ "workers": 4, "arm64": true }',
                '{"arm64":true,"workers":4}',
                "json-exact",
            )
        )
        self.assertFalse(
            _answer_matches(
                "arm64 true workers 4",
                '{"arm64":true,"workers":4}',
                "json-exact",
            )
        )
        self.assertFalse(
            _answer_matches(
                '{"arm64":true,"arm64":true,"workers":4}',
                '{"arm64":true,"workers":4}',
                "json-exact",
            )
        )

        artifact = _evaluation_payload()
        case = artifact["quality"]["cases"][0]
        case["accepted_answers"] = ['{"status":"ready"}']
        case["match_mode"] = "json-exact"
        case["response"] = "status ready"
        case["matched"] = True
        case["matched_answer"] = '{"status":"ready"}'
        with self.assertRaisesRegex(ValidationError, "matched outcome"):
            validate_server_evaluation(artifact)

    def test_expanded_behavior_suite_is_valid_and_balanced(self) -> None:
        suite = load_evaluation_suite(BEHAVIOR_SUITE)

        self.assertEqual(suite.suite_id, "paretopilot-qwen-behavior-v2")
        self.assertEqual(len(suite.quality_cases), 24)
        self.assertEqual(
            {case.case_id.split("-", 1)[0] for case in suite.quality_cases},
            {"instruction", "extraction", "classification", "arithmetic", "json", "fact"},
        )
        self.assertEqual(suite.generation_tokens, 64)
        self.assertEqual(suite.repetitions, 10)
        self.assertEqual(suite.warmups, 1)
        self.assertEqual(
            {case.match_mode for case in suite.quality_cases},
            {"trimmed-exact", "json-exact"},
        )

    def test_suite_rejects_duplicate_ids_and_unknown_fields(self) -> None:
        raw = json.loads(SUITE.read_text(encoding="utf-8"))
        raw["quality_cases"][1]["id"] = raw["quality_cases"][0]["id"]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "duplicate quality case"):
                load_evaluation_suite(path)

            raw["quality_cases"][1]["id"] = "unique"
            raw["surprise"] = True
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "unknown fields"):
                load_evaluation_suite(path)

    def test_evaluate_server_retains_quality_and_latency_samples(self) -> None:
        quality_answers = ["PARETOPILOT", "42", "Paris", "10", "YES"]
        responses: list[_Response] = [
            _Response(json.dumps({"choices": [{"message": {"content": answer}}]}).encode())
            for answer in quality_answers
        ]
        stream = [
            b'data: {"content":"Arm","tokens":[1],"stop":false}\n',
            b'data: {"content":" evidence","tokens":[2],"stop":true,'
            b'"timings":{"predicted_n":64}}\n',
        ]
        responses.extend(_Response(stream) for _ in range(11))
        clock_values = iter(float(index) / 100 for index in range(33))

        with (
            patch("paretopilot.server_eval.urlopen", side_effect=responses) as mocked_urlopen,
            patch("paretopilot.server_eval.sha256_file", return_value="a" * 64),
        ):
            payload = evaluate_server(
                "http://127.0.0.1:8080",
                SUITE,
                candidate_id="candidate-a",
                clock=lambda: next(clock_values),
            )

        self.assertEqual(payload["quality"]["score"], 1.0)
        self.assertEqual(len(payload["latency"]["samples"]), 10)
        self.assertAlmostEqual(payload["latency"]["ttft_ms_p50"], 10.0)
        self.assertAlmostEqual(payload["latency"]["e2e_latency_ms_p95"], 20.0)
        self.assertEqual(payload["latency"]["samples"][0]["content"], "Arm evidence")
        self.assertEqual(payload["latency"]["samples"][0]["predicted_tokens"], 64)
        self.assertFalse(payload["synthetic"])
        performance_request = mocked_urlopen.call_args_list[5].args[0]
        self.assertTrue(json.loads(performance_request.data)["ignore_eos"])

    def test_quality_matching_is_exact_after_simple_normalization(self) -> None:
        quality_answers = ["PARETOPILOT extra", "42", "Paris", "10", "YES"]
        responses: list[_Response] = [
            _Response(json.dumps({"choices": [{"message": {"content": answer}}]}).encode())
            for answer in quality_answers
        ]
        stream = [b'data: {"content":"x","tokens":[1],"stop":true,"timings":{"predicted_n":64}}\n']
        responses.extend(_Response(stream) for _ in range(11))
        clock_values = iter(float(index) / 100 for index in range(33))
        with patch("paretopilot.server_eval.urlopen", side_effect=responses):
            payload = evaluate_server(
                "http://127.0.0.1:8080",
                SUITE,
                candidate_id="candidate-a",
                clock=lambda: next(clock_values),
            )
        self.assertEqual(payload["quality"]["passed"], 4)
        self.assertEqual(payload["quality"]["score"], 0.8)

    def test_evaluate_server_rejects_a_short_fixed_length_generation(self) -> None:
        quality_answers = ["PARETOPILOT", "42", "Paris", "10", "YES"]
        responses: list[_Response] = [
            _Response(json.dumps({"choices": [{"message": {"content": answer}}]}).encode())
            for answer in quality_answers
        ]
        responses.append(
            _Response([b'data: {"content":"short","stop":true,"timings":{"predicted_n":63}}\n'])
        )
        clock_values = iter((0.0, 0.01, 0.02))
        with patch("paretopilot.server_eval.urlopen", side_effect=responses):
            with self.assertRaisesRegex(ValidationError, "expected exactly 64"):
                evaluate_server(
                    "http://127.0.0.1:8080",
                    SUITE,
                    candidate_id="candidate-a",
                    clock=lambda: next(clock_values),
                )

    def test_pool_server_evaluations_recomputes_balanced_pass_statistics(self) -> None:
        first_payload = _evaluation_payload()
        second_payload = _evaluation_payload(
            response="yes!",
            ttft_values=(30.0, 40.0),
            e2e_values=(300.0, 400.0),
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "pass-a.json", root / "pass-b.json"]
            for path, payload in zip(paths, (first_payload, second_payload), strict=True):
                path.write_text(json.dumps(payload), encoding="utf-8")
            originals = [path.read_bytes() for path in paths]

            pooled = pool_server_evaluations(paths)

            self.assertEqual([path.read_bytes() for path in paths], originals)
        self.assertEqual(pooled["suite"]["performance_repetitions"], 4)
        self.assertEqual(pooled["suite"]["performance_warmups"], 2)
        self.assertEqual(
            [sample["index"] for sample in pooled["latency"]["samples"]],
            [1, 2, 3, 4],
        )
        self.assertEqual(pooled["latency"]["ttft_ms_p50"], 25.0)
        self.assertEqual(pooled["latency"]["ttft_ms_p95"], 40.0)
        self.assertEqual(pooled["latency"]["e2e_latency_ms_p50"], 250.0)
        self.assertEqual(pooled["latency"]["e2e_latency_ms_p95"], 400.0)
        self.assertEqual(pooled["quality"]["cases"][0]["response"], "YES")

    def test_pool_server_evaluations_rejects_incompatible_or_untrusted_passes(self) -> None:
        scenarios: list[tuple[str, dict[str, object], str]] = []

        candidate = _evaluation_payload(candidate_id="candidate-b")
        scenarios.append(("candidate", candidate, "candidate_id"))

        suite = _evaluation_payload()
        suite["suite"]["seed"] = 7
        scenarios.append(("suite", suite, "suite identity"))

        counts = _evaluation_payload(warmups=2)
        scenarios.append(("per-pass-counts", counts, "per-pass counts"))

        outcome = _evaluation_payload(response="NO", matched=False)
        scenarios.append(("outcome", outcome, "matched outcomes"))

        synthetic = _evaluation_payload()
        synthetic["synthetic"] = True
        scenarios.append(("synthetic", synthetic, "synthetic must be false"))

        token_count = _evaluation_payload()
        token_count["latency"]["samples"][0]["predicted_tokens"] = 63
        scenarios.append(("token-count", token_count, "predicted_tokens"))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            first.write_text(json.dumps(_evaluation_payload()), encoding="utf-8")
            for name, payload, message in scenarios:
                with self.subTest(name=name):
                    other = root / f"{name}.json"
                    other.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValidationError, message):
                        pool_server_evaluations([first, other])

    def test_pool_server_evaluations_enforces_input_bounds_and_strict_json(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            first.write_text(json.dumps(_evaluation_payload()), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "from 2 to 8"):
                pool_server_evaluations([first])
            with self.assertRaisesRegex(ValidationError, "from 2 to 8"):
                pool_server_evaluations([first] * 9)
            with self.assertRaisesRegex(ValidationError, "distinct"):
                pool_server_evaluations([first, first])

            duplicate_key = root / "duplicate.json"
            duplicate_key.write_text('{"schema_version":"1.0","schema_version":"1.0"}')
            with self.assertRaisesRegex(ValidationError, "duplicate JSON object key"):
                pool_server_evaluations([first, duplicate_key])

    def test_parse_gnu_time_peak_rss(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "time.txt"
            path.write_text(
                "\tMaximum resident set size (kbytes): 1048576\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_gnu_time_peak_rss(path), 1024.0)

            path.write_text("no measurement\n", encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "exactly one"):
                parse_gnu_time_peak_rss(path)

    def test_invalid_base_url_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValidationError, "base_url"):
            evaluate_server("file:///tmp/server", SUITE, candidate_id="x")


if __name__ == "__main__":
    unittest.main()
