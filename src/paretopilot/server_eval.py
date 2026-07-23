"""Deterministic quality and request-latency evaluation for ``llama-server``.

The evaluator intentionally uses only the Python standard library.  It talks to
the pinned server's public HTTP API, retains every raw model response, and emits
the aggregate values consumed by the candidate assembler.  Peak RSS is captured
by the surrounding process supervisor because a process cannot reliably measure
its parent's maximum resident set after that parent exits.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from paretopilot.domain import ValidationError
from paretopilot.io import load_json_object, sha256_file


MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class QualityCase:
    case_id: str
    prompt: str
    accepted_answers: tuple[str, ...]
    match_mode: str


@dataclass(frozen=True)
class EvaluationSuite:
    suite_id: str
    license: str
    quality_cases: tuple[QualityCase, ...]
    performance_prompt: str
    generation_tokens: int
    repetitions: int
    warmups: int


def load_evaluation_suite(path: Path) -> EvaluationSuite:
    """Load and strictly validate a small, versioned evaluation suite."""

    raw = load_json_object(path)
    allowed = {
        "schema_version",
        "id",
        "license",
        "quality_cases",
        "performance",
    }
    _reject_unknown(raw, allowed, "evaluation suite")
    if raw.get("schema_version") != "1.0":
        raise ValidationError("evaluation suite schema_version must be '1.0'")
    suite_id = _nonempty_string(raw.get("id"), "evaluation suite id")
    license_name = _nonempty_string(raw.get("license"), "evaluation suite license")

    raw_cases = raw.get("quality_cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValidationError("quality_cases must be a non-empty array")
    if len(raw_cases) > 100:
        raise ValidationError("quality_cases must contain at most 100 cases")
    cases: list[QualityCase] = []
    ids: set[str] = set()
    for index, item in enumerate(raw_cases):
        if not isinstance(item, Mapping):
            raise ValidationError(f"quality_cases[{index}] must be an object")
        _reject_unknown(
            item,
            {"id", "prompt", "accepted_answers", "match_mode"},
            f"quality_cases[{index}]",
        )
        case_id = _nonempty_string(item.get("id"), f"quality_cases[{index}].id")
        if case_id in ids:
            raise ValidationError(f"duplicate quality case id {case_id!r}")
        ids.add(case_id)
        prompt = _bounded_text(item.get("prompt"), f"quality case {case_id!r} prompt", 4096)
        raw_answers = item.get("accepted_answers")
        if not isinstance(raw_answers, list) or not raw_answers:
            raise ValidationError(
                f"quality case {case_id!r} accepted_answers must be a non-empty array"
            )
        if len(raw_answers) > 20:
            raise ValidationError(
                f"quality case {case_id!r} accepted_answers must contain at most 20 values"
            )
        answers = tuple(
            _bounded_text(value, f"quality case {case_id!r} accepted answer", 256)
            for value in raw_answers
        )
        match_mode = _quality_match_mode(
            item.get("match_mode", "normalized-text"),
            f"quality case {case_id!r} match_mode",
        )
        if match_mode == "json-exact" and any(
            _canonical_json(answer) is None for answer in answers
        ):
            raise ValidationError(
                f"quality case {case_id!r} json-exact accepted answers must be valid JSON"
            )
        cases.append(QualityCase(case_id, prompt, answers, match_mode))

    performance = raw.get("performance")
    if not isinstance(performance, Mapping):
        raise ValidationError("performance must be an object")
    _reject_unknown(
        performance,
        {"prompt", "generation_tokens", "repetitions", "warmups"},
        "performance",
    )
    performance_prompt = _bounded_text(performance.get("prompt"), "performance.prompt", 32768)
    generation_tokens = _bounded_int(
        performance.get("generation_tokens"),
        "performance.generation_tokens",
        minimum=8,
        maximum=512,
    )
    repetitions = _bounded_int(
        performance.get("repetitions"),
        "performance.repetitions",
        minimum=5,
        maximum=50,
    )
    warmups = _bounded_int(
        performance.get("warmups"),
        "performance.warmups",
        minimum=1,
        maximum=10,
    )
    return EvaluationSuite(
        suite_id=suite_id,
        license=license_name,
        quality_cases=tuple(cases),
        performance_prompt=performance_prompt,
        generation_tokens=generation_tokens,
        repetitions=repetitions,
        warmups=warmups,
    )


def evaluate_server(
    base_url: str,
    suite_path: Path,
    *,
    candidate_id: str,
    timeout_seconds: float = 180.0,
    clock: Callable[[], float] = time.perf_counter,
) -> Mapping[str, Any]:
    """Evaluate one running server and return raw, deterministic JSON data.

    Wall-clock samples naturally vary.  Given the same captured samples, later
    assembly and report generation are deterministic.
    """

    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        raise ValidationError("base_url must be an http or https URL")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValidationError("candidate_id must be a non-empty string")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValidationError("timeout_seconds must be a positive finite number")

    suite = load_evaluation_suite(suite_path)
    quality_rows: list[Mapping[str, Any]] = []
    passed = 0
    for case in suite.quality_cases:
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "Follow the user instruction exactly and answer concisely.",
                },
                {"role": "user", "content": case.prompt},
            ],
            "max_tokens": 32,
            "seed": 4242,
            "temperature": 0,
            "stream": False,
        }
        response = _post_json(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            payload,
            timeout_seconds,
        )
        content = _chat_content(response, case.case_id)
        matched_answer = next(
            (
                answer
                for answer in case.accepted_answers
                if _answer_matches(content, answer, case.match_mode)
            ),
            None,
        )
        matched = matched_answer is not None
        passed += int(matched)
        quality_rows.append(
            {
                "id": case.case_id,
                "prompt": case.prompt,
                "accepted_answers": list(case.accepted_answers),
                "match_mode": case.match_mode,
                "response": content,
                "matched": matched,
                "matched_answer": matched_answer,
            }
        )

    performance_payload = {
        "prompt": suite.performance_prompt,
        "n_predict": suite.generation_tokens,
        "ignore_eos": True,
        "seed": 4242,
        "temperature": 0,
        "cache_prompt": False,
        "stream": True,
        "timings_per_token": False,
    }
    for _ in range(suite.warmups):
        _stream_completion(
            f"{base_url.rstrip('/')}/completion",
            performance_payload,
            timeout_seconds,
            clock,
            expected_predicted_tokens=suite.generation_tokens,
        )

    samples: list[Mapping[str, Any]] = []
    for index in range(suite.repetitions):
        sample = dict(
            _stream_completion(
                f"{base_url.rstrip('/')}/completion",
                performance_payload,
                timeout_seconds,
                clock,
                expected_predicted_tokens=suite.generation_tokens,
            )
        )
        sample["index"] = index + 1
        samples.append(sample)

    ttft_values = [float(sample["ttft_ms"]) for sample in samples]
    e2e_values = [float(sample["e2e_latency_ms"]) for sample in samples]
    return {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "synthetic": False,
        "suite": {
            "id": suite.suite_id,
            "license": suite.license,
            "sha256": sha256_file(suite_path),
            "quality_case_count": len(suite.quality_cases),
            "performance_repetitions": suite.repetitions,
            "performance_warmups": suite.warmups,
            "generation_tokens": suite.generation_tokens,
            "cache_prompt": False,
            "seed": 4242,
            "temperature": 0,
        },
        "quality": {
            "method": "fixed exact-answer smoke evaluation",
            "score": passed / len(suite.quality_cases),
            "passed": passed,
            "total": len(suite.quality_cases),
            "cases": quality_rows,
        },
        "latency": {
            "method": "single-client streamed HTTP requests",
            "ttft_ms_p50": _percentile(ttft_values, 50),
            "ttft_ms_p95": _percentile(ttft_values, 95),
            "e2e_latency_ms_p50": _percentile(e2e_values, 50),
            "e2e_latency_ms_p95": _percentile(e2e_values, 95),
            "samples": samples,
        },
    }


def pool_server_evaluations(paths: Sequence[Path]) -> Mapping[str, Any]:
    """Pool compatible balanced-pass server evaluations.

    Each input remains an independent raw artifact.  The returned mapping uses
    the existing server-evaluation schema, retaining the first pass's quality
    responses while requiring every pass to agree on case identity and match
    outcome.  Latency samples from all passes are copied, renumbered, and used
    to recompute the aggregate percentiles.
    """

    input_paths = [Path(path) for path in paths]
    if not 2 <= len(input_paths) <= 8:
        raise ValidationError("server evaluation pooling requires from 2 to 8 inputs")

    resolved_inputs: list[str] = []
    evaluations: list[Mapping[str, Any]] = []
    for path in input_paths:
        try:
            resolved = str(path.resolve(strict=True)).casefold()
        except OSError as exc:
            raise ValidationError(f"could not resolve server evaluation {path}: {exc}") from exc
        if resolved in resolved_inputs:
            raise ValidationError("server evaluation input paths must be distinct")
        resolved_inputs.append(resolved)
        try:
            raw = load_json_object(path)
        except ValidationError as exc:
            raise ValidationError(f"invalid server evaluation {path}: {exc}") from exc
        _validate_server_evaluation_artifact(raw, f"server evaluation {path}")
        evaluations.append(raw)

    first = evaluations[0]
    first_candidate = str(first["candidate_id"])
    first_suite = first["suite"]
    first_quality = first["quality"]
    first_latency = first["latency"]
    assert isinstance(first_suite, Mapping)
    assert isinstance(first_quality, Mapping)
    assert isinstance(first_latency, Mapping)

    first_cases = first_quality["cases"]
    assert isinstance(first_cases, list)
    quality_identity = [
        (
            case["id"],
            case["prompt"],
            tuple(case["accepted_answers"]),
            case.get("match_mode", "normalized-text"),
        )
        for case in first_cases
    ]
    quality_outcomes = [(case["matched"], case["matched_answer"]) for case in first_cases]

    pooled_samples: list[dict[str, Any]] = []
    repetitions_per_pass = int(first_suite["performance_repetitions"])
    warmups_per_pass = int(first_suite["performance_warmups"])
    for index, raw in enumerate(evaluations, start=1):
        if raw["candidate_id"] != first_candidate:
            raise ValidationError(
                f"server evaluation input {index} candidate_id does not match the first input"
            )
        suite = raw["suite"]
        quality = raw["quality"]
        latency = raw["latency"]
        assert isinstance(suite, Mapping)
        assert isinstance(quality, Mapping)
        assert isinstance(latency, Mapping)
        if suite != first_suite:
            raise ValidationError(
                f"server evaluation input {index} suite identity, settings, and per-pass "
                "counts must match the first input"
            )
        if quality["method"] != first_quality["method"]:
            raise ValidationError(
                f"server evaluation input {index} quality method does not match the first input"
            )
        cases = quality["cases"]
        assert isinstance(cases, list)
        identity = [
            (
                case["id"],
                case["prompt"],
                tuple(case["accepted_answers"]),
                case.get("match_mode", "normalized-text"),
            )
            for case in cases
        ]
        if identity != quality_identity:
            raise ValidationError(
                f"server evaluation input {index} quality case identities do not match "
                "the first input"
            )
        outcomes = [(case["matched"], case["matched_answer"]) for case in cases]
        if outcomes != quality_outcomes:
            raise ValidationError(
                f"server evaluation input {index} quality matched outcomes do not match "
                "the first input"
            )
        if latency["method"] != first_latency["method"]:
            raise ValidationError(
                f"server evaluation input {index} latency method does not match the first input"
            )
        samples = latency["samples"]
        assert isinstance(samples, list)
        for sample in samples:
            copied = deepcopy(dict(sample))
            copied["index"] = len(pooled_samples) + 1
            pooled_samples.append(copied)

    ttft_values = [float(sample["ttft_ms"]) for sample in pooled_samples]
    e2e_values = [float(sample["e2e_latency_ms"]) for sample in pooled_samples]
    pooled_suite = deepcopy(dict(first_suite))
    pooled_suite["performance_repetitions"] = repetitions_per_pass * len(evaluations)
    pooled_suite["performance_warmups"] = warmups_per_pass * len(evaluations)
    return {
        "schema_version": "1.0",
        "candidate_id": first_candidate,
        "synthetic": False,
        "suite": pooled_suite,
        "quality": deepcopy(dict(first_quality)),
        "latency": {
            "method": first_latency["method"],
            "ttft_ms_p50": _percentile(ttft_values, 50),
            "ttft_ms_p95": _percentile(ttft_values, 95),
            "e2e_latency_ms_p50": _percentile(e2e_values, 50),
            "e2e_latency_ms_p95": _percentile(e2e_values, 95),
            "samples": pooled_samples,
        },
    }


def validate_server_evaluation(raw: Mapping[str, Any]) -> None:
    """Validate one serialized server evaluation without pooling it."""

    _validate_server_evaluation_artifact(raw, "server evaluation")


def _validate_server_evaluation_artifact(raw: Mapping[str, Any], context: str) -> None:
    """Validate the complete schema before an artifact can enter a pool."""

    _require_exact_fields(
        raw,
        {"schema_version", "candidate_id", "synthetic", "suite", "quality", "latency"},
        context,
    )
    if raw.get("schema_version") != "1.0":
        raise ValidationError(f"{context}.schema_version must be '1.0'")
    _nonempty_string(raw.get("candidate_id"), f"{context}.candidate_id")
    if raw.get("synthetic") is not False:
        raise ValidationError(f"{context}.synthetic must be false")

    suite = _mapping(raw.get("suite"), f"{context}.suite")
    _require_exact_fields(
        suite,
        {
            "id",
            "license",
            "sha256",
            "quality_case_count",
            "performance_repetitions",
            "performance_warmups",
            "generation_tokens",
            "cache_prompt",
            "seed",
            "temperature",
        },
        f"{context}.suite",
    )
    _nonempty_string(suite.get("id"), f"{context}.suite.id")
    _nonempty_string(suite.get("license"), f"{context}.suite.license")
    digest = _nonempty_string(suite.get("sha256"), f"{context}.suite.sha256")
    if _SHA256.fullmatch(digest) is None:
        raise ValidationError(f"{context}.suite.sha256 must be a lowercase SHA-256 digest")
    case_count = _integer_between(
        suite.get("quality_case_count"),
        f"{context}.suite.quality_case_count",
        minimum=1,
        maximum=100,
    )
    repetitions = _integer_between(
        suite.get("performance_repetitions"),
        f"{context}.suite.performance_repetitions",
        minimum=1,
        maximum=10_000,
    )
    _integer_between(
        suite.get("performance_warmups"),
        f"{context}.suite.performance_warmups",
        minimum=1,
        maximum=10_000,
    )
    generation_tokens = _integer_between(
        suite.get("generation_tokens"),
        f"{context}.suite.generation_tokens",
        minimum=1,
        maximum=4096,
    )
    if suite.get("cache_prompt") is not False:
        raise ValidationError(f"{context}.suite.cache_prompt must be false")
    _integer_between(
        suite.get("seed"),
        f"{context}.suite.seed",
        minimum=0,
        maximum=2**63 - 1,
    )
    if _finite_number(suite.get("temperature"), f"{context}.suite.temperature") != 0.0:
        raise ValidationError(f"{context}.suite.temperature must be zero")

    quality = _mapping(raw.get("quality"), f"{context}.quality")
    _require_exact_fields(
        quality,
        {"method", "score", "passed", "total", "cases"},
        f"{context}.quality",
    )
    _nonempty_string(quality.get("method"), f"{context}.quality.method")
    score = _finite_number(
        quality.get("score"),
        f"{context}.quality.score",
        minimum=0.0,
        maximum=1.0,
    )
    passed = _integer_between(
        quality.get("passed"),
        f"{context}.quality.passed",
        minimum=0,
        maximum=case_count,
    )
    total = _integer_between(
        quality.get("total"),
        f"{context}.quality.total",
        minimum=1,
        maximum=100,
    )
    if total != case_count:
        raise ValidationError(f"{context}.quality.total must match suite quality_case_count")
    if not math.isclose(score, passed / total, rel_tol=1e-12, abs_tol=1e-12):
        raise ValidationError(f"{context}.quality.score must match passed/total")
    cases = quality.get("cases")
    if not isinstance(cases, list) or len(cases) != total:
        raise ValidationError(f"{context}.quality.cases must contain exactly {total} cases")
    seen_case_ids: set[str] = set()
    matched_count = 0
    for index, case_value in enumerate(cases):
        case_context = f"{context}.quality.cases[{index}]"
        case = _mapping(case_value, case_context)
        required_case_fields = {
            "id",
            "prompt",
            "accepted_answers",
            "response",
            "matched",
            "matched_answer",
        }
        missing_case_fields = sorted(required_case_fields - set(case))
        unknown_case_fields = sorted(set(case) - required_case_fields - {"match_mode"})
        if missing_case_fields or unknown_case_fields:
            details: list[str] = []
            if missing_case_fields:
                details.append("missing fields: " + ", ".join(missing_case_fields))
            if unknown_case_fields:
                details.append("unknown fields: " + ", ".join(unknown_case_fields))
            raise ValidationError(f"{case_context} has {'; '.join(details)}")
        case_id = _nonempty_string(case.get("id"), f"{case_context}.id")
        if case_id in seen_case_ids:
            raise ValidationError(f"{context}.quality.cases contains duplicate id {case_id!r}")
        seen_case_ids.add(case_id)
        _nonempty_string(case.get("prompt"), f"{case_context}.prompt")
        answers = case.get("accepted_answers")
        if not isinstance(answers, list) or not answers or len(answers) > 20:
            raise ValidationError(
                f"{case_context}.accepted_answers must contain from 1 to 20 strings"
            )
        accepted_answers = [
            _nonempty_string(answer, f"{case_context}.accepted_answers[{answer_index}]")
            for answer_index, answer in enumerate(answers)
        ]
        match_mode = _quality_match_mode(
            case.get("match_mode", "normalized-text"),
            f"{case_context}.match_mode",
        )
        if match_mode == "json-exact" and any(
            _canonical_json(answer) is None for answer in accepted_answers
        ):
            raise ValidationError(
                f"{case_context}.accepted_answers must be valid JSON for json-exact"
            )
        response = case.get("response")
        if not isinstance(response, str):
            raise ValidationError(f"{case_context}.response must be a string")
        matched = case.get("matched")
        if not isinstance(matched, bool):
            raise ValidationError(f"{case_context}.matched must be a boolean")
        matched_answer = case.get("matched_answer")
        matching_answers = [
            answer for answer in accepted_answers if _answer_matches(response, answer, match_mode)
        ]
        if matched:
            if not isinstance(matched_answer, str) or matched_answer not in accepted_answers:
                raise ValidationError(
                    f"{case_context}.matched_answer must be one accepted answer when matched"
                )
            if matched_answer not in matching_answers:
                raise ValidationError(f"{case_context}.matched outcome does not match response")
        elif matched_answer is not None:
            raise ValidationError(f"{case_context}.matched_answer must be null when unmatched")
        elif matching_answers:
            raise ValidationError(f"{case_context}.unmatched outcome does not match response")
        matched_count += int(matched)
    if matched_count != passed:
        raise ValidationError(f"{context}.quality.passed must match case outcomes")

    latency = _mapping(raw.get("latency"), f"{context}.latency")
    _require_exact_fields(
        latency,
        {
            "method",
            "ttft_ms_p50",
            "ttft_ms_p95",
            "e2e_latency_ms_p50",
            "e2e_latency_ms_p95",
            "samples",
        },
        f"{context}.latency",
    )
    _nonempty_string(latency.get("method"), f"{context}.latency.method")
    samples = latency.get("samples")
    if not isinstance(samples, list) or len(samples) != repetitions:
        raise ValidationError(
            f"{context}.latency.samples must contain exactly {repetitions} samples"
        )
    ttft_values: list[float] = []
    e2e_values: list[float] = []
    for index, sample_value in enumerate(samples, start=1):
        sample_context = f"{context}.latency.samples[{index - 1}]"
        sample = _mapping(sample_value, sample_context)
        _require_exact_fields(
            sample,
            {
                "index",
                "ttft_ms",
                "e2e_latency_ms",
                "event_count",
                "predicted_tokens",
                "content",
            },
            sample_context,
        )
        if (
            _integer_between(
                sample.get("index"), sample_context + ".index", minimum=1, maximum=repetitions
            )
            != index
        ):
            raise ValidationError(f"{context}.latency sample indexes must be consecutive from one")
        ttft = _finite_number(
            sample.get("ttft_ms"), sample_context + ".ttft_ms", strictly_positive=True
        )
        e2e = _finite_number(
            sample.get("e2e_latency_ms"),
            sample_context + ".e2e_latency_ms",
            strictly_positive=True,
        )
        if e2e < ttft:
            raise ValidationError(f"{sample_context}.e2e_latency_ms cannot be below TTFT")
        _integer_between(
            sample.get("event_count"),
            sample_context + ".event_count",
            minimum=1,
            maximum=10_000_000,
        )
        predicted_tokens = _integer_between(
            sample.get("predicted_tokens"),
            sample_context + ".predicted_tokens",
            minimum=1,
            maximum=4096,
        )
        if predicted_tokens != generation_tokens:
            raise ValidationError(
                f"{sample_context}.predicted_tokens must equal suite generation_tokens"
            )
        _nonempty_string(sample.get("content"), sample_context + ".content")
        ttft_values.append(ttft)
        e2e_values.append(e2e)

    expected_aggregates = {
        "ttft_ms_p50": _percentile(ttft_values, 50),
        "ttft_ms_p95": _percentile(ttft_values, 95),
        "e2e_latency_ms_p50": _percentile(e2e_values, 50),
        "e2e_latency_ms_p95": _percentile(e2e_values, 95),
    }
    for name, expected in expected_aggregates.items():
        actual = _finite_number(
            latency.get(name), f"{context}.latency.{name}", strictly_positive=True
        )
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValidationError(f"{context}.latency.{name} does not match raw samples")


def parse_gnu_time_peak_rss(path: Path) -> float:
    """Read GNU ``time -v`` output and return maximum RSS in MiB."""

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"GNU time output must be UTF-8 text: {path}") from exc
    except OSError as exc:
        raise ValidationError(f"could not read GNU time output {path}: {exc}") from exc
    matches = re.findall(r"^\s*Maximum resident set size \(kbytes\):\s*([0-9]+)\s*$", text, re.M)
    if len(matches) != 1:
        raise ValidationError("GNU time output must contain exactly one maximum RSS value")
    kib = int(matches[0])
    if kib <= 0:
        raise ValidationError("GNU time maximum RSS must be positive")
    return kib / 1024.0


def _post_json(url: str, payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
    request = Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise ValidationError(f"server returned HTTP {exc.code} for {url}: {detail}") from exc
    except (OSError, URLError) as exc:
        raise ValidationError(f"could not reach {url}: {exc}") from exc
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValidationError("server response exceeded the 4 MiB safety limit")
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"server returned invalid JSON for {url}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValidationError(f"server response for {url} must be an object")
    return raw


def _stream_completion(
    url: str,
    payload: Mapping[str, Any],
    timeout: float,
    clock: Callable[[], float],
    *,
    expected_predicted_tokens: int,
) -> Mapping[str, Any]:
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
    request = Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    start = clock()
    first_content_at: float | None = None
    content_parts: list[str] = []
    event_count = 0
    total_bytes = 0
    final_event: Mapping[str, Any] | None = None
    try:
        with urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                total_bytes += len(raw_line)
                if total_bytes > MAX_RESPONSE_BYTES:
                    raise ValidationError(
                        "streamed server response exceeded the 4 MiB safety limit"
                    )
                line = raw_line.decode("utf-8").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ValidationError(f"server returned invalid SSE JSON: {exc}") from exc
                if not isinstance(event, Mapping):
                    raise ValidationError("server SSE event must be an object")
                final_event = event
                event_count += 1
                content = event.get("content", "")
                if not isinstance(content, str):
                    raise ValidationError("server SSE content must be a string")
                if content and first_content_at is None:
                    first_content_at = clock()
                content_parts.append(content)
    except HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise ValidationError(f"server returned HTTP {exc.code} for {url}: {detail}") from exc
    except ValidationError:
        raise
    except (OSError, URLError, UnicodeDecodeError) as exc:
        raise ValidationError(f"could not read streamed response from {url}: {exc}") from exc
    end = clock()
    if first_content_at is None:
        raise ValidationError("server stream completed without generated content")
    if final_event is None:
        raise ValidationError("server stream completed without a JSON event")
    timings = final_event.get("timings")
    if not isinstance(timings, Mapping):
        raise ValidationError("final server SSE event is missing timings")
    predicted_tokens = timings.get("predicted_n")
    if isinstance(predicted_tokens, bool) or not isinstance(predicted_tokens, int):
        raise ValidationError("final server SSE timings.predicted_n must be an integer")
    if predicted_tokens != expected_predicted_tokens:
        raise ValidationError(
            "server generated "
            f"{predicted_tokens} tokens; expected exactly {expected_predicted_tokens}"
        )
    return {
        "ttft_ms": (first_content_at - start) * 1000.0,
        "e2e_latency_ms": (end - start) * 1000.0,
        "event_count": event_count,
        "predicted_tokens": predicted_tokens,
        "content": "".join(content_parts),
    }


def _chat_content(response: Mapping[str, Any], case_id: str) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValidationError(f"quality case {case_id!r} response must contain one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValidationError(f"quality case {case_id!r} choice must be an object")
    message = choice.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValidationError(f"quality case {case_id!r} response is missing message content")
    return str(message["content"])


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        raise ValidationError("cannot compute a percentile without samples")
    if percentile == 50:
        return float(statistics.median(values))
    ordered = sorted(values)
    rank = math.ceil((percentile / 100.0) * len(ordered))
    return float(ordered[max(0, rank - 1)])


def _normalize_answer(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _quality_match_mode(value: Any, context: str) -> str:
    if value not in {"normalized-text", "trimmed-exact", "json-exact"}:
        raise ValidationError(
            f"{context} must be 'normalized-text', 'trimmed-exact', or 'json-exact'"
        )
    return str(value)


def quality_answer_matches(
    response: str,
    accepted_answer: str,
    match_mode: str,
) -> bool:
    """Apply one validated deterministic quality-case matching contract."""

    if match_mode == "normalized-text":
        return _normalize_answer(response) == _normalize_answer(accepted_answer)
    if match_mode == "trimmed-exact":
        return response.strip() == accepted_answer.strip()
    if match_mode == "json-exact":
        response_json = _canonical_json(response)
        accepted_json = _canonical_json(accepted_answer)
        return response_json is not None and response_json == accepted_json
    raise ValidationError(f"unknown quality match mode {match_mode!r}")


def _answer_matches(response: str, accepted_answer: str, match_mode: str) -> bool:
    """Backward-compatible private alias used by focused unit tests."""

    return quality_answer_matches(response, accepted_answer, match_mode)


def _canonical_json(value: str) -> str | None:
    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
        parsed: dict[str, Any] = {}
        for key, item in pairs:
            if key in parsed:
                raise ValueError(f"duplicate JSON key {key!r}")
            parsed[key] = item
        return parsed

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    try:
        parsed = json.loads(
            value.strip(),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
        return json.dumps(
            parsed,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(str(key) for key in set(raw) - allowed)
    if unknown:
        raise ValidationError(f"{context} contains unknown fields: {', '.join(unknown)}")


def _require_exact_fields(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(raw)
    missing = sorted(expected - actual)
    unknown = sorted(str(key) for key in actual - expected)
    details: list[str] = []
    if missing:
        details.append("missing fields: " + ", ".join(missing))
    if unknown:
        details.append("unknown fields: " + ", ".join(unknown))
    if details:
        raise ValidationError(f"{context} has " + "; ".join(details))


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{context} must be an object")
    return value


def _integer_between(value: Any, context: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{context} must be an integer")
    if value < minimum or value > maximum:
        raise ValidationError(f"{context} must be from {minimum} to {maximum}")
    return value


def _finite_number(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{context} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{context} must be a finite number")
    if strictly_positive and number <= 0:
        raise ValidationError(f"{context} must be positive")
    if minimum is not None and number < minimum:
        raise ValidationError(f"{context} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValidationError(f"{context} must be at most {maximum}")
    return number


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _bounded_text(value: Any, field_name: str, maximum: int) -> str:
    text = _nonempty_string(value, field_name)
    if len(text) > maximum:
        raise ValidationError(f"{field_name} must contain at most {maximum} characters")
    return text


def _bounded_int(value: Any, field_name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    if value < minimum or value > maximum:
        raise ValidationError(f"{field_name} must be from {minimum} to {maximum}")
    return value
