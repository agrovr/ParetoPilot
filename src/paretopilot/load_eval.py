"""Bounded, deterministic load-sweep evaluation primitives.

The strict aggregation accepts an injected request runner, which keeps the
concurrency scheduler easy to test and reusable with compatible inference
servers.  A small standard-library adapter provides the live ``llama-server``
SSE path used by the CLI.

Every successful runner result represents exactly the requested output-token
count.  Expected request failures are returned as explicit failed measurements;
malformed measurements and unexpected runner exceptions fail the sweep closed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from paretopilot.domain import ValidationError
from paretopilot.io import load_json_object, sha256_file


_ALLOWED_CONCURRENCY = {1, 2, 4}
_MAX_PROMPTS = 16
_MAX_REQUESTS_PER_LEVEL = 256
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_SERVER_ARGV_ITEMS = 128
_MAX_SERVER_ARG_LENGTH = 4096
_ALLOWED_BINDING_OPTIONS = ("--host", "--port")


@dataclass(frozen=True)
class LoadRequest:
    """One request issued by a bounded load sweep."""

    prompt: str
    prompt_index: int
    output_tokens: int
    concurrency: int
    request_index: int
    warmup: bool


RequestRunner = Callable[[LoadRequest], Mapping[str, Any]]


@dataclass(frozen=True)
class LoadPlan:
    """Strict settings loaded by the native load-evaluation CLI."""

    prompts: tuple[str, ...]
    output_tokens: int
    warmup_requests_per_level: int
    measured_requests_per_level: int
    concurrency_levels: tuple[int, ...]
    timeout_seconds: float
    slo: Mapping[str, Any]


def load_load_plan(path: Path) -> LoadPlan:
    """Load a bounded load plan from strict JSON."""

    raw = load_json_object(path)
    _require_exact_fields(
        raw,
        {
            "schema_version",
            "prompts",
            "output_tokens",
            "warmup_requests_per_level",
            "measured_requests_per_level",
            "concurrency_levels",
            "timeout_seconds",
            "slo",
        },
        "load plan",
    )
    if raw.get("schema_version") != "1.0":
        raise ValidationError("load plan schema_version must be '1.0'")
    prompts = _validate_prompts(raw.get("prompts"))
    output_tokens = _integer_between(
        raw.get("output_tokens"),
        "load plan output_tokens",
        minimum=8,
        maximum=512,
    )
    warmups = _integer_between(
        raw.get("warmup_requests_per_level"),
        "load plan warmup_requests_per_level",
        minimum=0,
        maximum=32,
    )
    levels = _validate_concurrency_levels(raw.get("concurrency_levels"))
    requests = _integer_between(
        raw.get("measured_requests_per_level"),
        "load plan measured_requests_per_level",
        minimum=max(levels),
        maximum=_MAX_REQUESTS_PER_LEVEL,
    )
    for concurrency in levels:
        if requests % concurrency != 0:
            raise ValidationError(
                "load plan measured_requests_per_level must be divisible by every concurrency level"
            )
    timeout = _finite_number(
        raw.get("timeout_seconds"),
        "load plan timeout_seconds",
        strictly_positive=True,
    )
    if timeout > 3600.0:
        raise ValidationError("load plan timeout_seconds must be at most 3600")
    slo = _validate_slo(raw.get("slo"), "load plan slo")
    return LoadPlan(
        prompts=prompts,
        output_tokens=output_tokens,
        warmup_requests_per_level=warmups,
        measured_requests_per_level=requests,
        concurrency_levels=levels,
        timeout_seconds=timeout,
        slo=slo,
    )


def load_plan_evaluation_contract(plan: LoadPlan) -> Mapping[str, Any]:
    """Return the exact methodology and SLO a validated plan must produce.

    The live-request timeout is intentionally not repeated in the measured
    artifact. It remains bound through the checksummed plan SHA-256 recorded in
    the evidence binding.
    """

    if not isinstance(plan, LoadPlan):
        raise ValidationError("plan must be a validated LoadPlan")
    prompts = _validate_prompts(plan.prompts)
    levels = _validate_concurrency_levels(plan.concurrency_levels)
    output_tokens = _integer_between(
        plan.output_tokens,
        "load plan output_tokens",
        minimum=8,
        maximum=512,
    )
    warmups = _integer_between(
        plan.warmup_requests_per_level,
        "load plan warmup_requests_per_level",
        minimum=0,
        maximum=32,
    )
    requests = _integer_between(
        plan.measured_requests_per_level,
        "load plan measured_requests_per_level",
        minimum=max(levels),
        maximum=_MAX_REQUESTS_PER_LEVEL,
    )
    for concurrency in levels:
        if requests % concurrency != 0:
            raise ValidationError(
                "load plan measured_requests_per_level must be divisible by every concurrency level"
            )
    methodology = {
        "concurrency_levels": list(levels),
        "prompts": [
            {
                "index": index,
                "text": prompt,
                "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
            for index, prompt in enumerate(prompts, start=1)
        ],
        "output_tokens": output_tokens,
        "warmup_requests_per_level": warmups,
        "measured_requests_per_level": requests,
        "prompt_schedule": "round-robin",
        "client_scheduler": "bounded thread pool",
    }
    return {
        "methodology": methodology,
        "slo": _validate_slo(plan.slo, "load plan slo"),
    }


def validate_load_artifact_against_plan(
    raw: Mapping[str, Any],
    plan: LoadPlan,
    *,
    context: str = "load evaluation",
) -> None:
    """Require a single or combined load artifact to match its plan exactly."""

    if not isinstance(raw, Mapping):
        raise ValidationError(f"{context} must be an object")
    expected = load_plan_evaluation_contract(plan)
    if raw.get("methodology") != expected["methodology"]:
        raise ValidationError(f"{context} methodology does not match load-plan.json")
    if raw.get("slo") != expected["slo"]:
        raise ValidationError(f"{context} SLO does not match load-plan.json")


def load_server_command(path: Path) -> tuple[str, ...]:
    """Load one strict ``llama-server`` command document."""

    raw = load_json_object(path)
    _require_exact_fields(raw, {"schema_version", "argv"}, "server command")
    if raw.get("schema_version") != "1.0":
        raise ValidationError("server command schema_version must be '1.0'")
    return _validate_server_argv(raw.get("argv"), "server command argv")


def build_load_evidence_binding(
    *,
    base_url: str,
    plan_path: Path,
    server_command_path: Path,
    canonical_server_command_path: Path,
) -> Mapping[str, Any]:
    """Bind a load run to its plan and materially equivalent server commands."""

    # Validate all three source documents before recording their exact byte hashes.
    load_load_plan(plan_path)
    load_argv = load_server_command(server_command_path)
    canonical_argv = load_server_command(canonical_server_command_path)
    server_configuration = _derive_server_configuration(
        load_argv=load_argv,
        canonical_argv=canonical_argv,
        load_command_sha256=sha256_file(server_command_path),
        canonical_command_sha256=sha256_file(canonical_server_command_path),
    )
    request_base_url = _normalize_request_base_url(base_url)
    _validate_request_endpoint(
        request_base_url,
        server_configuration,
        "load evidence binding",
    )
    binding: Mapping[str, Any] = {
        "plan_sha256": sha256_file(plan_path),
        "request_base_url": request_base_url,
        "server_configuration": server_configuration,
    }
    _validate_evidence_binding(binding, "load evidence binding")
    return binding


def evaluate_llama_server_load(
    base_url: str,
    plan: LoadPlan,
    *,
    candidate_id: str,
    evidence_binding: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Evaluate one live ``llama-server`` using a validated load plan."""

    if not isinstance(plan, LoadPlan):
        raise ValidationError("plan must be a validated LoadPlan")
    if evidence_binding is not None:
        normalized_base_url = _normalize_request_base_url(base_url)
        if evidence_binding.get("request_base_url") != normalized_base_url:
            raise ValidationError("live load base_url does not match its evidence binding")
    runner = llama_server_request_runner(
        base_url,
        timeout_seconds=plan.timeout_seconds,
    )
    return evaluate_load(
        candidate_id=candidate_id,
        prompts=plan.prompts,
        output_tokens=plan.output_tokens,
        warmup_requests_per_level=plan.warmup_requests_per_level,
        measured_requests_per_level=plan.measured_requests_per_level,
        request_runner=runner,
        slo=plan.slo,
        concurrency_levels=plan.concurrency_levels,
        synthetic=False,
        evidence_binding=evidence_binding,
    )


def llama_server_request_runner(
    base_url: str,
    *,
    timeout_seconds: float = 180.0,
    clock: Callable[[], float] = time.perf_counter,
) -> RequestRunner:
    """Create a standard-library SSE runner for ``llama-server``.

    HTTP and network failures become explicit failed samples with measured
    timestamps.  Invalid JSON, invalid SSE fields, and contradictory token
    counts remain validation errors and fail the evaluation closed.
    """

    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        raise ValidationError("base_url must be an http or https URL")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValidationError("timeout_seconds must be a positive finite number")
    if not callable(clock):
        raise ValidationError("clock must be callable")
    endpoint = f"{base_url.rstrip('/')}/completion"

    def run(request_spec: LoadRequest) -> Mapping[str, Any]:
        if not isinstance(request_spec, LoadRequest):
            raise ValidationError("llama-server request must be a LoadRequest")
        payload = {
            "prompt": request_spec.prompt,
            "n_predict": request_spec.output_tokens,
            "ignore_eos": True,
            "seed": 4242,
            "temperature": 0,
            "cache_prompt": False,
            "stream": True,
            "timings_per_token": False,
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        http_request = Request(
            endpoint,
            data=encoded,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        started = clock()
        first_content_at: float | None = None
        final_event: Mapping[str, Any] | None = None
        observed_tokens = 0
        total_bytes = 0
        try:
            with urlopen(http_request, timeout=timeout_seconds) as response:
                for raw_line in response:
                    total_bytes += len(raw_line)
                    if total_bytes > _MAX_RESPONSE_BYTES:
                        raise ValidationError(
                            "streamed server response exceeded the 4 MiB safety limit"
                        )
                    try:
                        line = raw_line.decode("utf-8").strip()
                    except UnicodeDecodeError as exc:
                        raise ValidationError("server returned non-UTF-8 SSE data") from exc
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
                    content = event.get("content", "")
                    if not isinstance(content, str):
                        raise ValidationError("server SSE content must be a string")
                    tokens = event.get("tokens")
                    if tokens is not None:
                        if not isinstance(tokens, list) or any(
                            isinstance(token, bool) or not isinstance(token, int)
                            for token in tokens
                        ):
                            raise ValidationError("server SSE tokens must be an array of integers")
                        observed_tokens += len(tokens)
                    if content and first_content_at is None:
                        first_content_at = clock()
                    final_event = event
        except HTTPError as exc:
            finished = clock()
            try:
                detail = exc.read(4096).decode("utf-8", errors="replace").strip()
            except OSError:
                detail = ""
            message = f"HTTP {exc.code}"
            if detail:
                message += f": {detail}"
            return _failed_network_result(
                started=started,
                finished=finished,
                first_content_at=first_content_at,
                generated_tokens=observed_tokens,
                error=message,
            )
        except ValidationError:
            raise
        except (OSError, URLError, TimeoutError) as exc:
            finished = clock()
            return _failed_network_result(
                started=started,
                finished=finished,
                first_content_at=first_content_at,
                generated_tokens=observed_tokens,
                error=f"network error: {type(exc).__name__}: {exc}",
            )

        finished = clock()
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
        if predicted_tokens < 0:
            raise ValidationError("final server SSE timings.predicted_n must not be negative")
        if observed_tokens and observed_tokens != predicted_tokens:
            raise ValidationError("server SSE token arrays do not match final timings.predicted_n")
        completed = predicted_tokens == request_spec.output_tokens
        return {
            "completed": completed,
            "ttft_ms": (first_content_at - started) * 1000.0,
            "e2e_latency_ms": (finished - started) * 1000.0,
            "generated_tokens": predicted_tokens,
            "error": (
                None
                if completed
                else (
                    f"server generated {predicted_tokens} tokens; expected exactly "
                    f"{request_spec.output_tokens}"
                )
            ),
            "started_at_seconds": started,
            "finished_at_seconds": finished,
        }

    return run


def evaluate_load(
    *,
    candidate_id: str,
    prompts: Sequence[str],
    output_tokens: int,
    warmup_requests_per_level: int,
    measured_requests_per_level: int,
    request_runner: RequestRunner,
    slo: Mapping[str, Any],
    concurrency_levels: Sequence[int] = (1, 2, 4),
    peak_rss_mib_by_concurrency: Mapping[int, float] | None = None,
    synthetic: bool,
    evidence_binding: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Run and strictly aggregate a bounded 1/2/4-client load sweep.

    ``request_runner`` must return a mapping with these exact fields:
    ``completed``, ``ttft_ms``, ``e2e_latency_ms``, ``generated_tokens``,
    ``error``, ``started_at_seconds``, and ``finished_at_seconds``.  Timestamps
    must share one monotonic clock origin across the sweep.

    A failed request uses ``completed=False``, a non-empty ``error``, and may
    use ``None`` for ``ttft_ms``.  Runner exceptions are not silently converted
    into samples because doing so would discard the timing evidence needed for
    trustworthy throughput calculations.
    """

    normalized_candidate = _nonempty_text(candidate_id, "candidate_id", maximum=128)
    normalized_prompts = _validate_prompts(prompts)
    normalized_output_tokens = _integer_between(
        output_tokens,
        "output_tokens",
        minimum=8,
        maximum=512,
    )
    normalized_warmups = _integer_between(
        warmup_requests_per_level,
        "warmup_requests_per_level",
        minimum=0,
        maximum=32,
    )
    normalized_levels = _validate_concurrency_levels(concurrency_levels)
    normalized_requests = _integer_between(
        measured_requests_per_level,
        "measured_requests_per_level",
        minimum=max(normalized_levels),
        maximum=_MAX_REQUESTS_PER_LEVEL,
    )
    for concurrency in normalized_levels:
        if normalized_requests % concurrency != 0:
            raise ValidationError(
                "measured_requests_per_level must be divisible by every concurrency level"
            )
    if not callable(request_runner):
        raise ValidationError("request_runner must be callable")
    if not isinstance(synthetic, bool):
        raise ValidationError("synthetic must be a boolean")
    normalized_binding = _normalize_evidence_binding(
        evidence_binding,
        "evidence_binding",
    )
    normalized_slo = _validate_slo(slo, "slo")
    normalized_rss = _validate_peak_rss(
        peak_rss_mib_by_concurrency,
        normalized_levels,
    )

    prompt_rows = [
        {
            "index": index,
            "text": prompt,
            "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        for index, prompt in enumerate(normalized_prompts, start=1)
    ]
    rows: list[Mapping[str, Any]] = []
    for concurrency in normalized_levels:
        warmup_requests = _make_requests(
            normalized_prompts,
            output_tokens=normalized_output_tokens,
            concurrency=concurrency,
            count=normalized_warmups,
            warmup=True,
        )
        warmup_results = _run_requests(
            warmup_requests,
            concurrency=concurrency,
            request_runner=request_runner,
        )
        for request, raw_result in zip(warmup_requests, warmup_results, strict=True):
            result = _validate_runner_result(
                raw_result,
                output_tokens=normalized_output_tokens,
                context=(f"concurrency {concurrency} warmup request {request.request_index}"),
            )
            if not result["completed"]:
                raise ValidationError(
                    f"concurrency {concurrency} warmup request "
                    f"{request.request_index} did not complete"
                )

        measured_requests = _make_requests(
            normalized_prompts,
            output_tokens=normalized_output_tokens,
            concurrency=concurrency,
            count=normalized_requests,
            warmup=False,
        )
        measured_results = _run_requests(
            measured_requests,
            concurrency=concurrency,
            request_runner=request_runner,
        )
        samples: list[Mapping[str, Any]] = []
        for request, raw_result in zip(measured_requests, measured_results, strict=True):
            result = dict(
                _validate_runner_result(
                    raw_result,
                    output_tokens=normalized_output_tokens,
                    context=(f"concurrency {concurrency} measured request {request.request_index}"),
                )
            )
            result.update(
                {
                    "index": request.request_index,
                    "prompt_index": request.prompt_index,
                    "prompt_sha256": prompt_rows[request.prompt_index - 1]["sha256"],
                }
            )
            samples.append(result)

        rows.append(
            _aggregate_row(
                candidate_id=normalized_candidate,
                concurrency=concurrency,
                samples=samples,
                peak_rss_mib=normalized_rss.get(concurrency),
                slo=normalized_slo,
            )
        )

    passing_levels = [int(row["concurrency"]) for row in rows if row["slo_met"]]
    artifact: Mapping[str, Any] = {
        "schema_version": "1.0",
        "candidate_id": normalized_candidate,
        "synthetic": synthetic,
        "methodology": {
            "concurrency_levels": list(normalized_levels),
            "prompts": prompt_rows,
            "output_tokens": normalized_output_tokens,
            "warmup_requests_per_level": normalized_warmups,
            "measured_requests_per_level": normalized_requests,
            "prompt_schedule": "round-robin",
            "client_scheduler": "bounded thread pool",
        },
        "slo": normalized_slo,
        "rows": rows,
        "highest_slo_concurrency": max(passing_levels) if passing_levels else None,
        "evidence_binding": normalized_binding,
    }
    validate_load_evaluation(artifact)
    return artifact


def combine_load_evaluations(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    require_evidence_bindings: bool = False,
) -> Mapping[str, Any]:
    """Combine compatible single-candidate sweeps into one report artifact."""

    if not isinstance(require_evidence_bindings, bool):
        raise ValidationError("require_evidence_bindings must be a boolean")
    if not isinstance(evaluations, Sequence) or isinstance(evaluations, (str, bytes)):
        raise ValidationError("load evaluations must be an array")
    if not 2 <= len(evaluations) <= 32:
        raise ValidationError("load evaluation combination requires from 2 to 32 inputs")

    validated: list[Mapping[str, Any]] = []
    candidate_ids: set[str] = set()
    for index, evaluation in enumerate(evaluations):
        if not isinstance(evaluation, Mapping):
            raise ValidationError(f"load evaluations[{index}] must be an object")
        validate_load_evaluation(
            evaluation,
            require_evidence_binding=require_evidence_bindings,
        )
        candidate_id = str(evaluation["candidate_id"])
        if candidate_id in candidate_ids:
            raise ValidationError(f"duplicate load evaluation candidate_id {candidate_id!r}")
        candidate_ids.add(candidate_id)
        validated.append(evaluation)

    first = validated[0]
    methodology = first["methodology"]
    slo = first["slo"]
    synthetic = first["synthetic"]
    first_binding = first["evidence_binding"]
    bound = first_binding is not None
    plan_sha256 = str(first_binding["plan_sha256"]) if isinstance(first_binding, Mapping) else None
    for index, evaluation in enumerate(validated[1:], start=2):
        if evaluation["methodology"] != methodology:
            raise ValidationError(
                f"load evaluation input {index} methodology does not match the first input"
            )
        if evaluation["slo"] != slo:
            raise ValidationError(
                f"load evaluation input {index} SLO does not match the first input"
            )
        if evaluation["synthetic"] is not synthetic:
            raise ValidationError(
                f"load evaluation input {index} synthetic flag does not match the first input"
            )
        binding = evaluation["evidence_binding"]
        if (binding is not None) is not bound:
            raise ValidationError(
                "load evaluations must either all include evidence bindings or all omit them"
            )
        if isinstance(binding, Mapping) and binding["plan_sha256"] != plan_sha256:
            raise ValidationError(
                f"load evaluation input {index} plan SHA-256 does not match the first input"
            )

    ordered = sorted(validated, key=lambda item: str(item["candidate_id"]))
    combined: Mapping[str, Any] = {
        "schema_version": "1.0",
        "synthetic": synthetic,
        "methodology": deepcopy(methodology),
        "slo": deepcopy(slo),
        "rows": [deepcopy(row) for evaluation in ordered for row in evaluation["rows"]],
        "highest_slo_concurrency": {
            str(evaluation["candidate_id"]): evaluation["highest_slo_concurrency"]
            for evaluation in ordered
        },
        "evidence_bindings": (
            {
                "plan_sha256": plan_sha256,
                "candidate_server_configurations": {
                    str(evaluation["candidate_id"]): deepcopy(
                        evaluation["evidence_binding"]["server_configuration"]
                    )
                    for evaluation in ordered
                },
                "candidate_request_base_urls": {
                    str(evaluation["candidate_id"]): str(
                        evaluation["evidence_binding"]["request_base_url"]
                    )
                    for evaluation in ordered
                },
            }
            if bound
            else None
        ),
    }
    validate_combined_load_evaluation(
        combined,
        require_evidence_bindings=require_evidence_bindings,
    )
    return combined


def validate_load_evaluation(
    raw: Mapping[str, Any],
    *,
    require_evidence_binding: bool = False,
) -> None:
    """Validate and recompute every aggregate in one candidate load sweep."""

    if not isinstance(require_evidence_binding, bool):
        raise ValidationError("require_evidence_binding must be a boolean")
    if not isinstance(raw, Mapping):
        raise ValidationError("load evaluation must be an object")
    _require_exact_fields(
        raw,
        {
            "schema_version",
            "candidate_id",
            "synthetic",
            "methodology",
            "slo",
            "rows",
            "highest_slo_concurrency",
            "evidence_binding",
        },
        "load evaluation",
    )
    if raw.get("schema_version") != "1.0":
        raise ValidationError("load evaluation schema_version must be '1.0'")
    candidate_id = _nonempty_text(
        raw.get("candidate_id"),
        "load evaluation candidate_id",
        maximum=128,
    )
    if not isinstance(raw.get("synthetic"), bool):
        raise ValidationError("load evaluation synthetic must be a boolean")
    binding = raw.get("evidence_binding")
    if binding is None:
        if require_evidence_binding:
            raise ValidationError("load evaluation evidence_binding is required")
    else:
        _validate_evidence_binding(binding, "load evaluation evidence_binding")
    levels, prompts, output_tokens, request_count = _validate_methodology(
        raw.get("methodology"),
        "load evaluation methodology",
    )
    slo = _validate_slo(raw.get("slo"), "load evaluation slo")
    rows = raw.get("rows")
    if not isinstance(rows, list) or len(rows) != len(levels):
        raise ValidationError(
            "load evaluation rows must contain exactly one row per concurrency level"
        )
    prompt_hashes = {int(prompt["index"]): str(prompt["sha256"]) for prompt in prompts}

    recomputed_rows: list[Mapping[str, Any]] = []
    for index, (expected_concurrency, row_value) in enumerate(zip(levels, rows, strict=True)):
        context = f"load evaluation rows[{index}]"
        row = _mapping(row_value, context)
        if row.get("candidate_id") != candidate_id:
            raise ValidationError(f"{context}.candidate_id does not match the evaluation")
        if row.get("concurrency") != expected_concurrency:
            raise ValidationError(f"{context}.concurrency does not match methodology order")
        samples = _validate_samples(
            row.get("samples"),
            context=f"{context}.samples",
            request_count=request_count,
            output_tokens=output_tokens,
            prompt_hashes=prompt_hashes,
        )
        peak_rss = row.get("peak_rss_mib")
        if peak_rss is not None:
            peak_rss = _finite_number(
                peak_rss,
                f"{context}.peak_rss_mib",
                strictly_positive=True,
            )
        recomputed = _aggregate_row(
            candidate_id=candidate_id,
            concurrency=expected_concurrency,
            samples=samples,
            peak_rss_mib=peak_rss,
            slo=slo,
        )
        if row != recomputed:
            raise ValidationError(f"{context} aggregate fields do not match raw samples")
        recomputed_rows.append(recomputed)

    passing_levels = [int(row["concurrency"]) for row in recomputed_rows if row["slo_met"]]
    expected_highest = max(passing_levels) if passing_levels else None
    if raw.get("highest_slo_concurrency") != expected_highest:
        raise ValidationError(
            "load evaluation highest_slo_concurrency does not match row SLO results"
        )


def validate_combined_load_evaluation(
    raw: Mapping[str, Any],
    *,
    require_evidence_bindings: bool = False,
) -> None:
    """Strictly validate the multi-candidate document used by report rendering."""

    if not isinstance(require_evidence_bindings, bool):
        raise ValidationError("require_evidence_bindings must be a boolean")
    if not isinstance(raw, Mapping):
        raise ValidationError("combined load evaluation must be an object")
    _require_exact_fields(
        raw,
        {
            "schema_version",
            "synthetic",
            "methodology",
            "slo",
            "rows",
            "highest_slo_concurrency",
            "evidence_bindings",
        },
        "combined load evaluation",
    )
    if raw.get("schema_version") != "1.0":
        raise ValidationError("combined load evaluation schema_version must be '1.0'")
    synthetic = raw.get("synthetic")
    if not isinstance(synthetic, bool):
        raise ValidationError("combined load evaluation synthetic must be a boolean")
    levels, prompts, output_tokens, request_count = _validate_methodology(
        raw.get("methodology"),
        "combined load evaluation methodology",
    )
    slo = _validate_slo(raw.get("slo"), "combined load evaluation slo")
    highest = _mapping(
        raw.get("highest_slo_concurrency"),
        "combined load evaluation highest_slo_concurrency",
    )
    if not highest:
        raise ValidationError("combined load evaluation highest_slo_concurrency must not be empty")
    candidate_ids = [
        _nonempty_text(candidate_id, "combined candidate id", maximum=128)
        for candidate_id in highest
    ]
    if candidate_ids != sorted(candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValidationError("combined load evaluation candidate ids must be unique and sorted")
    evidence_bindings = raw.get("evidence_bindings")
    if evidence_bindings is None:
        if require_evidence_bindings:
            raise ValidationError("combined load evaluation evidence_bindings is required")
    else:
        _validate_combined_evidence_bindings(
            evidence_bindings,
            candidate_ids=candidate_ids,
            context="combined load evaluation evidence_bindings",
        )
    prompt_hashes = {int(prompt["index"]): str(prompt["sha256"]) for prompt in prompts}
    rows = raw.get("rows")
    expected_row_count = len(candidate_ids) * len(levels)
    if not isinstance(rows, list) or len(rows) != expected_row_count:
        raise ValidationError(
            "combined load evaluation rows must contain every candidate and concurrency"
        )

    expected_pairs = [
        (candidate_id, concurrency) for candidate_id in candidate_ids for concurrency in levels
    ]
    passing_by_candidate: dict[str, list[int]] = {
        candidate_id: [] for candidate_id in candidate_ids
    }
    for index, ((candidate_id, concurrency), row_value) in enumerate(
        zip(expected_pairs, rows, strict=True)
    ):
        context = f"combined load evaluation rows[{index}]"
        row = _mapping(row_value, context)
        if row.get("candidate_id") != candidate_id:
            raise ValidationError(f"{context}.candidate_id is not in canonical order")
        if row.get("concurrency") != concurrency:
            raise ValidationError(f"{context}.concurrency is not in canonical order")
        samples = _validate_samples(
            row.get("samples"),
            context=f"{context}.samples",
            request_count=request_count,
            output_tokens=output_tokens,
            prompt_hashes=prompt_hashes,
        )
        peak_rss = row.get("peak_rss_mib")
        if peak_rss is not None:
            peak_rss = _finite_number(
                peak_rss,
                f"{context}.peak_rss_mib",
                strictly_positive=True,
            )
        expected = _aggregate_row(
            candidate_id=candidate_id,
            concurrency=concurrency,
            samples=samples,
            peak_rss_mib=peak_rss,
            slo=slo,
        )
        if row != expected:
            raise ValidationError(f"{context} aggregate fields do not match raw samples")
        if row["slo_met"]:
            passing_by_candidate[candidate_id].append(concurrency)

    for candidate_id, passing in passing_by_candidate.items():
        expected = max(passing) if passing else None
        if highest.get(candidate_id) != expected:
            raise ValidationError(
                "combined load evaluation highest_slo_concurrency does not match "
                f"rows for candidate {candidate_id!r}"
            )


def _normalize_evidence_binding(
    value: Mapping[str, Any] | None,
    context: str,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    _validate_evidence_binding(value, context)
    return deepcopy(dict(value))


def _validate_evidence_binding(value: Any, context: str) -> None:
    binding = _mapping(value, context)
    _require_exact_fields(
        binding,
        {"plan_sha256", "request_base_url", "server_configuration"},
        context,
    )
    _sha256_digest(binding.get("plan_sha256"), f"{context}.plan_sha256")
    configuration = _mapping(
        binding.get("server_configuration"),
        f"{context}.server_configuration",
    )
    _validate_server_configuration(
        configuration,
        f"{context}.server_configuration",
    )
    _validate_request_endpoint(
        binding.get("request_base_url"),
        configuration,
        context,
    )


def _validate_combined_evidence_bindings(
    value: Any,
    *,
    candidate_ids: Sequence[str],
    context: str,
) -> None:
    bindings = _mapping(value, context)
    _require_exact_fields(
        bindings,
        {
            "plan_sha256",
            "candidate_server_configurations",
            "candidate_request_base_urls",
        },
        context,
    )
    _sha256_digest(bindings.get("plan_sha256"), f"{context}.plan_sha256")
    configurations = _mapping(
        bindings.get("candidate_server_configurations"),
        f"{context}.candidate_server_configurations",
    )
    configuration_ids = list(configurations)
    if configuration_ids != list(candidate_ids):
        raise ValidationError(
            f"{context}.candidate_server_configurations must contain every "
            "candidate in canonical order"
        )
    request_base_urls = _mapping(
        bindings.get("candidate_request_base_urls"),
        f"{context}.candidate_request_base_urls",
    )
    if list(request_base_urls) != list(candidate_ids):
        raise ValidationError(
            f"{context}.candidate_request_base_urls must contain every candidate in canonical order"
        )
    for candidate_id in candidate_ids:
        configuration = _mapping(
            configurations[candidate_id],
            f"{context}.candidate_server_configurations[{candidate_id!r}]",
        )
        _validate_server_configuration(
            configuration,
            f"{context}.candidate_server_configurations[{candidate_id!r}]",
        )
        _validate_request_endpoint(
            request_base_urls[candidate_id],
            configuration,
            f"{context}.candidate_request_base_urls[{candidate_id!r}]",
        )


def _derive_server_configuration(
    *,
    load_argv: Sequence[str],
    canonical_argv: Sequence[str],
    load_command_sha256: str,
    canonical_command_sha256: str,
) -> Mapping[str, Any]:
    normalized_load = _validate_server_argv(load_argv, "load server argv")
    normalized_canonical = _validate_server_argv(
        canonical_argv,
        "canonical server argv",
    )
    load_digest = _sha256_digest(
        load_command_sha256,
        "load server command SHA-256",
    )
    canonical_digest = _sha256_digest(
        canonical_command_sha256,
        "canonical server command SHA-256",
    )
    if len(normalized_load) != len(normalized_canonical):
        raise ValidationError(
            "load server command has a material argv length difference from "
            "the canonical deployment command"
        )

    load_model = _server_option(normalized_load, "--model", "load server argv")
    canonical_model = _server_option(
        normalized_canonical,
        "--model",
        "canonical server argv",
    )
    load_parallel_text = _server_option(
        normalized_load,
        "--parallel",
        "load server argv",
    )
    canonical_parallel_text = _server_option(
        normalized_canonical,
        "--parallel",
        "canonical server argv",
    )
    load_host = _server_option(normalized_load, "--host", "load server argv")
    canonical_host = _server_option(
        normalized_canonical,
        "--host",
        "canonical server argv",
    )
    load_port_text = _server_option(normalized_load, "--port", "load server argv")
    canonical_port_text = _server_option(
        normalized_canonical,
        "--port",
        "canonical server argv",
    )
    load_parallel = _decimal_integer(
        load_parallel_text,
        "load server --parallel",
        minimum=1,
        maximum=1024,
    )
    canonical_parallel = _decimal_integer(
        canonical_parallel_text,
        "canonical server --parallel",
        minimum=1,
        maximum=1024,
    )
    load_port = _decimal_integer(
        load_port_text,
        "load server --port",
        minimum=1,
        maximum=65535,
    )
    canonical_port = _decimal_integer(
        canonical_port_text,
        "canonical server --port",
        minimum=1,
        maximum=65535,
    )

    if load_parallel != canonical_parallel:
        raise ValidationError("load server --parallel must match the canonical deployment command")

    differing_options: set[str] = set()
    for index, (load_argument, canonical_argument) in enumerate(
        zip(normalized_load, normalized_canonical, strict=True)
    ):
        if load_argument == canonical_argument:
            continue
        preceding_option = (
            normalized_load[index - 1]
            if index > 0 and normalized_load[index - 1] == normalized_canonical[index - 1]
            else None
        )
        if preceding_option not in _ALLOWED_BINDING_OPTIONS:
            raise ValidationError(
                "load server command materially differs from the canonical "
                f"deployment command at argv[{index}]"
            )
        differing_options.add(preceding_option)

    ordered_differences = [
        option for option in _ALLOWED_BINDING_OPTIONS if option in differing_options
    ]
    return {
        "load_server_command_sha256": load_digest,
        "canonical_server_command_sha256": canonical_digest,
        "load_server_argv": list(normalized_load),
        "canonical_server_argv": list(normalized_canonical),
        "load_runtime_path": normalized_load[0],
        "canonical_runtime_path": normalized_canonical[0],
        "load_model_path": load_model,
        "canonical_model_path": canonical_model,
        "load_parallel": load_parallel,
        "canonical_parallel": canonical_parallel,
        "load_host": load_host,
        "canonical_host": canonical_host,
        "load_port": load_port,
        "canonical_port": canonical_port,
        "allowed_binding_options": list(_ALLOWED_BINDING_OPTIONS),
        "differing_binding_options": ordered_differences,
        "argv_equivalent_except_binding": True,
    }


def _validate_server_configuration(value: Any, context: str) -> None:
    configuration = _mapping(value, context)
    expected_fields = {
        "load_server_command_sha256",
        "canonical_server_command_sha256",
        "load_server_argv",
        "canonical_server_argv",
        "load_runtime_path",
        "canonical_runtime_path",
        "load_model_path",
        "canonical_model_path",
        "load_parallel",
        "canonical_parallel",
        "load_host",
        "canonical_host",
        "load_port",
        "canonical_port",
        "allowed_binding_options",
        "differing_binding_options",
        "argv_equivalent_except_binding",
    }
    _require_exact_fields(configuration, expected_fields, context)
    expected = _derive_server_configuration(
        load_argv=_validate_server_argv(
            configuration.get("load_server_argv"),
            f"{context}.load_server_argv",
        ),
        canonical_argv=_validate_server_argv(
            configuration.get("canonical_server_argv"),
            f"{context}.canonical_server_argv",
        ),
        load_command_sha256=_sha256_digest(
            configuration.get("load_server_command_sha256"),
            f"{context}.load_server_command_sha256",
        ),
        canonical_command_sha256=_sha256_digest(
            configuration.get("canonical_server_command_sha256"),
            f"{context}.canonical_server_command_sha256",
        ),
    )
    if configuration != expected:
        raise ValidationError(f"{context} derived fields do not match the recorded server commands")


def _validate_server_argv(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValidationError(f"{context} must be an array")
    if not 1 <= len(value) <= _MAX_SERVER_ARGV_ITEMS:
        raise ValidationError(
            f"{context} must contain from 1 to {_MAX_SERVER_ARGV_ITEMS} arguments"
        )
    argv: list[str] = []
    total_length = 0
    for index, argument in enumerate(value):
        argument_context = f"{context}[{index}]"
        text = _nonempty_text(
            argument,
            argument_context,
            maximum=_MAX_SERVER_ARG_LENGTH,
        )
        if text != text.strip():
            raise ValidationError(f"{argument_context} must not have surrounding whitespace")
        if any(ord(character) < 32 or ord(character) == 127 for character in text):
            raise ValidationError(f"{argument_context} contains a control character")
        total_length += len(text)
        argv.append(text)
    if total_length > 32_768:
        raise ValidationError(f"{context} exceeds the 32768-character total limit")
    return tuple(argv)


def _server_option(argv: Sequence[str], option: str, context: str) -> str:
    indices = [index for index, argument in enumerate(argv) if argument == option]
    if len(indices) != 1:
        raise ValidationError(f"{context} must contain exactly one {option} option")
    index = indices[0]
    if index + 1 >= len(argv):
        raise ValidationError(f"{context} {option} is missing its value")
    value = argv[index + 1]
    if value.startswith("--"):
        raise ValidationError(f"{context} {option} is missing its value")
    return value


def _decimal_integer(
    value: str,
    context: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if not value.isascii() or not value.isdecimal():
        raise ValidationError(f"{context} must be a decimal integer")
    converted = int(value)
    if converted < minimum or converted > maximum:
        raise ValidationError(f"{context} must be from {minimum} to {maximum}")
    return converted


def _normalize_request_base_url(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError("load request base_url must be a non-empty string")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValidationError(f"load request base_url is invalid: {exc}") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname is None
        or port is None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValidationError(
            "load request base_url must be an http(s) origin with an explicit port"
        )
    host = parsed.hostname.casefold()
    formatted_host = f"[{host}]" if ":" in host else host
    return f"{parsed.scheme}://{formatted_host}:{port}"


def _validate_request_endpoint(
    request_base_url: Any,
    configuration: Mapping[str, Any],
    context: str,
) -> None:
    normalized = _normalize_request_base_url(request_base_url)
    parsed = urlsplit(normalized)
    request_host = str(parsed.hostname).casefold()
    request_port = parsed.port
    load_host = str(configuration.get("load_host")).casefold()
    load_port = configuration.get("load_port")
    if request_port != load_port:
        raise ValidationError(f"{context} request base_url port must match the load server --port")
    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    wildcard_hosts = {"0.0.0.0", "::"}
    host_matches = (
        request_host == load_host
        or request_host in loopback_hosts
        and load_host in loopback_hosts | wildcard_hosts
    )
    if not host_matches:
        raise ValidationError(
            f"{context} request base_url host does not safely match the load server --host"
        )


def _sha256_digest(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValidationError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _make_requests(
    prompts: tuple[str, ...],
    *,
    output_tokens: int,
    concurrency: int,
    count: int,
    warmup: bool,
) -> list[LoadRequest]:
    return [
        LoadRequest(
            prompt=prompts[index % len(prompts)],
            prompt_index=(index % len(prompts)) + 1,
            output_tokens=output_tokens,
            concurrency=concurrency,
            request_index=index + 1,
            warmup=warmup,
        )
        for index in range(count)
    ]


def _run_requests(
    requests: Sequence[LoadRequest],
    *,
    concurrency: int,
    request_runner: RequestRunner,
) -> list[Mapping[str, Any]]:
    if not requests:
        return []

    def invoke(request: LoadRequest) -> Mapping[str, Any]:
        try:
            result = request_runner(request)
        except Exception as exc:
            raise ValidationError(
                f"request runner raised {type(exc).__name__} for "
                f"concurrency {request.concurrency} request {request.request_index}: {exc}"
            ) from exc
        if not isinstance(result, Mapping):
            raise ValidationError("request runner must return an object")
        return result

    with ThreadPoolExecutor(
        max_workers=concurrency,
        thread_name_prefix=f"paretopilot-load-{concurrency}",
    ) as executor:
        return list(executor.map(invoke, requests))


def _failed_network_result(
    *,
    started: float,
    finished: float,
    first_content_at: float | None,
    generated_tokens: int,
    error: str,
) -> Mapping[str, Any]:
    return {
        "completed": False,
        "ttft_ms": (None if first_content_at is None else (first_content_at - started) * 1000.0),
        "e2e_latency_ms": (finished - started) * 1000.0,
        "generated_tokens": generated_tokens,
        "error": error[:512],
        "started_at_seconds": started,
        "finished_at_seconds": finished,
    }


def _validate_runner_result(
    raw: Mapping[str, Any],
    *,
    output_tokens: int,
    context: str,
) -> Mapping[str, Any]:
    _require_exact_fields(
        raw,
        {
            "completed",
            "ttft_ms",
            "e2e_latency_ms",
            "generated_tokens",
            "error",
            "started_at_seconds",
            "finished_at_seconds",
        },
        context,
    )
    completed = raw.get("completed")
    if not isinstance(completed, bool):
        raise ValidationError(f"{context}.completed must be a boolean")
    started = _finite_number(
        raw.get("started_at_seconds"),
        f"{context}.started_at_seconds",
        minimum=0.0,
    )
    finished = _finite_number(
        raw.get("finished_at_seconds"),
        f"{context}.finished_at_seconds",
        strictly_positive=True,
    )
    if finished <= started:
        raise ValidationError(f"{context}.finished_at_seconds must be after its start")
    e2e = _finite_number(
        raw.get("e2e_latency_ms"),
        f"{context}.e2e_latency_ms",
        strictly_positive=True,
    )
    elapsed_ms = (finished - started) * 1000.0
    if not math.isclose(e2e, elapsed_ms, rel_tol=1e-6, abs_tol=0.05):
        raise ValidationError(f"{context}.e2e_latency_ms must match its start/finish timestamps")
    generated_tokens = _integer_between(
        raw.get("generated_tokens"),
        f"{context}.generated_tokens",
        minimum=0,
        maximum=output_tokens,
    )
    ttft_value = raw.get("ttft_ms")
    error = raw.get("error")
    if completed:
        ttft = _finite_number(
            ttft_value,
            f"{context}.ttft_ms",
            minimum=0.0,
        )
        if ttft > e2e:
            raise ValidationError(f"{context}.ttft_ms cannot exceed e2e_latency_ms")
        if generated_tokens != output_tokens:
            raise ValidationError(
                f"{context}.generated_tokens must equal the fixed output token count"
            )
        if error is not None:
            raise ValidationError(f"{context}.error must be null for a completed request")
    else:
        if ttft_value is None:
            ttft = None
        else:
            ttft = _finite_number(
                ttft_value,
                f"{context}.ttft_ms",
                minimum=0.0,
            )
            if ttft > e2e:
                raise ValidationError(f"{context}.ttft_ms cannot exceed e2e_latency_ms")
        error = _nonempty_text(error, f"{context}.error", maximum=512)
    return {
        "completed": completed,
        "ttft_ms": ttft,
        "e2e_latency_ms": e2e,
        "generated_tokens": generated_tokens,
        "error": error,
        "started_at_seconds": started,
        "finished_at_seconds": finished,
    }


def _aggregate_row(
    *,
    candidate_id: str,
    concurrency: int,
    samples: Sequence[Mapping[str, Any]],
    peak_rss_mib: float | None,
    slo: Mapping[str, Any],
) -> Mapping[str, Any]:
    request_count = len(samples)
    completed_samples = [sample for sample in samples if sample["completed"]]
    completed_count = len(completed_samples)
    failed_count = request_count - completed_count
    started = [float(sample["started_at_seconds"]) for sample in samples]
    finished = [float(sample["finished_at_seconds"]) for sample in samples]
    wall_time = max(finished) - min(started)
    if not math.isfinite(wall_time) or wall_time <= 0:
        raise ValidationError("load samples must cover a positive finite wall time")
    ttft_values = [float(sample["ttft_ms"]) for sample in completed_samples]
    e2e_values = [float(sample["e2e_latency_ms"]) for sample in completed_samples]
    completion_rate = completed_count / request_count
    error_rate = failed_count / request_count
    generated_tokens = sum(int(sample["generated_tokens"]) for sample in samples)
    ttft_p50 = _percentile_or_none(ttft_values, 50)
    ttft_p95 = _percentile_or_none(ttft_values, 95)
    e2e_p50 = _percentile_or_none(e2e_values, 50)
    e2e_p95 = _percentile_or_none(e2e_values, 95)
    failures = _slo_failures(
        completion_rate=completion_rate,
        ttft_ms_p95=ttft_p95,
        e2e_latency_ms_p95=e2e_p95,
        slo=slo,
    )
    return {
        "candidate_id": candidate_id,
        "concurrency": concurrency,
        "request_count": request_count,
        "completed_requests": completed_count,
        "failed_requests": failed_count,
        "completion_rate": completion_rate,
        "error_rate": error_rate,
        "wall_time_seconds": wall_time,
        "requests_per_second": completed_count / wall_time,
        "generated_tokens_per_second": generated_tokens / wall_time,
        "ttft_ms_p50": ttft_p50,
        "ttft_ms_p95": ttft_p95,
        "e2e_latency_ms_p50": e2e_p50,
        "e2e_latency_ms_p95": e2e_p95,
        "peak_rss_mib": peak_rss_mib,
        "slo_met": not failures,
        "slo_failures": failures,
        "samples": list(samples),
    }


def _slo_failures(
    *,
    completion_rate: float,
    ttft_ms_p95: float | None,
    e2e_latency_ms_p95: float | None,
    slo: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    if completion_rate < float(slo["min_completion_rate"]):
        failures.append("completion_rate_below_minimum")
    if ttft_ms_p95 is None:
        failures.append("no_completed_request_ttft")
    else:
        maximum_ttft = slo["max_ttft_ms_p95"]
        if maximum_ttft is not None and ttft_ms_p95 > float(maximum_ttft):
            failures.append("ttft_ms_p95_above_maximum")
    if e2e_latency_ms_p95 is None:
        failures.append("no_completed_request_e2e")
    elif e2e_latency_ms_p95 > float(slo["max_e2e_latency_ms_p95"]):
        failures.append("e2e_latency_ms_p95_above_maximum")
    return failures


def _validate_samples(
    value: Any,
    *,
    context: str,
    request_count: int,
    output_tokens: int,
    prompt_hashes: Mapping[int, str],
) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != request_count:
        raise ValidationError(f"{context} must contain exactly {request_count} samples")
    validated: list[Mapping[str, Any]] = []
    for index, sample_value in enumerate(value, start=1):
        sample_context = f"{context}[{index - 1}]"
        sample = _mapping(sample_value, sample_context)
        _require_exact_fields(
            sample,
            {
                "index",
                "prompt_index",
                "prompt_sha256",
                "completed",
                "ttft_ms",
                "e2e_latency_ms",
                "generated_tokens",
                "error",
                "started_at_seconds",
                "finished_at_seconds",
            },
            sample_context,
        )
        if sample.get("index") != index:
            raise ValidationError(f"{sample_context}.index must be sequential")
        prompt_index = _integer_between(
            sample.get("prompt_index"),
            f"{sample_context}.prompt_index",
            minimum=1,
            maximum=len(prompt_hashes),
        )
        expected_prompt_index = ((index - 1) % len(prompt_hashes)) + 1
        if prompt_index != expected_prompt_index:
            raise ValidationError(
                f"{sample_context}.prompt_index does not follow the round-robin schedule"
            )
        if sample.get("prompt_sha256") != prompt_hashes[prompt_index]:
            raise ValidationError(
                f"{sample_context}.prompt_sha256 does not match its configured prompt"
            )
        result_fields = {
            name: sample[name]
            for name in (
                "completed",
                "ttft_ms",
                "e2e_latency_ms",
                "generated_tokens",
                "error",
                "started_at_seconds",
                "finished_at_seconds",
            )
        }
        normalized = dict(
            _validate_runner_result(
                result_fields,
                output_tokens=output_tokens,
                context=sample_context,
            )
        )
        normalized.update(
            {
                "index": index,
                "prompt_index": prompt_index,
                "prompt_sha256": prompt_hashes[prompt_index],
            }
        )
        validated.append(normalized)
    return validated


def _validate_prompts(prompts: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(prompts, Sequence) or isinstance(prompts, (str, bytes)):
        raise ValidationError("prompts must be an array")
    if not 1 <= len(prompts) <= _MAX_PROMPTS:
        raise ValidationError(f"prompts must contain from 1 to {_MAX_PROMPTS} values")
    normalized = tuple(
        _nonempty_text(prompt, f"prompts[{index}]", maximum=8192)
        for index, prompt in enumerate(prompts)
    )
    if len(set(normalized)) != len(normalized):
        raise ValidationError("prompts must be unique")
    return normalized


def _validate_concurrency_levels(levels: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)):
        raise ValidationError("concurrency_levels must be an array")
    normalized = tuple(levels)
    if not normalized:
        raise ValidationError("concurrency_levels must not be empty")
    if any(isinstance(level, bool) or not isinstance(level, int) for level in normalized):
        raise ValidationError("concurrency_levels must contain integers")
    if any(level not in _ALLOWED_CONCURRENCY for level in normalized):
        raise ValidationError("concurrency_levels may contain only 1, 2, and 4")
    if normalized != tuple(sorted(set(normalized))):
        raise ValidationError("concurrency_levels must be unique and increasing")
    return normalized


def _validate_peak_rss(
    values: Mapping[int, float] | None,
    levels: tuple[int, ...],
) -> Mapping[int, float]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValidationError("peak_rss_mib_by_concurrency must be an object")
    if set(values) != set(levels):
        raise ValidationError(
            "peak_rss_mib_by_concurrency must contain exactly every concurrency level"
        )
    return {
        concurrency: _finite_number(
            values[concurrency],
            f"peak_rss_mib_by_concurrency[{concurrency}]",
            strictly_positive=True,
        )
        for concurrency in levels
    }


def _validate_methodology(
    value: Any,
    context: str,
) -> tuple[tuple[int, ...], list[Mapping[str, Any]], int, int]:
    methodology = _mapping(value, context)
    _require_exact_fields(
        methodology,
        {
            "concurrency_levels",
            "prompts",
            "output_tokens",
            "warmup_requests_per_level",
            "measured_requests_per_level",
            "prompt_schedule",
            "client_scheduler",
        },
        context,
    )
    levels = _validate_concurrency_levels(methodology.get("concurrency_levels"))
    prompt_values = methodology.get("prompts")
    if not isinstance(prompt_values, list):
        raise ValidationError(f"{context}.prompts must be an array")
    if not 1 <= len(prompt_values) <= _MAX_PROMPTS:
        raise ValidationError(f"{context}.prompts must contain from 1 to {_MAX_PROMPTS} values")
    prompts: list[Mapping[str, Any]] = []
    seen_text: set[str] = set()
    for index, prompt_value in enumerate(prompt_values, start=1):
        prompt_context = f"{context}.prompts[{index - 1}]"
        prompt = _mapping(prompt_value, prompt_context)
        _require_exact_fields(prompt, {"index", "text", "sha256"}, prompt_context)
        if prompt.get("index") != index:
            raise ValidationError(f"{prompt_context}.index must be sequential")
        text = _nonempty_text(prompt.get("text"), f"{prompt_context}.text", maximum=8192)
        if text in seen_text:
            raise ValidationError(f"{context}.prompts must be unique")
        seen_text.add(text)
        expected_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if prompt.get("sha256") != expected_sha:
            raise ValidationError(f"{prompt_context}.sha256 does not match prompt text")
        prompts.append({"index": index, "text": text, "sha256": expected_sha})
    output_tokens = _integer_between(
        methodology.get("output_tokens"),
        f"{context}.output_tokens",
        minimum=8,
        maximum=512,
    )
    _integer_between(
        methodology.get("warmup_requests_per_level"),
        f"{context}.warmup_requests_per_level",
        minimum=0,
        maximum=32,
    )
    request_count = _integer_between(
        methodology.get("measured_requests_per_level"),
        f"{context}.measured_requests_per_level",
        minimum=max(levels),
        maximum=_MAX_REQUESTS_PER_LEVEL,
    )
    for concurrency in levels:
        if request_count % concurrency != 0:
            raise ValidationError(
                f"{context}.measured_requests_per_level must be divisible by "
                "every concurrency level"
            )
    if methodology.get("prompt_schedule") != "round-robin":
        raise ValidationError(f"{context}.prompt_schedule must be 'round-robin'")
    if methodology.get("client_scheduler") != "bounded thread pool":
        raise ValidationError(f"{context}.client_scheduler must be 'bounded thread pool'")
    return levels, prompts, output_tokens, request_count


def _validate_slo(value: Any, context: str) -> Mapping[str, Any]:
    slo = _mapping(value, context)
    _require_exact_fields(
        slo,
        {
            "min_completion_rate",
            "max_ttft_ms_p95",
            "max_e2e_latency_ms_p95",
        },
        context,
    )
    completion_rate = _finite_number(
        slo.get("min_completion_rate"),
        f"{context}.min_completion_rate",
        minimum=0.0,
        maximum=1.0,
    )
    max_ttft_value = slo.get("max_ttft_ms_p95")
    max_ttft = (
        None
        if max_ttft_value is None
        else _finite_number(
            max_ttft_value,
            f"{context}.max_ttft_ms_p95",
            strictly_positive=True,
        )
    )
    max_e2e = _finite_number(
        slo.get("max_e2e_latency_ms_p95"),
        f"{context}.max_e2e_latency_ms_p95",
        strictly_positive=True,
    )
    if max_ttft is not None and max_ttft > max_e2e:
        raise ValidationError(f"{context}.max_ttft_ms_p95 cannot exceed max E2E")
    return {
        "min_completion_rate": completion_rate,
        "max_ttft_ms_p95": max_ttft,
        "max_e2e_latency_ms_p95": max_e2e,
    }


def _percentile_or_none(
    values: Sequence[float],
    percentile: int,
) -> float | None:
    if not values:
        return None
    if percentile == 50:
        return float(statistics.median(values))
    ordered = sorted(values)
    rank = math.ceil((percentile / 100.0) * len(ordered))
    return float(ordered[max(0, rank - 1)])


def _require_exact_fields(
    raw: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
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
    converted = float(value)
    if not math.isfinite(converted):
        raise ValidationError(f"{context} must be a finite number")
    if strictly_positive and converted <= 0:
        raise ValidationError(f"{context} must be positive")
    if minimum is not None and converted < minimum:
        raise ValidationError(f"{context} must be at least {minimum}")
    if maximum is not None and converted > maximum:
        raise ValidationError(f"{context} must be at most {maximum}")
    return converted


def _nonempty_text(value: Any, context: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{context} must be a non-empty string")
    if len(value) > maximum:
        raise ValidationError(f"{context} must contain at most {maximum} characters")
    return value
