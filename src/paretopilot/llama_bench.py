"""Strict, dependency-free parsing for ``llama-bench -o jsonl`` output.

The upstream format does not emit a canonical ``test`` or repetition-count
field.  Workload kind is therefore inferred from ``n_prompt`` and ``n_gen``,
and the two sample arrays are the source of truth for repetition count.
Unknown fields are deliberately ignored so additive upstream changes remain
compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

from paretopilot.domain import ValidationError


TestKind = Literal["pp", "tg", "pg"]

# llama-bench may serialize summary values at lower precision than individual
# samples. A 0.1% relative tolerance admits ordinary display rounding while
# still rejecting summaries that do not describe their attached samples.
_AVERAGE_REL_TOLERANCE = 1e-3
_MAX_FILE_BYTES = 64 * 1024 * 1024
_MAX_LINE_CHARACTERS = 2 * 1024 * 1024
_MAX_RECORDS = 10_000
_MAX_SAMPLES_PER_RECORD = 10_000


class LlamaBenchParseError(ValidationError):
    """Raised when a JSONL row does not satisfy the supported contract."""


@dataclass(frozen=True, slots=True)
class LlamaBenchRecord:
    """The stable subset of one upstream llama-bench JSONL row."""

    source_line: int
    build_commit: str
    model_filename: str
    n_prompt: int
    n_gen: int
    test_kind: TestKind
    avg_ns: float
    avg_ts: float
    samples_ns: tuple[float, ...]
    samples_ts: tuple[float, ...]
    n_threads: int | None = None
    n_batch: int | None = None
    n_ubatch: int | None = None
    n_gpu_layers: int | None = None
    devices: str | None = None
    no_op_offload: int | None = None
    synthetic_fixture: bool = False

    @property
    def repetition_count(self) -> int:
        """Return the measured repetitions represented by the sample arrays."""

        return len(self.samples_ns)


def _context(source: str, line_number: int) -> str:
    return f"{source}:{line_number}"


def _required_string(raw: Mapping[str, Any], name: str, *, context: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise LlamaBenchParseError(f"{context}: {name} must be a non-empty string")
    return value


def _required_nonnegative_int(raw: Mapping[str, Any], name: str, *, context: str) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LlamaBenchParseError(f"{context}: {name} must be a non-negative integer")
    return value


def _optional_int(raw: Mapping[str, Any], name: str, *, context: str) -> int | None:
    if name not in raw:
        return None
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise LlamaBenchParseError(f"{context}: {name} must be an integer")
    return value


def _optional_positive_int(raw: Mapping[str, Any], name: str, *, context: str) -> int | None:
    value = _optional_int(raw, name, context=context)
    if value is not None and value <= 0:
        raise LlamaBenchParseError(f"{context}: {name} must be a positive integer")
    return value


def _optional_binary_int(raw: Mapping[str, Any], name: str, *, context: str) -> int | None:
    value = _optional_int(raw, name, context=context)
    if value is not None and value not in {0, 1}:
        raise LlamaBenchParseError(f"{context}: {name} must be either 0 or 1")
    return value


def _optional_string(raw: Mapping[str, Any], name: str, *, context: str) -> str | None:
    if name not in raw:
        return None
    return _required_string(raw, name, context=context)


def _positive_number(value: Any, *, field_name: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LlamaBenchParseError(f"{context}: {field_name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise LlamaBenchParseError(f"{context}: {field_name} must be finite and positive")
    return converted


def _required_positive_number(raw: Mapping[str, Any], name: str, *, context: str) -> float:
    return _positive_number(raw.get(name), field_name=name, context=context)


def _required_samples(raw: Mapping[str, Any], name: str, *, context: str) -> tuple[float, ...]:
    values = raw.get(name)
    if not isinstance(values, list):
        raise LlamaBenchParseError(f"{context}: {name} must be an array")
    if len(values) > _MAX_SAMPLES_PER_RECORD:
        raise LlamaBenchParseError(
            f"{context}: {name} exceeds the {_MAX_SAMPLES_PER_RECORD}-sample safety limit"
        )
    return tuple(
        _positive_number(value, field_name=f"{name}[{index}]", context=context)
        for index, value in enumerate(values)
    )


def _validate_reported_average(
    reported: float,
    samples: tuple[float, ...],
    *,
    reported_name: str,
    samples_name: str,
    absolute_tolerance: float,
    context: str,
) -> None:
    calculated = math.fsum(samples) / len(samples)
    if not math.isclose(
        reported,
        calculated,
        rel_tol=_AVERAGE_REL_TOLERANCE,
        abs_tol=absolute_tolerance,
    ):
        raise LlamaBenchParseError(
            f"{context}: {reported_name} is inconsistent with the arithmetic mean of "
            f"{samples_name} (reported {reported:.12g}, calculated {calculated:.12g})"
        )


def _infer_test_kind(n_prompt: int, n_gen: int, *, context: str) -> TestKind:
    if n_prompt > 0 and n_gen == 0:
        return "pp"
    if n_prompt == 0 and n_gen > 0:
        return "tg"
    if n_prompt > 0 and n_gen > 0:
        return "pg"
    raise LlamaBenchParseError(
        f"{context}: invalid test shape; n_prompt and n_gen cannot both be zero"
    )


def parse_llama_bench_row(
    raw: Mapping[str, Any], *, source: str = "<memory>", line_number: int = 1
) -> LlamaBenchRecord:
    """Validate and convert one decoded llama-bench JSON object."""

    context = _context(source, line_number)
    if not isinstance(raw, Mapping):
        raise LlamaBenchParseError(f"{context}: row must be a JSON object")

    n_prompt = _required_nonnegative_int(raw, "n_prompt", context=context)
    n_gen = _required_nonnegative_int(raw, "n_gen", context=context)
    samples_ns = _required_samples(raw, "samples_ns", context=context)
    samples_ts = _required_samples(raw, "samples_ts", context=context)
    if not samples_ns:
        raise LlamaBenchParseError(f"{context}: sample arrays must not be empty")
    if len(samples_ns) != len(samples_ts):
        raise LlamaBenchParseError(
            f"{context}: samples_ns and samples_ts must have the same length"
        )

    avg_ns = _required_positive_number(raw, "avg_ns", context=context)
    avg_ts = _required_positive_number(raw, "avg_ts", context=context)
    synthetic_fixture = raw.get("synthetic_fixture", False)
    if not isinstance(synthetic_fixture, bool):
        raise LlamaBenchParseError(f"{context}: synthetic_fixture must be a boolean")
    _validate_reported_average(
        avg_ns,
        samples_ns,
        reported_name="avg_ns",
        samples_name="samples_ns",
        absolute_tolerance=0.5,
        context=context,
    )
    _validate_reported_average(
        avg_ts,
        samples_ts,
        reported_name="avg_ts",
        samples_name="samples_ts",
        absolute_tolerance=1e-12,
        context=context,
    )

    return LlamaBenchRecord(
        source_line=line_number,
        build_commit=_required_string(raw, "build_commit", context=context),
        model_filename=_required_string(raw, "model_filename", context=context),
        n_prompt=n_prompt,
        n_gen=n_gen,
        test_kind=_infer_test_kind(n_prompt, n_gen, context=context),
        avg_ns=avg_ns,
        avg_ts=avg_ts,
        samples_ns=samples_ns,
        samples_ts=samples_ts,
        n_threads=_optional_positive_int(raw, "n_threads", context=context),
        n_batch=_optional_positive_int(raw, "n_batch", context=context),
        n_ubatch=_optional_positive_int(raw, "n_ubatch", context=context),
        n_gpu_layers=_optional_int(raw, "n_gpu_layers", context=context),
        devices=_optional_string(raw, "devices", context=context),
        no_op_offload=_optional_binary_int(raw, "no_op_offload", context=context),
        synthetic_fixture=synthetic_fixture,
    )


def parse_llama_bench_jsonl(text: str, *, source: str = "<memory>") -> tuple[LlamaBenchRecord, ...]:
    """Parse JSONL text, ignoring blank lines and rejecting malformed rows."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    records: list[LlamaBenchRecord] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        context = _context(source, line_number)
        if len(line) > _MAX_LINE_CHARACTERS:
            raise LlamaBenchParseError(
                f"{context}: row exceeds the {_MAX_LINE_CHARACTERS}-character safety limit"
            )
        if len(records) >= _MAX_RECORDS:
            raise LlamaBenchParseError(f"{source}: exceeds the {_MAX_RECORDS}-record safety limit")
        try:
            raw = json.loads(
                line,
                object_pairs_hook=lambda pairs: _unique_json_object(pairs, context=context),
                parse_constant=lambda value: _reject_json_constant(value, context=context),
            )
        except LlamaBenchParseError:
            raise
        except json.JSONDecodeError as exc:
            raise LlamaBenchParseError(
                f"{context}: invalid JSON at column {exc.colno}: {exc.msg}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise LlamaBenchParseError(f"{context}: row must be a JSON object")
        records.append(parse_llama_bench_row(raw, source=source, line_number=line_number))

    if not records:
        raise LlamaBenchParseError(f"{source}: no llama-bench records found")
    return tuple(records)


def load_llama_bench_jsonl(path: Path) -> tuple[LlamaBenchRecord, ...]:
    """Read and parse a llama-bench JSONL artifact from disk."""

    try:
        size = path.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise LlamaBenchParseError(
                f"{path}: file exceeds the {_MAX_FILE_BYTES}-byte safety limit"
            )
        text = path.read_text(encoding="utf-8")
    except LlamaBenchParseError:
        raise
    except UnicodeDecodeError as exc:
        raise LlamaBenchParseError(f"{path}: file must be UTF-8 text") from exc
    except OSError as exc:
        raise LlamaBenchParseError(f"could not read {path}: {exc}") from exc
    return parse_llama_bench_jsonl(text, source=str(path))


def _unique_json_object(pairs: list[tuple[str, Any]], *, context: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LlamaBenchParseError(f"{context}: duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str, *, context: str) -> None:
    raise LlamaBenchParseError(f"{context}: non-standard JSON constant {value!r} is not allowed")
