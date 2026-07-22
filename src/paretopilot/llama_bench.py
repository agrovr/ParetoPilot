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


def _positive_number(value: Any, *, field_name: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LlamaBenchParseError(f"{context}: {field_name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise LlamaBenchParseError(f"{context}: {field_name} must be finite and positive")
    return converted


def _required_positive_number(raw: Mapping[str, Any], name: str, *, context: str) -> float:
    return _positive_number(raw.get(name), field_name=name, context=context)


def _required_samples(
    raw: Mapping[str, Any], name: str, *, context: str
) -> tuple[float, ...]:
    values = raw.get(name)
    if not isinstance(values, list):
        raise LlamaBenchParseError(f"{context}: {name} must be an array")
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
        synthetic_fixture=synthetic_fixture,
    )


def parse_llama_bench_jsonl(
    text: str, *, source: str = "<memory>"
) -> tuple[LlamaBenchRecord, ...]:
    """Parse JSONL text, ignoring blank lines and rejecting malformed rows."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    records: list[LlamaBenchRecord] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        context = _context(source, line_number)
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LlamaBenchParseError(
                f"{context}: invalid JSON at column {exc.colno}: {exc.msg}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise LlamaBenchParseError(f"{context}: row must be a JSON object")
        records.append(
            parse_llama_bench_row(raw, source=source, line_number=line_number)
        )

    if not records:
        raise LlamaBenchParseError(f"{source}: no llama-bench records found")
    return tuple(records)


def load_llama_bench_jsonl(path: Path) -> tuple[LlamaBenchRecord, ...]:
    """Read and parse a llama-bench JSONL artifact from disk."""

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise LlamaBenchParseError(f"file not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise LlamaBenchParseError(f"{path}: file must be UTF-8 text") from exc
    return parse_llama_bench_jsonl(text, source=str(path))
