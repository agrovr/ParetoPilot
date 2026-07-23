"""Compare compatible generic and KleidiAI ``llama-bench`` summaries."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence

from paretopilot.domain import ValidationError
from paretopilot.llama_bench import TestKind


_TEST_ORDER: tuple[TestKind, ...] = ("pp", "tg", "pg")


class LlamaBenchComparisonError(ValidationError):
    """Raised when summaries are malformed or cannot be compared safely."""


def _object(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LlamaBenchComparisonError(f"{context} must be an object")
    return value


def _non_empty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LlamaBenchComparisonError(f"{context} must be a non-empty string")
    return value


def _boolean(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise LlamaBenchComparisonError(f"{context} must be a boolean")
    return value


def _nonnegative_integer(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LlamaBenchComparisonError(f"{context} must be a non-negative integer")
    return value


def _positive_integer(value: Any, *, context: str) -> int:
    result = _nonnegative_integer(value, context=context)
    if result == 0:
        raise LlamaBenchComparisonError(f"{context} must be greater than zero")
    return result


def _positive_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LlamaBenchComparisonError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise LlamaBenchComparisonError(f"{context} must be finite and positive")
    return result


def _json_value(value: Any, *, context: str) -> Any:
    """Copy a settings value into a deterministic JSON representation."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LlamaBenchComparisonError(f"{context} must be finite")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LlamaBenchComparisonError(f"{context} keys must be strings")
            copied[key] = _json_value(item, context=f"{context}.{key}")
        return copied
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_value(item, context=f"{context}[{index}]") for index, item in enumerate(value)
        ]
    raise LlamaBenchComparisonError(f"{context} must be JSON-compatible")


def _settings_copy(value: Any, *, context: str) -> dict[str, Any]:
    copied = _json_value(_object(value, context=context), context=context)
    assert isinstance(copied, dict)
    return copied


def _settings_without_kleidiai(
    settings: Mapping[str, Any],
    *,
    role: str,
    expected_flag: bool,
) -> dict[str, Any]:
    """Return settings with only the required role-selecting flag removed."""

    copied = _settings_copy(settings, context=f"{role}.settings")
    build = copied.get("build")
    if not isinstance(build, dict):
        raise LlamaBenchComparisonError(f"{role}.settings.build must be an object")
    if build.get("kleidiai") is not expected_flag:
        expected_text = "true" if expected_flag else "false"
        raise LlamaBenchComparisonError(f"{role}.settings.build.kleidiai must be {expected_text}")
    del build["kleidiai"]
    return copied


@dataclass(frozen=True, slots=True)
class _MetricSnapshot:
    median: float
    sample_count: int

    @classmethod
    def from_mapping(
        cls,
        raw: Any,
        *,
        context: str,
    ) -> "_MetricSnapshot":
        mapping = _object(raw, context=context)
        return cls(
            median=_positive_number(mapping.get("median"), context=f"{context}.median"),
            sample_count=_positive_integer(
                mapping.get("sample_count"),
                context=f"{context}.sample_count",
            ),
        )


@dataclass(frozen=True, slots=True)
class _TestSnapshot:
    test_kind: TestKind
    n_prompt: int
    n_gen: int
    tokens_per_second: _MetricSnapshot
    duration_ns: _MetricSnapshot

    @classmethod
    def from_mapping(
        cls,
        test_kind: TestKind,
        raw: Any,
        *,
        context: str,
    ) -> "_TestSnapshot":
        mapping = _object(raw, context=context)
        encoded_kind = _non_empty_string(
            mapping.get("test_kind"),
            context=f"{context}.test_kind",
        )
        if encoded_kind != test_kind:
            raise LlamaBenchComparisonError(
                f"{context}.test_kind must match its object key {test_kind!r}"
            )
        n_prompt = _nonnegative_integer(mapping.get("n_prompt"), context=f"{context}.n_prompt")
        n_gen = _nonnegative_integer(mapping.get("n_gen"), context=f"{context}.n_gen")
        expected_kind = (
            "pp"
            if n_prompt > 0 and n_gen == 0
            else "tg"
            if n_prompt == 0 and n_gen > 0
            else "pg"
            if n_prompt > 0 and n_gen > 0
            else None
        )
        if expected_kind != test_kind:
            raise LlamaBenchComparisonError(
                f"{context} shape ({n_prompt}, {n_gen}) is not a {test_kind} workload"
            )
        tokens_per_second = _MetricSnapshot.from_mapping(
            mapping.get("tokens_per_second"),
            context=f"{context}.tokens_per_second",
        )
        duration_ns = _MetricSnapshot.from_mapping(
            mapping.get("duration_ns"),
            context=f"{context}.duration_ns",
        )
        if tokens_per_second.sample_count != duration_ns.sample_count:
            raise LlamaBenchComparisonError(f"{context} metric sample counts must match")
        return cls(
            test_kind=test_kind,
            n_prompt=n_prompt,
            n_gen=n_gen,
            tokens_per_second=tokens_per_second,
            duration_ns=duration_ns,
        )


@dataclass(frozen=True, slots=True)
class _VariantSnapshot:
    label: str
    build_commit: str
    model_filename: str
    settings: Mapping[str, Any]
    synthetic_fixture: bool
    tests: Mapping[TestKind, _TestSnapshot]

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        context: str,
    ) -> "_VariantSnapshot":
        if raw.get("schema_version") != "1.0":
            raise LlamaBenchComparisonError(f"{context}.schema_version must currently be '1.0'")
        settings = _settings_copy(raw.get("settings"), context=f"{context}.settings")
        raw_tests = _object(raw.get("tests"), context=f"{context}.tests")
        if not raw_tests:
            raise LlamaBenchComparisonError(f"{context}.tests must not be empty")
        if any(not isinstance(kind, str) for kind in raw_tests):
            raise LlamaBenchComparisonError(f"{context}.tests workload keys must be strings")
        unsupported = sorted(set(raw_tests) - set(_TEST_ORDER))
        if unsupported:
            raise LlamaBenchComparisonError(
                f"{context}.tests contains unsupported workload kinds: "
                f"{', '.join(str(kind) for kind in unsupported)}"
            )
        tests: dict[TestKind, _TestSnapshot] = {}
        for test_kind in _TEST_ORDER:
            if test_kind in raw_tests:
                tests[test_kind] = _TestSnapshot.from_mapping(
                    test_kind,
                    raw_tests[test_kind],
                    context=f"{context}.tests.{test_kind}",
                )
        return cls(
            label=_non_empty_string(raw.get("label"), context=f"{context}.label"),
            build_commit=_non_empty_string(
                raw.get("build_commit"), context=f"{context}.build_commit"
            ),
            model_filename=_non_empty_string(
                raw.get("model_filename"), context=f"{context}.model_filename"
            ),
            settings=settings,
            synthetic_fixture=_boolean(
                raw.get("synthetic_fixture"),
                context=f"{context}.synthetic_fixture",
            ),
            tests=tests,
        )


@dataclass(frozen=True, slots=True)
class LlamaBenchWorkloadComparison:
    """Median-based comparison for one matching workload shape."""

    test_kind: TestKind
    n_prompt: int
    n_gen: int
    generic: _TestSnapshot
    kleidiai: _TestSnapshot

    def to_mapping(self) -> dict[str, Any]:
        generic_throughput = self.generic.tokens_per_second.median
        kleidiai_throughput = self.kleidiai.tokens_per_second.median
        generic_duration = self.generic.duration_ns.median
        kleidiai_duration = self.kleidiai.duration_ns.median
        throughput_speedup = kleidiai_throughput / generic_throughput
        return {
            "test_kind": self.test_kind,
            "n_prompt": self.n_prompt,
            "n_gen": self.n_gen,
            "generic": {
                "tokens_per_second_median": generic_throughput,
                "tokens_per_second_sample_count": (self.generic.tokens_per_second.sample_count),
                "duration_ns_median": generic_duration,
                "duration_ns_sample_count": self.generic.duration_ns.sample_count,
            },
            "kleidiai": {
                "tokens_per_second_median": kleidiai_throughput,
                "tokens_per_second_sample_count": (self.kleidiai.tokens_per_second.sample_count),
                "duration_ns_median": kleidiai_duration,
                "duration_ns_sample_count": self.kleidiai.duration_ns.sample_count,
            },
            "median_throughput_speedup": throughput_speedup,
            "median_throughput_percent_change": (throughput_speedup - 1.0) * 100.0,
            "median_duration_percent_change": ((kleidiai_duration / generic_duration) - 1.0)
            * 100.0,
        }


@dataclass(frozen=True, slots=True)
class LlamaBenchComparison:
    """Validated comparison of one generic and one KleidiAI summary."""

    build_commit: str
    model_filename: str
    synthetic_fixture: bool
    generic_label: str
    kleidiai_label: str
    tests: tuple[LlamaBenchWorkloadComparison, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "comparison_type": "generic-vs-kleidiai",
            "build_commit": self.build_commit,
            "model_filename": self.model_filename,
            "synthetic_fixture": self.synthetic_fixture,
            "compatibility": {
                "validated": True,
                "same_build_commit": True,
                "same_model_filename": True,
                "same_benchmark_settings_except_kleidiai": True,
                "same_workload_shapes": True,
                "same_sample_counts": True,
                "same_synthetic_status": True,
            },
            "variants": {
                "generic": {"label": self.generic_label},
                "kleidiai": {"label": self.kleidiai_label},
            },
            "tests": {test.test_kind: test.to_mapping() for test in self.tests},
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_mapping()


def compare_llama_bench_summaries(
    generic: Mapping[str, Any],
    kleidiai: Mapping[str, Any],
) -> LlamaBenchComparison:
    """Validate and compare generic and KleidiAI variant summaries.

    The two summaries must identify the exact same llama.cpp commit, model
    filename, workload kinds and shapes, and synthetic-evidence status.  A
    mismatch is an error rather than a partially comparable report.
    """

    generic_summary = _VariantSnapshot.from_mapping(
        _object(generic, context="generic"),
        context="generic",
    )
    kleidiai_summary = _VariantSnapshot.from_mapping(
        _object(kleidiai, context="kleidiai"),
        context="kleidiai",
    )

    if generic_summary.build_commit != kleidiai_summary.build_commit:
        raise LlamaBenchComparisonError(
            "generic and KleidiAI summaries use different build commits: "
            f"{generic_summary.build_commit!r} and {kleidiai_summary.build_commit!r}"
        )
    if generic_summary.model_filename != kleidiai_summary.model_filename:
        raise LlamaBenchComparisonError(
            "generic and KleidiAI summaries use different model filenames: "
            f"{generic_summary.model_filename!r} and "
            f"{kleidiai_summary.model_filename!r}"
        )
    generic_settings = _settings_without_kleidiai(
        generic_summary.settings,
        role="generic",
        expected_flag=False,
    )
    kleidiai_settings = _settings_without_kleidiai(
        kleidiai_summary.settings,
        role="kleidiai",
        expected_flag=True,
    )
    if generic_settings != kleidiai_settings:
        raise LlamaBenchComparisonError(
            "generic and KleidiAI benchmark settings differ beyond build.kleidiai: "
            f"{json.dumps(generic_settings, sort_keys=True, separators=(',', ':'))} != "
            f"{json.dumps(kleidiai_settings, sort_keys=True, separators=(',', ':'))}"
        )
    if generic_summary.synthetic_fixture != kleidiai_summary.synthetic_fixture:
        raise LlamaBenchComparisonError(
            "generic and KleidiAI summaries have different synthetic status"
        )
    if set(generic_summary.tests) != set(kleidiai_summary.tests):
        raise LlamaBenchComparisonError(
            "generic and KleidiAI summaries contain different workload kinds"
        )

    comparisons: list[LlamaBenchWorkloadComparison] = []
    for test_kind in _TEST_ORDER:
        if test_kind not in generic_summary.tests:
            continue
        generic_test = generic_summary.tests[test_kind]
        kleidiai_test = kleidiai_summary.tests[test_kind]
        generic_shape = (generic_test.n_prompt, generic_test.n_gen)
        kleidiai_shape = (kleidiai_test.n_prompt, kleidiai_test.n_gen)
        if generic_shape != kleidiai_shape:
            raise LlamaBenchComparisonError(
                f"generic and KleidiAI {test_kind} workload shapes differ: "
                f"{generic_shape!r} and {kleidiai_shape!r}"
            )
        generic_sample_count = generic_test.tokens_per_second.sample_count
        kleidiai_sample_count = kleidiai_test.tokens_per_second.sample_count
        if generic_sample_count != kleidiai_sample_count:
            raise LlamaBenchComparisonError(
                f"generic and KleidiAI {test_kind} sample counts differ: "
                f"{generic_sample_count} and {kleidiai_sample_count}"
            )
        comparisons.append(
            LlamaBenchWorkloadComparison(
                test_kind=test_kind,
                n_prompt=generic_test.n_prompt,
                n_gen=generic_test.n_gen,
                generic=generic_test,
                kleidiai=kleidiai_test,
            )
        )

    return LlamaBenchComparison(
        build_commit=generic_summary.build_commit,
        model_filename=generic_summary.model_filename,
        synthetic_fixture=generic_summary.synthetic_fixture,
        generic_label=generic_summary.label,
        kleidiai_label=kleidiai_summary.label,
        tests=tuple(comparisons),
    )
