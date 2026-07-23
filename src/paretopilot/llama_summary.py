"""Aggregate compatible ``llama-bench`` artifacts without losing provenance.

The upstream JSONL output does not contain every setting needed to decide
whether benchmark runs are comparable.  Callers therefore attach an explicit
``settings`` mapping to every artifact.  A variant summary is produced only
when all artifacts agree on build commit, model, test shapes, settings, and
synthetic-evidence status.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from paretopilot.domain import ValidationError
from paretopilot.llama_bench import (
    LlamaBenchRecord,
    TestKind,
    load_llama_bench_jsonl,
)


_TEST_ORDER: tuple[TestKind, ...] = ("pp", "tg", "pg")


class LlamaBenchSummaryError(ValidationError):
    """Raised when benchmark artifacts cannot form one comparable summary."""


def _non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LlamaBenchSummaryError(f"{field_name} must be a non-empty string")
    return value


def _json_value(value: Any, *, context: str) -> Any:
    """Copy a value into a deterministic JSON-compatible representation."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LlamaBenchSummaryError(f"{context} must be finite")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LlamaBenchSummaryError(f"{context} keys must be strings")
            copied[key] = _json_value(item, context=f"{context}.{key}")
        return copied
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_value(item, context=f"{context}[{index}]") for index, item in enumerate(value)
        ]
    raise LlamaBenchSummaryError(f"{context} must be JSON-compatible")


def _settings_copy(settings: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(settings, Mapping):
        raise LlamaBenchSummaryError("settings must be an object")
    copied = _json_value(settings, context="settings")
    assert isinstance(copied, dict)
    return copied


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class LabeledLlamaBenchArtifact:
    """One named JSONL artifact and the settings under which it was produced."""

    label: str
    source_file: str
    records: tuple[LlamaBenchRecord, ...]
    settings: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _non_empty_string(self.label, field_name="label"))
        object.__setattr__(
            self,
            "source_file",
            _non_empty_string(self.source_file, field_name="source_file"),
        )
        records = tuple(self.records)
        if not records:
            raise LlamaBenchSummaryError(f"artifact {self.label!r} has no records")
        for index, record in enumerate(records):
            if not isinstance(record, LlamaBenchRecord):
                raise LlamaBenchSummaryError(
                    f"artifact {self.label!r} records[{index}] is not a LlamaBenchRecord"
                )
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "settings", _settings_copy(self.settings))

    @classmethod
    def from_path(
        cls,
        label: str,
        path: str | Path,
        *,
        settings: Mapping[str, Any],
    ) -> "LabeledLlamaBenchArtifact":
        """Load a labeled artifact while retaining its supplied file name."""

        resolved_path = Path(path)
        return cls(
            label=label,
            source_file=str(path),
            records=load_llama_bench_jsonl(resolved_path),
            settings=settings,
        )

    @classmethod
    def from_records(
        cls,
        label: str,
        records: Iterable[LlamaBenchRecord],
        *,
        source_file: str,
        settings: Mapping[str, Any],
    ) -> "LabeledLlamaBenchArtifact":
        """Build an artifact from records already validated by the parser."""

        return cls(
            label=label,
            source_file=source_file,
            records=tuple(records),
            settings=settings,
        )


@dataclass(frozen=True, slots=True)
class SampleStatistics:
    """Serializable descriptive statistics for one sample series."""

    sample_count: int
    mean: float
    median: float
    sample_stdev: float
    minimum: float
    maximum: float

    @classmethod
    def from_samples(cls, samples: Iterable[float]) -> "SampleStatistics":
        values = tuple(float(value) for value in samples)
        if not values:
            raise LlamaBenchSummaryError("cannot summarize an empty sample series")
        # A one-observation sample has no estimable dispersion.  Reporting 0.0
        # is the conventional, JSON-safe representation used by this schema.
        sample_stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        return cls(
            sample_count=len(values),
            mean=statistics.fmean(values),
            median=float(statistics.median(values)),
            sample_stdev=sample_stdev,
            minimum=min(values),
            maximum=max(values),
        )

    def to_dict(self) -> dict[str, int | float]:
        """Return built-in values ready for ``json.dumps``."""

        return {
            "sample_count": self.sample_count,
            "mean": self.mean,
            "median": self.median,
            "sample_stdev": self.sample_stdev,
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class LlamaBenchTestSummary:
    """Aggregated repetitions for one prompt/generation workload shape."""

    test_kind: TestKind
    n_prompt: int
    n_gen: int
    tokens_per_second: SampleStatistics
    duration_ns: SampleStatistics
    source_files: tuple[str, ...]
    source_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_kind": self.test_kind,
            "n_prompt": self.n_prompt,
            "n_gen": self.n_gen,
            "tokens_per_second": self.tokens_per_second.to_dict(),
            "duration_ns": self.duration_ns.to_dict(),
            "source_files": list(self.source_files),
            "source_labels": list(self.source_labels),
        }


@dataclass(frozen=True, slots=True)
class LlamaBenchVariantSummary:
    """A serializable summary of multiple compatible artifacts for one variant."""

    label: str
    build_commit: str
    model_filename: str
    settings: Mapping[str, Any]
    synthetic_fixture: bool
    source_files: tuple[str, ...]
    source_labels: tuple[str, ...]
    tests: tuple[LlamaBenchTestSummary, ...]

    def by_kind(self, test_kind: TestKind) -> LlamaBenchTestSummary:
        for test in self.tests:
            if test.test_kind == test_kind:
                return test
        raise KeyError(test_kind)

    @property
    def test_shapes(self) -> dict[TestKind, tuple[int, int]]:
        return {test.test_kind: (test.n_prompt, test.n_gen) for test in self.tests}

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, entirely JSON-compatible mapping."""

        return {
            "schema_version": "1.0",
            "label": self.label,
            "build_commit": self.build_commit,
            "model_filename": self.model_filename,
            "settings": _settings_copy(self.settings),
            "synthetic_fixture": self.synthetic_fixture,
            "source_files": list(self.source_files),
            "source_labels": list(self.source_labels),
            "tests": {test.test_kind: test.to_dict() for test in self.tests},
        }

    def to_mapping(self) -> dict[str, Any]:
        """Match the serialization convention used by other ParetoPilot reports."""

        return self.to_dict()


@dataclass(frozen=True, slots=True)
class _ArtifactContext:
    build_commit: str
    model_filename: str
    shapes: Mapping[TestKind, tuple[int, int]]
    settings_json: str
    synthetic_fixture: bool


def _artifact_context(artifact: LabeledLlamaBenchArtifact) -> _ArtifactContext:
    commits = {record.build_commit for record in artifact.records}
    if len(commits) != 1:
        raise LlamaBenchSummaryError(
            f"artifact {artifact.label!r} mixes build commits: {sorted(commits)!r}"
        )
    models = {record.model_filename for record in artifact.records}
    if len(models) != 1:
        raise LlamaBenchSummaryError(
            f"artifact {artifact.label!r} mixes models: {sorted(models)!r}"
        )
    synthetic_values = {record.synthetic_fixture for record in artifact.records}
    if len(synthetic_values) != 1:
        raise LlamaBenchSummaryError(
            f"artifact {artifact.label!r} mixes synthetic and measured records"
        )

    _validate_settings_against_records(artifact)

    shapes: dict[TestKind, tuple[int, int]] = {}
    for record in artifact.records:
        shape = (record.n_prompt, record.n_gen)
        previous = shapes.setdefault(record.test_kind, shape)
        if previous != shape:
            raise LlamaBenchSummaryError(
                f"artifact {artifact.label!r} has multiple {record.test_kind} "
                f"test shapes: {previous!r} and {shape!r}"
            )

    return _ArtifactContext(
        build_commit=next(iter(commits)),
        model_filename=next(iter(models)),
        shapes=shapes,
        settings_json=_canonical_json(artifact.settings),
        synthetic_fixture=next(iter(synthetic_values)),
    )


def _validate_settings_against_records(
    artifact: LabeledLlamaBenchArtifact,
) -> None:
    """Reconcile caller metadata with every runtime value reported by JSONL."""

    field_to_setting = {
        "n_threads": "threads",
        "n_batch": "batch_size",
        "n_ubatch": "ubatch_size",
        "n_gpu_layers": "n_gpu_layers",
        "devices": "devices",
        "no_op_offload": "no_op_offload",
    }
    derived_cpu_only = {
        "n_gpu_layers": 0,
        "devices": "none",
        "no_op_offload": 1,
    }
    for record_field, setting_name in field_to_setting.items():
        reported = {getattr(record, record_field) for record in artifact.records}
        if reported == {None}:
            continue
        if None in reported:
            raise LlamaBenchSummaryError(
                f"artifact {artifact.label!r} only partially reports {record_field}"
            )
        if len(reported) != 1:
            raise LlamaBenchSummaryError(
                f"artifact {artifact.label!r} reports multiple {record_field} values: "
                f"{sorted(reported, key=str)!r}"
            )
        expected = artifact.settings.get(setting_name)
        if expected is None and artifact.settings.get("cpu_only") is True:
            expected = derived_cpu_only.get(record_field)
        if expected is None:
            raise LlamaBenchSummaryError(
                f"artifact {artifact.label!r} reports {record_field} but settings omit "
                f"{setting_name}"
            )
        actual = next(iter(reported))
        if actual != expected:
            raise LlamaBenchSummaryError(
                f"artifact {artifact.label!r} reported {record_field}={actual!r} "
                f"but settings declare {setting_name}={expected!r}"
            )


def _describe_shape(shape: Mapping[TestKind, tuple[int, int]]) -> str:
    return ", ".join(
        f"{kind}=({shape[kind][0]},{shape[kind][1]})" for kind in _TEST_ORDER if kind in shape
    )


def summarize_llama_bench_variant(
    label: str,
    artifacts: Iterable[LabeledLlamaBenchArtifact],
) -> LlamaBenchVariantSummary:
    """Combine compatible repetitions into one summary for ``label``.

    Every artifact must contain the same set of test kinds and the same
    ``(n_prompt, n_gen)`` shape for each kind.  Repeated rows with that same
    shape are allowed and their sample arrays are pooled.
    """

    variant_label = _non_empty_string(label, field_name="variant label")
    artifact_values = tuple(artifacts)
    if not artifact_values:
        raise LlamaBenchSummaryError("at least one artifact is required")
    for index, artifact in enumerate(artifact_values):
        if not isinstance(artifact, LabeledLlamaBenchArtifact):
            raise LlamaBenchSummaryError(f"artifacts[{index}] is not a LabeledLlamaBenchArtifact")

    labels = [artifact.label for artifact in artifact_values]
    if len(labels) != len(set(labels)):
        raise LlamaBenchSummaryError("artifact labels must be unique")

    contexts = tuple(_artifact_context(artifact) for artifact in artifact_values)
    expected = contexts[0]
    for artifact, context in zip(artifact_values[1:], contexts[1:], strict=True):
        if context.build_commit != expected.build_commit:
            raise LlamaBenchSummaryError(
                f"artifact {artifact.label!r} build commit {context.build_commit!r} "
                f"does not match {expected.build_commit!r}"
            )
        if context.model_filename != expected.model_filename:
            raise LlamaBenchSummaryError(
                f"artifact {artifact.label!r} model {context.model_filename!r} "
                f"does not match {expected.model_filename!r}"
            )
        if dict(context.shapes) != dict(expected.shapes):
            raise LlamaBenchSummaryError(
                f"artifact {artifact.label!r} test shapes "
                f"{_describe_shape(context.shapes)!r} do not match "
                f"{_describe_shape(expected.shapes)!r}"
            )
        if context.settings_json != expected.settings_json:
            raise LlamaBenchSummaryError(
                f"artifact {artifact.label!r} settings do not match the first artifact"
            )
        if context.synthetic_fixture != expected.synthetic_fixture:
            raise LlamaBenchSummaryError(
                f"artifact {artifact.label!r} synthetic status does not match the first artifact"
            )

    summaries: list[LlamaBenchTestSummary] = []
    for test_kind in _TEST_ORDER:
        shape = expected.shapes.get(test_kind)
        if shape is None:
            continue
        samples_ts: list[float] = []
        samples_ns: list[float] = []
        source_files: list[str] = []
        source_labels: list[str] = []
        for artifact in artifact_values:
            matching = [record for record in artifact.records if record.test_kind == test_kind]
            if matching:
                source_files.append(artifact.source_file)
                source_labels.append(artifact.label)
            for record in matching:
                samples_ts.extend(record.samples_ts)
                samples_ns.extend(record.samples_ns)

        summaries.append(
            LlamaBenchTestSummary(
                test_kind=test_kind,
                n_prompt=shape[0],
                n_gen=shape[1],
                tokens_per_second=SampleStatistics.from_samples(samples_ts),
                duration_ns=SampleStatistics.from_samples(samples_ns),
                source_files=tuple(source_files),
                source_labels=tuple(source_labels),
            )
        )

    return LlamaBenchVariantSummary(
        label=variant_label,
        build_commit=expected.build_commit,
        model_filename=expected.model_filename,
        settings=_settings_copy(artifact_values[0].settings),
        synthetic_fixture=expected.synthetic_fixture,
        source_files=tuple(artifact.source_file for artifact in artifact_values),
        source_labels=tuple(labels),
        tests=tuple(summaries),
    )


def summarize_llama_bench_paths(
    label: str,
    labeled_paths: Mapping[str, str | Path] | Iterable[tuple[str, str | Path]],
    *,
    settings: Mapping[str, Any],
) -> LlamaBenchVariantSummary:
    """Load and summarize ``label -> JSONL path`` entries with shared settings."""

    items = labeled_paths.items() if isinstance(labeled_paths, Mapping) else labeled_paths
    artifacts = (
        LabeledLlamaBenchArtifact.from_path(
            artifact_label,
            path,
            settings=settings,
        )
        for artifact_label, path in items
    )
    return summarize_llama_bench_variant(label, artifacts)


def summarize_llama_bench_records(
    label: str,
    labeled_records: Mapping[str, Iterable[LlamaBenchRecord]]
    | Iterable[tuple[str, Iterable[LlamaBenchRecord]]],
    *,
    settings: Mapping[str, Any],
    source_files: Mapping[str, str] | None = None,
) -> LlamaBenchVariantSummary:
    """Summarize labeled, pre-validated record groups with shared settings."""

    items = labeled_records.items() if isinstance(labeled_records, Mapping) else labeled_records
    artifact_values: list[LabeledLlamaBenchArtifact] = []
    for artifact_label, records in items:
        source_file = (
            source_files.get(artifact_label, artifact_label)
            if source_files is not None
            else artifact_label
        )
        artifact_values.append(
            LabeledLlamaBenchArtifact.from_records(
                artifact_label,
                records,
                source_file=source_file,
                settings=settings,
            )
        )
    return summarize_llama_bench_variant(label, artifact_values)
