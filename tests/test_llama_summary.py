from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from paretopilot.llama_bench import LlamaBenchRecord, parse_llama_bench_row
from paretopilot.llama_summary import (
    LabeledLlamaBenchArtifact,
    LlamaBenchSummaryError,
    SampleStatistics,
    summarize_llama_bench_paths,
    summarize_llama_bench_records,
    summarize_llama_bench_variant,
)


COMMIT = "67b9b0e7f6ce45d929a4411907d3c48ec719e81c"
SETTINGS = {
    "threads": 8,
    "batch_size": 512,
    "warmup": True,
    "cpu_only": True,
    "build": {"kleidiai": False},
}


def record(
    kind: str,
    samples_ts: list[float],
    samples_ns: list[float],
    *,
    commit: str = COMMIT,
    model: str = "model.gguf",
    synthetic: bool = False,
    shape: tuple[int, int] | None = None,
) -> LlamaBenchRecord:
    shapes = {"pp": (512, 0), "tg": (0, 128), "pg": (64, 32)}
    n_prompt, n_gen = shape or shapes[kind]
    return parse_llama_bench_row(
        {
            "build_commit": commit,
            "model_filename": model,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "avg_ns": sum(samples_ns) / len(samples_ns),
            "avg_ts": sum(samples_ts) / len(samples_ts),
            "samples_ns": samples_ns,
            "samples_ts": samples_ts,
            "synthetic_fixture": synthetic,
        }
    )


def artifact(
    label: str,
    records: list[LlamaBenchRecord],
    *,
    settings: object = SETTINGS,
) -> LabeledLlamaBenchArtifact:
    assert isinstance(settings, dict)
    return LabeledLlamaBenchArtifact.from_records(
        label,
        records,
        source_file=f"{label}.jsonl",
        settings=settings,
    )


class SampleStatisticsTests(unittest.TestCase):
    def test_computes_descriptive_statistics_with_sample_stdev(self) -> None:
        summary = SampleStatistics.from_samples([10.0, 12.0, 14.0, 16.0])

        self.assertEqual(summary.sample_count, 4)
        self.assertEqual(summary.mean, 13.0)
        self.assertEqual(summary.median, 13.0)
        self.assertTrue(math.isclose(summary.sample_stdev, math.sqrt(20 / 3)))
        self.assertEqual(summary.minimum, 10.0)
        self.assertEqual(summary.maximum, 16.0)

    def test_single_sample_has_zero_dispersion(self) -> None:
        summary = SampleStatistics.from_samples([7.5])

        self.assertEqual(summary.sample_stdev, 0.0)


class LlamaBenchVariantSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = artifact(
            "run-01",
            [
                record("pp", [100.0, 110.0], [10.0, 12.0]),
                record("tg", [20.0, 22.0], [40.0, 42.0]),
                record("pg", [50.0, 52.0], [20.0, 22.0]),
            ],
        )
        self.second = artifact(
            "run-02",
            [
                record("pp", [120.0, 130.0], [14.0, 16.0]),
                record("tg", [24.0, 26.0], [44.0, 46.0]),
                record("pg", [54.0, 56.0], [24.0, 26.0]),
            ],
        )

    def test_combines_samples_by_kind_and_preserves_sources(self) -> None:
        summary = summarize_llama_bench_variant(
            "generic",
            [self.first, self.second],
        )

        self.assertEqual(summary.build_commit, COMMIT)
        self.assertEqual(summary.model_filename, "model.gguf")
        self.assertEqual(summary.test_shapes, {"pp": (512, 0), "tg": (0, 128), "pg": (64, 32)})
        self.assertEqual([test.test_kind for test in summary.tests], ["pp", "tg", "pg"])
        pp = summary.by_kind("pp")
        self.assertEqual(pp.tokens_per_second.sample_count, 4)
        self.assertEqual(pp.tokens_per_second.mean, 115.0)
        self.assertEqual(pp.tokens_per_second.median, 115.0)
        self.assertTrue(
            math.isclose(pp.tokens_per_second.sample_stdev, math.sqrt(500 / 3))
        )
        self.assertEqual(pp.tokens_per_second.minimum, 100.0)
        self.assertEqual(pp.tokens_per_second.maximum, 130.0)
        self.assertEqual(pp.duration_ns.sample_count, 4)
        self.assertEqual(pp.duration_ns.mean, 13.0)
        self.assertEqual(pp.source_files, ("run-01.jsonl", "run-02.jsonl"))
        self.assertEqual(summary.source_labels, ("run-01", "run-02"))

    def test_mapping_is_json_serializable_and_uses_explicit_metric_names(self) -> None:
        summary = summarize_llama_bench_variant("generic", [self.first, self.second])

        mapping = summary.to_dict()
        encoded = json.dumps(mapping, sort_keys=True, allow_nan=False)

        self.assertIn('"tokens_per_second"', encoded)
        self.assertIn('"duration_ns"', encoded)
        self.assertEqual(mapping["tests"]["pp"]["source_files"], [
            "run-01.jsonl",
            "run-02.jsonl",
        ])
        self.assertEqual(mapping["tests"]["pp"]["tokens_per_second"]["sample_count"], 4)
        self.assertEqual(mapping["tests"]["pp"]["tokens_per_second"]["min"], 100.0)
        self.assertEqual(summary.to_mapping(), mapping)

    def test_rejects_mixed_build_commits(self) -> None:
        incompatible = artifact(
            "other",
            [record("pp", [1.0, 2.0], [3.0, 4.0], commit="different")],
        )
        first = artifact(
            "first",
            [record("pp", [1.0, 2.0], [3.0, 4.0])],
        )

        with self.assertRaisesRegex(LlamaBenchSummaryError, "build commit"):
            summarize_llama_bench_variant("generic", [first, incompatible])

    def test_rejects_mixed_models(self) -> None:
        incompatible = artifact(
            "other",
            [record("pp", [1.0, 2.0], [3.0, 4.0], model="other.gguf")],
        )
        first = artifact(
            "first",
            [record("pp", [1.0, 2.0], [3.0, 4.0])],
        )

        with self.assertRaisesRegex(LlamaBenchSummaryError, "model"):
            summarize_llama_bench_variant("generic", [first, incompatible])

    def test_rejects_mismatched_test_shapes_or_missing_kinds(self) -> None:
        changed_shape = artifact(
            "changed",
            [
                record("pp", [1.0, 2.0], [3.0, 4.0], shape=(256, 0)),
                record("tg", [1.0, 2.0], [3.0, 4.0]),
                record("pg", [1.0, 2.0], [3.0, 4.0]),
            ],
        )

        with self.assertRaisesRegex(LlamaBenchSummaryError, "test shapes"):
            summarize_llama_bench_variant("generic", [self.first, changed_shape])

        missing_kind = artifact(
            "missing",
            [
                record("pp", [1.0, 2.0], [3.0, 4.0]),
                record("tg", [1.0, 2.0], [3.0, 4.0]),
            ],
        )
        with self.assertRaisesRegex(LlamaBenchSummaryError, "test shapes"):
            summarize_llama_bench_variant("generic", [self.first, missing_kind])

    def test_rejects_multiple_shapes_within_one_artifact(self) -> None:
        mixed = artifact(
            "mixed",
            [
                record("pp", [1.0, 2.0], [3.0, 4.0]),
                record("pp", [1.0, 2.0], [3.0, 4.0], shape=(256, 0)),
            ],
        )

        with self.assertRaisesRegex(LlamaBenchSummaryError, "multiple pp test shapes"):
            summarize_llama_bench_variant("generic", [mixed])

    def test_rejects_mismatched_settings_even_when_key_order_differs(self) -> None:
        reordered = artifact(
            "reordered",
            [record("pp", [1.0, 2.0], [3.0, 4.0])],
            settings={"cpu_only": True, "threads": 8},
        )
        equivalent = artifact(
            "equivalent",
            [record("pp", [1.0, 2.0], [3.0, 4.0])],
            settings={"threads": 8, "cpu_only": True},
        )
        summarize_llama_bench_variant("generic", [reordered, equivalent])

        changed = artifact(
            "changed",
            [record("pp", [1.0, 2.0], [3.0, 4.0])],
            settings={"threads": 4, "cpu_only": True},
        )
        with self.assertRaisesRegex(LlamaBenchSummaryError, "settings"):
            summarize_llama_bench_variant("generic", [reordered, changed])

    def test_rejects_mixed_synthetic_and_measured_records(self) -> None:
        measured = artifact(
            "measured",
            [record("pp", [1.0, 2.0], [3.0, 4.0])],
        )
        synthetic = artifact(
            "synthetic",
            [record("pp", [1.0, 2.0], [3.0, 4.0], synthetic=True)],
        )

        with self.assertRaisesRegex(LlamaBenchSummaryError, "synthetic status"):
            summarize_llama_bench_variant("generic", [measured, synthetic])

    def test_records_convenience_accepts_labels_and_source_names(self) -> None:
        summary = summarize_llama_bench_records(
            "generic",
            {
                "morning": [record("pp", [1.0, 2.0], [3.0, 4.0])],
                "evening": [record("pp", [3.0, 4.0], [5.0, 6.0])],
            },
            settings=SETTINGS,
            source_files={"morning": "a.jsonl", "evening": "b.jsonl"},
        )

        self.assertEqual(summary.source_files, ("a.jsonl", "b.jsonl"))
        self.assertEqual(summary.by_kind("pp").tokens_per_second.sample_count, 4)

    def test_paths_convenience_loads_and_preserves_paths(self) -> None:
        row = {
            "build_commit": COMMIT,
            "model_filename": "model.gguf",
            "n_prompt": 512,
            "n_gen": 0,
            "avg_ns": 11.0,
            "avg_ts": 105.0,
            "samples_ns": [10.0, 12.0],
            "samples_ts": [100.0, 110.0],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "run.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            summary = summarize_llama_bench_paths(
                "generic",
                {"run-01": path},
                settings=SETTINGS,
            )

        self.assertEqual(summary.source_files, (str(path),))
        self.assertEqual(summary.by_kind("pp").tokens_per_second.mean, 105.0)

    def test_rejects_non_json_settings(self) -> None:
        with self.assertRaisesRegex(LlamaBenchSummaryError, "JSON-compatible"):
            artifact(
                "bad",
                [record("pp", [1.0, 2.0], [3.0, 4.0])],
                settings={"path": Path("model.gguf")},
            )


if __name__ == "__main__":
    unittest.main()
