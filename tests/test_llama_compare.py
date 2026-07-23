from __future__ import annotations

from copy import deepcopy
import math
import unittest

from paretopilot.llama_compare import (
    LlamaBenchComparisonError,
    compare_llama_bench_summaries,
)


COMMIT = "67b9b0e7f6ce45d929a4411907d3c48ec719e81c"


def metric(median: float, sample_count: int = 20) -> dict[str, float | int]:
    return {
        "sample_count": sample_count,
        "mean": median,
        "median": median,
        "sample_stdev": 1.0,
        "min": median - 1.0,
        "max": median + 1.0,
    }


def summary(
    label: str,
    *,
    pp_throughput: float,
    pp_duration: float,
    tg_throughput: float = 20.0,
    tg_duration: float = 100.0,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "label": label,
        "build_commit": COMMIT,
        "model_filename": "model-q4_0.gguf",
        "settings": {
            "threads": 4,
            "batch_size": 512,
            "repetitions": 10,
            "build": {"kleidiai": label == "optimized", "native": True},
        },
        "synthetic_fixture": False,
        "source_files": [f"{label}.jsonl"],
        "source_labels": [label],
        "tests": {
            "pp": {
                "test_kind": "pp",
                "n_prompt": 512,
                "n_gen": 0,
                "tokens_per_second": metric(pp_throughput, 20),
                "duration_ns": metric(pp_duration, 20),
                "source_files": [f"{label}.jsonl"],
                "source_labels": [label],
            },
            "tg": {
                "test_kind": "tg",
                "n_prompt": 0,
                "n_gen": 128,
                "tokens_per_second": metric(tg_throughput, 18),
                "duration_ns": metric(tg_duration, 18),
                "source_files": [f"{label}.jsonl"],
                "source_labels": [label],
            },
        },
    }


class LlamaBenchComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generic = summary(
            "generic",
            pp_throughput=100.0,
            pp_duration=10.0,
        )
        self.kleidiai = summary(
            "optimized",
            pp_throughput=125.0,
            pp_duration=8.0,
            tg_throughput=24.0,
            tg_duration=80.0,
        )

    def test_computes_median_changes_and_preserves_sample_counts(self) -> None:
        payload = compare_llama_bench_summaries(
            self.generic,
            self.kleidiai,
        ).to_mapping()

        self.assertEqual(payload["comparison_type"], "generic-vs-kleidiai")
        self.assertEqual(payload["build_commit"], COMMIT)
        self.assertTrue(payload["compatibility"]["validated"])
        self.assertTrue(payload["compatibility"]["same_benchmark_settings_except_kleidiai"])
        self.assertTrue(payload["compatibility"]["same_sample_counts"])
        self.assertEqual(list(payload["tests"]), ["pp", "tg"])
        pp = payload["tests"]["pp"]
        self.assertEqual(pp["median_throughput_speedup"], 1.25)
        self.assertEqual(pp["median_throughput_percent_change"], 25.0)
        self.assertTrue(math.isclose(pp["median_duration_percent_change"], -20.0))
        self.assertEqual(pp["generic"]["tokens_per_second_sample_count"], 20)
        self.assertEqual(pp["kleidiai"]["duration_ns_sample_count"], 20)

    def test_rejects_different_build_commits(self) -> None:
        incompatible = deepcopy(self.kleidiai)
        incompatible["build_commit"] = "different"

        with self.assertRaisesRegex(LlamaBenchComparisonError, "different build commits"):
            compare_llama_bench_summaries(self.generic, incompatible)

    def test_rejects_different_model_filenames(self) -> None:
        incompatible = deepcopy(self.kleidiai)
        incompatible["model_filename"] = "other.gguf"

        with self.assertRaisesRegex(LlamaBenchComparisonError, "different model filenames"):
            compare_llama_bench_summaries(self.generic, incompatible)

    def test_rejects_benchmark_setting_differences_beyond_kleidiai(self) -> None:
        changes = (
            ("threads", 8),
            ("batch_size", 256),
            ("repetitions", 20),
        )
        for key, value in changes:
            with self.subTest(key=key):
                incompatible = deepcopy(self.kleidiai)
                incompatible["settings"][key] = value
                with self.assertRaisesRegex(
                    LlamaBenchComparisonError,
                    "settings differ beyond build.kleidiai",
                ):
                    compare_llama_bench_summaries(self.generic, incompatible)

        incompatible_build = deepcopy(self.kleidiai)
        incompatible_build["settings"]["build"]["native"] = False
        with self.assertRaisesRegex(
            LlamaBenchComparisonError,
            "settings differ beyond build.kleidiai",
        ):
            compare_llama_bench_summaries(self.generic, incompatible_build)

    def test_requires_explicit_role_correct_kleidiai_flags(self) -> None:
        wrong_generic = deepcopy(self.generic)
        wrong_generic["settings"]["build"]["kleidiai"] = True
        with self.assertRaisesRegex(
            LlamaBenchComparisonError,
            r"generic\.settings\.build\.kleidiai must be false",
        ):
            compare_llama_bench_summaries(wrong_generic, self.kleidiai)

        wrong_optimized = deepcopy(self.kleidiai)
        wrong_optimized["settings"]["build"]["kleidiai"] = False
        with self.assertRaisesRegex(
            LlamaBenchComparisonError,
            r"kleidiai\.settings\.build\.kleidiai must be true",
        ):
            compare_llama_bench_summaries(self.generic, wrong_optimized)

        missing_flag = deepcopy(self.generic)
        del missing_flag["settings"]["build"]["kleidiai"]
        with self.assertRaisesRegex(
            LlamaBenchComparisonError,
            r"generic\.settings\.build\.kleidiai must be false",
        ):
            compare_llama_bench_summaries(missing_flag, self.kleidiai)

    def test_rejects_different_workload_kinds_or_shapes(self) -> None:
        missing_test = deepcopy(self.kleidiai)
        del missing_test["tests"]["tg"]
        with self.assertRaisesRegex(LlamaBenchComparisonError, "different workload kinds"):
            compare_llama_bench_summaries(self.generic, missing_test)

        changed_shape = deepcopy(self.kleidiai)
        changed_shape["tests"]["pp"]["n_prompt"] = 256
        with self.assertRaisesRegex(LlamaBenchComparisonError, "workload shapes differ"):
            compare_llama_bench_summaries(self.generic, changed_shape)

    def test_rejects_different_synthetic_status(self) -> None:
        incompatible = deepcopy(self.kleidiai)
        incompatible["synthetic_fixture"] = True

        with self.assertRaisesRegex(LlamaBenchComparisonError, "synthetic status"):
            compare_llama_bench_summaries(self.generic, incompatible)

    def test_rejects_nonpositive_medians_and_malformed_shapes(self) -> None:
        invalid_metric = deepcopy(self.generic)
        invalid_metric["tests"]["pp"]["tokens_per_second"]["median"] = 0.0
        with self.assertRaisesRegex(LlamaBenchComparisonError, "finite and positive"):
            compare_llama_bench_summaries(invalid_metric, self.kleidiai)

        invalid_shape = deepcopy(self.generic)
        invalid_shape["tests"]["pp"]["n_prompt"] = 0
        with self.assertRaisesRegex(LlamaBenchComparisonError, "not a pp workload"):
            compare_llama_bench_summaries(invalid_shape, self.kleidiai)

    def test_rejects_metric_sample_count_mismatch(self) -> None:
        incompatible = deepcopy(self.generic)
        incompatible["tests"]["pp"]["duration_ns"]["sample_count"] = 19

        with self.assertRaisesRegex(LlamaBenchComparisonError, "sample counts must match"):
            compare_llama_bench_summaries(incompatible, self.kleidiai)

    def test_rejects_cross_variant_sample_count_mismatch(self) -> None:
        incompatible = deepcopy(self.kleidiai)
        incompatible["tests"]["pp"]["tokens_per_second"]["sample_count"] = 19
        incompatible["tests"]["pp"]["duration_ns"]["sample_count"] = 19

        with self.assertRaisesRegex(LlamaBenchComparisonError, "pp sample counts differ"):
            compare_llama_bench_summaries(self.generic, incompatible)


if __name__ == "__main__":
    unittest.main()
