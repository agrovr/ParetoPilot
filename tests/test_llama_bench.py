from __future__ import annotations

from pathlib import Path
import unittest

from paretopilot.llama_bench import (
    LlamaBenchParseError,
    load_llama_bench_jsonl,
    parse_llama_bench_jsonl,
    parse_llama_bench_row,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "llama_bench.synthetic.jsonl"


def valid_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "build_commit": "67b9b0e7f6ce45d929a4411907d3c48ec719e81c",
        "model_filename": "synthetic-model.gguf",
        "n_prompt": 512,
        "n_gen": 0,
        "avg_ns": 100_000_000,
        "avg_ts": 5_120.0,
        "samples_ns": [101_000_000, 99_000_000],
        "samples_ts": [5_069.3, 5_171.7],
    }
    row.update(overrides)
    return row


class LlamaBenchParserTests(unittest.TestCase):
    def test_valid_row_is_typed_and_unknown_fields_are_tolerated(self) -> None:
        record = parse_llama_bench_row(
            valid_row(future_upstream_field={"anything": True}),
            source="synthetic.jsonl",
            line_number=7,
        )

        self.assertEqual(record.source_line, 7)
        self.assertEqual(record.test_kind, "pp")
        self.assertEqual(record.repetition_count, 2)
        self.assertEqual(record.samples_ns, (101_000_000.0, 99_000_000.0))
        self.assertFalse(record.synthetic_fixture)

    def test_synthetic_fixture_covers_all_inferred_test_kinds(self) -> None:
        records = load_llama_bench_jsonl(FIXTURE_PATH)

        self.assertEqual([record.test_kind for record in records], ["pp", "tg", "pg"])
        self.assertTrue(all(record.repetition_count == 2 for record in records))
        self.assertTrue(all(record.synthetic_fixture for record in records))

    def test_rejects_non_boolean_synthetic_fixture_marker(self) -> None:
        with self.assertRaisesRegex(LlamaBenchParseError, "synthetic_fixture must be a boolean"):
            parse_llama_bench_row(valid_row(synthetic_fixture="yes"))

    def test_rejects_invalid_test_shape(self) -> None:
        with self.assertRaisesRegex(LlamaBenchParseError, "invalid test shape"):
            parse_llama_bench_row(valid_row(n_prompt=0, n_gen=0))

    def test_rejects_mismatched_sample_counts(self) -> None:
        with self.assertRaisesRegex(LlamaBenchParseError, "same length"):
            parse_llama_bench_row(valid_row(samples_ts=[5_120.0]))

    def test_rejects_empty_sample_arrays(self) -> None:
        with self.assertRaisesRegex(LlamaBenchParseError, "must not be empty"):
            parse_llama_bench_row(valid_row(samples_ns=[], samples_ts=[]))

    def test_allows_integer_rounding_in_reported_nanosecond_average(self) -> None:
        record = parse_llama_bench_row(
            valid_row(
                avg_ns=100_000_000,
                samples_ns=[100_000_000, 100_000_001],
            )
        )

        self.assertEqual(record.avg_ns, 100_000_000.0)

    def test_rejects_inconsistent_nanosecond_average(self) -> None:
        with self.assertRaisesRegex(LlamaBenchParseError, "avg_ns is inconsistent.*samples_ns"):
            parse_llama_bench_row(valid_row(avg_ns=10_000_000))

    def test_rejects_inconsistent_throughput_average(self) -> None:
        with self.assertRaisesRegex(LlamaBenchParseError, "avg_ts is inconsistent.*samples_ts"):
            parse_llama_bench_row(valid_row(avg_ts=500.0))

    def test_bad_json_reports_its_line(self) -> None:
        with self.assertRaisesRegex(LlamaBenchParseError, r"bad\.jsonl:2: invalid JSON"):
            parse_llama_bench_jsonl("\n{not-json}\n", source="bad.jsonl")

    def test_rejects_duplicate_keys_and_nonstandard_numbers(self) -> None:
        duplicate = (
            '{"build_commit":"abc","build_commit":"def","model_filename":"m",'
            '"n_prompt":1,"n_gen":0,"avg_ns":1,"avg_ts":1,'
            '"samples_ns":[1],"samples_ts":[1]}\n'
        )
        with self.assertRaisesRegex(LlamaBenchParseError, "duplicate JSON object key"):
            parse_llama_bench_jsonl(duplicate, source="duplicate.jsonl")

        nonstandard = duplicate.replace(
            '"build_commit":"abc","build_commit":"def"',
            '"build_commit":"abc"',
        ).replace('"avg_ns":1', '"avg_ns":NaN')
        with self.assertRaisesRegex(LlamaBenchParseError, "non-standard JSON constant"):
            parse_llama_bench_jsonl(nonstandard, source="nan.jsonl")

    def test_rejects_unreasonably_large_sample_arrays(self) -> None:
        row = valid_row(
            samples_ns=[1.0] * 10_001,
            samples_ts=[1.0] * 10_001,
            avg_ns=1.0,
            avg_ts=1.0,
        )
        with self.assertRaisesRegex(LlamaBenchParseError, "sample safety limit"):
            parse_llama_bench_row(row)

    def test_truncated_json_reports_its_line(self) -> None:
        valid = (
            '{"build_commit":"abc","model_filename":"m.gguf","n_prompt":8,'
            '"n_gen":0,"avg_ns":1,"avg_ts":8,"samples_ns":[1],"samples_ts":[8]}'
        )
        with self.assertRaisesRegex(LlamaBenchParseError, r"truncated\.jsonl:2: invalid JSON"):
            parse_llama_bench_jsonl(valid + '\n{"build_commit":', source="truncated.jsonl")

    def test_requires_numeric_sample_values(self) -> None:
        with self.assertRaisesRegex(LlamaBenchParseError, r"samples_ts\[1\]"):
            parse_llama_bench_row(valid_row(samples_ts=[5_120.0, "fast"]))

    def test_parses_reported_runtime_settings(self) -> None:
        record = parse_llama_bench_row(
            valid_row(
                n_threads=4,
                n_batch=512,
                n_ubatch=128,
                n_gpu_layers=0,
                devices="none",
                no_op_offload=1,
            )
        )

        self.assertEqual(record.n_threads, 4)
        self.assertEqual(record.n_batch, 512)
        self.assertEqual(record.n_ubatch, 128)
        self.assertEqual(record.n_gpu_layers, 0)
        self.assertEqual(record.devices, "none")
        self.assertEqual(record.no_op_offload, 1)

    def test_rejects_boolean_runtime_integer(self) -> None:
        with self.assertRaisesRegex(LlamaBenchParseError, "n_threads must be an integer"):
            parse_llama_bench_row(valid_row(n_threads=True))


if __name__ == "__main__":
    unittest.main()
