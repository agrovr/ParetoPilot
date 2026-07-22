from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from paretopilot.domain import ValidationError
from paretopilot.io import load_benchmarks, sha256_file, write_json


def valid_benchmark_json(*, parameters: str = "{}") -> str:
    return (
        '{"schema_version":"1.0","baseline_id":"baseline","synthetic":true,'
        '"candidates":[{"id":"baseline","parameters":'
        + parameters
        + ',"metrics":{"quality_score":1.0}}]}'
    )


class StrictJsonInputTests(unittest.TestCase):
    def test_rejects_duplicate_keys_at_any_depth(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(
                valid_benchmark_json(parameters='{"threads":4,"threads":8}'),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValidationError, "duplicate JSON object key 'threads'"):
                load_benchmarks(path)

    def test_rejects_non_standard_constants_in_nested_values(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nan.json"
            path.write_text(
                valid_benchmark_json(parameters='{"temperature":NaN}'),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValidationError, "non-standard JSON constant 'NaN'"):
                load_benchmarks(path)

    def test_rejects_finite_json_number_that_overflows_to_infinity(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "overflow.json"
            path.write_text(
                valid_benchmark_json(parameters='{"temperature":1e10000}'),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValidationError, "must not contain NaN or Infinity"):
                load_benchmarks(path)

    def test_converts_numeric_overflow_to_validation_error(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "huge-metric.json"
            path.write_text(
                valid_benchmark_json().replace(
                    '"quality_score":1.0',
                    '"quality_score":' + "1" + "0" * 400,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValidationError, "invalid benchmark data"):
                load_benchmarks(path)

    def test_normalizes_filesystem_read_errors(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValidationError, "could not read"):
                load_benchmarks(Path(directory))


class StrictJsonOutputTests(unittest.TestCase):
    def test_writes_utf8_json_with_stable_trailing_newline(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "output.json"
            write_json(path, {"label": "Café", "warnings": ("one",)})

            raw = path.read_bytes()
            self.assertTrue(raw.endswith(b"\n"))
            self.assertIn("Café".encode(), raw)
            self.assertEqual(json.loads(raw), {"label": "Café", "warnings": ["one"]})

    def test_refuses_existing_destination_and_preserves_it(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            path.write_text("original\n", encoding="utf-8")

            with self.assertRaisesRegex(ValidationError, "refusing to overwrite"):
                write_json(path, {"replacement": True})

            self.assertEqual(path.read_text(encoding="utf-8"), "original\n")
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_explicit_overwrite_replaces_existing_destination(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            path.write_text("original\n", encoding="utf-8")

            write_json(path, {"replacement": True}, overwrite=True)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"replacement": True})

    def test_rejects_nested_nonfinite_and_unsupported_values(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            with self.assertRaisesRegex(ValidationError, "NaN or Infinity"):
                write_json(path, {"nested": [{"value": float("inf")} ]})
            with self.assertRaisesRegex(ValidationError, "unsupported JSON value type"):
                write_json(path, {"nested": {"not-json"}})
            self.assertFalse(path.exists())


class Sha256Tests(unittest.TestCase):
    def test_hashes_file_and_normalizes_read_errors(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"abc")

            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )
            with self.assertRaisesRegex(ValidationError, "could not hash"):
                sha256_file(Path(directory) / "missing.bin")


if __name__ == "__main__":
    unittest.main()
