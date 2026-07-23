from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from paretopilot.analysis import recommend
from paretopilot.domain import BenchmarkSet, Constraints
from paretopilot.io import load_json_object, sha256_file
from paretopilot.llama_compare import compare_llama_bench_summaries
from paretopilot.study import StudyAssemblyError, assemble_study


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PAIRED_STUDY_FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "paired-study"
REQUIRED_PATHS = {
    "status.json",
    "manifest.json",
    "provenance.json",
    "environment/runner.json",
    "settings/generic.json",
    "settings/kleidiai.json",
    "summary/comparison-pooled.json",
    "summary/comparison-pair-1.json",
    "summary/comparison-pair-2.json",
    "summary/generic.json",
    "summary/kleidiai.json",
    "summary/generic-pass-1.json",
    "summary/kleidiai-pass-1.json",
    "summary/generic-pass-2.json",
    "summary/kleidiai-pass-2.json",
}
COMPARISON_INPUTS = {
    "summary/comparison-pooled.json": (
        "summary/generic.json",
        "summary/kleidiai.json",
    ),
    "summary/comparison-pair-1.json": (
        "summary/generic-pass-1.json",
        "summary/kleidiai-pass-1.json",
    ),
    "summary/comparison-pair-2.json": (
        "summary/generic-pass-2.json",
        "summary/kleidiai-pass-2.json",
    ),
}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _rewrite_checksums(root: Path) -> None:
    entries = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file() and item.name != "SHA256SUMS"),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        entries.append(f"{sha256_file(path)}  {relative}")
    (root / "SHA256SUMS").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _minimal_bundle(destination: Path) -> Path:
    destination.mkdir(parents=True)
    for relative in sorted(REQUIRED_PATHS):
        source = PAIRED_STUDY_FIXTURE / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _rewrite_checksums(destination)
    return destination


def _replace_json(root: Path, relative: str, update: object) -> None:
    payload = dict(load_json_object(root / relative))
    if not callable(update):
        raise TypeError("update must be callable")
    update(payload)
    _write_json(root / relative, payload)


def _rebuild_comparison(root: Path, comparison_path: str) -> None:
    generic_path, kleidiai_path = COMPARISON_INPUTS[comparison_path]
    generic = load_json_object(root / generic_path)
    kleidiai = load_json_object(root / kleidiai_path)
    comparison = compare_llama_bench_summaries(generic, kleidiai).to_mapping()
    comparison["input_fingerprints"] = {
        "generic_summary_sha256": sha256_file(root / generic_path),
        "kleidiai_summary_sha256": sha256_file(root / kleidiai_path),
    }
    _write_json(root / comparison_path, comparison)


def _make_consistent_two_percent_improvement(root: Path) -> None:
    def improve(payload: dict[str, object]) -> None:
        tests = payload["tests"]
        tg = tests["tg"]
        if "pass-2" in str(payload["label"]):
            generic = load_json_object(root / "summary/generic-pass-2.json")
        else:
            generic = load_json_object(root / "summary/generic.json")
        generic_tg = generic["tests"]["tg"]
        tg["tokens_per_second"]["median"] = generic_tg["tokens_per_second"]["median"] * 1.02
        tg["duration_ns"]["median"] = generic_tg["duration_ns"]["median"] / 1.02

    _replace_json(root, "summary/kleidiai-pass-2.json", improve)
    _replace_json(root, "summary/kleidiai.json", improve)
    _rebuild_comparison(root, "summary/comparison-pair-2.json")
    _rebuild_comparison(root, "summary/comparison-pooled.json")
    _rewrite_checksums(root)


class PairedStudyFixtureTests(unittest.TestCase):
    def test_assembles_measured_fixture_and_selects_safe_baseline(self) -> None:
        with TemporaryDirectory() as directory:
            root = _minimal_bundle(Path(directory) / "bundle")
            assembly = assemble_study(root)

            benchmarks = BenchmarkSet.from_mapping(assembly.benchmark_set)
            constraints = Constraints.from_mapping(assembly.constraints)
            result = recommend(benchmarks, constraints)

            generic = benchmarks.by_id("generic-baseline")
            kleidiai = benchmarks.by_id("kleidiai-optimized")
            self.assertEqual(generic.metrics["prompt_tokens_per_second"], 113.6605)
            self.assertEqual(generic.metrics["generation_tokens_per_second"], 35.1134)
            self.assertEqual(generic.metrics["prompt_duration_ms"], 4504.6401985)
            self.assertEqual(kleidiai.metrics["prompt_tokens_per_second"], 114.25800000000001)
            self.assertEqual(
                kleidiai.metrics["generation_tokens_per_second"],
                35.130700000000004,
            )
            self.assertEqual(kleidiai.metrics["generation_duration_ms"], 3643.539655)
            self.assertEqual(assembly.adoption_gate["status"], "inconclusive")
            self.assertEqual(
                assembly.adoption_gate["pair_percent_changes"],
                [2.158336929354876, -1.2144520927712676],
            )
            self.assertEqual(
                assembly.adoption_gate["pooled_percent_change"],
                0.04926894006278548,
            )
            self.assertEqual(result["selected_id"], "generic-baseline")
            self.assertEqual(
                result["rejected"],
                {
                    "kleidiai-optimized": [
                        "confidence_and_practical_effect_gate=0 is below minimum 1"
                    ]
                },
            )
            self.assertFalse(assembly.benchmark_set["synthetic"])

    def test_metadata_preserves_truthful_provenance_hashes_and_quality_basis(self) -> None:
        with TemporaryDirectory() as directory:
            root = _minimal_bundle(Path(directory) / "bundle")
            assembly = assemble_study(root)
            metadata = assembly.benchmark_set["metadata"]

            self.assertEqual(metadata["repository"]["run_id"], "29940067201")
            self.assertEqual(
                metadata["provenance"]["run"]["url"],
                "https://github.com/agrovr/ParetoPilot/actions/runs/29940067201",
            )
            self.assertEqual(metadata["runner"]["machine"], "aarch64")
            self.assertEqual(
                metadata["runner"]["captured_environment"]["RUNNER_ARCH"],
                "ARM64",
            )
            self.assertEqual(metadata["model"]["quantization"], "Q4_0")
            self.assertEqual(metadata["quality_evidence"]["basis"], "identical-model-sha256")
            self.assertFalse(metadata["quality_evidence"]["direct_quality_evaluation"])
            self.assertEqual(
                metadata["quality_evidence"]["metric"],
                "model_identity_quality_retention",
            )
            integrity = metadata["evidence_integrity"]
            self.assertEqual(integrity["verified_entry_count"], len(REQUIRED_PATHS))
            self.assertEqual(set(integrity["verified_files"]), REQUIRED_PATHS)
            manifest_bytes = (root / "SHA256SUMS").read_bytes()
            self.assertEqual(
                integrity["checksum_manifest_sha256"],
                hashlib.sha256(manifest_bytes).hexdigest(),
            )

    def test_output_is_deterministic_across_bundle_locations(self) -> None:
        with TemporaryDirectory() as directory:
            first = _minimal_bundle(Path(directory) / "one")
            second = _minimal_bundle(Path(directory) / "elsewhere" / "two")

            self.assertEqual(
                assemble_study(first).to_mapping(), assemble_study(second).to_mapping()
            )


class IntegrityTests(unittest.TestCase):
    def test_rejects_checksum_tampering_missing_and_unlisted_files(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)

            tampered = _minimal_bundle(base / "tampered")
            (tampered / "status.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(StudyAssemblyError, "SHA256 mismatch for status.json"):
                assemble_study(tampered)

            missing = _minimal_bundle(base / "missing")
            (missing / "manifest.json").unlink()
            with self.assertRaisesRegex(StudyAssemblyError, "entry is missing: manifest.json"):
                assemble_study(missing)

            unlisted = _minimal_bundle(base / "unlisted")
            (unlisted / "surprise.txt").write_text("not checksummed\n", encoding="utf-8")
            with self.assertRaisesRegex(StudyAssemblyError, "files missing from SHA256SUMS"):
                assemble_study(unlisted)

    def test_rejects_traversal_and_duplicate_checksum_entries(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)

            traversal = _minimal_bundle(base / "traversal")
            with (traversal / "SHA256SUMS").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(f"{'0' * 64}  ../outside.json\n")
            with self.assertRaisesRegex(StudyAssemblyError, "unsafe relative path"):
                assemble_study(traversal)

            duplicate = _minimal_bundle(base / "duplicate")
            first_line = (duplicate / "SHA256SUMS").read_text(encoding="utf-8").splitlines()[0]
            with (duplicate / "SHA256SUMS").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(first_line + "\n")
            with self.assertRaisesRegex(StudyAssemblyError, "duplicate entry"):
                assemble_study(duplicate)

    def test_rejects_invalid_canonical_status_after_valid_checksum_update(self) -> None:
        with TemporaryDirectory() as directory:
            root = _minimal_bundle(Path(directory) / "bundle")

            def invalidate(payload: dict[str, object]) -> None:
                payload["status"] = "partial"

            _replace_json(root, "status.json", invalidate)
            _rewrite_checksums(root)
            with self.assertRaisesRegex(
                StudyAssemblyError,
                "status.json status must be 'complete'",
            ):
                assemble_study(root)

    def test_rejects_pair_that_uses_a_different_model(self) -> None:
        with TemporaryDirectory() as directory:
            root = _minimal_bundle(Path(directory) / "bundle")

            def change_model(payload: dict[str, object]) -> None:
                payload["model_filename"] = "/tmp/different-q4_0.gguf"

            _replace_json(root, "summary/generic-pass-2.json", change_model)
            _replace_json(root, "summary/kleidiai-pass-2.json", change_model)
            _rebuild_comparison(root, "summary/comparison-pair-2.json")
            _rewrite_checksums(root)

            with self.assertRaisesRegex(StudyAssemblyError, "same exact model"):
                assemble_study(root)


class AdoptionThresholdTests(unittest.TestCase):
    def test_consistent_practical_effect_passes_then_stricter_threshold_rejects(self) -> None:
        with TemporaryDirectory() as directory:
            root = _minimal_bundle(Path(directory) / "bundle")
            _make_consistent_two_percent_improvement(root)

            passing = assemble_study(root, practical_effect_threshold_percent=1.0)
            passing_result = recommend(
                BenchmarkSet.from_mapping(passing.benchmark_set),
                Constraints.from_mapping(passing.constraints),
            )
            self.assertTrue(passing.adoption_gate["paired_direction_consistent"])
            self.assertTrue(passing.adoption_gate["meets_practical_effect_threshold"])
            self.assertEqual(passing.adoption_gate["status"], "eligible-for-adoption")
            self.assertEqual(passing_result["selected_id"], "kleidiai-optimized")

            rejected = assemble_study(root, practical_effect_threshold_percent=3.0)
            rejected_result = recommend(
                BenchmarkSet.from_mapping(rejected.benchmark_set),
                Constraints.from_mapping(rejected.constraints),
            )
            self.assertTrue(rejected.adoption_gate["paired_direction_consistent"])
            self.assertFalse(rejected.adoption_gate["meets_practical_effect_threshold"])
            self.assertEqual(
                rejected.adoption_gate["status"],
                "below-practical-effect-threshold",
            )
            self.assertEqual(rejected_result["selected_id"], "generic-baseline")

    def test_rejects_invalid_threshold_values(self) -> None:
        with TemporaryDirectory() as directory:
            root = _minimal_bundle(Path(directory) / "bundle")
            for value in (-1.0, float("inf"), True):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(
                        StudyAssemblyError,
                        "practical_effect_threshold_percent",
                    ):
                        assemble_study(root, practical_effect_threshold_percent=value)


if __name__ == "__main__":
    unittest.main()
