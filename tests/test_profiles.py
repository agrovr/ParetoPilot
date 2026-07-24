from __future__ import annotations

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from paretopilot import cli
from paretopilot.analysis import recommend
from paretopilot.domain import BenchmarkSet, Constraints, ValidationError
from paretopilot.io import write_json
from paretopilot.profiles import (
    PolicySet,
    evaluate_policy_profiles,
    load_policy_set,
)


def _benchmarks(
    *,
    synthetic: bool = True,
    classification: str | None = None,
) -> BenchmarkSet:
    candidates = (
        ("q8-generic", 100.0, 20.0, 4000.0, 50.0, 100.0),
        ("q4-generic", 99.8, 18.0, 2000.0, 44.0, 110.0),
        ("q4-kleidiai", 102.0, 17.8, 2001.0, 40.0, 111.0),
        ("q4-kleidiai-tuned", 100.2, 18.2, 2001.0, 40.2, 130.0),
    )
    return BenchmarkSet.from_mapping(
        {
            "schema_version": "1.0",
            "baseline_id": "q8-generic",
            "synthetic": synthetic,
            "metadata": ({"classification": classification} if classification is not None else {}),
            "candidates": [
                {
                    "id": candidate_id,
                    "label": candidate_id,
                    "parameters": {},
                    "metrics": {
                        "quality_score": 1.0,
                        "e2e_latency_ms_p95": e2e,
                        "generation_tps": generation,
                        "peak_rss_mib": memory,
                        "model_size_mib": memory / 2.0,
                        "ttft_ms_p95": ttft,
                        "prompt_tps": prompt,
                    },
                }
                for candidate_id, e2e, generation, memory, ttft, prompt in candidates
            ],
        }
    )


def _constraints() -> Constraints:
    return Constraints.from_mapping(
        {
            "min_quality_retention": 1.0,
            "quality_metric": "quality_score",
            "max_values": {
                "e2e_latency_ms_p95": 150.0,
                "peak_rss_mib": 5000.0,
            },
            "min_values": {},
            "objective": {
                "metric": "e2e_latency_ms_p95",
                "direction": "min",
            },
            "objective_tolerance_percent": 1.0,
            "preference_order": [
                "q8-generic",
                "q4-generic",
                "q4-kleidiai",
                "q4-kleidiai-tuned",
            ],
            "frontier_metrics": {
                "e2e_latency_ms_p95": "min",
                "generation_tps": "max",
                "model_size_mib": "min",
                "peak_rss_mib": "min",
                "quality_score": "max",
                "ttft_ms_p95": "min",
            },
        }
    )


def _config_mapping() -> dict[str, object]:
    repository = Path(__file__).parents[1]
    return json.loads((repository / "configs" / "policies.arm64.json").read_text(encoding="utf-8"))


class PolicyProfileTests(unittest.TestCase):
    def test_arm64_profiles_make_expected_honest_recommendations(self) -> None:
        policy_set = PolicySet.from_mapping(_config_mapping())

        result = evaluate_policy_profiles(_benchmarks(), _constraints(), policy_set)

        selected = {
            profile["id"]: profile["recommendation"]["selected_id"]
            for profile in result["profiles"]
        }
        self.assertEqual(
            selected,
            {
                "canonical-latency": "q8-generic",
                "memory-first": "q4-generic",
                "first-token-first": "q4-kleidiai",
                "prompt-ingest-first": "q4-kleidiai-tuned",
                "decode-first": "q8-generic",
            },
        )
        canonical = result["profiles"][0]
        self.assertEqual(canonical["classification"], "canonical")
        self.assertEqual(
            canonical["scenario_notice"],
            "Primary predeclared policy; source evidence is not canonical.",
        )
        self.assertEqual(
            canonical["recommendation"],
            recommend(_benchmarks(), _constraints()),
        )
        for profile in result["profiles"][1:]:
            self.assertEqual(profile["classification"], "derived-non-canonical")
            self.assertIn(
                "does not replace the primary predeclared policy",
                profile["scenario_notice"],
            )
            self.assertEqual(
                profile["recommendation"]["constraints"]["preference_order"],
                [],
            )

    def test_canonical_measured_evidence_uses_submission_decision_notices(self) -> None:
        data = _benchmarks(synthetic=False, classification="canonical")
        result = evaluate_policy_profiles(
            data, _constraints(), PolicySet.from_mapping(_config_mapping())
        )

        self.assertEqual(
            result["profiles"][0]["scenario_notice"],
            "Canonical submission decision.",
        )
        for profile in result["profiles"][1:]:
            self.assertIn(
                "does not replace the canonical decision",
                profile["scenario_notice"],
            )

    def test_policy_set_rejects_unknown_missing_duplicate_and_ambiguous_fields(self) -> None:
        cases: list[tuple[str, dict[str, object], str]] = []

        unknown_top = _config_mapping()
        unknown_top["extra"] = True
        cases.append(("unknown top-level", unknown_top, "unknown extra"))

        unknown_profile = _config_mapping()
        profiles = unknown_profile["profiles"]
        assert isinstance(profiles, list)
        profiles[0]["extra"] = True
        cases.append(("unknown profile", unknown_profile, "unknown extra"))

        duplicate = _config_mapping()
        profiles = duplicate["profiles"]
        assert isinstance(profiles, list)
        profiles[1]["id"] = profiles[0]["id"]
        cases.append(("duplicate id", duplicate, "profile ids must be unique"))

        multiple_canonical = _config_mapping()
        profiles = multiple_canonical["profiles"]
        assert isinstance(profiles, list)
        profiles[1]["classification"] = "canonical"
        profiles[1]["preference_policy"] = "canonical"
        cases.append(("multiple canonical", multiple_canonical, "exactly one canonical"))

        wrong_pointer = _config_mapping()
        wrong_pointer["canonical_profile_id"] = "memory-first"
        cases.append(
            (
                "wrong canonical pointer",
                wrong_pointer,
                "must identify the profile classified as canonical",
            )
        )

        derived_preferences = _config_mapping()
        profiles = derived_preferences["profiles"]
        assert isinstance(profiles, list)
        profiles[1]["preference_policy"] = "canonical"
        cases.append(
            (
                "derived preferences",
                derived_preferences,
                "derived profile must not inherit",
            )
        )

        for label, mapping, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValidationError, message):
                    PolicySet.from_mapping(mapping)

    def test_canonical_profile_must_exactly_match_canonical_constraints(self) -> None:
        raw = _config_mapping()
        profiles = raw["profiles"]
        assert isinstance(profiles, list)
        profiles[0]["objective_tolerance_percent"] = 0.0
        configured = PolicySet.from_mapping(raw)

        with self.assertRaisesRegex(ValidationError, "tolerance does not match"):
            evaluate_policy_profiles(_benchmarks(), _constraints(), configured)

        raw = _config_mapping()
        profiles = raw["profiles"]
        assert isinstance(profiles, list)
        profiles[0]["objective"] = {
            "metric": "ttft_ms_p95",
            "direction": "min",
        }
        configured = PolicySet.from_mapping(raw)
        with self.assertRaisesRegex(ValidationError, "objective does not match"):
            evaluate_policy_profiles(_benchmarks(), _constraints(), configured)

    def test_profile_rejects_missing_metric_and_conflicting_direction(self) -> None:
        missing_raw = {
            "schema_version": "1.0",
            "baseline_id": "q8-generic",
            "synthetic": True,
            "candidates": [
                {
                    "id": candidate.candidate_id,
                    "metrics": {
                        key: value
                        for key, value in candidate.metrics.items()
                        if not (candidate.candidate_id == "q4-generic" and key == "prompt_tps")
                    },
                }
                for candidate in _benchmarks().candidates
            ],
        }
        with self.assertRaisesRegex(
            ValidationError,
            r"prompt_tps.*missing from candidates: q4-generic",
        ):
            evaluate_policy_profiles(
                BenchmarkSet.from_mapping(missing_raw),
                _constraints(),
                PolicySet.from_mapping(_config_mapping()),
            )

        conflict = _config_mapping()
        profiles = conflict["profiles"]
        assert isinstance(profiles, list)
        profiles[4]["objective"]["direction"] = "min"
        configured = PolicySet.from_mapping(conflict)
        with self.assertRaisesRegex(ValidationError, "direction conflicts"):
            evaluate_policy_profiles(_benchmarks(), _constraints(), configured)

    def test_loader_reuses_strict_json_rules(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policies.json"
            path.write_text(
                '{"schema_version":"1.0","schema_version":"1.0",'
                '"canonical_profile_id":"canonical-latency","profiles":[]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValidationError, "duplicate JSON object key"):
                load_policy_set(path)

            path.write_text(
                '{"schema_version":"1.0","canonical_profile_id":"canonical-latency",'
                '"profiles":[{"id":"canonical-latency","label":"x","description":"x",'
                '"classification":"canonical","objective":{"metric":"latency",'
                '"direction":"min"},"objective_tolerance_percent":NaN,'
                '"preference_policy":"canonical"}]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValidationError, "non-standard JSON constant"):
                load_policy_set(path)

    def test_profiles_cli_writes_fingerprinted_output_and_refuses_overwrite(self) -> None:
        repository = Path(__file__).parents[1]
        policies = repository / "configs" / "policies.arm64.json"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_path = root / "benchmarks.json"
            constraints_path = root / "constraints.json"
            output_path = root / "profiles.json"
            data = _benchmarks()
            policy = _constraints()
            write_json(
                benchmark_path,
                {
                    "schema_version": data.schema_version,
                    "baseline_id": data.baseline_id,
                    "synthetic": data.synthetic,
                    "metadata": dict(data.metadata),
                    "candidates": [
                        {
                            "id": candidate.candidate_id,
                            "label": candidate.label,
                            "parameters": dict(candidate.parameters),
                            "metrics": dict(candidate.metrics),
                        }
                        for candidate in data.candidates
                    ],
                },
            )
            write_json(
                constraints_path,
                {
                    "min_quality_retention": policy.min_quality_retention,
                    "quality_metric": policy.quality_metric,
                    "max_values": dict(policy.max_values),
                    "min_values": dict(policy.min_values),
                    "objective": {
                        "metric": policy.objective.metric,
                        "direction": policy.objective.direction,
                    },
                    "frontier_metrics": dict(policy.frontier_metrics),
                    "objective_tolerance_percent": policy.objective_tolerance_percent,
                    "preference_order": list(policy.preference_order),
                },
            )

            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                exit_code = cli.main(
                    [
                        "profiles",
                        str(benchmark_path),
                        "--constraints",
                        str(constraints_path),
                        "--policies",
                        str(policies),
                        "--output",
                        str(output_path),
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, json.loads(output_path.read_text(encoding="utf-8")))
            self.assertEqual(
                set(payload["input_fingerprints"]),
                {"benchmarks_sha256", "constraints_sha256", "policies_sha256"},
            )

            stderr = io.StringIO()
            with patch("sys.stdout", io.StringIO()), patch("sys.stderr", stderr):
                overwrite_exit = cli.main(
                    [
                        "profiles",
                        str(benchmark_path),
                        "--constraints",
                        str(constraints_path),
                        "--policies",
                        str(policies),
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertEqual(overwrite_exit, 2)
            self.assertIn("refusing to overwrite", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
