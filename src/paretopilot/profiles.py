"""Validated deployment-policy profiles derived from one benchmark set.

Policy profiles answer different deployment questions without changing the
measured evidence.  Exactly one profile is canonical; every other profile is
explicitly labeled as a derived, non-canonical scenario.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from paretopilot.analysis import recommend
from paretopilot.domain import BenchmarkSet, Constraints, Objective, ValidationError
from paretopilot.io import load_json_object


_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_CLASSIFICATIONS = {"canonical", "derived-non-canonical"}
_PREFERENCE_POLICIES = {"canonical", "none"}
_DERIVED_NOTICE = "Derived, non-canonical scenario; it does not replace the canonical decision."


def _exact_fields(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(raw)
    if actual == expected:
        return
    details: list[str] = []
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        details.append("missing " + ", ".join(missing))
    if unknown:
        details.append("unknown " + ", ".join(unknown))
    raise ValidationError(f"{context} has an invalid schema: {'; '.join(details)}")


def _non_empty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{context} must be a non-empty string")
    return value


def _percentage(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{context} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 100.0:
        raise ValidationError(f"{context} must be finite and between 0 and 100")
    return converted


@dataclass(frozen=True, slots=True)
class PolicyProfile:
    """One validated objective applied to a shared benchmark set."""

    profile_id: str
    label: str
    description: str
    classification: str
    objective: Objective
    objective_tolerance_percent: float
    preference_policy: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, index: int) -> "PolicyProfile":
        context = f"profiles[{index}]"
        _exact_fields(
            raw,
            {
                "id",
                "label",
                "description",
                "classification",
                "objective",
                "objective_tolerance_percent",
                "preference_policy",
            },
            context,
        )
        profile_id = _non_empty_string(raw.get("id"), f"{context}.id")
        if _PROFILE_ID.fullmatch(profile_id) is None:
            raise ValidationError(
                f"{context}.id must use lowercase letters, digits, and single hyphens"
            )
        classification = _non_empty_string(raw.get("classification"), f"{context}.classification")
        if classification not in _CLASSIFICATIONS:
            raise ValidationError(
                f"{context}.classification must be 'canonical' or 'derived-non-canonical'"
            )
        preference_policy = _non_empty_string(
            raw.get("preference_policy"), f"{context}.preference_policy"
        )
        if preference_policy not in _PREFERENCE_POLICIES:
            raise ValidationError(f"{context}.preference_policy must be 'canonical' or 'none'")
        if classification == "canonical" and preference_policy != "canonical":
            raise ValidationError(f"{context} canonical profile must use canonical preferences")
        if classification == "derived-non-canonical" and preference_policy != "none":
            raise ValidationError(
                f"{context} derived profile must not inherit canonical preferences"
            )

        raw_objective = raw.get("objective")
        if not isinstance(raw_objective, Mapping):
            raise ValidationError(f"{context}.objective must be an object")
        try:
            objective = Objective.from_mapping(raw_objective)
        except ValidationError as exc:
            raise ValidationError(f"invalid {context}.objective: {exc}") from exc

        return cls(
            profile_id=profile_id,
            label=_non_empty_string(raw.get("label"), f"{context}.label"),
            description=_non_empty_string(raw.get("description"), f"{context}.description"),
            classification=classification,
            objective=objective,
            objective_tolerance_percent=_percentage(
                raw.get("objective_tolerance_percent"),
                f"{context}.objective_tolerance_percent",
            ),
            preference_policy=preference_policy,
        )


@dataclass(frozen=True, slots=True)
class PolicySet:
    """Closed policy-profile configuration."""

    schema_version: str
    canonical_profile_id: str
    profiles: tuple[PolicyProfile, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PolicySet":
        _exact_fields(
            raw,
            {"schema_version", "canonical_profile_id", "profiles"},
            "policy set",
        )
        if raw.get("schema_version") != "1.0":
            raise ValidationError("policy set schema_version must currently be '1.0'")
        canonical_profile_id = _non_empty_string(
            raw.get("canonical_profile_id"), "canonical_profile_id"
        )
        if _PROFILE_ID.fullmatch(canonical_profile_id) is None:
            raise ValidationError("canonical_profile_id is not a valid profile id")

        raw_profiles = raw.get("profiles")
        if not isinstance(raw_profiles, Sequence) or isinstance(raw_profiles, (str, bytes)):
            raise ValidationError("profiles must be an array")
        if not 1 <= len(raw_profiles) <= 32:
            raise ValidationError("profiles must contain between 1 and 32 entries")

        profiles: list[PolicyProfile] = []
        for index, value in enumerate(raw_profiles):
            if not isinstance(value, Mapping):
                raise ValidationError(f"profiles[{index}] must be an object")
            profiles.append(PolicyProfile.from_mapping(value, index=index))

        ids = [profile.profile_id for profile in profiles]
        if len(ids) != len(set(ids)):
            raise ValidationError("profile ids must be unique")
        canonical = [profile for profile in profiles if profile.classification == "canonical"]
        if len(canonical) != 1:
            raise ValidationError("policy set must contain exactly one canonical profile")
        if canonical[0].profile_id != canonical_profile_id:
            raise ValidationError(
                "canonical_profile_id must identify the profile classified as canonical"
            )

        return cls(
            schema_version="1.0",
            canonical_profile_id=canonical_profile_id,
            profiles=tuple(profiles),
        )


def load_policy_set(path: Path) -> PolicySet:
    """Load a policy set using ParetoPilot's strict JSON parser."""

    try:
        return PolicySet.from_mapping(load_json_object(path))
    except ValidationError:
        raise
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValidationError(f"invalid policy set in {path}: {exc}") from exc


def _constraints_for_profile(
    base: Constraints,
    profile: PolicyProfile,
) -> Constraints:
    if profile.classification == "canonical":
        if profile.objective != base.objective:
            raise ValidationError(
                f"canonical profile {profile.profile_id!r} objective does not match "
                "the canonical constraints"
            )
        if not math.isclose(
            profile.objective_tolerance_percent,
            base.objective_tolerance_percent,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValidationError(
                f"canonical profile {profile.profile_id!r} tolerance does not match "
                "the canonical constraints"
            )
        return base

    frontier_metrics = dict(base.frontier_metrics)
    existing_direction = frontier_metrics.get(profile.objective.metric)
    if existing_direction is not None and existing_direction != profile.objective.direction:
        raise ValidationError(
            f"profile {profile.profile_id!r} direction conflicts with canonical "
            f"frontier metric {profile.objective.metric!r}"
        )
    frontier_metrics[profile.objective.metric] = profile.objective.direction
    return Constraints(
        min_quality_retention=base.min_quality_retention,
        quality_metric=base.quality_metric,
        max_values=dict(base.max_values),
        min_values=dict(base.min_values),
        objective=profile.objective,
        frontier_metrics=frontier_metrics,
        objective_tolerance_percent=profile.objective_tolerance_percent,
        preference_order=(),
    )


def evaluate_policy_profiles(
    benchmarks: BenchmarkSet,
    canonical_constraints: Constraints,
    policy_set: PolicySet,
) -> Mapping[str, Any]:
    """Precompute deterministic recommendations for every deployment profile."""

    results: list[dict[str, Any]] = []
    candidate_ids = [candidate.candidate_id for candidate in benchmarks.candidates]
    for profile in policy_set.profiles:
        missing = [
            candidate_id
            for candidate_id in candidate_ids
            if profile.objective.metric not in benchmarks.by_id(candidate_id).metrics
        ]
        if missing:
            raise ValidationError(
                f"profile {profile.profile_id!r} objective metric "
                f"{profile.objective.metric!r} is missing from candidates: " + ", ".join(missing)
            )
        constraints = _constraints_for_profile(canonical_constraints, profile)
        decision = dict(recommend(benchmarks, constraints))
        results.append(
            {
                "id": profile.profile_id,
                "label": profile.label,
                "description": profile.description,
                "classification": profile.classification,
                "scenario_notice": (
                    "Canonical submission decision."
                    if profile.classification == "canonical"
                    else _DERIVED_NOTICE
                ),
                "recommendation": decision,
            }
        )

    canonical_result = next(
        result for result in results if result["id"] == policy_set.canonical_profile_id
    )
    return {
        "schema_version": "1.0",
        "source_schema_version": benchmarks.schema_version,
        "synthetic_source": benchmarks.synthetic,
        "canonical_profile_id": policy_set.canonical_profile_id,
        "canonical_selected_id": canonical_result["recommendation"]["selected_id"],
        "profiles": results,
    }
