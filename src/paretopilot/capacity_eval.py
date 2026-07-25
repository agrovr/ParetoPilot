"""Strict supplementary Arm64 server-capacity evidence.

The frozen v1.1 load contract intentionally requires ``--parallel`` to match a
candidate's canonical deployment command.  This module keeps that contract
unchanged and assembles a separate two-dimensional study from already
validated load artifacts.

The supplementary study varies only server slots, total context required to
hold per-slot context constant, and the local binding port. It preserves two
counterbalanced passes, exact commands, request-level load artifacts, quality
gates, process RSS, and an observed KleidiAI model-buffer marker without
changing the canonical recommendation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from paretopilot.domain import ValidationError
from paretopilot.io import (
    load_json_object,
    load_json_object_snapshot,
    sha256_file,
    sha256_json,
)
from paretopilot.load_eval import (
    load_plan_evaluation_contract,
    load_load_plan,
    validate_load_artifact_against_plan,
    validate_load_evaluation,
)
from paretopilot.server_eval import parse_gnu_time_peak_rss, validate_server_evaluation


_SHA256_LENGTH = 64
_ALLOWED_LEVELS = (1, 2, 4)
_ALLOWED_COMMAND_DIFFERENCES = ("--ctx-size", "--parallel", "--port")
_MODEL_BUFFER_MARKER = b"CPU_KLEIDIAI model buffer"
_MAX_TEXT = 4096


@dataclass(frozen=True)
class CapacityCandidate:
    """One predeclared deployment candidate."""

    id: str
    label: str
    role: str
    kleidiai_expected: bool
    canonical_server_command_sha256: str
    canonical_server_argv_sha256: str


@dataclass(frozen=True)
class CapacityPass:
    """One counterbalanced matrix pass."""

    id: str
    candidate_order: tuple[str, ...]
    server_parallel_order: tuple[int, ...]
    client_concurrency_order: tuple[int, ...]


@dataclass(frozen=True)
class CapacityPlan:
    """Validated capacity-study plan."""

    classification: str
    scope: str
    candidates: tuple[CapacityCandidate, ...]
    server_parallel_levels: tuple[int, ...]
    client_concurrency_levels: tuple[int, ...]
    per_slot_context_tokens: int
    passes: tuple[CapacityPass, ...]
    minimum_quality_score: float
    minimum_quality_retention: float
    reference_candidate_id: str
    require_identical_quality_outcomes: bool
    max_server_peak_rss_mib: float
    require_every_pass_slo: bool
    max_throughput_relative_spread_percent: float
    max_e2e_relative_spread_percent: float
    load_slo: Mapping[str, Any]
    objective: str
    direction: str
    objective_tolerance_percent: float
    tie_breakers: tuple[str, ...]
    pins: Mapping[str, str]


def load_capacity_plan(path: Path) -> CapacityPlan:
    """Load one strict, bounded supplementary capacity plan."""

    return _capacity_plan_from_mapping(load_json_object(path), "capacity plan")


def capacity_plan_mapping(plan: CapacityPlan) -> Mapping[str, Any]:
    """Return the canonical JSON representation embedded in evidence."""

    return {
        "schema_version": "1.1",
        "classification": plan.classification,
        "scope": plan.scope,
        "candidates": [
            {
                "id": candidate.id,
                "label": candidate.label,
                "role": candidate.role,
                "kleidiai_expected": candidate.kleidiai_expected,
                "canonical_server_command_sha256": (candidate.canonical_server_command_sha256),
                "canonical_server_argv_sha256": candidate.canonical_server_argv_sha256,
            }
            for candidate in plan.candidates
        ],
        "server_parallel_levels": list(plan.server_parallel_levels),
        "client_concurrency_levels": list(plan.client_concurrency_levels),
        "per_slot_context_tokens": plan.per_slot_context_tokens,
        "passes": [
            {
                "id": pass_spec.id,
                "candidate_order": list(pass_spec.candidate_order),
                "server_parallel_order": list(pass_spec.server_parallel_order),
                "client_concurrency_order": list(pass_spec.client_concurrency_order),
            }
            for pass_spec in plan.passes
        ],
        "quality_gate": {
            "minimum_score": plan.minimum_quality_score,
            "minimum_retention_vs_reference": plan.minimum_quality_retention,
            "reference_candidate_id": plan.reference_candidate_id,
            "require_identical_outcomes_across_parallel": (plan.require_identical_quality_outcomes),
        },
        "capacity_gate": {
            "max_server_peak_rss_mib": plan.max_server_peak_rss_mib,
            "require_every_pass_slo": plan.require_every_pass_slo,
            "max_throughput_relative_spread_percent": (plan.max_throughput_relative_spread_percent),
            "max_e2e_relative_spread_percent": plan.max_e2e_relative_spread_percent,
        },
        "load_slo": dict(plan.load_slo),
        "selection": {
            "objective": plan.objective,
            "direction": plan.direction,
            "objective_tolerance_percent": plan.objective_tolerance_percent,
            "tie_breakers": list(plan.tie_breakers),
        },
        "pins": dict(plan.pins),
    }


def bind_capacity_quality(
    *,
    evaluation_path: Path,
    server_command_path: Path,
    base_url: str,
    pass_id: str,
    candidate_id: str,
    server_parallel: int,
    run_id: str,
    run_attempt: int,
) -> Mapping[str, Any]:
    """Bind one strict quality evaluation to its exact capacity server run."""

    evaluation_path = Path(evaluation_path)
    server_command_path = Path(server_command_path)
    if _resolved_path_key(evaluation_path) == _resolved_path_key(server_command_path):
        raise ValidationError("capacity quality evaluation and server command paths must differ")

    evaluation, evaluation_sha256 = load_json_object_snapshot(evaluation_path)
    validate_server_evaluation(evaluation)
    normalized_candidate_id = _text(candidate_id, "capacity quality candidate_id", maximum=128)
    if evaluation.get("candidate_id") != normalized_candidate_id:
        raise ValidationError("capacity quality evaluation candidate_id does not match its binding")

    command, command_sha256 = load_json_object_snapshot(server_command_path)
    command_argv = _server_command_document(command, "capacity quality server command")
    normalized_parallel = _integer(
        server_parallel,
        "capacity quality server_parallel",
        minimum=1,
        maximum=1024,
    )
    if (
        _option_integer(command_argv, "--parallel", "capacity quality server argv")
        != normalized_parallel
    ):
        raise ValidationError("capacity quality server --parallel does not match its binding")
    normalized_base_url = _normalize_base_url(base_url, "capacity quality request_base_url")
    _validate_request_endpoint(
        normalized_base_url,
        command_argv,
        "capacity quality binding",
    )
    normalized_pass_id = _text(pass_id, "capacity quality pass_id", maximum=64)
    if normalized_pass_id != "quality":
        raise ValidationError("capacity quality pass_id must be 'quality'")

    wrapper: Mapping[str, Any] = {
        "schema_version": "1.0",
        "classification": "capacity-quality-binding",
        "pass_id": normalized_pass_id,
        "candidate_id": normalized_candidate_id,
        "server_parallel": normalized_parallel,
        "run_id": _text(run_id, "capacity quality run_id", maximum=128),
        "run_attempt": _integer(
            run_attempt,
            "capacity quality run_attempt",
            minimum=1,
            maximum=1_000_000,
        ),
        "request_base_url": normalized_base_url,
        "evaluation_sha256": evaluation_sha256,
        "evaluation_content_sha256": sha256_json(evaluation),
        "server_command_sha256": command_sha256,
        "server_command_content_sha256": sha256_json(command),
        "server_argv_sha256": _argv_sha256(command_argv),
        "server_argv": list(command_argv),
        "evaluation": deepcopy(dict(evaluation)),
    }
    _validate_quality_wrapper(wrapper, "capacity quality binding")
    return wrapper


def assemble_capacity_study(
    *,
    plan_path: Path,
    load_plan_path: Path,
    manifest_path: Path,
    load_artifacts: Sequence[tuple[str, Path]],
    rss_artifacts: Sequence[tuple[str, Path]],
    server_logs: Sequence[tuple[str, Path]],
    quality_artifacts: Sequence[tuple[str, Path]],
) -> Mapping[str, Any]:
    """Assemble a deterministic capacity study from measured source artifacts."""

    plan = load_capacity_plan(plan_path)
    load_plan = load_load_plan(load_plan_path)
    if sha256_file(load_plan_path) != plan.pins["load_plan_sha256"]:
        raise ValidationError("load plan SHA-256 does not match the capacity plan pin")
    if dict(load_plan.slo) != dict(plan.load_slo):
        raise ValidationError("load plan SLO does not match the predeclared capacity SLO")
    load_contract = load_plan_evaluation_contract(load_plan)

    manifest = load_json_object(manifest_path)
    provenance = _validate_manifest(manifest, plan)
    expected_run_labels = _expected_run_labels(plan)
    expected_quality_labels = _expected_quality_labels(plan)

    load_paths = _labeled_paths(load_artifacts, expected_run_labels, "load artifacts")
    rss_paths = _labeled_paths(rss_artifacts, expected_run_labels, "RSS artifacts")
    log_paths = _labeled_paths(server_logs, expected_run_labels, "server logs")
    quality_paths = _labeled_paths(
        quality_artifacts,
        expected_quality_labels,
        "quality artifacts",
    )
    _reject_duplicate_resolved_input_paths(
        {
            "capacity plan": Path(plan_path),
            "load plan": Path(load_plan_path),
            "manifest": Path(manifest_path),
            **{f"load:{label}": path for label, path in load_paths.items()},
            **{f"rss:{label}": path for label, path in rss_paths.items()},
            **{f"log:{label}": path for label, path in log_paths.items()},
            **{f"quality:{label}": path for label, path in quality_paths.items()},
        }
    )

    load_fingerprints: dict[str, str] = {}
    rss_fingerprints: dict[str, str] = {}
    log_fingerprints: dict[str, str] = {}
    quality_fingerprints: dict[str, str] = {}
    source_loads: dict[str, Mapping[str, Any]] = {}
    source_quality: dict[str, Mapping[str, Any]] = {}
    server_configurations: list[Mapping[str, Any]] = []
    load_rows: dict[tuple[str, str, int, int], Mapping[str, Any]] = {}
    load_identities: set[tuple[str, str, str]] = set()

    candidates = {candidate.id: candidate for candidate in plan.candidates}
    canonical_commands = provenance["canonical_commands"]
    passes = {pass_spec.id: pass_spec for pass_spec in plan.passes}

    for label in _expected_run_sequence(plan):
        pass_id, candidate_id, server_parallel = _parse_run_label(label)
        candidate = candidates[candidate_id]
        load_path = load_paths[label]
        raw, load_sha256 = load_json_object_snapshot(load_path)
        validate_load_evaluation(raw, require_evidence_binding=True)
        validate_load_artifact_against_plan(raw, load_plan, context=f"load artifact {label}")
        if raw.get("synthetic") is not False:
            raise ValidationError(f"load artifact {label} must be measured")
        if raw.get("candidate_id") != candidate_id:
            raise ValidationError(f"load artifact {label} candidate_id does not match its label")
        if raw.get("slo") != plan.load_slo:
            raise ValidationError(f"load artifact {label} SLO does not match the capacity plan")
        execution_order = _load_execution_order(raw, f"load artifact {label}")
        if execution_order != passes[pass_id].client_concurrency_order:
            raise ValidationError(
                f"load artifact {label} execution_order does not match its capacity pass"
            )

        binding = _mapping(raw.get("evidence_binding"), f"load artifact {label}.evidence_binding")
        if binding.get("plan_sha256") != plan.pins["load_plan_sha256"]:
            raise ValidationError(f"load artifact {label} plan SHA-256 does not match")
        server_configuration = _mapping(
            binding.get("server_configuration"),
            f"load artifact {label}.server_configuration",
        )
        configuration = _capacity_server_configuration(
            pass_id=pass_id,
            candidate=candidate,
            server_parallel=server_parallel,
            request_base_url=_text(
                binding.get("request_base_url"),
                f"load artifact {label}.request_base_url",
            ),
            server_configuration=server_configuration,
            canonical_command=_mapping(
                canonical_commands[candidate_id],
                f"manifest canonical command {candidate_id}",
            ),
            per_slot_context_tokens=plan.per_slot_context_tokens,
            rss_path=rss_paths[label],
            log_path=log_paths[label],
            source_label=label,
            load_evaluation_sha256=load_sha256,
            load_evaluation_content_sha256=sha256_json(raw),
        )
        server_configurations.append(configuration)

        load_identity = (
            candidate_id,
            str(binding.get("request_base_url")),
            str(configuration["load_command_sha256"]),
        )
        if load_identity in load_identities:
            raise ValidationError(f"duplicate capacity load identity for {label!r}")
        load_identities.add(load_identity)

        load_fingerprints[label] = load_sha256
        rss_fingerprints[label] = sha256_file(rss_paths[label])
        log_fingerprints[label] = sha256_file(log_paths[label])
        source_loads[label] = {
            "input_sha256": load_sha256,
            "content_sha256": sha256_json(raw),
            "evaluation": deepcopy(dict(raw)),
        }
        for row_value in raw["rows"]:
            row = _mapping(row_value, f"load artifact {label}.row")
            client_concurrency = _integer(
                row.get("concurrency"),
                f"load artifact {label}.row.concurrency",
                minimum=1,
                maximum=1024,
            )
            if client_concurrency not in plan.client_concurrency_levels:
                raise ValidationError(
                    f"load artifact {label} contains unplanned client concurrency"
                )
            key = (pass_id, candidate_id, server_parallel, client_concurrency)
            if key in load_rows:
                raise ValidationError(f"duplicate capacity load row for {key!r}")
            load_rows[key] = row

    configuration_by_key = {
        (
            str(configuration["pass_id"]),
            str(configuration["candidate_id"]),
            int(configuration["server_parallel"]),
        ): configuration
        for configuration in server_configurations
    }
    quality_checks = _quality_checks(
        plan=plan,
        paths=quality_paths,
        fingerprints=quality_fingerprints,
        sources=source_quality,
        provenance=provenance,
        canonical_commands=canonical_commands,
        configuration_by_key=configuration_by_key,
    )
    quality_by_key = {
        (str(check["candidate_id"]), int(check["server_parallel"])): check
        for check in quality_checks
    }

    cells: list[Mapping[str, Any]] = []
    for candidate in plan.candidates:
        for server_parallel in plan.server_parallel_levels:
            quality = quality_by_key[(candidate.id, server_parallel)]
            for client_concurrency in plan.client_concurrency_levels:
                pass_metrics: list[Mapping[str, Any]] = []
                for pass_spec in plan.passes:
                    key = (
                        pass_spec.id,
                        candidate.id,
                        server_parallel,
                        client_concurrency,
                    )
                    row = load_rows[key]
                    configuration = configuration_by_key[
                        (pass_spec.id, candidate.id, server_parallel)
                    ]
                    pass_metrics.append(
                        _pass_metrics(
                            pass_id=pass_spec.id,
                            row=row,
                            server_peak_rss_mib=float(configuration["server_peak_rss_mib"]),
                            source_label=str(configuration["source_load_label"]),
                            source_sha256=str(configuration["load_evaluation_sha256"]),
                        )
                    )
                cells.append(
                    _capacity_cell(
                        candidate_id=candidate.id,
                        server_parallel=server_parallel,
                        client_concurrency=client_concurrency,
                        pass_metrics=pass_metrics,
                        quality_gate_met=bool(quality["gate_met"]),
                        plan=plan,
                    )
                )

    selections = _capacity_selections(plan, cells)
    artifact: Mapping[str, Any] = {
        "schema_version": "1.1",
        "classification": "supplementary-capacity",
        "synthetic": False,
        "plan": capacity_plan_mapping(plan),
        "provenance": provenance,
        "load_contract": {
            "load_plan_sha256": plan.pins["load_plan_sha256"],
            "methodology": deepcopy(dict(load_contract["methodology"])),
            "slo": deepcopy(dict(load_contract["slo"])),
        },
        "source_artifacts": {
            "load_evaluations": source_loads,
            "quality_wrappers": source_quality,
        },
        "quality_checks": quality_checks,
        "server_configurations": sorted(
            server_configurations,
            key=lambda item: (
                _pass_index(plan, str(item["pass_id"])),
                _candidate_index(plan, str(item["candidate_id"])),
                int(item["server_parallel"]),
            ),
        ),
        "cells": cells,
        "selections": selections,
        "input_fingerprints": {
            "capacity_plan_sha256": sha256_file(plan_path),
            "capacity_plan_content_sha256": sha256_json(capacity_plan_mapping(plan)),
            "load_plan_sha256": sha256_file(load_plan_path),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_content_sha256": sha256_json(provenance),
            "load_artifacts": dict(sorted(load_fingerprints.items())),
            "rss_artifacts": dict(sorted(rss_fingerprints.items())),
            "server_logs": dict(sorted(log_fingerprints.items())),
            "quality_artifacts": dict(sorted(quality_fingerprints.items())),
        },
        "canonical_outputs_modified": False,
        "boundary_caveat": (
            "This is a bounded fixed-concurrency study on one native Arm64 runner. "
            "It is not an open-loop production-capacity benchmark, a cost or energy "
            "claim, or a replacement for the frozen canonical v1.1 recommendation. "
            "The KleidiAI log marker establishes only that llama.cpp reported a "
            "CPU_KLEIDIAI model buffer; it is not kernel-level acceleration proof."
        ),
    }
    validate_capacity_study(artifact)
    return artifact


def validate_capacity_study(raw: Mapping[str, Any]) -> None:
    """Validate and recompute a serialized capacity-study artifact."""

    study = _mapping(raw, "capacity study")
    _exact_fields(
        study,
        {
            "schema_version",
            "classification",
            "synthetic",
            "plan",
            "provenance",
            "load_contract",
            "source_artifacts",
            "quality_checks",
            "server_configurations",
            "cells",
            "selections",
            "input_fingerprints",
            "canonical_outputs_modified",
            "boundary_caveat",
        },
        "capacity study",
    )
    if study.get("schema_version") != "1.1":
        raise ValidationError("capacity study schema_version must be '1.1'")
    if study.get("classification") != "supplementary-capacity":
        raise ValidationError("capacity study classification must be supplementary-capacity")
    if study.get("synthetic") is not False:
        raise ValidationError("capacity study synthetic must be false")
    if study.get("canonical_outputs_modified") is not False:
        raise ValidationError("capacity study must not modify canonical outputs")
    _text(study.get("boundary_caveat"), "capacity study boundary_caveat")

    plan = _capacity_plan_from_mapping(
        _mapping(study.get("plan"), "capacity study plan"),
        "capacity study plan",
    )
    provenance = _validate_manifest(
        _mapping(study.get("provenance"), "capacity study provenance"),
        plan,
    )
    load_contract = _mapping(
        study.get("load_contract"),
        "capacity study load_contract",
    )
    _exact_fields(
        load_contract,
        {"load_plan_sha256", "methodology", "slo"},
        "capacity study load_contract",
    )
    if (
        _digest(
            load_contract.get("load_plan_sha256"),
            "capacity study load_contract.load_plan_sha256",
        )
        != plan.pins["load_plan_sha256"]
    ):
        raise ValidationError("capacity study load contract plan hash does not match")
    load_slo = _validate_slo(
        load_contract.get("slo"),
        "capacity study load_contract.slo",
    )
    if load_slo != plan.load_slo:
        raise ValidationError("capacity study load contract SLO does not match its plan")
    fingerprints = _validate_fingerprints(
        study.get("input_fingerprints"),
        plan,
        provenance=provenance,
    )
    load_sources, quality_sources = _validate_source_artifacts(
        study.get("source_artifacts"),
        plan=plan,
        provenance=provenance,
        fingerprints=fingerprints,
        load_contract=load_contract,
    )
    configurations = _validate_serialized_server_configurations(
        study.get("server_configurations"),
        plan,
        load_sources=load_sources,
        fingerprints=fingerprints,
    )
    configuration_by_key = {
        (
            str(configuration["pass_id"]),
            str(configuration["candidate_id"]),
            int(configuration["server_parallel"]),
        ): configuration
        for configuration in configurations
    }
    _crosslink_quality_commands(
        plan=plan,
        wrappers=quality_sources,
        configuration_by_key=configuration_by_key,
    )
    quality_checks = _derive_quality_checks(
        plan=plan,
        wrappers=quality_sources,
        fingerprints=_mapping(
            fingerprints["quality_artifacts"],
            "capacity study quality fingerprints",
        ),
    )
    if study.get("quality_checks") != quality_checks:
        raise ValidationError("capacity study quality_checks do not match embedded quality sources")
    quality_by_key = {
        (str(check["candidate_id"]), int(check["server_parallel"])): check
        for check in quality_checks
    }
    cells = _cells_from_sources(
        plan=plan,
        load_sources=load_sources,
        quality_by_key=quality_by_key,
        configuration_by_key=configuration_by_key,
    )
    if study.get("cells") != cells:
        raise ValidationError("capacity study cells do not match embedded load sources")
    expected_selections = _capacity_selections(plan, cells)
    if study.get("selections") != expected_selections:
        raise ValidationError("capacity study selections do not match recomputed cells")


def _capacity_plan_from_mapping(raw_value: Any, context: str) -> CapacityPlan:
    raw = _mapping(raw_value, context)
    _exact_fields(
        raw,
        {
            "schema_version",
            "classification",
            "scope",
            "candidates",
            "server_parallel_levels",
            "client_concurrency_levels",
            "per_slot_context_tokens",
            "passes",
            "quality_gate",
            "capacity_gate",
            "load_slo",
            "selection",
            "pins",
        },
        context,
    )
    if raw.get("schema_version") != "1.1":
        raise ValidationError(f"{context} schema_version must be '1.1'")
    if raw.get("classification") != "supplementary-capacity":
        raise ValidationError(f"{context} classification must be supplementary-capacity")
    scope = _text(raw.get("scope"), f"{context}.scope")

    candidate_values = raw.get("candidates")
    if not isinstance(candidate_values, list) or len(candidate_values) != 2:
        raise ValidationError(f"{context}.candidates must contain exactly two candidates")
    candidates: list[CapacityCandidate] = []
    candidate_ids: set[str] = set()
    roles: set[str] = set()
    for index, value in enumerate(candidate_values):
        candidate_context = f"{context}.candidates[{index}]"
        candidate = _mapping(value, candidate_context)
        _exact_fields(
            candidate,
            {
                "id",
                "label",
                "role",
                "kleidiai_expected",
                "canonical_server_command_sha256",
                "canonical_server_argv_sha256",
            },
            candidate_context,
        )
        candidate_id = _text(candidate.get("id"), f"{candidate_context}.id", maximum=128)
        label = _text(candidate.get("label"), f"{candidate_context}.label", maximum=256)
        role = _text(candidate.get("role"), f"{candidate_context}.role", maximum=64)
        if role not in {"canonical-reference", "resource-alternative"}:
            raise ValidationError(f"{candidate_context}.role is not supported")
        if candidate_id in candidate_ids or role in roles:
            raise ValidationError(f"{context}.candidates must have unique ids and roles")
        candidate_ids.add(candidate_id)
        roles.add(role)
        kleidiai_expected = candidate.get("kleidiai_expected")
        if not isinstance(kleidiai_expected, bool):
            raise ValidationError(f"{candidate_context}.kleidiai_expected must be boolean")
        candidates.append(
            CapacityCandidate(
                id=candidate_id,
                label=label,
                role=role,
                kleidiai_expected=kleidiai_expected,
                canonical_server_command_sha256=_digest(
                    candidate.get("canonical_server_command_sha256"),
                    f"{candidate_context}.canonical_server_command_sha256",
                ),
                canonical_server_argv_sha256=_digest(
                    candidate.get("canonical_server_argv_sha256"),
                    f"{candidate_context}.canonical_server_argv_sha256",
                ),
            )
        )
    if roles != {"canonical-reference", "resource-alternative"}:
        raise ValidationError(f"{context}.candidates must declare both supported roles")

    server_levels = _levels(
        raw.get("server_parallel_levels"),
        f"{context}.server_parallel_levels",
    )
    client_levels = _levels(
        raw.get("client_concurrency_levels"),
        f"{context}.client_concurrency_levels",
    )
    per_slot_context = _integer(
        raw.get("per_slot_context_tokens"),
        f"{context}.per_slot_context_tokens",
        minimum=128,
        maximum=32768,
    )
    pass_values = raw.get("passes")
    if not isinstance(pass_values, list) or len(pass_values) != 2:
        raise ValidationError(f"{context}.passes must contain exactly two counterbalanced passes")
    passes: list[CapacityPass] = []
    pass_ids: set[str] = set()
    for index, value in enumerate(pass_values):
        pass_context = f"{context}.passes[{index}]"
        pass_raw = _mapping(value, pass_context)
        _exact_fields(
            pass_raw,
            {
                "id",
                "candidate_order",
                "server_parallel_order",
                "client_concurrency_order",
            },
            pass_context,
        )
        pass_id = _text(pass_raw.get("id"), f"{pass_context}.id", maximum=64)
        if pass_id in pass_ids:
            raise ValidationError(f"{context}.passes must have unique ids")
        pass_ids.add(pass_id)
        candidate_order = _text_array(
            pass_raw.get("candidate_order"),
            f"{pass_context}.candidate_order",
        )
        if set(candidate_order) != candidate_ids or len(candidate_order) != len(candidate_ids):
            raise ValidationError(
                f"{pass_context}.candidate_order must be a permutation of candidates"
            )
        server_order = _integer_array(
            pass_raw.get("server_parallel_order"),
            f"{pass_context}.server_parallel_order",
        )
        if set(server_order) != set(server_levels) or len(server_order) != len(server_levels):
            raise ValidationError(
                f"{pass_context}.server_parallel_order must be a permutation of levels"
            )
        client_order = _integer_array(
            pass_raw.get("client_concurrency_order"),
            f"{pass_context}.client_concurrency_order",
        )
        if set(client_order) != set(client_levels) or len(client_order) != len(client_levels):
            raise ValidationError(
                f"{pass_context}.client_concurrency_order must be a permutation of levels"
            )
        passes.append(
            CapacityPass(
                id=pass_id,
                candidate_order=candidate_order,
                server_parallel_order=server_order,
                client_concurrency_order=client_order,
            )
        )
    if (
        passes[1].candidate_order != tuple(reversed(passes[0].candidate_order))
        or passes[1].server_parallel_order != tuple(reversed(passes[0].server_parallel_order))
        or passes[1].client_concurrency_order != tuple(reversed(passes[0].client_concurrency_order))
    ):
        raise ValidationError(f"{context}.passes must use exact counterbalanced orders")

    quality = _mapping(raw.get("quality_gate"), f"{context}.quality_gate")
    _exact_fields(
        quality,
        {
            "minimum_score",
            "minimum_retention_vs_reference",
            "reference_candidate_id",
            "require_identical_outcomes_across_parallel",
        },
        f"{context}.quality_gate",
    )
    minimum_score = _finite(
        quality.get("minimum_score"),
        f"{context}.quality_gate.minimum_score",
        minimum=0.0,
        maximum=1.0,
    )
    minimum_retention = _finite(
        quality.get("minimum_retention_vs_reference"),
        f"{context}.quality_gate.minimum_retention_vs_reference",
        minimum=0.0,
        maximum=1.0,
    )
    reference_candidate_id = _text(
        quality.get("reference_candidate_id"),
        f"{context}.quality_gate.reference_candidate_id",
        maximum=128,
    )
    reference_candidate = next(
        (candidate for candidate in candidates if candidate.id == reference_candidate_id),
        None,
    )
    if reference_candidate is None or reference_candidate.role != "canonical-reference":
        raise ValidationError(
            f"{context}.quality_gate.reference_candidate_id must name the canonical reference"
        )
    require_identical = quality.get("require_identical_outcomes_across_parallel")
    if require_identical is not True:
        raise ValidationError(
            f"{context}.quality_gate must require identical outcomes across parallel levels"
        )

    capacity_gate = _mapping(raw.get("capacity_gate"), f"{context}.capacity_gate")
    _exact_fields(
        capacity_gate,
        {
            "max_server_peak_rss_mib",
            "require_every_pass_slo",
            "max_throughput_relative_spread_percent",
            "max_e2e_relative_spread_percent",
        },
        f"{context}.capacity_gate",
    )
    max_rss = _finite(
        capacity_gate.get("max_server_peak_rss_mib"),
        f"{context}.capacity_gate.max_server_peak_rss_mib",
        strictly_positive=True,
    )
    if capacity_gate.get("require_every_pass_slo") is not True:
        raise ValidationError(f"{context}.capacity_gate must require every pass SLO")
    max_throughput_spread = _finite(
        capacity_gate.get("max_throughput_relative_spread_percent"),
        f"{context}.capacity_gate.max_throughput_relative_spread_percent",
        minimum=0.0,
        maximum=100.0,
    )
    max_e2e_spread = _finite(
        capacity_gate.get("max_e2e_relative_spread_percent"),
        f"{context}.capacity_gate.max_e2e_relative_spread_percent",
        minimum=0.0,
        maximum=100.0,
    )
    load_slo = _validate_slo(raw.get("load_slo"), f"{context}.load_slo")

    selection = _mapping(raw.get("selection"), f"{context}.selection")
    _exact_fields(
        selection,
        {
            "objective",
            "direction",
            "objective_tolerance_percent",
            "tie_breakers",
        },
        f"{context}.selection",
    )
    objective = _text(selection.get("objective"), f"{context}.selection.objective")
    direction = _text(selection.get("direction"), f"{context}.selection.direction")
    objective_tolerance = _finite(
        selection.get("objective_tolerance_percent"),
        f"{context}.selection.objective_tolerance_percent",
        minimum=0.0,
        maximum=100.0,
    )
    tie_breakers = _text_array(
        selection.get("tie_breakers"),
        f"{context}.selection.tie_breakers",
    )
    expected_tie_breakers = (
        "e2e_latency_ms_p95_median",
        "ttft_ms_p95_median",
        "server_peak_rss_mib_max",
        "server_parallel",
        "client_concurrency",
    )
    if (
        objective != "generated_tokens_per_second_median"
        or direction != "max"
        or objective_tolerance != 1.0
        or tie_breakers != expected_tie_breakers
    ):
        raise ValidationError(f"{context}.selection must use the fixed capacity objective")

    pins = _mapping(raw.get("pins"), f"{context}.pins")
    expected_pins = {
        "load_plan_sha256",
        "evaluation_suite_sha256",
        "canonical_evidence_lock_sha256",
        "canonical_release_sha256",
    }
    _exact_fields(pins, expected_pins, f"{context}.pins")
    normalized_pins = {
        name: _digest(pins.get(name), f"{context}.pins.{name}") for name in sorted(expected_pins)
    }

    return CapacityPlan(
        classification="supplementary-capacity",
        scope=scope,
        candidates=tuple(candidates),
        server_parallel_levels=server_levels,
        client_concurrency_levels=client_levels,
        per_slot_context_tokens=per_slot_context,
        passes=tuple(passes),
        minimum_quality_score=minimum_score,
        minimum_quality_retention=minimum_retention,
        reference_candidate_id=reference_candidate_id,
        require_identical_quality_outcomes=True,
        max_server_peak_rss_mib=max_rss,
        require_every_pass_slo=True,
        max_throughput_relative_spread_percent=max_throughput_spread,
        max_e2e_relative_spread_percent=max_e2e_spread,
        load_slo=load_slo,
        objective=objective,
        direction=direction,
        objective_tolerance_percent=objective_tolerance,
        tie_breakers=tie_breakers,
        pins=normalized_pins,
    )


def _validate_manifest(raw_value: Any, plan: CapacityPlan) -> Mapping[str, Any]:
    raw = _mapping(raw_value, "capacity manifest")
    _exact_fields(
        raw,
        {
            "schema_version",
            "classification",
            "synthetic",
            "source",
            "runner",
            "runtime",
            "optimization_library",
            "toolchain",
            "candidates",
            "canonical_commands",
            "canonical_evidence",
        },
        "capacity manifest",
    )
    if raw.get("schema_version") != "1.1":
        raise ValidationError("capacity manifest schema_version must be '1.1'")
    if raw.get("classification") != "supplementary-capacity":
        raise ValidationError("capacity manifest classification must be supplementary-capacity")
    if raw.get("synthetic") is not False:
        raise ValidationError("capacity manifest synthetic must be false")

    source = _mapping(raw.get("source"), "capacity manifest.source")
    _exact_fields(
        source,
        {
            "repository",
            "revision",
            "workflow",
            "run_id",
            "run_attempt",
            "generated_at_utc",
        },
        "capacity manifest.source",
    )
    _text(source.get("repository"), "capacity manifest.source.repository")
    _git_revision(source.get("revision"), "capacity manifest.source.revision")
    workflow = _text(source.get("workflow"), "capacity manifest.source.workflow")
    if workflow != ".github/workflows/capacity-study-arm64.yml":
        raise ValidationError("capacity manifest source workflow is not the capacity workflow")
    _text(source.get("run_id"), "capacity manifest.source.run_id")
    _integer(
        source.get("run_attempt"),
        "capacity manifest.source.run_attempt",
        minimum=1,
        maximum=1_000_000,
    )
    _text(source.get("generated_at_utc"), "capacity manifest.source.generated_at_utc")

    runner = _mapping(raw.get("runner"), "capacity manifest.runner")
    _exact_fields(
        runner,
        {"os", "architecture", "cpu", "cpu_count"},
        "capacity manifest.runner",
    )
    _text(runner.get("os"), "capacity manifest.runner.os")
    architecture = _text(runner.get("architecture"), "capacity manifest.runner.architecture")
    if architecture.casefold() not in {"arm64", "aarch64"}:
        raise ValidationError("capacity manifest runner must be native Arm64")
    _text(runner.get("cpu"), "capacity manifest.runner.cpu")
    _integer(
        runner.get("cpu_count"),
        "capacity manifest.runner.cpu_count",
        minimum=1,
        maximum=4096,
    )

    runtime = _mapping(raw.get("runtime"), "capacity manifest.runtime")
    _exact_fields(
        runtime,
        {"name", "repository", "revision"},
        "capacity manifest.runtime",
    )
    if _text(runtime.get("name"), "capacity manifest.runtime.name") != "llama.cpp":
        raise ValidationError("capacity manifest runtime must be llama.cpp")
    _text(runtime.get("repository"), "capacity manifest.runtime.repository")
    _git_revision(runtime.get("revision"), "capacity manifest.runtime.revision")

    library = _mapping(
        raw.get("optimization_library"),
        "capacity manifest.optimization_library",
    )
    _exact_fields(
        library,
        {
            "name",
            "repository",
            "version",
            "source_archive_sha256",
            "size_bytes",
        },
        "capacity manifest.optimization_library",
    )
    if _text(library.get("name"), "capacity manifest.optimization_library.name") != "KleidiAI":
        raise ValidationError("capacity manifest optimization library must be KleidiAI")
    _text(library.get("repository"), "capacity manifest.optimization_library.repository")
    _text(library.get("version"), "capacity manifest.optimization_library.version")
    _digest(
        library.get("source_archive_sha256"),
        "capacity manifest.optimization_library.source_archive_sha256",
    )
    _integer(
        library.get("size_bytes"),
        "capacity manifest.optimization_library.size_bytes",
        minimum=1,
        maximum=10_000_000_000,
    )

    toolchain = _mapping(raw.get("toolchain"), "capacity manifest.toolchain")
    _exact_fields(
        toolchain,
        {
            "gcc_version_sha256",
            "gxx_version_sha256",
            "cmake_version_sha256",
            "ninja_version_sha256",
        },
        "capacity manifest.toolchain",
    )
    for name in (
        "gcc_version_sha256",
        "gxx_version_sha256",
        "cmake_version_sha256",
        "ninja_version_sha256",
    ):
        _digest(toolchain.get(name), f"capacity manifest.toolchain.{name}")

    candidates = _mapping(raw.get("candidates"), "capacity manifest.candidates")
    expected_ids = {candidate.id for candidate in plan.candidates}
    if set(candidates) != expected_ids:
        raise ValidationError("capacity manifest candidates do not match the plan")
    for candidate in plan.candidates:
        context = f"capacity manifest.candidates[{candidate.id!r}]"
        value = _mapping(candidates[candidate.id], context)
        _exact_fields(value, {"model", "build"}, context)
        model = _mapping(value.get("model"), f"{context}.model")
        _exact_fields(
            model,
            {"family", "repository", "revision", "filename", "sha256", "size_bytes"},
            f"{context}.model",
        )
        _text(model.get("family"), f"{context}.model.family")
        _text(model.get("repository"), f"{context}.model.repository")
        _git_revision(model.get("revision"), f"{context}.model.revision")
        _text(model.get("filename"), f"{context}.model.filename")
        _digest(model.get("sha256"), f"{context}.model.sha256")
        _integer(
            model.get("size_bytes"),
            f"{context}.model.size_bytes",
            minimum=1,
            maximum=1_000_000_000_000,
        )
        build = _mapping(value.get("build"), f"{context}.build")
        _exact_fields(
            build,
            {
                "label",
                "server_binary_sha256",
                "kleidiai_enabled",
                "cmake_cache_sha256",
                "configure_log_sha256",
                "compile_log_sha256",
            },
            f"{context}.build",
        )
        _text(build.get("label"), f"{context}.build.label")
        _digest(build.get("server_binary_sha256"), f"{context}.build.server_binary_sha256")
        for name in (
            "cmake_cache_sha256",
            "configure_log_sha256",
            "compile_log_sha256",
        ):
            _digest(build.get(name), f"{context}.build.{name}")
        if build.get("kleidiai_enabled") is not candidate.kleidiai_expected:
            raise ValidationError(f"{context}.build KleidiAI flag does not match the plan")

    canonical_commands = _mapping(
        raw.get("canonical_commands"),
        "capacity manifest.canonical_commands",
    )
    if set(canonical_commands) != expected_ids:
        raise ValidationError("capacity manifest canonical commands do not match the plan")
    for candidate in plan.candidates:
        context = f"capacity manifest.canonical_commands[{candidate.id!r}]"
        command = _mapping(canonical_commands[candidate.id], context)
        _exact_fields(command, {"sha256", "argv"}, context)
        digest = _digest(command.get("sha256"), f"{context}.sha256")
        if digest != candidate.canonical_server_command_sha256:
            raise ValidationError(f"{context}.sha256 does not match the plan")
        argv = _argv(command.get("argv"), f"{context}.argv")
        expected_command_sha256 = sha256_json({"schema_version": "1.0", "argv": list(argv)})
        if digest != expected_command_sha256:
            raise ValidationError(f"{context}.sha256 does not match the canonical JSON command")
        if _argv_sha256(argv) != candidate.canonical_server_argv_sha256:
            raise ValidationError(f"{context}.argv does not match the plan argv hash")

    evidence = _mapping(
        raw.get("canonical_evidence"),
        "capacity manifest.canonical_evidence",
    )
    _exact_fields(
        evidence,
        {"lock_sha256", "release_sha256", "run_id", "release_tag"},
        "capacity manifest.canonical_evidence",
    )
    if (
        _digest(
            evidence.get("lock_sha256"),
            "capacity manifest.canonical_evidence.lock_sha256",
        )
        != plan.pins["canonical_evidence_lock_sha256"]
    ):
        raise ValidationError("capacity manifest canonical evidence lock does not match the plan")
    if (
        _digest(
            evidence.get("release_sha256"),
            "capacity manifest.canonical_evidence.release_sha256",
        )
        != plan.pins["canonical_release_sha256"]
    ):
        raise ValidationError("capacity manifest canonical release does not match the plan")
    _text(evidence.get("run_id"), "capacity manifest.canonical_evidence.run_id")
    _text(evidence.get("release_tag"), "capacity manifest.canonical_evidence.release_tag")
    return raw


def _capacity_server_configuration(
    *,
    pass_id: str,
    candidate: CapacityCandidate,
    server_parallel: int,
    request_base_url: str,
    server_configuration: Mapping[str, Any],
    canonical_command: Mapping[str, Any],
    per_slot_context_tokens: int,
    rss_path: Path,
    log_path: Path,
    source_label: str,
    load_evaluation_sha256: str,
    load_evaluation_content_sha256: str,
) -> Mapping[str, Any]:
    load_parallel = _integer(
        server_configuration.get("load_parallel"),
        "capacity server load_parallel",
        minimum=1,
        maximum=1024,
    )
    canonical_parallel = _integer(
        server_configuration.get("canonical_parallel"),
        "capacity server canonical_parallel",
        minimum=1,
        maximum=1024,
    )
    if load_parallel != server_parallel or canonical_parallel != server_parallel:
        raise ValidationError("capacity server parallelism does not match its planned label")

    load_argv = _argv(
        server_configuration.get("load_server_argv"),
        "capacity server load argv",
    )
    binding_argv = _argv(
        server_configuration.get("canonical_server_argv"),
        "capacity server binding argv",
    )
    if load_argv != binding_argv:
        raise ValidationError("capacity load command must bind to itself before aggregation")
    if server_configuration.get("load_server_command_sha256") != (
        server_configuration.get("canonical_server_command_sha256")
    ):
        raise ValidationError("capacity load command hashes must match their self-binding")
    load_command_sha256 = _digest(
        server_configuration.get("load_server_command_sha256"),
        "capacity server load command SHA-256",
    )
    load_command_content_sha256 = sha256_json({"schema_version": "1.0", "argv": list(load_argv)})
    canonical_argv = _argv(canonical_command.get("argv"), "canonical deployment argv")
    differences = _command_differences(
        capacity_argv=load_argv,
        canonical_argv=canonical_argv,
        server_parallel=server_parallel,
        per_slot_context_tokens=per_slot_context_tokens,
    )
    _validate_request_endpoint(request_base_url, load_argv, "capacity load artifact")

    peak_rss = parse_gnu_time_peak_rss(rss_path)
    try:
        log_bytes = log_path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"could not read capacity server log {log_path}: {exc}") from exc
    marker_count = log_bytes.count(_MODEL_BUFFER_MARKER)
    if candidate.kleidiai_expected and marker_count < 1:
        raise ValidationError(
            f"{candidate.id} capacity server did not record the KleidiAI model-buffer marker"
        )
    if not candidate.kleidiai_expected and marker_count:
        raise ValidationError(
            f"{candidate.id} capacity server unexpectedly recorded a KleidiAI model-buffer marker"
        )
    total_context = _option_integer(load_argv, "--ctx-size", "capacity server argv")
    return {
        "pass_id": pass_id,
        "candidate_id": candidate.id,
        "server_parallel": server_parallel,
        "source_load_label": source_label,
        "load_evaluation_sha256": _digest(
            load_evaluation_sha256,
            "capacity load evaluation SHA-256",
        ),
        "load_evaluation_content_sha256": _digest(
            load_evaluation_content_sha256,
            "capacity load evaluation content SHA-256",
        ),
        "total_context_tokens": total_context,
        "per_slot_context_tokens": per_slot_context_tokens,
        "request_base_url": request_base_url,
        "load_command_sha256": load_command_sha256,
        "load_command_content_sha256": load_command_content_sha256,
        "load_argv_sha256": _argv_sha256(load_argv),
        "canonical_command_sha256": _digest(
            canonical_command.get("sha256"),
            "capacity server canonical command SHA-256",
        ),
        "canonical_argv_sha256": _argv_sha256(canonical_argv),
        "load_argv": list(load_argv),
        "canonical_argv": list(canonical_argv),
        "differing_options": list(differences),
        "server_peak_rss_mib": peak_rss,
        "rss_sha256": sha256_file(rss_path),
        "server_log_sha256": sha256_file(log_path),
        "kleidiai_model_buffer_marker_count": marker_count,
    }


def _command_differences(
    *,
    capacity_argv: Sequence[str],
    canonical_argv: Sequence[str],
    server_parallel: int,
    per_slot_context_tokens: int,
) -> tuple[str, ...]:
    if len(capacity_argv) != len(canonical_argv):
        raise ValidationError("capacity command length differs from canonical deployment")
    allowed_value_indexes: dict[int, str] = {}
    for option in _ALLOWED_COMMAND_DIFFERENCES:
        capacity_index = _option_index(capacity_argv, option, "capacity server argv")
        canonical_index = _option_index(canonical_argv, option, "canonical server argv")
        if capacity_index != canonical_index:
            raise ValidationError(f"capacity command option {option} changed position")
        allowed_value_indexes[capacity_index + 1] = option

    differences: set[str] = set()
    for index, (capacity_value, canonical_value) in enumerate(
        zip(capacity_argv, canonical_argv, strict=True)
    ):
        if capacity_value == canonical_value:
            continue
        option = allowed_value_indexes.get(index)
        if option is None:
            raise ValidationError(
                f"capacity command materially differs from canonical argv at index {index}"
            )
        differences.add(option)

    if _option_integer(capacity_argv, "--parallel", "capacity server argv") != server_parallel:
        raise ValidationError("capacity command --parallel does not match the planned level")
    expected_context = server_parallel * per_slot_context_tokens
    if _option_integer(capacity_argv, "--ctx-size", "capacity server argv") != expected_context:
        raise ValidationError(
            "capacity command --ctx-size must preserve the planned per-slot context"
        )
    if _option_integer(canonical_argv, "--parallel", "canonical server argv") != 1:
        raise ValidationError("canonical deployment command must use --parallel 1")
    if (
        _option_integer(canonical_argv, "--ctx-size", "canonical server argv")
        != per_slot_context_tokens
    ):
        raise ValidationError("canonical deployment context does not match per-slot context")
    _option_integer(capacity_argv, "--port", "capacity server argv")
    return tuple(option for option in _ALLOWED_COMMAND_DIFFERENCES if option in differences)


def _quality_checks(
    *,
    plan: CapacityPlan,
    paths: Mapping[str, Path],
    fingerprints: dict[str, str],
    sources: dict[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
    canonical_commands: Mapping[str, Any],
    configuration_by_key: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    wrappers: dict[str, Mapping[str, Any]] = {}
    for label in _expected_quality_sequence(plan):
        candidate_id, server_parallel = _parse_quality_label(label)
        wrapper, input_sha256 = load_json_object_snapshot(paths[label])
        _validate_quality_wrapper(
            wrapper,
            f"quality wrapper {label}",
            plan=plan,
            provenance=provenance,
            expected_candidate_id=candidate_id,
            expected_server_parallel=server_parallel,
            canonical_command=_mapping(
                canonical_commands[candidate_id],
                f"capacity canonical command {candidate_id}",
            ),
        )
        wrappers[label] = wrapper
        fingerprints[label] = input_sha256
        sources[label] = {
            "input_sha256": input_sha256,
            "content_sha256": sha256_json(wrapper),
            "wrapper": deepcopy(dict(wrapper)),
        }
    _crosslink_quality_commands(
        plan=plan,
        wrappers=wrappers,
        configuration_by_key=configuration_by_key,
    )
    return _derive_quality_checks(
        plan=plan,
        wrappers=wrappers,
        fingerprints=fingerprints,
    )


def _derive_quality_checks(
    *,
    plan: CapacityPlan,
    wrappers: Mapping[str, Mapping[str, Any]],
    fingerprints: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    expected_labels = _expected_quality_sequence(plan)
    if list(wrappers) != expected_labels:
        raise ValidationError("capacity quality wrapper coverage or order is incomplete")
    if set(fingerprints) != set(expected_labels):
        raise ValidationError("capacity quality wrapper fingerprints are incomplete")

    raw_checks: dict[tuple[str, int], Mapping[str, Any]] = {}
    outcomes: dict[tuple[str, int], tuple[tuple[str, bool, str | None], ...]] = {}
    for label in expected_labels:
        candidate_id, server_parallel = _parse_quality_label(label)
        wrapper = wrappers[label]
        raw = _mapping(wrapper.get("evaluation"), f"quality wrapper {label}.evaluation")
        quality = _mapping(raw.get("quality"), f"quality wrapper {label}.quality")
        cases = quality.get("cases")
        if not isinstance(cases, list):
            raise ValidationError(f"quality wrapper {label}.quality.cases must be an array")
        outcome = tuple(
            (
                _text(_mapping(case, "quality case").get("id"), "quality case id"),
                bool(_mapping(case, "quality case").get("matched")),
                (
                    None
                    if _mapping(case, "quality case").get("matched_answer") is None
                    else _text(
                        _mapping(case, "quality case").get("matched_answer"),
                        "quality case matched_answer",
                    )
                ),
            )
            for case in cases
        )
        raw_checks[(candidate_id, server_parallel)] = raw
        outcomes[(candidate_id, server_parallel)] = outcome

    candidate_reference_outcomes = {
        candidate.id: outcomes[(candidate.id, plan.server_parallel_levels[0])]
        for candidate in plan.candidates
    }
    checks: list[Mapping[str, Any]] = []
    for candidate in plan.candidates:
        for server_parallel in plan.server_parallel_levels:
            raw = raw_checks[(candidate.id, server_parallel)]
            quality = _mapping(raw["quality"], "quality artifact quality")
            suite = _mapping(raw["suite"], "quality artifact suite")
            score = _finite(
                quality.get("score"),
                "quality artifact score",
                minimum=0.0,
                maximum=1.0,
            )
            passed = _integer(
                quality.get("passed"),
                "quality artifact passed",
                minimum=0,
                maximum=1000,
            )
            total = _integer(
                quality.get("total"),
                "quality artifact total",
                minimum=1,
                maximum=1000,
            )
            reference_raw = raw_checks[(plan.reference_candidate_id, server_parallel)]
            reference_score = float(
                _mapping(reference_raw["quality"], "reference quality").get("score")
            )
            retention = score / reference_score if reference_score > 0 else None
            outcomes_match = (
                outcomes[(candidate.id, server_parallel)]
                == candidate_reference_outcomes[candidate.id]
            )
            failures: list[str] = []
            if score < plan.minimum_quality_score:
                failures.append("quality_score_below_minimum")
            if candidate.id != plan.reference_candidate_id and (
                retention is None or retention < plan.minimum_quality_retention
            ):
                failures.append("quality_retention_below_minimum")
            if plan.require_identical_quality_outcomes and not outcomes_match:
                failures.append("quality_outcomes_changed_across_parallel_levels")
            label = _quality_label(candidate.id, server_parallel)
            wrapper = wrappers[label]
            checks.append(
                {
                    "candidate_id": candidate.id,
                    "server_parallel": server_parallel,
                    "suite_id": _text(suite.get("id"), "quality suite id"),
                    "suite_sha256": _digest(suite.get("sha256"), "quality suite SHA-256"),
                    "score": score,
                    "passed": passed,
                    "total": total,
                    "retention_vs_reference": retention,
                    "outcomes_match_candidate_reference": outcomes_match,
                    "outcomes_sha256": _outcomes_sha256(outcomes[(candidate.id, server_parallel)]),
                    "gate_met": not failures,
                    "failure_reasons": failures,
                    "source_label": label,
                    "input_sha256": _digest(
                        fingerprints[label],
                        f"quality wrapper {label} input SHA-256",
                    ),
                    "source_content_sha256": sha256_json(wrapper),
                    "evaluation_sha256": _digest(
                        wrapper.get("evaluation_sha256"),
                        f"quality wrapper {label} evaluation SHA-256",
                    ),
                }
            )
    return checks


def _pass_metrics(
    *,
    pass_id: str,
    row: Mapping[str, Any],
    server_peak_rss_mib: float,
    source_label: str,
    source_sha256: str,
) -> Mapping[str, Any]:
    return {
        "pass_id": pass_id,
        "source_load_label": source_label,
        "source_load_sha256": source_sha256,
        "request_count": int(row["request_count"]),
        "completed_requests": int(row["completed_requests"]),
        "completion_rate": float(row["completion_rate"]),
        "requests_per_second": float(row["requests_per_second"]),
        "generated_tokens_per_second": float(row["generated_tokens_per_second"]),
        "ttft_ms_p95": (None if row["ttft_ms_p95"] is None else float(row["ttft_ms_p95"])),
        "e2e_latency_ms_p95": (
            None if row["e2e_latency_ms_p95"] is None else float(row["e2e_latency_ms_p95"])
        ),
        "slo_met": bool(row["slo_met"]),
        "slo_failures": list(row["slo_failures"]),
        "server_peak_rss_mib": server_peak_rss_mib,
    }


def _capacity_cell(
    *,
    candidate_id: str,
    server_parallel: int,
    client_concurrency: int,
    pass_metrics: Sequence[Mapping[str, Any]],
    quality_gate_met: bool,
    plan: CapacityPlan,
) -> Mapping[str, Any]:
    if len(pass_metrics) != len(plan.passes):
        raise ValidationError("capacity cell does not contain every counterbalanced pass")
    pass_ids = tuple(str(metric["pass_id"]) for metric in pass_metrics)
    if pass_ids != tuple(pass_spec.id for pass_spec in plan.passes):
        raise ValidationError("capacity cell pass order does not match the plan")

    all_slo = all(bool(metric["slo_met"]) for metric in pass_metrics)
    max_rss = max(float(metric["server_peak_rss_mib"]) for metric in pass_metrics)
    completion_min = min(float(metric["completion_rate"]) for metric in pass_metrics)
    throughput_values = [float(metric["generated_tokens_per_second"]) for metric in pass_metrics]
    request_rate_values = [float(metric["requests_per_second"]) for metric in pass_metrics]
    ttft_values = [metric["ttft_ms_p95"] for metric in pass_metrics]
    e2e_values = [metric["e2e_latency_ms_p95"] for metric in pass_metrics]
    throughput_spread = _relative_spread(throughput_values)
    e2e_spread = _optional_relative_spread(e2e_values)
    failures: list[str] = []
    if plan.require_every_pass_slo and not all_slo:
        failures.append("one_or_more_passes_failed_load_slo")
    if max_rss > plan.max_server_peak_rss_mib:
        failures.append("server_peak_rss_above_maximum")
    if not quality_gate_met:
        failures.append("quality_gate_failed")
    if throughput_spread > plan.max_throughput_relative_spread_percent:
        failures.append("throughput_relative_spread_above_maximum")
    if e2e_spread is None or e2e_spread > plan.max_e2e_relative_spread_percent:
        failures.append("e2e_relative_spread_above_maximum")
    for metric in pass_metrics:
        for failure in metric["slo_failures"]:
            qualified = f"{metric['pass_id']}:{failure}"
            if qualified not in failures:
                failures.append(qualified)

    summary = {
        "every_pass_slo_met": all_slo,
        "quality_gate_met": quality_gate_met,
        "completion_rate_min": completion_min,
        "requests_per_second_median": statistics.median(request_rate_values),
        "generated_tokens_per_second_median": statistics.median(throughput_values),
        "ttft_ms_p95_median": _optional_median(ttft_values),
        "e2e_latency_ms_p95_median": _optional_median(e2e_values),
        "server_peak_rss_mib_max": max_rss,
        "throughput_relative_spread_percent": throughput_spread,
        "ttft_relative_spread_percent": _optional_relative_spread(ttft_values),
        "e2e_relative_spread_percent": e2e_spread,
        "capacity_gate_met": not failures,
        "failure_reasons": failures,
    }
    return {
        "candidate_id": candidate_id,
        "server_parallel": server_parallel,
        "client_concurrency": client_concurrency,
        "pass_metrics": [dict(metric) for metric in pass_metrics],
        "summary": summary,
    }


def _capacity_selections(
    plan: CapacityPlan,
    cells: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    selections: list[Mapping[str, Any]] = []
    for candidate in plan.candidates:
        candidate_cells = [cell for cell in cells if cell["candidate_id"] == candidate.id]
        eligible = [
            cell
            for cell in candidate_cells
            if _mapping(cell["summary"], "capacity cell summary").get("capacity_gate_met") is True
        ]
        reference = next(
            cell
            for cell in candidate_cells
            if cell["server_parallel"] == 1 and cell["client_concurrency"] == 1
        )
        if eligible:
            numeric_best = max(
                float(
                    _mapping(cell["summary"], "eligible capacity cell summary")[
                        "generated_tokens_per_second_median"
                    ]
                )
                for cell in eligible
            )
            objective_floor = numeric_best * (1.0 - plan.objective_tolerance_percent / 100.0)
            within_tolerance = [
                cell
                for cell in eligible
                if float(
                    _mapping(cell["summary"], "eligible capacity cell summary")[
                        "generated_tokens_per_second_median"
                    ]
                )
                >= objective_floor
                or math.isclose(
                    float(
                        _mapping(cell["summary"], "eligible capacity cell summary")[
                            "generated_tokens_per_second_median"
                        ]
                    ),
                    objective_floor,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ]
            best = min(within_tolerance, key=_selection_preference_key)
            selected = {
                "server_parallel": int(best["server_parallel"]),
                "client_concurrency": int(best["client_concurrency"]),
                "generated_tokens_per_second_median": float(
                    _mapping(best["summary"], "best cell summary")[
                        "generated_tokens_per_second_median"
                    ]
                ),
            }
            comparison = _selection_comparison(
                _mapping(best["summary"], "best cell summary"),
                _mapping(reference["summary"], "reference cell summary"),
            )
        else:
            numeric_best = None
            within_tolerance = []
            selected = None
            comparison = None
        selections.append(
            {
                "candidate_id": candidate.id,
                "reference_cell": {
                    "server_parallel": 1,
                    "client_concurrency": 1,
                },
                "selected_cell": selected,
                "comparison_to_reference_percent": comparison,
                "eligible_cell_count": len(eligible),
                "numeric_best_generated_tokens_per_second_median": numeric_best,
                "objective_tolerance_percent": plan.objective_tolerance_percent,
                "within_tolerance_cell_count": len(within_tolerance),
                "selection_basis": (
                    "Find the best median generated tokens/s among cells where both "
                    "counterbalanced passes meet the load SLO, quality, RSS, and spread "
                    "gates; "
                    f"treat cells within {plan.objective_tolerance_percent:.1f}% of that "
                    "best as equivalent, then apply the fixed lower-is-better tie-breakers."
                ),
            }
        )
    return selections


def _selection_preference_key(
    cell: Mapping[str, Any],
) -> tuple[float, float, float, int, int]:
    summary = _mapping(cell["summary"], "capacity cell summary")
    e2e = summary["e2e_latency_ms_p95_median"]
    ttft = summary["ttft_ms_p95_median"]
    if e2e is None or ttft is None:
        raise ValidationError("eligible capacity cells must have latency measurements")
    return (
        float(e2e),
        float(ttft),
        float(summary["server_peak_rss_mib_max"]),
        int(cell["server_parallel"]),
        int(cell["client_concurrency"]),
    )


def _selection_comparison(
    selected: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> Mapping[str, float]:
    metrics = (
        "generated_tokens_per_second_median",
        "e2e_latency_ms_p95_median",
        "ttft_ms_p95_median",
        "server_peak_rss_mib_max",
    )
    result: dict[str, float] = {}
    for metric in metrics:
        selected_value = selected[metric]
        reference_value = reference[metric]
        if selected_value is None or reference_value is None or float(reference_value) == 0:
            raise ValidationError("capacity selection comparison requires nonzero measurements")
        result[metric] = (
            (float(selected_value) - float(reference_value)) / float(reference_value) * 100.0
        )
    return result


def _validate_source_artifacts(
    value: Any,
    *,
    plan: CapacityPlan,
    provenance: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
    load_contract: Mapping[str, Any],
) -> tuple[Mapping[str, Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]:
    sources = _mapping(value, "capacity study source_artifacts")
    _exact_fields(
        sources,
        {"load_evaluations", "quality_wrappers"},
        "capacity study source_artifacts",
    )
    load_values = _mapping(
        sources.get("load_evaluations"),
        "capacity study source_artifacts.load_evaluations",
    )
    expected_loads = _expected_run_sequence(plan)
    if len(load_values) != len(expected_loads) or set(load_values) != set(expected_loads):
        raise ValidationError("capacity study embedded load evaluation coverage is incomplete")
    quality_values = _mapping(
        sources.get("quality_wrappers"),
        "capacity study source_artifacts.quality_wrappers",
    )
    expected_quality = _expected_quality_sequence(plan)
    if len(quality_values) != len(expected_quality) or set(quality_values) != set(expected_quality):
        raise ValidationError("capacity study embedded quality wrapper coverage is incomplete")

    load_fingerprints = _mapping(
        fingerprints.get("load_artifacts"),
        "capacity study load fingerprints",
    )
    quality_fingerprints = _mapping(
        fingerprints.get("quality_artifacts"),
        "capacity study quality fingerprints",
    )
    passes = {pass_spec.id: pass_spec for pass_spec in plan.passes}
    load_sources: dict[str, Mapping[str, Any]] = {}
    load_identities: set[tuple[str, str, str]] = set()
    for label in expected_loads:
        context = f"capacity study source_artifacts.load_evaluations[{label!r}]"
        source = _mapping(load_values[label], context)
        _exact_fields(
            source,
            {"input_sha256", "content_sha256", "evaluation"},
            context,
        )
        input_sha256 = _digest(source.get("input_sha256"), f"{context}.input_sha256")
        if input_sha256 != load_fingerprints[label]:
            raise ValidationError(f"{context}.input_sha256 is not cross-linked")
        evaluation = _mapping(source.get("evaluation"), f"{context}.evaluation")
        content_sha256 = _digest(
            source.get("content_sha256"),
            f"{context}.content_sha256",
        )
        if content_sha256 != sha256_json(evaluation):
            raise ValidationError(f"{context}.content_sha256 is not recomputable")
        validate_load_evaluation(evaluation, require_evidence_binding=True)
        if evaluation.get("synthetic") is not False:
            raise ValidationError(f"{context}.evaluation must be measured")
        pass_id, candidate_id, server_parallel = _parse_run_label(label)
        if evaluation.get("candidate_id") != candidate_id:
            raise ValidationError(f"{context}.evaluation candidate_id does not match")
        if evaluation.get("slo") != plan.load_slo:
            raise ValidationError(f"{context}.evaluation SLO does not match the plan")
        if evaluation.get("methodology") != load_contract.get("methodology"):
            raise ValidationError(
                f"{context}.evaluation methodology does not match the load contract"
            )
        if evaluation.get("slo") != load_contract.get("slo"):
            raise ValidationError(f"{context}.evaluation SLO does not match the load contract")
        if _load_execution_order(evaluation, context) != passes[pass_id].client_concurrency_order:
            raise ValidationError(f"{context}.evaluation execution_order does not match its pass")
        binding = _mapping(
            evaluation.get("evidence_binding"),
            f"{context}.evaluation.evidence_binding",
        )
        if binding.get("plan_sha256") != plan.pins["load_plan_sha256"]:
            raise ValidationError(f"{context}.evaluation load-plan pin does not match")
        configuration = _mapping(
            binding.get("server_configuration"),
            f"{context}.evaluation.server_configuration",
        )
        if (
            configuration.get("load_parallel") != server_parallel
            or configuration.get("canonical_parallel") != server_parallel
        ):
            raise ValidationError(f"{context}.evaluation parallel identity does not match")
        load_argv = _argv(
            configuration.get("load_server_argv"),
            f"{context}.evaluation load argv",
        )
        binding_argv = _argv(
            configuration.get("canonical_server_argv"),
            f"{context}.evaluation binding argv",
        )
        if load_argv != binding_argv:
            raise ValidationError(f"{context}.evaluation must use self-bound commands")
        command_sha256 = _digest(
            configuration.get("load_server_command_sha256"),
            f"{context}.evaluation command SHA-256",
        )
        if command_sha256 != configuration.get("canonical_server_command_sha256"):
            raise ValidationError(f"{context}.evaluation command hashes are not self-bound")
        request_base_url = _normalize_base_url(
            binding.get("request_base_url"),
            f"{context}.evaluation request_base_url",
        )
        _validate_request_endpoint(request_base_url, load_argv, context)
        identity = (
            candidate_id,
            request_base_url,
            command_sha256,
        )
        if identity in load_identities:
            raise ValidationError(f"duplicate embedded capacity load identity for {label!r}")
        load_identities.add(identity)
        load_sources[label] = source

    canonical_commands = _mapping(
        provenance.get("canonical_commands"),
        "capacity study provenance canonical_commands",
    )
    quality_sources: dict[str, Mapping[str, Any]] = {}
    for label in expected_quality:
        context = f"capacity study source_artifacts.quality_wrappers[{label!r}]"
        source = _mapping(quality_values[label], context)
        _exact_fields(
            source,
            {"input_sha256", "content_sha256", "wrapper"},
            context,
        )
        input_sha256 = _digest(source.get("input_sha256"), f"{context}.input_sha256")
        if input_sha256 != quality_fingerprints[label]:
            raise ValidationError(f"{context}.input_sha256 is not cross-linked")
        wrapper = _mapping(source.get("wrapper"), f"{context}.wrapper")
        content_sha256 = _digest(
            source.get("content_sha256"),
            f"{context}.content_sha256",
        )
        if content_sha256 != sha256_json(wrapper):
            raise ValidationError(f"{context}.content_sha256 is not recomputable")
        candidate_id, server_parallel = _parse_quality_label(label)
        _validate_quality_wrapper(
            wrapper,
            f"{context}.wrapper",
            plan=plan,
            provenance=provenance,
            expected_candidate_id=candidate_id,
            expected_server_parallel=server_parallel,
            canonical_command=_mapping(
                canonical_commands[candidate_id],
                f"capacity canonical command {candidate_id}",
            ),
        )
        quality_sources[label] = wrapper
    return load_sources, quality_sources


def _validate_quality_wrapper(
    value: Any,
    context: str,
    *,
    plan: CapacityPlan | None = None,
    provenance: Mapping[str, Any] | None = None,
    expected_candidate_id: str | None = None,
    expected_server_parallel: int | None = None,
    canonical_command: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    wrapper = _mapping(value, context)
    _exact_fields(
        wrapper,
        {
            "schema_version",
            "classification",
            "pass_id",
            "candidate_id",
            "server_parallel",
            "run_id",
            "run_attempt",
            "request_base_url",
            "evaluation_sha256",
            "evaluation_content_sha256",
            "server_command_sha256",
            "server_command_content_sha256",
            "server_argv_sha256",
            "server_argv",
            "evaluation",
        },
        context,
    )
    if wrapper.get("schema_version") != "1.0":
        raise ValidationError(f"{context}.schema_version must be '1.0'")
    if wrapper.get("classification") != "capacity-quality-binding":
        raise ValidationError(f"{context}.classification is not supported")
    if wrapper.get("pass_id") != "quality":
        raise ValidationError(f"{context}.pass_id must be 'quality'")
    candidate_id = _text(wrapper.get("candidate_id"), f"{context}.candidate_id", maximum=128)
    server_parallel = _integer(
        wrapper.get("server_parallel"),
        f"{context}.server_parallel",
        minimum=1,
        maximum=1024,
    )
    run_id = _text(wrapper.get("run_id"), f"{context}.run_id", maximum=128)
    run_attempt = _integer(
        wrapper.get("run_attempt"),
        f"{context}.run_attempt",
        minimum=1,
        maximum=1_000_000,
    )
    if expected_candidate_id is not None and candidate_id != expected_candidate_id:
        raise ValidationError(f"{context}.candidate_id does not match its source label")
    if expected_server_parallel is not None and server_parallel != expected_server_parallel:
        raise ValidationError(f"{context}.server_parallel does not match its source label")
    if plan is not None and server_parallel not in plan.server_parallel_levels:
        raise ValidationError(f"{context}.server_parallel is not predeclared")
    if provenance is not None:
        source = _mapping(provenance.get("source"), "capacity provenance source")
        if run_id != source.get("run_id") or run_attempt != source.get("run_attempt"):
            raise ValidationError(f"{context} workflow run identity does not match provenance")

    evaluation = _mapping(wrapper.get("evaluation"), f"{context}.evaluation")
    validate_server_evaluation(evaluation)
    if evaluation.get("candidate_id") != candidate_id:
        raise ValidationError(f"{context}.evaluation candidate_id does not match")
    _digest(wrapper.get("evaluation_sha256"), f"{context}.evaluation_sha256")
    if _digest(
        wrapper.get("evaluation_content_sha256"),
        f"{context}.evaluation_content_sha256",
    ) != sha256_json(evaluation):
        raise ValidationError(f"{context}.evaluation_content_sha256 is not recomputable")
    if plan is not None:
        suite = _mapping(evaluation.get("suite"), f"{context}.evaluation.suite")
        if suite.get("sha256") != plan.pins["evaluation_suite_sha256"]:
            raise ValidationError(f"{context}.evaluation suite SHA-256 does not match the plan")

    argv = _argv(wrapper.get("server_argv"), f"{context}.server_argv")
    if _option_integer(argv, "--parallel", f"{context}.server_argv") != server_parallel:
        raise ValidationError(f"{context}.server argv parallel does not match")
    if _digest(wrapper.get("server_argv_sha256"), f"{context}.server_argv_sha256") != _argv_sha256(
        argv
    ):
        raise ValidationError(f"{context}.server_argv_sha256 is not recomputable")
    command_content_sha256 = sha256_json({"schema_version": "1.0", "argv": list(argv)})
    if (
        _digest(
            wrapper.get("server_command_content_sha256"),
            f"{context}.server_command_content_sha256",
        )
        != command_content_sha256
    ):
        raise ValidationError(f"{context}.server command content hash is not recomputable")
    _digest(
        wrapper.get("server_command_sha256"),
        f"{context}.server_command_sha256",
    )
    request_base_url = _normalize_base_url(
        wrapper.get("request_base_url"),
        f"{context}.request_base_url",
    )
    if wrapper.get("request_base_url") != request_base_url:
        raise ValidationError(f"{context}.request_base_url must be normalized")
    _validate_request_endpoint(request_base_url, argv, context)
    if plan is not None and canonical_command is not None:
        canonical_argv = _argv(
            canonical_command.get("argv"),
            f"{context}.canonical_server_argv",
        )
        _command_differences(
            capacity_argv=argv,
            canonical_argv=canonical_argv,
            server_parallel=server_parallel,
            per_slot_context_tokens=plan.per_slot_context_tokens,
        )
    return wrapper


def _crosslink_quality_commands(
    *,
    plan: CapacityPlan,
    wrappers: Mapping[str, Mapping[str, Any]],
    configuration_by_key: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> None:
    for label in _expected_quality_sequence(plan):
        candidate_id, server_parallel = _parse_quality_label(label)
        quality_argv = _argv(
            wrappers[label].get("server_argv"),
            f"quality wrapper {label}.server_argv",
        )
        for pass_spec in plan.passes:
            configuration = configuration_by_key[(pass_spec.id, candidate_id, server_parallel)]
            load_argv = _argv(
                configuration.get("load_argv"),
                f"capacity server configuration {label}.load_argv",
            )
            _require_argv_equivalent_except(
                quality_argv,
                load_argv,
                allowed_options=("--port",),
                context=f"quality wrapper {label} and {pass_spec.id} load command",
            )


def _cells_from_sources(
    *,
    plan: CapacityPlan,
    load_sources: Mapping[str, Mapping[str, Any]],
    quality_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
    configuration_by_key: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    cells: list[Mapping[str, Any]] = []
    for candidate in plan.candidates:
        for server_parallel in plan.server_parallel_levels:
            quality = quality_by_key[(candidate.id, server_parallel)]
            for client_concurrency in plan.client_concurrency_levels:
                pass_metrics: list[Mapping[str, Any]] = []
                for pass_spec in plan.passes:
                    label = _run_label(pass_spec.id, candidate.id, server_parallel)
                    source = load_sources[label]
                    evaluation = _mapping(
                        source.get("evaluation"),
                        f"embedded load source {label}.evaluation",
                    )
                    rows = evaluation.get("rows")
                    if not isinstance(rows, list):
                        raise ValidationError(f"embedded load source {label}.rows must be an array")
                    matching_rows = [
                        _mapping(row, f"embedded load source {label}.row")
                        for row in rows
                        if _mapping(row, f"embedded load source {label}.row").get("concurrency")
                        == client_concurrency
                    ]
                    if len(matching_rows) != 1:
                        raise ValidationError(
                            f"embedded load source {label} must have one row for "
                            f"concurrency {client_concurrency}"
                        )
                    configuration = configuration_by_key[
                        (pass_spec.id, candidate.id, server_parallel)
                    ]
                    pass_metrics.append(
                        _pass_metrics(
                            pass_id=pass_spec.id,
                            row=matching_rows[0],
                            server_peak_rss_mib=float(configuration["server_peak_rss_mib"]),
                            source_label=label,
                            source_sha256=str(source["input_sha256"]),
                        )
                    )
                cells.append(
                    _capacity_cell(
                        candidate_id=candidate.id,
                        server_parallel=server_parallel,
                        client_concurrency=client_concurrency,
                        pass_metrics=pass_metrics,
                        quality_gate_met=bool(quality["gate_met"]),
                        plan=plan,
                    )
                )
    return cells


def _validate_serialized_quality_checks(
    value: Any,
    plan: CapacityPlan,
) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValidationError("capacity study quality_checks must be an array")
    expected = [
        (candidate.id, level)
        for candidate in plan.candidates
        for level in plan.server_parallel_levels
    ]
    if len(value) != len(expected):
        raise ValidationError("capacity study quality_checks coverage is incomplete")
    checks: list[Mapping[str, Any]] = []
    for index, ((candidate_id, level), raw_value) in enumerate(zip(expected, value, strict=True)):
        context = f"capacity study quality_checks[{index}]"
        raw = _mapping(raw_value, context)
        _exact_fields(
            raw,
            {
                "candidate_id",
                "server_parallel",
                "suite_id",
                "suite_sha256",
                "score",
                "passed",
                "total",
                "retention_vs_reference",
                "outcomes_match_candidate_reference",
                "outcomes_sha256",
                "gate_met",
                "failure_reasons",
                "input_sha256",
            },
            context,
        )
        if raw.get("candidate_id") != candidate_id or raw.get("server_parallel") != level:
            raise ValidationError(f"{context} identity is not in canonical order")
        _text(raw.get("suite_id"), f"{context}.suite_id")
        if (
            _digest(raw.get("suite_sha256"), f"{context}.suite_sha256")
            != plan.pins["evaluation_suite_sha256"]
        ):
            raise ValidationError(f"{context}.suite_sha256 does not match the plan")
        score = _finite(raw.get("score"), f"{context}.score", minimum=0.0, maximum=1.0)
        passed = _integer(raw.get("passed"), f"{context}.passed", minimum=0, maximum=1000)
        total = _integer(raw.get("total"), f"{context}.total", minimum=1, maximum=1000)
        if not math.isclose(score, passed / total, rel_tol=1e-12, abs_tol=1e-12):
            raise ValidationError(f"{context}.score does not match passed/total")
        retention = raw.get("retention_vs_reference")
        if retention is not None:
            _finite(retention, f"{context}.retention_vs_reference", minimum=0.0)
        if not isinstance(raw.get("outcomes_match_candidate_reference"), bool):
            raise ValidationError(f"{context}.outcomes_match_candidate_reference must be boolean")
        _digest(raw.get("outcomes_sha256"), f"{context}.outcomes_sha256")
        if not isinstance(raw.get("gate_met"), bool):
            raise ValidationError(f"{context}.gate_met must be boolean")
        failures = _text_list(raw.get("failure_reasons"), f"{context}.failure_reasons")
        if bool(failures) == bool(raw.get("gate_met")):
            raise ValidationError(f"{context}.gate_met does not match failure reasons")
        _digest(raw.get("input_sha256"), f"{context}.input_sha256")
        checks.append(raw)
    return checks


def _validate_serialized_server_configurations(
    value: Any,
    plan: CapacityPlan,
    *,
    load_sources: Mapping[str, Mapping[str, Any]],
    fingerprints: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValidationError("capacity study server_configurations must be an array")
    expected = [
        (pass_spec.id, candidate.id, level)
        for pass_spec in plan.passes
        for candidate in plan.candidates
        for level in plan.server_parallel_levels
    ]
    if len(value) != len(expected):
        raise ValidationError("capacity study server configuration coverage is incomplete")
    candidates = {candidate.id: candidate for candidate in plan.candidates}
    configurations: list[Mapping[str, Any]] = []
    for index, ((pass_id, candidate_id, level), raw_value) in enumerate(
        zip(expected, value, strict=True)
    ):
        context = f"capacity study server_configurations[{index}]"
        raw = _mapping(raw_value, context)
        _exact_fields(
            raw,
            {
                "pass_id",
                "candidate_id",
                "server_parallel",
                "source_load_label",
                "load_evaluation_sha256",
                "load_evaluation_content_sha256",
                "total_context_tokens",
                "per_slot_context_tokens",
                "request_base_url",
                "load_command_sha256",
                "load_command_content_sha256",
                "load_argv_sha256",
                "canonical_command_sha256",
                "canonical_argv_sha256",
                "load_argv",
                "canonical_argv",
                "differing_options",
                "server_peak_rss_mib",
                "rss_sha256",
                "server_log_sha256",
                "kleidiai_model_buffer_marker_count",
            },
            context,
        )
        if (
            raw.get("pass_id") != pass_id
            or raw.get("candidate_id") != candidate_id
            or raw.get("server_parallel") != level
        ):
            raise ValidationError(f"{context} identity is not in canonical order")
        if raw.get("per_slot_context_tokens") != plan.per_slot_context_tokens:
            raise ValidationError(f"{context}.per_slot_context_tokens does not match the plan")
        if raw.get("total_context_tokens") != level * plan.per_slot_context_tokens:
            raise ValidationError(f"{context}.total_context_tokens does not preserve per-slot size")
        source_label = _run_label(pass_id, candidate_id, level)
        if raw.get("source_load_label") != source_label:
            raise ValidationError(f"{context}.source_load_label does not match its identity")
        source = _mapping(load_sources[source_label], f"embedded load source {source_label}")
        evaluation = _mapping(
            source.get("evaluation"),
            f"embedded load source {source_label}.evaluation",
        )
        binding = _mapping(
            evaluation.get("evidence_binding"),
            f"embedded load source {source_label}.evidence_binding",
        )
        bound_configuration = _mapping(
            binding.get("server_configuration"),
            f"embedded load source {source_label}.server_configuration",
        )
        load_fingerprints = _mapping(
            fingerprints.get("load_artifacts"),
            "capacity study load fingerprints",
        )
        rss_fingerprints = _mapping(
            fingerprints.get("rss_artifacts"),
            "capacity study RSS fingerprints",
        )
        log_fingerprints = _mapping(
            fingerprints.get("server_logs"),
            "capacity study server-log fingerprints",
        )
        if raw.get("load_evaluation_sha256") != load_fingerprints[source_label]:
            raise ValidationError(f"{context}.load_evaluation_sha256 is not cross-linked")
        if raw.get("load_evaluation_sha256") != source.get("input_sha256"):
            raise ValidationError(f"{context}.load_evaluation_sha256 does not match its source")
        if raw.get("load_evaluation_content_sha256") != source.get("content_sha256"):
            raise ValidationError(
                f"{context}.load_evaluation_content_sha256 does not match its source"
            )
        request_base_url = _normalize_base_url(
            raw.get("request_base_url"),
            f"{context}.request_base_url",
        )
        if request_base_url != binding.get("request_base_url"):
            raise ValidationError(f"{context}.request_base_url does not match its load source")
        load_digest = _digest(
            raw.get("load_command_sha256"),
            f"{context}.load_command_sha256",
        )
        if load_digest != bound_configuration.get("load_server_command_sha256"):
            raise ValidationError(f"{context}.load_command_sha256 does not match its source")
        canonical_digest = _digest(
            raw.get("canonical_command_sha256"),
            f"{context}.canonical_command_sha256",
        )
        if canonical_digest != candidates[candidate_id].canonical_server_command_sha256:
            raise ValidationError(f"{context}.canonical_command_sha256 does not match the plan")
        load_argv = _argv(raw.get("load_argv"), f"{context}.load_argv")
        canonical_argv = _argv(raw.get("canonical_argv"), f"{context}.canonical_argv")
        if list(load_argv) != bound_configuration.get("load_server_argv"):
            raise ValidationError(f"{context}.load_argv does not match its source")
        if _digest(
            raw.get("load_command_content_sha256"),
            f"{context}.load_command_content_sha256",
        ) != sha256_json({"schema_version": "1.0", "argv": list(load_argv)}):
            raise ValidationError(f"{context}.load_command_content_sha256 is not recomputable")
        if _digest(raw.get("load_argv_sha256"), f"{context}.load_argv_sha256") != _argv_sha256(
            load_argv
        ):
            raise ValidationError(f"{context}.load_argv_sha256 is not recomputable")
        if (
            _digest(
                raw.get("canonical_argv_sha256"),
                f"{context}.canonical_argv_sha256",
            )
            != candidates[candidate_id].canonical_server_argv_sha256
            or _argv_sha256(canonical_argv) != candidates[candidate_id].canonical_server_argv_sha256
        ):
            raise ValidationError(f"{context}.canonical_argv_sha256 does not match the plan")
        expected_differences = _command_differences(
            capacity_argv=load_argv,
            canonical_argv=canonical_argv,
            server_parallel=level,
            per_slot_context_tokens=plan.per_slot_context_tokens,
        )
        if raw.get("differing_options") != list(expected_differences):
            raise ValidationError(f"{context}.differing_options are not recomputable")
        _finite(
            raw.get("server_peak_rss_mib"),
            f"{context}.server_peak_rss_mib",
            strictly_positive=True,
        )
        if (
            _digest(raw.get("rss_sha256"), f"{context}.rss_sha256")
            != rss_fingerprints[source_label]
        ):
            raise ValidationError(f"{context}.rss_sha256 is not cross-linked")
        if (
            _digest(
                raw.get("server_log_sha256"),
                f"{context}.server_log_sha256",
            )
            != log_fingerprints[source_label]
        ):
            raise ValidationError(f"{context}.server_log_sha256 is not cross-linked")
        markers = _integer(
            raw.get("kleidiai_model_buffer_marker_count"),
            f"{context}.kleidiai_model_buffer_marker_count",
            minimum=0,
            maximum=1_000_000,
        )
        if candidates[candidate_id].kleidiai_expected != (markers > 0):
            raise ValidationError(f"{context} KleidiAI model-buffer marker does not match the plan")
        _validate_request_endpoint(request_base_url, load_argv, context)
        configurations.append(raw)
    return configurations


def _validate_serialized_cells(
    value: Any,
    plan: CapacityPlan,
    *,
    quality_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
    configuration_by_key: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValidationError("capacity study cells must be an array")
    expected = [
        (candidate.id, server_parallel, client_concurrency)
        for candidate in plan.candidates
        for server_parallel in plan.server_parallel_levels
        for client_concurrency in plan.client_concurrency_levels
    ]
    if len(value) != len(expected):
        raise ValidationError("capacity study cell coverage is incomplete")
    cells: list[Mapping[str, Any]] = []
    for index, ((candidate_id, server_parallel, client_concurrency), raw_value) in enumerate(
        zip(expected, value, strict=True)
    ):
        context = f"capacity study cells[{index}]"
        raw = _mapping(raw_value, context)
        _exact_fields(
            raw,
            {
                "candidate_id",
                "server_parallel",
                "client_concurrency",
                "pass_metrics",
                "summary",
            },
            context,
        )
        if (
            raw.get("candidate_id") != candidate_id
            or raw.get("server_parallel") != server_parallel
            or raw.get("client_concurrency") != client_concurrency
        ):
            raise ValidationError(f"{context} identity is not in canonical order")
        pass_values = raw.get("pass_metrics")
        if not isinstance(pass_values, list) or len(pass_values) != len(plan.passes):
            raise ValidationError(f"{context}.pass_metrics coverage is incomplete")
        pass_metrics: list[Mapping[str, Any]] = []
        for pass_index, (pass_spec, metric_value) in enumerate(
            zip(plan.passes, pass_values, strict=True)
        ):
            metric_context = f"{context}.pass_metrics[{pass_index}]"
            metric = _mapping(metric_value, metric_context)
            _exact_fields(
                metric,
                {
                    "pass_id",
                    "request_count",
                    "completed_requests",
                    "completion_rate",
                    "requests_per_second",
                    "generated_tokens_per_second",
                    "ttft_ms_p95",
                    "e2e_latency_ms_p95",
                    "slo_met",
                    "slo_failures",
                    "server_peak_rss_mib",
                },
                metric_context,
            )
            if metric.get("pass_id") != pass_spec.id:
                raise ValidationError(f"{metric_context}.pass_id does not match the plan")
            requests = _integer(
                metric.get("request_count"),
                f"{metric_context}.request_count",
                minimum=1,
                maximum=1_000_000,
            )
            completed = _integer(
                metric.get("completed_requests"),
                f"{metric_context}.completed_requests",
                minimum=0,
                maximum=requests,
            )
            completion = _finite(
                metric.get("completion_rate"),
                f"{metric_context}.completion_rate",
                minimum=0.0,
                maximum=1.0,
            )
            if not math.isclose(completion, completed / requests, rel_tol=1e-12, abs_tol=1e-12):
                raise ValidationError(f"{metric_context}.completion_rate is inconsistent")
            _finite(
                metric.get("requests_per_second"),
                f"{metric_context}.requests_per_second",
                minimum=0.0,
            )
            _finite(
                metric.get("generated_tokens_per_second"),
                f"{metric_context}.generated_tokens_per_second",
                minimum=0.0,
            )
            for name in ("ttft_ms_p95", "e2e_latency_ms_p95"):
                if metric.get(name) is not None:
                    _finite(
                        metric.get(name),
                        f"{metric_context}.{name}",
                        strictly_positive=True,
                    )
            if not isinstance(metric.get("slo_met"), bool):
                raise ValidationError(f"{metric_context}.slo_met must be boolean")
            failures = _text_list(
                metric.get("slo_failures"),
                f"{metric_context}.slo_failures",
            )
            if bool(failures) == bool(metric.get("slo_met")):
                raise ValidationError(f"{metric_context}.slo_met conflicts with failures")
            rss = _finite(
                metric.get("server_peak_rss_mib"),
                f"{metric_context}.server_peak_rss_mib",
                strictly_positive=True,
            )
            expected_rss = float(
                configuration_by_key[(pass_spec.id, candidate_id, server_parallel)][
                    "server_peak_rss_mib"
                ]
            )
            if not math.isclose(rss, expected_rss, rel_tol=1e-12, abs_tol=1e-12):
                raise ValidationError(f"{metric_context}.server_peak_rss_mib is inconsistent")
            pass_metrics.append(metric)
        expected_cell = _capacity_cell(
            candidate_id=candidate_id,
            server_parallel=server_parallel,
            client_concurrency=client_concurrency,
            pass_metrics=pass_metrics,
            quality_gate_met=bool(quality_by_key[(candidate_id, server_parallel)]["gate_met"]),
            plan=plan,
        )
        if raw != expected_cell:
            raise ValidationError(f"{context} summary does not match recomputed pass metrics")
        cells.append(raw)
    return cells


def _validate_fingerprints(
    value: Any,
    plan: CapacityPlan,
    *,
    provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    fingerprints = _mapping(value, "capacity study input_fingerprints")
    _exact_fields(
        fingerprints,
        {
            "capacity_plan_sha256",
            "capacity_plan_content_sha256",
            "load_plan_sha256",
            "manifest_sha256",
            "manifest_content_sha256",
            "load_artifacts",
            "rss_artifacts",
            "server_logs",
            "quality_artifacts",
        },
        "capacity study input_fingerprints",
    )
    _digest(
        fingerprints.get("capacity_plan_sha256"),
        "capacity study input_fingerprints.capacity_plan_sha256",
    )
    if _digest(
        fingerprints.get("capacity_plan_content_sha256"),
        "capacity study input_fingerprints.capacity_plan_content_sha256",
    ) != sha256_json(capacity_plan_mapping(plan)):
        raise ValidationError("capacity study plan content fingerprint is not recomputable")
    if (
        _digest(
            fingerprints.get("load_plan_sha256"),
            "capacity study input_fingerprints.load_plan_sha256",
        )
        != plan.pins["load_plan_sha256"]
    ):
        raise ValidationError("capacity study load-plan fingerprint does not match the plan")
    _digest(
        fingerprints.get("manifest_sha256"),
        "capacity study input_fingerprints.manifest_sha256",
    )
    if _digest(
        fingerprints.get("manifest_content_sha256"),
        "capacity study input_fingerprints.manifest_content_sha256",
    ) != sha256_json(provenance):
        raise ValidationError("capacity study provenance content fingerprint is not recomputable")
    expected_run_labels = _expected_run_labels(plan)
    expected_quality_labels = _expected_quality_labels(plan)
    for name in ("load_artifacts", "rss_artifacts", "server_logs"):
        values = _mapping(
            fingerprints.get(name),
            f"capacity study input_fingerprints.{name}",
        )
        if set(values) != expected_run_labels:
            raise ValidationError(f"capacity study {name} fingerprint coverage is incomplete")
        for label, digest in values.items():
            _digest(digest, f"capacity study {name}[{label!r}]")
    quality = _mapping(
        fingerprints.get("quality_artifacts"),
        "capacity study input_fingerprints.quality_artifacts",
    )
    if set(quality) != expected_quality_labels:
        raise ValidationError("capacity study quality fingerprint coverage is incomplete")
    for label, digest in quality.items():
        _digest(digest, f"capacity study quality_artifacts[{label!r}]")
    return fingerprints


def _validate_slo(value: Any, context: str) -> Mapping[str, Any]:
    slo = _mapping(value, context)
    _exact_fields(
        slo,
        {"min_completion_rate", "max_ttft_ms_p95", "max_e2e_latency_ms_p95"},
        context,
    )
    min_completion = _finite(
        slo.get("min_completion_rate"),
        f"{context}.min_completion_rate",
        minimum=0.0,
        maximum=1.0,
    )
    ttft_value = slo.get("max_ttft_ms_p95")
    ttft = None
    if ttft_value is not None:
        ttft = _finite(
            ttft_value,
            f"{context}.max_ttft_ms_p95",
            strictly_positive=True,
        )
    max_e2e = _finite(
        slo.get("max_e2e_latency_ms_p95"),
        f"{context}.max_e2e_latency_ms_p95",
        strictly_positive=True,
    )
    return {
        "min_completion_rate": min_completion,
        "max_ttft_ms_p95": ttft,
        "max_e2e_latency_ms_p95": max_e2e,
    }


def _server_command_document(value: Any, context: str) -> tuple[str, ...]:
    command = _mapping(value, context)
    _exact_fields(command, {"schema_version", "argv"}, context)
    if command.get("schema_version") != "1.0":
        raise ValidationError(f"{context}.schema_version must be '1.0'")
    return _argv(command.get("argv"), f"{context}.argv")


def _argv_sha256(argv: Sequence[str]) -> str:
    normalized = _argv(list(argv), "server argv")
    serialized = json.dumps(
        list(normalized),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_base_url(value: Any, context: str) -> str:
    text = _text(value, context, maximum=2048)
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise ValidationError(f"{context} is invalid: {exc}") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname is None
        or port is None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValidationError(f"{context} must be an http(s) origin with an explicit port")
    host = parsed.hostname.casefold()
    formatted_host = f"[{host}]" if ":" in host else host
    return f"{parsed.scheme}://{formatted_host}:{port}"


def _validate_request_endpoint(
    request_base_url: Any,
    argv: Sequence[str],
    context: str,
) -> None:
    normalized = _normalize_base_url(request_base_url, f"{context} request_base_url")
    parsed = urlsplit(normalized)
    request_host = str(parsed.hostname).casefold()
    request_port = parsed.port
    server_host = _option_value(argv, "--host", f"{context} server argv").casefold()
    server_port = _option_integer(argv, "--port", f"{context} server argv")
    if request_port != server_port:
        raise ValidationError(f"{context} request base_url port must match the server --port")
    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    wildcard_hosts = {"0.0.0.0", "::"}
    if not (
        request_host == server_host
        or request_host in loopback_hosts
        and server_host in loopback_hosts | wildcard_hosts
    ):
        raise ValidationError(
            f"{context} request base_url host does not safely match the server --host"
        )


def _load_execution_order(raw: Mapping[str, Any], context: str) -> tuple[int, ...]:
    methodology = _mapping(raw.get("methodology"), f"{context}.methodology")
    levels = _integer_array(
        methodology.get("concurrency_levels"),
        f"{context}.methodology.concurrency_levels",
    )
    if raw.get("schema_version") == "1.1":
        return _integer_array(raw.get("execution_order"), f"{context}.execution_order")
    return levels


def _require_argv_equivalent_except(
    left: Sequence[str],
    right: Sequence[str],
    *,
    allowed_options: Sequence[str],
    context: str,
) -> None:
    left_argv = _argv(list(left), f"{context} left argv")
    right_argv = _argv(list(right), f"{context} right argv")
    if len(left_argv) != len(right_argv):
        raise ValidationError(f"{context} argv lengths differ")
    allowed_indexes: set[int] = set()
    for option in allowed_options:
        left_index = _option_index(left_argv, option, f"{context} left argv")
        right_index = _option_index(right_argv, option, f"{context} right argv")
        if left_index != right_index:
            raise ValidationError(f"{context} option {option} changed position")
        allowed_indexes.add(left_index + 1)
    for index, (left_value, right_value) in enumerate(zip(left_argv, right_argv, strict=True)):
        if left_value != right_value and index not in allowed_indexes:
            raise ValidationError(f"{context} materially differs at argv[{index}]")


def _resolved_path_key(path: Path) -> str:
    try:
        resolved = Path(path).resolve()
    except OSError as exc:
        raise ValidationError(f"could not resolve capacity input path {path}: {exc}") from exc
    return str(resolved).casefold()


def _reject_duplicate_resolved_input_paths(values: Mapping[str, Path]) -> None:
    seen: dict[str, str] = {}
    for label, path in values.items():
        key = _resolved_path_key(path)
        previous = seen.get(key)
        if previous is not None:
            raise ValidationError(
                f"capacity inputs {previous!r} and {label!r} resolve to the same path"
            )
        seen[key] = label


def _expected_run_sequence(plan: CapacityPlan) -> list[str]:
    return [
        _run_label(pass_spec.id, candidate_id, level)
        for pass_spec in plan.passes
        for candidate_id in pass_spec.candidate_order
        for level in pass_spec.server_parallel_order
    ]


def _expected_quality_sequence(plan: CapacityPlan) -> list[str]:
    return [
        _quality_label(candidate.id, level)
        for candidate in plan.candidates
        for level in plan.server_parallel_levels
    ]


def _expected_run_labels(plan: CapacityPlan) -> set[str]:
    return {
        _run_label(pass_spec.id, candidate.id, level)
        for pass_spec in plan.passes
        for candidate in plan.candidates
        for level in plan.server_parallel_levels
    }


def _expected_quality_labels(plan: CapacityPlan) -> set[str]:
    return {
        _quality_label(candidate.id, level)
        for candidate in plan.candidates
        for level in plan.server_parallel_levels
    }


def _run_label(pass_id: str, candidate_id: str, server_parallel: int) -> str:
    return f"{pass_id}/{candidate_id}/p{server_parallel}"


def _quality_label(candidate_id: str, server_parallel: int) -> str:
    return f"{candidate_id}/p{server_parallel}"


def _parse_run_label(label: str) -> tuple[str, str, int]:
    parts = label.split("/")
    if len(parts) != 3 or not parts[2].startswith("p"):
        raise ValidationError(f"invalid capacity run label {label!r}")
    return parts[0], parts[1], _decimal(parts[2][1:], f"capacity run label {label!r}")


def _parse_quality_label(label: str) -> tuple[str, int]:
    parts = label.split("/")
    if len(parts) != 2 or not parts[1].startswith("p"):
        raise ValidationError(f"invalid capacity quality label {label!r}")
    return parts[0], _decimal(parts[1][1:], f"capacity quality label {label!r}")


def _labeled_paths(
    values: Sequence[tuple[str, Path]],
    expected: set[str],
    context: str,
) -> Mapping[str, Path]:
    result: dict[str, Path] = {}
    for label, path in values:
        if label in result:
            raise ValidationError(f"{context} label {label!r} must be unique")
        result[label] = Path(path)
    if set(result) != expected:
        missing = sorted(expected - set(result))
        unknown = sorted(set(result) - expected)
        detail: list[str] = []
        if missing:
            detail.append("missing: " + ", ".join(missing))
        if unknown:
            detail.append("unknown: " + ", ".join(unknown))
        raise ValidationError(f"{context} do not match the plan ({'; '.join(detail)})")
    return result


def _outcomes_sha256(outcomes: Sequence[tuple[str, bool, str | None]]) -> str:
    serialized = "\n".join(
        f"{case_id}\t{str(matched).lower()}\t{matched_answer or ''}"
        for case_id, matched, matched_answer in outcomes
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _optional_median(values: Sequence[Any]) -> float | None:
    if any(value is None for value in values):
        return None
    return float(statistics.median(float(value) for value in values))


def _relative_spread(values: Sequence[float]) -> float:
    low = min(values)
    high = max(values)
    median = statistics.median(values)
    if median == 0:
        return 0.0 if high == low else math.inf
    return (high - low) / median * 100.0


def _optional_relative_spread(values: Sequence[Any]) -> float | None:
    if any(value is None for value in values):
        return None
    return _relative_spread([float(value) for value in values])


def _pass_index(plan: CapacityPlan, pass_id: str) -> int:
    return next(index for index, pass_spec in enumerate(plan.passes) if pass_spec.id == pass_id)


def _candidate_index(plan: CapacityPlan, candidate_id: str) -> int:
    return next(
        index for index, candidate in enumerate(plan.candidates) if candidate.id == candidate_id
    )


def _option_index(argv: Sequence[str], option: str, context: str) -> int:
    indexes = [index for index, value in enumerate(argv) if value == option]
    if len(indexes) != 1 or indexes[0] + 1 >= len(argv):
        raise ValidationError(f"{context} must contain {option} exactly once with a value")
    return indexes[0]


def _option_integer(argv: Sequence[str], option: str, context: str) -> int:
    index = _option_index(argv, option, context)
    return _decimal(argv[index + 1], f"{context} {option}")


def _option_value(argv: Sequence[str], option: str, context: str) -> str:
    index = _option_index(argv, option, context)
    return _text(argv[index + 1], f"{context} {option}")


def _argv(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 128:
        raise ValidationError(f"{context} must contain from 1 to 128 arguments")
    return tuple(
        _text(item, f"{context}[{index}]", maximum=4096) for index, item in enumerate(value)
    )


def _levels(value: Any, context: str) -> tuple[int, ...]:
    levels = _integer_array(value, context)
    if levels != _ALLOWED_LEVELS:
        raise ValidationError(f"{context} must be exactly [1, 2, 4]")
    return levels


def _integer_array(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{context} must be a non-empty array")
    return tuple(
        _integer(item, f"{context}[{index}]", minimum=1, maximum=1024)
        for index, item in enumerate(value)
    )


def _text_array(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{context} must be a non-empty array")
    result = tuple(_text(item, f"{context}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ValidationError(f"{context} values must be unique")
    return result


def _text_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list):
        raise ValidationError(f"{context} must be an array")
    return [_text(item, f"{context}[{index}]") for index, item in enumerate(value)]


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{context} must be an object")
    return value


def _exact_fields(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(raw)
    missing = sorted(expected - actual)
    unknown = sorted(str(name) for name in actual - expected)
    details: list[str] = []
    if missing:
        details.append("missing fields: " + ", ".join(missing))
    if unknown:
        details.append("unknown fields: " + ", ".join(unknown))
    if details:
        raise ValidationError(f"{context} has " + "; ".join(details))


def _text(value: Any, context: str, *, maximum: int = _MAX_TEXT) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{context} must be a non-empty string")
    text = value.strip()
    if len(text) > maximum:
        raise ValidationError(f"{context} must contain at most {maximum} characters")
    return text


def _digest(value: Any, context: str) -> str:
    digest = _text(value, context, maximum=_SHA256_LENGTH)
    if len(digest) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValidationError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _git_revision(value: Any, context: str) -> str:
    revision = _text(value, context, maximum=40)
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValidationError(f"{context} must be a lowercase 40-character Git revision")
    return revision


def _integer(
    value: Any,
    context: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{context} must be an integer")
    if value < minimum or value > maximum:
        raise ValidationError(f"{context} must be from {minimum} to {maximum}")
    return value


def _decimal(value: str, context: str) -> int:
    if not value.isdecimal():
        raise ValidationError(f"{context} must end in a decimal integer")
    return int(value, 10)


def _finite(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{context} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{context} must be a finite number")
    if strictly_positive and number <= 0:
        raise ValidationError(f"{context} must be positive")
    if minimum is not None and number < minimum:
        raise ValidationError(f"{context} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValidationError(f"{context} must be at most {maximum}")
    return number
