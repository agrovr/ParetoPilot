"""Inspect whether the current host can produce Arm64 benchmark evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import platform
from typing import Any


ARM64_MACHINE_NAMES = frozenset({"aarch64", "arm64"})


@dataclass(frozen=True)
class EnvironmentReport:
    """A portable snapshot of the host used to run ParetoPilot."""

    machine_architecture: str
    processor: str
    platform: str
    operating_system: str
    os_release: str
    python_version: str
    is_arm64: bool
    evidence_eligible: bool
    warnings: tuple[str, ...]
    architecture_eligible: bool = field(init=False)

    def __post_init__(self) -> None:
        """Keep the legacy Arm64 flag and explicit eligibility flag consistent."""

        object.__setattr__(self, "architecture_eligible", self.is_arm64)

    def to_mapping(self) -> dict[str, Any]:
        """Return a serialization-friendly representation of the report."""

        return asdict(self)


def inspect_environment() -> EnvironmentReport:
    """Collect host facts and classify whether they are valid Arm64 evidence."""

    machine_architecture = _value_or_unknown(platform.machine())
    operating_system = _value_or_unknown(platform.system())
    is_arm64 = machine_architecture.casefold() in ARM64_MACHINE_NAMES
    architecture_eligible = is_arm64
    evidence_eligible = architecture_eligible and operating_system.casefold() == "linux"

    warnings: tuple[str, ...] = ()
    if not architecture_eligible:
        warnings = (
            "Non-Arm64 architecture detected; this environment is smoke-test-only "
            "and is not eligible for benchmark evidence.",
        )
    elif not evidence_eligible:
        warnings = (
            f"Arm64 architecture detected on {operating_system}, but benchmark evidence "
            "requires native Arm64 Linux; this environment is compatibility-test-only.",
        )

    return EnvironmentReport(
        machine_architecture=machine_architecture,
        processor=_value_or_unknown(platform.processor()),
        platform=_value_or_unknown(platform.platform()),
        operating_system=operating_system,
        os_release=_value_or_unknown(platform.release()),
        python_version=_value_or_unknown(platform.python_version()),
        is_arm64=is_arm64,
        evidence_eligible=evidence_eligible,
        warnings=warnings,
    )


def _value_or_unknown(value: str) -> str:
    """Normalize platform strings without pretending missing facts were detected."""

    normalized = value.strip()
    return normalized if normalized else "unknown"
