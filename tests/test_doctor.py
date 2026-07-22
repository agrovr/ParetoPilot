from __future__ import annotations

import unittest
from unittest.mock import patch

from paretopilot import doctor


class EnvironmentDoctorTests(unittest.TestCase):
    def test_arm64_host_is_eligible_for_benchmark_evidence(self) -> None:
        with (
            patch.object(doctor.platform, "machine", return_value="AARCH64"),
            patch.object(doctor.platform, "processor", return_value="Neoverse-N1"),
            patch.object(doctor.platform, "platform", return_value="Linux-6.8-aarch64"),
            patch.object(doctor.platform, "system", return_value="Linux"),
            patch.object(doctor.platform, "release", return_value="6.8.0"),
            patch.object(doctor.platform, "python_version", return_value="3.12.9"),
        ):
            report = doctor.inspect_environment()

        self.assertEqual(report.machine_architecture, "AARCH64")
        self.assertEqual(report.processor, "Neoverse-N1")
        self.assertEqual(report.platform, "Linux-6.8-aarch64")
        self.assertEqual(report.operating_system, "Linux")
        self.assertEqual(report.os_release, "6.8.0")
        self.assertEqual(report.python_version, "3.12.9")
        self.assertTrue(report.is_arm64)
        self.assertTrue(report.architecture_eligible)
        self.assertTrue(report.evidence_eligible)
        self.assertEqual(report.warnings, ())

    def test_arm64_alias_is_accepted(self) -> None:
        with (
            patch.object(doctor.platform, "machine", return_value="arm64"),
            patch.object(doctor.platform, "system", return_value="Linux"),
        ):
            report = doctor.inspect_environment()

        self.assertTrue(report.is_arm64)
        self.assertTrue(report.architecture_eligible)
        self.assertTrue(report.evidence_eligible)

    def test_arm64_non_linux_host_is_compatible_but_not_evidence_eligible(self) -> None:
        for operating_system in ("Windows", "Darwin"):
            with self.subTest(operating_system=operating_system):
                with (
                    patch.object(doctor.platform, "machine", return_value="ARM64"),
                    patch.object(doctor.platform, "system", return_value=operating_system),
                ):
                    report = doctor.inspect_environment()

                self.assertTrue(report.is_arm64)
                self.assertTrue(report.architecture_eligible)
                self.assertFalse(report.evidence_eligible)
                self.assertEqual(len(report.warnings), 1)
                self.assertIn("requires native Arm64 Linux", report.warnings[0])
                self.assertIn("compatibility-test-only", report.warnings[0])

    def test_x86_host_is_explicitly_smoke_only(self) -> None:
        with (
            patch.object(doctor.platform, "machine", return_value="AMD64"),
            patch.object(doctor.platform, "processor", return_value="x86_64 Family 6"),
            patch.object(doctor.platform, "platform", return_value="Windows-11-10.0.26100"),
            patch.object(doctor.platform, "system", return_value="Windows"),
            patch.object(doctor.platform, "release", return_value="11"),
            patch.object(doctor.platform, "python_version", return_value="3.12.9"),
        ):
            report = doctor.inspect_environment()

        self.assertFalse(report.is_arm64)
        self.assertFalse(report.architecture_eligible)
        self.assertFalse(report.evidence_eligible)
        self.assertEqual(len(report.warnings), 1)
        self.assertIn("smoke-test-only", report.warnings[0])
        self.assertIn("not eligible for benchmark evidence", report.warnings[0])


if __name__ == "__main__":
    unittest.main()
