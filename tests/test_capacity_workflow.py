from __future__ import annotations

import hashlib
from pathlib import Path
import re
import textwrap
import unittest

from paretopilot.capacity_eval import load_capacity_plan


REPOSITORY = Path(__file__).parents[1]
WORKFLOW_PATH = REPOSITORY / ".github" / "workflows" / "capacity-study-arm64.yml"


def _workflow() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _environment_value(workflow: str, name: str) -> str:
    match = re.search(rf"^  {re.escape(name)}:\s+([^\r\n]+)$", workflow, re.MULTILINE)
    if match is None:
        raise AssertionError(f"workflow environment value is missing: {name}")
    return match.group(1).strip().strip('"')


class CapacityWorkflowContractTests(unittest.TestCase):
    def test_repository_input_pins_match_exact_bytes(self) -> None:
        workflow = _workflow()
        inputs = (
            ("CAPACITY_PLAN", "CAPACITY_PLAN_SHA256"),
            ("LOAD_PLAN", "LOAD_PLAN_SHA256"),
            ("EVAL_SUITE", "EVAL_SUITE_SHA256"),
            ("EVIDENCE_LOCK", "EVIDENCE_LOCK_SHA256"),
        )

        for path_name, digest_name in inputs:
            with self.subTest(input=path_name):
                source = REPOSITORY / _environment_value(workflow, path_name)
                expected = _environment_value(workflow, digest_name)
                self.assertTrue(source.is_file())
                self.assertEqual(
                    hashlib.sha256(source.read_bytes()).hexdigest(),
                    expected,
                )

    def test_performance_and_quality_commands_follow_the_plan_exactly(self) -> None:
        workflow = _workflow()
        plan = load_capacity_plan(REPOSITORY / "configs" / "capacity.arm64.json")

        performance_pattern = re.compile(
            r"^\s*run_configuration\s+"
            r"(forward|reverse)\s+"
            r"([a-z0-9-]+)\s+"
            r"([a-z0-9-]+)\s+"
            r"(\"\$[A-Z0-9_]+\")\s+"
            r"(\d+)\s+(\d+)\s+"
            r"([124])\s+(\d+)\s+([124](?:,[124]){2})$",
            re.MULTILINE,
        )
        performance_matches = performance_pattern.findall(workflow)
        candidate_runtime = {
            "q8-generic": ("generic", '"$Q8_FILENAME"', 512, 128),
            "q4-kleidiai-tuned": ("kleidiai", '"$Q4_FILENAME"', 512, 512),
        }
        observed_performance = [
            (
                pass_id,
                candidate_id,
                build,
                model,
                int(batch),
                int(ubatch),
                int(parallel),
                execution_order,
            )
            for (
                pass_id,
                candidate_id,
                build,
                model,
                batch,
                ubatch,
                parallel,
                _port,
                execution_order,
            ) in performance_matches
        ]
        expected_performance = [
            (
                pass_spec.id,
                candidate_id,
                *candidate_runtime[candidate_id],
                parallel,
                ",".join(str(level) for level in pass_spec.client_concurrency_order),
            )
            for pass_spec in plan.passes
            for candidate_id in pass_spec.candidate_order
            for parallel in pass_spec.server_parallel_order
        ]
        self.assertEqual(observed_performance, expected_performance)

        quality_pattern = re.compile(
            r"^\s*run_quality_configuration\s+"
            r"([a-z0-9-]+)\s+"
            r"([a-z0-9-]+)\s+"
            r"(\"\$[A-Z0-9_]+\")\s+"
            r"(\d+)\s+(\d+)\s+([124])\s+(\d+)$",
            re.MULTILINE,
        )
        quality_matches = quality_pattern.findall(workflow)
        observed_quality = [
            (
                candidate_id,
                build,
                model,
                int(batch),
                int(ubatch),
                int(parallel),
            )
            for candidate_id, build, model, batch, ubatch, parallel, _port in quality_matches
        ]
        expected_quality = [
            (candidate.id, *candidate_runtime[candidate.id], parallel)
            for candidate in plan.candidates
            for parallel in plan.server_parallel_levels
        ]
        self.assertEqual(observed_quality, expected_quality)

        ports = [int(match[7]) for match in performance_matches]
        ports.extend(int(match[6]) for match in quality_matches)
        self.assertEqual(len(ports), len(set(ports)))
        self.assertTrue(all(1024 <= port <= 65535 for port in ports))

    def test_every_embedded_python_program_compiles(self) -> None:
        lines = _workflow().splitlines()
        programs: list[str] = []
        index = 0
        while index < len(lines):
            if "<<'PY'" not in lines[index]:
                index += 1
                continue
            body: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != "PY":
                body.append(lines[index])
                index += 1
            self.assertLess(index, len(lines), "unterminated Python heredoc")
            programs.append(textwrap.dedent("\n".join(body)) + "\n")
            index += 1

        self.assertEqual(len(programs), 11)
        for number, program in enumerate(programs, start=1):
            with self.subTest(program=number):
                compile(program, f"{WORKFLOW_PATH.name}:python-heredoc-{number}", "exec")

    def test_llama_server_target_is_enabled_by_the_pinned_build_contract(self) -> None:
        workflow = _workflow()
        build_section = workflow[
            workflow.index("- name: Build pinned generic and KleidiAI servers") : workflow.index(
                "- name: Download and verify pinned Q8 and Q4 models"
            )
        ]

        self.assertIn("-DLLAMA_BUILD_SERVER=ON", build_section)
        self.assertIn("-DLLAMA_BUILD_TOOLS=ON", build_section)
        self.assertNotIn("-DLLAMA_BUILD_TOOLS=OFF", build_section)
        self.assertIn("--target llama-server", build_section)

    def test_manifest_failure_and_success_contracts_are_truthful(self) -> None:
        workflow = _workflow()
        manifest_section = workflow[
            workflow.index("- name: Build capacity source manifest") : workflow.index(
                "- name: Measure the slot and client matrix in mirrored order"
            )
        ]
        self.assertIn('"schema_version": "1.1"', manifest_section)

        failure_section = workflow[
            workflow.index("- name: Refresh failed status and checksums") : workflow.index(
                "- name: Upload incomplete capacity diagnostics"
            )
        ]
        self.assertIn('Path("artifacts/current-stage.txt")', failure_section)
        self.assertIn('"failed_stage": failed_stage', failure_section)

        success_section = workflow[
            workflow.index("- name: Finalize checksummed capacity bundle") : workflow.index(
                "- name: Upload validated capacity study"
            )
        ]
        self.assertIn('"eligible_cell_counts": eligible_cell_counts', success_section)
        self.assertIn('"selected_operating_points": selected_operating_points', success_section)
        self.assertNotIn("integrity gates passed", success_section)
        self.assertIn("each cell's checks were recorded", success_section)

    def test_incomplete_diagnostics_cover_cancellation_and_preserve_raw_quality(self) -> None:
        workflow = _workflow()
        recovery_section = workflow[workflow.index("- name: Refresh failed status and checksums") :]

        self.assertEqual(recovery_section.count("${{ failure() || cancelled() }}"), 2)
        self.assertIn(
            'local raw_evaluation="$directory/quality-evaluation.raw.json"',
            workflow,
        )
        self.assertNotIn('local raw_evaluation="$RUNNER_TEMP/quality-', workflow)
        self.assertIn(
            "the Arm64 capacity study did not complete",
            recovery_section,
        )


if __name__ == "__main__":
    unittest.main()
