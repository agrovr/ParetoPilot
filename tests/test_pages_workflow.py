from __future__ import annotations

import json
from pathlib import Path
import textwrap
import unittest


REPOSITORY = Path(__file__).parents[1]
WORKFLOW_PATH = REPOSITORY / ".github" / "workflows" / "pages.yml"
CAPACITY_LOCK_PATH = REPOSITORY / "results" / "published" / "30144901854" / "evidence.json"


def _workflow() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _compact(value: str) -> str:
    return "".join(line.strip() for line in value.splitlines())


class PagesWorkflowContractTests(unittest.TestCase):
    def test_capacity_release_is_the_exact_committed_v14_asset(self) -> None:
        workflow = _workflow()
        compact = _compact(workflow).replace('"', "").replace("'", "")
        lock = json.loads(CAPACITY_LOCK_PATH.read_text(encoding="utf-8"))
        archive = lock["archive"]

        self.assertIn(
            "CAPACITY_LOCK: results/published/30144901854/evidence.json",
            workflow,
        )
        self.assertEqual(archive["release_tag"], "v1.4.0")
        self.assertEqual(
            archive["release_asset_name"],
            "paretopilot-v1.4.0-arm64-capacity-30144901854.zip",
        )
        for value in (
            archive["release_tag"],
            archive["release_asset_name"],
            archive["release_asset_url"],
            archive["release_url"],
            archive["sha256"],
            str(archive["size_bytes"]),
        ):
            with self.subTest(value=value):
                self.assertIn(str(value), compact)

        capacity_step = workflow[
            workflow.index(
                "- name: Download, replay, and verify locked capacity evidence"
            ) : workflow.index("- name: Rebuild, compare, and present every v1.1 output")
        ]
        self.assertIn('Request(\n              archive["release_asset_url"]', capacity_step)
        self.assertIn(
            "PYTHONPATH=src python -m paretopilot replay-capacity",
            capacity_step,
        )
        self.assertIn("output/pages/capacity-evidence", capacity_step)
        self.assertIn("--output-dir output/pages/capacity-replayed", capacity_step)

    def test_capacity_proof_is_published_and_wired_into_the_showcase(self) -> None:
        workflow = _workflow()
        build_section = workflow[
            workflow.index(
                "- name: Rebuild, compare, and present every v1.1 output"
            ) : workflow.index("- name: Upload Pages artifact")
        ]
        expected_copies = (
            (
                "output/pages/capacity-evidence/capacity-study.json",
                "_site/evidence/capacity-study.json",
            ),
            (
                "output/pages/capacity-evidence/capacity-receipt.md",
                "_site/evidence/capacity-receipt.md",
            ),
        )
        for source, destination in expected_copies:
            with self.subTest(destination=destination):
                self.assertIn(source, build_section)
                self.assertIn(destination, build_section)
                self.assertLess(build_section.index(source), build_section.index(destination))

        showcase = build_section[
            build_section.index("PYTHONPATH=src python -m paretopilot showcase-v11") :
        ]
        expected_flags = {
            "--capacity-study": "output/pages/capacity-evidence/capacity-study.json",
            "--capacity-evidence-lock": '"$CAPACITY_LOCK"',
            "--capacity-study-href": "evidence/capacity-study.json",
            "--capacity-receipt-href": "evidence/capacity-receipt.md",
        }
        for flag, value in expected_flags.items():
            with self.subTest(flag=flag):
                self.assertEqual(showcase.count(f"{flag} "), 1)
                self.assertIn(f"{flag} {value}", showcase)

        self.assertIn('test "$(find _site -type f | wc -l)" -eq 5', build_section)
        self.assertNotIn('test "$(find _site -type f | wc -l)" -eq 3', build_section)

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

        self.assertEqual(len(programs), 4)
        for number, program in enumerate(programs, start=1):
            with self.subTest(program=number):
                compile(program, f"{WORKFLOW_PATH.name}:python-heredoc-{number}", "exec")


if __name__ == "__main__":
    unittest.main()
