from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPOSITORY = Path(__file__).parents[1]
WORKFLOW_PATH = REPOSITORY / ".github" / "workflows" / "pages.yml"
CAPACITY_LOCK_PATH = REPOSITORY / "results" / "published" / "30144901854" / "evidence.json"
VALID_INDEX = """\
<html><body>
<main id="main-content">
<a href="#main-content">Self</a>
<a href="evidence/report-v1.1.html#report">Report</a>
<a href="evidence/optimization-receipt.md">Optimization receipt</a>
<a href="evidence/capacity-study.json">Capacity study</a>
<a href="evidence/capacity-receipt.md">Capacity receipt</a>
<a href="https://github.com/agrovr/ParetoPilot">External</a>
<link rel="icon" href="data:,">
</main>
</body></html>
"""


def _workflow() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _compact(value: str) -> str:
    return "".join(line.strip() for line in value.splitlines())


def _embedded_python_programs() -> list[str]:
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
        if index >= len(lines):
            raise AssertionError("unterminated Python heredoc")
        programs.append(textwrap.dedent("\n".join(body)) + "\n")
        index += 1
    return programs


def _run_stage_validator(
    program: str,
    index_html: str = VALID_INDEX,
    *,
    extra_file: bool = False,
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        site = root / "_site"
        evidence = site / "evidence"
        evidence.mkdir(parents=True)

        (site / "index.html").write_text(index_html, encoding="utf-8")
        (evidence / "report-v1.1.html").write_text(
            '<main id="report"><a href="../index.html#main-content">Back</a></main>',
            encoding="utf-8",
        )
        (evidence / "optimization-receipt.md").write_text("receipt\n", encoding="utf-8")
        (evidence / "capacity-study.json").write_text("{}\n", encoding="utf-8")
        (evidence / "capacity-receipt.md").write_text("capacity\n", encoding="utf-8")
        if extra_file:
            (site / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")

        return subprocess.run(
            [sys.executable, "-c", program],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )


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
            workflow.index("- name: Verify and replay the v1.4 capacity release") : workflow.index(
                "- name: Build the published results page"
            )
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
            workflow.index("- name: Build the published results page") : workflow.index(
                "- name: Upload Pages artifact"
            )
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

        self.assertIn("Pages manifest mismatch", build_section)
        self.assertIn("Pages link validation failed", build_section)
        self.assertNotIn('test "$(find _site -type f | wc -l)"', build_section)

    def test_pages_stage_validator_checks_manifest_links_and_fragments(self) -> None:
        validators = [
            program
            for program in _embedded_python_programs()
            if "Pages manifest mismatch" in program
        ]
        self.assertEqual(len(validators), 1)
        validator = validators[0]

        valid = _run_stage_validator(validator)
        self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)

        extra = _run_stage_validator(validator, extra_file=True)
        self.assertNotEqual(extra.returncode, 0)
        self.assertIn("Pages manifest mismatch", extra.stderr)

        missing_file = _run_stage_validator(
            validator,
            VALID_INDEX.replace("capacity-study.json", "missing.json"),
        )
        self.assertNotEqual(missing_file.returncode, 0)
        self.assertIn("missing local target", missing_file.stderr)

        missing_fragment = _run_stage_validator(
            validator,
            VALID_INDEX.replace("#report", "#missing"),
        )
        self.assertNotEqual(missing_fragment.returncode, 0)
        self.assertIn("missing fragment target", missing_fragment.stderr)

    def test_every_embedded_python_program_compiles(self) -> None:
        programs = _embedded_python_programs()

        self.assertEqual(len(programs), 5)
        for number, program in enumerate(programs, start=1):
            with self.subTest(program=number):
                compile(program, f"{WORKFLOW_PATH.name}:python-heredoc-{number}", "exec")


if __name__ == "__main__":
    unittest.main()
