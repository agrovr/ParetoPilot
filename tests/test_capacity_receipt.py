from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from paretopilot.capacity_eval import validate_capacity_study
from paretopilot.capacity_receipt import render_capacity_receipt
from paretopilot.io import load_json_object, write_json
from test_capacity_eval import CapacityFixture


class CapacityReceiptTests(unittest.TestCase):
    def test_receipt_is_deterministic_complete_and_keeps_canonical_scope(self) -> None:
        with TemporaryDirectory() as directory:
            study = CapacityFixture(Path(directory)).assemble()

            first = render_capacity_receipt(study)
            second = render_capacity_receipt(study)

            self.assertEqual(first, second)
            self.assertTrue(first.startswith("# ParetoPilot Arm64 Capacity Receipt\n"))
            self.assertIn("SUPPLEMENTARY EVIDENCE · CANONICAL v1.1 UNCHANGED", first)
            self.assertIn("## Capacity envelope — Q8 generic reference", first)
            self.assertIn(
                "## Capacity envelope — Q4 with KleidiAI and 512-token micro-batch",
                first,
            )
            self.assertIn("P4 / C2 is the best observed passing point", first)
            self.assertIn("one_or_more_passes_failed_load_slo", first)
            self.assertIn("paretopilot assemble-capacity", first)
            self.assertIn("Canonical outputs modified: No", first)
            self.assertNotIn("production capacity", first.casefold())
            self.assertNotIn("optimal", first.casefold())

    def test_canonical_json_round_trip_remains_valid_and_renderable(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            study = CapacityFixture(root).assemble()
            study_path = root / "capacity-study.json"

            write_json(study_path, study)
            reloaded = load_json_object(study_path)

            validate_capacity_study(reloaded)
            receipt = render_capacity_receipt(reloaded)
            self.assertTrue(receipt.startswith("# ParetoPilot Arm64 Capacity Receipt\n"))


if __name__ == "__main__":
    unittest.main()
