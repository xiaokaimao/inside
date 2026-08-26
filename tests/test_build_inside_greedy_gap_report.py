from __future__ import annotations

import unittest

from experiments.build_inside_greedy_gap_report import build_artifact


class InsideGreedyGapReportTests(unittest.TestCase):
    def test_artifact_has_bounded_canonical_datasets_and_required_sections(
        self,
    ) -> None:
        artifact = build_artifact()
        manifest = artifact["manifest"]
        snapshot = artifact["snapshot"]

        self.assertEqual(manifest["surface"], "report")
        self.assertEqual(
            manifest["blocks"][0]["body"],
            f"# {manifest['title']}",
        )
        self.assertGreaterEqual(len(manifest["charts"]), 1)
        self.assertEqual(snapshot["status"], "ready")
        self.assertEqual(len(snapshot["datasets"]["formal_gap"]), 15)
        self.assertEqual(
            len(snapshot["datasets"]["synthetic_results"]), 6
        )
        self.assertEqual(
            len(snapshot["datasets"]["ablation_effects"]), 12
        )
        self.assertTrue(
            all(
                isinstance(rows, list) and len(rows) <= 2_000
                for rows in snapshot["datasets"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
