import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "dca_threshold"


class DCAReleaseTests(unittest.TestCase):
    def test_protocol_scope_is_explicit(self):
        protocol = json.loads(
            (ROOT / "provenance" / "dca_threshold_protocol.json").read_text()
        )
        self.assertEqual(protocol["release"], "v1.3.0")
        self.assertEqual(protocol["external_episodes"], 240)
        self.assertFalse(protocol["clinical_deployment_validated"])
        self.assertTrue(protocol["query_labels_read_after_threshold_lock"])

    def test_summary_exports_have_expected_rows(self):
        with (RESULTS / "dca_summary.csv").open(newline="") as stream:
            dca = list(csv.DictReader(stream))
        with (RESULTS / "locked_threshold_summary.csv").open(newline="") as stream:
            thresholds = list(csv.DictReader(stream))
        self.assertGreater(len(dca), 0)
        self.assertGreater(len(thresholds), 0)
        self.assertTrue({"NB_mean", "standardized_NB_mean"} <= set(dca[0]))
        self.assertTrue({"specificity_mean", "sensitivity_mean"} <= set(thresholds[0]))


if __name__ == "__main__":
    unittest.main()
