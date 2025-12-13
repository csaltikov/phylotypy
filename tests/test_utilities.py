from pathlib import Path
import unittest

from phylotypy import utilities

class TestTrainingData(unittest.TestCase):
    def setUp(self) -> None:
        self.taxaname = "Shewanella"
        self.taxanames = ["Shewanella", "Escherichia"]
        self.badtaxa = ["Shewanella", "Bad", "Escherichia"]

    def test_get_taxa_ids(self):
        observed = utilities.get_taxa_ids(self.taxaname)
        self.assertIsInstance(observed, dict)
        uids_observed = observed.get("result", None).get("uids", [])
        self.assertEqual(['22'], uids_observed)

        observed = utilities.get_taxa_ids(self.taxanames)
        self.assertIsInstance(observed, dict)
        # uids_observed = observed.get("result", None).get("uids", [])
        # self.assertEqual(['561', '22'], uids_observed)

        observed = utilities.get_taxa_ids(self.badtaxa)
        self.assertIsInstance(observed, dict)
        # uids_observed = observed.get("result", None).get("uids", [])
        # self.assertEqual(['561', '22'], uids_observed)


if __name__ == "__main__":
    print("hello")
