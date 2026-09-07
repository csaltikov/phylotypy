import unittest

import numpy as np
from phylotypy import bootstrap


class TestBootstrap(unittest.TestCase):
    def setUp(self) -> None:
        self.ref_genera = np.array(["A;a;A", "A;a;B", "A;a;C", "A;b;A", "A;b;B", "A;b;C"])

    def test_split_taxa(self):
        observed = bootstrap.split_taxa_arr(self.ref_genera)
        expected = np.array([["A", "a", "A"],
                             ["A", "a", "B"],
                             ["A", "a", "C"],
                             ["A", "b", "A"],
                             ["A", "b", "B"],
                             ["A", "b", "C"]])
        self.assertTrue(np.array_equal(observed, expected))

    def test_bootstrap_consensus_confidence_is_percentage_non_default_num_bs(self):
        # num_bs=10 (not 100): unanimous support must report confidence=100,
        # not the raw vote count of 10.
        num_bs = 10
        classified_bs_kmers = np.zeros(num_bs, dtype=int)  # all votes -> "A;a;A"

        result = bootstrap.bootstrap_consensus(classified_bs_kmers, self.ref_genera)

        self.assertTrue(np.array_equal(result["confidence"], np.array([100, 100, 100])))

    def test_bootstrap_consensus_confidence_partial_support(self):
        num_bs = 10
        # 7 votes for genus 0 ("A;a;A"), 3 votes for genus 3 ("A;b;A"):
        # agree at level 0 ("A"), split below that.
        classified_bs_kmers = np.array([0] * 7 + [3] * 3)

        result = bootstrap.bootstrap_consensus(classified_bs_kmers, self.ref_genera)

        self.assertEqual(result["confidence"][0], 100)
        self.assertEqual(result["confidence"][1], 70)

    def test_bootstrap_consensus_batch_confidence_is_percentage_non_default_num_bs(self):
        num_bs = 10
        bs_res = np.zeros((1, num_bs), dtype=int)  # all votes -> "A;a;A"

        result = bootstrap.bootstrap_consensus_batch(bs_res, self.ref_genera)

        self.assertTrue(np.array_equal(result["confidence"], np.array([[100, 100, 100]])))

    def test_bootstrap_consensus_batch_confidence_partial_support(self):
        num_bs = 10
        bs_res = np.array([[0] * 7 + [3] * 3])

        result = bootstrap.bootstrap_consensus_batch(bs_res, self.ref_genera)

        self.assertEqual(result["confidence"][0, 0], 100)
        self.assertEqual(result["confidence"][0, 1], 70)


if __name__ == "__main__":
    unittest.main()
