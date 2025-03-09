import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, verbose=1)

import unittest

import numpy as np
import pandas as pd
from phylotypy import kmers, conditional_prob


class TestGetKmers(unittest.TestCase):
    def setUp(self) -> None:
        self.kmers = kmers
        self.sequence = "ATGCGCTAGTAGCATGC"
        self.expected_cond_prods = {
            25: (np.array([1, 2]) + 0.875) / (np.array([1, 2]) + 1),  # [0.9375, 0.95833333]
            28: (np.array([1, 0]) + 0.375) / (np.array([1, 2]) + 1),  # [0.6875, 0.125 ]
            29: (np.array([0, 2]) + 0.625) / (np.array([1, 2]) + 1),  # [0.3125, 0.875 ]
            62: (np.array([0, 0]) + 0.125) / (np.array([1, 2]) + 1)  # [0.0625, 0.04166667]
        }
        self.kmer_size = 3
        self.sequences = ["ATGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        self.genera = ["A", "B", "B"]
        self.sequences_df = pd.DataFrame(dict(id=self.genera, sequence=self.sequences))

    def test_make_kmer_database(self):
        observed_db = conditional_prob.make_kmers_database(self.sequences_df, kmer_size=self.kmer_size)
        expected_id = np.array([0,1,1])
        self.assertTrue(np.array_equal(observed_db[:,0], expected_id))

    def test_calc_priors(self):
        detect_arr = conditional_prob.make_kmers_database(sequences_db=self.sequences_df, kmer_size=self.kmer_size)
        observed_db = conditional_prob.calc_priors(np.array(detect_arr), kmer_size=self.kmer_size)

        self.assertEqual(observed_db[25], 0.875)

    def test_conditional_prob(self):
        observed = conditional_prob.make_kmers_database(self.sequences_df, kmer_size=self.kmer_size)
        expected_idx = np.array([0,1,1])
        self.assertTrue(np.array_equal(observed[:,0], expected_idx))

    def test_build_database(self):
        observed_arr = conditional_prob.build_database(self.sequences_df, kmer_size=self.kmer_size)
        observed = np.exp(observed_arr.conditional_prob[25]).astype(np.float16)
        expected = self.expected_cond_prods[25].astype(np.float16)
        self.assertTrue(np.array_equal(observed, expected))


if __name__ == "__main__":
    unittest.main()
