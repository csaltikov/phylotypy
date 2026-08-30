import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import unittest
import numpy as np
import pandas as pd
from phylotypy import kmers, conditional_prob


class TestGetKmers(unittest.TestCase):
    def setUp(self) -> None:
        self.kmers = kmers
        self.sequence = "ATGCGCTAGTAGCATGC"
        self.kmer_size = 3
        self.sequences = ["ATGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        self.genera = ["A", "B", "B"]
        self.sequences_df = pd.DataFrame(dict(id=self.genera, sequence=self.sequences))

    def test_make_kmer_database(self):
        observed_idx, observed_kmers = conditional_prob.seq_to_kmers_database(self.sequences_df, kmer_size=self.kmer_size)
        expected_idx = [0,1,1]
        self.assertEqual(observed_kmers[:,0].tolist(), expected_idx)

    def test_calc_priors(self):
        _, observed_kmers = conditional_prob.seq_to_kmers_database(sequences_db=self.sequences_df, kmer_size=self.kmer_size)
        observed_db = conditional_prob.calc_priors(np.array(observed_kmers), kmer_size=self.kmer_size)
        self.assertEqual(observed_db[25], 0.875)

    def test_conditional_prob(self):
        observed_idx, observed_kmers = conditional_prob.seq_to_kmers_database(self.sequences_df, kmer_size=self.kmer_size)
        expected_idx = np.array([0,1,1])
        self.assertTrue(np.array_equal(observed_kmers[:,0], expected_idx))

if __name__ == "__main__":
    unittest.main()
