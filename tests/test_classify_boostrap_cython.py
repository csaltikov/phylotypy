import unittest

import numpy as np
import pandas as pd

from phylotypy import kmers, bootstrap, conditional_prob
from phylotypy.classify_bootstraps import classify_bootstraps_cython


class TestClassifyBootstrapCy(unittest.TestCase):
    def setUp(self):
        self.kmer_size = 3
        self.sequences = ["ATGCGCTA", "GTCTATTC", "GTCTATTC"]
        self.ref_genera = ["A", "B", "B"]
        self.sequences_df = pd.DataFrame({"id": self.ref_genera,
                                     "sequence": self.sequences})

    def test_classify_bs_cython(self):
        test_database = kmers.build_kmer_database(self.sequences,
                                                  self.ref_genera,
                                                  kmer_size=self.kmer_size)

        genera_idx_test, detected_kmers_test = conditional_prob.seq_to_kmers_database(self.sequences_df,
                                                                                      kmer_size=self.kmer_size)
        # ref A
        seq_kmer = detected_kmers_test[0, 1:]

        bootstrapped = bootstrap.bootstrap(seq_kmer, divider=3)
        classified_kmers = classify_bootstraps_cython(bootstrapped, test_database.conditional_prob)

        observed = np.argmax(np.bincount(classified_kmers))
        expected = 0

        self.assertEqual( self.ref_genera[observed],  self.ref_genera[expected])


if __name__ == "__main__":
    unittest.main()
