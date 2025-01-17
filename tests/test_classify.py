import numpy as np
from pathlib import Path
import unittest
from phylotypy import kmers



class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.db_name = "rdp"
        self.mod_file = Path("model_raw.rbf")
        self.genera_file = Path("ref_genera.npy")

    def test_db_files(self):
        kmer_size = 3
        sequences = ["ATGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        genera_idx = [0, 1, 1]
        ref_genera = np.array(["A", "B", "B"])

        detect_list = kmers.detect_kmers_across_sequences_mp(sequences, kmer_size)
        priors = kmers.calc_word_specific_priors(detect_list, kmer_size)

        # (m(wi) + Pi) / (M + 1)

        conditional_prob = kmers.calc_genus_conditional_prob(detect_list,
                                                             genera_idx,
                                                             priors)
        conditional_prob.tofile(self.mod_file)

        # np.save(self.mod_file, kmers.genera_str_to_index(ref_genera))
        mod_shape = conditional_prob.shape

        db_ = kmers.KmerDB(conditional_prob=np.memmap(self.mod_file,
                                                      mode="c",
                                                      dtype=np.float16,
                                                      shape=mod_shape),
                           genera_idx=kmers.genera_str_to_index(ref_genera.tolist()),
                           genera_names=ref_genera
                           )
        self.assertEqual(db_.conditional_prob.shape, (4**kmer_size, 2))


if __name__ == '__main__':
    unittest.main()
