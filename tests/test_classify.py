import numpy as np
from pathlib import Path
import unittest

from phylotypy import kmers, read_fasta
from phylotypy import classifier


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.current_dir = Path(__file__).parent
        self.fasta_dir = self.current_dir / 'test_fasta'
        self.db_name = "rdp"
        self.mod_file = Path("model_raw.rbf")
        self.genera_file = Path("ref_genera.npy")
        self.test_fasta = self.fasta_dir / "test_fasta.fa"
        self.test_ref = read_fasta.read_taxa_fasta(self.test_fasta)

    def test_db_files(self):
        kmer_size = 3
        sequences = ["ATGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        genera_idx = [0, 1, 1]
        ref_genera = np.array(["A", "B", "B"])

        detect_list = kmers.detect_kmers_across_sequences_mp(sequences, kmer_size)
        priors = kmers.calc_word_specific_priors(detect_list, kmer_size)

        # (m(wi) + Pi) / (M + 1)

        conditional_prob = kmers.calc_genus_conditional_prob_mp(detect_list,
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

    def test_make_classifier(self):
        observed = classifier.make_classifier(self.test_ref)

        n_genera_exp = 3
        n_genera_obs = len(observed.genera_names)
        self.assertEqual(n_genera_exp, n_genera_obs)

    def test_classify_sequence(self):
        database = classifier.make_classifier(self.test_ref)

        res = classifier.classify_sequences(self.test_ref, database)
        observed = res.iloc[0, 1]
        print(observed)
        expected_res = "Bacteria(100);Actinomycetota(100);Actinobacteria(100);Mycobacteriales(100);Mycobacteriaceae(100);Mycobacterium(100)"

        self.assertEqual(observed, expected_res)


if __name__ == '__main__':
    unittest.main()
