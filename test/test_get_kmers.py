import sys
import unittest

import numpy as np
import kmers

sys.path.append('../')


class TestGetKmers(unittest.TestCase):
    def setUp(self) -> None:
        self.kmers = kmers
        self.sequence = "ATGCGCTAGTAGCATGC"

    def test_get_all_kmers(self):
        """Test that can extract all possible 8-mers from a sequence"""
        all_kmers = kmers.get_all_kmers(self.sequence, kmer_size=8)

        expected_kmers = [
            "ATGCGCTA",
            "TGCGCTAG",
            "GCGCTAGT",
            "CGCTAGTA",
            "GCTAGTAG",
            "CTAGTAGC",
            "TAGTAGCA",
            "AGTAGCAT",
            "GTAGCATG",
            "TAGCATGC"
        ]

        self.assertEqual(all_kmers, expected_kmers)

    # def test_get_kmers(self):
    #     """Test that can extract specific kmer from a starting position and size"""
    #     kmer = get_kmers.get_kmer(self.sequence, 0, kmer_size=8)
    #     self.assertEqual(kmer, "ATGCGCTA", msg="Kmer is equal")

    def test_seq_to_base4(self):
        """Test that can conversion works between DNA sequence and quarternary"""
        expected = "03212130230210321"
        base4_seq = kmers.seq_to_base4(self.sequence)
        self.assertEqual(base4_seq, expected, msg="base4_seq is not equal to expected")

        seq_with_R = "ATGCGCTRGTAGCATGC"
        expected = "0321213N230210321"
        base4_seq = kmers.seq_to_base4(seq_with_R)
        self.assertEqual(base4_seq, expected, msg="base4_seq is not equal to expected")

        seq_lower = "ATGCGCTRGTAGCATGC".lower()
        expected = "0321213N230210321"
        base4_seq = kmers.seq_to_base4(seq_lower)
        self.assertEqual(base4_seq, expected, msg="base4_seq is not equal to expected")

    def test_base4_to_index(self):
        """Test that can generate base10 values from kmers in base4"""
        x = ["0000"]
        expect = [0]
        actual = kmers.base4_to_index(x)
        self.assertEqual(expect, actual,
                         msg=f"base4_to_index concersion of {x} is not equal to expected {expect}")

        x = ["1000"]
        expect = [64]
        actual = kmers.base4_to_index(x)
        self.assertEqual(expect, actual)

        x = ["0123"]
        expect = [27]
        actual = kmers.base4_to_index(x)
        self.assertEqual(expect, actual)

        x = ["0123", "1000", "0000"]
        expect = [27, 64, 0]
        actual = kmers.base4_to_index(x)
        self.assertTrue(np.array_equal(expect, actual))

    def test_detect_kmers(self):
        """Accurately detect kmers from a sequence"""
        sequence = "ATGCGCTAGTAGCATGC"
        kmer = kmers.get_all_kmers(kmers.seq_to_base4(sequence))
        indices = kmers.base4_to_index(kmer)

        detected = kmers.detect_kmers(sequence)

        self.assertEqual(len(detected), len(indices))
        self.assertTrue(np.array_equal(indices, detected))

        sequence = "ATGCGCTAGTAGCATGCN"
        kmer = kmers.get_all_kmers(kmers.seq_to_base4(sequence), kmer_size=7)
        indices = kmers.base4_to_index(kmer)

        detected = kmers.detect_kmers(sequence, kmer_size=7)
        self.assertEqual(len(detected), len(indices))
        self.assertTrue(np.array_equal(indices, detected))

    def test_get_kmers_from_across_sequences(self):
        """Test Accurately detect kmers across multiple sequences"""
        kmer_size = 3
        sequences = ["ATGCGCTA", "ATGCGCTC"]
        base4_sequences = kmers.seq_to_base4(sequences)
        # expected = np.zeros(2 * (4**kmer_size)).reshape(4**kmer_size, 2)

        expected = [None] * 2

        expected[0] = kmers.base4_to_index(kmers.get_all_kmers(base4_sequences[0], kmer_size))
        expected[1] = kmers.base4_to_index(kmers.get_all_kmers(base4_sequences[1], kmer_size))

        detect_list = kmers.detect_kmers_across_sequences(sequences, kmer_size)

        self.assertTrue(np.array_equal(detect_list, expected))

    def test_calc_word_specific_priors(self):
        """Test Calcuate word specific priors"""
        kmer_size = 3
        sequences = ["ATGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        detect_list = kmers.detect_kmers_across_sequences(sequences, kmer_size)
        detect_matrix = np.array(detect_list).T
        self.assertEqual((6, 3), detect_matrix.shape)

        # 26 - all 3 = (3+0.5) / (1 + 3) =0.875
        # 29 - only 1 = 0.375
        # 30 - only 2 and 3 = 0.625
        # 64 - none = 0.125

        # expected = expected.reshape(-1, 1)
        priors = kmers.calc_word_specific_priors(detect_list, kmer_size)

        self.assertEqual(priors[25], 0.875)
        self.assertEqual(priors[28], 0.375)
        self.assertEqual(priors[29], 0.625)
        self.assertEqual(priors[62], 0.125)

    def test_calc_genus_conditional_prob(self):
        """Calculate genus-specific conditional probabilities"""
        kmer_size = 3
        sequences = ["ATGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        genera = [0, 1, 1]

        detect_list = kmers.detect_kmers_across_sequences(sequences, kmer_size)
        priors = kmers.calc_word_specific_priors(detect_list, kmer_size)

        # (m(wi) + Pi) / (M + 1)

        pos25 = (np.array([1, 2]) + 0.875) / (np.array([1, 2]) + 1)
        pos28 = (np.array([1, 0]) + 0.375) / (np.array([1, 2]) + 1)
        pos29 = (np.array([0, 2]) + 0.625) / (np.array([1, 2]) + 1)
        pos62 = (np.array([0, 0]) + 0.125) / (np.array([1, 2]) + 1)

        conditional_prob = kmers.calc_genus_conditional_prob(detect_list,
                                                             genera,
                                                             priors)
        self.assertTrue(np.array_equal(conditional_prob[25,], pos25))
        self.assertTrue(np.array_equal(conditional_prob[28,], pos28))
        self.assertTrue(np.array_equal(conditional_prob[29,], pos29))
        self.assertTrue(np.array_equal(conditional_prob[62,], pos62))

    def test_build_kmer_database(self):
        kmer_size = 3
        sequences = ["ATGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        genera = ["A", "B", "B"]

        # list
        db = kmers.build_kmer_database(sequences, genera, kmer_size)

        pos25 = (np.array([1, 2]) + 0.875) / (np.array([1, 2]) + 1)
        pos28 = (np.array([1, 0]) + 0.375) / (np.array([1, 2]) + 1)
        pos29 = (np.array([0, 2]) + 0.625) / (np.array([1, 2]) + 1)
        pos62 = (np.array([0, 0]) + 0.125) / (np.array([1, 2]) + 1)

        self.assertTrue(np.array_equal(db["conditional_prob"][25,], pos25))
        self.assertTrue(np.array_equal(db["conditional_prob"][28,], pos28))
        self.assertTrue(np.array_equal(db["conditional_prob"][29,], pos29))
        self.assertTrue(np.array_equal(db["conditional_prob"][62,], pos62))

        self.assertEqual(db["genera_names"][0], "A")
        self.assertEqual(db["genera_names"][1], "B")


if __name__ == '__main__':
    unittest.main()
