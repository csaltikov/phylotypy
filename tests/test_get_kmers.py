import unittest

import numpy as np
from phylotypy import kmers


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

    def test_create_kmer_database(self):
        kmer_size = 3
        sequences = ["ATGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        genera = ["A", "B", "B"]
        db = kmers.build_kmer_database(sequences, genera, kmer_size)

        observed = db["conditional_prob"][25,]  # np.array
        expected = (np.array([1, 2]) + 0.875) / (np.array([1, 2]) + 1)
        self.assertTrue(np.array_equal(observed, expected))

        observed = db["conditional_prob"][28,]  # np.array
        expected = (np.array([1, 0]) + 0.375) / (np.array([1, 2]) + 1)
        self.assertTrue(np.array_equal(observed, expected))

        observed = db["conditional_prob"][29,]  # np.array
        expected = (np.array([0, 2]) + 0.625) / (np.array([1, 2]) + 1)
        self.assertTrue(np.array_equal(observed, expected))

        observed = db["conditional_prob"][63,]  # np.array
        expected = (np.array([0, 0]) + 0.125) / (np.array([1, 2]) + 1)
        self.assertTrue(np.array_equal(observed, expected))

        observed = db["genera_idx"][0]
        expected = 0
        self.assertTrue(np.array_equal(observed, expected))

        observed = db["genera_idx"][1]
        expected = 1
        self.assertTrue(np.array_equal(observed, expected))

        observed = db["genera_idx"][2]
        expected = 1
        self.assertTrue(np.array_equal(observed, expected))

        observed = db["genera_names"][0]
        expected = "A"
        self.assertTrue(np.array_equal(observed, expected))

        observed = db["genera_names"][1]
        expected = "B"
        self.assertTrue(np.array_equal(observed, expected))

    def test_bootstrap_kmers(self):
        kmers_arr = np.arange(100)
        kmer_size = 8
        expected_n_kmers = 100 // 8

        detected = kmers.bootstrap_kmers(kmers_arr, kmer_size)
        self.assertEqual(detected.shape[0], expected_n_kmers)

        detected_in_kmers = np.isin(detected, kmers_arr)
        self.assertTrue(np.all(detected_in_kmers))

    def test_bootstrap_sampler(self):
        kmer_size = 3
        sequences = ["ATGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        genera = ["A", "B", "B"]
        db = kmers.build_kmer_database(sequences, genera, kmer_size)

        unknown_kmers = kmers.detect_kmers("ATGCGCTC", kmer_size)
        expected_classification = 1

        detected_classification = kmers.classify_bs(unknown_kmers, db)
        self.assertEqual(detected_classification, expected_classification)

    def test_consensus_classified_bootstrap_samples(self):
        db = dict()
        db["genera"] = np.array(["A;a;A", "A;a;B", "A;a;C", "A;b;A", "A;b;B", "A;b;C"])

        bs_class = np.array([0, 0, 0, 0, 3], dtype=int)

        expected = dict()
        expected["taxonomy"] = np.array(["A", "a", "A"])
        expected["confidence"] = np.array([100, 80, 80])

        observed = kmers.consensus_bs_class(bs_class, db)
        print(observed["taxonomy"])
        self.assertTrue(np.array_equal(expected["confidence"], observed["confidence"]))

        self.assertTrue(np.array_equal(expected["taxonomy"], observed["taxonomy"]))

    def test_consensus_taxonomy(self):
        oscillospiraceae = dict(
            taxonomy=np.array(["Bacteria", "Bacillota", "Clostridia", "Eubacteriales", "Oscillospiraceae"]),
            confidence=np.array([100, 100, 99, 99, 98])
            )

        expected = "Bacteria(100);Bacillota(100);Clostridia(99);Eubacteriales(99);Oscillospiraceae(98);Oscillospiraceae_unclassified(98)"

        tax_string = kmers.print_taxonomy(oscillospiraceae, n_levels=6)

        self.assertEqual(expected, tax_string)

        bacteroidales = dict(taxonomy=np.array(["Bacteria", "Bacteroidota", "Bacteroidia", "Bacteroidales"]),
                             confidence=np.array([100, 100, 97, 97]))

    def test_consensus_bs_to_print_taxonomy(self):
        db = dict()
        db["genera"] = np.array(["A;a;A", "A;a;B", "A;a;C", "A;b;A", "A;b;B", "A;b;C"])

        bs_class = np.array([0, 0, 0, 2, 3], dtype=int)

        expected = "A(100);a(80);a_unclassified(80)"

        classified = kmers.consensus_bs_class(bs_class, db)
        filtered_classified = kmers.filter_taxonomy(classified)
        tax_string = kmers.print_taxonomy(filtered_classified, n_levels=3)

        self.assertEqual(expected, tax_string)

if __name__ == '__main__':
    unittest.main()