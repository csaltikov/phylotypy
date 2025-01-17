import unittest

import numpy as np
from phylotypy import kmers


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

        seq_with_r = "ATGCGCTRGTAGCATGC"
        expected = "0321213N230210321"
        base4_seq = kmers.seq_to_base4(seq_with_r)
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
        num_kmers = len(sequences[0]) - kmer_size + 1

        expected = np.zeros((len(sequences), num_kmers), dtype=int)

        expected[0] = kmers.base4_to_index(kmers.get_all_kmers(base4_sequences[0], kmer_size))
        expected[1] = kmers.base4_to_index(kmers.get_all_kmers(base4_sequences[1], kmer_size))

        detect_list = kmers.detect_kmers_across_sequences_mp(sequences, kmer_size)

        self.assertTrue(np.array_equal(detect_list, expected))

        detect_list_mp = kmers.detect_kmers_across_sequences_mp(sequences, kmer_size, 6)
        self.assertTrue(np.array_equal(detect_list_mp, expected))

    def test_calc_word_specific_priors(self):
        """Test Calcuate word specific priors"""
        kmer_size = 3
        sequences = ["ATGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        detect_list = kmers.detect_kmers_across_sequences_mp(sequences, kmer_size)
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

    def test_calc_genus_conditional_prob_old(self):
        """Calculate genus-specific conditional probabilities"""
        kmer_size = 3
        sequences = ["ATGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        genera = [0, 1, 1]

        detect_list = kmers.detect_kmers_across_sequences_mp(sequences, kmer_size)
        priors = kmers.calc_word_specific_priors(detect_list, kmer_size)

        # (m(wi) + Pi) / (M + 1)

        conditional_prob = kmers.calc_genus_conditional_prob_old(detect_list,
                                                                 genera,
                                                                 priors)

        for pos, cond_prod in self.expected_cond_prods.items():
            log_cond_prod = np.log(cond_prod)
            self.assertTrue(np.array_equal(conditional_prob[pos,], log_cond_prod.astype(np.float16)))

        self.assertEqual(conditional_prob.dtype, np.float16)

    def test_calc_genus_conditional_prob(self):
        """Calculate genus-specific conditional probabilities"""
        kmer_size = 3
        sequences = ["ATGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        genera = [0, 1, 1]

        detect_list = kmers.detect_kmers_across_sequences_mp(sequences, kmer_size)
        priors = kmers.calc_word_specific_priors(detect_list, kmer_size)

        # (m(wi) + Pi) / (M + 1)
        conditional_prob = kmers.calc_genus_conditional_prob(detect_list,
                                                             genera,
                                                             priors)
        for pos, cond_prod in self.expected_cond_prods.items():
            log_cond_prod = np.log(cond_prod)
            self.assertTrue(np.array_equal(conditional_prob[pos,], log_cond_prod.astype(np.float16)))

    def test_build_kmer_database(self):
        kmer_size = 3
        sequences = ["ATGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        genera = ["A", "B", "B"]

        # list
        db = kmers.build_kmer_database(sequences, genera, kmer_size)

        for pos, cond_prod in self.expected_cond_prods.items():
            log_cond_prod = np.log(cond_prod).astype(np.float16)
            print(log_cond_prod, db.conditional_prob[pos,])
            self.assertTrue(np.array_equal(db.conditional_prob[pos,],
                                           log_cond_prod)
                            )

        self.assertEqual(db.genera_names[0], "A")
        self.assertEqual(db.genera_names[1], "B")

    def test_create_kmer_database(self):
        kmer_size = 3
        sequences = ["ATGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        genera = ["A", "B", "B"]
        db = kmers.build_kmer_database(sequences, genera, kmer_size)

        observed = db.conditional_prob[25,]  # np.array
        expected = (np.array([1, 2], dtype=float) + 0.875) / (np.array([1, 2]) + 1)
        self.assertTrue(np.array_equal(observed, np.log(expected).astype(np.float16)))

        observed = db.conditional_prob[28,]  # np.array
        expected = (np.array([1, 0]) + 0.375) / (np.array([1, 2]) + 1)
        self.assertTrue(np.array_equal(observed, np.log(expected).astype(np.float16)))

        observed = db.conditional_prob[29,]  # np.array
        expected = (np.array([0, 2]) + 0.625) / (np.array([1, 2]) + 1)
        self.assertTrue(np.array_equal(observed, np.log(expected).astype(np.float16)))

        observed = db.conditional_prob[63,]  # np.array
        expected = (np.array([0, 0]) + 0.125) / (np.array([1, 2]) + 1)
        self.assertTrue(np.array_equal(observed, np.log(expected).astype(np.float16)))

        observed = db.genera_idx[0]
        expected = 0
        self.assertTrue(np.array_equal(observed, expected))

        observed = db.genera_idx[1]
        expected = 1
        self.assertTrue(np.array_equal(observed, expected))

        observed = db.genera_idx[2]
        expected = 1
        self.assertTrue(np.array_equal(observed, expected))

        observed = db.genera_names[0]
        expected = "A"
        self.assertTrue(np.array_equal(observed, expected))

        observed = db.genera_names[1]
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

    def test_classify_bs(self):
        kmer_size = 2
        sequences = ["CTGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        genera = ["A", "B", "B"]
        db = kmers.build_kmer_database(sequences, genera, kmer_size)
        kmers_list = [3, 6, 13, 14]
        expected = 1
        observed = kmers.classify_bs(kmers_list, db)
        self.assertEqual(observed, expected)

        kmers_list = [9, 12, 15]
        expected = 0
        observed = kmers.classify_bs(kmers_list, db)
        self.assertEqual(observed, expected)

    def test_consensus_classified_bootstrap_samples(self):
        ref_genera = np.array(["A;a;A", "A;a;B", "A;a;C", "A;b;A", "A;b;B", "A;b;C"])

        db = kmers.KmerDB(genera_names=ref_genera,
                          conditional_prob=np.empty(0),
                          genera_idx=list(ref_genera))

        bs_class = np.array([0, 0, 0, 0, 3], dtype=int)

        expected = dict()
        expected["taxonomy"] = np.array(["A", "a", "A"])
        expected["confidence"] = np.array([100, 80, 80])

        observed = kmers.consensus_bs_class(bs_class, db.genera_names)
        print(observed["taxonomy"])
        self.assertTrue(np.array_equal(expected["confidence"], observed["confidence"]))

        self.assertTrue(np.array_equal(expected["taxonomy"], observed["taxonomy"]))

    def test_get_consensus(self):
        taxa_cumm_join_arr = np.array(
            [['A', 'A;a', 'A;a;A'],
             ['A', 'A;a', 'A;a;A'],
             ['A', 'A;a', 'A;a;A'],
             ['A', 'A;a', 'A;a;B']])

        expected = np.array(['A;a;A', 75], dtype='<U20')
        observed = kmers.get_consensus(taxa_cumm_join_arr[:, 2]).astype('<U20')

        truth = np.array_equal(expected, observed)
        self.assertTrue(truth)

    def test_consensus_taxonomy(self):
        oscillospiraceae = dict(
            taxonomy=np.array(["Bacteria", "Bacillota", "Clostridia", "Eubacteriales", "Oscillospiraceae"]),
            confidence=np.array([100, 100, 99, 99, 98])
        )

        expected = "Bacteria(100);Bacillota(100);Clostridia(99);Eubacteriales(99);Oscillospiraceae(98);Oscillospiraceae_unclassified(98)"

        tax_string = kmers.print_taxonomy(oscillospiraceae, n_levels=6)

        self.assertEqual(expected, tax_string)

        # bacteroidales = dict(taxonomy=np.array(["Bacteria", "Bacteroidota", "Bacteroidia", "Bacteroidales"]),
        #                      confidence=np.array([100, 100, 97, 97]))

    def test_consensus_bs_to_print_taxonomy(self):
        genera = np.array(["A;a;A", "A;a;B", "A;a;C", "A;b;A", "A;b;B", "A;b;C"])
        db = kmers.KmerDB(genera_names=genera,
                          conditional_prob=np.empty(genera.shape),
                          genera_idx=kmers.genera_str_to_index(list(genera))
                          )

        bs_class = np.array([0, 0, 0, 2, 3], dtype=int)

        expected = "A(100);a(80);a_unclassified(80)"

        classified = kmers.consensus_bs_class(bs_class, db.genera_names)
        filtered_classified = kmers.filter_taxonomy(classified)
        tax_string = kmers.print_taxonomy(filtered_classified, n_levels=3)

        self.assertEqual(expected, tax_string)

    def test_base10_base4_nucleotide(self):
        my_seq = "AAAATTTT"
        my_base4 = "00003333"
        my_kmer = 255

        observed = kmers.base10_base4(my_kmer)
        self.assertEqual(my_base4, observed)

        observed = kmers.base4_to_nucleotide(my_base4)
        self.assertEqual(my_seq, observed)

    def test_genera_str_to_index(self):
        genera = ["A;a;A", "A;a;B", "A;a;C", "A;a;A", "A;a;B", "A;b;C"]
        observed: list = kmers.genera_str_to_index(genera)
        expected: list = [0, 1, 2, 0, 1, 3]
        self.assertEqual(expected, observed)


if __name__ == '__main__':
    unittest.main()
