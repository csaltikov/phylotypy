from collections import defaultdict, Counter
import sys
import unittest

import numpy as np

import kmers
import phylotypy

sys.path.append('../')


class TestGetKmers(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = phylotypy.Phylotypy()
        self.X_train = ["CCGCTGA", "CCGCTGA", "GTGGAAT", "GTGGAAT", "TATGCAC"]
        self.y_train = ["A;a;A", "A;a;B", "A;a;C", "A;b;A", "A;b;B", "A;b;C"]
        self.detect_list = kmers.detect_kmers_across_sequences(self.X_train, 2)
        # self.db = kmers.build_kmer_database(self.X_train, self.y_train, 2)

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
        expected["confidence"] = np.array([1, 0.8, 0.8])

        observed = kmers.consensus_bs_class(bs_class, db)
        self.assertTrue(np.array_equal(observed["frac"], expected["confidence"]))

    def test_classifier_fit(self):
        self.classifier.fit(self.X_train, self.y_train, kmer_size=2)

        # check shape of the model
        observed = self.classifier.model.shape
        expected = (16, 3)
        self.assertEqual(observed, expected)

        # check that the references were selected properly
        ref_genera_idx = self.classifier.ref_genera_idx

        observed = ref_genera_idx
        expected = [0, 0, 1, 1, 2]
        self.assertTrue(observed, expected)

        # y_train gets sorted alphabetically
        observed = str(self.classifier.ref_genera[0])
        expected = "A;B;C;d"
        self.assertEqual(observed, expected)

        observed = str(self.classifier.ref_genera[2])
        expected = "A;C;A;a"
        self.assertEqual(observed, expected)

    # def test_bootstrap(self):
    #
    #     kmer_list = self.detect_list[0]
    #
    #     observed = self.classifier.bootstrap(kmer_list, 5)
    #
    #     expected = []
    #     self.assertEqual(expected, observed)

    # def test_predict(self):
    #
    #
