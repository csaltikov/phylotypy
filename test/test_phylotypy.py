import unittest

import numpy as np
# from statsmodels.tools import sequences

import kmers
import phylotypy


class TestGetKmers(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = phylotypy.Phylotypy()
        self.X_train = ["CCGCTGA", "CCGCTGA", "GTGGAAT", "GTGGAAT", "TATGCAC"]
        self.y_train = ["A;a;A", "A;a;B", "A;a;C", "A;b;A", "A;b;B", "A;b;C"]
        self.detect_list = kmers.detect_kmers_across_sequences(self.X_train, 2)
        # self.db = kmers.build_kmer_database(self.X_train, self.y_train, 2)

    def test_classifier_fit(self):
        kmer_size = 3
        sequences = ["ATGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        genera = ["A", "B", "B"]
        self.classifier.fit(sequences, genera, kmer_size=kmer_size)

        # check shape of the model
        observed = self.classifier.model.shape
        expected = (64, 2)
        self.assertEqual(observed, expected)

        # check that the references were selected properly
        ref_genera_idx = self.classifier.ref_genera_idx

        observed = ref_genera_idx
        expected = [0, 1]
        self.assertTrue(observed, expected)

        # genera gets sorted alphabetically
        observed = str(self.classifier.ref_genera[0])
        expected = "A"
        self.assertEqual(observed, expected)

        observed = str(self.classifier.ref_genera[1])
        expected = "B"
        self.assertEqual(observed, expected)


    def test_classifier_bootstrap(self):
        # TODO
        kmer_size = 3
        bs_class = [0, 0, 0, 0, 3]
        genera = np.array(["A;a;A", "A;a;B", "A;a;C", "A;b;A", "A;b;B", "A;b;C"])

        classifier = phylotypy.Phylotypy()
        classifier.n_levels = 3
        classifier.ref_genera = genera
        observed_taxa, observed_scores = classifier.consensus_bs_class(bs_class)
        expected_classification = "A;a;A"
        expected_score = 100 * 4/5

        self.assertEqual(observed_taxa, expected_classification)
        self.assertEqual(observed_scores, expected_score)

    #TODO add more testing
