import unittest

import numpy as np
# from statsmodels.tools import sequences
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


try:
    import phylotypy.kmers as kmers
    import phylotypy.phylotypy as phylotypy
except ImportError:
    import kmers
    import phylotypy


class TestGetKmers(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = phylotypy.Classify()
        self.X_train = ["CCGCTGA", "CCGCTGA", "GTGGAAT", "GTGGAAT", "TATGCAC"]
        self.y_train = ["A;a;A", "A;a;B", "A;a;C", "A;b;A", "A;b;B", "A;b;C"]
        self.detect_list = kmers.detect_kmers_across_sequences(self.X_train, 2)
        # self.db = kmers.build_kmer_database(self.X_train, self.y_train, 2)

    def test_classifier_fit(self):
        kmer_size = 3
        sequences = ["ATGCGCTA", "ATGCGCTC", "ATGCGCTC"]
        genera = ["A;B;c", "B;C;d", "B;C;d"]
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
        expected = "A;B;c"
        self.assertEqual(observed, expected)

        observed = str(self.classifier.ref_genera[1])
        expected = "B;C;d"
        self.assertEqual(observed, expected)


    def test_classifier_bootstrap(self):
        # TODO
        kmer_size = 3
        bs_class = [0, 0, 0, 0, 3]
        genera = np.array(["A;a;A", "A;a;B", "A;a;C", "A;b;A", "A;b;B", "A;b;C"])

        classifier = phylotypy.Classify()
        classifier.n_levels = 3
        classifier.ref_genera = genera
        observed = classifier.consensus_bs_class(bs_class)
        expected_classification = ["A", "a", "A"]
        expected_scores = [100, 80, 80]

        self.assertEqual(expected_classification, observed["taxonomy"].tolist())
        self.assertEqual(expected_scores, observed["confidence"].tolist())

        consensus = dict(taxonomy=np.array(["A", "a", "A"]),
                         confidence=np.array([100, 80, 70]))

        classification_filtered = kmers.filter_taxonomy(consensus, 80)
        expected_classification_filtered = np.array(["A", "a"])

        test = np.array_equal(expected_classification_filtered, classification_filtered["taxonomy"])
        self.assertTrue(test)

    def test_print_taxonomy(self):
        consensus = dict(taxonomy=np.array(["A", "a"]),
                         confidence=np.array([100, 80]))
        expected = "A(100);a(80);a_unclassified(80)"
        observed = kmers.print_taxonomy(consensus, 3)

        self.assertEqual(expected, observed)
