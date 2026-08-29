from pathlib import Path
import unittest

from phylotypy import read_fasta, classifier
from phylotypy.batch_classifier import ClassifyAll


class TestClassifyAll(unittest.TestCase):
    def setUp(self):
        self.current_dir = Path(__file__).parent
        self.fasta_dir = self.current_dir / 'test_fasta'
        self.test_fasta = self.fasta_dir / "test_fasta.fa"
        self.test_ref = read_fasta.read_taxa_fasta(self.test_fasta)
        self.expected_classification = (
            "Bacteria(100);Actinomycetota(100);Actinobacteria(100);"
            "Mycobacteriales(100);Mycobacteriaceae(100);Mycobacterium(100)"
        )

    def test_classify_matches_known_taxonomy(self):
        database = classifier.make_classifier(self.test_ref)

        classify_seqs = ClassifyAll()
        classify_seqs.classify(self.test_ref, database)
        res = classify_seqs.results()

        self.assertEqual(len(res), len(self.test_ref))
        observed = res.loc[res["id"] == self.test_ref["id"].iloc[0], "classification"].iloc[0]
        self.assertEqual(observed, self.expected_classification)

    def test_classify_with_small_chunks_matches_full_batch(self):
        """chunk smaller than num_bs forces classify_all() through multiple
        chunked passes instead of one; result should be unaffected."""
        database = classifier.make_classifier(self.test_ref)

        classify_seqs = ClassifyAll()
        classify_seqs.classify(self.test_ref, database, num_bs=100, chunk=5)
        res = classify_seqs.results()

        observed = res.loc[res["id"] == self.test_ref["id"].iloc[0], "classification"].iloc[0]
        self.assertEqual(observed, self.expected_classification)


if __name__ == '__main__':
    unittest.main()
