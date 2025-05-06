from pathlib import Path
import unittest

from phylotypy import read_fasta
from phylotypy import classifier


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.current_dir = Path(__file__).parent
        self.fasta_dir = self.current_dir / 'test_fasta'
        self.test_fasta = self.fasta_dir / "test_fasta.fa"
        self.test_ref = read_fasta.read_taxa_fasta(self.test_fasta)

    def test_make_classifier(self):
        observed = classifier.make_classifier(self.test_ref)

        n_genera_exp = 3
        n_genera_obs = len(observed.genera_names)
        self.assertEqual(n_genera_exp, n_genera_obs)

    def test_classify_sequence(self):
        database = classifier.make_classifier(self.test_ref, multiprocess=True, n_cpu=8, verbose=True)

        res = classifier.classify_sequences(self.test_ref, database)
        observed = res.iloc[0, 1]
        expected_res = "Bacteria(100);Actinomycetota(100);Actinobacteria(100);Mycobacteriales(100);Mycobacteriaceae(100);Mycobacterium(100)"

        self.assertEqual(observed, expected_res)


if __name__ == '__main__':
    unittest.main()
