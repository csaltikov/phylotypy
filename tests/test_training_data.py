from pathlib import Path
import unittest

from phylotypy.utilities import read_fasta


class TestTrainingData(unittest.TestCase):
    def setUp(self) -> None:
        self.current_dir = Path(__file__).parent
        self.fasta_dir = self.current_dir / 'test_fasta'
        self.fasta_file_suborder = self.fasta_dir / "test_fasta_suborder.fa"  # has suborder
        self.fasta_file_short = self.fasta_dir / "test_fasta_short_taxa.fa" # missing some taxa levels

    def test_is_gzip(self):
        gz_file = self.current_dir / "test_fasta" / "test_fasta_short_taxa.fa.gz"
        observed = read_fasta.is_gzip_file(gz_file)
        self.assertTrue(observed)

        non_gz_file = self.current_dir / "test_fasta" / "test_fasta_short_taxa.fa"
        observed = read_fasta.is_gzip_file(non_gz_file)
        self.assertFalse(observed)

    def test_fasta_id(self):
        # fasta string has suborder__
        ref = read_fasta.read_taxa_fasta(self.fasta_file_suborder)
        fasta_string = ref["id"].iloc[0].split(";")
        self.assertTrue(len(fasta_string), 6)

        # fasta string taxonomy string missing taxa levels
        ref = read_fasta.read_taxa_fasta(self.fasta_file_short)
        fasta_string = ref["id"].iloc[0].split(";")
        self.assertLess(len(fasta_string), 6)

        ref = read_fasta.read_taxa_fasta(self.fasta_dir / "test_fasta_short_taxa.fa.gz")
        fasta_string = ref["id"].iloc[0].split(";")
        self.assertLess(len(fasta_string), 6)


if __name__ == "__main__":
    unittest.main()
