import gzip
from pathlib import Path
import unittest
import tempfile
import pandas as pd
from phylotypy import df_to_fasta
from phylotypy import read_taxa_fasta


class TestWriteFasta(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"id": ["seq1", "seq2"], "sequence": ["ACTGCG", "TTTGCCGAAGCAGCT"]})
        self.test_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.test_dir.name)
        self.file_path = Path().joinpath(self.test_dir.name, "test.fasta")

        # 2. Register the assertion wrapper FIRST (Runs LAST)
        self.addCleanup(self.verify_dir_is_gone, self.dir_path)
        self.addCleanup(self.test_dir.cleanup)

    def verify_dir_is_gone(self, path: Path):
        directory_still_exists = path.exists()
        self.assertFalse(directory_still_exists, f"Leaked temporary directory at {path}")

    def test_write_fasta(self):
        df_to_fasta(self.df, self.file_path)

        expected = ">seq1\nACTGCG\n>seq2\nTTTGCCGAAGCAGCT\n"

        with open(self.file_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), expected)

    def test_write_read_fasta(self):
        df_to_fasta(self.df, self.file_path)
        fasta_df = read_taxa_fasta(self.file_path)
        pd.testing.assert_frame_equal(fasta_df, self.df)
        self.assertTrue(self.file_path.exists())

    def test_line_wrap(self):
        long_seq = "A" * 130
        df = pd.DataFrame({"id": ["seq1"], "sequence": [long_seq]})
        df_to_fasta(df, self.file_path, line_width=60)
        lines = self.file_path.read_text().splitlines()

        self.assertEqual(lines[0], ">seq1")
        self.assertEqual(lines[1], "A" * 60)
        self.assertEqual(lines[2], "A" * 60)
        self.assertEqual(lines[3], "A" * 10)  # remainder
        self.assertEqual(len(lines), 4)

    def test_no_line_wrap(self):
        long_seq = "A" * 130
        df = pd.DataFrame({"id": ["seq1"], "sequence": [long_seq]})
        df_to_fasta(df, self.file_path, line_width=0)
        lines = self.file_path.read_text().splitlines()
        self.assertEqual(lines[1], long_seq)

    def test_write_gz_fasta(self):
        gz_file = Path().joinpath(self.test_dir.name, "out.fa.gz")
        df_to_fasta(self.df, gz_file)

        with gzip.open(gz_file, "rt") as f:
            content = f.read()

        self.assertIn(">seq1\nACTGCG", content)

    def test_write_bad_fasta(self):
        bad_df = pd.DataFrame({"id": ["seq1"], "seq": ["ACGT"]})
        with self.assertRaises(ValueError):
            df_to_fasta(bad_df, self.file_path)

        bad_df2 = pd.DataFrame({"sequence": ["ACGT"]})
        with self.assertRaises(ValueError):
            df_to_fasta(bad_df2, self.file_path)


if __name__ == "__main__":
    unittest.main()
