import tempfile
import unittest
from pathlib import Path

import pandas as pd

from phylotypy.utilities import helpers


class TestLoadAndValidateSeqs(unittest.TestCase):
    def setUp(self) -> None:
        self.current_dir = Path(__file__).parent
        self.fasta_dir = self.current_dir / "test_fasta"
        self.fasta_file = self.fasta_dir / "test_fasta_suborder.fa"
        self.fasta_file_gz = self.fasta_dir / "test_fasta_short_taxa.fa.gz"
        self.valid_df = pd.DataFrame({"id": ["seq1", "seq2"], "sequence": ["ACGT", "TGCA"]})

    # DataFrame input

    def test_dataframe_passthrough(self):
        observed = helpers.load_and_validate_seqs(self.valid_df)
        pd.testing.assert_frame_equal(observed, self.valid_df)

    def test_dataframe_missing_columns_raises(self):
        bad_df = pd.DataFrame({"id": ["seq1"], "seq": ["ACGT"]})
        with self.assertRaises(ValueError):
            helpers.load_and_validate_seqs(bad_df)

    def test_non_dataframe_non_path_raises_type_error(self):
        with self.assertRaises(TypeError):
            helpers.load_and_validate_seqs(["not", "a", "dataframe"])

    # FASTA input

    def test_fasta_file(self):
        observed = helpers.load_and_validate_seqs(self.fasta_file)
        self.assertIsInstance(observed, pd.DataFrame)
        self.assertEqual({"id", "sequence"}, set(observed.columns))
        self.assertGreater(len(observed), 0)

    def test_fasta_file_str_path(self):
        observed = helpers.load_and_validate_seqs(str(self.fasta_file))
        self.assertIsInstance(observed, pd.DataFrame)

    def test_fasta_gz_file(self):
        observed = helpers.load_and_validate_seqs(self.fasta_file_gz)
        self.assertIsInstance(observed, pd.DataFrame)
        self.assertGreater(len(observed), 0)

    def test_fasta_extension_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            upper_case_copy = Path(tmp_dir) / "MY_SEQS.FA"
            upper_case_copy.write_text(self.fasta_file.read_text())
            observed = helpers.load_and_validate_seqs(upper_case_copy)
            self.assertIsInstance(observed, pd.DataFrame)

    # CSV / TSV input

    def test_csv_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "seqs.csv"
            self.valid_df.to_csv(csv_path, index=False)
            observed = helpers.load_and_validate_seqs(csv_path)
            pd.testing.assert_frame_equal(observed, self.valid_df)

    def test_tsv_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tsv_path = Path(tmp_dir) / "seqs.tsv"
            self.valid_df.to_csv(tsv_path, sep="\t", index=False)
            observed = helpers.load_and_validate_seqs(tsv_path)
            pd.testing.assert_frame_equal(observed, self.valid_df)

    def test_csv_file_missing_columns_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "bad_seqs.csv"
            pd.DataFrame({"id": ["seq1"], "seq": ["ACGT"]}).to_csv(csv_path, index=False)
            with self.assertRaises(ValueError):
                helpers.load_and_validate_seqs(csv_path)

    # Unsupported input

    def test_unsupported_extension_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = Path(tmp_dir) / "seqs.txt"
            bad_path.write_text("id,sequence\nseq1,ACGT\n")
            with self.assertRaises(ValueError):
                helpers.load_and_validate_seqs(bad_path)


if __name__ == "__main__":
    unittest.main()
