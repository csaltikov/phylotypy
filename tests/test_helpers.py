import os
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


class TestPathHelper(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name).resolve()
        self.existing = self.tmp_dir / "seqs.fa"
        self.existing.write_text(">s1\nACGT\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_str_is_resolved_to_absolute_path(self):
        out = helpers.path_helper(str(self.existing))
        self.assertIsInstance(out, Path)
        self.assertTrue(out.is_absolute())
        self.assertEqual(out, self.existing)

    def test_pathlike_passthrough(self):
        self.assertEqual(helpers.path_helper(self.existing), self.existing)

    def test_bytes_input(self):
        self.assertEqual(helpers.path_helper(os.fsencode(self.existing)), self.existing)

    def test_strips_whitespace(self):
        self.assertEqual(helpers.path_helper(f"  {self.existing}\n"), self.existing)

    def test_expands_tilde(self):
        out = helpers.path_helper("~/mypath/fasta.fa")
        self.assertEqual(out, (Path.home() / "mypath" / "fasta.fa").resolve())
        self.assertNotIn("~", str(out))

    def test_expands_env_var(self):
        os.environ["PHYLOTYPY_TEST_DIR"] = str(self.tmp_dir)
        try:
            out = helpers.path_helper("$PHYLOTYPY_TEST_DIR/seqs.fa")
        finally:
            del os.environ["PHYLOTYPY_TEST_DIR"]
        self.assertEqual(out, self.existing)

    def test_relative_path_resolved_against_cwd(self):
        out = helpers.path_helper("some/relative/file.fa")
        self.assertEqual(out, (Path.cwd() / "some/relative/file.fa").resolve())

    def test_none_raises_type_error(self):
        with self.assertRaises(TypeError):
            helpers.path_helper(None)

    def test_non_pathlike_raises_type_error(self):
        with self.assertRaises(TypeError):
            helpers.path_helper(123)

    def test_empty_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            helpers.path_helper("   ")

    def test_must_exist_passes_for_existing_file(self):
        self.assertEqual(helpers.path_helper(self.existing, must_exist=True), self.existing)

    def test_must_exist_raises_for_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            helpers.path_helper(self.tmp_dir / "nope.fa", must_exist=True)

    def test_missing_file_allowed_by_default(self):
        out = helpers.path_helper(self.tmp_dir / "nope.fa")
        self.assertEqual(out, self.tmp_dir / "nope.fa")
