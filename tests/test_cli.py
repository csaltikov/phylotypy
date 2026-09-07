import contextlib
import io
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from phylotypy import cli, classifier, kmers, read_fasta


class TestCliBase(unittest.TestCase):
    def setUp(self):
        self.current_dir = Path(__file__).parent
        self.fasta_dir = self.current_dir / "test_fasta"
        self.test_fasta = self.fasta_dir / "test_fasta.fa"
        self.test_ref = read_fasta.read_taxa_fasta(self.test_fasta)
        self.expected_classification = (
            "Bacteria(100);Actinomycetota(100);Actinobacteria(100);"
            "Mycobacteriales(100);Mycobacteriaceae(100);Mycobacterium(100)"
        )

        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.tmp_path = Path(self.tmp_dir.name)

        # Query fasta: same sequences as the reference, but with plain (non-taxonomy)
        # ids, standing in for representative sequences to classify.
        self.query_fasta = self.tmp_path / "query.fa"
        with open(self.query_fasta, "w") as f:
            for i, (seq_id, sequence) in enumerate(zip(self.test_ref["id"], self.test_ref["sequence"]), 1):
                f.write(f">query_seq{i}\n{sequence}\n")

    def run_cli(self, argv):
        """Run cli.main(argv), returning (exit_code, stdout)."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(argv)
        return code, buf.getvalue()


class TestBuildCommand(TestCliBase):
    def test_build_creates_a_loadable_pickled_database(self):
        out_pkl = self.tmp_path / "db.pkl"
        code, _ = self.run_cli([
            "build", "-i", str(self.test_fasta), "-o", str(out_pkl), "--threads", "1",
        ])

        self.assertEqual(code, 0)
        self.assertTrue(out_pkl.exists())

        with open(out_pkl, "rb") as f:
            database = pickle.load(f)

        self.assertIsInstance(database, kmers.KmerDB)
        self.assertEqual(len(set(database.genera_idx)), self.test_ref["id"].nunique())

    def test_build_creates_missing_output_directories(self):
        out_pkl = self.tmp_path / "nested" / "dir" / "db.pkl"
        code, _ = self.run_cli([
            "build", "-i", str(self.test_fasta), "-o", str(out_pkl), "--threads", "1",
        ])

        self.assertEqual(code, 0)
        self.assertTrue(out_pkl.exists())

    def test_build_rejects_a_missing_input_file(self):
        with self.assertRaises(SystemExit) as ctx:
            self.run_cli(["build", "-i", "does_not_exist.fa", "-o", str(self.tmp_path / "db.pkl")])
        self.assertEqual(ctx.exception.code, 2)


class TestClassifyCommand(TestCliBase):
    def setUp(self):
        super().setUp()
        self.db_pkl = self.tmp_path / "db.pkl"
        database = classifier.make_classifier(self.test_ref, multiprocess=False)
        with open(self.db_pkl, "wb") as f:
            pickle.dump(database, f)

    def test_classify_with_prebuilt_db_writes_tsv_results(self):
        out_tsv = self.tmp_path / "results.tsv"
        code, stdout = self.run_cli([
            "classify", "-i", str(self.query_fasta), "-d", str(self.db_pkl),
            "-o", str(out_tsv), "--min-consensus", "70",
        ])

        self.assertEqual(code, 0)
        self.assertIn("Classified 4 sequences", stdout)

        results = pd.read_csv(out_tsv, sep="\t")
        self.assertEqual(list(results.columns), ["id", "classification"])
        self.assertEqual(len(results), 4)
        observed = results.loc[results["id"] == "query_seq1", "classification"].iloc[0]
        self.assertEqual(observed, self.expected_classification)

    def test_classify_out_csv_extension_uses_comma_separator(self):
        out_csv = self.tmp_path / "results.csv"
        code, _ = self.run_cli([
            "classify", "-i", str(self.query_fasta), "-d", str(self.db_pkl),
            "-o", str(out_csv),
        ])

        self.assertEqual(code, 0)
        first_line = out_csv.read_text().splitlines()[0]
        self.assertEqual(first_line, "id,classification")

    def test_classify_builds_on_the_fly_from_a_raw_reference_fasta(self):
        out_tsv = self.tmp_path / "results.tsv"
        save_db = self.tmp_path / "cached_db.pkl"
        code, stdout = self.run_cli([
            "classify", "-i", str(self.query_fasta), "-d", str(self.test_fasta),
            "-o", str(out_tsv), "--save-db", str(save_db), "-v",
        ])

        self.assertEqual(code, 0)
        self.assertIn("doesn't look like a prebuilt database", stdout)
        self.assertTrue(save_db.exists())

        results = pd.read_csv(out_tsv, sep="\t")
        self.assertEqual(len(results), 4)

    def test_classify_rejects_a_missing_db_file(self):
        with self.assertRaises(SystemExit) as ctx:
            self.run_cli([
                "classify", "-i", str(self.query_fasta), "-d", "does_not_exist.pkl",
                "-o", str(self.tmp_path / "out.tsv"),
            ])
        self.assertEqual(ctx.exception.code, 2)

    def test_classify_reports_errors_from_classification_without_a_traceback(self):
        out_tsv = self.tmp_path / "results.tsv"
        with patch.object(classifier, "classify_sequences", side_effect=ValueError("boom")):
            code, _ = self.run_cli([
                "classify", "-i", str(self.query_fasta), "-d", str(self.db_pkl), "-o", str(out_tsv),
            ])
        self.assertEqual(code, 1)
        self.assertFalse(out_tsv.exists())


class TestIsPickledDb(unittest.TestCase):
    def test_recognizes_pkl_and_pickle_extensions(self):
        self.assertTrue(cli._is_pickled_db(Path("db.pkl")))
        self.assertTrue(cli._is_pickled_db(Path("db.pickle")))
        self.assertTrue(cli._is_pickled_db(Path("db.PKL")))

    def test_rejects_fasta_extensions(self):
        self.assertFalse(cli._is_pickled_db(Path("ref.fa")))
        self.assertFalse(cli._is_pickled_db(Path("ref.fasta")))
        self.assertFalse(cli._is_pickled_db(Path("ref.fa.gz")))


class TestVersionAndHelp(unittest.TestCase):
    def test_version_flag_exits_cleanly(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--version"])
        self.assertEqual(ctx.exception.code, 0)

    def test_no_subcommand_is_an_error(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main([])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
