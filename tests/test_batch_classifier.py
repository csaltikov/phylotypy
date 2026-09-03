from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from phylotypy import read_fasta, classifier
from phylotypy import batch_classifier
from phylotypy.batch_classifier import (
    ClassifyAll,
    _classify_all_kernel,
    _check_bootstrap_memory,
    classify_all,
)


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

    def test_classify_all_kernel_matches_reference_computation(self):
        """Direct correctness check for the numba classify_all kernel against an
        independent, pure-numpy reference: sum cond_prob rows for each bootstrap
        replicate's sampled kmers, then argmax. Not tied to any particular
        implementation, so it stays valid even if the kernel is rewritten again.
        """
        rng = np.random.default_rng(0)
        n_kmers, n_genera, n_rows, n_sub = 50, 4, 20, 6
        cond_prob = rng.random((n_kmers, n_genera)).astype(np.float32)
        BS = rng.integers(0, n_kmers, size=(n_rows, n_sub)).astype(np.int64)

        observed = _classify_all_kernel(BS, cond_prob)
        expected = np.array([cond_prob[BS[i]].sum(axis=0).argmax() for i in range(n_rows)])

        self.assertTrue(np.array_equal(observed, expected))


class TestBootstrapMemoryCheck(unittest.TestCase):
    """_check_bootstrap_memory guards against classify_all's bootstrap array
    (n_sequences x num_bs x n_sub, int64) exceeding a safe fraction of total
    system RAM. These tests control the "total RAM" reading directly so they're
    deterministic regardless of the machine actually running them.
    """

    def test_raises_when_projection_exceeds_safety_fraction(self):
        eight_gb = 8 * 1024 ** 3
        with patch.object(batch_classifier, "_total_system_memory_bytes", return_value=eight_gb):
            # 1M sequences x 100 bootstraps x 200 kmers x 8 bytes ~= 149 GB, far over budget
            with self.assertRaises(MemoryError) as ctx:
                _check_bootstrap_memory(n_sequences=1_000_000, num_bs=100, n_sub=200, force=False)
            self.assertIn("force=True", str(ctx.exception))

    def test_force_bypasses_the_check(self):
        eight_gb = 8 * 1024 ** 3
        with patch.object(batch_classifier, "_total_system_memory_bytes", return_value=eight_gb):
            _check_bootstrap_memory(n_sequences=1_000_000, num_bs=100, n_sub=200, force=True)
            # no exception raised

    def test_allows_a_reasonably_sized_request(self):
        eight_gb = 8 * 1024 ** 3
        with patch.object(batch_classifier, "_total_system_memory_bytes", return_value=eight_gb):
            # 4,625 sequences x 100 bootstraps x 57 kmers x 8 bytes ~= 211 MB, well within budget
            _check_bootstrap_memory(n_sequences=4625, num_bs=100, n_sub=57, force=False)
            # no exception raised

    def test_skips_gracefully_when_total_memory_is_unknown(self):
        with patch.object(batch_classifier, "_total_system_memory_bytes", return_value=None):
            _check_bootstrap_memory(n_sequences=1_000_000, num_bs=100, n_sub=200, force=False)
            # no exception raised -- best-effort only, never breaks on unsupported platforms


class TestClassifyAllKmerSize(unittest.TestCase):
    def test_bootstrap_all_receives_the_actual_kmer_size(self):
        """classify_all() must forward its kmer_size to bootstrap_all() instead
        of relying on bootstrap_all's own default (8) -- otherwise n_sub is
        computed against the wrong kmer_size whenever a non-default one is used.
        """
        kmer_mat = np.tile(np.arange(30), (5, 1))  # 5 sequences, width 30
        mock_database = type("MockDB", (), {"conditional_prob": np.ones((100, 3), dtype=np.float32)})()

        with patch.object(batch_classifier, "bootstrap_all", wraps=batch_classifier.bootstrap_all) as mock_bootstrap:
            classify_all(kmer_mat, mock_database, num_bs=2, kmer_size=3)
            mock_bootstrap.assert_called_once()
            self.assertEqual(mock_bootstrap.call_args.kwargs["kmer_size"], 3)


if __name__ == '__main__':
    unittest.main()
