import os
import numpy as np
import numba as nb
from pathlib import Path
from phylotypy import kmers, conditional_prob, bootstrap
from collections import defaultdict
import pandas as pd

_MEMORY_SAFETY_FRACTION = 0.3


def _total_system_memory_bytes():
    """Best-effort total system RAM in bytes via POSIX sysconf. Returns None on
    platforms where this isn't available (e.g. Windows), so the safety check
    below degrades gracefully rather than crashing. Deliberately dependency-free
    (no psutil) -- this only reads OS-exposed aggregate memory stats, the same
    information any unprivileged process/tool (top, Activity Monitor) can see.
    """
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _check_bootstrap_memory(n_sequences, num_bs, n_sub, force):
    """Refuse to run classify_all's bootstrap step if it would need more than
    _MEMORY_SAFETY_FRACTION of total system RAM just for the sampled-kmer array
    -- that risks a crash or severe swap thrashing, not just a slow run. Pass
    force=True to bypass this if you know your machine can handle it.
    """
    if force:
        return

    total_memory = _total_system_memory_bytes()
    if total_memory is None:
        return  # can't determine total RAM on this platform; best-effort only

    projected_bytes = n_sequences * num_bs * n_sub * 8  # bs array is int64
    budget = total_memory * _MEMORY_SAFETY_FRACTION

    if projected_bytes > budget:
        max_num_bs = max(1, int(budget // (n_sequences * n_sub * 8)))
        raise MemoryError(
            f"Classifying {n_sequences:,} sequences with num_bs={num_bs} would need "
            f"~{projected_bytes / 1e9:.1f} GB for bootstrap sampling alone -- more than "
            f"{_MEMORY_SAFETY_FRACTION:.0%} of this machine's {total_memory / 1e9:.1f} GB "
            f"of total RAM. This risks crashing or severely slowing your system, not just "
            f"running slowly.\n"
            f"Try num_bs<={max_num_bs} with this many sequences, or classify fewer "
            f"sequences at a time. If you understand the risk and want to proceed anyway, "
            f"pass force=True."
        )


def bootstrap_all(kmer_mat, kmer_size: int = 8, num_bs: int = 100, rng=None):
    """Bootstrap sample kmer indices. Take an 2D array of kmers where each row
    contains the indices for each sequence. The first column is the index of the sequence
    """
    rng = np.random.default_rng(rng) if rng is None else rng
    seq, length = kmer_mat.shape
    n_sub = length // kmer_size
    valid_counts = (kmer_mat != -1).sum(1)
    pos = (rng.random((seq, num_bs, n_sub)) * valid_counts[:, None, None]).astype(np.int64)
    return kmer_mat[np.arange(seq)[:, None, None], pos]


@nb.njit(parallel=True, fastmath=True)
def _classify_all_kernel(BS, cond_prob):
    """For each bootstrap replicate (row of BS), sum the cond_prob rows for its
    sampled kmer indices and take the argmax genus. Parallelized across rows.
    """
    M, n_sub = BS.shape
    n_genera = cond_prob.shape[1]
    genus = np.empty(M, dtype=np.int64)
    for i in nb.prange(M):
        acc = np.zeros(n_genera, dtype=np.float32)
        for j in range(n_sub):
            k = BS[i, j]
            acc += cond_prob[k, :]
        best = 0
        best_val = acc[0]
        for g in range(1, n_genera):
            if acc[g] > best_val:
                best_val = acc[g]
                best = g
        genus[i] = best
    return genus


def classify_all(kmer_mat, database, num_bs: int = 100, kmer_size: int = 8,
                 verbose: bool = False, force: bool = False):
    cond_prob = database.conditional_prob
    seq, length = kmer_mat.shape
    n_sub = length // kmer_size

    _check_bootstrap_memory(seq, num_bs, n_sub, force)

    if verbose:
        print(f"Processing {seq} sequences")

    bs = bootstrap_all(kmer_mat, kmer_size=kmer_size, num_bs=num_bs)
    BS = np.ascontiguousarray(bs.reshape(seq * num_bs, n_sub))

    genus = _classify_all_kernel(BS, cond_prob)

    if verbose:
        print(f"Done classifying {seq} sequences")
    return genus.reshape(seq, num_bs)


def summarize(bs_res, sequences, database, min_confidence=80, n_levels=None, verbose=False):
    if n_levels is None:
        counts = np.char.count(database.genera_names, ";")
        n_levels = int(np.bincount(counts).argmax()) + 1

    batch_consensus = bootstrap.bootstrap_consensus_batch(bs_res, database.genera_names)
    taxa_consensus = batch_consensus["taxonomy"]
    confidence_consensus = batch_consensus["confidence"]

    classified = defaultdict(list)
    classified["id"] = sequences["id"].to_numpy()
    for i in range(bs_res.shape[0]):
        if verbose:
            if i % 100 == 0:
                print(f"Classifying sequence {i} of {len(classified['id'])}")
        consensus = dict(taxonomy=taxa_consensus[i], confidence=confidence_consensus[i])
        filtered = kmers.filter_taxonomy(classification=consensus, min_confidence=min_confidence)
        classified["classification"].append(kmers.print_taxonomy(consensus=filtered, n_levels=n_levels))
    return pd.DataFrame(classified)


class ClassifyAll:
    def __init__(self):
        self.database: kmers.KmerDB | dict = None
        self.kmer_mat = None
        self.bs_results = None
        self.sequences: pd.DataFrame | str | Path = None
        self.genera_idx = None
        self.detected_kmers_test = None

    def calc_kmer_mat(self, sequences, seq_col: str = "sequence", id_col: str = "id",
                      kmer_size: int = 8, verbose: bool = False):
        self.genera_idx, self.kmer_mat = conditional_prob.seq_to_kmers_database(
            sequences, seq_col=seq_col, id_col=id_col, kmer_size=kmer_size, verbose=verbose
        )

    def classify(self, sequences, database, num_bs: int = 100,
                seq_col: str = "sequence", id_col: str = "id",
                kmer_size: int = 8, verbose: bool = False, force: bool = False):
        self.database = database
        self.sequences = sequences
        self.calc_kmer_mat(sequences, seq_col=seq_col, id_col=id_col, kmer_size=kmer_size, verbose=verbose)

        self.bs_results = classify_all(
            kmer_mat=self.kmer_mat,
            database=self.database,
            num_bs=num_bs,
            kmer_size=kmer_size,
            verbose=verbose,
            force=force,
        )

    def results(self, min_confidence=80, n_levels=None, verbose=False):
        return summarize(
            self.bs_results,
            self.sequences,
            self.database,
            min_confidence=min_confidence,
            n_levels=n_levels,
            verbose=verbose
        )
