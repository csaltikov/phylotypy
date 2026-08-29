import numpy as np
from scipy import sparse
from pathlib import Path
from phylotypy import kmers, conditional_prob, bootstrap
from collections import defaultdict
import pandas as pd

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


def classify_all(kmer_mat, database, num_bs: int = 100, chunk=20_000, verbose: bool = False):
    cond_prob = database.conditional_prob
    n_kmers = cond_prob.shape[0]
    seq = kmer_mat.shape[0]

    bs = bootstrap_all(kmer_mat, num_bs=num_bs)
    n_sub = bs.shape[2]
    BS = bs.reshape(seq * num_bs, n_sub)

    genus = np.empty(seq * num_bs, dtype=np.int64)

    if verbose:
        print(f"Processing {kmer_mat.shape[0]} sequences")

    n_chunks = -(-BS.shape[0] // chunk)  # ceil division
    # if chunk equals num_bs, one sequence is processed per loop
    for chunk_idx, start in enumerate(range(0, BS.shape[0], chunk), start=1):
        block = BS[start:start + chunk]
        M = block.shape[0]
        rows = np.repeat(np.arange(M), n_sub)
        C = sparse.csr_matrix(
            (np.ones(M * n_sub, np.float32), (rows, block.ravel())),
            shape=(M, n_kmers),
        )
        genus[start:start + M] = (C @ cond_prob).argmax(axis=1)

        if verbose:
            sequences_done = min(seq, (start + M) // num_bs)
            print(f"Chunk {chunk_idx}/{n_chunks}: {sequences_done}/{seq} sequences processed")
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

    def calc_kmer_mat(self, sequences, **kwargs):
        self.genera_idx, self.kmer_mat = conditional_prob.seq_to_kmers_database(sequences, **kwargs)

    def classify(self, sequences, database, num_bs: int = 100, chunk=20_000, **kwargs):
        verbose = kwargs.get("verbose", False)
        self.database = database
        self.sequences = sequences
        self.calc_kmer_mat(sequences, **kwargs)

        self.bs_results = classify_all(
            kmer_mat=self.kmer_mat,
            database=self.database,
            num_bs=num_bs,
            chunk=chunk,
            verbose=verbose
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
