#!/usr/bin/env python3
import multiprocessing
import platform


if platform.system() == "Darwin":
    multiprocessing.set_start_method('spawn', force=True)
    
    
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import numba as nb

from phylotypy import kmers
from phylotypy import cond_prob_cython
from phylotypy import _worker_pool


def calc_priors(detected_kmers: np.ndarray, kmer_size: int = 8):
    num_seqs = len(detected_kmers)
    max_value = 4 ** kmer_size

    # get all kmers in the corpus of sequences
    flat_kmers = detected_kmers[:, 1:].flatten() # ignore col 0, row indices
    # remove negative values
    flat_kmers = flat_kmers[flat_kmers !=-1]  # -1 are encoded as NA values
    counts = np.bincount(flat_kmers, minlength=max_value)

    # expected-likelihood estimate using Jeffreys-Perks law of succession
    # 0 < Pi < 1
    return (counts + 0.5) / (num_seqs + 1)


##
def fix_kmers_length(kmer_arr, seq_len: int = 1400):
    n_missing = seq_len - len(kmer_arr)
    return np.concatenate([kmer_arr, np.full(n_missing, -1, dtype=int)])


##
def seq_to_kmers_database(sequences_db, seq_col: str = 'sequence', id_col: str = 'id',
                          kmer_size: int = 8, verbose: bool = False):

    if verbose:
        print(f"kmer_size is set to {kmer_size}")

    db = sequences_db

    if isinstance(sequences_db, str):
        if ".csv" in Path(sequences_db).suffix:
            db = pd.read_csv(sequences_db)
        if ".tsv" in Path(sequences_db).suffix:
            db = pd.read_csv(sequences_db, sep="\t")
        else:
            db = pd.read_csv(sequences_db, sep=None, engine='python')

    pool = _worker_pool.get_pool()
    kmer_results = pool.starmap(kmers.detect_kmer_indices, [(s, kmer_size) for s in db[seq_col]])
    kmer_series = pd.Series(kmer_results, index=db.index)
    # Calculate max length
    max_seq_len = kmer_series.str.len().max().astype(int)

    # pad each sequence's kmer array to a common length (cheap; no pool needed)
    detected_kmers = kmer_series.apply(lambda x: fix_kmers_length(x, max_seq_len))
    genera_idx = kmers.genera_str_to_index(db[id_col])

    print("Done with detecting k-mers")
    # Create final array first column are the sequence indices
    all_kmers_arr = np.hstack((
        np.array(genera_idx).reshape(-1, 1),
        np.stack(detected_kmers.to_numpy())
    ), dtype=int)
    return [genera_idx, all_kmers_arr]


##
class GenusCondProb:
    """Calculates the genus conditional probability matrix
    for a corpus of kmer indices of all the unique genera"""
    def __init__(self, kmers_arr: np.ndarray, priors: np.ndarray, kmer_size: int = 8):
        self.kmers_arr = kmers_arr
        self.priors = priors
        self.kmer_size = kmer_size
        self.m_1 = None
        self.wi_pi = None

    def calculate_genus_counts(self):
        genus_ids = self.kmers_arr[:, 0]
        kmers = self.kmers_arr[:, 1:]
        unique_genera, inverse, self.genera_idx_counts = np.unique(genus_ids, return_inverse=True, return_counts=True)
        kmer_flat = kmers.flatten()
        mask = kmer_flat != -1
        kmer_clean = kmer_flat[mask]
        genus_map_clean = inverse[np.arange(len(genus_ids))].repeat(kmers.shape[1])[mask]
        self.counts = np.zeros((4 ** self.kmer_size, len(unique_genera)), dtype=int)
        np.add.at(self.counts, (kmer_clean, genus_map_clean), 1)
        return self.counts

    def calculate(self):
        self.calculate_genus_counts()
        self.wi_pi = (self.counts + self.priors.reshape(-1, 1))
        self.m_1 = (self.genera_idx_counts + 1)
        divided = np.divide(self.wi_pi, self.m_1)
        return np.log(divided).astype(np.float32)

    def calculate_genus_counts_(self):
        self.uniq_idx, self.uniq_idx_counts = np.unique(self.kmers_arr[:, 0], return_counts=True)  # first column are the seq ids
        self.genus_kmer_counts = np.zeros((4 ** self.kmer_size, self.uniq_idx.shape[0]), dtype=int)  # id, kmer_indices, counts

        for uniq in self.uniq_idx:
            kmers_arr_ = self.kmers_arr[self.kmers_arr[:, 0] == uniq, 1:]  # col 0 are the sequence ids
            kmers_arr_ = kmers_arr_[kmers_arr_ != -1].flatten()
            self.genus_kmer_counts[:, uniq] = np.bincount(kmers_arr_[1:], minlength=4**self.kmer_size)

    def calculate_(self):
        self.calculate_genus_counts_()
        self.wi_pi = (self.genus_kmer_counts + self.priors.reshape(-1, 1))
        self.m_1 = (self.uniq_idx_counts + 1)
        divided = np.divide(self.wi_pi, self.m_1)
        return np.log(divided).astype(np.float32)


##
@nb.njit(parallel=True)
def genus_counts_parallel(detect_list, genera_idx, n_kmers, n_genera):
    """Efficiently count kmers per genus using parallel processing"""
    genus_count = np.zeros((n_kmers, n_genera), dtype=np.float32)
    for i in nb.prange(len(genera_idx)):
        for kmer_idx in detect_list[i]:
            if kmer_idx != 0:
                genus_count[kmer_idx, genera_idx[i]] += 1
    return genus_count


##
def calc_genus_conditional_prob_jt(detect_list: list[list[int]],
                                   genera_idx: list,
                                   kmer_size: int = 8) -> np.ndarray:

    unique_genera, genus_counts = np.unique(genera_idx, return_counts=True)
    n_genera = len(unique_genera)
    n_kmers = 4 ** kmer_size

    # Create mapping from original genera indices to contiguous 0...n_genera-1 indices
    # This ensures we can use the indices directly in our array
    genera_mapping = {g: i for i, g in enumerate(unique_genera)}
    mapped_genera = np.array([genera_mapping[g] for g in genera_idx], dtype=np.int32)

    # Update genus counts using parallelized Numba function
    genus_count = genus_counts_parallel(detect_list, mapped_genera, n_kmers, n_genera)

    return genus_count


##
if __name__ == "__main__":
    proj_dir = Path.cwd() # training data is in the top level data directory
    seq = pd.read_csv(proj_dir / "data/trainset19_072023_small_db.csv", index_col=0)
    sequences = seq.sample(1000)
    print(sequences.shape)
    ##
    kmers_size = 8
    genera_idx, kmers_list = seq_to_kmers_database(sequences, kmer_size=kmers_size)
    ##
    priors = calc_priors(kmers_list, kmers_size)
    ##
    def c_prob(kmers_list, genera_idx, kmers_size):
        n_genera = np.unique(genera_idx).shape[0]
        counts = genus_counts_parallel(kmers_list[:,1:], genera_idx, 4**kmers_size, n_genera)
        wi_pi = (counts + priors.reshape(-1, 1))
        genus_counts = np.unique(genera_idx, return_counts=True)[1]
        m_1 = (genus_counts + 1)
        wi_pi /= m_1
        return np.log(wi_pi)

    ## speed check for using numba
    start = perf_counter()
    cond_prob = c_prob(kmers_list, genera_idx, kmers_size)
    end = perf_counter()
    print(f"{end - start:.3f} s")

    ## speed check for using non-parallel version of code
    detect_list = kmers.detect_kmers_across_sequences_mp(sequences["sequence"])
    start = perf_counter()
    cond_prob_2 = kmers.calc_genus_conditional_prob(detect_list, genera_idx, priors)
    end = perf_counter()
    print(f"{end - start:.3f} s")

    ## speed check using cython version of code
    start = perf_counter()
    cond_prob_3 = cond_prob_cython.calc_genus_conditional_prob(detect_list,
                                                               np.array(genera_idx, dtype=np.int32),
                                                               priors.astype(np.float32))
    end = perf_counter()
    print(f"{end - start:.3f} s")
