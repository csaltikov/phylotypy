#!/usr/bin/env python3
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import numba as nb

from phylotypy import kmers
from phylotypy import cond_prob_cython

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False, verbose=1)


def build_database(sequences, kmer_size: int = 8, **kwargs):
    """
    Builds a k-mer database for genetic sequence classification by calculating genus-specific conditional
    probabilities, priors, and mapping genus names to indices.

    Args:
        sequences: Input genetic sequences provided as a DataFrame containing sequence data and metadata.
            Must include an identifier column specified by the `id_col`.
        kmer_size: Size of k-mers to generate from the input sequences. Represents the window size for
            slicing sequences into smaller subsequences.
        **kwargs: Optional keyword arguments:
            - id_col (str): Name of the column in `sequences` DataFrame identifying genera. Defaults to "id".

    Returns:
        KmerDB: A database object containing calculated conditional probabilities, genus indices, and genus names.
    """
    id_col = kwargs.get('id_col', "id")

    genera = sequences[id_col]
    genera_names = kmers.index_genus_mapper(genera)

    genera_idx, detected_kmers = seq_to_kmers_database(sequences, kmer_size=kmer_size)
    priors = calc_priors(detected_kmers, kmer_size)

    cond_prob_arr = GenusCondProb(detected_kmers, priors, kmer_size).calculate()

    return kmers.KmerDB(conditional_prob=cond_prob_arr,
                        genera_idx=genera_idx,
                        genera_names=genera_names)


def calc_priors(detected_kmers: np.ndarray, kmer_size: int = 8):
    num_seqs = len(detected_kmers)
    max_value = 4 ** kmer_size

    flat_kmers = detected_kmers[:, 1:].flatten() # ignore col 0, row indices
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
def seq_to_kmers_database(sequences_db, **kwargs):
    seq_col = kwargs.get('seq_col', 'sequence')
    id_col = kwargs.get('id_col', 'id')
    kmer_size = kwargs.get('kmer_size', 8)

    db = sequences_db

    if isinstance(sequences_db, str):
        if ".csv" in Path(sequences_db).suffix:
            db = pd.read_csv(sequences_db)
        if ".tsv" in Path(sequences_db).suffix:
            db = pd.read_csv(sequences_db, sep="\t")

    kmer_series = db[seq_col].parallel_apply(lambda x: kmers.detect_kmer_indices(x, k=kmer_size))
    # Calculate max length
    max_seq_len = kmer_series.str.len().max().astype(int)

    # apply function that makes the list objects the same length
    detected_kmers = kmer_series.parallel_apply(lambda x: fix_kmers_length(np.array(x), max_seq_len))
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
    proj_dir = Path("../../") # training data is in the top level data directory
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
