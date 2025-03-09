#!/usr/bin/env python3
from pathlib import Path
import itertools
from time import perf_counter

import numpy as np
import pandas as pd
import jax.numpy as jnp

from phylotypy import kmers

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, verbose=1)


def build_database(sequences, kmer_size, **kwargs):
    id_col = kwargs.get('id_col', "id")

    genera = sequences[id_col]
    genera_idx = kmers.genera_str_to_index(genera)
    genera_names = kmers.index_genus_mapper(genera)

    detected_kmers = make_kmers_database(sequences, kmer_size=kmer_size)
    priors = calc_priors(detected_kmers, kmer_size)

    cond_prob_arr = GenusCondProb(detected_kmers, priors, kmer_size).calculate()

    return kmers.KmerDB(conditional_prob=cond_prob_arr,
                        genera_idx=genera_idx,
                        genera_names=genera_names)


def parallel_conditional_prob(kmers_df: pd.DataFrame, priors) -> np.ndarray:
    genera_idx = kmers.genera_str_to_index(kmers_df["id"])
    unique_genus_counts = np.unique(genera_idx, return_counts=True)[1]
    n_genera = len(unique_genus_counts)
    n_kmers = len(priors)

    ## Building genus conditional probabilities
    def join_list(df):
        return list(itertools.chain.from_iterable(df["kmers_list"]))

    def counts_idx(s):
        return np.unique(s, return_counts=True, return_index=False)

    print("Starting the calculation")
    results = (kmers_df.groupby("id")
               .parallel_apply(join_list)
               .reset_index(name="kmers_list")
               .loc[:, "kmers_list"]
               .parallel_apply(counts_idx)
               )

    ##
    # numpy way:
    print("Filling in the genus_count array")
    genus_count = np.zeros((n_kmers, n_genera), dtype=np.float32)

    for i, (idx, counts) in enumerate(results):
        genus_count[idx, i] += counts

    wi_pi = (genus_count + priors.reshape(-1, 1))
    m_1 = (unique_genus_counts + 1)

    print("Final calculation...")
    genus_cond_prob = np.log(np.divide(wi_pi, m_1)).astype(np.float16)

    return genus_cond_prob


def fix_kmers_length(kmer_arr, seq_len: int = 1400):
    n_missing = seq_len - len(kmer_arr)
    return np.concatenate([kmer_arr, np.full(n_missing, -1, dtype=int)])


## New code
def calc_priors(detected_kmers: np.ndarray, kmer_size: int = 8):
    num_seqs = detected_kmers.shape[0]
    max_value = 4 ** kmer_size

    flat_kmers = detected_kmers.flatten()
    flat_kmers = flat_kmers[flat_kmers !=-1]
    counts = np.bincount(flat_kmers, minlength=max_value)

    # expected-likelihood estimate using Jeffreys-Perks law of succession
    # 0 < Pi < 1
    return (counts + 0.5) / (num_seqs + 1)


##
def make_kmers_database(sequences_db, **kwargs):
    seq_col = kwargs.get('seq_col', 'sequence')
    kmer_col = kwargs.get('kmer_col', 'kmers')
    id_col = kwargs.get('id_col', 'id')
    kmer_size = kwargs.get('kmer_size', 8)

    db = sequences_db

    if isinstance(sequences_db, str):
        if ".csv" in Path(sequences_db).suffix:
            db = pd.read_csv(sequences_db)
        if ".tsv" in Path(sequences_db).suffix:
            db = pd.read_csv(sequences_db, sep="\t")

    # db[kmer_col] = kmers.detect_kmers_across_sequences_mp(db[seq_col])
    db[kmer_col] = db[seq_col].parallel_apply(kmers.detect_kmers, kmer_size=kmer_size)
    # Calculate max length
    max_seq_len = db[kmer_col].str.len().max().astype(int)

    # apply function that makes the list objects the same length
    db[kmer_col] = db[kmer_col].apply(lambda x: fix_kmers_length(np.array(x), max_seq_len))
    db["idx"] = kmers.genera_str_to_index(db[id_col])

    # Create final array first column are the sequence indices
    all_kmers_arr = np.hstack((
        db["idx"].to_numpy().reshape(-1, 1),
        np.stack(db[kmer_col].to_numpy())
    ))
    return all_kmers_arr


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
        uniq_idx, uniq_idx_counts = np.unique(self.kmers_arr[:, 0], return_counts=True)  # first column are the seq ids
        genus_kmer_counts = np.zeros((4 ** self.kmer_size, uniq_idx.shape[0]), dtype=int)  # id, kmer_indices, counts

        for uniq in uniq_idx:
            kmers_arr_ = self.kmers_arr[self.kmers_arr[:, 0] == uniq, 1:]  # col 0 are the sequence ids
            # kmers_arr_ = kmers_arr_[kmers_arr_ != -1]  # remove -1, it's a placeholder for emtpy space
            # idx, counts = np.unique(kmers_arr_, return_counts=True)
            # genus_kmer_counts[idx[1:], uniq] = counts[1:]
            kmers_arr_ = kmers_arr_[kmers_arr_ != -1].flatten()
            genus_kmer_counts[:, uniq] = np.bincount(kmers_arr_[1:], minlength=4**self.kmer_size)

        print(f"Done making the genus kmers counts array {genus_kmer_counts.shape}")
        self.wi_pi = (genus_kmer_counts + self.priors.reshape(-1, 1))
        self.m_1 = (uniq_idx_counts + 1)

    def calculate(self):
        print("Computing the log probability matrix")
        self.calculate_genus_counts()
        divided = jnp.divide(self.wi_pi, self.m_1)
        return jnp.log(divided)


##
if __name__ == "__main__":
    ##
    home_dir = Path.home()

    sequences = pd.read_csv(home_dir / "PycharmProjects/phylotypy/data/trainset19_072023_small_db.csv")
    print(sequences.head())

    kmer_size = 8
    # kmers_db = make_kmers_database(rdp, kmer_size=kmers_size)
    # priors = calc_priors(kmers_db, kmers_size)
    #
    # cond_prob_arr = GenusCondProb(kmers_db, priors, kmers_size).calculate()
    # print(cond_prob_arr.shape)
    ##
    start = perf_counter()
    database = build_database(sequences, kmer_size=kmer_size)
    end = perf_counter()
    print(f"{end-start:.2f}")
    #
    # ##
    # start = perf_counter()
    # database2 = kmers.build_kmer_database(sequences["sequence"], sequences["id"])
    # end = perf_counter()
    # print(f"{end-start:.2f}")

    ##
    # kmer_size = 8
    # detected_kmers = make_kmers_database(sequences, kmer_size=kmer_size)
    # start = perf_counter()
    # priors = calc_priors(detected_kmers)
    # end = perf_counter()
    # print(f"{end-start:.2f}")
