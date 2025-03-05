#!/usr/bin/env python3
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from phylotypy import kmers
from phylotypy.utilities import read_fasta

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, verbose=1)


def parallel_conditional_prob(kmers_df: pd.DataFrame, priors) -> np.ndarray:
    genera_idx = kmers.genera_str_to_index(kmers_df["id"])
    unique_genus_counts = np.unique(genera_idx, return_counts=True)[1]
    n_genera = len(unique_genus_counts)
    n_kmers = len(priors)

    ## Building genus conditional probablities
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
    n_missing = seq_len - kmer_arr.shape[0]
    return np.concatenate([kmer_arr, np.full(n_missing, -1, dtype=int)])


## New code
def calc_priors(detected_kmers: np.ndarray, kmer_size: int = 8):
    num_seqs = detected_kmers.shape[0]
    priors_arr = np.zeros(4 ** kmer_size)
    for idx_list in detected_kmers:
        priors_arr[idx_list] += 1

    # expected-likelihood estimate using Jeffreys-Perks law of succession
    # 0 < Pi < 1
    return (priors_arr + 0.5) / (num_seqs + 1)

##
def make_kmers_database(sequences_db, **kwargs):
    seq_col = kwargs.get('seq_col', 'sequence')
    kmer_col = kwargs.get('kmer_col', 'kmers')
    id_col = kwargs.get('id_col', 'id')
    kmer_size = kwargs.get('kmer_size', 8)

    if isinstance(sequences_db, str):
        if ".csv" in Path(sequences_db).suffix:
            db = pd.read_csv(sequences_db)
    else:
        db = sequences_db

    # db[kmer_col] = kmers.detect_kmers_across_sequences_mp(db[seq_col])
    db[kmer_col] = db[seq_col].parallel_apply(kmers.detect_kmers, kmer_size=kmer_size)
    # Calculate max length
    max_seq_len = db[kmer_col].str.len().max().astype(int)

    # apply function that makes the list objects the same length
    db[kmer_col] = db[kmer_col].apply(lambda x: fix_kmers_length(np.array(x), max_seq_len))
    db["idx"] = kmers.genera_str_to_index(db[id_col])

    # Create final array
    all_kmers_arr = np.hstack((
        db["idx"].to_numpy().reshape(-1, 1),
        np.stack(db[kmer_col].to_numpy())
    ))
    return all_kmers_arr


##
class GenusCondProb:
    '''Calculates the geneus conditional probability matrix
    for a corpus of kmer indices of all the unique genera'''
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
            idx, counts = np.unique(kmers_arr_, return_counts=True)
            genus_kmer_counts[idx[1:], uniq] = counts[1:]

        print(f"Done making the genus kmers counts array {genus_kmer_counts.shape}")
        self.wi_pi = (genus_kmer_counts + self.priors.reshape(-1, 1))
        self.m_1 = (uniq_idx_counts + 1)

    def calculate(self):
        print("Computing the log probability matrix")
        self.calculate_genus_counts()
        divided = np.divide(self.wi_pi, self.m_1)
        log_trans = np.log(divided)
        return log_trans


##
if __name__ == "__main__":
    test_fasta = read_fasta.read_taxa_fasta("../../data/test_fasta.fa")
    print(test_fasta)

    kmers_size = 5
    kmers_db = make_kmers_database(test_fasta, kmer_size=kmers_size)
    priors = calc_priors(kmers_db, kmers_size)

    cond_prob_arr = GenusCondProb(kmers_db, priors, kmers_size).calculate()
    print(cond_prob_arr.shape)

