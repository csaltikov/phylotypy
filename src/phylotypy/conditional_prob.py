#!/usr/bin/env python3
from pathlib import Path
from time import perf_counter


import numpy as np
import pandas as pd

from phylotypy import kmers

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


## New code
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
    # kmer_list = db[seq_col].parallel_apply(kmers.detect_kmers, kmer_size=kmer_size)
    kmer_list = kmers.detect_kmers_across_sequences_mp(db[seq_col], kmer_size=kmer_size)
    kmer_series = pd.Series(kmer_list)
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
    ))
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
        return np.log(divided)

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
        return np.log(divided)


##
if __name__ == "__main__":
    ##
    import timeit

    home_dir = Path.home()

    sequences = pd.read_csv(home_dir / "PycharmProjects/phylotypy/data/trainset19_072023_small_db.csv", index_col=0)
    print(sequences.head())
    ##
    kmers_size = 8
    kmers_db = seq_to_kmers_database(sequences, kmer_size=kmers_size)

    make_class = timeit.timeit(lambda: seq_to_kmers_database(sequences, kmer_size=kmers_size), number=5)
    print(make_class)

    #
    # ##
    # priors = calc_priors(kmers_db, kmers_size)
    # start = perf_counter()
    # cond_prob_arr = GenusCondProb(kmers_db, priors, kmers_size).calculate_alt()
    # end = perf_counter()
    # print(cond_prob_arr.shape)
    # print(f"{end-start:.2f}")
    #
    # ##
    # start = perf_counter()
    # cond_prob_arr = GenusCondProb(kmers_db, priors, kmers_size).calculate()
    # end = perf_counter()
    # print(cond_prob_arr.shape)
    # print(f"{end-start:.2f}")

    ##
    start = perf_counter()
    database = build_database(sequences, kmer_size=kmers_size)
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

    import matplotlib.pyplot as plt
    from pathlib import Path
    from phylotypy import classifier, results, kmers, conditional_prob
    from phylotypy.utilities import read_fasta

    rdp_fasta = Path("data/rdp_16S_v19.dada2.fasta")
    moving_pics = read_fasta.read_taxa_fasta("data/dna_moving_pictures.fasta")
    rdp_df = read_fasta.read_taxa_fasta(rdp_fasta)

    kmer_arr = conditional_prob.seq_to_kmers_database(rdp_df)
    priors = conditional_prob.calc_priors(kmer_arr)

