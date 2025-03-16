#!/usr/bin/env python3

##
from collections import Counter
from functools import partial
from itertools import repeat
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pandas as pd

import numba as nb

from phylotypy import kmers, conditional_prob, classifier
from phylotypy.utilities import read_fasta


##
def bootstrap(arr: list | np.ndarray, divider: int = 8, num_bs: int = 100) -> NDArray:
    bootstrap_fn = partial(bootstrap_kmers, arr, divider)
    return np.array(list(map(lambda _: bootstrap_fn(), repeat(1, num_bs))))


def bootstrap_kmers(kmers: np.array, kmer_size: int = 8):
    '''Performs a single bootstrap sampling on a kmers array'''
    valid_kmers = kmers[kmers != -1]
    num_removed_kmers = len(kmers) - len(valid_kmers)
    n_kmers = len(valid_kmers) + num_removed_kmers
    return np.random.choice(valid_kmers, n_kmers // kmer_size, replace=True)


def split_taxa_arr(taxa_arr: np.ndarray) -> np.ndarray:
    """
    Split array of taxonomy strings into a 2D array where each row is a
    sample and each column is a taxonomic level.
    """
    return np.array([taxa_str.split(";") for taxa_str in taxa_arr])


def bootstrap_consensus_helper(arr):
    ids, scores = np.apply_along_axis(kmers.get_consensus, axis=0, arr=arr)
    return ids, scores


def sort_by_indices_helper(values_to_sort, indices_array):
    sorted_indices = np.argsort(indices_array)
    return values_to_sort[sorted_indices]


##
@nb.jit(nopython=True, parallel=True)
def classify_bootstraps_numba(bs_indices, conditional_prob):
    classifications = np.empty(bs_indices.shape[0], dtype=np.int64)

    for i in nb.prange(bs_indices.shape[0]):
        sums = np.sum(conditional_prob[bs_indices[i]], axis=0)
        classifications[i] = np.argmax(sums)

    return classifications


##
def bootstrap_consensus(classified_bs_kmers: np.ndarray, genera_names: np.ndarray):
    """
    Find consensus taxonomy from bootstrap samples.

    Parameters:
    classified_bs_kmers: 1D array of integers representing bootstrap samples
    database: Database object containing genera_names mapping;
                database = {conditional_prob:np.array
                        genera_idx mapping}

    Returns:
    Dictionary with taxonomy consensus and confidence values
    """
    # Convert the 1D array of ints (classified_bs_kmers) to taxonomy strings
    # using genera_names 1D array
    taxa = genera_names[classified_bs_kmers]

    # Split into a 2D array of (100,6)
    res = split_taxa_arr(taxa)
    n_levels = res.shape[1]

    taxa_consensus = np.empty(n_levels, dtype=object)
    confidence_consensus = np.zeros(n_levels, dtype=int)

    # Use Counter directly instead of apply_along_axis with custom function
    for i in range(n_levels):
        counter = Counter(res[:, i])
        most_common = counter.most_common(1)[0]  # Returns (taxon, count)
        taxa_consensus[i] = most_common[0]
        confidence_consensus[i] = most_common[1]

    return dict(taxonomy=taxa_consensus, confidence=confidence_consensus)


##
if __name__ == "__main__":
    from time import perf_counter

    print(Path().absolute())
    rdp_fasta = Path("../../data/rdp_16S_v19.dada2.fasta")
    rdp_small = Path("../../data/trainset19_072023_small_db.csv")

    moving_pics = read_fasta.read_taxa_fasta("../../data/dna_moving_pictures.fasta")
    rdp_df = read_fasta.read_taxa_fasta(rdp_fasta)
    rdp_small_df = pd.read_csv(rdp_small, index_col=0)

    ##
    database = conditional_prob.build_database(rdp_small_df)
    ##
    # seqs = rdp_small_df.sample(500, random_state=1234)

    classified = classifier.classify_sequences(moving_pics, database)

    ##
    # from phylotypy import conditional_prob
    # from classify_bootstraps import classify_bootstraps_cython
    # from phylotypy import kmers
    # from collections import defaultdict
    # ##
    # classified_seqs = defaultdict(list)
    # genera_idx_test, detected_kmers_test = conditional_prob.seq_to_kmers_database(moving_pics)
    # for i, idx in enumerate(genera_idx_test):
    #     print(i)
    #     seq_kmer = detected_kmers_test[detected_kmers_test[:,0]==idx, 1:].flatten()
    #     name = seqs.iloc[i]["id"]
    #     classified_seqs["id"].append(name)
    #     bootstrapped = bootstrap(seq_kmer)
    #     classified_kmers = classify_bootstraps_cython(bootstrapped, database.conditional_prob)
    #     consensus = bootstrap_consensus(classified_kmers, database.genera_names)
    #     filtered = filter_taxonomy(consensus)
    #     classified_seqs["taxonomy"].append(kmers.print_taxonomy(filtered))
    #     # classified["classification"].append(classify_sequence(seq_kmer, database))

    # # res = pd.DataFrame(classified)
