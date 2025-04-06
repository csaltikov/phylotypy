#!/usr/bin/env python3

##
from collections import Counter
from functools import partial
from itertools import repeat
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from phylotypy import kmers, conditional_prob, classifier
from phylotypy.utilities import read_fasta


##
def bootstrap(arr: list | np.ndarray, divider: int = 8, num_bs: int = 100) -> NDArray:
    bootstrap_fn = partial(bootstrap_kmers, arr, divider)
    return np.array(list(map(lambda _: bootstrap_fn(), repeat(1, num_bs))))


def bootstrap_kmers(kmer_arr: np.array, kmer_size: int = 8):
    '''Performs a single bootstrap sampling on a kmers array'''
    valid_kmers = kmer_arr[kmer_arr != -1]
    num_removed_kmers = len(kmer_arr) - len(valid_kmers)
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
    rdp_small_fasta = Path("../../data/trainset19_072023_small_db.fasta")

    moving_pics = read_fasta.read_taxa_fasta("../../data/dna_moving_pictures.fasta")
    rdp_df = read_fasta.read_taxa_fasta(rdp_small_fasta)

    start = perf_counter()
    database = classifier.make_classifier(rdp_df)
    end = perf_counter()
    print(f"Time taken: {(end - start):.2f}")

    start = perf_counter()
    database1 = kmers.build_kmer_database(rdp_df["sequence"], rdp_df["id"], verbose=True)
    end = perf_counter()
    print(f"Time taken: {(end - start):.2f}")

    print(np.array_equal(database.conditional_prob, database1.conditional_prob))
    print(np.array_equal(database.genera_idx, database1.genera_idx))
    print(np.array_equal(database.genera_names, database1.genera_names))
