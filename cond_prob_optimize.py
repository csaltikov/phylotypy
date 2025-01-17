##
import functools
from collections import defaultdict, Counter
import itertools
import logging
import multiprocessing as mp
import os
from time import perf_counter
import timeit

import numpy as np

import pandas as pd

import kmers

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)


##
def calc_genus_conditional_prob(detect_list: list,
                                genera_idx: list,
                                word_specific_priors: np.array,
                                verbose: bool = False) -> np.array:
    if verbose:
        print("Calculating genus conditional probability")
    # genus_arr = np.array(genera)  # indices not the taxa names
    genus_counts = np.unique(genera_idx, return_counts=True)[1]  # get counts of each unique genera
    n_genera = len(genus_counts)
    n_sequences = len(genera_idx)
    n_kmers = len(word_specific_priors)

    # Create an array with zeros, rows are kmer indices, columns number of unique genera
    genus_count = np.zeros((n_kmers, n_genera), dtype=np.float32)

    # loop through the incoming genera
    # i is a specific organism
    # get the list of kmer indices and fill in 1 or 0
    for i in range(n_sequences):
        genus_count[detect_list[i], genera_idx[i]] = genus_count[detect_list[i], genera_idx[i]] + 1
        # np.add.at(genus_count, (detect_list[i], genera[i]), 1)

    wi_pi = genus_count + word_specific_priors.reshape(-1, 1)
    m_1 = (genus_counts + 1)

    genus_cond_prob = np.log((np.divide(wi_pi, m_1)).astype(np.float16))

    return genus_cond_prob


def calc_genus_cond_prob_opt(detect_list: list,
                             genera_idx: list,
                             word_specific_priors: np.array) -> np.array:
    """
    Calculate genus conditional probability

    Args:
        detect_list: a list containing a list of kmer indices of varying lengths
        genera_idx: list of genera index numbers
        word_specific_priors: 1D np.array of length 4**kmer_size default is 8

    Returns:
        genus_cond_prob: np.array of the genus conditional probability
    """
    genus_counts = np.unique(genera_idx, return_counts=True)[1]
    n_genera = len(genus_counts)
    n_kmers = len(word_specific_priors)

    genus_count = np.zeros((n_kmers, n_genera), dtype=np.int16)

    for i in range(len(genera_idx)):
        np.add.at(genus_count, (detect_list[i], genera_idx[i]), 1)

    wi_pi = genus_count + word_specific_priors.reshape(-1, 1)
    m_1 = (genus_counts + 1)

    genus_cond_prob = np.log((wi_pi / m_1).astype(np.float16))

    return genus_cond_prob


def detect_kmers_across_sequences(sequences: list, kmer_size: int = 8, verbose: bool = False) -> list:
    """Find all kmers for a list of nucleotide sequences"""
    if verbose:
        print("Detecting kmers across sequences")
    n_sequences = len(sequences)
    kmer_list = []
    for i, seq in enumerate(sequences):
        kmer_list.append(kmers.detect_kmers(seq, kmer_size))
    return kmer_list


def pool_detect_kmers_across_seqs(sequences: list,
                                  kmer_size: int = 8,
                                  num_processes: int = 4) -> np.array:
    with mp.Pool(num_processes) as pool:
        args_list = [(seq, kmer_size) for seq in sequences]
        collected_results = pool.starmap_async(kmers.detect_kmers, args_list)
        results = collected_results.get()
    return results


##
if __name__ == "__main__":
    ##
    refs = pd.read_csv('data/trainset19_072023_db.csv')

    ref_seqs = refs.sample(n=5000, replace=False, random_state=2025)

    sequences = ref_seqs["sequences"]
    genera = ref_seqs["taxonomy"]
    kmer_size = 8

    start = perf_counter()
    detected_kmers = detect_kmers_across_sequences(sequences, kmer_size=kmer_size, verbose=True)
    end = perf_counter()
    print(f"Standard Time taken: {(end - start):.2f}")

    priors = kmers.calc_word_specific_priors(detected_kmers, kmer_size=kmer_size, verbose=True)
    genera_idx = kmers.genera_str_to_index(genera)

    ##
    kmers_arr = np.empty((len(detected_kmers), 1), dtype=object)
    kmers_arr[:, 0] = detected_kmers
    np.savez('data/sample_5000.npz',
             priors=priors,
             detected_kmers=kmers_arr,
             genera_idx=np.array(genera_idx),
             allow_pickle=True
             )
    ##
    db_arr = np.load('data/sample_5000.npz', allow_pickle=True)
    detected_kmers_loaded = db_arr['detected_kmers'].flatten().tolist()

    ##
    cond_prob = functools.partial(calc_genus_cond_prob_opt,
                                  detect_list=detected_kmers,
                                  genera_idx=genera_idx,
                                  word_specific_priors=priors)
    res = timeit.timeit(cond_prob, number=5)/5
    print(res)
