from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Any, Type

import multiprocessing as mp
import platform

if platform.system() == "Darwin":
    mp.set_start_method('spawn', force=True)

import re

import numpy as np
from numpy.dtypes import StringDType

from phylotypy import _worker_pool

'''
Naive Bayes Classifier for DNA sequences. The project is inspired by the 
following paper. The approach is to convert a DNA sequence to kmers and 
determine the probability of each kmer in the sequence.

Wang Q, Garrity GM, Tiedje JM, Cole JR. 
Naive Bayesian classifier for rapid assignment of rRNA sequences into 
the new bacterial taxonomy. 
Appl Environ Microbiol. 2007 Aug;73(16):5261-7. 
doi: 10.1128/AEM.00062-07. Epub 2007 Jun 22.
PMID: 17586664; PMCID: PMC1950982.
https://pubmed.ncbi.nlm.nih.gov/17586664/
'''


@dataclass(slots=True)
class KmerDB:
    conditional_prob: np.ndarray
    genera_idx: list
    genera_names: np.ndarray

    data: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.data = {
            "conditional_prob": self.conditional_prob,
            "genera_idx": self.genera_idx,
            "genera_names": self.genera_names
        }



@dataclass
class KmerIndices:
    """CSR-style compressed kmer storage — replaces list of lists."""
    indices: np.ndarray   # flat int32 array of all kmer indices
    offsets: np.ndarray   # int32 array of length n_sequences+1

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i):
        return self.indices[self.offsets[i]:self.offsets[i+1]]


def build_kmer_database(sequences: list[str], genera: list[str],
                        kmer_size: int = 8, verbose: bool = False,
                        num_processes: int = mp.cpu_count(),
                        **kwargs) -> KmerDB:
    """Creates a conditional probability matrix from DNA sequences and associated genera.

    This function processes DNA sequences to build a k-mer database that can be used
    for taxonomic classification. It supports both single-process and multi-process
    k-mer detection.

    Args:
        sequences: List of DNA sequences as strings (e.g., ["ATCGGA", "ATCGGA"])
        genera: List of genera names for building a model or sequence IDs for prediction
        kmer_size: Size of DNA subsequences to analyze (default: 8 nucleotides)
        use_multiprocessing: Whether to use parallel processing (default: True)
        num_processes: Number of processes to use in parallel mode (default: CPU count)
        verbose: Enable progress output (default: False)

    Returns:
        KmerDB: Database containing conditional probabilities and genera information
    """
    m_proc = kwargs.get('m_proc', True)
    if m_proc:
        detected_kmers = detect_kmers_across_sequences_mp(sequences,
                                                          kmer_size=kmer_size,
                                                          num_processes=num_processes,
                                                          verbose=verbose)
    else:
        detected_kmers = detect_kmers_across_sequences(sequences,
                                                       kmer_size=kmer_size,
                                                       verbose=verbose)

    priors = calc_word_specific_priors(detected_kmers, kmer_size=kmer_size, verbose=verbose)

    genera_idx = genera_str_to_index(genera)

    cond_prob = calc_genus_conditional_prob(detected_kmers,
                                               genera_idx,
                                               priors,
                                               verbose=verbose)

    genera_names = index_genus_mapper(genera)

    return KmerDB(conditional_prob=cond_prob,
                  genera_idx=genera_idx,
                  genera_names=genera_names)


def get_all_kmers(sequence: str, kmer_size: int = 8) -> list:
    return [sequence[i: i + kmer_size] for i in range(len(sequence) - kmer_size + 1)]


def seq_to_base4(sequence: str | list):
    def convert_dna(dna_seq):
        dna_seq = dna_seq.upper()
        dna_seq = re.sub(r"[^ACGT]", "N", dna_seq)
        dna = "ACGT"
        base4 = "0123"
        translation_mapping = dna_seq.maketrans(dna, base4)
        return dna_seq.translate(translation_mapping)

    # check if 'sequence' is list or string
    if isinstance(sequence, str):
        return convert_dna(sequence)
    elif isinstance(sequence, list):
        converted = []
        for seq in sequence:
            converted.append(convert_dna(seq))
        return converted
    else:
        raise ValueError("Input should be a list or string")


def base4_to_index(base4_str: list) -> list:
    """Converts base4 string to a numpy array of indices but ignores kmers with an N"""
    converted_list: list = [int(item, 4) for item in base4_str if "N" not in item]
    return converted_list


def detect_kmers(sequence: str, kmer_size: int = 8) -> np.ndarray:
    """Detects kmers in a DNA sequence"""
    # Converts ACGT sequence data to base4
    kmers_: list = get_all_kmers(seq_to_base4(sequence), kmer_size)
    # Detected kmer indices, base 10, which are positions in a matrix
    kmers_ = base4_to_index(kmers_)
    return np.unique(kmers_).astype(np.int32)


def detect_kmer_indices(sequence: str, k: int=8) -> np.ndarray:
    base4_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.array([base4_map.get(nuc, -1) for nuc in sequence], dtype=np.int8)
    if arr.size < k:
        return np.array([], dtype=np.int32)  # Return empty array for short sequences

    # Create a sliding window view
    shape = (arr.size - k + 1, k)
    strides = (arr.strides[0], arr.strides[0])
    kmers = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    exclude = np.any(kmers == -1, axis=1)

    # Convert to base10 indices
    valid_kmers = kmers[~exclude]
    if valid_kmers.shape[0] == 0:
        return np.array([], dtype=np.int32)

    powers = 4 ** np.arange(k - 1, -1, -1)
    valid_kmers = np.dot(valid_kmers, powers)
    return np.unique(valid_kmers).astype(np.int32)


def detect_kmers_across_sequences(sequences: list, kmer_size: int = 8, verbose: bool = False) -> list:
    """Find all kmers for a list of nucleotide sequences"""
    if verbose:
        print("Detecting kmers across sequences (standard)")
    
    n_sequences = len(sequences)
    offsets = np.zeros(n_sequences + 1, dtype=np.int32)
    results: list = [None] * n_sequences
    
    for i, seq in enumerate(sequences):
        r = detect_kmer_indices(seq, k=kmer_size)
        results[i] = r
        offsets[i+1] = offsets[i] + len(r)
        
    indices = np.empty(offsets[-1], dtype=np.int32)
    for i, r in enumerate(results):
        indices[offsets[i]:offsets[i+1]] = r
    
    del results
    
    if verbose:
        print("Done detecting kmers")
    return KmerIndices(indices=indices, offsets=offsets)


def detect_kmers_across_sequences_mp(sequences: list,
                                     kmer_size: int = 8,
                                     num_processes: int = 4,
                                     verbose: bool = False) -> KmerIndices:
    if verbose:
        print("Detecting kmers across sequences mp")
        print(f"Kmer size is set to {kmer_size}")

    n_sequences = len(sequences)
    # Cheap upper bound on total kmer indices (no kmer detection needed to compute it),
    # so the output buffer can be preallocated instead of materializing every
    # per-sequence result in a list before compacting them.
    upper_bound = sum(max(0, len(seq) - kmer_size + 1) for seq in sequences)

    pool = _worker_pool.get_pool(num_processes)
    func = partial(detect_kmers, kmer_size=kmer_size)
    chunksize = max(1, n_sequences // (num_processes * 4))

    indices = np.empty(upper_bound, dtype=np.int32)
    offsets = np.zeros(n_sequences + 1, dtype=np.int32)

    cursor = 0
    for i, r in enumerate(pool.imap(func, sequences, chunksize=chunksize)):
        n = len(r)
        indices[cursor:cursor + n] = r
        cursor += n
        offsets[i + 1] = cursor

    # upper_bound over-counts kmers dropped by N-filtering/dedup (~2% in practice);
    # a view is enough to trim to the real total, no need to pay for a full copy.
    indices = indices[:cursor]

    if verbose:
        print("Done detecting kmers")
    return KmerIndices(indices=indices, offsets=offsets)


def calc_word_specific_priors(detected_kmers_list: list,
                              kmer_size: int = 8,
                              verbose: bool = False) -> np.ndarray:
    """
    Calculates the prior likelihoods for each kmer in the coprus of reference sequences

    Args:
        detected_kmers_list: a list of list where each sublist is a list of all kmers within a sequence
        kmer_size: size of the kmer, e.g. 8 nucleotides if kmer_size = 8
        verbose: do you want to see the progress

    Returns:
        a 1D array of prior probabilities of size 4^kmer_size
    """
    if verbose:
        print("Calculating word specific priors")
    # For the list of list to subset each list to just the unique kmers within each list
    # detected_kmers_list = [np.unique(kmer_indices) for kmer_indices in detected_kmers_list]

    n_seqs = len(detected_kmers_list)  # the corpus of N sequences
    priors = np.zeros(4 ** kmer_size)

    for i in range(n_seqs):
        idx_list = detected_kmers_list[i]
        priors[idx_list] +=1
    priors = (priors + 0.5) / (n_seqs + 1)
    return priors


def calc_genus_conditional_prob(detect_list: list[list[int]],
                                genera_idx: list,
                                word_specific_priors: np.ndarray,
                                verbose: bool = False) -> np.ndarray:
    if verbose:
        print("Calculating genus conditional probability")
    # genus_arr = np.array(genera_idx)  # indices not the taxa names
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
    # Calculate the likelihood for a genus to have a specific kmer
    # (m(wi) + Pi) / (M + 1)
    wi_pi = (genus_count + word_specific_priors.reshape(-1, 1))
    m_1 = (genus_counts + 1)

    divided = np.divide(wi_pi, m_1)
    genus_cond_prob = np.log(divided).astype(np.float32)

    return genus_cond_prob


def genera_str_to_index(genera: list) -> list:
    """Find unique genera, and convert the genera names to indices"""
    unique_genera = np.unique(genera)
    factor_map = {genus: idx for idx, genus in enumerate(unique_genera)}
    # Convert genera to factors using the mapping
    genera_factors = [factor_map[val] for val in genera]
    return genera_factors


def index_genus_mapper(genera_list: list):
    # Create a dictionary mapping unique values to integers
    unique_genera = np.unique(genera_list)
    # return factor_map
    return unique_genera


def consensus_bs_class(bs_class: np.array, genera_names) -> dict[str, list | Any]:
    """Convert the indices in the bootstrap array to taxonomy"""
    taxonomy: np.array = genera_names[bs_class]
    # sometimes taxonomy is empty or has "none" value
    mask = taxonomy != None
    taxonomy_filtered = taxonomy[mask]

    taxonomy_split = np.array([line.split(";") for line in taxonomy_filtered])

    def cumulative_join(col):
        join_taxa = [";".join(col[:i + 1]) for i in range(len(col))]
        return np.array(join_taxa, dtype=StringDType) #'<U300'

    taxa_cum_join_arr = np.apply_along_axis(cumulative_join, 1, taxonomy_split)

    taxa_string, confidence = np.apply_along_axis(get_consensus, axis=0, arr=taxa_cum_join_arr)

    return dict(taxonomy=np.array(taxa_string[-1].split(";")),
                confidence=confidence)


def get_consensus(taxa_cumm_join_arr: np.ndarray):
    """Helper for consensus_bs_class to find the best taxon and confidence level"""
    # get best ID and score for each column of the taxa array
    taxonomy, counts = np.unique(taxa_cumm_join_arr, return_counts=True)
    max_id = np.argmax(counts)

    id_fraction_arr = np.full(2, fill_value=["unclassified", 0], dtype=list)

    fraction = counts[max_id] / counts.sum()

    id_fraction_arr[0] = taxonomy[max_id]
    id_fraction_arr[1] = int(100 * fraction)

    return id_fraction_arr


def filter_taxonomy(classification: dict, min_confidence: float = 80) -> Dict:
    """Helper for create_con determines finds the best id"""

    high_confidence = np.where(classification["confidence"] >= min_confidence)[0]

    # in case all confidence scores are below min_confidence
    if high_confidence.size == 0:
        first_taxon = classification["taxonomy"][0]
        first_confidence = classification["confidence"][0]
        taxonomy = np.array([first_taxon], dtype=StringDType)
        confidence =  np.array([first_confidence])
    else:
        taxonomy = classification["taxonomy"][high_confidence]
        confidence = classification["confidence"][high_confidence]
    return dict(
        taxonomy=taxonomy,
        confidence=confidence
    )


def print_taxonomy(consensus: dict, n_levels=6) -> str:
    original_levels = len(consensus["taxonomy"])
    given_levels = original_levels
    extra_levels = n_levels - given_levels

    taxa_idx = np.arange(original_levels)

    last_taxa = consensus["taxonomy"][taxa_idx[-1]]
    last_confidence = consensus["confidence"][taxa_idx[-1]]
    unclassified = f"{last_taxa}_unclassified"

    taxonomy = np.concatenate((consensus["taxonomy"], [unclassified] * extra_levels), axis=0)
    confidence = np.concatenate((consensus["confidence"], [last_confidence] * extra_levels), axis=0)

    # Construct the classification string
    updated_classification = [f"{taxa}({conf})" for taxa, conf in zip(taxonomy, confidence.astype(int))]

    return ";".join(updated_classification)


def print_taxonomy_unsplit(taxonomy, n_levels=6):
    taxonomy_split: list = re.findall(r'[^;]+', taxonomy)
    n_taxa_levels = len(taxonomy_split)
    last_taxa = taxonomy[-1]
    updated_taxonomy = taxonomy_split + [f"{last_taxa}_unclassified"] * (n_levels - n_taxa_levels)
    return ";".join(updated_taxonomy)


def fix_taxonomy(taxa_string: str, n_levels: int = 6) -> str:
    taxa_string = taxa_string.rstrip(";")
    original_levels = taxa_string.count(";") + 1
    given_levels = original_levels
    extra_levels = n_levels - given_levels
    taxonomy = taxa_string.split(";")
    last_taxa = taxonomy[-1]
    unclassified = f"{last_taxa}_unclassified"
    taxonomy = taxonomy + [unclassified] * extra_levels
    updated_classification = ";".join(taxonomy)
    return updated_classification


def get_ref_genera(genera: list):
    genera_names = index_genus_mapper(genera)
    return genera_names


def base10_base4(kmer: int, kmer_size: int = 8) -> str:
    base4: str = ""
    i = 1
    while i <= kmer_size:
        kmer, remainder = divmod(kmer, 4)
        base4 += str(remainder)
        i += 1
    return base4[::-1]


def base4_to_nucleotide(base4_seq: str | list):
    def convert_base4(base4_str: str):
        dna = "ACGT"
        base4 = "0123"
        translation_mapping = base4_str.maketrans(base4, dna)
        return base4_str.translate(translation_mapping)

    # check if 'sequence' is list or string
    if isinstance(base4_seq, str):
        return convert_base4(base4_seq)
    elif isinstance(base4_seq, list):
        converted = []
        for seq in base4_seq:
            converted.append(convert_base4(seq))
        return converted
    else:
        raise ValueError("Input should be a list or string")


if __name__ == "__main__":
    # Example usage of the functions
    sequences = ["ATCGGA", "ATCGGA", "ATCGGA", "CTCGGA"]
    genera = ["genus1", "genus2", "genus2", "genus3"]

    # Example of how to use the build_kmer_database function
    database = build_kmer_database(
        sequences=sequences,
        genera=genera,
        kmer_size=8,
        verbose=True
    )
    print("Database created successfully")
