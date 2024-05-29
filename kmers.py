import re
from typing import Dict, Any

import numpy as np


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


def build_kmer_database(sequences: list, genera: list, kmer_size: int = 8, verbose: bool = False):
    detected_kmers = detect_kmers_across_sequences(sequences, kmer_size=kmer_size)

    priors = calc_word_specific_priors(detected_kmers, kmer_size, verbose)

    genera_idx = genera_str_to_index(genera)

    cond_prob = calc_genus_conditional_prob(detected_kmers, genera_idx, priors, verbose)
    genera_names = index_genus_mapper(genera)

    return dict(conditional_prob=cond_prob, genera_idx=genera_idx, genera_names=genera_names)


def get_all_kmers(sequence: str, kmer_size: int = 8) -> list:
    kmer_list: list = []
    for i in range(len(sequence) - kmer_size + 1):
        kmer_list.append(sequence[i: i + kmer_size])
    return kmer_list


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
        raise ValueError(f"Input should be a list or string")


def base4_to_index(base4_str: list) -> list:
    """Converts base4 string to a numpy array of indices"""
    keep_kmers: list = []
    # Ignore kmers with N, can't convert those to base10
    for item in base4_str:
        if "N" not in item:
            keep_kmers.append(str(item))
    # Each base4 kmer has a unique number
    converted_list = [int(s, 4) for s in keep_kmers]
    return converted_list


def detect_kmers(sequence: str, kmer_size: int = 8) -> np.array:
    """Detects kmers in a DNA sequence"""
    # Converts ACGT sequence data to base4
    kmers: list = get_all_kmers(seq_to_base4(sequence), kmer_size)
    # Detected kmer indices, base 10, which are positions in a matrix
    return base4_to_index(kmers)


def detect_kmers_across_sequences(sequences: list, kmer_size: int = 8, verbose: bool = False) -> list:
    """Find all kmers for a list of nucleotide sequences"""
    if verbose:
        print("Detecting kmers across sequences")
    n_sequences = len(sequences)
    kmer_list: list = [None] * n_sequences

    for i, seq in enumerate(sequences):
        kmer_list[i] = detect_kmers(seq, kmer_size)

    return kmer_list


def calc_word_specific_priors(detect_list, kmer_size, verbose: bool = False):
    if verbose:
        print("Calculating word specific priors")
    kmer_list = [item for sublist in detect_list for item in sublist]
    n_seqs = len(detect_list)
    kmer_idx, counts = np.unique(kmer_list, return_counts=True)
    priors = np.zeros(4 ** kmer_size)
    priors[kmer_idx] = counts

    return (priors + 0.5) / (n_seqs + 1)


def calc_genus_conditional_prob(detect_list: list,
                                genera: list,
                                word_specific_priors,
                                verbose: bool = False) -> np.array:
    if verbose:
        print("Calculating genus conditional probability")
    genus_arr = np.array(genera)  # put the list of genera into an array
    genus_counts = np.unique(genus_arr, return_counts=True)[1]  # get counts of each unique genera
    n_genera = len(genus_counts)  # get number of unique genera
    n_sequences = len(genera)
    n_kmers = len(word_specific_priors)

    # Create an array with zeros, rows are kmer indicies, columns number of unique genera
    genus_count = np.zeros((n_kmers, n_genera))

    # loop through the incoming genera
    # i is a specific organism
    # get the list of kmer indices and fill in 1 or 0
    for i in range(n_sequences):
        genus_count[detect_list[i], genera[i]] = genus_count[detect_list[i], genera[i]] + 1

    # Calculate the likelihood for a genus to have a specific kmer
    # (m(wi) + Pi) / (M + 1)
    wi_pi = (genus_count + np.array(word_specific_priors).reshape(-1, 1)).T
    m_1 = (genus_counts + 1).reshape(-1, 1)
    genus_cond_prob = (wi_pi / m_1).T

    return genus_cond_prob


def genera_str_to_index(genera: list) -> list:
    # Create a dictionary mapping unique values to integers
    unique_genera = np.unique(genera)
    factor_map = {val: idx for idx, val in enumerate(unique_genera)}
    # Convert genera to factors using the mapping
    genera_factors = [factor_map[val] for val in genera]
    return genera_factors


def index_genus_mapper(genera_list: list) -> dict:
    # Create a dictionary mapping unique values to integers
    unique_genera = np.unique(genera_list)
    # return factor_map
    return unique_genera


def bootstrap_kmers(kmers: np.array, kmer_size: int = 8):
    n_kmers = kmers.shape[0] // kmer_size
    return np.random.choice(kmers, n_kmers, replace=True)


def classify_bs(kmer_index: list, db):
    """Screens a tests sequence against all classes in the model"""
    model_mask = db["conditional_prob"][kmer_index, :]
    class_log = np.log(model_mask)
    class_sum = np.sum(class_log, axis=0, keepdims=True)
    max_idx = np.argmax(class_sum)

    return max_idx


def consensus_bs_class(bs_class: np.array, db) -> dict[str, list | Any]:
    # Convert the indices in the bootstrap array to taxonomy
    taxonomy: np.array = db["genera"][bs_class]
    # sometimes taxonomy is empty or has "none" value
    mask = taxonomy != None
    taxonomy_filtered = taxonomy[mask]

    taxonomy_split = np.array([line.split(";") for line in taxonomy_filtered])

    def cumulative_join(col):
        join_taxa = [";".join(col[:i + 1]) for i in range(len(col))]
        return join_taxa

    taxa_cum_join_arr = np.apply_along_axis(cumulative_join, 1, taxonomy_split)

    taxa_string, confidence = np.apply_along_axis(get_consensus, axis=0, arr=taxa_cum_join_arr)

    return dict(taxonomy=np.array(taxa_string[-1].split(";")),
                confidence=confidence)


def get_consensus(taxa_cum_join_arr: np.ndarray[str]):
    """Helper for consensus_bs_class determines best taxon and confidence level"""
    # get best ID and score for each column of the taxa array
    taxonomy_table = np.unique(taxa_cum_join_arr, return_counts=True)
    max_id = np.argmax(taxonomy_table[1])

    id_fraction_arr = np.full(2, fill_value=["unclassified", 0], dtype=list)

    fraction = taxonomy_table[1][max_id].item() / taxonomy_table[1].sum()

    id_fraction_arr[0] = taxonomy_table[0][max_id].item()
    id_fraction_arr[1] = int(100 * fraction)

    return id_fraction_arr


def filter_taxonomy(classification: dict, min_confidence: float = 80) -> Dict:
    """Helper for create_con determines finds the best id"""

    n_levels = classification["confidence"].shape[0]

    high_confidence = np.where(classification["confidence"] >= min_confidence)[0]

    if high_confidence.size == 0:
        taxonomy = np.array(["unclassified"] * n_levels)
        confidence = np.zeros(n_levels, dtype=int)
    else:
        taxonomy = classification["taxonomy"][high_confidence]
        confidence = classification["confidence"][high_confidence]

    return dict(
        taxonomy=taxonomy,
        confidence=confidence
    )


def print_taxonomy(consensus: dict, n_levels=6) -> str:
    original_levels = consensus["taxonomy"].shape[0]
    given_levels = original_levels
    extra_levels = n_levels - given_levels

    taxa_idx = np.arange(consensus["taxonomy"].shape[0])

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

