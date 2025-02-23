from dataclasses import dataclass

import pandas as pd

import multiprocessing as mp
from pathlib import Path
import re
from typing import Dict, Any

import numpy as np

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

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


def build_kmer_database(sequences: list[str], genera: list[str],
                        kmer_size: int = 8, verbose: bool = False,
                        num_processes: int = mp.cpu_count(),
                        **kwargs) -> KmerDB:
    """Creates a conditional probablity matrix from DNA sequences and associated genera or sequence ids.
    The m_proc kwarg sets the kmer detection to multiprocessing.

    @param sequences: DNA seqeunce as strings in a list ["ATCGGA", "ATCGGA"]
    @param genera: list of genera for building a model or list of ids for predicting genera
    @param kmer_size: the 'chunk' size of DNA, 8 nucleotide kmers are the default
    @param m_proc: whether to use multiprocessing or not
    @param num_processes: the number of processes to use
    @param verbose: set to True if you want to see the outputs
    @return: kmers database dictionary
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


def classify_sequences(uknown_df: pd.DataFrame,
                       database):
    conditional_prob = database.conditional_prob
    genera_names = database.genera_names

    uknown_df["classification"] = (uknown_df["sequence"]
                               .parallel_apply(detect_kmers)
                               .parallel_apply(bootstrap)
                               .parallel_apply( lambda x: classify_bootstraps(x, conditional_prob))
                               .parallel_apply(bootstrap)
                               .parallel_apply(lambda x: consensus_bs_class(x, genera_names))
                               .parallel_apply(lambda x: print_taxonomy(filter_taxonomy(x)))
                               )
    return uknown_df


def classify(bs_kmer, database: KmerDB):
    classified_list = classify_bootstraps(bs_kmer, database.conditional_prob)
    return classified_list


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
        raise ValueError(f"Input should be a list or string")


def base4_to_index(base4_str: list) -> list:
    """Converts base4 string to a numpy array of indices but ignores kmers with an N"""
    converted_list: list = [int(item, 4) for item in base4_str if "N" not in item]
    return converted_list


def detect_kmers(sequence: str, kmer_size: int = 8) -> list:
    """Detects kmers in a DNA sequence"""
    # Converts ACGT sequence data to base4
    kmers_: list = get_all_kmers(seq_to_base4(sequence), kmer_size)
    # Detected kmer indices, base 10, which are positions in a matrix
    kmers_ = base4_to_index(kmers_)
    return np.unique(kmers_).tolist()


def detect_kmers_across_sequences(sequences: list, kmer_size: int = 8, verbose: bool = False) -> list:
    """Find all kmers for a list of nucleotide sequences"""
    if verbose:
        print("Detecting kmers across sequences")
    n_sequences = len(sequences)
    kmer_list: list = [None] * n_sequences
    for i, seq in enumerate(sequences):
        kmer_list[i] = detect_kmers(seq, kmer_size)
    return kmer_list


def detect_kmers_across_sequences_mp(sequences: list,
                                     kmer_size: int = 8,
                                     num_processes: int = 4,
                                     verbose: bool = False) -> list:
    if verbose:
        print("Detecting kmers across sequences mp")
    with mp.Pool(num_processes) as pool:
        args_list = [(seq, kmer_size) for seq in sequences]
        results = pool.starmap(detect_kmers, args_list)
    return results


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

    for idx_list in detected_kmers_list:
        priors[idx_list] +=1

    # expected-likelihood estimate using Jeffreys-Perks law of succession
    # 0 < Pi < 1
    return (priors + 0.5) / (n_seqs + 1)


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

    genus_cond_prob = np.log(np.divide(wi_pi, m_1)).astype(np.float16)

    return genus_cond_prob


def process_chunk(args):
    start, end, detect_list, genera_idx, genus_count_dat, shape = args
    genus_count = np.memmap(genus_count_dat, mode='r+', dtype=np.float32, shape=shape)
    for i in range(start, end):
        genus_count[detect_list[i], genera_idx[i]] += 1
    genus_count.flush()


def calc_genus_conditional_prob_mp(detect_list: list[list[int]],
                                   genera_idx: list,
                                   word_specific_priors: np.ndarray,
                                   verbose: bool = False,
                                   **kwargs) -> np.array:
    genus_arr = np.array(genera_idx)  # indices not the taxa names
    genus_counts = np.unique(genus_arr, return_counts=True)[1]  # get counts of each unique genera
    n_genera = len(genus_counts)
    n_sequences = len(genera_idx)
    n_kmers = len(word_specific_priors)

    if verbose:
        print(f"Calculating genus conditional probability for {n_genera} unique genera")

    # Create an array with zeros, rows are kmer indices, columns number of unique genera
    genus_count_dat = Path("genus_count.dat")

    genus_count = np.memmap(genus_count_dat,
                            dtype=np.float32,
                            mode='w+',
                            shape=(n_kmers, n_genera))

    if verbose:
        print("Genus conditional probability: start looping")
        print(f"db size {genus_count.shape}")

    # Prepare chunks for parallel processing or memory savings
    chunk_size: int = kwargs.get('chunk_size', 20000)
    multi: bool = kwargs.get('multi', False)
    n_cpu: int = kwargs.get('n_cpu', 4)

    if multi:
        print(f"multiple processing: {multi} with {n_cpu} cpus")
        chunks = [(start, min(start + chunk_size, n_sequences), detect_list, genera_idx, genus_count_dat, genus_count.shape)
                  for start in range(0, n_sequences, chunk_size)]

        with mp.Pool(processes=n_cpu) as pool:
            pool.map(process_chunk, chunks)
    else:
        # Start filling out genus_count array
        # Process data in chunks
        for start in range(0, n_sequences, chunk_size):
            end = min(start + chunk_size, n_sequences)

            # Process a chunk of the data
            for i in range(start, end):
                genus_count[detect_list[i], genera_idx[i]] += 1

            # Flush changes to disk periodically
            genus_count.flush()

    if verbose:
        print("Genus conditional probability: done looping")

    genus_count.flush()

    # Calculate the likelihood for a genus to have a specific kmer
    # (m(wi) + Pi) / (M + 1)
    wi_pi = (genus_count + word_specific_priors.reshape(-1, 1))
    m_1 = (genus_counts + 1)

    genus_cond_prob = np.log(np.divide(wi_pi, m_1)).astype(np.float16)

    # Delete the memmap object
    del genus_count
    # Remove the temporary file
    if genus_count_dat.exists():
        genus_count_dat.unlink()

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


def bootstrap_kmers(kmers: np.array, kmer_size: int = 8):
    '''Performs a single bootstrap sampling on a kmers array'''
    n_kmers = kmers.shape[0] // kmer_size
    return np.random.choice(kmers, n_kmers, replace=True)


def bootstrap(kmer_index: list, n_bootstraps: int = 100, fraction: int = 8, **kwargs) -> np.ndarray:
    ''''Performs multiple bootstrap samplings on a list of kmers'''
    # if kwargs.get('seed'):
    #     print(kwargs.get('seed'))
    #     np.random.seed(kwargs.get('seed'))
    n_samples = len(kmer_index) // fraction
    kmer_sample_arr = np.zeros((n_bootstraps, n_samples), dtype=int)
    for i in range(n_bootstraps):
        kmer_sample_arr[i, :] = np.random.choice(kmer_index, n_samples, replace=True)
    return kmer_sample_arr


def classify_bs(kmer_index: list, db):
    """Classify a single bootstrap sample of kmers from a sample"""
    model_mask = db.conditional_prob[kmer_index, :]
    class_sum = np.sum(model_mask, axis=0)
    max_idx = np.argmax(class_sum)
    return max_idx


def classify_bootstraps(bs_indices: np.array, conditional_prob):
    """"Classify an array of kmer bootstraps from a sample"""
    return np.argmax(np.sum(conditional_prob[bs_indices], axis=1), axis=1)


def consensus_bs_class(bs_class: np.array, genera_names) -> dict[str, list | Any]:
    """Convert the indices in the bootstrap array to taxonomy"""
    taxonomy: np.array = genera_names[bs_class]
    # sometimes taxonomy is empty or has "none" value
    mask = taxonomy != None
    taxonomy_filtered = taxonomy[mask]

    taxonomy_split = np.array([line.split(";") for line in taxonomy_filtered])

    def cumulative_join(col):
        join_taxa = [";".join(col[:i + 1]) for i in range(len(col))]
        return np.array(join_taxa, dtype='<U300')

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

    fraction = counts[max_id].item() / counts.sum()

    id_fraction_arr[0] = taxonomy[max_id].item()
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
        raise ValueError(f"Input should be a list or string")


if __name__ == "__main__":
    mp.freeze_support()

    from phylotypy import classifier
    from phylotypy.utilities import read_fasta

    database = classifier.load_classifier("../../local_items/database.pkl")

    conditional_prob = database.conditional_prob
    genera_names = database.genera_names


    seqs = read_fasta.read_taxa_fasta("../../data/dna_moving_pictures.fasta")
    res = classify_sequences(seqs, database)
