#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path
import pickle
import time

import pandas as pd
from phylotypy import kmers, conditional_prob, bootstrap
from phylotypy.classify_bootstraps import classify_bootstraps_cython
from phylotypy.utilities import read_fasta

from pandarallel import pandarallel


def classify_sequences(sequences_df, database, verbose=False):
    genera_idx_test, detected_kmers_test = conditional_prob.seq_to_kmers_database(sequences_df)

    classified = defaultdict(list)

    for i, idx in enumerate(genera_idx_test):
        if verbose:
            if i % 100 == 0:
                print(f"Classifying sequence {i} of {len(genera_idx_test)}")
        seq_kmer = detected_kmers_test[detected_kmers_test[:,0]==idx, 1:].flatten()
        name = sequences_df.iloc[i]["id"]
        classified["id"].append(name)
        classified["classification"].append(classify_sequence(seq_kmer, database))

    res = pd.DataFrame(classified)
    # classification = results.summarize_predictions(res)
    return res


def classify_sequence(seq_kmer, database):
    bootstrapped = bootstrap.bootstrap(seq_kmer)
    # classified_kmers = kmers.classify_bootstraps(bootstrapped, database.conditional_prob)
    classified_kmers = classify_bootstraps_cython(bootstrapped, database.conditional_prob)
    consensus = bootstrap.bootstrap_consensus(classified_kmers, database.genera_names)
    filtered = kmers.filter_taxonomy(consensus)
    return kmers.print_taxonomy(filtered)


def classify_sequences_timed(seq_kmer, database):
    start_time = time.perf_counter()
    bootstrapped = bootstrap.bootstrap(seq_kmer)
    bootstrap_time = time.perf_counter() - start_time
    print(f"Bootstrap took {bootstrap_time:.4f} seconds.")

    start_time = time.perf_counter()
    # classified_kmers = kmers.classify_bootstraps(bootstrapped, database.conditional_prob)
    classified_kmers = bootstrap.classify_bootstraps_numba(bootstrapped, database.conditional_prob)
    classification_time = time.perf_counter() - start_time
    print(f"Classification took {classification_time:.4f} seconds.")

    start_time = time.perf_counter()
    consensus = bootstrap.bootstrap_consensus(classified_kmers, database.genera_names)
    consensus_time = time.perf_counter() - start_time
    print(f"Consensus took {consensus_time:.4f} seconds.")

    start_time = time.perf_counter()
    filtered = kmers.filter_taxonomy(consensus)
    filtering_time = time.perf_counter() - start_time
    print(f"Filtering took {filtering_time:.4f} seconds.")

    start_time = time.perf_counter()
    result = kmers.print_taxonomy(filtered)
    printing_time = time.perf_counter() - start_time
    print(f"Printing took {printing_time:.4f} seconds.")

    return result

def make_classifier(ref_fasta: Path | str, out_dir: Path | str, **kwargs):
    """
    Constructs the classifier database: conditional_probabilities array and unique reference genera

    Args:
        ref_fasta: path/to/fasta/file either as Path() or string
        out_dir: path/to/output directory either as Path() or string

    Returns:
        kmers.KmerDb database, saved as a pkl file in the out_dir
    """
    kmers_size = kwargs.get('kmers_size', 8)
    if check_path(ref_fasta) and check_path(out_dir):
        ref_db = read_fasta.read_taxa_fasta(ref_fasta)

        ids, kmers_db = conditional_prob.seq_to_kmers_database(ref_db, kmer_size=kmers_size)
        priors = conditional_prob.calc_priors(kmers_db, kmers_size)

        cond_prob_arr = conditional_prob.GenusCondProb(kmers_db, priors, kmers_size).calculate_alt()

        all_genera_names = ref_db["id"].to_list()

        genera_idx = kmers.genera_str_to_index(all_genera_names)
        genera_names = kmers.index_genus_mapper(all_genera_names)

        database = kmers.KmerDB(conditional_prob=cond_prob_arr,
                                genera_names=genera_names,
                                genera_idx=genera_idx)

        with open(out_dir / "database.pkl", "wb") as f:
            pickle.dump(database, f)
        return database
    else:
        print("Please check path to fasta file and/or output directory")


def check_path(object_path):
    if isinstance(object_path, Path):
        if not object_path.exists():
            return False
    if isinstance(object_path, str):
        if not Path(object_path).exists():
            return False
    return True


def load_classifier(db_path: Path | str):
    if check_path(db_path):
        with open(db_path, "rb") as f:
            return pickle.load(f)
    else:
        print("Error: file not found, check db path")


def initialize_pandarallel(**kwargs):
    '''Initialize pandarallel with a configurable number of
    workers If nb_workers is not provided, pandarallel
    will use the default (typically CPU core count)'''
    progress_bar = kwargs.get('progress_bar', True)
    verbose = kwargs.get('verbose', 1)
    nb_workers = kwargs.get('nb_workers', 4)
    pandarallel.initialize(progress_bar=progress_bar, verbose=verbose, nb_workers=nb_workers)


def process_sequence(sequence, conditional_prob, genera_names, min_confidence, n_levels):
    kmers_detected = kmers.detect_kmers(sequence)
    bootstrapped = kmers.bootstrap(kmers_detected)
    classified = kmers.classify_bootstraps(bootstrapped, conditional_prob)
    consensus = kmers.consensus_bs_class(classified, genera_names)
    filtered = kmers.filter_taxonomy(consensus, min_confidence)
    return kmers.print_taxonomy(filtered, n_levels)


def classify_sequences_(sequences_df: pd.DataFrame, database, **kwargs):
    n_levels = kwargs.get('n_levels', 6)
    min_confidence = kwargs.get('min_confidence', 80)
    nb_workers = kwargs.get('nb_workers', 4)
    initialize_pandarallel(nb_workers=nb_workers, **kwargs)
    conditional_prob = database.conditional_prob
    genera_names = database.genera_names

    sequences_df["classification"] = sequences_df["sequence"].parallel_apply(
        process_sequence, args=(conditional_prob, genera_names, min_confidence, n_levels)
    )
    return sequences_df



if __name__ == "__main__":
    print(__name__)
