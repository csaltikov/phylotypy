#!/usr/bin/env python3

from pathlib import Path
import pickle

import pandas as pd
from phylotypy import kmers
from phylotypy.utilities import read_fasta

from pandarallel import pandarallel


def initialize_pandarallel(nb_workers=None, **kwargs):
    '''Initialize pandarallel with a configurable number of
    workers If nb_workers is not provided, pandarallel
    will use the default (typically CPU core count)'''
    progress_bar = kwargs.get('progress_bar', True)
    verbose = kwargs.get('verbose', 1)
    pandarallel.initialize(progress_bar=progress_bar, verbose=verbose, nb_workers=nb_workers)


def process_sequence(sequence, conditional_prob, genera_names, min_confidence, n_levels):
    kmers_detected = kmers.detect_kmers(sequence)
    bootstrapped = kmers.bootstrap(kmers_detected)
    classified = kmers.classify_bootstraps(bootstrapped, conditional_prob)
    consensus = kmers.consensus_bs_class(classified, genera_names)
    filtered = kmers.filter_taxonomy(consensus, min_confidence)
    return kmers.print_taxonomy(filtered, n_levels)


def classify_sequences(sequences_df: pd.DataFrame, database, **kwargs):
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


def make_classifier(ref_fasta: Path | str, out_dir: Path | str):
    """
    Constructs the classifier database: conditional_probabilities array and unique reference genera

    Args:
        ref_fasta: path/to/fasta/file either as Path() or string
        out_dir: path/to/output directory either as Path() or string

    Returns:
        kmers.KmerDb database, saved as a pkl file in the out_dir
    """
    if check_path(ref_fasta) and check_path(out_dir):
        ref_db = read_fasta.read_taxa_fasta(ref_fasta)
        database = kmers.build_kmer_database(sequences=["sequence"],
                                             genera=ref_db["id"],
                                             verbose=True,
                                             m_proc=True)
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


if __name__ == "__main__":
    print(__name__)
