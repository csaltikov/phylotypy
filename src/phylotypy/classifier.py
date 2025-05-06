#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from PyQt5.pyrcc_main import verbose

from phylotypy import kmers, conditional_prob, bootstrap
from phylotypy import cond_prob_cython
from phylotypy import classify_bootstraps_cython
from phylotypy import read_fasta

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False, verbose=1)


def classify_sequences(sequences, database: kmers.KmerDB | dict, verbose=False, **kwargs, ):
    """
    Classify 16S rRNA DNA sequences against a reference database.

    This function takes a DataFrame of sequences, processes them into k-mers, and
    classifies each sequence based on the provided reference database. It returns
    a DataFrame with classification results for each sequence, including their
    corresponding identifiers and classifications. Verbose mode allows for
    progress tracking during classification.

    Args:
        sequences: pd.DataFrame
            A DataFrame where each row represents a sequence with at least an
            "id" column holding sequence identifiers and a "sequence" column.
        database: dict
            The reference database to classify against, containing the necessary
            information for sequence classification.
        verbose: bool, optional
            If set to True, displays progress updates during sequence classification.
            Default is False.
        **kwargs: Any
            Additional keyword arguments that are passed to the internal k-mer
            conversion function.

    Returns:
        pd.DataFrame:
            A DataFrame containing classification results. It includes a column
            for the sequence "id" and a "classification" column with the predicted
            classification for each sequence.

    Examples:
        >>> from phylotypy import read_fasta, classifier
        >>> seqs = read_fasta.read_taxa_fasta("my_sequences.fa")
        >>> ref_seqs = read_fasta.read_taxa_fasta("my_reference_sequences.fa")
        >>> database = classifier.make_classifier(ref_seqs)
        >>> classified = classifier.classify_sequences(seqs, database)
    """
    genera_idx_test, detected_kmers_test = conditional_prob.seq_to_kmers_database(sequences, **kwargs)

    classified = defaultdict(list)

    for i, idx in enumerate(genera_idx_test):
        if verbose:
            if i % 100 == 0:
                print(f"Classifying sequence {i} of {len(genera_idx_test)}")
        seq_kmer = detected_kmers_test[i, 1:].flatten()
        name = sequences.iloc[i]["id"]
        classified["id"].append(name)
        classified["classification"].append(classify_sequence(seq_kmer, database))

    res = pd.DataFrame(classified)
    return res


def classify_sequence(seq_kmer, database):
    bootstrapped = bootstrap.bootstrap(seq_kmer)
    classified_kmers = classify_bootstraps_cython(bootstrapped, database.conditional_prob)
    consensus = bootstrap.bootstrap_consensus(classified_kmers, database.genera_names)
    filtered = kmers.filter_taxonomy(consensus)
    return kmers.print_taxonomy(filtered)


def make_classifier(ref_db: pd.DataFrame | str | Path, **kwargs):
    """
    Creates a k-mer based classifier database from a DNA sequence reference database.

    This function processes a reference database of sequences and their IDs, validates its structure, and
    builds a k-mer based classifier database. The k-mer size can be adjusted using keyword arguments.
    The output is a database containing conditional probabilities for genera classification.

    Args:
        ref_db (pd.DataFrame | str | Path): The reference database. It can be a DataFrame with 'id'
            and 'sequence' columns or a file path to a FASTA file containing sequence data with taxonomy.
        **kwargs: Additional configuration options:
            - kmers_size (int): Size of k-mers to use in the analysis (default: 8)
            - multiprocess (bool): Whether to use multiprocessing for k-mer detection (default: True)
            - n_cpu (int): Number of CPU cores to use for multiprocessing (default: 4)
            - verbose (bool): Whether to show progress messages during processing (default: False)

    Returns:
        KmerDB: A k-mer based database object that contains the genus conditional probabilities,
            genus indices, and genus names.

    Raises:
        ValueError: If the reference database does not contain the required 'id' and 'sequence' columns.

    Examples:
        >>> from phylotypy import read_fasta, classifier
        >>> ref_seqs = read_fasta.read_taxa_fasta("my_reference_sequences.fa", multiprocess=True)
        >>> database = classifier.make_classifier(ref_seqs)

        >>> # Save the database for later use:
        >>> import pickle

        >>> with open("database.pkl", "wb") as f:
        >>>     pickle.dump(database, f)

        >>> # Load the database later:
        >>> import pickle
        >>> with open("database.pkl", "rb") as f:
        >>>     database = pickle.load(f)
    """
    if isinstance(ref_db, str | Path):
        ref_db = read_fasta.read_taxa_fasta(ref_db)

    ref_db_cols = ref_db.columns.to_list()
    if not set(ref_db_cols) == {"id", "sequence"}:
        raise ValueError("Reference database must contain 'id' and 'sequence' columns")

    kmer_size = kwargs.get('kmers_size', 8)
    multiprocess = kwargs.get('multiprocess', True)
    n_cpu = kwargs.get('n_cpu', 4)
    verbose = kwargs.get('verbose', False)

    print("Building classifier database...")

    if multiprocess:
        # detect_list = ref_db["sequence"].parallel_apply(lambda df: kmers.detect_kmer_indices(df, k=kmer_size))
        detect_list = kmers.detect_kmers_across_sequences_mp(ref_db["sequence"],
                                                          kmer_size=kmer_size,
                                                          verbose=verbose)
    else:
        detect_list = kmers.detect_kmers_across_sequences(ref_db["sequence"],
                                                          kmer_size=kmer_size,
                                                          verbose=verbose)

    genera_idx = np.array(kmers.genera_str_to_index(ref_db["id"]), dtype=np.int32)
    genera_names = kmers.index_genus_mapper(ref_db["id"].to_list())

    priors = kmers.calc_word_specific_priors(detect_list, kmer_size=kmer_size, verbose=verbose)

    genus_cond_prob = cond_prob_cython.calc_genus_conditional_prob(detect_list, genera_idx, priors.astype(np.float32))

    database = kmers.KmerDB(conditional_prob=genus_cond_prob, genera_idx=genera_idx.tolist(),
                            genera_names=genera_names)

    print("Done building classifier")
    return database


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
