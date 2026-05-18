#!/usr/bin/env python3
import multiprocessing
import platform

if platform.system() == "Darwin":
    multiprocessing.set_start_method("spawn", force=True)
    
from collections import defaultdict
from pathlib import Path
import pickle

import pandas as pd
import numpy as np

from phylotypy import kmers, conditional_prob, bootstrap
from phylotypy import cond_prob_cython
from phylotypy import classify_bootstraps_cython
from phylotypy import read_fasta
from phylotypy import training_data

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False, verbose=1)


def detect_n_levels(genera_names: np.ndarray) -> int:
    """Infer the number of taxonomic levels from the majority semicolon count in genera_names."""
    counts = np.char.count(genera_names, ";")
    majority = np.bincount(counts).argmax()
    return int(majority) + 1


def classify_sequences(sequences: pd.DataFrame | str | Path,
                       database: kmers.KmerDB | dict,
                       verbose=False, **kwargs):
    """
    Classify 16S rRNA DNA sequences against a reference database.

    This function takes a DataFrame of sequences of fasta file, processes them into k-mers, and
    classifies each sequence based on the provided reference database. It returns
    a DataFrame with classification results for each sequence, including their
    corresponding identifiers and classifications. Verbose mode allows for
    progress tracking during classification.

    Args:
        sequences: pd.DataFrame or str Path
            Input is either a str/Path to the fasta file to classify. Or a pandas
            DataFrame. If str/Path, the fasta will be converted to a datafame.
            Each row is a sequence with at least an "id" column holding sequence
            identifiers and a "sequence" column.
        database: dict or kmers.KmerDB
            The reference database to classify against, containing the necessary
            information for sequence classification.
        verbose: bool, optional
            If set to True, displays progress updates during sequence classification.
            Default is False.
        **kwargs: min_confidence, n_levels
            Additional keyword arguments that are passed to the internal k-mer
            conversion function. Use min_confidence (default = 80) for filtering bootstrap
            consensus. Use n_levels to set taxonomic levels (default=6).

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
    if isinstance(sequences, str | Path):
        sequences = read_fasta.read_taxa_fasta(sequences)

    if "n_levels" not in kwargs:
        kwargs["n_levels"] = detect_n_levels(database.genera_names)

    genera_idx_test, detected_kmers_test = conditional_prob.seq_to_kmers_database(
        sequences, **kwargs
        )

    classified = defaultdict(list)

    for i, idx in enumerate(genera_idx_test):
        if verbose:
            if i % 100 == 0:
                print(f"Classifying sequence {i} of {len(genera_idx_test)}")
        seq_kmer = detected_kmers_test[i, 1:].flatten()
        name = sequences.iloc[i]["id"]
        classified["id"].append(name)
        classified["classification"].append(classify_sequence(seq_kmer=seq_kmer,
                                                              database=database,
                                                              **kwargs))
    res = pd.DataFrame(classified)
    return res


def classify_sequence(seq_kmer, database, **kwargs):
    min_confidence = kwargs.get("min_confidence", 80)
    n_levels = kwargs.get("n_levels", 6)
    bootstrapped = bootstrap.bootstrap(seq_kmer)
    classified_kmers = classify_bootstraps_cython(bootstrapped, database.conditional_prob)
    consensus = bootstrap.bootstrap_consensus(classified_kmers, database.genera_names)
    filtered = kmers.filter_taxonomy(classification=consensus, min_confidence=min_confidence)
    return kmers.print_taxonomy(filtered, n_levels=n_levels)


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
            - filter_db (bool): filter records used to create the database; see training_data.py
            - max_per_genus (int): used for down sampling and max records per unique genera
            - random_state (int): for down-sampling database, set to 42 
            

    Returns:
        KmerDB: A k-mer based database object that contains the genus conditional probabilities,
            genus indices, and genus names.

    Raises:
        ValueError: If the reference database does not contain the required 'id' and 'sequence' columns.

    Examples:
        >>> from phylotypy import read_fasta, classifier
        >>> ref_seqs = read_fasta.read_taxa_fasta("my_reference_sequences.fa", multiprocess=True)
        >>> database = classifier.make_classifier(ref_seqs)
        
        >>> # downsample and filter option
        >>> database_filt = classifier.make_classifier(ref_seqs,
        >>>                                            filter_db=True,
        >>>                                            max_per_genus=50)

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
        
    filter_db: bool = kwargs.get("filter_db", False)
    max_per_genus: int = kwargs.get("max_per_genus", 200)
    random_state: int = kwargs.get("random_state", 42)
    mmap_threshold_gb: float|int = kwargs.get("mmap_threshold_gb", 8.0)
    kmer_size: int = kwargs.get('kmers_size', 8)
    multiprocess: bool = kwargs.get('multiprocess', True)
    n_cpu: int = kwargs.get('n_cpu', 4)
    verbose: bool() = kwargs.get('verbose', False)
    
    if filter_db:
        print(f"Before filter: {len(ref_db):,} sequences, {ref_db['id'].nunique():,} genera")
        ref_db = training_data.filter_train_set(ref_db)
        print(f"After filter: {len(ref_db):,} sequences, {ref_db['id'].nunique():,} genera")
        ref_db = training_data.down_sample(ref_db, col="id", 
                                           n=max_per_genus,
                                           random_state=random_state)
        print(f"After downsample: {len(ref_db):,} sequences, {ref_db['id'].nunique():,} genera")
        print(f"Matrix will be: {65536 * ref_db['id'].nunique() * 4 / 1e9:.2f} GB")

    ref_db_cols = ref_db.columns.to_list()
    required_cols = {"id", "sequence"}
    if not required_cols.issubset(set(ref_db_cols)):
        raise ValueError("Reference database must contain 'id' and 'sequence' columns")

    print("Building classifier database...")
    print(f"Mutiprocessing is set to: {multiprocess}")

    if multiprocess:
        # detect_list = ref_db["sequence"].parallel_apply(lambda df: kmers.detect_kmer_indices(df, k=kmer_size))
        detect_list = kmers.detect_kmers_across_sequences_mp(ref_db["sequence"],
                                                          kmer_size=kmer_size,
                                                          verbose=verbose,
                                                          num_processes=n_cpu)
    else:
        detect_list = kmers.detect_kmers_across_sequences(ref_db["sequence"],
                                                          kmer_size=kmer_size,
                                                          verbose=verbose)

    genera_idx = np.array(kmers.genera_str_to_index(ref_db["id"]), dtype=np.int32)
    genera_names = kmers.index_genus_mapper(ref_db["id"].to_list())
    
    del ref_db
    
    priors = kmers.calc_word_specific_priors(detect_list, kmer_size=kmer_size, verbose=verbose)
    
    n_kmers = 4 ** kmer_size
    n_genera = len(set(genera_idx))
    matrix_gb = (4 ** kmer_size * n_genera * 4) / 1e9
    if verbose:
        print(f"genus_count matrix will be: {matrix_gb:.2f} GB")
        print(f"n_sequences: {len(detect_list)}")
        print(f"n_genera: {n_genera}")
        print(f"n_kmers: {n_kmers}")
        print(f"total kmer indices: {detect_list.indices.shape[0]:,}")
    
    if matrix_gb > mmap_threshold_gb:
        print("Large matrix detected, using memmap...")
        genus_cond_prob = cond_prob_cython.calc_genus_conditional_prob_mmap(
            detect_list.indices, detect_list.offsets,
            genera_idx, priors.astype(np.float32)
        )
    else:
        genus_cond_prob = cond_prob_cython.calc_genus_conditional_prob(
            detect_list.indices, detect_list.offsets,
            genera_idx, priors.astype(np.float32)
        )

    del detect_list
    del priors
    
    import gc
    gc.collect()  # force Python to actually release the memory

    database = kmers.KmerDB(conditional_prob=genus_cond_prob,
                            genera_idx=genera_idx,
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
    """
    Load a classifier object from a pickle file.

    Parameters
    ----------
    db_path : Path or str
        Path to the pickle file containing the classifier.

    Returns
    -------
    object
        The unpickled classifier object.

    Raises
    ------
    FileNotFoundError
        If the provided file path does not exist.
    pickle.UnpicklingError
        If there is a problem loading the pickle file.
    Exception
        For any other issues that might arise during loading.
    """
    db_path = Path(db_path)
    if check_path(db_path):
        try:
            with open(db_path, "rb") as f:
                return pickle.load(f)
        except pickle.UnpicklingError:
            raise pickle.UnpicklingError(f"Unable to unpickle file: {db_path}")
        except Exception as e:
            raise Exception(f"An error occurred: {e}")
    else:
        raise FileNotFoundError(f"Classifier file not found: {db_path}")


if __name__ == "__main__":
    print(__name__)
