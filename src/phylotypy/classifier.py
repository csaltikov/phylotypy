#!/usr/bin/env python3
import multiprocessing
import platform
import warnings

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
from phylotypy.batch_classifier import ClassifyAll


def detect_n_levels(genera_names: np.ndarray | pd.Series | list) -> int:
    """Infer the number of taxonomic levels from the majority semicolon count in genera_names."""
    genera_names = np.asarray(genera_names, dtype=str)
    counts = np.char.count(genera_names, ";")
    majority = np.bincount(counts).argmax()
    return int(majority) + 1


def check_taxonomic_levels(ref_db: pd.DataFrame, id_col: str = "id") -> None:
    """
    Verify every taxonomy string in ref_db[id_col] has the same number of
    ';'-delimited taxonomic levels. A ragged reference database builds fine
    here but fails later in consensus_bs_class, which stacks the per-sequence
    taxonomy splits into a single array and requires them to be the same length.
    """
    counts_summary = read_fasta.count_taxonomic_levels(ref_db[id_col])
    if len(counts_summary) > 1:
        raise ValueError(
            "ref_db contains taxonomy strings with inconsistent numbers of taxonomic "
            f"levels:\n{counts_summary.to_string()}\n"
            "All 'id' entries must have the same number of ';'-delimited levels before "
            "building a classifier, or classify_sequences() will fail later on.\n"
            "Options:\n"
            "  - Fix/pad the taxonomy strings in ref_db so every entry has the same depth.\n"
            "  - Call make_classifier(ref_db, filter_db=True) to filter ref_db down to a "
            "single consistent depth (see training_data.filter_train_set).\n"
            "  - Pass n_levels=<int> together with filter_db=True to control which depth is kept."
        )


def classify_sequences(sequences: pd.DataFrame | str | Path,
                       database: kmers.KmerDB | dict,
                       *,
                       verbose: bool = False,
                       min_confidence: float = 80,
                       n_levels: int | None = None,
                       kmer_size: int = 8,
                       seq_col: str = "sequence",
                       id_col: str = "id",
                       num_bs: int = 100,
                       chunk: int = 20_000):
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
        min_confidence: float, optional
            Bootstrap confidence threshold (0-100) below which a taxonomic rank is
            dropped in favor of an "_unclassified" placeholder. Default: 80.
        n_levels: int, optional
            Number of taxonomic levels in the output classification string. If
            None (default), it's auto-detected from the reference database.
        kmer_size: int, optional
            K-mer size used to convert sequences to k-mer indices; must match the
            kmer_size the reference database was built with. Default: 8.
        seq_col: str, optional
            Name of the column in `sequences` holding the DNA sequence. Default: "sequence".
        id_col: str, optional
            Name of the column in `sequences` holding the sequence identifier. Default: "id".
        num_bs: int, optional
            Number of bootstrap replicates used for the confidence estimate. Default: 100.
        chunk: int, optional
            Number of bootstrap rows processed per batch during classification;
            tune down to reduce peak memory on very large inputs. Default: 20_000.

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

    n_levels = n_levels or detect_n_levels(database.genera_names)

    classify_seqs = ClassifyAll()
    classify_seqs.classify(sequences=sequences,
                           database=database,
                           verbose=verbose,
                           kmer_size=kmer_size,
                           seq_col=seq_col,
                           id_col=id_col,
                           num_bs=num_bs,
                           chunk=chunk)

    res = classify_seqs.results(
        min_confidence=min_confidence,
        verbose=verbose
    )
    if verbose:
        print(f"Classified {len(res)} sequences to {n_levels} taxonomic levels")
    return res


def classify_sequence(seq_kmer, database, **kwargs):
    warnings.warn(
        "classify_sequence() is deprecated as of version 0.4.0 and will be removed in 1.0.0. "
        "Pass single-sequence DataFrames or FASTA files directly to classify_sequences() instead.",
        category=DeprecationWarning,
        stacklevel=2
    )
    min_confidence = kwargs.get("min_confidence", 80)
    n_levels = kwargs.get("n_levels", 6)
    bootstrapped = bootstrap.bootstrap(seq_kmer)
    classified_kmers = classify_bootstraps_cython(bootstrapped, database.conditional_prob)
    consensus = bootstrap.bootstrap_consensus(classified_kmers, database.genera_names)
    filtered = kmers.filter_taxonomy(classification=consensus, min_confidence=min_confidence)
    return kmers.print_taxonomy(filtered, n_levels=n_levels)


def make_classifier(ref_db: pd.DataFrame | str | Path, *,
                    kmer_size: int = 8,
                    multiprocess: bool = True,
                    n_cpu: int = 4,
                    verbose: bool = False,
                    filter_db: bool = False,
                    max_per_genus: int = None,
                    random_state: int = 2112,
                    mmap_threshold_gb: float | int = 8.0,
                    n_levels: int | None = None):
    """
    Creates a k-mer based classifier database from a DNA sequence reference database.

    This function processes a reference database of sequences and their IDs, validates its structure, and
    builds a k-mer based classifier database. The k-mer size can be adjusted using keyword arguments.
    The output is a database containing conditional probabilities for genera classification.

    Args:
        ref_db (pd.DataFrame | str | Path): The reference database. It can be a DataFrame with 'id'
            and 'sequence' columns or a file path to a FASTA file containing sequence data with taxonomy.
        kmer_size (int): Size of k-mers to use in the analysis (default: 8)
        multiprocess (bool): Whether to use multiprocessing for k-mer detection (default: True)
        n_cpu (int): Number of CPU cores to use for multiprocessing (default: 4)
        verbose (bool): Whether to show progress messages during processing (default: False)
        filter_db (bool): filter records used to create the database; see training_data.py
        max_per_genus (int): used for down sampling and max records per unique genera, filter_db must be True
        random_state (int): for down-sampling database, set to 42
        mmap_threshold_gb (float | int): if the conditional-probability matrix would exceed this
            size in GB, build it memory-mapped instead of in RAM (default: 8.0)
        n_levels (int, optional): taxonomic depth to filter ref_db down to when filter_db=True.
            If None (default), it's auto-detected from ref_db's taxonomy strings.

    Returns:
        KmerDB: A k-mer based database object that contains the genus conditional probabilities,
            genus indices, and genus names.

    Raises:
        ValueError: If the reference database does not contain the required 'id' and 'sequence' columns.

    Examples:
        >>> from phylotypy import read_fasta, classifier
        >>> ref_seqs = read_fasta.read_taxa_fasta("my_reference_sequences.fa")
        >>> database = classifier.make_classifier(ref_seqs, multiprocess=True)

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

    if filter_db:
        n_levels = n_levels or detect_n_levels(ref_db["id"])
        print(f"Before filter: {len(ref_db):,} sequences, {ref_db['id'].nunique():,} genera")
        ref_db = training_data.filter_train_set(ref_db, n_levels=n_levels, verbose=verbose)
        print(f"After filter: {len(ref_db):,} sequences, {ref_db['id'].nunique():,} genera")
    if max_per_genus:
        ref_db = training_data.down_sample(ref_db, col="id", 
                                           n=max_per_genus,
                                           random_state=random_state)
        print(f"After downsample: {len(ref_db):,} sequences, {ref_db['id'].nunique():,} genera")
    print(f"Matrix will be: {(4**kmer_size) * ref_db['id'].nunique() * 4 / 1e9:.2f} GB")

    ref_db_cols = ref_db.columns.to_list()
    required_cols = {"id", "sequence"}
    if not required_cols.issubset(set(ref_db_cols)):
        raise ValueError("Reference database must contain 'id' and 'sequence' columns")

    check_taxonomic_levels(ref_db)

    print("Building classifier database...")
    print(f"Mutiprocessing is set to: {multiprocess}")

    if multiprocess:
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
