#!/usr/bin/env python3

from pathlib import Path
import pickle

import pandas as pd
from phylotypy import kmers
from phylotypy.utilities import read_fasta

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


def classify_sequences(unknown_df: pd.DataFrame,
                       database):
    conditional_prob = database.conditional_prob
    genera_names = database.genera_names

    unknown_df["classification"] = (unknown_df["sequence"]
                                    .parallel_apply(kmers.detect_kmers)
                                    .parallel_apply(kmers.bootstrap)
                                    .parallel_apply(lambda x: kmers.classify_bootstraps(x, conditional_prob))
                                    .parallel_apply(kmers.bootstrap)
                                    .parallel_apply(lambda x: kmers.consensus_bs_class(x, genera_names))
                                    .parallel_apply(lambda x: kmers.print_taxonomy(kmers.filter_taxonomy(x)))
                                    )
    return unknown_df


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
        database = kmers.build_kmer_database(ref_db["sequence"], ref_db["id"],
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
