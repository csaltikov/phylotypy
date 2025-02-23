#!/usr/bin/env python3

from pathlib import Path
import pickle

from phylotypy import kmers
from phylotypy.utilities import read_fasta


def make_classifier(ref_fasta: Path | str, out_dir: Path | str):
    """
    Constructs the classifier database: conditional_probabilities array and unique reference genera

    Args:
        ref_fasta: path/to/fasta/file either as Path() or string
        out_dir: path/to/output directory either as Path() or string

    Returns:
        kmers.KmerDb database, saved as a pkl file in the outdir
    """
    if check_path(ref_fasta) and check_path(out_dir):
        refdb = read_fasta.read_taxa_fasta(ref_fasta)
        database = kmers.build_kmer_database(refdb["sequence"], refdb["id"],
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
