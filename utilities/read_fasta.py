import sys
from collections import defaultdict

import pandas as pd


def read_fasta_file(fasta_file):
    """Read a fasta file and return a dictionary of sequences."""
    fasta_data = defaultdict(list)

    with open(fasta_file, "r") as f:
        big_line = ""
        first_line = f.readline()
        if not first_line.startswith(">"):
            raise ValueError("Fasta file does not start with '>'.")
        first_line = (first_line
                      .rstrip()
                      .replace(">", "")
                      .split("\t")
                      )
        fasta_data["id"].append(first_line[0])
        if len(first_line) > 1:
            fasta_data["comment"].append(first_line[1])
        for line in f.readlines():
            if line.startswith(">"):
                line = (line.rstrip()
                        .replace(">", "")
                        .replace(" ", "\t", 1)
                        .split("\t")
                        )
                fasta_data["id"].append(line[0])
                if len(first_line) > 1:
                    fasta_data["comment"].append(line[1])
                fasta_data["sequence"].append(big_line)
                big_line = ""
            else:
                big_line += line.rstrip()
        fasta_data["sequence"].append(big_line)

    return pd.DataFrame(fasta_data)


def read_taxonomy(taxonomy_file, sep="\t"):
    taxa_df = pd.read_csv(taxonomy_file, sep=sep, names=["id", "taxonomy"], header=None)
    taxa_df["taxonomy"] = taxa_df["taxonomy"].str.rstrip(";")
    return taxa_df


if __name__ == "__main__":
    fasta_data_file = "../data/test_fasta.fa" # sys.argv[1]
    try:
        fasta_data_loaded = read_fasta_file(fasta_data_file)
        print(fasta_data_loaded)
    except Exception as e:
        print(e)
