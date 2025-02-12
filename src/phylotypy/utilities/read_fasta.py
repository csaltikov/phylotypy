import gzip
import pathlib
import re
from collections import defaultdict

import pandas as pd


def read_taxa_fasta(fasta_file: str | pathlib.Path) -> pd.DataFrame:
    """Read a fasta file and return a dictionary of sequences.

    Parameters:
        fasta_file (string): path to fasta file

    Returns:
        pd.DataFrame: dictionary of sequences
    """
    fasta_data = defaultdict(list)

    def get_taxa_string(fasta_line: str):
        fasta_list = (fasta_line.rstrip()
                      .replace(">", "")
                      .rstrip(";"))
        fasta_list = re.split(r"\t|\s{2,}", fasta_list)
        for item in fasta_list:
            if re.findall("Bacteria|Archaea|Eukar", item):
                return re.sub(r" suborder\w+;", "", item) # fasta id might have suborder

    def clean_id(fasta_line: str):
        clean_line = fasta_line.rstrip().replace(">", "")
        return re.split(r"\t|\s{2,}", clean_line)[0]

    gz_file = is_gzip_file(fasta_file)

    open_func = gzip.open if gz_file else open
    mode = 'rt' if gz_file else 'r'  # 'rt' for text mode in gzip

    with open_func(fasta_file, mode) as f:
        big_line = ""

        # First sequence
        first_line = f.readline()

        if not first_line.startswith(">"):
            raise ValueError("The file does not start with '>'.")

        taxa_string = get_taxa_string(first_line)
        if taxa_string:
            fasta_data["id"].append(taxa_string)
        else:
            fasta_data["id"].append(clean_id(first_line))

        # the remaining sequences
        for line in f.readlines():
            if line.startswith(">"):
                taxa_string = get_taxa_string(line)
                if taxa_string:
                    fasta_data["id"].append(taxa_string)
                else:
                    fasta_data["id"].append(clean_id(line))

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


def is_gzip_file(file_path):
    try:
        with gzip.open(file_path, 'rb') as f:
            f.read(1)
        return True
    except gzip.BadGzipFile:
        return False


if __name__ == "__main__":
    fasta_data_file = ["../data/test_fasta.fa",
                       "../data/test_2_fasta.fa",
                       "../data/test_3_fasta.fa",
                       "../data/test_4_fasta.fa"] # sys.argv[1]
    try:
        for fasta in fasta_data_file:
            fasta_data_loaded = read_taxa_fasta(fasta)
            print(fasta_data_loaded["id"].head())
    except Exception as e:
        print(e)
