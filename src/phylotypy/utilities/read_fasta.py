import gzip
from pathlib import Path
import re
from collections import defaultdict

import pandas as pd


def read_taxa_fasta(fasta_file: str | Path) -> pd.DataFrame:
    """Read a fasta file and return a pandas dataframe:

    Parameters:
        fasta_file (string or pathlib.Path()): path to fasta file, can be a .gz file as well

    Returns:
        pd.DataFrame: two column dataframe: id and sequence

    Examples:
        >>> read_taxa_fasta("my_fasta.fa")
            id                 sequence
        0  seq1  TACGGAGGATCCGAGCGTTA...
        1  seq2  TACGTAGGGTGCGAGCGTTA...
        2  seq3  TACGTAGGTCCCGAGCGTTG...
        3  seq4  TACGGAGGATCCGAGCGTTA...
        4  seq5  TACGGAGGATCCGAGCGTTA...
    """
    fasta_data = defaultdict(list)

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
        if "U" in big_line:
            big_line = big_line.replace("U", "T")
        fasta_data["sequence"].append(big_line)

    return pd.DataFrame(fasta_data)


def is_gzip_file(file_path):
    try:
        with gzip.open(file_path, 'rb') as f:
            f.readlines(1)
        return True
    except:
        return False


def get_taxa_string(fasta_line: str):
    fasta_list = (fasta_line.rstrip()
                  .replace(">", "")
                  .rstrip(";"))
    fasta_list = re.split(r"\t|\s{2,}", fasta_list)
    # clean up the fasta description if it has a taxon string
    for item in fasta_list:
        if re.findall("Bacteria|Archaea|Eukar", item):
            return re.sub(r" suborder\w+;", "", item) # fasta id might have suborder


def clean_id(fasta_line: str):
    clean_line = fasta_line.rstrip().replace(">", "")
    return re.split(r"\t|\s{2,}", clean_line)[0]


if __name__ == "__main__":
    fasta_data_file = ["data/test_2_fasta.fa",
                       "data/test_3_fasta.fa",
                       "data/test_4_fasta.fa"] # sys.argv[1]
    for f in fasta_data_file:
        print(Path(f).is_file())

    try:
        for fasta in fasta_data_file:
            fasta_data_loaded = read_taxa_fasta(fasta)
            print(fasta_data_loaded["id"].head())
    except Exception as e:
        print(e)
