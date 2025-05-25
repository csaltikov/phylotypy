import io
import gzip
from pathlib import Path
import pickle
import re
import subprocess

import pandas as pd


def dataframe_to_fasta(df, fasta_file):
    with open(fasta_file, "w") as f:
        for index, row in df.iterrows():
            f.write(f">{row['id']}\n{row['sequence']}\n")


def taxa_to_dataframe(taxa_file):
    file_path = Path(taxa_file)
    if not file_path.is_file():
        print("Taxa file not found")
        return None
    taxa_table_df = pd.read_csv(file_path, sep="\t", names=["id", "taxonomy"])
    taxa_levels = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "_"]
    taxa_table_df[taxa_levels] = taxa_table_df['taxonomy'].str.split(';', expand=True)
    taxa_table_df['taxonomy'] = taxa_table_df['taxonomy'].str.rstrip(';')
    taxa_table_df.drop(["_"], axis="columns", inplace=True)
    return taxa_table_df


def fix_qiime_taxa(taxa_string):
    taxonomy = re.sub(r"\s*\w+__", "", taxa_string)
    return taxonomy


def convert_dict_keys_to_int(d):
    """
    Convert the keys of a dictionary from strings to integers.

    Parameters:
    d (dict): The dictionary with string keys to be converted.

    Returns:
    dict: A new dictionary with integer keys.
    """
    return {int(key): value for key, value in d.items()}


def pickle_and_compress(obj, output_file: str | Path):
    """
    Pickles a Python object and compresses it into a .pkl.gz file using gzip or pigz.

    Parameters:
        obj (object): The Python object to pickle and compress.
        output_file (str): The path to the output compressed .pkl.gz file.
    """
    # Start the subprocess to compress the output with pigz (parallel gzip)
    with subprocess.Popen(['pigz', '-c'], stdin=subprocess.PIPE, stdout=open(output_file, 'wb')) as proc:
        # Create an in-memory buffer to pickle the object
        with io.BytesIO() as buffer:
            pickle.dump(obj, buffer)  # Pickle the object into the buffer
            buffer.seek(0)  # Go to the start of the buffer before writing
            proc.stdin.write(buffer.read())  # Write the pickled object to the subprocess for compression


def unpickle_and_decompress(input_file: str | Path):
    """
    Unpickles and decompresses a .pkl.gz file to retrieve the original Python object.

    Parameters:
        input_file (str): The path to the compressed .pkl.gz file.

    Returns:
        object: The Python object that was pickled and compressed.
    """
    # Open the gzipped file and unpickle the object
    with gzip.open(input_file, 'rb') as f:
        obj = pickle.load(f)  # Unpickle and load the object from the file
    return obj


if __name__ == "__main__":
    print(f"Support tools for phylotypy package")
