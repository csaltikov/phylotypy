from pathlib import Path

from Bio import SeqIO, Seq
import pandas as pd


def fasta_to_dataframe_taxa(fasta_file, taxafile, save: bool = False):
    data_dir = Path("data")
    fasta_file = Path(fasta_file)
    if not fasta_file.is_file() or not Path(taxafile).is_file():
        print("Taxa files not found")
        return None
    records = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        records.append((record.id, str(record.seq)))
    df = pd.DataFrame(records, columns=["accession", "sequences"])
    taxtable_df = taxa_to_dataframe(taxafile)
    refdb = df.merge(taxtable_df, on="accession", how="left")
    if save:
        save_file = data_dir.joinpath(fasta_file.name.split(".")[0] + "_db.csv")
        refdb.to_csv(save_file, index=False)
    return refdb


def translate_sequence(dna_sequence):
    """
    Translate a DNA sequence to a protein sequence using BioPython.

    Parameters:
    - dna_sequence (str): DNA sequence to be translated.

    Returns:
    - str: Translated protein sequence.
    """
    # Create a Bio.Seq object from the DNA sequence
    seq_obj = Seq.translate(dna_sequence)

    # Translate the sequence to a protein sequence
    protein_sequence = str(seq_obj)

    return protein_sequence


def fasta_to_dataframe(fasta_file, **kwargs):
    """
    Read a FASTA file and store sequence names and sequences in a pandas DataFrame.

    Parameters:
    - fasta_file (str): Path to the input FASTA file.
    - translate (bool): Whether to translate DNA sequences to protein sequences.

    Returns:
    - pandas DataFrame: DataFrame containing sequence names and sequences.
    """
    # Initialize lists to store sequence names and sequences
    seq_names = []
    seq_sequences = []

    # Iterate over sequences in the FASTA file
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Append sequence name and sequence to lists
        seq_names.append(record.id)
        sequence = str(record.seq)
        if "translate" in kwargs:
            if kwargs["translate"]:
                sequence = translate_sequence(sequence)
        seq_sequences.append(sequence)

    # Create a DataFrame from the lists
    df = pd.DataFrame({'SequenceName': seq_names,
                       'Sequence': seq_sequences})
    if "gene" in kwargs:
        df["gene"] = kwargs["gene"]
        return df

    return df


def taxa_to_dataframe(taxa_file):
    file_path = Path(taxa_file)
    if not file_path.is_file():
        print("Taxa file not found")
        return None
    taxa_table_df = pd.read_csv(file_path, sep="\t", names=["accession", "taxonomy"])
    taxa_levels = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "_"]
    taxa_table_df[taxa_levels] = taxa_table_df['taxonomy'].str.split(';', expand=True)
    taxa_table_df['taxonomy'] = taxa_table_df['taxonomy'].str.rstrip(';')
    taxa_table_df.drop(["_"], axis="columns", inplace=True)
    return taxa_table_df


def convert_dict_keys_to_int(d):
    """
    Convert the keys of a dictionary from strings to integers.

    Parameters:
    d (dict): The dictionary with string keys to be converted.

    Returns:
    dict: A new dictionary with integer keys.
    """
    return {int(key): value for key, value in d.items()}
