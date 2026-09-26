import gzip
import os
from pathlib import Path
import pandas as pd
from . import helpers


def df_to_fasta(df: pd.DataFrame, fasta_file: str | Path, line_width: int = 60) -> None:
    """Write a pandas dataframe to a fasta file.

    Parameters:
        df (pd.DataFrame): must contain 'id' and 'sequence' columns.
        fasta_file (string or pathlib.Path()): output path. Written as gzip if the
            path ends in '.gz'.
        line_width (int): wrap sequence lines to this many characters (default: 60).
            Set to 0 to write each sequence on a single line.

    Examples:
        >>> df_to_fasta(ref_db, "my_fasta.fa")
    """
    if "id" not in df.columns or "sequence" not in df.columns:
        raise ValueError("df must contain 'id' and 'sequence' columns.")

    gz_file = str(fasta_file).endswith(".gz")
    open_func = gzip.open if gz_file else open
    mode = "wt" if gz_file else "w"

    fasta = helpers.path_helper(fasta_file)

    with open_func(Path(fasta), mode) as f:
        for seq_id, sequence in zip(df["id"], df["sequence"]):
            f.write(f">{seq_id}\n")
            if line_width > 0:
                for i in range(0, len(sequence), line_width):
                    f.write(sequence[i:i + line_width] + "\n")
            else:
                f.write(sequence + "\n")
