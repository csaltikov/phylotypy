##
from multiprocessing import freeze_support
from pathlib import Path
import time

import pandas as pd

from utilities import fasta_to_dataframe
import phylotypy

if __name__ == "__main__":
    freeze_support()

    ##
    """
    Notes:
    Load data, can use training_data.py to download and produce trainset19_072023_db.csv
    or do it yourself another way.
    """

    db_file_path = Path("data/trainset19_072023_db.csv")
    db = pd.read_csv(db_file_path)
    print(f"Size of the database: {db.shape}")

    ##
    # Reload the module in case I edit the code
    kmer_size = 8
    classify = phylotypy.Classify()
    classify.multi_processing = True
    start = time.time()
    classify.fit(db["sequences"], db["taxonomy"], kmer_size=kmer_size, verbose=True)
    classify.verbose = False
    end = time.time()
    print(f"Run time {(end - start):.1f} seconds")

    ##
    # Classifying QIIME2 Moving Pictures rep-seqs-dada2.qza
    # https://docs.qiime2.org/2024.2/tutorials/moving-pictures/
    moving_pic = fasta_to_dataframe("data/dna_moving_pictures.fasta")

    # prepare the sequences and sequence name lists
    X_mov_pic = moving_pic["Sequence"]
    y_mov_pic = moving_pic["SequenceName"]

    # Classify
    predict_mov_pic = classify.predict(X_mov_pic, y_mov_pic)

    # Put results in a dataframe
    predict_mov_pic_df = phylotypy.summarize_predictions(predict_mov_pic)
    print(predict_mov_pic_df[["id", "Genus"]].head())

    ##
    # Testing a single organism sequences
    orio = fasta_to_dataframe("data/orio_16s.txt")  # pd.Dataframe

    # prepare the sequences and sequence name lists
    X_unk = orio["Sequence"]
    y_unk = orio["SequenceName"]

    # Classify the sequence
    predict_orio = classify.predict(X_unk, y_unk)

    # Report the results in a dataframe
    predict_orio_df = phylotypy.summarize_predictions(predict_orio)
    print(predict_orio_df[["id", "Genus"]])
