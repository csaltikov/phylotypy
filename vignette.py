##
import time
from multiprocessing import freeze_support
from pathlib import Path

import pandas as pd

from phylotypy import predict
from phylotypy.utilities import utilities, read_fasta

if __name__ == "__main__":
    freeze_support()

    ##
    """
    Notes:
    Load data, can use training_data.py to download and produce trainset19_072023_db.csv
    or do it yourself another way.
    """
    home = Path.home()

    silva_data = Path("data/silva_138.2_ssuref_sub_prok.csv")
    print(f"ref db {silva_data.name} fount: {silva_data.exists()}")

    rdp_data = Path("data/trainset19_072023_db.csv")
    print(f"ref db {rdp_data.name} fount: {rdp_data.exists()}")

    pds_data = home / "Documents/Projects/phylotypyr/phylotypr_proj/data/trainset9_pds.csv"

    db = pd.read_csv(pds_data)
    print(f"Size of the database: {db.shape}")

    ##
    # Reload the module in case I edit the code
    kmer_size = 8
    classify = predict.Classify()
    classify.multi_processing = True
    classify.save_db = Path.home()

    start = time.time()
    classify.fit(db["sequence"],
                 db["taxonomy"],
                 multi=True,
                 n_cpu=6)
    end = time.time()
    print(f"Run time {(end - start):.1f} seconds")

    ##
    # Classifying QIIME2 Moving Pictures rep-seqs-dada2.qza
    # https://docs.qiime2.org/2024.2/tutorials/moving-pictures/
    moving_pic = utilities.fasta_to_dataframe("data/dna_moving_pictures.fasta")

    # prepare the sequences and sequence name lists
    X_mov_pic = moving_pic["sequence"]
    y_mov_pic = moving_pic["id"]

    ##
    # Classify
    predict_mov_pic = classify.predict(X_mov_pic[0:20], y_mov_pic[0:20])

    # Put results in a dataframe
    predict_mov_pic_df = predict.summarize_predictions(predict_mov_pic)
    print(predict_mov_pic_df[["id", "Genus"]].head())

    ##
    # Testing a single organism sequences
    orio = utilities.fasta_to_dataframe("data/orio_16s.txt")  # pd.Dataframe

    # prepare the sequences and sequence name lists
    X_unk = orio["sequence"]
    y_unk = orio["id"]

    ##
    # Classify the sequence, set multi_p to false, it's faster
    predict_orio = classify.predict(X_unk, y_unk, multi_p=False)

    # Report the results in a dataframe
    predict_orio_df = predict.summarize_predictions(predict_orio)
    print(predict_orio_df[["id", "classification"]])

    ##
    test_seq = read_fasta.read_taxa_fasta("data/unknown_barnesiella.fa")
    test_res = classify.predict(test_seq["sequence"], test_seq["id"], multi_p=False, min_confid=50)
    test_res_df = predict.summarize_predictions(test_res)
    print(test_res_df[["id", "classification"]])
