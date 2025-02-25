#!/usr/bin/env python3
from pathlib import Path
from time import perf_counter
import itertools

import numpy as np
import pandas as pd

from phylotypy import kmers
from phylotypy.utilities import read_fasta

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, verbose=1)


def parallel_conditional_prob(kmers_df: pd.DataFrame, priors) -> np.ndarray:
    genera_idx = kmers.genera_str_to_index(kmers_df["id"])
    unique_genus_counts = np.unique(genera_idx, return_counts=True)[1]
    n_genera = len(unique_genus_counts)
    n_kmers = len(priors)

    ## Building genus conditional probablities
    def join_list(df):
        return list(itertools.chain.from_iterable(df["kmers_list"]))

    def counts_idx(s):
        return np.unique(s, return_counts=True, return_index=False)

    print("Starting the calculation")
    results = (kmers_df.groupby("id")
               .parallel_apply(join_list)
               .reset_index(name="kmers_list")
               .loc[:, "kmers_list"]
               .parallel_apply(counts_idx)
               )

    ##
    # Pandas way:
    # genus_count = pd.DataFrame(0, index=range(n_kmers),
    #                            columns=range(n_genera),
    #                            dtype=np.float32)
    #
    # for i, (idx, counts) in enumerate(results):
    #     genus_count.iloc[idx, i] = counts
    #
    # numpy way:
    print("Filling in the genus_count array")
    genus_count = np.zeros((n_kmers, n_genera), dtype=np.float32)

    for i, (idx, counts) in enumerate(results):
        genus_count[idx, i] += counts

    wi_pi = (genus_count + priors.reshape(-1, 1))
    m_1 = (unique_genus_counts + 1)

    print("Final calculation...")
    genus_cond_prob = np.log(np.divide(wi_pi, m_1)).astype(np.float16)

    return genus_cond_prob


##
if __name__ == "__main__":
    ##
    import pickle

    home = Path.home()
    rdp_fasta = Path("../data/rdp_16S_v19.dada2.fasta")
    silva_fasta = home / "silva_nr99_v138.2_toGenus_trainset.fa"

    ##
    ref_fasta = read_fasta.read_taxa_fasta(silva_fasta)
    kmers_df = ref_fasta[~ref_fasta["id"].str.contains("Eukar|Incertae|Unknown")].copy()
    kmers_df["kmers_list"] = kmers_df["sequence"].parallel_apply(kmers.detect_kmers)

    priors = kmers.calc_word_specific_priors(kmers_df["kmers_list"].to_list())

    start = perf_counter()
    conditional_prob = parallel_conditional_prob(kmers_df, priors)
    end = perf_counter()
    print(f"Function took {end-start:.2f} seconds")

    with open("database_silva.pkl", "wb") as f:
        pickle.dump(conditional_prob, f)
    ##
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50)
    #
    # import cProfile
    #
    # seqs = X_test.tolist()
    # cprofile = cProfile.run('kmers.classify_sequence(seqs[0], database)')
