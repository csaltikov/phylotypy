from collections import Counter, defaultdict
import re
from typing import Tuple, Any, List, Dict

import numpy as np
import pandas as pd

import kmers


class Phylotypy:
    def __init__(self):
        self.y_refs = None
        self.kmer_size = None
        self.model = None
        self.ref_genera = None
        self.detect_list = None
        self.ref_genera_idx = None
        self.boot = 10
        self.verbose = True
        self.n_levels = 6

    def fit(self, X, y, kmer_size: int = 8, verbose: bool = False):
        self.verbose = verbose
        print("Fitting model")
        self.y_refs = y
        self.kmer_size = kmer_size
        db_model = kmers.build_kmer_database(X, self.y_refs, self.kmer_size)
        self.model = db_model["conditional_prob"]
        self.ref_genera = db_model["genera_names"]
        self.ref_genera_idx = db_model["genera_idx"]
        print("Done fitting model")

    def predict(self, nuc_sequences: list, y_test: list, kmer_size: int = 8, **kwargs):
        if "boot" in kwargs:
            self.boot: int = kwargs["boot"]
        # Convert each nucleotide sequence in the database to base4
        # Get a list of kmer indices for each sequence in the database, a list of lists
        self.detect_list = kmers.detect_kmers_across_sequences(nuc_sequences, kmer_size, self.verbose)
        # Get the posteriors for each genera in the database
        posteriors = self.predict_(y_test)
        if self.verbose:
            print("Done predicting")
        return posteriors

    def predict_(self, genera_list: list) -> list[Any]:
        # dictionary to hold the predictions/posteriors
        if self.verbose:
            print("Calculating posterior probabilities p(S|G)")

        # store prediction results and certainty percent
        predict_taxa_arr = np.empty(len(genera_list), dtype=object)
        certainty_arr = np.zeros(len(genera_list), dtype=float)

        # loop through each genus
        for i, test_org in enumerate(genera_list):
            # Show progress if set to True
            # get the list of kmer indices for the specific test sequence
            kmer_index = self.detect_list[i]
            # stores all the replicate max_idx's as an array
            max_idx_arr = self.bootstrap(kmer_index, n_bootstraps=self.boot)
            # get the max_idx from the list and it's percent
            predict_taxa, certainty = self.consensus_bs_class(max_idx_arr)
            # In case you want to see the progress
            if i % (len(genera_list) // 50) == 0 and self.verbose:
                print(f"Processed {i * 100 / len(genera_list):.1f}% of the sequences")
            predict_taxa_arr[i] = predict_taxa
            certainty_arr[i] = certainty

        return [predict_taxa_arr, certainty_arr]

    def bootstrap(self, kmer_index: list, n_bootstraps: int = 100) -> np.array:
        """Random sample a test sequence and find the best match to the db of class"""
        bootstrap_max_ids = np.empty((n_bootstraps,), dtype=int)
        n_samples = len(kmer_index) // 8
        i = 0
        # Set the seed
        # np.random.seed(2112)
        while i < n_bootstraps:
            kmer_samples = np.random.choice(kmer_index, size=n_samples, replace=True)
            max_id = self.classify_bs(kmer_samples)
            bootstrap_max_ids[i] = max_id
            i += 1
        return bootstrap_max_ids

    def classify_bs(self, kmer_index: list):
        """Screens a test sequence against all classes in the model"""
        model_mask = self.model[kmer_index, :]
        class_prod = np.prod(model_mask, axis=0, keepdims=True)
        max_idx = np.argmax(class_prod)
        return max_idx

    def consensus_bs_class(self, incoming_bootstrap: np.array) -> np.array:
        # Convert the indices in the bootstrap array to taxonomy
        taxonomy: np.array = self.ref_genera[incoming_bootstrap]
        # sometimes taxonomy is empty or has "none" value
        mask = taxonomy != None
        taxonomy_filtered = taxonomy[mask]
        taxonomy_split = np.array([line.split(";") for line in taxonomy_filtered])
        # this is a ND array with rows equal to number of bootstraps
        # and columns of cumulative join taxonomy lineage
        return self.create_con(taxonomy_split)

    def create_con(self, arr: np.array):
        taxa = np.empty(arr.shape[1], dtype=object)
        scores = np.empty(arr.shape[1], dtype=object)

        def cumulative_join(col):
            return [";".join(col[:i + 1]) for i in range(len(col))]

        taxa_arr = np.apply_along_axis(cumulative_join, 1, arr)

        # get best ID and score for each column of the taxa array
        for k in range(taxa_arr.shape[1]):
            taxa[k], scores[k] = self.get_max(taxa_arr[:, k])

        best_id = self.filter_scores(scores)
        best_taxa = self.print_taxonomy(taxa[best_id])
        return best_taxa, scores[best_id]

    @staticmethod
    def get_max(col):
        """Helper for create_con determines id and fraction"""
        freq = Counter(col)
        id, frac = freq.most_common(1)[0]
        return id, int(frac / len(col) * 100)

    @staticmethod
    def filter_scores(scores: np.array):
        """Helper for create_con determines finds the best id"""
        threshold = scores >= 80
        scores_filt = np.where(threshold)[0]
        if scores_filt.size > 0:
            return scores_filt[-1]
        else:
            return np.argmax(scores)

    def print_taxonomy(self, taxonomy: np.array,) -> str:
        taxonomy_split: list = re.findall(r'[^;]+', taxonomy)
        n_taxa_levels = len(taxonomy_split)
        updated_taxonomy = taxonomy_split + ["unclassified"] * (self.n_levels - n_taxa_levels)
        return ";".join(updated_taxonomy)


def summarize_predictions(predictions: list[str | int], genera_list: list[str]) -> pd.DataFrame:
    predicted_taxa, certainty = predictions
    data = {"id": genera_list, "full lineage": predicted_taxa, "stat": certainty}
    classified_df = pd.DataFrame.from_dict(data)
    taxa_levels = ["Domain", "Phylum", "Class", "Order", "Family", "Genus"]
    classified_df[taxa_levels] = classified_df["full lineage"].str.split(";", expand=True)
    return classified_df


def genera_index_mapper(genera_list: list) -> dict:
    # Create a dictionary mapping unique values to integers
    unique_genera = np.unique(genera_list)
    factor_map = {val: idx for idx, val in enumerate(unique_genera)}

    return factor_map


# def highest_counts(idx_array: np.array, **kwargs):
#     """Calculate frequency of each factor"""
#     # In case the idx_array is empty or the last element
#     if np.ndim(idx_array) == 0:
#         return "unclassified", 0
#
#     frequency = Counter(idx_array)
#
#     # Get factor with the highest count
#     most_common_factor, highest_count = frequency.most_common(1)[0]
#
#     # Calculate percentage
#     highest_count_percent: float = (highest_count / idx_array.shape[0]) * 100
#
#     if "verbose" in kwargs:
#         if kwargs["verbose"]:
#             print("**highest_counts**", most_common_factor, highest_count_percent)
#
#     return most_common_factor, highest_count_percent


# def classify_column(column: np.array):
#     level, score = highest_counts(column, verbose=False)
#     if score >= 70:
#         return level, score
#     else:
#         return "unclassified", 0


# def update_lineage(incoming_array: np.array):
#     # need to get the scores for the columns
#     classified_column, scores = np.apply_along_axis(classify_column, axis=0, arr=incoming_array)
#     best_index = np.where(classified_column == "unclassified")[0]
#     if best_index.sum() == 0:
#         score = scores[-1]
#     else:
#         score = scores[best_index][0]
#     return ";".join(classified_column), score
