import json
from typing import Any

import numpy as np
import pandas as pd
import re

from numpy import ndarray, dtype

try:
    import phylotypy.kmers as kmers
except ImportError:
    import kmers


class Classify:
    def __init__(self):
        self.kmer_size = None
        self.model = None
        self.ref_genera = None
        self.detect_list = None
        self.ref_genera_idx = None
        self.boot = 100
        self.save_db = True
        self.verbose = True
        self.n_levels = 6  # defaults levels down to genus

    def fit(self, X, y, kmer_size: int = 8, verbose: bool = False, **kwargs):
        self.verbose = verbose
        print("Fitting model")
        self.kmer_size = kmer_size
        db_model = kmers.build_kmer_database(X, y, self.kmer_size, self.verbose)

        self.model = db_model["conditional_prob"]
        self.ref_genera = db_model["genera_names"]
        self.ref_genera_idx = db_model["genera_idx"]
        print("Done fitting model")

        # Save config file
        if self.save_db:
            db_model["conditional_prob"].tofile("model_raw.rbf")
            np.save("ref_genera.npy", self.ref_genera)
            json_config = kwargs.get('config', "phylotypy_config.json")
            self.save_config(json_config)

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

    def predict_(self, genera_list: list, min_confid: int = 80) -> dict[str, Any]:
        # dictionary to hold the predictions/posteriors
        n_sequences = len(genera_list)
        if self.verbose:
            print(f"Classifying {len(genera_list)} sequences")

        # store prediction results and certainty percent
        predict_consensus = np.empty(n_sequences, dtype=list)
        test_org_arr = np.empty(n_sequences, dtype=list)

        # loop through each genus
        for i, test_org in enumerate(genera_list):
            # Show progress if set to True
            # get the list of kmer indices for the specific tests sequence
            kmer_index = self.detect_list[i]

            # Gest the best ref_genera_idx values as a 1D array of len(boot)
            max_idx_arr = self.bootstrap(kmer_index, n_bootstraps=self.boot)

            try:
                # Classify boostrap samples
                classification = self.consensus_bs_class(max_idx_arr)

                # Filter classification by min confidence score
                classification_filtered = kmers.filter_taxonomy(classification, min_confid)

                # Format consensus
                consensus = kmers.print_taxonomy(classification_filtered, self.n_levels)

                # add the consensus taxonomy to the array
                predict_consensus[i] = consensus
                test_org_arr[i] = test_org

            except Exception as e:
                # Catch errors along the work flow
                print(f"Couldn't classify {test_org}, trying increasing bootstraps samples: {e}")
                predict_consensus[i] = ";".join(np.array(["unclassified(0)"] * self.n_levels))
                test_org_arr[i] = test_org
                continue

            try:
                if i != 0 and i % (n_sequences // 50) == 0 and self.verbose:
                    print(f"Processed {i * 100 / n_sequences:.1f}% of the sequences")
            except ZeroDivisionError:
                pass

        return dict(id=test_org_arr, classification=list(predict_consensus))

    def bootstrap(self, kmer_index: list, n_bootstraps: int = 100) -> np.ndarray:
        """Random sample a tests sequence and find the best match to the db of class"""
        bootstrap_max_ids = np.empty(n_bootstraps, dtype=int)
        n_samples = len(kmer_index) // 8

        for i in range(n_bootstraps):
            kmer_samples = np.random.choice(kmer_index, size=n_samples, replace=True)
            max_id = self.classify_bs(kmer_samples)
            bootstrap_max_ids[i] = max_id
        return bootstrap_max_ids

    def classify_bs(self, kmer_index: list):
        """Screens a tests sequence against all classes in the model"""
        class_prod = np.sum(self.model[kmer_index, :], axis=0)
        max_idx = np.argmax(class_prod)
        return max_idx

    def consensus_bs_class(self, incoming_bootstrap: np.array) -> dict[str, ndarray[Any, dtype[Any]] | Any]:
        # Convert the indices in the bootstrap array to taxonomy
        taxonomy: np.array = self.ref_genera[incoming_bootstrap]
        # sometimes taxonomy is empty or has "none" value
        mask = taxonomy != None
        taxonomy_filtered = taxonomy[mask]

        taxonomy_split = np.array([line.split(";") for line in taxonomy_filtered])

        def cumulative_join(col):
            join_taxa = [";".join(col[:i + 1]) for i in range(len(col))]
            return join_taxa

        taxa_cum_join_arr = np.apply_along_axis(cumulative_join, 1, taxonomy_split)

        # taxa_string, confidence = np.apply_along_axis(kmers.get_consensus, axis=0, arr=taxa_cum_join_arr)
        taxa_confidence = np.apply_along_axis(kmers.get_consensus, axis=0, arr=taxa_cum_join_arr)

        return {'taxonomy': np.array(taxa_confidence[0][-1].split(";")), 'confidence': taxa_confidence[1]}

    def save_config(self, config_file):
        model_shape = self.model.shape
        config = {
            "model_file": "data/model_raw.rbf",
            "genera": "data/ref_genera.npy",
            "model_shape": list(model_shape)
        }

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_file}")


def summarize_predictions(classified: dict, n_levels: int = 6):
    classified_df = pd.DataFrame(classified)
    taxa_levels_full = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    taxa_levels = taxa_levels_full[:n_levels]
    tax_level_codes = [f"{t[0].lower()}__" for t in taxa_levels]
    classified_df[taxa_levels] = classified_df["classification"].str.split(";", expand=True)

    def join_taxa(taxa_split):
        return ";".join([f'{tax_level_codes[i]}{tax}' for i, tax in enumerate(taxa_split)])

    def remove_confidence(col):
        return col.str.replace(r"\(\d+\)", "", regex=True)

    classified_df[taxa_levels] = classified_df[taxa_levels].apply(remove_confidence)
    classified_df["observed"] = classified_df[taxa_levels].apply(lambda row: ';'.join(row.values), axis=1)
    classified_df["lineage"] = classified_df[taxa_levels].apply(lambda row: join_taxa(row.values), axis=1)
    return classified_df


def genera_index_mapper(genera_list: list) -> dict:
    # Create a dictionary mapping unique values to integers
    unique_genera = np.unique(genera_list)
    factor_map = {val: idx for idx, val in enumerate(unique_genera)}

    return factor_map

if __name__ == "__main__":
    print("Phylotypy v1.0")
