from collections import defaultdict
from functools import partial
import json
import multiprocessing as mp
import numpy as np
import re
from time import perf_counter
from typing import Optional

from pathlib import Path
from phylotypy import kmers
from phylotypy.utilities import read_fasta


global db


class GetKmerDB:
    _instance: Optional["GetKmerDB"] = None
    _is_initialized: bool = False

    def __new__(cls, mod_file, mod_shape, genera_file) -> "GetKmerDB":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            return cls._instance
        else:
            print("database already exists")
            return cls._instance

    def __init__(self, mod_file, mod_shape, genera_file) -> None:
        if not self._is_initialized:
            self.mod_file = mod_file
            self.mod_shape = mod_shape
            self.genera_file = genera_file
            self.db_ = self.load_db()

    def load_db(self) -> kmers.KmerDB:
        db_ = kmers.KmerDB(conditional_prob=np.memmap(self.mod_file,
                                                      dtype=np.float16,
                                                      mode="c",
                                                      shape=self.mod_shape),
                           genera_idx=kmers.genera_str_to_index(np.load(self.genera_file, allow_pickle=True)),
                           genera_names=np.load(self.genera_file, allow_pickle=True)
                           )
        return db_

    @property
    def genera_names(self):
        return self.db_.genera_names

    @property
    def conditional_prob(self):
        return self.db_.conditional_prob

    @property
    def genera_idx(self):
        return self.db_.genera_idx


class Predict:
    def __init__(self):
        self.db = None
        self.mod_shape = None
        self.mod_file = None
        self.genera_file = None
        return

    def load_db(self, mod_file, mod_shape, genera_file):
        self.mod_file = mod_file
        self.mod_shape = mod_shape
        self.genera_file = genera_file
        self.db = GetKmerDB(self.mod_file, self.mod_shape, self.genera_file)

    @staticmethod
    def init_worker(mod_file, mod_shape, genera_file):
        global db
        db = GetKmerDB(mod_file, mod_shape, genera_file)

    @staticmethod
    def classify(bs_kmer: list | np.ndarray,
                 min_confid: int = 80,
                 n_levels: int = 6) -> str:
        classified_list = kmers.classify_bootstraps(bs_kmer, db.conditional_prob)
        classification = kmers.consensus_bs_class(classified_list, db.genera_names)
        classification_filtered = kmers.filter_taxonomy(classification, min_confid)
        return kmers.print_taxonomy(classification_filtered, n_levels)

    def predict(self, sequences: list | np.ndarray):
        kmer_list = kmers.detect_kmers_across_sequences_mp(sequences, verbose=True)
        bootstrap_func = partial(kmers.bootstrap, n_bootstraps=100)

        print("Bootstrapping sequences...")
        with mp.Pool(processes=8,
                     initializer=self.init_worker,
                     initargs=(self.mod_file, self.mod_shape, self.genera_file)) as pool:
            bs_results_collected = pool.imap(bootstrap_func, kmer_list, chunksize=10)
            bs_kmers = [np.array(result) for result in bs_results_collected]

            print("Classifying sequences...")
            classify_partial = partial(self.classify)
            classified_results = pool.map(classify_partial, bs_kmers, chunksize=10)

        return classified_results


def load_db(conf_file: str | Path) -> dict:
    with open(conf_file, 'r') as f:
        config = json.load(f)

    mod_files = defaultdict(object)
    # Extract the necessary information from the config
    mod_files["model_dir"] = Path(config['model_dir'])
    mod_files["model_file"] = Path(config['model_dir']).joinpath(config['model'])
    mod_files["genera"] = Path(config['model_dir']).joinpath(config['genera'])
    mod_files["model_shape"] = tuple(config['model_shape'])

    return mod_files


if __name__ == "__main__":
    db_files = load_db(Path.home() / "PycharmProjects/phylotypy_data/local_data/models/rdp/model_config.json")

    classifier = Predict()
    classifier.load_db(db_files["model_file"], db_files["model_shape"], db_files["genera"])

    test_seqs = Path.home() / "PycharmProjects/phylotypy_data/data/dna_moving_pictures.fasta"
    seqs = read_fasta.read_taxa_fasta(test_seqs)

    ##

    start = perf_counter()
    classify_seq = classifier.predict(seqs["sequence"])
    end = perf_counter()
    print(f"Time taken: {(end - start):.2f}")

    ##
    def remove_confidence(col):
        return re.sub(r"\(\d+\)", "", col)


    seqs["classification"] = classify_seq
    seqs["taxonomy"] = seqs["classification"].apply(remove_confidence)
    print(seqs.head())
    print(seqs.shape)
