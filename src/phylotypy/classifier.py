import sys
from collections import defaultdict
from functools import partial
import json
import multiprocessing as mp
import numpy as np
import re
from time import perf_counter

from pathlib import Path
from phylotypy import kmers, get_kmer_db
from phylotypy.utilities import read_fasta


global db


class Predict:
    def __init__(self):
        self.db = None
        self.mod_shape = None
        self.mod_file = None
        self.genera_file = None
        return

    def load_db(self, config):
        db_config = get_db_files(config)
        self.mod_file = db_config["model_file"]
        self.mod_shape = db_config["model_shape"]
        self.genera_file = db_config["genera"]
        self.db = get_kmer_db.GetKmerDB(self.mod_file, self.genera_file, self.mod_shape)

    @staticmethod
    def init_worker(mod_file, genera_file, mod_shape):
        global db
        db = get_kmer_db.GetKmerDB(mod_file, genera_file, mod_shape)

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

        print(f"Bootstrapping {len(kmer_list)} sequences")
        with mp.Pool(processes=8,
                     initializer=self.init_worker,
                     initargs=(self.mod_file, self.genera_file, self.mod_shape)) as pool:
            bs_results_collected = pool.imap(bootstrap_func, kmer_list, chunksize=10)
            bs_kmers = [np.array(result) for result in bs_results_collected]

            print("Classifying sequences")
            classify_partial = partial(self.classify)
            classified_results = pool.map(classify_partial, bs_kmers, chunksize=10)
            print("Done classifying sequences")

        return classified_results


def get_db_files(conf_file: str | Path | dict) -> dict:
    if isinstance(conf_file, Path):
        if conf_file.exists():
            with open(conf_file, 'r') as f:
                config = json.load(f)
        else:
            print("File not found")
    elif isinstance(conf_file, str):
        if Path(conf_file).exists():
            with open(conf_file, 'r') as f:
                config = json.load(f)
        else:
            print("File not found")
    elif isinstance(conf_file, dict):
        config = conf_file
        print("Config is dict")
    else:
        print("Config file must be a JSON file or a dictionary.")
        raise TypeError

    mod_files = defaultdict(object)
    # Extract the necessary information from the config
    mod_files["model_dir"] = Path(config['model_dir'])
    mod_files["model_file"] = Path(config['model_dir']).joinpath(config['model'])
    mod_files["genera"] = Path(config['model_dir']).joinpath(config['genera'])

    for key, value in mod_files.items():
        if not isinstance(value, Path):
            print(f"{value} is not a valid path")
            sys.exit(1)
        if not Path(value).exists():
            print(f"{value.name} was not found")
            sys.exit(1)

    mod_files["model_shape"] = tuple(config['model_shape'])

    return mod_files


if __name__ == "__main__":
    config_file = "../../training_data/models/rdp/model_config.json"
    print(Path(config_file).exists())

    config_dir = "../../training_data/models/rdp"

    config_dict = {
        "db_name": "mini_rdp",
        "model": "model_raw.rbf",
        "genera": "ref_genera.npy",
        "model_shape": [ 65536, 3883],
        "model_dir": config_dir
    }

    classifier = Predict()
    classifier.load_db(config_file)

    test_seqs = "../../data/dna_moving_pictures.fasta"
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
