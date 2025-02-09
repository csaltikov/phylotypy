##
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from phylotypy.utilities import read_fasta
from phylotypy import predict, get_kmer_db


##
def make_model(train_data_json: json, kmer_size: int = 8):
    # check the user config file
    with open(train_data_json, 'r') as f:
        config = json.load(f)

    # mandatory
    fasta = config["fasta"]
    output_dir = Path(config["output"])
    db_name = config['db_name']

    # read in the fasta and taxa to make the classifier model
    X, y = get_db_sequences(fasta)
    classifier = predict.Classify()
    classifier.multi_processing = True
    classifier.fit(X, y, multi=True, n_cpu=12, chunk_size=20000)
    classifier.verbose = True

    # save the model and genera to dir specified in the config
    model = classifier.model
    print(model.shape)
    model_path = output_dir / "model_raw.rbf"  # must be raw binary format
    model.tofile(model_path)

    ref_genera = classifier.ref_genera
    ref_genera_path = output_dir / "ref_genera.npy"
    np.save(ref_genera_path, ref_genera)

    ref_genera_idx = classifier.ref_genera_idx
    ref_genera_idx_path = output_dir / "ref_genera_idx.npy"
    np.save(ref_genera_idx_path, ref_genera_idx)

    # save paths to a config file for the classifier
    model_json = output_dir / "model_config.json"
    model_config = {
        "db_name": db_name,
        "model": "model_raw.rbf",
        "genera": "ref_genera.npy",
        "model_shape": [4**kmer_size, len(ref_genera)],
        "model_dir": str(model_path.parent)
    }

    with open(model_json, 'w') as j:
        json.dump(model_config, j, indent=4)
        print(f"Training data files saved in {model_config['model_dir']}")

    return classifier


def get_db_sequences(fasta):
    db = read_fasta.read_taxa_fasta(fasta)

    X_train = db["sequence"].tolist()
    y_train = db["id"].tolist()

    if ";" not in y_train[0]:
        print("Must taxonomy must contain ; to separate taxa levels")
        sys.exit()

    print(db.head())
    print(f"Size of the database: {db.shape[0]} sequences")

    return X_train, y_train


def check_train_config_file(config_path):
    # Check if the configuration file exists
    if not Path(config_path).is_file():
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)

    # Load the configuration file
    with open(config_path, 'r') as file:
        try:
            config = json.load(file)
        except json.JSONDecodeError:
            print(f"Error: Configuration file '{config_path}' is not a valid JSON.")
            sys.exit(1)

    # Check if required keys are present
    required_keys = ["fasta", "output"]
    for key in required_keys:
        if key not in config:
            print(f"Error: Key '{key}' not found in the configuration file.")
            print(f"Config must contain file paths: ", *required_keys)
            sys.exit(1)

    # Check if the paths specified in the keys exist
    for key in required_keys:
        if not Path(config[key]).exists():
            print(f"Error: File path specified for '{key}' does not exist: {config[key]}")
            sys.exit(1)

    print("Training data config passed successfully.")


def get_db(config_file):
    db = get_kmer_db.load_db(config_file)
    return db


##
if __name__ == "__main__":
    ##
    # What the training data config looks like:
    # output is the path to where the genera and model files are stored
    """
    {
    "db_name": "rdp",
    "fasta": "training_data/trainset19_072023.rdp/trainset19_072023.rdp.fasta",
    "output": "models/rdp"
    }
    """
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file.json>")
        sys.exit(1)

    config_file_mkdb = sys.argv[1]
    check_train_config_file(config_file_mkdb)

    nb_classifier = make_model(config_file_mkdb)


