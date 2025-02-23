from collections import defaultdict
import json
from typing import Optional
from pathlib import Path

import numpy as np

from phylotypy import kmers


class GetKmerDB:
    _instance: Optional["GetKmerDB"] = None
    _is_initialized: bool = False

    def __new__(cls, mod_file, genera_file, mod_shape) -> "GetKmerDB":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            return cls._instance
        else:
            # print("database already exists")
            return cls._instance

    def __init__(self, mod_file, genera_file, mod_shape) -> None:
        if not self._is_initialized:
            self.mod_file = mod_file
            self.mod_shape = mod_shape
            self.genera_file = genera_file
            self.db_ = self.load_db_()

    def load_db_(self) -> kmers.KmerDB:
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


def load_db(config: dict|Path, **kwargs):
    if isinstance(config, Path):
        if config.exists():
            with open(config, 'r') as f:
                config = json.load(f)
        else:
            print("File not found")
    elif isinstance(config, dict):
        config = config
        print("Config is dict")
    else:
        print("Config file must be a JSON file or a dictionary.")
        raise TypeError

    # Extract the necessary information from the config
    model_files = defaultdict()
    model_file_dir = kwargs.get("mod_dir", Path(config["model_dir"]))
    if not model_file_dir.exists():
        raise FileNotFoundError

    print(model_file_dir)
    print(config["model"])
    model_files["mod_data"] = model_file_dir / config["model"]
    model_files["genera"] = model_file_dir / config["genera"]

    for key, value in model_files.items():
        if not isinstance(value, Path):
            print(f"{key} is not a path")
            raise TypeError
        if not Path(value).exists():
            print(f"{key} is not a file")
            raise FileNotFoundError

    db = GetKmerDB(model_files["mod_data"], model_files["genera"], config["model_shape"])

    return db


if __name__ == "__main__":
    db = load_db(Path.home() / "PycharmProjects/phylotypy_data/local_data/models/rdp/model_config.json")
    print(db.genera_names[0:10])
    print(db.conditional_prob[0:10])

    db_dict = {
        "db_name": "mini_rdp",
        "model": "model_raw.rbf",
        "genera": "ref_genera.npy",
        "model_dir": Path.home() / "PycharmProjects/phylotypy_data/local_data/models/rdp",
        "model_shape": [65536, 3883] }

    db2 = load_db(db_dict)
    print(db2.genera_names[0:10])
    print(db2.conditional_prob[0:10])
