import json
from typing import Optional
from pathlib import Path

import numpy as np

from phylotypy import kmers


class GetKmerDB:
    _instance: Optional["GetKmerDB"] = None
    _is_initialized: bool = False

    def __new__(cls, mod_file, mod_shape, genera_file) -> "GetKmerDB":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            return cls._instance
        else:
            # print("database already exists")
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


def load_db(conf_file):
    with open(conf_file, 'r') as f:
        config = json.load(f)

    print(*config)
    # Extract the necessary information from the config
    mod_dir = Path(config["model_dir"])
    mod_file = mod_dir / config["model"]
    genera = mod_dir / config["genera"]
    model_shape = config["model_shape"]

    db = GetKmerDB(mod_file, model_shape, genera)

    return db


if __name__ == "__main__":
    db = load_db(Path.home() / "PycharmProjects/phylotypy_data/local_data/models/rdp/model_config.json")
    print(db.genera_names[0:10])
    print(db.conditional_prob[0:10])
