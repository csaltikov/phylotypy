import unittest
import numpy as np
import pandas as pd
from phylotypy import training_data


class TestTrainingData(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = {"id": ["Bacteria;Incertae Sedis",
                            "Bacteria;Candidatus",
                            "Eukaryota;worm",
                            "Prokaryota;Shewanella"]}

    def test_filter_train_set(self):
        id_df = pd.DataFrame(self.ids)

        observed = training_data.filter_train_set(id_df)
        obs_true = observed["id"].str.contains("Shewanella").values

        expected = np.array([True])
        self.assertEqual(obs_true, expected)

        obs_true = observed["id"].str.contains("Eukaryota").values
        self.assertFalse(obs_true, expected)


if __name__ == "__main__":
    print("hello")
