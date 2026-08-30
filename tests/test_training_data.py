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

        observed = training_data.filter_train_set(id_df, n_levels=2)
        obs_true = observed["id"].str.contains("Shewanella").values

        expected = np.any([True])
        self.assertEqual(obs_true, expected)

        obs_true = observed["id"].str.contains("Eukaryota").values
        self.assertFalse(obs_true, expected)

    def test_filter_train_set_n_levels_7_handles_full_level_range(self):
        """Real reference databases (e.g. raw SILVA releases) span a wide range
        of taxonomy depths before cleaning, including occasional malformed
        entries with more than 7 levels. filter_train_set(n_levels=7) must
        filter these down to genus/species-level entries without crashing.

        Taxa strings below are real examples pulled from
        silva_nr99_v138.2_toSpecies_trainset.fa.gz (levels 1-7); the 8-level
        entry is that same real 7-level string with an extra field appended,
        since no naturally-occurring >7-level entries exist in that file.
        """
        too_shallow = [
            "Archaea",                                                                    # 1 level
            "Bacteria;PAUC34f",                                                           # 2 levels
            "Bacteria;Bacillota;BRH-c20a",                                                # 3 levels
            "Bacteria;Cyanobacteriota;Cyanobacteriia;Chloroplast",                        # 4 levels
            "Bacteria;Chlamydiota;Chlamydiia;Chlamydiales;Simkaniaceae",                  # 5 levels
        ]
        six_level = "Bacteria;Bacillota;Bacilli;Bacillales;Bacillaceae;Anoxybacillus"
        seven_level = ("Bacteria;Pseudomonadota;Gammaproteobacteria;Enterobacterales;"
                       "Vibrionaceae;Vibrio;halioticoli")
        too_deep = seven_level + ";strainX"  # 8 levels: malformed

        # six_level/seven_level appear twice each so they survive the
        # post-collapse singleton filter (anything seen only once is dropped).
        ids = too_shallow + [six_level, six_level, seven_level, seven_level, too_deep]
        df = pd.DataFrame({
            "id": ids,
            "sequence": ["ACGT" * 10 for _ in ids],
        })

        # threshold=1 disables the low-representation species collapse so the
        # surviving ids are simple to assert on directly.
        observed = training_data.filter_train_set(df, n_levels=7, threshold=1)

        for shallow_id in too_shallow:
            self.assertFalse((observed["id"] == shallow_id).any())
        self.assertFalse(observed["id"].str.contains("strainX").any())

        self.assertTrue((observed["id"] == f"{six_level};Anoxybacillus_sp").any())
        self.assertTrue((observed["id"] == f"{seven_level.rsplit(';', 1)[0]};Vibrio_halioticoli").any())


if __name__ == "__main__":
    print("hello")
