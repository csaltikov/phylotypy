import unittest

import numpy as np
from phylotypy import bootstrap


class TestBootstrap(unittest.TestCase):
    def setUp(self) -> None:
        self.ref_genera = np.array(["A;a;A", "A;a;B", "A;a;C", "A;b;A", "A;b;B", "A;b;C"])

    def test_split_taxa(self):
        observed = bootstrap.split_taxa_arr(self.ref_genera)
        expected = np.array([["A", "a", "A"],
                             ["A", "a", "B"],
                             ["A", "a", "C"],
                             ["A", "b", "A"],
                             ["A", "b", "B"],
                             ["A", "b", "C"]])
        self.assertTrue(np.array_equal(observed, expected))


if __name__ == "__main__":
    unittest.main()
