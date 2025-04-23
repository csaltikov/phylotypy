import unittest
import pandas as pd

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, verbose=1)
import math
import numpy as np


class TestPandarella(unittest.TestCase):
    def setUp(self) -> None:
        df_size = int(100)
        self.df = pd.DataFrame(dict(a=np.random.randint(1, 8, df_size),
                               b=np.random.rand(df_size)))

    def func(self, x):
        return math.sin(x.a ** 2) + math.sin(x.b ** 2)


    def test_pandarella(self):
        res = self.df.apply(self.func, axis=1)
        res_parallel = self.df.parallel_apply(self.func, axis=1)
        # self.assertTrue(res.equals(res_parallel))


if __name__ == "__main__":
    unittest.main()
