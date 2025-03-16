from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "phylotypy.classify_bootstraps.classify_bootstraps",
        ["src/phylotypy/classify_bootstraps/classify_bootstraps.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="phylotypy",
    packages=["phylotypy", "phylotypy.classify_bootstraps"],
    package_dir={"": "src"},
    ext_modules=cythonize(extensions),
)
