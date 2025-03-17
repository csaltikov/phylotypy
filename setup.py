from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "phylotypy.classify_bootstraps.classify_bootstraps",
        ["src/phylotypy/classify_bootstraps/classify_bootstraps.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="phylotypy",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=cythonize(extensions),
)
