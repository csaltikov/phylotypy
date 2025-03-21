from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        "phylotypy.classify_bootstraps.classify_bootstraps",
        ["src/phylotypy/classify_bootstraps/classify_bootstraps.pyx"],
        include_dirs=[np.get_include()],
    ),
        # New cond_prob_cython Cython extension
    Extension(
        "phylotypy.cond_prob_c.cond_prob_cython",
        ["src/phylotypy/cond_prob_c/cond_prob_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3'],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

setup(
    name="phylotypy",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"},
        annotate=True),
)
