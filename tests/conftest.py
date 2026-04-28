import multiprocessing
import platform
import pytest

def pytest_configure(config):
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn", force=True)
