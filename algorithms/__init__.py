"""The four assigned algorithms (Table II). Each module provides multiply_seq and multiply_mp."""
import importlib


def load(name):
    """Import algorithms/<name>.py, e.g. load("strassen")."""
    return importlib.import_module(f"{__name__}.{name}")
