"""[A] Input matrices: generation, storage and loading of the groups G0..G9.

Each group G_g is stored as data/G<g>.npz holding the arrays A0, B0, ..., A7, B7, where
pair i has size MATRIX_SIZES[i]. Elements are integers in [VALUE_MIN, VALUE_MAX] and are
stored as uint8 (0..255 fits exactly); they are widened to int64 when loaded, so products
and sums never overflow.
"""
import hashlib

import numpy as np

import config


def group_path(g):
    """Location of the .npz file holding group G_g."""
    return config.DATA_DIR / f"G{g}.npz"


def generate_group(g):
    """Return the pairs (A_i, B_i) of group G_g as uint8 arrays.

    Every group has its own generator seeded with (SEED, g), so a group is reproducible
    on its own and identical on every machine that uses the pinned NumPy version.
    """
    rng = np.random.default_rng([config.SEED, g])
    return [
        tuple(rng.integers(config.VALUE_MIN, config.VALUE_MAX, size=(m, m),
                           dtype=np.uint8, endpoint=True) for _ in range(2))
        for m in config.MATRIX_SIZES
    ]


def save_group(g, pairs):
    """Write the pairs of group G_g to data/G<g>.npz."""
    arrays = {}
    for i, (A, B) in enumerate(pairs):
        arrays[f"A{i}"] = A
        arrays[f"B{i}"] = B
    np.savez(group_path(g), **arrays)


def load_group(g, max_size=config.MATRIX_SIZES[-1]):
    """Load the pairs of G_g with size <= max_size, in the format the algorithms receive."""
    with np.load(group_path(g)) as archive:
        return [(_convert(archive[f"A{i}"]), _convert(archive[f"B{i}"]))
                for i, m in enumerate(config.MATRIX_SIZES) if m <= max_size]


def _convert(matrix):
    """uint8 on disk -> int64 array, or list of lists when MATRIX_FORMAT is "list"."""
    matrix = matrix.astype(np.int64)
    return matrix.tolist() if config.MATRIX_FORMAT == "list" else matrix


def fingerprint():
    """Short SHA-256 of all generated matrices; equal fingerprints mean identical data."""
    digest = hashlib.sha256()
    for g in range(config.NUM_GROUPS):
        with np.load(group_path(g)) as archive:
            for i in range(len(config.MATRIX_SIZES)):
                digest.update(archive[f"A{i}"].tobytes())
                digest.update(archive[f"B{i}"].tobytes())
    return digest.hexdigest()[:16]
