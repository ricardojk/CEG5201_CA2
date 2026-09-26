"""Correctness check: compare multiply_seq and multiply_mp with NumPy on the small pairs of G0.

np.matmul is used here only to produce the reference answer. The benchmarked code in
algorithms/ must not use it.

Usage:
    python verify.py --algo strassen
    python verify.py --algo all --max-size 512 --procs 2 6
"""
import config  # first import: pins BLAS threads before NumPy loads

import argparse
import functools
import sys

import numpy as np

from algorithms import load as load_algorithm
from common import bench, data


def check(name, label, multiply, pairs):
    """Print one verdict line; return True when every product equals NumPy's."""
    wrong = [len(A) for A, B in pairs
             if not np.array_equal(np.asarray(multiply(A, B)),
                                   np.matmul(np.asarray(A), np.asarray(B)))]
    print(f"  {name:<22}{label:<9}" + (f"WRONG at sizes {wrong}" if wrong else "OK"))
    return not wrong


def main():
    parser = argparse.ArgumentParser(description="Check the algorithms against NumPy.")
    parser.add_argument("--algo", required=True, choices=[*config.ALGORITHMS.values(), "all"])
    parser.add_argument("--max-size", type=int, default=config.VERIFY_MAX_SIZE,
                        help="largest G0 pair to check (default: config.VERIFY_MAX_SIZE)")
    parser.add_argument("--procs", type=int, nargs="+", default=config.PROCESS_COUNTS,
                        help="process counts to check (default: config.PROCESS_COUNTS)")
    args = parser.parse_args()

    names = list(config.ALGORITHMS.values()) if args.algo == "all" else [args.algo]
    pairs = data.load_group(0, args.max_size)
    print(f"Checking G0 pairs of size {[len(A) for A, _ in pairs]}")

    passed = True
    for name in names:
        algo = load_algorithm(name)
        passed &= check(name, "seq", algo.multiply_seq, pairs)
        for p in args.procs:
            pool = bench.create_pool(p, algo.__name__)
            multiply = functools.partial(algo.multiply_mp, pool=pool, p=p)
            passed &= check(name, f"mp p={p}", multiply, pairs)
            pool.close()
            pool.join()

    print("All checks passed." if passed else "Some checks FAILED.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
