"""[C-n.1] Locate the bottleneck of a sequential implementation with cProfile.

Runs multiply_seq once on the G0 pair of the chosen size under cProfile and lists the
functions that take the most time, as a share of the total. "Own" time is spent in the
function itself; "total" time includes everything it calls. The same table is saved to
results/<algorithm>/profile/seq_<size>.txt for the report.

Usage:
    python profile_seq.py --algo cannon --size 512
    python profile_seq.py --algo strassen --size 1024 --top 15
"""
import config  # first import: pins BLAS threads before NumPy loads

import argparse
import cProfile
import pstats
import time
from pathlib import Path

from algorithms import load as load_algorithm
from common import data


def label(key):
    """Readable name of a pstats entry: "module.function" or the built-in's name."""
    filename, _, name = key
    return name if filename == "~" else f"{Path(filename).stem}.{name}"


def main():
    parser = argparse.ArgumentParser(description="Profile the sequential version of an algorithm.")
    parser.add_argument("--algo", required=True, choices=list(config.ALGORITHMS.values()))
    parser.add_argument("--size", type=int, default=512, choices=config.MATRIX_SIZES)
    parser.add_argument("--top", type=int, default=12, help="number of functions to list")
    args = parser.parse_args()

    algo = load_algorithm(args.algo)
    A, B = data.load_group(0, args.size)[-1]

    start = time.perf_counter()
    algo.multiply_seq(A, B)
    plain = time.perf_counter() - start

    profiler = cProfile.Profile()
    profiler.runcall(algo.multiply_seq, A, B)
    stats = pstats.Stats(profiler).stats     # key -> (primitive calls, calls, own, total, callers)
    total = sum(own for _, _, own, _, _ in stats.values())

    lines = [
        f"cProfile of {args.algo}.multiply_seq on the {args.size} x {args.size} pair of G0",
        f"run time without profiler: {plain:.3f} s | with profiler: {total:.3f} s",
        "",
        f"{'function':<52}{'calls':>10}{'own (s)':>10}{'own %':>8}{'total (s)':>11}{'total %':>9}",
    ]
    ranked = sorted(stats.items(), key=lambda item: item[1][2], reverse=True)[:args.top]
    for key, (_, calls, own, cumulative, _) in ranked:
        lines.append(f"{label(key):<52}{calls:>10}{own:>10.3f}{own / total:>8.1%}"
                     f"{cumulative:>11.3f}{cumulative / total:>9.1%}")
    report = "\n".join(lines)
    print(report)

    out_dir = config.RESULTS_DIR / args.algo / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"seq_{args.size}.txt"
    path.write_text(report + "\n", encoding="utf-8")
    print(f"\n-> {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
