"""One-click full test of Cannon's algorithm: verify -> benchmark -> tables and figures.

Steps (each one must succeed before the next starts):
  1. verify.py        correctness of multiply_seq / multiply_mp against NumPy
  2. run_benchmark.py sequential run, then p = 1..6, on G0..G9  (about 75 min)
  3. analyze.py       tables and figures in results/cannon/

On macOS the computer is kept awake (caffeinate) until this script ends.

Usage:
    python run_cannon.py            # full test
    python run_cannon.py --quick    # smoke test: G0-G1, sizes up to 256 -> results_quick/
"""
import config  # first import: pins BLAS threads before NumPy loads

import argparse
import os
import platform
import subprocess
import sys
import time


def step(title, *args):
    """Run one project script with this Python; stop everything if it fails."""
    print(f"\n=== {title}", flush=True)
    start = time.perf_counter()
    subprocess.run([sys.executable, *args], cwd=config.ROOT, check=True)
    print(f"=== {title}: done in {time.perf_counter() - start:.0f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Full test of Cannon's algorithm.")
    parser.add_argument("--quick", action="store_true", help="small smoke test instead")
    args = parser.parse_args()
    quick = ["--quick"] if args.quick else []

    if platform.system() == "Darwin":
        # Prevent idle sleep for as long as this process is alive.
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])

    start = time.perf_counter()
    step("1/3 verify", "verify.py", "--algo", "cannon")
    step("2/3 benchmark", "run_benchmark.py", "--algo", "cannon", *quick)
    step("3/3 tables and figures", "analyze.py", "--algo", "cannon", *quick)
    results = config.QUICK_RESULTS_DIR if args.quick else config.RESULTS_DIR
    print(f"\nAll done in {(time.perf_counter() - start) / 60:.1f} min. "
          f"Results: {(results / 'cannon').relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
