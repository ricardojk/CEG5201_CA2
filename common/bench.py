"""[B-n] / [C-n] Timing harness shared by all four algorithms.

Timing protocol (identical for every member, see [O] in the report):
  * a group is loaded from disk and converted to MATRIX_FORMAT before the clock starts;
  * the clock starts once per group and is read after every pair, which gives the
    cumulative elapsed times T_0..T_7 of Tables B-n.1 / C-n.1 (the group total is T_7);
  * multiprocessing: a fresh Pool of p workers is created for every group; whether its
    creation is inside the timed region is set by config.TIME_POOL_CREATION;
  * nothing but the multiplications runs inside the timed region; the products are
    discarded there and checked separately by verify.py.
"""
import functools
import gc
import importlib
import json
import multiprocessing as mp
import time
from datetime import datetime

import config
from algorithms import load as load_algorithm
from common import data, sysinfo

# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------
_barrier = None     # set in every worker process by _init_worker


def _init_worker(barrier, module_name):
    """Runs once in each new worker: keep the start-up barrier, import the algorithm."""
    global _barrier
    _barrier = barrier
    importlib.import_module(module_name)


def _wait_for_all_workers(_):
    """Blocks until p workers hold one of these tasks at the same time."""
    _barrier.wait()


def create_pool(p, module_name):
    """Start a Pool of p worker processes and return once every worker is running.

    Pool() returns while its workers are still launching Python and importing modules.
    Handing out p tasks that meet at a p-party barrier forces every worker to be up,
    so this start-up cost never leaks into the first timed multiplication.
    """
    context = mp.get_context(config.START_METHOD)
    barrier = context.Barrier(p)
    pool = context.Pool(p, initializer=_init_worker, initargs=(barrier, module_name))
    pool.map(_wait_for_all_workers, range(p), chunksize=1)
    return pool


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_pairs(multiply, pairs, start):
    """Multiply every (A, B) in order; return cumulative times T_i measured from `start`."""
    cumulative = []
    for A, B in pairs:
        multiply(A, B)
        cumulative.append(time.perf_counter() - start)
    return cumulative


def time_group(algo, p, pairs):
    """Time one group. p=None runs multiply_seq; otherwise multiply_mp with p workers.

    Returns (cumulative times T_0..T_n, pool start-up time in seconds or None).
    """
    gc.collect()
    if p is None:
        return time_pairs(algo.multiply_seq, pairs, time.perf_counter()), None

    before_pool = time.perf_counter()
    pool = create_pool(p, algo.__name__)
    pool_ready = time.perf_counter()
    start = before_pool if config.TIME_POOL_CREATION else pool_ready
    multiply = functools.partial(algo.multiply_mp, pool=pool, p=p)
    cumulative = time_pairs(multiply, pairs, start)
    pool.close()
    pool.join()
    return cumulative, pool_ready - before_pool


def run(name, p, groups, max_size, out_dir):
    """Benchmark algorithm `name` on `groups` and save out_dir/seq.json or mp_p<p>.json.

    The JSON file is rewritten after every group, so an interrupted run keeps the
    groups that finished.
    """
    algo = load_algorithm(name)
    label = "seq" if p is None else f"mp_p{p}"
    record = {
        "algorithm": name,
        "mode": "seq" if p is None else "mp",
        "p": p,
        "started": datetime.now().isoformat(timespec="seconds"),
        "machine": sysinfo.machine(),
        "protocol": sysinfo.protocol(),
        "data_fingerprint": data.fingerprint(),
        "sizes": [m for m in config.MATRIX_SIZES if m <= max_size],
        "groups": [],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{label}.json"

    for g in groups:
        pairs = data.load_group(g, max_size)
        print(f"  {name} {label:<6} G{g} ...", end="", flush=True)
        cumulative, pool_startup = time_group(algo, p, pairs)
        record["groups"].append({"group": g, "cumulative_s": cumulative,
                                 "pool_startup_s": pool_startup})
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f" {cumulative[-1]:10.4f} s")
    print(f"  -> {path.relative_to(config.ROOT)}")
