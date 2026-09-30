# CEG5201 CA2: team workspace

One shared framework for four algorithms. Data generation, timing, tables and figures
are common code, so every member's numbers are measured the same way. The only code
each person writes is their own file in `algorithms/`.

Set-up and run commands for Windows and macOS are in [readme.txt](readme.txt) (this is
also the readme we submit). This file covers how we work on the code together.

## 1. First steps (everyone)

1. Install exactly **Python 3.14.7** (see requirements.txt; the interpreter version
   changes the measured times, e.g. a run on 3.13 was clearly slower per operation),
   then follow README.md section 2 to create the environment and run
   `pip install -r requirements.txt`. Check with `python --version`.
2. `python show_setup.py` shows which algorithm is yours.
3. `python generate_data.py` prints a **data fingerprint**. Post it in the group chat:
   everyone's must match (same seed + same NumPy version = identical G0..G9).

## 2. Project layout

```
config.py              <- ALL shared settings (the only settings file)
show_setup.py          assignment + machine + protocol
generate_data.py       [A]  data/G0.npz .. G9.npz
verify.py              correctness check against NumPy
profile_seq.py         [C-n.1] cProfile of your multiply_seq: where the time goes
run_benchmark.py       [B] [C]  timing runs -> results/<algo>/raw/*.json
analyze.py             [D] [E]  tables (CSV) + figures (PNG)
make_submission.py     builds the Canvas zip with the required names
common/                shared harness: data.py, bench.py, sysinfo.py, analysis.py
algorithms/            one file per algorithm  <- your code goes here
```

## 3. Settings: `config.py` only

- **Team section** (`GROUP_ID`, `MEMBERS`): filled in once, then left alone.
- **Protocol** (`MATRIX_FORMAT`, `TIME_POOL_CREATION`, `PROCESS_COUNTS`, `START_METHOD`,
  `SEED`, ...): must be identical for everyone. Change these only after the team agrees.
- **Algorithm parameters** (`STRASSEN_LEAF_SIZE`, `CW_LEAF_SIZE`, `CANNON_GRID`,
  `FOX_GRID`): each owner edits only their own line. Read yours with `import config`
  in your algorithm file. Add new knobs here too, not as magic numbers inside your
  algorithm file.

Never hard-code paths, sizes or process counts elsewhere; read them from `config`.

## 4. Implementing your algorithm

Fill in the two functions in `algorithms/<your algorithm>.py`:

```python
def multiply_seq(A, B):          # [B-n] sequential version -> returns C = A x B
def multiply_mp(A, B, pool, p):  # [C-n] same product using `pool`, a ready Pool of p workers
```

- `A`, `B`: square M x M matrices (M = 16 ... 2048, always a power of two) in
  `config.MATRIX_FORMAT` (int64 NumPy arrays by default).
- Return the exact product, as a NumPy array or a list of lists. `verify.py` compares
  it with NumPy.

Rules (from the CA2 statement, plus what the harness needs):

1. **No library matrix product** in either function: no `np.matmul`, `np.dot`, `@`,
   `np.einsum`, `np.tensordot`, `np.inner`, `np.linalg`. Whether element-wise NumPy
   (`+`, `-`, `*`, `.sum()`) counts as "implementing it yourself" is a **team decision
   to confirm at the 13 Oct consultation**. Until then, keep the core loops your own.
2. Parallel work goes through the given `pool` (`map`, `starmap`, `apply_async`, ...).
   Do not create your own Pool, and do not use threads.
3. Functions you send to the pool must be **top-level functions of your module**: no
   lambdas, no nested functions. Windows and macOS use `spawn`, which sends
   functions to workers by name.
4. Send the matrices to the workers **inside `multiply_mp`** (as task arguments,
   `multiprocessing.shared_memory`, ...), so the transfer cost is timed like everyone
   else's. Do not pre-load data in initializers or globals.
5. Do not print inside the two functions (printing adds to the measured time).
6. Test through `verify.py`. Algorithm files are modules, not scripts. Any new script
   must keep its code under `if __name__ == "__main__":`, which spawn requires.

If you adapt code from anywhere, cite it in your file and in your [F-n] page.

**Finding your bottleneck ([C-n.1]).** The statement asks you to identify the
bottleneck of your sequential version before parallelising it. Run
`python profile_seq.py --algo <yours> --size 512` (and `--size 1024`): it lists the
functions with the largest share of the run time and saves the table to
`results/<algo>/profile/` for the report. Parallelise the part that dominates, then
check that the overheads you add (task dispatch, data transfer, uneven task sizes)
stay small.

## 5. Day-to-day workflow

```
python verify.py --algo cannon                  # correctness, G0 pairs up to 256
python run_benchmark.py --algo cannon --quick   # under a minute -> results_quick/
python analyze.py --quick                       # tables/figures of the smoke test
```

The **final benchmark** runs on the one machine the team chose (statement [O]), once
every file passes `verify.py`:

```
python run_benchmark.py --algo all              # several hours
python analyze.py
```

On that machine, plug in the power, close other applications and disable sleep.
On macOS: `caffeinate -i python run_benchmark.py --algo all`. On Windows: Settings ->
Power -> Sleep: Never. Raw results are saved after every group. A single part can be
re-run, e.g. `--mode mp --procs 5 6`.

## 6. Timing protocol (fixed; describe it in the report under [O])

- Loading and format conversion happen before the clock starts.
- One clock per group, read after every pair, gives the cumulative times T_0..T_7.
- Multiprocessing: a fresh Pool per group, fully started before timing
  (`TIME_POOL_CREATION = False`); its start-up time is recorded separately in
  `C-n_pool_startup.csv`.
- Order: G0..G9, small to large pairs, sequential first, then p = 1..6.

## 7. What `analyze.py` gives you for the report

| Report item | File |
|---|---|
| Table B-n.1 / B-n.2 | `results/<algo>/tables/B-n.1.csv`, `B-n.2.csv` |
| Table C-n.1 / C-n.2 | `results/<algo>/tables/C-n.1.csv`, `C-n.2.csv` |
| D-n.1 (i), (ii) | `results/<algo>/figures/D-n.1_speedup_vs_p.png`, `D-n.1_speedup_vs_k.png`, table `D-n.1.csv` |
| D-n.2 | `results/<algo>/tables/D-n.2.csv`, figure `D-n.2_speedup_vs_p.png` |
| [E] comparison | `results/comparison/E_*.png`, `E_summary.csv` |
| Bottleneck discussion | `results/<algo>/profile/seq_<size>.txt` (from `profile_seq.py`), `C-n_pool_startup.csv` (process start-up cost per p) |

## 8. Appendix A errata (checked numerically)

- The text says Z = Y·X, but every block equation computes **Z = X·Y**.
- Strassen: correct, apart from the Z label typo in the first line (Z21 printed where
  Z12 is meant).
- Coppersmith-Winograd: this is really the Strassen-Winograd variant. Correct lines:
  `S4 = X12 - S2`, `Z21 = M1 + M6 + M7 - M4`, and the last line is `Z22` (printed Z12).
- Cannon and Fox: correct as printed.

## 9. Git and submission

- `.gitignore` excludes `data/` (regenerate it), `results_quick/`, `dist/` and `.venv/`.
  Commit `results/` after the final run so everyone can write [E].
- To submit, export the report to PDF, then run
  `python make_submission.py path/to/report.pdf`. This writes
  `dist/CA2_<GroupID>_<matric numbers>.zip` with the report renamed
  `CA2_Report_<GroupID>_<matric numbers>.pdf`, all `.py` files, `readme.txt` and
  `requirements.txt`.
