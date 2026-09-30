"""[B-3] / [C-3]  Cannon's algorithm  (n = 3 in Table II).

Owner: ZHANG FANGYU (Member ID 2)

Reference: L. E. Cannon, "A Cellular Computer to Implement the Kalman Filter
Algorithm," PhD thesis, Montana State University, 1969.

How the algorithm works
-----------------------
A, B and C are cut into a q x q grid of blocks (q = config.CANNON_GRID). Every grid
position (i, j) holds one block of A and one block of B at a time:

1. Initial alignment (skew): block row i of A is rotated left by i positions and block
   column j of B is rotated up by j positions. Position (i, j) now holds A[i][k] and
   B[k][j] with k = (i + j) mod q, a pair that can be multiplied.
2. q stages: every position multiplies the pair it holds and adds the product to its
   block C[i][j]; then all A blocks move one position left and all B blocks one position
   up (circularly). At stage s position (i, j) holds the pair with k = (i + j + s) mod q,
   so over the q stages it meets every k exactly once: C[i][j] = sum_k A[i][k] B[k][j].

At every stage each block of A and of B sits at exactly one position, so no block is
ever needed by two positions at once (Fox's algorithm broadcasts blocks instead).

Implementation notes
--------------------
* All arithmetic is pure Python: blocks are lists of Python ints and every entry of a
  block product is one dot product, sum(map(mul, row_of_A, column_of_B)). NumPy is only
  used to store the input and cut it into blocks (no NumPy arithmetic).
* Block boundaries are t * M // q, so q does not have to divide M; blocks then differ
  by at most one row/column and every product stays conformable. (Appendix A's q = 3
  example is correct.)
* Sequential version: the skew and the shifts are carried out literally, by rotating
  the grids of block references (no data is copied).

Bottleneck of the sequential version (python profile_seq.py --algo cannon)
-----------------------------------------------------------------------------
The q^2 block multiply-adds of the q stages (_multiply_add) take 99.5 % of the time
at M = 512 and 99.8 % at M = 1024; the dot products inside them alone take 92-96 %.
Skew, shifts, NumPy -> list conversion and joining are O(M^2) and together < 0.5 %.
Within one stage the q^2 block products are independent, and the q products of one
grid position only add into its own C block, so this is exactly the work to spread
over processes.

Multiprocessing version
-----------------------
* Every grid position (i, j) becomes one pool task, a "virtual processor". The main
  process sends it, in stage order, the q block pairs that the skew and the shifts
  would bring to it; the task runs its q multiply-add stages and returns C[i][j]. The
  q^2 tasks are independent, so no barrier is needed between stages.
* Measured overheads of this scheme are small: building tasks, pickling the blocks
  and joining C cost < 1 % of the run time and mostly overlap with the computation.
  The loss that remained was load imbalance: with q = 4 there are 16 equal tasks, so
  5 or 6 workers finish in uneven rounds and sit idle 10-16 % of the time. Hence
  q = CANNON_GRID = 8 (64 tasks, handed out one at a time with chunksize=1), which
  lets faster workers pick up more tasks.
* Small matrices would be cut into tiny tasks whose dispatch costs more than their
  arithmetic, so blocks are kept at least CANNON_MIN_BLOCK wide (see _grid_size).
  The sequential version uses the same grid, so both run the same algorithm.
"""
from itertools import chain
from operator import mul

import numpy as np

import config


# ---------------------------------------------------------------------------
# Block helpers
# ---------------------------------------------------------------------------
def _grid_size(n):
    """Block grid q for an n x n matrix: CANNON_GRID, but blocks at least CANNON_MIN_BLOCK wide."""
    return max(1, min(config.CANNON_GRID, n // config.CANNON_MIN_BLOCK))


def _edges(n, q):
    """Boundaries of the q block rows (= block columns) of an n x n matrix."""
    return [n * t // q for t in range(q + 1)]


def _block(M, edges, i, k):
    """Block (i, k) of the NumPy matrix M, as a view (no copy)."""
    return M[edges[i]:edges[i + 1], edges[k]:edges[k + 1]]


def _zeros(rows, cols):
    return [[0] * cols for _ in range(rows)]


def _multiply_add(C, A, Bt):
    """Return C + A x B for list blocks, where Bt is B transposed (Bt[c] = column c of B).

    Storing B by columns turns every entry of the product into one dot product of two
    Python lists: sum(map(mul, row, column)).
    """
    return [[c + sum(map(mul, a_row, b_col)) for c, b_col in zip(c_row, Bt)]
            for c_row, a_row in zip(C, A)]


def _join(grid):
    """Glue a q x q grid of list blocks back into one matrix (list of rows)."""
    return [list(chain.from_iterable(row_parts))
            for block_row in grid for row_parts in zip(*block_row)]


# ---------------------------------------------------------------------------
# [B-3] Sequential version
# ---------------------------------------------------------------------------
def multiply_seq(A, B):
    """[B-3] Sequential Cannon: skew once, then q stages of multiply-add and shift."""
    A, B = np.asarray(A), np.asarray(B)
    q = _grid_size(len(A))
    edges = _edges(len(A), q)

    # Initial alignment: position (i, j) receives A[i][(i+j)%q] and B[(i+j)%q][j].
    # B blocks are stored transposed (by columns) for the dot-product kernel.
    A_grid = [[_block(A, edges, i, (i + j) % q).tolist() for j in range(q)] for i in range(q)]
    B_grid = [[_block(B, edges, (i + j) % q, j).T.tolist() for j in range(q)] for i in range(q)]
    C_grid = [[_zeros(edges[i + 1] - edges[i], edges[j + 1] - edges[j]) for j in range(q)]
              for i in range(q)]

    for _ in range(q):
        # Every position multiplies the pair it currently holds into its C block ...
        C_grid = [[_multiply_add(C_grid[i][j], A_grid[i][j], B_grid[i][j]) for j in range(q)]
                  for i in range(q)]
        # ... then every A block moves one position left, every B block one position up.
        A_grid = [row[1:] + row[:1] for row in A_grid]
        B_grid = B_grid[1:] + B_grid[:1]

    return _join(C_grid)


# ---------------------------------------------------------------------------
# [C-3] Multiprocessing version
# ---------------------------------------------------------------------------
def _grid_position(a_blocks, b_blocks):
    """Worker task: all q stages of one grid position (i, j); returns block C[i][j].

    a_blocks[s] and b_blocks[s] are the NumPy blocks that the skew and s shifts bring
    to this position, i.e. A[i][k] and B[k][j] with k = (i + j + s) mod q.
    """
    C = _zeros(a_blocks[0].shape[0], b_blocks[0].shape[1])
    for a, b in zip(a_blocks, b_blocks):
        C = _multiply_add(C, a.tolist(), b.T.tolist())
    return C


def multiply_mp(A, B, pool, p):
    """[C-3] Cannon with its q x q grid positions run as independent tasks on `pool`."""
    A, B = np.asarray(A), np.asarray(B)
    q = _grid_size(len(A))
    edges = _edges(len(A), q)

    # One task per grid position, with its block pairs listed in stage order.
    tasks = []
    for i in range(q):
        for j in range(q):
            ks = [(i + j + s) % q for s in range(q)]
            tasks.append(([_block(A, edges, i, k) for k in ks],
                          [_block(B, edges, k, j) for k in ks]))

    blocks = pool.starmap(_grid_position, tasks, chunksize=1)   # row-major (i, j) order
    return _join([blocks[i * q:(i + 1) * q] for i in range(q)])
