"""[B-3] / [C-3]  Cannon's algorithm  (n = 3 in Table II).

Owner: ZHANG FANGYU (Member ID 2)

Reference: L. E. Cannon, "A Cellular Computer to Implement the Kalman Filter
Algorithm," PhD thesis, Montana State University, 1969.

Appendix A (q = 3 example) is correct: skew once (block row i of X shifted left by i,
block column j of Y shifted up by j), then q stages of "multiply matching blocks,
accumulate, shift X left by one and Y up by one". All sizes are powers of two, so
a power-of-two q divides every matrix evenly.

Tuning parameter: config.CANNON_GRID (q).
Interface and rules: README.md, section "Implementing your algorithm".
"""


def multiply_seq(A, B):
    """[B-3] Sequential Cannon. Return the product A x B."""
    raise NotImplementedError("TODO [B-3]: sequential Cannon")


def multiply_mp(A, B, pool, p):
    """[C-3] Cannon accelerated with `pool`, a ready multiprocessing.Pool of p workers."""
    raise NotImplementedError("TODO [C-3]: multiprocessing Cannon")
