"""[B-1] / [C-1]  Strassen's algorithm  (n = 1 in Table II).

Owner: <your name>

Reference: V. Strassen, "Gaussian elimination is not optimal," Numer. Math., vol. 13,
no. 4, pp. 354-356, 1969.

Appendix A of the statement gives a valid 7-product variant; its only slip is the
top-right block of Z in the first line (printed Z21, meaning Z12).

Tuning parameter: config.STRASSEN_LEAF_SIZE.
Interface and rules: README.md, section "Implementing your algorithm".
"""


def multiply_seq(A, B):
    """[B-1] Sequential Strassen. Return the product A x B."""
    raise NotImplementedError("TODO [B-1]: sequential Strassen")


def multiply_mp(A, B, pool, p):
    """[C-1] Strassen accelerated with `pool`, a ready multiprocessing.Pool of p workers."""
    raise NotImplementedError("TODO [C-1]: multiprocessing Strassen")
