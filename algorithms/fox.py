"""[B-4] / [C-4]  Fox's algorithm  (n = 4 in Table II).

Owner: <your name>

Reference: G. C. Fox, S. W. Otto, and A. J. G. Hey, "Matrix algorithms on a hypercube
I: Matrix multiplication," Parallel Comput., vol. 4, no. 1, pp. 17-31, 1987.

Appendix A (q = 3 example) is correct. In stage s it broadcasts one block of Y down
each block column and rolls X left by one block per stage - the transpose of the
textbook form (broadcast a block of X along each block row, roll Y up); both give
X x Y. All sizes are powers of two, so a power-of-two q divides every matrix evenly.

Tuning parameter: config.FOX_GRID (q).
Interface and rules: README.md, section "Implementing your algorithm".
"""


def multiply_seq(A, B):
    """[B-4] Sequential Fox. Return the product A x B."""
    raise NotImplementedError("TODO [B-4]: sequential Fox")


def multiply_mp(A, B, pool, p):
    """[C-4] Fox accelerated with `pool`, a ready multiprocessing.Pool of p workers."""
    raise NotImplementedError("TODO [C-4]: multiprocessing Fox")
