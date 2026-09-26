"""[B-2] / [C-2]  Coppersmith-Winograd algorithm  (n = 2 in Table II).

Owner: <your name>

Reference: D. Coppersmith and S. Winograd, "Matrix multiplication via arithmetic
progressions," J. Symb. Comput., vol. 9, no. 3, pp. 251-280, 1990.

Appendix A actually describes the Strassen-Winograd variant (7 products, 15 block
additions), not the asymptotic algorithm of the 1990 paper. As printed it has errors
(checked numerically against NumPy); the correct lines are:
    S4  = X12 - S2              (printed: X12 + S2)
    Z21 = M1 + M6 + M7 - M4     (printed: M7 - M5)
    Z22 = M1 + M5 + M6 + M7     (printed with the label Z12; the formula is right)

Tuning parameter: config.CW_LEAF_SIZE.
Interface and rules: README.md, section "Implementing your algorithm".
"""


def multiply_seq(A, B):
    """[B-2] Sequential Coppersmith-Winograd. Return the product A x B."""
    raise NotImplementedError("TODO [B-2]: sequential Coppersmith-Winograd")


def multiply_mp(A, B, pool, p):
    """[C-2] Coppersmith-Winograd accelerated with `pool`, a ready Pool of p workers."""
    raise NotImplementedError("TODO [C-2]: multiprocessing Coppersmith-Winograd")
