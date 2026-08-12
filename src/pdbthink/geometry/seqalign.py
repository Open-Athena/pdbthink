"""Global sequence alignment used for two-state residue mapping (A.27.1).

A dependency-free Needleman-Wunsch with affine gaps. V1 only ever aligns two
observations of the same protein (>= 90% identity is enforced by A.27), so an
identity substitution model is sufficient and keeps the mapping reproducible
without pinning an external substitution matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

MATCH = 2.0
MISMATCH = -1.0
GAP_OPEN = -10.0
GAP_EXTEND = -0.5
ALGORITHM = "identity_nw_affine_v1"


@dataclass
class Alignment:
    pairs: list[tuple[int | None, int | None]]
    identity: float
    score: float
    algorithm: str = ALGORITHM

    @property
    def matched(self) -> list[tuple[int, int]]:
        return [(i, j) for i, j in self.pairs if i is not None and j is not None]


def align(seq_a: str, seq_b: str) -> Alignment:
    """Align two one-letter sequences and return index pairs."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return Alignment(pairs=[], identity=0.0, score=0.0)

    neg = float("-inf")
    # M: ends aligned, X: gap in b (consume a), Y: gap in a (consume b)
    mm = [[neg] * (m + 1) for _ in range(n + 1)]
    xx = [[neg] * (m + 1) for _ in range(n + 1)]
    yy = [[neg] * (m + 1) for _ in range(n + 1)]
    ptr_m = [[0] * (m + 1) for _ in range(n + 1)]
    ptr_x = [[0] * (m + 1) for _ in range(n + 1)]
    ptr_y = [[0] * (m + 1) for _ in range(n + 1)]

    mm[0][0] = 0.0
    for i in range(1, n + 1):
        xx[i][0] = GAP_OPEN + GAP_EXTEND * (i - 1)
        ptr_x[i][0] = 1 if i > 1 else 0
    for j in range(1, m + 1):
        yy[0][j] = GAP_OPEN + GAP_EXTEND * (j - 1)
        ptr_y[0][j] = 2 if j > 1 else 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = MATCH if seq_a[i - 1] == seq_b[j - 1] else MISMATCH
            best_prev = max(
                (mm[i - 1][j - 1], 0), (xx[i - 1][j - 1], 1), (yy[i - 1][j - 1], 2)
            )
            mm[i][j] = best_prev[0] + sub
            ptr_m[i][j] = best_prev[1]

            open_x = mm[i - 1][j] + GAP_OPEN
            extend_x = xx[i - 1][j] + GAP_EXTEND
            if open_x >= extend_x:
                xx[i][j], ptr_x[i][j] = open_x, 0
            else:
                xx[i][j], ptr_x[i][j] = extend_x, 1

            open_y = mm[i][j - 1] + GAP_OPEN
            extend_y = yy[i][j - 1] + GAP_EXTEND
            if open_y >= extend_y:
                yy[i][j], ptr_y[i][j] = open_y, 0
            else:
                yy[i][j], ptr_y[i][j] = extend_y, 2

    end = max((mm[n][m], 0), (xx[n][m], 1), (yy[n][m], 2))
    score, state = end
    i, j = n, m
    pairs: list[tuple[int | None, int | None]] = []
    while i > 0 or j > 0:
        if state == 0 and i > 0 and j > 0:
            pairs.append((i - 1, j - 1))
            state = ptr_m[i][j]
            i, j = i - 1, j - 1
        elif state == 1 and i > 0:
            pairs.append((i - 1, None))
            state = ptr_x[i][j]
            i -= 1
        elif state == 2 and j > 0:
            pairs.append((None, j - 1))
            state = ptr_y[i][j]
            j -= 1
        elif i > 0:                      # only gaps in b remain
            pairs.append((i - 1, None))
            i -= 1
            state = 1
        else:                            # only gaps in a remain
            pairs.append((None, j - 1))
            j -= 1
            state = 2
    pairs.reverse()

    matched = [(a, b) for a, b in pairs if a is not None and b is not None]
    identical = sum(1 for a, b in matched if seq_a[a] == seq_b[b])
    identity = identical / max(1, min(n, m))
    return Alignment(pairs=pairs, identity=identity, score=float(score))
