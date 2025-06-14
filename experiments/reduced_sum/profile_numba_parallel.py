from __future__ import annotations

import sys

import numpy as np
from numba import jit


@jit("f8[::1](f8[::1], f8[::1], f8[::1])", nopython=True, nogil=True, parallel=True)
def f(
    m: np.ndarray[tuple[int], np.float64],
    k1: np.ndarray[tuple[int], np.float64],
    k2: np.ndarray[tuple[int], np.float64],
) -> np.ndarray[tuple[int], np.float64]:
    """
    memory used: MK
    FLOPS: 4MK
    """
    A_mk = np.square(np.outer(m, k1))
    return A_mk @ k2


def ref(M, K):
    return np.square(np.arange(1, M + 1)) * K


def run(
    M: int = 47,
    K: int = 101,
):
    m = np.arange(1, M + 1, dtype=np.float64)
    k1 = np.arange(1, K + 1, dtype=np.float64)
    k2 = np.square(np.reciprocal(k1))
    np.testing.assert_allclose(f(m, k1, k2), ref(M, K))

    get_ipython().run_line_magic("timeit", "f(m, k1, k2)")


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} M K", file=sys.stderr)
        sys.exit(1)
    run(int(sys.argv[1]), int(sys.argv[2]))


if __name__ == "__main__":
    main()
