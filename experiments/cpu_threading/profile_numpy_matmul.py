from __future__ import annotations

import sys

import numpy as np


def f(
    A: np.ndarray[tuple[int, int], np.float64],
    B: np.ndarray[tuple[int, int], np.float64],
) -> np.ndarray[tuple[int, int], np.float64]:
    return A @ B


def gen_A(M: int, N: int) -> np.ndarray[tuple[int, int], np.float64]:
    return np.reciprocal(np.arange(1, M * N + 1, dtype=np.float64).reshape(M, N))


def gen_B(N: int, K: int) -> np.ndarray[tuple[int, int], np.float64]:
    return np.arange(1, N * K + 1, dtype=np.float64).reshape(N, K)


def run(
    M: int = 64,
    N: int = 64,
    K: int = 64,
):
    A = gen_A(M, N)
    B = gen_B(N, K)
    C = f(A, B)
    get_ipython().run_line_magic("timeit", "f(A, B)")


def main() -> None:
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} M N K", file=sys.stderr)
        print(sys.argv)
        sys.exit(1)
    run(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))


if __name__ == "__main__":
    main()
