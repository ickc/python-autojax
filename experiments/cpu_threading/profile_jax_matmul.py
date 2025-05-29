from __future__ import annotations

import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


@jax.jit
def f(
    A: np.ndarray[tuple[int, int], np.float64],
    B: np.ndarray[tuple[int, int], np.float64],
) -> np.ndarray[tuple[int, int], np.float64]:
    return A @ B


@partial(jax.jit, static_argnames=("M", "N"))
def gen_A(M: int, N: int) -> np.ndarray[tuple[int, int], np.float64]:
    return jnp.reciprocal(jnp.arange(1, M * N + 1, dtype=jnp.float64).reshape(M, N))


@partial(jax.jit, static_argnames=("N", "K"))
def gen_B(N: int, K: int) -> np.ndarray[tuple[int, int], np.float64]:
    return jnp.arange(1, N * K + 1, dtype=jnp.float64).reshape(N, K)


def run(
    M: int = 64,
    N: int = 64,
    K: int = 64,
):
    A = gen_A(M, N)
    B = gen_B(N, K)
    C = f(A, B)
    get_ipython().run_line_magic("timeit", "f(A, B).block_until_ready()")


def main() -> None:
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} M N K", file=sys.stderr)
        print(sys.argv)
        sys.exit(1)
    run(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))


if __name__ == "__main__":
    main()
