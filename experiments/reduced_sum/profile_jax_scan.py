from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


@jax.jit
def f(
    m: np.ndarray[tuple[int], np.float64],
    k1: np.ndarray[tuple[int], np.float64],
    k2: np.ndarray[tuple[int], np.float64],
) -> np.ndarray[tuple[int], np.float64]:
    """
    memory used: M
    FLOPS: 4MK
    """

    def f_k(k1: float, k2: float) -> np.ndarray[tuple[int], np.float64]:
        return jnp.square(m * k1) * k2

    def f_scan(
        sum_: np.ndarray[tuple[int], np.float64],
        args: tuple[float, float],
    ) -> tuple[np.ndarray[tuple[int], np.float64], None]:
        k1, k2 = args
        return sum_ + f_k(k1, k2), None

    res, _ = jax.lax.scan(f_scan, jnp.zeros_like(m), (k1, k2))
    return res


def ref(M, K):
    return np.square(jnp.arange(1, M + 1)) * K


def run(
    M: int = 47,
    K: int = 101,
):
    m = jnp.arange(1, M + 1, dtype=np.float64)
    k1 = jnp.arange(1, K + 1, dtype=np.float64)
    k2 = jnp.square(jnp.reciprocal(k1))
    np.testing.assert_allclose(f(m, k1, k2), ref(M, K))

    get_ipython().run_line_magic("timeit", "f(m, k1, k2)")


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} M K", file=sys.stderr)
        sys.exit(1)
    run(int(sys.argv[1]), int(sys.argv[2]))


if __name__ == "__main__":
    main()
