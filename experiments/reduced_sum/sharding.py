# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: autojax
#     language: python
#     name: autojax
# ---

# %%
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec

# %%
jax.config.update("jax_enable_x64", True)

# %%
N = 10
jax.config.update("jax_num_cpu_devices", N)
assert jax.device_count() == N
devices = jax.devices()


# %%
@jax.jit
def f(
    m: np.ndarray[tuple[int], np.float64],
    k1: np.ndarray[tuple[int], np.float64],
    k2: np.ndarray[tuple[int], np.float64],
) -> np.ndarray[tuple[int], np.float64]:
    """
    memory used: MK
    FLOPS: 4MK
    """
    A_mk = jnp.square(jnp.outer(m, k1))
    return A_mk @ k2


# %%
M = 8000
K = 8000

# %%
m = jnp.arange(1, M + 1, dtype=np.float64)
k1 = jnp.arange(1, K + 1, dtype=np.float64)
k2 = jnp.square(jnp.reciprocal(k1))

# %%
jax.debug.visualize_array_sharding(k1)

# %%
# %%timeit
f(m, k1, k2).block_until_ready()

# %%
# mesh = jax.make_mesh((N,), ("x",))
mesh = Mesh(devices, axis_names=("i",))
mesh

# %%
sharding = jax.sharding.NamedSharding(mesh, PartitionSpec("i"))
sharding

# %%
k1_sharded = jax.device_put(k1, sharding)
k2_sharded = jax.device_put(k2, sharding)

# %%
jax.debug.visualize_array_sharding(k1_sharded)

# %%
# %%timeit
f(m, k1_sharded, k2_sharded).block_until_ready()

# %%
sharding_all = jax.sharding.NamedSharding(mesh, PartitionSpec())
sharding_all

# %%
m_sharded = jax.device_put(m, sharding_all)

# %%
jax.debug.visualize_array_sharding(m_sharded)

# %%
# %%timeit
f(m_sharded, k1_sharded, k2_sharded).block_until_ready()
