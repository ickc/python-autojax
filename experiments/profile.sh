#!/usr/bin/env bash

M=8192

# 0.5 GiB, 1 GiB, 2 GiB
for K in 8192 16384 32768; do
    echo "=========== Numpy    M=${M}, K=${K} ==========="
    command time -v ipython profile_numpy.py ${M} ${K}
    echo "=========== JAX      M=${M}, K=${K} ==========="
    command time -v ipython profile_jax.py ${M} ${K}
    echo "=========== JAX vmap M=${M}, K=${K} ==========="
    command time -v ipython profile_jax_vmap.py ${M} ${K}
    echo "=========== JAX scan M=${M}, K=${K} ==========="
    command time -v ipython profile_jax_scan.py ${M} ${K}
    echo "=========== JAX fori M=${M}, K=${K} ==========="
    command time -v ipython profile_jax_fori.py ${M} ${K}
done
