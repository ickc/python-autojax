#!/usr/bin/env bash

set -e

M=8192

: > profile_${M}.log

# baseline where no env vars are set
# 0.5 GiB, 1 GiB, 2 GiB
for K in 8192 16384 32768; do
    for case in \
        numpy \
        numba \
        numba_parallel \
        numba_fori \
        numba_fori_parallel \
        jax \
        jax_vmap \
        jax_fori \
        jax_scan; do
        echo "=========== ${case} M=${M}, K=${K}, NUM_THREADS=-1 ===========" >> profile_${M}.log
        command time -v ipython profile_${case}.py ${M} ${K} &>> profile_${M}.log
    done
done

# non-JAX case where env var is clear to set
for NUM_THREADS in 1 2 4 8 10 14; do
    export MKL_NUM_THREADS=${NUM_THREADS}
    export MKL_DOMAIN_NUM_THREADS="MKL_BLAS=${NUM_THREADS}"
    export MKL_DYNAMIC=FALSE

    export OMP_NUM_THREADS=${NUM_THREADS}
    export OMP_PLACES=threads
    export OMP_PROC_BIND=spread
    export OMP_DYNAMIC=FALSE

    export NUMEXPR_NUM_THREADS=${NUM_THREADS}

    export OPENBLAS_NUM_THREADS=${NUM_THREADS}

    export NUMBA_NUM_THREADS=${NUM_THREADS}
    # 0.5 GiB, 1 GiB, 2 GiB
    for K in 8192 16384 32768; do
        for case in \
            numpy \
            numba \
            numba_parallel \
            numba_fori \
            numba_fori_parallel; do
            echo "=========== ${case} M=${M}, K=${K}, NUM_THREADS=${NUM_THREADS} ===========" >> profile_${M}.log
            command time -v ipython profile_${case}.py ${M} ${K} &>> profile_${M}.log
        done
    done
done

# JAX cases where we tests multiple ways to set env var
for NUM_THREADS in 1 2 4 8 10 14; do
    export MKL_NUM_THREADS=${NUM_THREADS}
    export MKL_DOMAIN_NUM_THREADS="MKL_BLAS=${NUM_THREADS}"
    export MKL_DYNAMIC=FALSE

    export OMP_NUM_THREADS=${NUM_THREADS}
    export OMP_PLACES=threads
    export OMP_PROC_BIND=spread
    export OMP_DYNAMIC=FALSE

    export NUMEXPR_NUM_THREADS=${NUM_THREADS}

    export OPENBLAS_NUM_THREADS=${NUM_THREADS}

    export NUMBA_NUM_THREADS=${NUM_THREADS}

    export NPROC=${NUM_THREADS}
    export JAX_NUM_CPU_DEVICES=1
    export TF_NUM_INTEROP_THREADS=1
    export TF_NUM_INTRAOP_THREADS=${NUM_THREADS}
    # 0.5 GiB, 1 GiB, 2 GiB
    for K in 8192 16384 32768; do
        for case in \
            jax \
            jax_vmap \
            jax_fori \
            jax_scan; do
            echo "=========== ${case} M=${M}, K=${K}, NUM_THREADS=${NUM_THREADS} ===========" >> profile_${M}.log
            command time -v ipython profile_${case}.py ${M} ${K} &>> profile_${M}.log
        done
    done
done

# JAX cases where we tests multiple ways to set env var
for NUM_THREADS in 1 2 4 8 10 14; do
    export MKL_NUM_THREADS=${NUM_THREADS}
    export MKL_DOMAIN_NUM_THREADS="MKL_BLAS=${NUM_THREADS}"
    export MKL_DYNAMIC=FALSE

    export OMP_NUM_THREADS=${NUM_THREADS}
    export OMP_PLACES=threads
    export OMP_PROC_BIND=spread
    export OMP_DYNAMIC=FALSE

    export NUMEXPR_NUM_THREADS=${NUM_THREADS}

    export OPENBLAS_NUM_THREADS=${NUM_THREADS}

    export NUMBA_NUM_THREADS=${NUM_THREADS}

    export NPROC=${NUM_THREADS}
    export JAX_NUM_CPU_DEVICES=${NUM_THREADS}
    export TF_NUM_INTEROP_THREADS=1
    export TF_NUM_INTRAOP_THREADS=${NUM_THREADS}
    # 0.5 GiB, 1 GiB, 2 GiB
    for K in 8192 16384 32768; do
        for case in \
            jax \
            jax_vmap \
            jax_fori \
            jax_scan; do
            echo "=========== ${case}_all_cpu_devices M=${M}, K=${K}, NUM_THREADS=${NUM_THREADS} ===========" >> profile_${M}.log
            command time -v ipython profile_${case}.py ${M} ${K} &>> profile_${M}.log
        done
    done
done
