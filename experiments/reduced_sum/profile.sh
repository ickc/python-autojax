#!/usr/bin/env bash

M=8192

for NUM_THREADS in 1 14; do
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
    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${NUM_THREADS} --xla_force_host_platform_device_count=${NUM_THREADS}"
    export TF_NUM_INTEROP_THREADS=1
    export TF_NUM_INTRAOP_THREADS=${NUM_THREADS}
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
            echo "=========== ${case} M=${M}, K=${K}, NUM_THREADS=${NUM_THREADS} ==========="
            command time -v ipython profile_${case}.py ${M} ${K}
        done
    done
done 2>&1 | tee profile_${M}.log
