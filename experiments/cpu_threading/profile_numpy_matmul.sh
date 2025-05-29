#!/usr/bin/env bash

set -e

M=1024
N=1024
K=1024

echo "=========== Numpy    M=${M}, N=${N}, K=${K}, NUM_THREADS=None ==========="
command time -v ipython profile_numpy_matmul.py ${M} ${K} ${N}

for NUM_THREADS in 1 2 4 8 10 14; do
    echo "=========== Numpy    M=${M}, N=${N}, K=${K}, NUM_THREADS=${NUM_THREADS} ==========="

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

    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=${NUM_THREADS} --xla_force_host_platform_device_count=${NUM_THREADS}"
    export TF_NUM_INTEROP_THREADS=1
    export TF_NUM_INTRAOP_THREADS=${NUM_THREADS}

    command time -v ipython profile_numpy_matmul.py ${M} ${K} ${N}
done
