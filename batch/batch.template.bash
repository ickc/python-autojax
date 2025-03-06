#!/bin/bash -l

#SBATCH -N 1
#SBATCH -J autojax-bench-{name}
#SBATCH -o batch/{name}.%J.out
#SBATCH -e batch/{name}.%J.err
#SBATCH -p mi300x
#SBATCH -A do018
#SBATCH -t 00:30:00

# set no. of threads ###########################################################

export MKL_NUM_THREADS={num_threads}
export MKL_DOMAIN_NUM_THREADS="MKL_BLAS={num_threads}"
export MKL_DYNAMIC=FALSE

export OMP_NUM_THREADS={num_threads}
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export OMP_DYNAMIC=FALSE

export NUMEXPR_NUM_THREADS={num_threads}

export OPENBLAS_NUM_THREADS={num_threads}

export NUMBA_NUM_THREADS={num_threads}

export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={num_threads} --xla_force_host_platform_device_count={num_threads}"
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS={num_threads}

# set generated data sizes #####################################################

export AUTOJAX_NO_LOAD_DATA=1
# N
export AUTOJAX_GRID_SIZE={grid_size}
# B
export AUTOJAX_N_MAPPING_NEIGHBORS={n_mapping_neighbors}
# K
export AUTOJAX_DATA_SIZE={data_size}
# P
export AUTOJAX_NEIGHBOR_SIZE={neighbor_size}
# S
export AUTOJAX_SRC_IMG_SIZE={src_img_size}

# run pytest ###################################################################

pytest --benchmark-save={name} -vv "TestBenchWTilde or TestBenchCurvatureMatrix"
