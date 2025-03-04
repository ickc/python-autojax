#!/usr/bin/env python3

import defopt

template: str = """#!/bin/bash -l

#SBATCH -N 1
#SBATCH -J autojax-bench-{name}
#SBATCH -o batch/{name}.%J.out
#SBATCH -e batch/{name}.%J.err
#SBATCH -p cosma8-milan
#SBATCH -A do018
#SBATCH --exclusive
#SBATCH -t 72:00:00

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

pytest --benchmark-save={name} -vv
pytest-benchmark compare {name} --csv=batch/{name}.csv --columns=mean,stddev,ops,rounds,iterations --sort=mean
"""


def gen_batch(
    *,
    num_threads: int = 64,
    grid_size: int = 128,
    n_mapping_neighbors: int = 3,
    data_size: int = 1024,
    neighbor_size: int = 10,
    src_img_size: int = 256,
) -> None:
    name = f"N={grid_size}_B={n_mapping_neighbors}_K={data_size}_P={neighbor_size}_S={src_img_size}_NUM_THREADS={num_threads}"
    string = template.format(
        name=name,
        num_threads=num_threads,
        grid_size=grid_size,
        n_mapping_neighbors=n_mapping_neighbors,
        data_size=data_size,
        neighbor_size=neighbor_size,
        src_img_size=src_img_size,
    )
    with open(f"batch/{name}.sh", "w") as f:
        f.write(string)


if __name__ == "__main__":
    defopt.run(gen_batch)
