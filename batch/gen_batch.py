#!/usr/bin/env python3


import defopt

SBATDCH_TEMPLATE: str = """
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=autojax-bench-{name}
#SBATCH --output=batch/{name}.%J.out
#SBATCH --error=batch/{name}.%J.err
#SBATCH --partition=cosma8-shm
#SBATCH --account=dr004
#SBATCH --time=3-00:00:00
"""

NUM_THREADS_TEMPLATE: str = """
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
"""

BENCHMARK_TEMPLATE: str = """
# set generated data sizes #####################################################

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

pixi run --environment={environment} pytest --benchmark-save={name} -vv -k 'DataGenerated'
"""


def gen_batch(
    *,
    sbatch: bool = False,
    num_threads: int | None = None,
    grid_size: int = 30,
    n_mapping_neighbors: int = 3,
    data_size: int = 1024,
    neighbor_size: int = 32,
    src_img_size: int = 256,
    environment: str = "default",
) -> None:
    template_list: list[str] = []
    if sbatch:
        template_list.append(SBATDCH_TEMPLATE)
    if num_threads is not None:
        template_list.append(NUM_THREADS_TEMPLATE)
    template_list.append(BENCHMARK_TEMPLATE)
    template = "\n".join(template_list)
    name = f"N={grid_size}_B={n_mapping_neighbors}_K={data_size}_P={neighbor_size}_S={src_img_size}_NUM_THREADS={num_threads}_{environment}"
    string = template.format(
        name=name,
        num_threads=num_threads,
        grid_size=grid_size,
        n_mapping_neighbors=n_mapping_neighbors,
        data_size=data_size,
        neighbor_size=neighbor_size,
        src_img_size=src_img_size,
        environment=environment,
    )
    with open(f"batch/{name}.sh", "w", encoding="utf-8") as f:
        f.write(string)


if __name__ == "__main__":
    defopt.run(gen_batch)
