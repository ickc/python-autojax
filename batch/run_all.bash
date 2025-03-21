#!/usr/bin/env bash

# usage:
# 
# on cosma8, starts an interactive job
# $ srun --partition=cosma8-shm --account=dr004 --time=3-00:00:00 --exclusive --pty /bin/bash
# then run
# $ pixi run batch/run_all.bash

# default small case
grid_size=30
data_size=1024
n_mapping_neighbors=3
for num_threads in None 1 128; do
    for environment in default cuda; do
        echo "Running with grid_size=$grid_size, data_size=$data_size, n_mapping_neighbors=$n_mapping_neighbors, num_threads=$num_threads, environment=$environment"
        if [[ $num_threads == "None" ]]; then
            batch/gen_batch.py --grid-size $grid_size --data-size $data_size --n-mapping-neighbors $n_mapping_neighbors --environment $environment
        else
            batch/gen_batch.py --grid-size $grid_size --data-size $data_size --n-mapping-neighbors $n_mapping_neighbors --environment $environment --num-threads $num_threads
        fi
        bash "batch/N=${grid_size}_B=${n_mapping_neighbors}_K=${data_size}_P=32_S=256_NUM_THREADS=${num_threads}_${environment}.sh"
    done
done


for grid_size in 32 64; do
    data_size=$((8 * grid_size * grid_size))
    for n_mapping_neighbors in 3 300; do
        for num_threads in None 1 128; do
            for environment in default cuda; do
                echo "Running with grid_size=$grid_size, data_size=$data_size, n_mapping_neighbors=$n_mapping_neighbors, num_threads=$num_threads, environment=$environment"
                if [[ $num_threads == "None" ]]; then
                    batch/gen_batch.py --grid-size $grid_size --data-size $data_size --n-mapping-neighbors $n_mapping_neighbors --environment $environment
                else
                    batch/gen_batch.py --grid-size $grid_size --data-size $data_size --n-mapping-neighbors $n_mapping_neighbors --environment $environment --num-threads $num_threads
                fi
                bash "batch/N=${grid_size}_B=${n_mapping_neighbors}_K=${data_size}_P=32_S=256_NUM_THREADS=${num_threads}_${environment}.sh"
            done
        done
    done
done
