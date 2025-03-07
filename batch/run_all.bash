#!/usr/bin/env bash

# usage: pixi run batch/run_all.bash

for grid_size in 32 64 128 256 512 1024 2048; do
    for num_threads in 1 256; do
        data_size=$((8 * grid_size * grid_size))
        echo "Running with grid_size=$grid_size, data_size=$data_size, num_threads=$num_threads"
        batch/gen_batch.py --grid-size $grid_size --data-size $data_size --num-threads $num_threads
        sbatch "batch/N=${grid_size}_B=3_K=${data_size}_P=32_S=256_NUM_THREADS=${num_threads}.sh"
    done
done
