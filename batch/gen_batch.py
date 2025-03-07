#!/usr/bin/env python3

from pathlib import Path

import defopt


def gen_batch(
    *,
    template_path: Path = Path(__file__).parent / "batch.template.bash",
    num_threads: int = 256,
    grid_size: int = 30,
    n_mapping_neighbors: int = 3,
    data_size: int = 1024,
    neighbor_size: int = 32,
    src_img_size: int = 256,
) -> None:
    with template_path.open("r") as f:
        template = f.read()
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
    with open(f"batch/{name}.sh", "w", encoding="utf-8") as f:
        f.write(string)


if __name__ == "__main__":
    defopt.run(gen_batch)
