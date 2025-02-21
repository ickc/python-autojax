# arg parse
import argparse

import numpy as np
from jax import numpy as jnp
from jax.experimental import sparse

import autojax

from .test_all import DataGenerated


def parse_args():
    parser = argparse.ArgumentParser()
    # N, N_PRIME, K, P, S
    parser.add_argument("-N", type=int, default=30)
    parser.add_argument("--N-PRIME", type=int, default=100)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("-P", type=int, default=32)
    parser.add_argument("-S", type=int, default=256)
    return parser.parse_args()


def main(args=None):
    data = (
        DataGenerated()
        if args is None
        else DataGenerated(
            N_=args.N,
            N_PRIME_=args.N_PRIME,
            K_=args.K,
            P_=args.P,
            S_=args.S,
        )
    )
    data_dict = data.dict()

    noise_map_real = data_dict["noise_map_real"]
    uv_wavelengths = data_dict["uv_wavelengths"]
    grid_radians_2d = data_dict["grid_radians_2d"]
    native_index_for_slim_index = data_dict["native_index_for_slim_index"]
    mapping_matrix = data_dict["mapping_matrix"]
    pixel_scale = data._pixel_scales

    for i in ("noise_map_real", "uv_wavelengths", "grid_radians_2d", "native_index_for_slim_index", "mapping_matrix"):
        var = locals()[i]
        print("===", i, "===")
        print(np.info(var))

    noise_map_real = jnp.array(noise_map_real)
    uv_wavelengths = jnp.array(uv_wavelengths)
    grid_radians_2d = jnp.array(grid_radians_2d)
    native_index_for_slim_index = jnp.array(native_index_for_slim_index)
    mapping_matrix = sparse.BCOO.fromdense(mapping_matrix)

    def run():
        w_compact = autojax.jax.w_tilde_curvature_compact_interferometer_from(
            noise_map_real,
            uv_wavelengths,
            pixel_scale,
            grid_radians_2d,
        )
        curvature_matrix = autojax.jax.curvature_matrix_via_w_compact_from(
            w_compact,
            native_index_for_slim_index,
            mapping_matrix,
        )
        return curvature_matrix.block_until_ready()

    run()


if __name__ == "__main__":
    args = parse_args()
    main(args)
