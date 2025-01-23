from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    import numpy as np

jax.config.update("jax_enable_x64", True)


@jax.jit
def w_tilde_data_interferometer_from(
    visibilities_real: np.ndarray[tuple[int], np.float64],
    noise_map_real: np.ndarray[tuple[int], np.float64],
    uv_wavelengths: np.ndarray[tuple[int, int], np.float64],
    grid_radians_slim: np.ndarray[tuple[int, int], np.float64],
    native_index_for_slim_index: np.ndarray[tuple[int, int], np.int64],
) -> np.ndarray[tuple[int], np.float64]:
    r"""
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF convolution of
    every pair of image pixels given the noise map. This can be used to efficiently compute the curvature matrix via
    the mappings between image and source pixels, in a way that omits having to perform the PSF convolution on every
    individual source pixel. This provides a significant speed up for inversions of imaging datasets.

    When w_tilde is used to perform an inversion, the mapping matrices are not computed, meaning that they cannot be
    used to compute the data vector. This method creates the vector `w_tilde_data` which allows for the data
    vector to be computed efficiently without the mapping matrix.

    The matrix w_tilde_data is dimensions [image_pixels] and encodes the PSF convolution with the `weight_map`,
    where the weights are the image-pixel values divided by the noise-map values squared:

    weight = image / noise**2.0

    .. math::
        \tilde{w}_{\text{data},i} = \sum_{j=1}^N \left(\frac{N_{r,j}^2}{V_{r,j}}\right)^2 \cos\left(2\pi(g_{i,1}u_{j,0} + g_{i,0}u_{j,1})\right)

    Parameters
    ----------
    visibilities_real : ndarray, shape (N,), dtype=float64
        The two dimensional masked image of values which `w_tilde_data` is computed from.
    noise_map_real : ndarray, shape (N,), dtype=float64
        The two dimensional masked noise-map of values which `w_tilde_data` is computed from.
    uv_wavelengths : ndarray, shape (N, 2), dtype=float64
    grid_radians_slim : ndarray, shape (M, 2), dtype=float64
    native_index_for_slim_index : ndarray, shape (M, 2), dtype=int64
        An array that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray, shape (M,), dtype=float64
        A matrix that encodes the PSF convolution values between the imaging divided by the noise map**2 that enables
        efficient calculation of the data vector.
    """
    g_i = grid_radians_slim.reshape(-1, 1, 2)
    u_j = uv_wavelengths.reshape(1, -1, 2)
    return (
        # (1, j∊N)
        jnp.square(jnp.square(noise_map_real) / visibilities_real).reshape(1, -1) *
        jnp.cos(
            (2.0 * jnp.pi) *
            # (i∊M, j∊N)
            (
                g_i[:, :, 0] * u_j[:, :, 1] +
                g_i[:, :, 1] * u_j[:, :, 0]
            )
        )
    ).sum(axis=1)  # sum over j
