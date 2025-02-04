from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import numba
import jax
import jax.numpy as jnp
from jax import lax

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


@jax.jit
def w_tilde_curvature_interferometer_from(
    noise_map_real: np.ndarray[tuple[int], np.float64],
    uv_wavelengths: np.ndarray[tuple[int, int], np.float64],
    grid_radians_slim: np.ndarray[tuple[int, int], np.float64],
) -> np.ndarray[tuple[int, int], np.float64]:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the NUFFT of every pair of
    image pixels given the noise map. This can be used to efficiently compute the curvature matrix via the mappings
    between image and source pixels, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_preload_interferometer_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    .. math::
        W̃_{ij} = \sum_{k=1}^N \frac{1}{n_k^2} \cos(2\pi[(g_{i1} - g_{j1})u_{k0} + (g_{i0} - g_{j0})u_{k1}])

    Parameters
    ----------
    noise_map_real : ndarray, shape (N,), dtype=float64
        The real noise-map values of the interferometer data.
    uv_wavelengths : ndarray, shape (N, 2), dtype=float64
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_slim : ndarray, shape (M, 2), dtype=float64
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    ndarray : ndarray, shape (M, M), dtype=float64
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """
    # (i∊M, j∊M, 1, 2)
    g_ij =  grid_radians_slim.reshape(-1, 1, 1, 2) - grid_radians_slim.reshape(1, -1, 1, 2)
    # (1, 1, k∊N, 2)
    u_k = uv_wavelengths.reshape(1, 1, -1, 2)
    return (
        jnp.cos(
            (2.0 * jnp.pi) *
            # (M, M, N)
            (
                g_ij[:, :, :, 0] * u_k[:, :, :, 1] +
                g_ij[:, :, :, 1] * u_k[:, :, :, 0]
            )
        ) /
        # (1, 1, k∊N)
        jnp.square(noise_map_real).reshape(1, 1, -1)
    ).sum(2)  # sum over k


@jax.jit
def data_vector_from(
    mapping_matrix: np.ndarray[tuple[int, int], np.float64],
    dirty_image: np.ndarray[tuple[int], np.float64],
) -> np.ndarray[tuple[int], np.float64]:
    """
    The `data_vector` is a 1D vector whose values are solved for by the simultaneous linear equations constructed
    by this object.

    The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf), where the
    data vector is given by equation (4) and the letter D.

    If there are multiple linear objects the `data_vectors` are concatenated ensuring their values are solved
    for simultaneously.

    The calculation is described in more detail in `inversion_util.w_tilde_data_interferometer_from`.
    """
    return dirty_image @ mapping_matrix


@jax.jit
def curvature_matrix_via_w_tilde_from(
    w_tilde: np.ndarray[tuple[int, int], np.float64],
    mapping_matrix: np.ndarray[tuple[int, int], np.float64],
) -> np.ndarray[tuple[int, int], np.float64]:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) from `w_tilde`.

    The dimensions of `w_tilde` are [image_pixels, image_pixels], meaning that for datasets with many image pixels
    this matrix can take up 10's of GB of memory. The calculation of the `curvature_matrix` via this function will
    therefore be very slow, and the method `curvature_matrix_via_w_tilde_curvature_preload_imaging_from` should be used
    instead.

    Parameters
    ----------
    w_tilde
        A matrix of dimensions [image_pixels, image_pixels] that encodes the convolution or NUFFT of every image pixel
        pair on the noise map.
    mapping_matrix
        The matrix representing the mappings between sub-grid pixels and pixelization pixels.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """
    return mapping_matrix.T @ w_tilde @ mapping_matrix


@jax.jit
def _idxs_from_neighbors(
    neighbors: np.ndarray[[int, int], np.int64],
) -> np.ndarray[[int, int], np.float64]:
    """
    Converts a neighbors array into a 2D index array for facilitating vectorized matrix updates.

    The original neighbors array defines adjacency relationships where:
        for i in range(M):
            for j in range(neighbors_sizes[i]):
                k = neighbors[i, j]
                if k != -1:  # neighbors_sizes guarantees that k will not be -1
                    mat[i, k] = ...

    This function reformulates the indexing pattern by returning a 2D array `idxs`, enabling:
        for k in range(idxs.shape[1]):
            if idxs[1, k] < M:  # this is no longer guaranteed as we aren't using neighbors_sizes
                mat[idxs[0, k], idxs[1, k]] = ...

    This transformation allows an alternative, more structured way to iterate over the indices
    rather than directly using the original neighbors array.

    **Handling Missing Neighbors:**
    Entries of `-1` in `neighbors` (indicating no neighbor) are replaced with an out-of-bounds index.
    This ensures that JAX can efficiently drop these entries during matrix updates.

    Parameters
    ----------
    neighbors : np.ndarray of shape (M, N)
        A 2D array where each row lists the neighboring indices of a pixel in the Voronoi grid.
        Entries of `-1` indicate the absence of a neighbor.

    Returns
    -------
    np.ndarray of shape (2, M * N)
        A 2D array of indices `idxs` that establishes the identity:
        `matrix[i, neighbors[i, j]] == matrix[idxs[0, k], idxs[1, k]]`,
        relating the original neighbors format to the transformed format.
    """
    M, N = neighbors.shape
    OUT_OF_BOUND_IDX = M
    neighbors = neighbors.flatten()
    return jnp.stack((jnp.repeat(jnp.arange(M), N), jnp.where(neighbors == -1, OUT_OF_BOUND_IDX, neighbors)))


@jax.jit
def constant_regularization_matrix_from(
    coefficient: float,
    neighbors: np.ndarray[[int, int], np.int64],
    neighbors_sizes: np.ndarray[[int], np.int64],
) -> np.ndarray[[int, int], np.float64]:
    """
    From the pixel-neighbors array, setup the regularization matrix using the instance regularization scheme.

    A complete description of regularizatin and the `regularization_matrix` can be found in the `Regularization`
    class in the module `autoarray.inversion.regularization`.

    Parameters
    ----------
    coefficient
        The regularization coefficients which controls the degree of smoothing of the inversion reconstruction.
    neighbors
        An array of length (total_pixels) which provides the index of all neighbors of every pixel in
        the Voronoi grid (entries of -1 correspond to no neighbor).
    neighbors_sizes
        An array of length (total_pixels) which gives the number of neighbors of every pixel in the
        Voronoi grid.

    Returns
    -------
    np.ndarray
        The regularization matrix computed using Regularization where the effective regularization
        coefficient of every source pixel is the same.
    """
    regularization_coefficient = coefficient * coefficient

    neighbor_idxs = _idxs_from_neighbors(neighbors)
    mat = jnp.diag(1e-8 + regularization_coefficient * neighbors_sizes)
    mat = mat.at[neighbor_idxs[0], neighbor_idxs[1]].add(-regularization_coefficient, mode="drop", unique_indices=True)
    return mat
