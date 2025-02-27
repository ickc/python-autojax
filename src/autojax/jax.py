from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    import numpy as np

jax.config.update("jax_enable_x64", True)

TWO_PI = 2.0 * jnp.pi
LOG_TWO_PI = jnp.log(TWO_PI)


@jax.jit
def mask_2d_centres_from(
    shape_native: tuple[int, int],
    pixel_scales: tuple[float, float],
    centre: tuple[float, float],
) -> tuple[float, float]:
    """
    Returns the (y,x) scaled central coordinates of a mask from its shape, pixel-scales and centre.

    The coordinate system is defined such that the positive y axis is up and positive x axis is right.

    Parameters
    ----------
    shape_native
        The (y,x) shape of the 2D array the scaled centre is computed for.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D array.
    centre : (float, flloat)
        The (y,x) centre of the 2D mask.

    Returns
    -------
    tuple (float, float)
        The (y,x) scaled central coordinates of the input array.

    Examples
    --------
    centres_scaled = centres_from(shape=(5,5), pixel_scales=(0.5, 0.5), centre=(0.0, 0.0))
    """
    return (
        0.5 * (shape_native[0] - 1) - (centre[0] / pixel_scales[0]),
        0.5 * (shape_native[1] - 1) + (centre[1] / pixel_scales[1]),
    )


@partial(jax.jit, static_argnums=0)
def mask_2d_circular_from(
    shape_native: tuple[int, int],
    pixel_scales: tuple[float, float],
    radius: float,
    centre: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Returns a circular mask from the 2D mask array shape and radius of the circle.

    This creates a 2D array where all values within the mask radius are unmasked and therefore `False`.

    Parameters
    ----------
    shape_native: Tuple[int, int]
        The (y,x) shape of the mask in units of pixels. Usually this is (N_PRIME, N_PRIME).
    pixel_scales
        The scaled units to pixel units conversion factor of each pixel, in arcsec.
    radius
        The radius (in scaled units) of the circle within which pixels unmasked, in arcsec.
    centre
            The centre of the circle used to mask pixels, in arcsec.

    Returns
    -------
    ndarray
        The 2D mask array whose central pixels are masked as a circle.

    Examples
    --------
    mask = mask_circular_from(
        shape=(10, 10), pixel_scales=0.1, radius=0.5, centre=(0.0, 0.0))
    """
    centres_scaled = mask_2d_centres_from(shape_native, pixel_scales, centre)
    ys, xs = jnp.indices(shape_native)
    return (radius * radius) < (
        jnp.square((ys - centres_scaled[0]) * pixel_scales[0])  #
        + jnp.square((xs - centres_scaled[1]) * pixel_scales[1])
    )


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

    The function is written in a way that the memory use does not depend on size of data K.

    Parameters
    ----------
    visibilities_real : ndarray, shape (K,), dtype=float64
        The two dimensional masked image of values which `w_tilde_data` is computed from.
    noise_map_real : ndarray, shape (K,), dtype=float64
        The two dimensional masked noise-map of values which `w_tilde_data` is computed from.
    uv_wavelengths : ndarray, shape (K, 2), dtype=float64
    grid_radians_slim : ndarray, shape (M, 2), dtype=float64
    native_index_for_slim_index : ndarray, shape (M, 2), dtype=int64
        An array that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray, shape (M,), dtype=float64
        A matrix that encodes the PSF convolution values between the imaging divided by the noise map**2 that enables
        efficient calculation of the data vector.
    """
    M = grid_radians_slim.shape[0]
    g_2pi = TWO_PI * grid_radians_slim
    g_2pi_y = g_2pi[:, 0]
    g_2pi_x = g_2pi[:, 1]

    def f_k(
        visibilities_real: float,
        noise_map_real: float,
        uv_wavelengths: np.ndarray[tuple[int], np.float64],
    ) -> np.ndarray[tuple[int], np.float64]:
        return jnp.cos(g_2pi_x * uv_wavelengths[0] + g_2pi_y * uv_wavelengths[1]) * (
            jnp.square(jnp.square(noise_map_real) / visibilities_real)
        )

    def f_scan(
        sum_: np.ndarray[tuple[int], np.float64],
        args: tuple[float, float, np.ndarray[tuple[int], np.float64]],
    ) -> tuple[np.ndarray[tuple[int], np.float64], None]:
        visibilities_real, noise_map_real, uv_wavelengths = args
        return sum_ + f_k(visibilities_real, noise_map_real, uv_wavelengths), None

    res, _ = jax.lax.scan(
        f_scan,
        jnp.zeros(M),
        (
            visibilities_real,
            noise_map_real,
            uv_wavelengths,
        ),
    )
    return res


@jax.jit
def w_tilde_curvature_interferometer_from(
    noise_map_real: np.ndarray[tuple[int], np.float64],
    uv_wavelengths: np.ndarray[tuple[int, int], np.float64],
    grid_radians_slim: np.ndarray[tuple[int, int], np.float64],
) -> np.ndarray[tuple[int, int], np.float64]:
    r"""
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the NUFFT of every pair of
    image pixels given the noise map. This can be used to efficiently compute the curvature matrix via the mappings
    between image and source pixels, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_preload_interferometer_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    Note that the current implementation does not take advantage of the fact that w_tilde is symmetric,
    due to the use of vectorized operations.

    .. math::
        \tilde{W}_{ij} = \sum_{k=1}^N \frac{1}{n_k^2} \cos(2\pi[(g_{i1} - g_{j1})u_{k0} + (g_{i0} - g_{j0})u_{k1}])

    The function is written in a way that the memory use does not depend on size of data K.

    Parameters
    ----------
    noise_map_real : ndarray, shape (K,), dtype=float64
        The real noise-map values of the interferometer data.
    uv_wavelengths : ndarray, shape (K, 2), dtype=float64
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_slim : ndarray, shape (M, 2), dtype=float64
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    curvature_matrix : ndarray, shape (M, M), dtype=float64
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """
    M = grid_radians_slim.shape[0]
    g_2pi = TWO_PI * grid_radians_slim
    δg_2pi = g_2pi.reshape(M, 1, 2) - g_2pi.reshape(1, M, 2)
    δg_2pi_y = δg_2pi[:, :, 0]
    δg_2pi_x = δg_2pi[:, :, 1]

    def f_k(
        noise_map_real: float,
        uv_wavelengths: np.ndarray[tuple[int], np.float64],
    ) -> np.ndarray[tuple[int, int], np.float64]:
        return jnp.cos(δg_2pi_x * uv_wavelengths[0] + δg_2pi_y * uv_wavelengths[1]) * jnp.reciprocal(
            jnp.square(noise_map_real)
        )

    def f_scan(
        sum_: np.ndarray[tuple[int, int], np.float64],
        args: tuple[float, np.ndarray[tuple[int], np.float64]],
    ) -> tuple[np.ndarray[tuple[int, int], np.float64], None]:
        noise_map_real, uv_wavelengths = args
        return sum_ + f_k(noise_map_real, uv_wavelengths), None

    res, _ = jax.lax.scan(
        f_scan,
        jnp.zeros((M, M)),
        (
            noise_map_real,
            uv_wavelengths,
        ),
    )
    return res


@partial(jax.jit, static_argnums=0)
def w_tilde_curvature_compact_interferometer_from(
    grid_size: int,
    noise_map_real: np.ndarray[tuple[int], np.float64],
    uv_wavelengths: np.ndarray[tuple[int, int], np.float64],
    pixel_scale: float,
) -> np.ndarray[tuple[int, int], np.float64]:
    N = grid_size
    OFFSET = N - 1
    # no. of elements after taking the difference of a point in a grid to another
    N_DIFF = 2 * N - 1
    # This converts from arcsec to radian too
    TWOPI_D = (jnp.pi * jnp.pi * pixel_scale) / 324000.0

    δ_mn0 = (TWOPI_D * jnp.arange(grid_size, dtype=jnp.float64)).reshape(-1, 1)
    # shift the centre in the 1-axis
    δ_mn1 = TWOPI_D * (jnp.arange(N_DIFF, dtype=jnp.float64) - OFFSET)

    def f_k(
        noise_map_real: float,
        uv_wavelengths: np.ndarray[tuple[int], np.float64],
    ) -> np.ndarray[tuple[int, int], np.float64]:
        return jnp.cos(δ_mn1 * uv_wavelengths[0] - δ_mn0 * uv_wavelengths[1]) * jnp.square(
            jnp.reciprocal(noise_map_real)
        )

    def f_scan(
        sum_: np.ndarray[tuple[int, int], np.float64],
        args: tuple[float, np.ndarray[tuple[int], np.float64]],
    ) -> tuple[np.ndarray[tuple[int, int], np.float64], None]:
        noise_map_real, uv_wavelengths = args
        return sum_ + f_k(noise_map_real, uv_wavelengths), None

    w_compact, _ = jax.lax.scan(
        f_scan,
        jnp.zeros((N, N_DIFF)),
        (
            noise_map_real,
            uv_wavelengths,
        ),
    )
    return w_compact


@jax.jit
def w_tilde_via_compact_from(
    w_compact: np.ndarray[tuple[int, int], np.float64],
    native_index_for_slim_index: np.ndarray[tuple[int, int], np.int64],
) -> np.ndarray[tuple[int, int], np.float64]:
    N = w_compact.shape[0]
    OFFSET = N - 1
    p_ij = native_index_for_slim_index.reshape(-1, 1, 2) - native_index_for_slim_index.reshape(1, -1, 2)
    # flip i, j if first index is negative as cos(-x) = cos(x)
    # this essentially moved the sign of the first index to the second index, and then adds an offset to the second index
    p_ij_1 = jnp.where(jnp.signbit(p_ij[:, :, 0]), -p_ij[:, :, 1], p_ij[:, :, 1]) + OFFSET
    p_ij_0 = jnp.abs(p_ij[:, :, 0])
    return w_compact[p_ij_0, p_ij_1]


@jax.jit
def curvature_matrix_via_w_compact_from(
    w_compact: np.ndarray[tuple[int, int], np.float64],
    native_index_for_slim_index: np.ndarray[tuple[int, int], np.int64],
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
    w_tilde : ndarray, shape (M, M), dtype=float64
        A matrix of dimensions [image_pixels, image_pixels] that encodes the convolution or NUFFT of every image pixel
        pair on the noise map.
    mapping_matrix : ndarray, shape (M, S), dtype=float64
        The matrix representing the mappings between sub-grid pixels and pixelization pixels.

    Returns
    -------
    curvature_matrix : ndarray, shape (S, S), dtype=float64
        The curvature matrix `F` (see Warren & Dye 2003).
    """
    w_tilde = w_tilde_via_compact_from(w_compact, native_index_for_slim_index)
    return curvature_matrix_via_w_tilde_from(w_tilde, mapping_matrix)


@jax.jit
def w_tilde_via_preload_from(
    w_tilde_preload: np.ndarray[tuple[int, int], np.float64],
    native_index_for_slim_index: np.ndarray[tuple[int, int], np.int64],
) -> np.ndarray[tuple[int, int], np.float64]:
    """
    Use the preloaded w_tilde matrix (see `w_tilde_preload_interferometer_from`) to compute
    w_tilde (see `w_tilde_interferometer_from`) efficiently.

    Parameters
    ----------
    w_tilde_preload : ndarray, shape (2N, 2N), dtype=float64
        The preloaded values of the NUFFT that enable efficient computation of w_tilde.
    native_index_for_slim_index : ndarray, shape (M, 2), dtype=int64
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.

    Returns
    -------
    ndarray : shape (M, M), dtype=float64
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """
    y_i = native_index_for_slim_index[:, 0]
    x_i = native_index_for_slim_index[:, 1]
    Δy_ij = y_i.reshape(-1, 1) - y_i
    Δx_ij = x_i.reshape(-1, 1) - x_i
    return w_tilde_preload[Δy_ij, Δx_ij]


@partial(jax.jit, static_argnums=2)
def mapping_matrix_from(
    pix_indexes_for_sub_slim_index: np.ndarray[tuple[int, int], np.int64],
    pix_weights_for_sub_slim_index: np.ndarray[np.ndarray[tuple[int, int], np.float64]],
    pixels: int,
) -> np.ndarray[tuple[int, int], np.float64]:
    """
    Returns the mapping matrix, which is a matrix representing the mapping between every unmasked sub-pixel of the data
    and the pixels of a pixelization. Non-zero entries signify a mapping, whereas zeros signify no mapping.

    For example, if the data has 5 unmasked pixels (with `sub_size=1` so there are not sub-pixels) and the pixelization
    3 pixels, with the following mappings:

    data pixel 0 -> pixelization pixel 0
    data pixel 1 -> pixelization pixel 0
    data pixel 2 -> pixelization pixel 1
    data pixel 3 -> pixelization pixel 1
    data pixel 4 -> pixelization pixel 2

    The mapping matrix (which is of dimensions [data_pixels, pixelization_pixels]) would appear as follows:

    [1, 0, 0] [0->0]
    [1, 0, 0] [1->0]
    [0, 1, 0] [2->1]
    [0, 1, 0] [3->1]
    [0, 0, 1] [4->2]

    The mapping matrix is actually built using the sub-grid of the grid, whereby each pixel is divided into a grid of
    sub-pixels which are all paired to pixels in the pixelization. The entries in the mapping matrix now become
    fractional values dependent on the sub-pixel sizes.

    For example, for a 2x2 sub-pixels in each pixel means the fractional value is 1.0/(2.0^2) = 0.25, if we have the
    following mappings:

    data pixel 0 -> data sub pixel 0 -> pixelization pixel 0
    data pixel 0 -> data sub pixel 1 -> pixelization pixel 1
    data pixel 0 -> data sub pixel 2 -> pixelization pixel 1
    data pixel 0 -> data sub pixel 3 -> pixelization pixel 1
    data pixel 1 -> data sub pixel 0 -> pixelization pixel 1
    data pixel 1 -> data sub pixel 1 -> pixelization pixel 1
    data pixel 1 -> data sub pixel 2 -> pixelization pixel 1
    data pixel 1 -> data sub pixel 3 -> pixelization pixel 1
    data pixel 2 -> data sub pixel 0 -> pixelization pixel 2
    data pixel 2 -> data sub pixel 1 -> pixelization pixel 2
    data pixel 2 -> data sub pixel 2 -> pixelization pixel 3
    data pixel 2 -> data sub pixel 3 -> pixelization pixel 3

    The mapping matrix (which is still of dimensions [data_pixels, pixelization_pixels]) appears as follows:

    [0.25, 0.75, 0.0, 0.0] [1 sub-pixel maps to pixel 0, 3 map to pixel 1]
    [ 0.0,  1.0, 0.0, 0.0] [All sub-pixels map to pixel 1]
    [ 0.0,  0.0, 0.5, 0.5] [2 sub-pixels map to pixel 2, 2 map to pixel 3]

    For certain pixelizations each data sub-pixel maps to multiple pixelization pixels in a weighted fashion, for
    example a Delaunay pixelization where there are 3 mappings per sub-pixel whose weights are determined via a
    nearest neighbor interpolation scheme.

    In this case, each mapping value is multiplied by this interpolation weight (which are in the array
    `pix_weights_for_sub_slim_index`) when the mapping matrix is constructed.

    Parameters
    ----------
    pix_indexes_for_sub_slim_index : ndarray, shape (M, B), dtype=int64
        The mappings from a data sub-pixel index to a pixelization pixel index.
    pix_weights_for_sub_slim_index : ndarray, shape (M, B), dtype=float64
        The weights of the mappings of every data sub pixel and pixelization pixel.
    pixels
        The number of pixels in the pixelization.
    total_mask_pixels
        The number of datas pixels in the observed datas and thus on the grid.
    """
    M = pix_indexes_for_sub_slim_index.shape[0]
    S = pixels
    # as the mapping matrix is M by S, S would be out of bound (any out of bound index would do)
    OUT_OF_BOUND_IDX = S
    B = pix_indexes_for_sub_slim_index.shape[1]
    pix_indexes_for_sub_slim_index = pix_indexes_for_sub_slim_index.flatten()
    pix_indexes_for_sub_slim_index = jnp.where(
        pix_indexes_for_sub_slim_index == -1, OUT_OF_BOUND_IDX, pix_indexes_for_sub_slim_index
    )

    I_IDX = jnp.repeat(jnp.arange(M), B)
    return (
        jnp.zeros((M, S)).at[I_IDX, pix_indexes_for_sub_slim_index]
        # unique indices should be guranteed by pix_indexes_for_sub_slim_index-spec
        .set(pix_weights_for_sub_slim_index.flatten(), mode="drop", unique_indices=True)
    )


@jax.jit
def data_vector_from(
    mapping_matrix: np.ndarray[tuple[int, int], np.float64],
    dirty_image: np.ndarray[tuple[int], np.float64],
) -> np.ndarray[tuple[int], np.float64]:
    """
    Parameters
    ----------
    mapping_matrix : ndarray, shape (M, S), dtype=float64
        Matrix representing mappings between sub-grid pixels and pixelization pixels
    dirty_image : ndarray, shape (M,), dtype=float64
        The dirty image used to compute the data vector

    Returns
    -------
    data_vector : ndarray, shape (S,), dtype=float64
        The `data_vector` is a 1D vector whose values are solved for by the simultaneous linear equations constructed
        by this object.

    The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf), where the
    data vector is given by equation (4) and the letter D.

    If there are multiple linear objects the `data_vectors` are concatenated ensuring their values are solved
    for simultaneously.

    The calculation is described in more detail in `inversion_util.w_tilde_data_interferometer_from`.

    FLOPS: 2MS
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
    w_tilde : ndarray, shape (M, M), dtype=float64
        A matrix of dimensions [image_pixels, image_pixels] that encodes the convolution or NUFFT of every image pixel
        pair on the noise map.
    mapping_matrix : ndarray, shape (M, S), dtype=float64
        The matrix representing the mappings between sub-grid pixels and pixelization pixels.

    Returns
    -------
    curvature_matrix : ndarray, shape (S, S), dtype=float64
        The curvature matrix `F` (see Warren & Dye 2003).
    """
    return mapping_matrix.T @ w_tilde @ mapping_matrix


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

    Memory requirement: 2SP + S^2
    FLOPS: 1 + 2S + 2SP

    Parameters
    ----------
    coefficient
        The regularization coefficients which controls the degree of smoothing of the inversion reconstruction.
    neighbors : ndarray, shape (S, P), dtype=int64
        An array of length (total_pixels) which provides the index of all neighbors of every pixel in
        the Voronoi grid (entries of -1 correspond to no neighbor).
    neighbors_sizes : ndarray, shape (S,), dtype=int64
        An array of length (total_pixels) which gives the number of neighbors of every pixel in the
        Voronoi grid.

    Returns
    -------
    regularization_matrix : ndarray, shape (S, S), dtype=float64
        The regularization matrix computed using Regularization where the effective regularization
        coefficient of every source pixel is the same.
    """
    S, P = neighbors.shape
    # as the regularization matrix is S by S, S would be out of bound (any out of bound index would do)
    OUT_OF_BOUND_IDX = S
    regularization_coefficient = coefficient * coefficient

    # flatten it for feeding into the matrix as j indices
    neighbors = neighbors.flatten()
    # now create the corresponding i indices
    I_IDX = jnp.repeat(jnp.arange(S), P)
    # Entries of `-1` in `neighbors` (indicating no neighbor) are replaced with an out-of-bounds index.
    # This ensures that JAX can efficiently drop these entries during matrix updates.
    neighbors = jnp.where(neighbors == -1, OUT_OF_BOUND_IDX, neighbors)
    return (
        jnp.diag(1e-8 + regularization_coefficient * neighbors_sizes).at[I_IDX, neighbors]
        # unique indices should be guranteed by neighbors-spec
        .add(-regularization_coefficient, mode="drop", unique_indices=True)
    )


@jax.jit
def reconstruction_positive_negative_from(
    data_vector: np.ndarray[tuple[int], np.float64],
    curvature_reg_matrix: np.ndarray[tuple[int, int], np.float64],
) -> np.ndarray[tuple[int], np.float64]:
    """
    Solve the linear system [F + reg_coeff*H] S = D -> S = [F + reg_coeff*H]^-1 D given by equation (12)
    of https://arxiv.org/pdf/astro-ph/0302587.pdf

    S is the vector of reconstructed inversion values.

    This reconstruction uses a linear algebra solver that allows for negative and positives values in the solution.
    By allowing negative values, the solver is efficient, but there are many inference problems where negative values
    are nonphysical or undesirable.

    This function checks that the solution does not give a linear algebra error (e.g. because the input matrix is
    not positive-definitive).

    It also explicitly checks solutions where all reconstructed values go to the same value, and raises an exception if
    this occurs. This solution occurs in many scenarios when it is clear not a valid solution, and therefore is checked
    for and removed.

    Memory requirement: S
    FLOPS: O(S^3)

    Parameters
    ----------
    data_vector : ndarray, shape (S,), dtype=float64
        The `data_vector` D which is solved for.
    curvature_reg_matrix : ndarray, shape (S, S), dtype=float64
        The sum of the curvature and regularization matrices.

    Returns
    -------
    reconstruction : ndarray, shape (S,), dtype=float64
    """
    return jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(curvature_reg_matrix, lower=True), data_vector)


@jax.jit
def noise_normalization_complex_from(
    noise_map: np.ndarray[[int], np.complex128],
) -> float:
    """
    Returns the noise-map normalization terms of a complex noise-map, summing the noise_map value in every pixel as:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise_map : ndarray, shape (K,), dtype=complex128
        The masked noise-map of the dataset.

    Returns
    -------
    noise_normalization : float
    """
    return 2.0 * (
        noise_map.size * LOG_TWO_PI
        + jnp.log(jnp.absolute(noise_map.real)).sum()
        + jnp.log(jnp.absolute(noise_map.imag)).sum()
    )


@jax.jit
def log_likelihood_function(
    dirty_image: np.ndarray[tuple[int], np.float64],
    data: np.ndarray[tuple[int], np.complex128],
    noise_map: np.ndarray[tuple[int], np.complex128],
    w_compact: np.ndarray[tuple[int, int], np.float64],
    native_index_for_slim_index: np.ndarray[tuple[int, int], np.int64],
    mapping_matrix: np.ndarray[tuple[int, int], np.float64],
    neighbors: np.ndarray[tuple[int, int], np.int64],
    neighbors_sizes: np.ndarray[tuple[int], np.int64],
) -> float:
    """Calculates the log likelihood of interferometer data given a model.

    This function combines several steps:
    1. Calculates noise normalization from the complex noise map
    2. Computes the curvature matrix using w_tilde and mapping matrix
    3. Creates a regularization matrix using constant regularization
    4. Solves for the reconstruction allowing positive and negative values
    5. Combines terms to compute the final log likelihood

    Parameters
    ----------
    dirty_image : ndarray, shape (M,), dtype=float64
        The dirty image used to compute the data vector
    data : ndarray, shape (K,), dtype=complex128
        The complex interferometer data being fitted
    noise_map : ndarray, shape (K,), dtype=complex128
        The complex noise map of the data
    uv_wavelengths : ndarray, shape (K, 2), dtype=float64
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset
    grid_radians_slim : ndarray, shape (M, 2), dtype=float64
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask
    mapping_matrix : ndarray, shape (M, S), dtype=float64
        Matrix representing mappings between sub-grid pixels and pixelization pixels
    neighbors : ndarray, shape (S, P), dtype=int64
        Array providing indices of neighbors for each pixel
    neighbors_sizes : ndarray, shape (S,), dtype=int64
        Array giving number of neighbors for each pixel

    Returns
    -------
    float
        The log likelihood value of the model fit to the data

    Notes
    -----
    The log likelihood calculation follows the formalism described in Warren & Dye 2003
    (https://arxiv.org/pdf/astro-ph/0302587.pdf) with additional terms for interferometric data.

    Typical sizes: (716 -> 70000 means 716 in the test dataset, but can go up to 70000 in science case)

    M = number of image pixels in real_space_mask = 716 -> ~70000 => M^2 ~ 5e9
    K = number of visibilitiies = 190 -> ~1e7 (but this is only used to compute w_tilde otuside the likelihood function)
    P = number of neighbors = 10 -> 3 (for Delaunay) but can go up to 300 for Voronoi (but we can just focus on delaunay for now)
    S = number of source pixels (e.g. reconstruction.shape) = 716 -> 1000

    With these numbers,
    ``w_tilde`` is 36.5GiB,
    ``mapping_matrix`` is 0.52GiB.
    """
    coefficient = 1.0

    noise_normalization: float = noise_normalization_complex_from(noise_map)

    # (S, S)
    curvature_matrix = curvature_matrix_via_w_compact_from(w_compact, native_index_for_slim_index, mapping_matrix)

    # shape: (S, S)
    # memory requirement: O(S^2)
    # FLOPS: O(2SP)
    regularization_matrix = constant_regularization_matrix_from(
        coefficient,
        neighbors,
        neighbors_sizes,
    )
    # shape: (S, S)
    # FLOPS: S^2
    curvature_reg_matrix = curvature_matrix + regularization_matrix
    # shape: (S,)
    # FLOPS: 2MS
    data_vector = data_vector_from(mapping_matrix, dirty_image)
    # FLOPS: O(S^3)
    log_regularization_matrix_term_over2 = jnp.log(
        jnp.diag(jax.scipy.linalg.cholesky(regularization_matrix, lower=True))
    ).sum()
    # we don't call reconstruction_positive_negative_from
    # as we want to keep curvature_reg_matrix_L for the logdet
    # shape: (S,)
    # FLOPS: O(S^3)
    curvature_reg_matrix_L = jax.scipy.linalg.cholesky(curvature_reg_matrix, lower=True)
    # shape: (S,)
    # FLOPS: O(S^2)
    reconstruction = jax.scipy.linalg.cho_solve((curvature_reg_matrix_L, True), data_vector)
    # FLOPS: O(S)
    log_curvature_reg_matrix_term_over2 = jnp.log(jnp.diag(curvature_reg_matrix_L)).sum()

    # (K,)
    chi_real = data.real / noise_map.real
    # (K,)
    chi_imag = data.imag / noise_map.imag
    regularization_term_plus_chi_squared: float = (
        chi_real @ chi_real + chi_imag @ chi_imag - reconstruction @ data_vector
    )

    return (
        -0.5 * (regularization_term_plus_chi_squared + noise_normalization)
        + log_regularization_matrix_term_over2
        - log_curvature_reg_matrix_term_over2
    )
