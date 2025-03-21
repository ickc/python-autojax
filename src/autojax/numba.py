from __future__ import annotations

import numba
import numpy as np

TWO_PI = 2.0 * np.pi
LOG_TWO_PI = np.log(TWO_PI)


@numba.jit(
    "UniTuple(f8, 2)(UniTuple(i8, 2), UniTuple(f8, 2), UniTuple(f8, 2))", nopython=True, nogil=True, parallel=False
)
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


@numba.jit(
    "b1[:, ::1](UniTuple(i8, 2), UniTuple(f8, 2), f8, UniTuple(f8, 2))", nopython=True, nogil=True, parallel=False
)
def mask_2d_circular_from(
    shape_native: tuple[int, int],
    pixel_scales: tuple[float, float],
    radius: float,
    centre: tuple[float, float],
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
    ys, xs = np.indices(shape_native)
    return (radius * radius) < (
        np.square((ys - centres_scaled[0]) * pixel_scales[0]) + np.square((xs - centres_scaled[1]) * pixel_scales[1])
    )


@numba.jit(
    "float64[::1](float64[::1], float64[::1], float64[:,::1], float64[:,::1])",
    nopython=True,
    nogil=True,
    parallel=True,
)
def w_tilde_data_interferometer_from(
    visibilities_real: np.ndarray[tuple[int], np.float64],
    noise_map_real: np.ndarray[tuple[int], np.float64],
    uv_wavelengths: np.ndarray[tuple[int, int], np.float64],
    grid_radians_slim: np.ndarray[tuple[int, int], np.float64],
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
    K = uv_wavelengths.shape[0]
    g_2pi = TWO_PI * grid_radians_slim

    res = np.zeros(M)
    # as K is larger, stream the data only once
    for k in range(K):
        uv_k_y = uv_wavelengths[k, 0]
        uv_k_x = uv_wavelengths[k, 1]
        n_k = noise_map_real[k]
        v_k = visibilities_real[k]
        w_k = np.square(np.square(n_k) / v_k)
        for i in range(M):
            res[i] += np.cos(g_2pi[i, 1] * uv_k_y + g_2pi[i, 0] * uv_k_x) * w_k
    return res


@numba.jit("f8[:, ::1](f8[::1], f8[:, ::1], f8[:, ::1])", nopython=True, nogil=True, parallel=True)
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
    K = uv_wavelengths.shape[0]
    g_2pi = TWO_PI * grid_radians_slim
    δg_2pi = g_2pi.reshape(-1, 1, 2) - g_2pi.reshape(1, -1, 2)

    w = np.zeros((M, M))
    for k in numba.prange(K):
        w += np.cos(δg_2pi[:, :, 1] * uv_wavelengths[k, 0] + δg_2pi[:, :, 0] * uv_wavelengths[k, 1]) * np.reciprocal(
            np.square(noise_map_real[k])
        )
    return w


@numba.jit("f8[:, ::1](i8, f8[::1], f8[:, ::1], f8)", nopython=True, nogil=True, parallel=True)
def w_compact_curvature_interferometer_from(
    grid_size: int,
    noise_map_real: np.ndarray[tuple[int], np.float64],
    uv_wavelengths: np.ndarray[tuple[int, int], np.float64],
    pixel_scale: float,
) -> np.ndarray[tuple[int, int], np.float64]:
    K = uv_wavelengths.shape[0]
    N = grid_size
    OFFSET = N - 1
    # no. of elements after taking the difference of a point in a grid to another
    N_DIFF = 2 * N - 1
    # This converts from arcsec to radian too
    TWOPI_D = (np.pi * np.pi * pixel_scale) / 324000.0

    δ_mn0 = (TWOPI_D * np.arange(grid_size, dtype=np.float64)).reshape(-1, 1)
    # shift the centre in the 1-axis
    δ_mn1 = TWOPI_D * (np.arange(N_DIFF, dtype=np.float64) - OFFSET)

    w_compact = np.zeros((N, N_DIFF))
    for k in numba.prange(K):
        w_compact += np.cos(δ_mn1 * uv_wavelengths[k, 0] - δ_mn0 * uv_wavelengths[k, 1]) * np.square(
            np.reciprocal(noise_map_real[k])
        )
    return w_compact


@numba.jit("f8[:, ::1](f8[:, ::1], i8[:, ::1])", nopython=True, nogil=True, parallel=False)
def w_tilde_via_compact_from(
    w_compact: np.ndarray[tuple[int, int], np.float64],
    native_index_for_slim_index: np.ndarray[tuple[int, int], np.int64],
) -> np.ndarray[tuple[int, int], np.float64]:
    M = native_index_for_slim_index.shape[0]
    OFFSET = w_compact.shape[0] - 1
    w = np.empty((M, M))
    for i in range(M):
        i_0 = native_index_for_slim_index[i, 0]
        i_1 = native_index_for_slim_index[i, 1]
        for j in range(M):
            j_0 = native_index_for_slim_index[j, 0]
            j_1 = native_index_for_slim_index[j, 1]
            m = i_0 - j_0
            # flip i, j if m < 0 as cos(-x) = cos(x)
            n = (i_1 - j_1 if m >= 0 else j_1 - i_1) + OFFSET
            w[i, j] = w_compact[np.abs(m), n]
    return w


@numba.jit("f8[:, ::1](f8[:, ::1], i8[:, ::1])", nopython=True, nogil=True, parallel=False)
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
    M = native_index_for_slim_index.shape[0]
    w = np.empty((M, M))
    for i in range(M):
        y_i, x_i = native_index_for_slim_index[i]
        for j in range(M):
            y_j, x_j = native_index_for_slim_index[j]
            Δy = y_j - y_i
            Δx = x_j - x_i
            w[i, j] = w_tilde_preload[Δy, Δx]
    return w


@numba.jit("f8[:, ::1](i8[:, ::1], f8[:, ::1], i8)", nopython=True, nogil=True, parallel=True)
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
    # B = 3 for Delaunay
    M, B = pix_indexes_for_sub_slim_index.shape
    S = pixels

    mapping_matrix = np.zeros((M, S))
    for m in range(M):
        for b in range(B):
            s = pix_indexes_for_sub_slim_index[m, b]
            if s == -1:
                break
            w = pix_weights_for_sub_slim_index[m, b]
            mapping_matrix[m, s] = w
    return mapping_matrix


@numba.jit("f8[::1](f8[:, ::1], f8[::1])", nopython=True, nogil=True)
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
    """
    return dirty_image @ mapping_matrix


@numba.jit("f8[:, ::1](f8[:, ::1], f8[:, ::1])", nopython=True, nogil=True)
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


@numba.jit("f8[:, ::1](f8[:, ::1], i8[:, ::1], i8[:, ::1], f8[:, ::1], i8)", nopython=True, nogil=True, parallel=False)
def curvature_matrix_via_w_compact_sparse_mapping_matrix_direct_from(
    w_compact: np.ndarray[tuple[int, int], np.float64],
    native_index_for_slim_index: np.ndarray[tuple[int, int], np.int64],
    pix_indexes_for_sub_slim_index: np.ndarray[tuple[int, int], np.int64],
    pix_weights_for_sub_slim_index: np.ndarray[np.ndarray[tuple[int, int], np.float64]],
    pixels: int,
) -> np.ndarray[tuple[int, int], np.float64]:
    """Calculate the curvature matrix using the compact w_tilde matrix and the sparse mapping matrix.

    This calculates it directly without expanding anything in memory. It optimizes for low memory usage but requires more FLOPS.

    Memory cost: S^2 <- the output matrix size, i.e. no extra memory is used.

    FLOP cost: (4 + 3B^2) M^2, B = pix_size_for_sub_slim_index.mean(), B=3 for Delaunay.
    """
    M, B = pix_indexes_for_sub_slim_index.shape
    S: int = pixels
    OFFSET: int = w_compact.shape[0] - 1

    b1: int
    b2: int
    m1_0: int
    m1_1: int
    m2_0: int
    m2_1: int
    n1: int
    n2: int
    s1: int
    s2: int
    t_m1_s1: float
    t_m2_s2: float
    w_m1_m2: float

    F = np.zeros((S, S))
    for m1 in range(M):
        m1_0 = native_index_for_slim_index[m1, 0]
        m1_1 = native_index_for_slim_index[m1, 1]
        for m2 in range(M):
            m2_0 = native_index_for_slim_index[m2, 0]
            m2_1 = native_index_for_slim_index[m2, 1]

            n1 = m1_0 - m2_0
            # flip i, j if n1 < 0 as cos(-x) = cos(x)
            n2 = (m1_1 - m2_1 if n1 >= 0 else m2_1 - m1_1) + OFFSET
            w_m1_m2 = w_compact[np.abs(n1), n2]

            for b1 in range(B):
                s1 = pix_indexes_for_sub_slim_index[m1, b1]
                if s1 == -1:
                    break
                t_m1_s1 = pix_weights_for_sub_slim_index[m1, b1]
                for b2 in range(B):
                    s2 = pix_indexes_for_sub_slim_index[m2, b2]
                    if s2 == -1:
                        break
                    t_m2_s2 = pix_weights_for_sub_slim_index[m2, b2]
                    F[s1, s2] += t_m1_s1 * w_m1_m2 * t_m2_s2
    return F


@numba.jit("f8[:, ::1](f8[:, ::1], i8[:, ::1], i8[:, ::1], f8[:, ::1], i8)", nopython=True, nogil=True, parallel=False)
def _w_compact_matmul_sparse_mapping_matrix_from(
    w_compact: np.ndarray[tuple[int, int], np.float64],
    native_index_for_slim_index: np.ndarray[tuple[int, int], np.int64],
    pix_indexes_for_sub_slim_index: np.ndarray[tuple[int, int], np.int64],
    pix_weights_for_sub_slim_index: np.ndarray[np.ndarray[tuple[int, int], np.float64]],
    pixels: int,
) -> np.ndarray[tuple[int, int], np.float64]:
    """Calculate w_tilde @ mapping_matrix using the compact w_tilde matrix and the sparse mapping matrix.

    This expands w_tilde @ mapping_matrix as M by S dense matrix.

    Memory cost: MS <- the output matrix size

    FLOP cost: 2(2 + B)M^2, B = pix_size_for_sub_slim_index.mean(), B=3 for Delaunay.
    """
    M, B = pix_indexes_for_sub_slim_index.shape
    S: int = pixels
    OFFSET: int = w_compact.shape[0] - 1

    b2: int
    m1_0: int
    m1_1: int
    m2_0: int
    m2_1: int
    n1: int
    n2: int
    s2: int
    t_m2_s2: float
    w_m1_m2: float

    Ω = np.zeros((M, S))
    for m1 in range(M):
        m1_0 = native_index_for_slim_index[m1, 0]
        m1_1 = native_index_for_slim_index[m1, 1]
        for m2 in range(M):
            m2_0 = native_index_for_slim_index[m2, 0]
            m2_1 = native_index_for_slim_index[m2, 1]

            n1 = m1_0 - m2_0
            # flip i, j if n1 < 0 as cos(-x) = cos(x)
            n2 = (m1_1 - m2_1 if n1 >= 0 else m2_1 - m1_1) + OFFSET
            w_m1_m2 = w_compact[np.abs(n1), n2]

            for b2 in range(B):
                s2 = pix_indexes_for_sub_slim_index[m2, b2]
                if s2 == -1:
                    break
                t_m2_s2 = pix_weights_for_sub_slim_index[m2, b2]
                Ω[m1, s2] += w_m1_m2 * t_m2_s2
    return Ω


@numba.jit("f8[:, ::1](f8[:, ::1], i8[:, ::1], f8[:, ::1], i8)", nopython=True, nogil=True, parallel=False)
def sparse_mapping_matrix_transpose_matmul(
    matrix: np.ndarray[tuple[int, int], np.float64],
    pix_indexes_for_sub_slim_index: np.ndarray[tuple[int, int], np.int64],
    pix_weights_for_sub_slim_index: np.ndarray[np.ndarray[tuple[int, int], np.float64]],
    pixels: int,
) -> np.ndarray[tuple[int, int], np.float64]:
    """Calculate T^T @ matrix using the sparse mapping matrix representation.

    Assuming matrix is M by S2,

    Memory cost: S1 S2 <- the output matrix size (no extra memory is used).

    FLOP cost: 2M B S2, B = pix_size_for_sub_slim_index.mean(), B=3 for Delaunay.
    """
    Ω = matrix
    M, S2 = Ω.shape
    B = pix_indexes_for_sub_slim_index.shape[1]
    S1: int = pixels

    b1: int
    m: int
    s1: int
    s2: int
    t_m_s1: float

    F = np.zeros((S1, S2))
    for m in range(M):
        for b1 in range(B):
            s1 = pix_indexes_for_sub_slim_index[m, b1]
            if s1 == -1:
                break
            t_m_s1 = pix_weights_for_sub_slim_index[m, b1]
            for s2 in range(S2):
                F[s1, s2] += t_m_s1 * Ω[m, s2]
    return F


@numba.jit("f8[:, ::1](f8[:, ::1], i8[:, ::1], f8[:, ::1], i8)", nopython=True, nogil=True, parallel=False)
def curvature_matrix_via_w_wilde_sparse_mapping_matrix_from(
    w_tilde: np.ndarray[tuple[int, int], np.float64],
    pix_indexes_for_sub_slim_index: np.ndarray[tuple[int, int], np.int64],
    pix_weights_for_sub_slim_index: np.ndarray[np.ndarray[tuple[int, int], np.float64]],
    pixels: int,
) -> np.ndarray[tuple[int, int], np.float64]:
    """Calculate the curvature matrix using the w_tilde matrix and the sparse mapping matrix.

    Memory cost: MS + S^2

    FLOP cost: 2BM(M + S), B = pix_size_for_sub_slim_index.mean(), B=3 for Delaunay.
    """
    TW = sparse_mapping_matrix_transpose_matmul(
        w_tilde,
        pix_indexes_for_sub_slim_index,
        pix_weights_for_sub_slim_index,
        pixels,
    )
    return sparse_mapping_matrix_transpose_matmul(
        np.ascontiguousarray(TW.T),
        pix_indexes_for_sub_slim_index,
        pix_weights_for_sub_slim_index,
        pixels,
    )


@numba.jit("f8[:, ::1](f8[:, ::1], i8[:, ::1], i8[:, ::1], f8[:, ::1], i8)", nopython=True, nogil=True, parallel=False)
def curvature_matrix_via_w_compact_sparse_mapping_matrix_from(
    w_compact: np.ndarray[tuple[int, int], np.float64],
    native_index_for_slim_index: np.ndarray[tuple[int, int], np.int64],
    pix_indexes_for_sub_slim_index: np.ndarray[tuple[int, int], np.int64],
    pix_weights_for_sub_slim_index: np.ndarray[np.ndarray[tuple[int, int], np.float64]],
    pixels: int,
) -> np.ndarray[tuple[int, int], np.float64]:
    """Calculate the curvature matrix using the compact w_tilde matrix and the sparse mapping matrix.

    This calculates T^T @ w_tilde @ T as two matrix multiplications, WT = w_tilde @ T and F = T^T @ WT.

    This reduces FLOP requirements (by ~(4 + 3B^2)/(2(2 + B)) = 3.1-fold for Delaunay) at the cost of more memory usage by extra amount of MS.

    Memory cost: MS + S^2

    FLOP cost: 2(2 + B)M^2 + 2MBS, B = pix_size_for_sub_slim_index.mean(), B=3 for Delaunay.

    """
    return sparse_mapping_matrix_transpose_matmul(
        _w_compact_matmul_sparse_mapping_matrix_from(
            w_compact,
            native_index_for_slim_index,
            pix_indexes_for_sub_slim_index,
            pix_weights_for_sub_slim_index,
            pixels,
        ),
        pix_indexes_for_sub_slim_index,
        pix_weights_for_sub_slim_index,
        pixels,
    )


# for benchmark, as this is the function in closest correspondence to the original function
curvature_matrix_via_w_tilde_curvature_preload_interferometer_from = (
    curvature_matrix_via_w_compact_sparse_mapping_matrix_from
)


@numba.jit("f8[:, ::1](f8, i8[:, ::1], i8[::1])", nopython=True, nogil=True, parallel=False)
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
    regularization_coefficient = coefficient * coefficient

    regularization_matrix = np.diag(1e-8 + regularization_coefficient * neighbors_sizes)
    for i in range(S):
        for j in range(P):
            k = neighbors[i, j]
            if k == -1:
                break
            regularization_matrix[i, k] -= regularization_coefficient
    return regularization_matrix


@numba.jit("f8[::1](f8[::1], f8[:, ::1])", nopython=True, nogil=True)
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
    return np.linalg.solve(curvature_reg_matrix, data_vector)


@numba.jit("f8(c16[::1])", nopython=True, nogil=True, parallel=False)
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
    return 2.0 * (noise_map.size * LOG_TWO_PI + np.log(np.abs(noise_map.view(np.float64))).sum())


@numba.jit(
    "f8(f8[::1], c16[::1], c16[::1], f8[:, ::1], i8[:, ::1], f8[:, ::1], i8[:, ::1], i8[::1])",
    nopython=True,
    nogil=True,
    parallel=True,
)
def log_likelihood_function_via_w_tilde_from(
    dirty_image: np.ndarray[tuple[int], np.float64],
    data: np.ndarray[tuple[int], np.complex128],
    noise_map: np.ndarray[tuple[int], np.complex128],
    w_tilde: np.ndarray[tuple[int, int], np.float64],
    pix_indexes_for_sub_slim_index: np.ndarray[tuple[int, int], np.int64],
    pix_weights_for_sub_slim_index: np.ndarray[np.ndarray[tuple[int, int], np.float64]],
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
    pix_indexes_for_sub_slim_index : ndarray, shape (M, B), dtype=int64
        The mappings from a data sub-pixel index to a pixelization pixel index.
    pix_weights_for_sub_slim_index : ndarray, shape (M, B), dtype=float64
        The weights of the mappings of every data sub pixel and pixelization pixel.
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

    M = number of image pixels in real_space_mask = 716 -> ~70000
    K = number of visibilitiies = 190 -> ~1e7 (but this is only used to compute w_tilde otuside the likelihood function)
    P = number of neighbors = 10 -> 3 (for Delaunay) but can go up to 300 for Voronoi (but we can just focus on delaunay for now)
    S = number of source pixels (e.g. reconstruction.shape) = 716 -> 1000
    """
    S = neighbors_sizes.size
    coefficient = 1.0

    noise_normalization: float = noise_normalization_complex_from(noise_map)

    # (M, S)
    mapping_matrix = mapping_matrix_from(
        pix_indexes_for_sub_slim_index,
        pix_weights_for_sub_slim_index,
        S,
    )
    # (S, S)
    curvature_matrix = curvature_matrix_via_w_tilde_from(w_tilde, mapping_matrix)

    # (S, S)
    regularization_matrix = constant_regularization_matrix_from(
        coefficient,
        neighbors,
        neighbors_sizes,
    )
    # (S, S)
    curvature_reg_matrix = curvature_matrix + regularization_matrix
    # (S,)
    data_vector = data_vector_from(mapping_matrix, dirty_image)
    # (S,)
    reconstruction = reconstruction_positive_negative_from(data_vector, curvature_reg_matrix)

    # (2K,)
    chi = data.view(np.float64) / noise_map.view(np.float64)
    regularization_term_plus_chi_squared = chi @ chi - reconstruction @ data_vector

    log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
    log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]

    return -0.5 * (
        regularization_term_plus_chi_squared
        + log_curvature_reg_matrix_term
        - log_regularization_matrix_term
        + noise_normalization
    )


@numba.jit(
    "f8(f8[::1], c16[::1], c16[::1], f8[:, ::1], i8[:, ::1], i8[:, ::1], f8[:, ::1], i8[:, ::1], i8[::1])",
    nopython=True,
    nogil=True,
    parallel=True,
)
def log_likelihood_function_via_w_compact_from(
    dirty_image: np.ndarray[tuple[int], np.float64],
    data: np.ndarray[tuple[int], np.complex128],
    noise_map: np.ndarray[tuple[int], np.complex128],
    w_compact: np.ndarray[tuple[int, int], np.float64],
    native_index_for_slim_index: np.ndarray[tuple[int, int], np.int64],
    pix_indexes_for_sub_slim_index: np.ndarray[tuple[int, int], np.int64],
    pix_weights_for_sub_slim_index: np.ndarray[np.ndarray[tuple[int, int], np.float64]],
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
    grid_radians_slim : ndarray, shape (M, 2), dtype=float64
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask
    pix_indexes_for_sub_slim_index : ndarray, shape (M, B), dtype=int64
        The mappings from a data sub-pixel index to a pixelization pixel index.
    pix_weights_for_sub_slim_index : ndarray, shape (M, B), dtype=float64
        The weights of the mappings of every data sub pixel and pixelization pixel.
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

    M = number of image pixels in real_space_mask = 716 -> ~70000
    K = number of visibilitiies = 190 -> ~1e7 (but this is only used to compute w_tilde otuside the likelihood function)
    P = number of neighbors = 10 -> 3 (for Delaunay) but can go up to 300 for Voronoi (but we can just focus on delaunay for now)
    S = number of source pixels (e.g. reconstruction.shape) = 716 -> 1000
    """
    S = neighbors_sizes.size
    coefficient = 1.0

    noise_normalization: float = noise_normalization_complex_from(noise_map)

    # (S, S)
    curvature_matrix = curvature_matrix_via_w_compact_sparse_mapping_matrix_from(
        w_compact,
        native_index_for_slim_index,
        pix_indexes_for_sub_slim_index,
        pix_weights_for_sub_slim_index,
        S,
    )

    # (S, S)
    regularization_matrix = constant_regularization_matrix_from(
        coefficient,
        neighbors,
        neighbors_sizes,
    )
    # (S, S)
    curvature_reg_matrix = curvature_matrix + regularization_matrix
    # (M, S)
    mapping_matrix = mapping_matrix_from(
        pix_indexes_for_sub_slim_index,
        pix_weights_for_sub_slim_index,
        S,
    )
    # (S,)
    data_vector = data_vector_from(mapping_matrix, dirty_image)
    # (S,)
    reconstruction = reconstruction_positive_negative_from(data_vector, curvature_reg_matrix)

    # (2K,)
    chi = data.view(np.float64) / noise_map.view(np.float64)
    regularization_term_plus_chi_squared = chi @ chi - reconstruction @ data_vector

    log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
    log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]

    return -0.5 * (
        regularization_term_plus_chi_squared
        + log_curvature_reg_matrix_term
        - log_regularization_matrix_term
        + noise_normalization
    )
