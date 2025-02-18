from __future__ import annotations

import numpy as np
from numba import jit

TWO_PI = 2.0 * np.pi
LOG_TWO_PI = np.log(TWO_PI)


@jit("UniTuple(f8, 2)(UniTuple(i8, 2), UniTuple(f8, 2), UniTuple(f8, 2))", nopython=True, nogil=True, parallel=False)
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


@jit("b1[:, ::1](UniTuple(i8, 2), UniTuple(f8, 2), f8, UniTuple(f8, 2))", nopython=True, nogil=True, parallel=False)
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
        The (y,x) shape of the mask in units of pixels.
    pixel_scales
        The scaled units to pixel units conversion factor of each pixel.
    radius
        The radius (in scaled units) of the circle within which pixels unmasked.
    centre
            The centre of the circle used to mask pixels.

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


@jit(
    "float64[::1](float64[::1], float64[::1], float64[:,::1], float64[:,::1], int64[:,::1])",
    nopython=True,
    nogil=True,
    parallel=True,
)
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
    # assume M < K to put TWO_PI multiplication there
    g_i = TWO_PI * grid_radians_slim.reshape(-1, 1, 2)
    u_k = uv_wavelengths.reshape(1, -1, 2)
    # A_ik, i<M, k<N
    # A = g_i[:, :, 0] * u_k[:, :, 1] + g_i[:, :, 1] * u_k[:, :, 0]
    return np.cos(g_i[:, :, 0] * u_k[:, :, 1] + g_i[:, :, 1] * u_k[:, :, 0]) @ np.square(
        np.square(noise_map_real) / visibilities_real
    )


@jit("f8[:, ::1](f8[::1], f8[:, ::1], f8[:, ::1])", nopython=True, nogil=True, parallel=True)
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
    # assume M < K to put TWO_PI multiplication there
    g_i = TWO_PI * grid_radians_slim.reshape(-1, 1, 2)
    u_k = uv_wavelengths.reshape(1, -1, 2)
    # A_ik, i<M, k<N
    A = g_i[:, :, 0] * u_k[:, :, 1] + g_i[:, :, 1] * u_k[:, :, 0]

    noise_map_real_inv = np.reciprocal(noise_map_real)
    C = np.cos(A) * noise_map_real_inv
    S = np.sin(A) * noise_map_real_inv

    curvature_matrix = C @ C.T + S @ S.T
    return curvature_matrix


@jit("f8[::1](f8[:, ::1], f8[::1])", nopython=True, nogil=True)
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


@jit("f8[:, ::1](f8[:, ::1], f8[:, ::1])", nopython=True, nogil=True)
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


@jit("f8[:, ::1](f8, i8[:, ::1], i8[::1])", nopython=True, nogil=True, parallel=False)
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
    S = neighbors_sizes.size
    regularization_coefficient = coefficient * coefficient

    regularization_matrix = np.diag(1e-8 + regularization_coefficient * neighbors_sizes)
    for i in range(S):
        for j in range(neighbors_sizes[i]):
            regularization_matrix[i, neighbors[i, j]] -= regularization_coefficient
    return regularization_matrix


@jit("f8[::1](f8[::1], f8[:, ::1])", nopython=True, nogil=True)
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


@jit("f8(c16[::1])", nopython=True, nogil=True, parallel=False)
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


@jit(
    "f8(f8[::1], c16[::1], c16[::1], f8[:, ::1], f8[:, ::1], f8[:, ::1], i8[:, ::1], i8[::1])",
    nopython=True,
    nogil=True,
    parallel=True,
)
def log_likelihood_function(
    dirty_image: np.ndarray[tuple[int], np.float64],
    data: np.ndarray[tuple[int], np.complex128],
    noise_map: np.ndarray[tuple[int], np.complex128],
    uv_wavelengths: np.ndarray[tuple[int, int], np.float64],
    grid_radians_slim: np.ndarray[tuple[int, int], np.float64],
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

    M = number of image pixels in real_space_mask = 716 -> ~70000
    K = number of visibilitiies = 190 -> ~1e7 (but this is only used to compute w_tilde otuside the likelihood function)
    P = number of neighbors = 10 -> 3 (for Delaunay) but can go up to 300 for Voronoi (but we can just focus on delaunay for now)
    S = number of source pixels (e.g. reconstruction.shape) = 716 -> 1000
    """
    coefficient = 1.0

    noise_normalization: float = noise_normalization_complex_from(noise_map)

    # (M, M)
    w_tilde = w_tilde_curvature_interferometer_from(
        np.ascontiguousarray(noise_map.real),
        uv_wavelengths,
        grid_radians_slim,
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
