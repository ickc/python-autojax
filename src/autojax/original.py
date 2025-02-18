from __future__ import annotations

import numpy as np
from numba import jit


@jit(nopython=True, nogil=True, parallel=False)
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
    y_centre_scaled = (float(shape_native[0] - 1) / 2) - (centre[0] / pixel_scales[0])
    x_centre_scaled = (float(shape_native[1] - 1) / 2) + (centre[1] / pixel_scales[1])

    return (y_centre_scaled, x_centre_scaled)


@jit(nopython=True, nogil=True, parallel=False)
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
    shape_native: tuple[int, int]
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

    mask_2d = np.full(shape_native, True)

    centres_scaled = mask_2d_centres_from(
        shape_native=mask_2d.shape,
        pixel_scales=pixel_scales,
        centre=centre,
    )

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
            x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

            r_scaled = np.sqrt(x_scaled**2 + y_scaled**2)

            if r_scaled <= radius:
                mask_2d[y, x] = False

    return mask_2d


@jit(nopython=True, nogil=True, parallel=True)
def w_tilde_data_interferometer_from(
    visibilities_real: np.ndarray,
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
    native_index_for_slim_index,
) -> np.ndarray:
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

    image_pixels = len(native_index_for_slim_index)

    w_tilde_data = np.zeros(image_pixels)

    weight_map_real = visibilities_real / noise_map_real**2.0

    for ip0 in range(image_pixels):
        value = 0.0

        y = grid_radians_slim[ip0, 1]
        x = grid_radians_slim[ip0, 0]

        for vis_1d_index in range(uv_wavelengths.shape[0]):
            value += weight_map_real[vis_1d_index] ** -2.0 * np.cos(
                2.0 * np.pi * (y * uv_wavelengths[vis_1d_index, 0] + x * uv_wavelengths[vis_1d_index, 1])
            )

        w_tilde_data[ip0] = value

    return w_tilde_data


@jit(nopython=True, nogil=True, parallel=True)
def w_tilde_curvature_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
) -> np.ndarray:
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

    w_tilde = np.zeros((grid_radians_slim.shape[0], grid_radians_slim.shape[0]))

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            y_offset = grid_radians_slim[i, 1] - grid_radians_slim[j, 1]
            x_offset = grid_radians_slim[i, 0] - grid_radians_slim[j, 0]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                w_tilde[i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (y_offset * uv_wavelengths[vis_1d_index, 0] + x_offset * uv_wavelengths[vis_1d_index, 1])
                )

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            w_tilde[j, i] = w_tilde[i, j]

    return w_tilde


@jit(nopython=True, nogil=True, parallel=True)
def w_tilde_curvature_preload_interferometer_from(
    noise_map_real: np.ndarray[tuple[int], np.float64],
    uv_wavelengths: np.ndarray[tuple[int, int], np.float64],
    shape_masked_pixels_2d: tuple[int, int],
    grid_radians_2d: np.ndarray[tuple[int, int, int], np.float64],
) -> np.ndarray[tuple[int, int], np.float64]:
    """Computes a preload matrix for efficient calculation of the NUFFT w_tilde matrix.

    The matrix w_tilde is a matrix of dimensions [unmasked_image_pixels, unmasked_image_pixels] that encodes the
    NUFFT of every pair of image pixels given the noise map. This can be used to efficiently compute the curvature
    matrix via the mapping matrix, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely expensive. This method
    creates a preload matrix that can compute the matrix w_tilde via an efficient preloading scheme which exploits the
    symmetries in the NUFFT.

    To compute w_tilde, one first defines a real space mask where every False entry is an unmasked pixel which is
    used in the calculation, for example:

    Mask Example:
        | x | x | x | x | x | x | x | x | x | x |
        | x | x | x | x | x | x | x | x | x | x |      This is an imaging.Mask2D, where:
        | x | x | x | x | x | x | x | x | x | x |
        | x | x | x | x | x | x | x | x | x | x |      x = `True` (Pixel is masked and excluded from lens)
        | x | x | x | o | o | o | x | x | x | x |      o = `False` (Pixel is not masked and included in lens)
        | x | x | x | o | o | o | x | x | x | x |
        | x | x | x | o | o | o | x | x | x | x |
        | x | x | x | x | x | x | x | x | x | x |
        | x | x | x | x | x | x | x | x | x | x |
        | x | x | x | x | x | x | x | x | x | x |

    Here, there are 9 unmasked pixels. Indexing of each unmasked pixel goes from the top-left corner right and
    downwards, therefore:

    Indexing Example:
        | x | x | x | x | x | x | x | x | x | x |
        | x | x | x | x | x | x | x | x | x | x |
        | x | x | x | x | x | x | x | x | x | x |
        | x | x | x | x | x | x | x | x | x | x |
        | x | x | x | 0 | 1 | 2 | x | x | x | x |
        | x | x | x | 3 | 4 | 5 | x | x | x | x |
        | x | x | x | 6 | 7 | 8 | x | x | x | x |
        | x | x | x | x | x | x | x | x | x | x |
        | x | x | x | x | x | x | x | x | x | x |
        | x | x | x | x | x | x | x | x | x | x |

    In the standard calculation of `w_tilde` it is a matrix of dimensions [unmasked_image_pixels, unmasked_pixel_images],
    therefore for the example mask above it would be dimensions [9, 9]. One performs a double for loop over
    `unmasked_image_pixels`, using the (y,x) spatial offset between every possible pair of unmasked image pixels to
    precompute values that depend on the properties of the NUFFT.

    This calculation has a lot of redundancy, because it uses the (y,x) *spatial offset* between the image pixels. For
    example, if two image pixels are next to one another by the same spacing the same value will be computed via the
    NUFFT. For the example mask above:

    - The value precomputed for pixel pair [0,1] is the same as pixel pairs [1,2], [3,4], [4,5], [6,7] and [7,8].
    - The value precomputed for pixel pair [0,3] is the same as pixel pairs [1,4], [2,5], [3,6], [4,7] and [5,8].
    - The values of pixels paired with themselves are also computed repeatedly for the standard calculation (e.g. 9
      times using the mask above).

    The `w_tilde_preload` method instead only computes each value once. To do this, it stores the preload values in a
    matrix of dimensions [shape_masked_pixels_y, shape_masked_pixels_x, 2], where `shape_masked_pixels` is the (y,x)
    size of the vertical and horizontal extent of unmasked pixels, e.g. the spatial extent over which the real space
    grid extends.

    Each entry in the matrix `w_tilde_preload[:,:,0]` provides the precomputed NUFFT value mapping an image pixel
    to a pixel offset by that much in the y and x directions, for example:

    - w_tilde_preload[0,0,0] gives the precomputed values of image pixels that are offset in the y direction by 0 and
      in the x direction by 0 - the values of pixels paired with themselves.
    - w_tilde_preload[1,0,0] gives the precomputed values of image pixels that are offset in the y direction by 1 and
      in the x direction by 0 - the values of pixel pairs [0,3], [1,4], [2,5], [3,6], [4,7] and [5,8]
    - w_tilde_preload[0,1,0] gives the precomputed values of image pixels that are offset in the y direction by 0 and
      in the x direction by 1 - the values of pixel pairs [0,1], [1,2], [3,4], [4,5], [6,7] and [7,8].

    Flipped pairs:
    The above preloaded values pair all image pixel NUFFT values when a pixel is to the right and/or down of the
    first image pixel. However, one must also precompute pairs where the paired pixel is to the left of the host
    pixels. These pairings are stored in `w_tilde_preload[:,:,1]`, and the ordering of these pairings is flipped in the
    x direction to make it straight forward to use this matrix when computing w_tilde.

    Parameters
    ----------
    noise_map_real : ndarray
        The real noise-map values of the interferometer data.
        Shape: (K,)
        dtype: float64
    uv_wavelengths : ndarray
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
        Shape: (K, 2)
        dtype: float64
    shape_masked_pixels_2d : tuple
        The (y,x) shape corresponding to the extent of unmasked pixels that go vertically and horizontally across the
        mask. E.g. (N, N), N = 30
    grid_radians_2d : ndarray
        The 2D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed. N_PRIME >= N. E.g. N_PRIME = 100
        Shape: (N_PRIME, N_PRIME, 2)
        dtype: float64

    Returns
    -------
    ndarray
        A matrix that precomputes the values for fast computation of w_tilde.
        Shape: (2N, 2N)
        dtype: float64
    """

    y_shape = shape_masked_pixels_2d[0]
    x_shape = shape_masked_pixels_2d[1]

    curvature_preload = np.zeros((y_shape * 2, x_shape * 2))

    #  For the second preload to index backwards correctly we have to extracted the 2D grid to its shape.
    grid_radians_2d = grid_radians_2d[0:y_shape, 0:x_shape]

    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]

    for i in range(y_shape):
        for j in range(x_shape):
            y_offset = grid_radians_2d[0, 0, 0] - grid_radians_2d[i, j, 0]
            x_offset = grid_radians_2d[0, 0, 1] - grid_radians_2d[i, j, 1]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                curvature_preload[i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (x_offset * uv_wavelengths[vis_1d_index, 0] + y_offset * uv_wavelengths[vis_1d_index, 1])
                )

    for i in range(y_shape):
        for j in range(x_shape):
            if j > 0:
                y_offset = grid_radians_2d[0, -1, 0] - grid_radians_2d[i, grid_x_shape - j - 1, 0]
                x_offset = grid_radians_2d[0, -1, 1] - grid_radians_2d[i, grid_x_shape - j - 1, 1]

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload[i, -j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (x_offset * uv_wavelengths[vis_1d_index, 0] + y_offset * uv_wavelengths[vis_1d_index, 1])
                    )

    for i in range(y_shape):
        for j in range(x_shape):
            if i > 0:
                y_offset = grid_radians_2d[-1, 0, 0] - grid_radians_2d[grid_y_shape - i - 1, j, 0]
                x_offset = grid_radians_2d[-1, 0, 1] - grid_radians_2d[grid_y_shape - i - 1, j, 1]

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload[-i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (x_offset * uv_wavelengths[vis_1d_index, 0] + y_offset * uv_wavelengths[vis_1d_index, 1])
                    )

    for i in range(y_shape):
        for j in range(x_shape):
            if i > 0 and j > 0:
                y_offset = grid_radians_2d[-1, -1, 0] - grid_radians_2d[grid_y_shape - i - 1, grid_x_shape - j - 1, 0]
                x_offset = grid_radians_2d[-1, -1, 1] - grid_radians_2d[grid_y_shape - i - 1, grid_x_shape - j - 1, 1]

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload[-i, -j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (x_offset * uv_wavelengths[vis_1d_index, 0] + y_offset * uv_wavelengths[vis_1d_index, 1])
                    )

    return curvature_preload


@jit(nopython=True, nogil=True, parallel=True)
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

    slim_size = len(native_index_for_slim_index)

    w_tilde_via_preload = np.zeros((slim_size, slim_size))

    for i in range(slim_size):
        i_y, i_x = native_index_for_slim_index[i]

        for j in range(i, slim_size):
            j_y, j_x = native_index_for_slim_index[j]

            y_diff = j_y - i_y
            x_diff = j_x - i_x

            w_tilde_via_preload[i, j] = w_tilde_preload[y_diff, x_diff]

    for i in range(slim_size):
        for j in range(i, slim_size):
            w_tilde_via_preload[j, i] = w_tilde_via_preload[i, j]

    return w_tilde_via_preload


def data_vector_from(mapping_matrix, dirty_image):
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
    return np.dot(mapping_matrix.T, dirty_image)


def curvature_matrix_via_w_tilde_from(w_tilde: np.ndarray, mapping_matrix: np.ndarray) -> np.ndarray:
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
    return np.dot(mapping_matrix.T, np.dot(w_tilde, mapping_matrix))


@jit(nopython=True, nogil=True, parallel=True)
def curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
    curvature_preload: np.ndarray[tuple[int, int], np.float64],
    pix_indexes_for_sub_slim_index: np.ndarray[tuple[int, int], np.int64],
    pix_size_for_sub_slim_index: np.ndarray[tuple[int], np.int64],
    pix_weights_for_sub_slim_index: np.ndarray[tuple[int, int], np.float64],
    native_index_for_slim_index: np.ndarray[tuple[int, int], np.int64],
    pix_pixels: int,
) -> np.ndarray[tuple[int, int], np.float64]:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) by computing it using `w_tilde_preload`
    (see `w_tilde_preload_interferometer_from`) for an interferometer inversion.

    To compute the curvature matrix via w_tilde the following matrix multiplication is normally performed:

    curvature_matrix = mapping_matrix.T * w_tilde * mapping matrix

    This function speeds this calculation up in two ways:

    1) Instead of using `w_tilde` (dimensions [image_pixels, image_pixels] it uses `w_tilde_preload` (dimensions
       [image_pixels, 2]). The massive reduction in the size of this matrix in memory allows for much fast computation.

    2) It omits the `mapping_matrix` and instead uses directly the 1D vector that maps every image pixel to a source
       pixel `native_index_for_slim_index`. This exploits the sparsity in the `mapping_matrix` to directly
       compute the `curvature_matrix` (e.g. it condenses the triple matrix multiplication into a double for loop!).

    Parameters
    ----------
    curvature_preload : ndarray, shape (2N, 2N), dtype=float64
        A matrix that precomputes the values for fast computation of w_tilde, which in this function is used to bypass
        the creation of w_tilde altogether and go directly to the `curvature_matrix`.
    pix_indexes_for_sub_slim_index : ndarray, shape (M, 3), dtype=int64
        The mappings from a data sub-pixel index to a pixelization's mesh pixel index.
    pix_size_for_sub_slim_index : ndarray, shape (M,), dtype=int64
        The number of mappings between each data sub pixel and pixelization pixel.
    pix_weights_for_sub_slim_index : ndarray, shape (M, 3), dtype=float64
        The weights of the mappings of every data sub pixel and pixelization pixel.
    native_index_for_slim_index : ndarray, shape (M, 2), dtype=int64
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.
    pix_pixels
        The total number of pixels in the pixelization's mesh that reconstructs the data. pix_pixels = S.

    Returns
    -------
    curvature_matrix : ndarray, shape (S, S), dtype=float64
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    preload = curvature_preload[0, 0]

    curvature_matrix = np.zeros((pix_pixels, pix_pixels))

    image_pixels = len(native_index_for_slim_index)

    for ip0 in range(image_pixels):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]

        for ip0_pix in range(pix_size_for_sub_slim_index[ip0]):
            sp0 = pix_indexes_for_sub_slim_index[ip0, ip0_pix]

            ip0_weight = pix_weights_for_sub_slim_index[ip0, ip0_pix]

            for ip1 in range(ip0 + 1, image_pixels):
                ip1_y, ip1_x = native_index_for_slim_index[ip1]

                for ip1_pix in range(pix_size_for_sub_slim_index[ip1]):
                    sp1 = pix_indexes_for_sub_slim_index[ip1, ip1_pix]

                    ip1_weight = pix_weights_for_sub_slim_index[ip1, ip1_pix]

                    y_diff = ip1_y - ip0_y
                    x_diff = ip1_x - ip0_x

                    curvature_matrix[sp0, sp1] += curvature_preload[y_diff, x_diff] * ip0_weight * ip1_weight

    curvature_matrix_new = np.zeros((pix_pixels, pix_pixels))

    for i in range(pix_pixels):
        for j in range(pix_pixels):
            curvature_matrix_new[i, j] = curvature_matrix[i, j] + curvature_matrix[j, i]

    curvature_matrix = curvature_matrix_new

    for ip0 in range(image_pixels):
        for ip0_pix in range(pix_size_for_sub_slim_index[ip0]):
            for ip1_pix in range(pix_size_for_sub_slim_index[ip0]):
                sp0 = pix_indexes_for_sub_slim_index[ip0, ip0_pix]
                sp1 = pix_indexes_for_sub_slim_index[ip0, ip1_pix]

                ip0_weight = pix_weights_for_sub_slim_index[ip0, ip0_pix]
                ip1_weight = pix_weights_for_sub_slim_index[ip0, ip1_pix]

                if sp0 > sp1:
                    curvature_matrix[sp0, sp1] += preload * ip0_weight * ip1_weight

                    curvature_matrix[sp1, sp0] += preload * ip0_weight * ip1_weight

                elif sp0 == sp1:
                    curvature_matrix[sp0, sp1] += preload * ip0_weight * ip1_weight

    return curvature_matrix


def constant_regularization_matrix_from(
    coefficient: float, neighbors: np.ndarray, neighbors_sizes: np.ndarray
) -> np.ndarray:
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

    parameters = len(neighbors)

    regularization_matrix = np.zeros(shape=(parameters, parameters))

    regularization_coefficient = coefficient**2.0

    for i in range(parameters):
        regularization_matrix[i, i] += 1e-8
        for j in range(neighbors_sizes[i]):
            neighbor_index = neighbors[i, j]
            regularization_matrix[i, i] += regularization_coefficient
            regularization_matrix[i, neighbor_index] -= regularization_coefficient

    return regularization_matrix


def reconstruction_positive_negative_from(
    data_vector: np.ndarray,
    curvature_reg_matrix: np.ndarray,
):
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


def noise_normalization_complex_from(noise_map: np.ndarray) -> float:
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
    noise_normalization_real = np.sum(np.log(2 * np.pi * noise_map.real**2.0))
    noise_normalization_imag = np.sum(np.log(2 * np.pi * noise_map.imag**2.0))
    return noise_normalization_real + noise_normalization_imag


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

    data_vector = data_vector_from(mapping_matrix, dirty_image)
    w_tilde = w_tilde_curvature_interferometer_from(
        noise_map.real,
        uv_wavelengths,
        grid_radians_slim,
    )
    curvature_matrix = curvature_matrix_via_w_tilde_from(w_tilde, mapping_matrix)
    regularization_matrix = constant_regularization_matrix_from(
        coefficient,
        neighbors,
        neighbors_sizes,
    )
    curvature_reg_matrix = curvature_matrix + regularization_matrix
    reconstruction = reconstruction_positive_negative_from(data_vector, curvature_reg_matrix)

    chi_squared_term_1 = np.linalg.multi_dot(
        [
            reconstruction.T,  # NOTE: shape = (M, )
            curvature_matrix,  # NOTE: shape = (M, M)
            reconstruction,  # NOTE: shape = (M, )
        ]
    )
    chi_squared_term_2 = -2.0 * np.linalg.multi_dot(
        [reconstruction.T, data_vector]  # NOTE: shape = (M, )  # NOTE: i.e. dirty_image
    )
    chi_squared_term_3 = np.add(  # NOTE: i.e. noise_normalization
        np.sum(data.real**2.0 / noise_map.real**2.0),
        np.sum(data.imag**2.0 / noise_map.imag**2.0),
    )

    chi_squared = chi_squared_term_1 + chi_squared_term_2 + chi_squared_term_3

    regularization_term = np.matmul(reconstruction.T, np.matmul(regularization_matrix, reconstruction))

    log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
    log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]

    noise_normalization = noise_normalization_complex_from(noise_map)

    return float(
        -0.5
        * (
            chi_squared
            + regularization_term
            + log_curvature_reg_matrix_term
            - log_regularization_matrix_term
            + noise_normalization
        )
    )


def log_likelihood_function_via_preload_method(
    dirty_image: np.ndarray[tuple[int], np.float64],
    data: np.ndarray[tuple[int], np.complex128],
    noise_map: np.ndarray[tuple[int], np.complex128],
    shape_masked_pixels_2d: tuple[int, int],
    grid_radians_2d: np.ndarray[tuple[int, int, int], np.float64],
    uv_wavelengths: np.ndarray[tuple[int, int], np.float64],
    mapping_matrix: np.ndarray[tuple[int, int], np.float64],
    neighbors: np.ndarray[tuple[int, int], np.int64],
    neighbors_sizes: np.ndarray[tuple[int], np.int64],
    pix_indexes_for_sub_slim_index: np.ndarray[tuple[int, int], np.int64],
    pix_size_for_sub_slim_index: np.ndarray[tuple[int], np.int64],
    pix_weights_for_sub_slim_index: np.ndarray[tuple[int, int], np.float64],
    native_index_for_slim_index: np.ndarray[tuple[int, int], np.int64],
    pix_pixels: int,
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
    shape_masked_pixels_2d : tuple(int, int)
        The (y,x) shape corresponding to the extent of unmasked pixels that go vertically and horizontally across the
        mask. E.g. (N, N), N = 30
    grid_radians_2d: ndarray, shape (N_PRIME, N_PRIME, 2), dtype=float64
        The 2D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed. N_PRIME >= N. E.g. N_PRIME = 100
    uv_wavelengths : ndarray, shape (K, 2), dtype=float64
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset
    mapping_matrix : ndarray, shape (M, S), dtype=float64
        Matrix representing mappings between sub-grid pixels and pixelization pixels
    neighbors : ndarray, shape (S, P), dtype=int64
        Array providing indices of neighbors for each pixel
    neighbors_sizes : ndarray, shape (S,), dtype=int64
        Array giving number of neighbors for each pixel
    pix_indexes_for_sub_slim_index : ndarray, shape (M, 3), dtype=int64
        The mappings from a data sub-pixel index to a pixelization's mesh pixel index.
    pix_size_for_sub_slim_index : ndarray, shape (M,), dtype=int64
        The number of mappings between each data sub pixel and pixelization pixel.
    pix_weights_for_sub_slim_index : ndarray, shape (M, 3), dtype=float64
        The weights of the mappings of every data sub pixel and pixelization pixel.
    native_index_for_slim_index : ndarray, shape (M, 2), dtype=int64
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.
    pix_pixels
        The total number of pixels in the pixelization's mesh that reconstructs the data. pix_pixels = S.

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
    N = The (y,x) shape corresponding to the extent of unmasked pixels that go vertically and horizontally across the mask. E.g. (N, N), N = 30. N ~ SQRT(M)
    K = number of visibilitiies = 190 -> ~1e7 (but this is only used to compute w_tilde otuside the likelihood function)
    P = number of neighbors = 10 -> 3 (for Delaunay) but can go up to 300 for Voronoi (but we can just focus on delaunay for now)
    S = number of source pixels (e.g. reconstruction.shape) = 716 -> 1000
    """
    coefficient = 1.0

    data_vector = data_vector_from(mapping_matrix, dirty_image)
    w_tilde_preload = w_tilde_curvature_preload_interferometer_from(
        noise_map.real,
        uv_wavelengths,
        shape_masked_pixels_2d,
        grid_radians_2d,
    )
    curvature_matrix = curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
        w_tilde_preload,
        pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index,
        pix_weights_for_sub_slim_index,
        native_index_for_slim_index,
        pix_pixels,
    )
    regularization_matrix = constant_regularization_matrix_from(
        coefficient,
        neighbors,
        neighbors_sizes,
    )
    curvature_reg_matrix = curvature_matrix + regularization_matrix
    reconstruction = reconstruction_positive_negative_from(data_vector, curvature_reg_matrix)

    chi_squared_term_1 = np.linalg.multi_dot(
        [
            reconstruction.T,  # NOTE: shape = (M, )
            curvature_matrix,  # NOTE: shape = (M, M)
            reconstruction,  # NOTE: shape = (M, )
        ]
    )
    chi_squared_term_2 = -2.0 * np.linalg.multi_dot(
        [reconstruction.T, data_vector]  # NOTE: shape = (M, )  # NOTE: i.e. dirty_image
    )
    chi_squared_term_3 = np.add(  # NOTE: i.e. noise_normalization
        np.sum(data.real**2.0 / noise_map.real**2.0),
        np.sum(data.imag**2.0 / noise_map.imag**2.0),
    )

    chi_squared = chi_squared_term_1 + chi_squared_term_2 + chi_squared_term_3

    regularization_term = np.matmul(reconstruction.T, np.matmul(regularization_matrix, reconstruction))

    log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
    log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]

    noise_normalization = noise_normalization_complex_from(noise_map)

    return float(
        -0.5
        * (
            chi_squared
            + regularization_term
            + log_curvature_reg_matrix_term
            - log_regularization_matrix_term
            + noise_normalization
        )
    )
