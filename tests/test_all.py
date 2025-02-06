from __future__ import annotations

import hashlib

import numpy as np
import pytest

from autojax import jax, numba, original


def create_M(n):
    """
    Create the n x n symmetric tridiagonal matrix M defined by:
      M[i, i] = 2 for i = 0,...,n-1,
      M[i, i+1] = M[i+1, i] = -1 for i = 0,...,n-2.
    """
    # Initialize an n x n matrix filled with zeros
    M = np.zeros((n, n))

    # Set the diagonal entries to 2
    np.fill_diagonal(M, 2)

    # Set the sub- and super-diagonals to -1
    for i in range(n - 1):
        M[i, i + 1] = -1
        M[i + 1, i] = -1

    return M


def create_M_inv(n):
    """
    Create the inverse of the n x n matrix M with entries:
      (M_inv)_{ij} = (min(i, j) * (n + 1 - max(i, j))) / (n + 1)
    where i, j are 1-indexed. Adjust indices for Python (0-indexed).
    """
    M_inv = np.zeros((n, n))

    # i and j will range from 0 to n-1, corresponding to i+1 and j+1 in the formula.
    for i in range(n):
        for j in range(n):
            # Convert indices to 1-indexed values
            ip1, jp1 = i + 1, j + 1
            M_inv[i, j] = (min(ip1, jp1) * (n + 1 - max(ip1, jp1))) / (n + 1)
    return M_inv


def deterministic_seed(string: str, *numbers: int) -> int:
    """Generate a deterministic seed from the class name."""
    hash_value = hashlib.md5(repr((string, numbers)).encode()).hexdigest()  # Get a hash from class name
    return int(hash_value, 16) % (2**32)  # Convert to an integer within a reasonable range


def gen_mapping_matrix(N: int) -> np.ndarray[tuple[int, int], np.float64]:
    """Generate a random mapping matrix of size N.

    The actual mapping matrix is quite sparse and have columns sum to 1.
    We aren't producing the sparse-ness here.
    """
    rng = np.random.default_rng(deterministic_seed("mapping_matrix", N, N))
    mapping_matrix = rng.random((N, N))
    mapping_matrix /= mapping_matrix.sum(axis=1).reshape(-1, 1)
    np.testing.assert_allclose(mapping_matrix.sum(axis=1), 1.0)
    return mapping_matrix


def gen_dirty_image(N: int) -> np.ndarray[tuple[int], np.float64]:
    """Generate a random dirty image of size N."""
    rng = np.random.default_rng(deterministic_seed("dirty_image", N))
    return rng.random(N)


def gen_visibilities_real(N: int) -> np.ndarray[tuple[int], np.float64]:
    """Generate random real visibilities of size N."""
    rng = np.random.default_rng(deterministic_seed("visibilities_real", N))
    return rng.random(N)


def gen_noise_map_real(N: int) -> np.ndarray[tuple[int], np.float64]:
    """Generate random real noise map of size N."""
    rng = np.random.default_rng(deterministic_seed("noise_map_real", N))
    return rng.random(N)


def gen_uv_wavelengths(N: int) -> np.ndarray[tuple[int, int], np.float64]:
    """Generate random uv wavelengths of size N."""
    rng = np.random.default_rng(deterministic_seed("uv_wavelengths", N, 2))
    return rng.random((N, 2))


def gen_grid_radians_slim(M: int) -> np.ndarray[tuple[int, int], np.float64]:
    """Generate random slim grid of size M."""
    rng = np.random.default_rng(deterministic_seed("grid_radians_slim", M, 2))
    return rng.random((M, 2))


def gen_native_index_for_slim_index(M: int) -> np.ndarray[tuple[int, int], np.int64]:
    """Generate random native index for slim index of size M."""
    rng = np.random.default_rng(deterministic_seed("native_index_for_slim_index", M, 2))
    return rng.integers(0, M, size=(M, 2))


def gen_w_tilde(N: int) -> np.ndarray[tuple[int, int], np.float64]:
    """Generate random w_tilde of size N."""
    rng = np.random.default_rng(deterministic_seed("w_tilde", N, N))
    return rng.random((N, N))


def gen_neighbors(M: int, N: int) -> tuple[
    np.ndarray[tuple[int], np.int64],
    np.ndarray[tuple[int, int], np.int64],
]:
    """Generate random neighbors and neighbors_sizes of sizes M and N."""
    rng = np.random.default_rng(deterministic_seed("neighbors", M, N))
    neighbors_sizes = rng.integers(0, N + 1, M)
    neighbors = np.full((M, N), -1, dtype=np.int64)
    for i in range(M):
        neighbors[i, : neighbors_sizes[i]] = np.sort(
            rng.choice(
                M,
                neighbors_sizes[i],
                replace=False,
            )
        )
    return neighbors, neighbors_sizes


def gen_data_vector(N: int) -> np.ndarray[tuple[int], np.float64]:
    """Generate random data vector of size N."""
    rng = np.random.default_rng(deterministic_seed("data_vector", N))
    return rng.random(N)


def gen_noise_map(N: int) -> np.ndarray[tuple[int], np.complex128]:
    """Generate random real noise map of size N."""
    rng = np.random.default_rng(deterministic_seed("noise_map", N))
    return rng.random(2 * N).view(np.complex128)


class TestMask2DCircularFrom:
    shape_native = (100, 100)
    pixel_scales = (0.2, 0.2)
    radius = 3.0
    centre = (0.0, 0.0)

    @pytest.fixture
    def setup_data(self):
        """Fixture to set up test data"""
        shape_native = self.shape_native
        pixel_scales = self.pixel_scales
        radius = self.radius
        centre = self.centre

        ref = original.mask_2d_circular_from(shape_native, pixel_scales, radius)

        return {
            "shape_native": shape_native,
            "pixel_scales": pixel_scales,
            "radius": radius,
            "centre": centre,
            "ref": ref,
        }

    @pytest.mark.benchmark(group="mask_2d_circular_from")
    def test_mask_2d_circular_from_original(self, setup_data, benchmark):
        """Benchmark the original mask_2d_circular_from function"""
        shape_native = setup_data["shape_native"]
        pixel_scales = setup_data["pixel_scales"]
        radius = setup_data["radius"]
        centre = setup_data["centre"]

        ref = setup_data["ref"]

        def run():
            return original.mask_2d_circular_from(shape_native, pixel_scales, radius, centre)

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="mask_2d_circular_from")
    def test_mask_2d_circular_from_numba(self, setup_data, benchmark):
        """Benchmark the numba mask_2d_circular_from function"""
        shape_native = setup_data["shape_native"]
        pixel_scales = setup_data["pixel_scales"]
        radius = setup_data["radius"]
        centre = setup_data["centre"]

        ref = setup_data["ref"]

        def run():
            return numba.mask_2d_circular_from(shape_native, pixel_scales, radius, centre)

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="mask_2d_circular_from")
    def test_mask_2d_circular_from_jax(self, setup_data, benchmark):
        """Benchmark the jax mask_2d_circular_from function"""
        shape_native = setup_data["shape_native"]
        pixel_scales = setup_data["pixel_scales"]
        radius = setup_data["radius"]
        centre = setup_data["centre"]

        ref = setup_data["ref"]

        def run():
            return jax.mask_2d_circular_from(shape_native, pixel_scales, radius, centre).block_until_ready()

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)


class TestWTildeDataInterferometer:
    M = 1000
    N = 1000

    @pytest.fixture
    def setup_data(self):
        """Fixture to set up test data"""
        M = self.M
        N = self.N

        visibilities_real = gen_visibilities_real(N)
        noise_map_real = gen_noise_map_real(N)
        uv_wavelengths = gen_uv_wavelengths(N)
        grid_radians_slim = gen_grid_radians_slim(M)
        native_index_for_slim_index = gen_native_index_for_slim_index(M)

        ref = original.w_tilde_data_interferometer_from(
            visibilities_real,
            noise_map_real,
            uv_wavelengths,
            grid_radians_slim,
            native_index_for_slim_index,
        )
        return {
            "visibilities_real": visibilities_real,
            "noise_map_real": noise_map_real,
            "uv_wavelengths": uv_wavelengths,
            "grid_radians_slim": grid_radians_slim,
            "native_index_for_slim_index": native_index_for_slim_index,
            "ref": ref,
        }

    @pytest.mark.benchmark(group="w_tilde_data_interferometer_from")
    def test_w_tilde_data_interferometer_from_original(self, setup_data, benchmark):
        """Benchmark the original w_tilde_data_interferometer_from function"""
        visibilities_real = setup_data["visibilities_real"]
        noise_map_real = setup_data["noise_map_real"]
        uv_wavelengths = setup_data["uv_wavelengths"]
        grid_radians_slim = setup_data["grid_radians_slim"]
        native_index_for_slim_index = setup_data["native_index_for_slim_index"]

        ref = setup_data["ref"]

        def run():
            return original.w_tilde_data_interferometer_from(
                visibilities_real,
                noise_map_real,
                uv_wavelengths,
                grid_radians_slim,
                native_index_for_slim_index,
            )

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="w_tilde_data_interferometer_from")
    def test_w_tilde_data_interferometer_from_numba(self, setup_data, benchmark):
        """Benchmark the numba w_tilde_data_interferometer_from function"""
        visibilities_real = setup_data["visibilities_real"]
        noise_map_real = setup_data["noise_map_real"]
        uv_wavelengths = setup_data["uv_wavelengths"]
        grid_radians_slim = setup_data["grid_radians_slim"]
        native_index_for_slim_index = setup_data["native_index_for_slim_index"]

        ref = setup_data["ref"]

        def run():
            return numba.w_tilde_data_interferometer_from(
                visibilities_real,
                noise_map_real,
                uv_wavelengths,
                grid_radians_slim,
                native_index_for_slim_index,
            )

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="w_tilde_data_interferometer_from")
    def test_w_tilde_data_interferometer_from_jax(self, setup_data, benchmark):
        """Benchmark the jax w_tilde_data_interferometer_from function"""
        visibilities_real = setup_data["visibilities_real"]
        noise_map_real = setup_data["noise_map_real"]
        uv_wavelengths = setup_data["uv_wavelengths"]
        grid_radians_slim = setup_data["grid_radians_slim"]
        native_index_for_slim_index = setup_data["native_index_for_slim_index"]

        ref = setup_data["ref"]

        def run():
            return jax.w_tilde_data_interferometer_from(
                visibilities_real,
                noise_map_real,
                uv_wavelengths,
                grid_radians_slim,
                native_index_for_slim_index,
            ).block_until_ready()

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)


class TestWTildeCurvatureInterferometer:
    M = 200
    N = 100

    @pytest.fixture
    def setup_data(self):
        """Fixture to set up test data"""
        M = self.M
        N = self.N

        noise_map_real = gen_noise_map_real(N)
        uv_wavelengths = gen_uv_wavelengths(N)
        grid_radians_slim = gen_grid_radians_slim(M)

        ref = original.w_tilde_curvature_interferometer_from(
            noise_map_real,
            uv_wavelengths,
            grid_radians_slim,
        )
        return {
            "noise_map_real": noise_map_real,
            "uv_wavelengths": uv_wavelengths,
            "grid_radians_slim": grid_radians_slim,
            "ref": ref,
        }

    @pytest.mark.benchmark(group="w_tilde_curvature_interferometer_from")
    def test_w_tilde_curvature_interferometer_from_original(self, setup_data, benchmark):
        """Benchmark the original w_tilde_curvature_interferometer_from function"""
        noise_map_real = setup_data["noise_map_real"]
        uv_wavelengths = setup_data["uv_wavelengths"]
        grid_radians_slim = setup_data["grid_radians_slim"]

        ref = setup_data["ref"]

        def run():
            return original.w_tilde_curvature_interferometer_from(
                noise_map_real,
                uv_wavelengths,
                grid_radians_slim,
            )

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="w_tilde_curvature_interferometer_from")
    def test_w_tilde_curvature_interferometer_from_numba(self, setup_data, benchmark):
        """Benchmark the numba w_tilde_curvature_interferometer_from function"""
        noise_map_real = setup_data["noise_map_real"]
        uv_wavelengths = setup_data["uv_wavelengths"]
        grid_radians_slim = setup_data["grid_radians_slim"]

        ref = setup_data["ref"]

        def run():
            return numba.w_tilde_curvature_interferometer_from(
                noise_map_real,
                uv_wavelengths,
                grid_radians_slim,
            )

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="w_tilde_curvature_interferometer_from")
    def test_w_tilde_curvature_interferometer_from_jax(self, setup_data, benchmark):
        """Benchmark the jax w_tilde_curvature_interferometer_from function"""
        noise_map_real = setup_data["noise_map_real"]
        uv_wavelengths = setup_data["uv_wavelengths"]
        grid_radians_slim = setup_data["grid_radians_slim"]

        ref = setup_data["ref"]

        def run():
            return jax.w_tilde_curvature_interferometer_from(
                noise_map_real,
                uv_wavelengths,
                grid_radians_slim,
            ).block_until_ready()

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)


class TestDataVectorFrom:
    N: int = 1000

    @pytest.fixture
    def setup_data(self):
        """Fixture to set up test data"""
        N = self.N
        mapping_matrix = gen_mapping_matrix(N)
        dirty_image = gen_dirty_image(N)

        ref = original.data_vector_from(mapping_matrix, dirty_image)

        return {
            "mapping_matrix": mapping_matrix,
            "dirty_image": dirty_image,
            "ref": ref,
        }

    @pytest.mark.benchmark(group="data_vector_from")
    def test_data_vector_from_original(self, setup_data, benchmark):
        """Benchmark the original data_vector_from function"""
        mapping_matrix = setup_data["mapping_matrix"]
        dirty_image = setup_data["dirty_image"]

        ref = setup_data["ref"]

        def run():
            return original.data_vector_from(mapping_matrix, dirty_image)

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="data_vector_from")
    def test_data_vector_from_numba(self, setup_data, benchmark):
        """Benchmark the numba data_vector_from function"""
        mapping_matrix = setup_data["mapping_matrix"]
        dirty_image = setup_data["dirty_image"]

        ref = setup_data["ref"]

        def run():
            return numba.data_vector_from(mapping_matrix, dirty_image)

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="data_vector_from")
    def test_data_vector_from_jax(self, setup_data, benchmark):
        """Benchmark the jax data_vector_from function"""
        mapping_matrix = setup_data["mapping_matrix"]
        dirty_image = setup_data["dirty_image"]

        ref = setup_data["ref"]

        def run():
            return jax.data_vector_from(mapping_matrix, dirty_image).block_until_ready()

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)


class TestCurvatureMatrixViaWTildeFrom:
    N: int = 1000

    @pytest.fixture
    def setup_data(self):
        """Fixture to set up test data"""
        N = self.N

        w_tilde = gen_w_tilde(N)
        mapping_matrix = gen_mapping_matrix(N)

        ref = original.curvature_matrix_via_w_tilde_from(w_tilde, mapping_matrix)

        return {
            "w_tilde": w_tilde,
            "mapping_matrix": mapping_matrix,
            "ref": ref,
        }

    @pytest.mark.benchmark(group="curvature_matrix_via_w_tilde_from")
    def test_curvature_matrix_via_w_tilde_from_original(self, setup_data, benchmark):
        """Benchmark the original curvature_matrix_via_w_tilde_from function"""
        w_tilde = setup_data["w_tilde"]
        mapping_matrix = setup_data["mapping_matrix"]

        ref = setup_data["ref"]

        def run():
            return original.curvature_matrix_via_w_tilde_from(w_tilde, mapping_matrix)

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="curvature_matrix_via_w_tilde_from")
    def test_curvature_matrix_via_w_tilde_from_numba(self, setup_data, benchmark):
        """Benchmark the numba curvature_matrix_via_w_tilde_from function"""
        w_tilde = setup_data["w_tilde"]
        mapping_matrix = setup_data["mapping_matrix"]

        ref = setup_data["ref"]

        def run():
            return numba.curvature_matrix_via_w_tilde_from(w_tilde, mapping_matrix)

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="curvature_matrix_via_w_tilde_from")
    def test_curvature_matrix_via_w_tilde_from_jax(self, setup_data, benchmark):
        """Benchmark the jax curvature_matrix_via_w_tilde_from function"""
        w_tilde = setup_data["w_tilde"]
        mapping_matrix = setup_data["mapping_matrix"]

        ref = setup_data["ref"]

        def run():
            return jax.curvature_matrix_via_w_tilde_from(w_tilde, mapping_matrix).block_until_ready()

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)


class TestConstantRegularizationMatrixFrom:
    M: int = 1000
    N: int = 10
    coefficient: float = 1.0

    @pytest.fixture
    def setup_data(self):
        """Fixture to set up test data"""
        M = self.M
        N = self.N

        neighbors, neighbors_sizes = gen_neighbors(M, N)

        ref = original.constant_regularization_matrix_from(self.coefficient, neighbors, neighbors_sizes)

        return {
            "neighbors": neighbors,
            "neighbors_sizes": neighbors_sizes,
            "ref": ref,
        }

    @pytest.mark.benchmark(group="constant_regularization_matrix_from")
    def test_constant_regularization_matrix_from_original(self, setup_data, benchmark):
        """Benchmark the original constant_regularization_matrix_from function"""
        coefficient = self.coefficient
        neighbors = setup_data["neighbors"]
        neighbors_sizes = setup_data["neighbors_sizes"]

        ref = setup_data["ref"]

        def run():
            return original.constant_regularization_matrix_from(coefficient, neighbors, neighbors_sizes)

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="constant_regularization_matrix_from")
    def test_constant_regularization_matrix_from_numba(self, setup_data, benchmark):
        """Benchmark the numba constant_regularization_matrix_from function"""
        coefficient = self.coefficient
        neighbors = setup_data["neighbors"]
        neighbors_sizes = setup_data["neighbors_sizes"]

        ref = setup_data["ref"]

        def run():
            return numba.constant_regularization_matrix_from(coefficient, neighbors, neighbors_sizes)

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="constant_regularization_matrix_from")
    def test_constant_regularization_matrix_from_jax(self, setup_data, benchmark):
        """Benchmark the jax constant_regularization_matrix_from function"""
        coefficient = self.coefficient
        neighbors = setup_data["neighbors"]
        neighbors_sizes = setup_data["neighbors_sizes"]

        ref = setup_data["ref"]

        def run():
            return jax.constant_regularization_matrix_from(coefficient, neighbors, neighbors_sizes).block_until_ready()

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)


class TestReconstructionPositiveNegativeFrom:
    M = 1000

    @pytest.fixture
    def setup_data(self):
        """Fixture to set up test data"""
        M = self.M

        data_vector = gen_data_vector(M)
        curvature_matrix = create_M(M)
        inv = create_M_inv(M)

        ref = inv @ data_vector

        return {
            "data_vector": data_vector,
            "curvature_matrix": curvature_matrix,
            "ref": ref,
        }

    @pytest.mark.benchmark(group="reconstruction_positive_negative_from")
    def test_reconstruction_positive_negative_from_original(self, setup_data, benchmark):
        """Benchmark the original reconstruction_positive_negative_from function"""
        data_vector = setup_data["data_vector"]
        curvature_matrix = setup_data["curvature_matrix"]

        ref = setup_data["ref"]

        def run():
            return original.reconstruction_positive_negative_from(data_vector, curvature_matrix)

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="reconstruction_positive_negative_from")
    def test_reconstruction_positive_negative_from_numba(self, setup_data, benchmark):
        """Benchmark the numba reconstruction_positive_negative_from function"""
        data_vector = setup_data["data_vector"]
        curvature_matrix = setup_data["curvature_matrix"]

        ref = setup_data["ref"]

        def run():
            return numba.reconstruction_positive_negative_from(data_vector, curvature_matrix)

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="reconstruction_positive_negative_from")
    def test_reconstruction_positive_negative_from_jax(self, setup_data, benchmark):
        """Benchmark the jax reconstruction_positive_negative_from function"""
        data_vector = setup_data["data_vector"]
        curvature_matrix = setup_data["curvature_matrix"]

        ref = setup_data["ref"]

        def run():
            return jax.reconstruction_positive_negative_from(data_vector, curvature_matrix).block_until_ready()

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)


class TestNoiseNormalizationComplexFrom:
    M = 100

    @pytest.fixture
    def setup_data(self):
        """Fixture to set up test data"""
        M = self.M

        noise_map = gen_noise_map(M)

        ref = original.noise_normalization_complex_from(noise_map)

        return {
            "noise_map": noise_map,
            "ref": ref,
        }

    @pytest.mark.benchmark(group="noise_normalization_complex_from")
    def test_noise_normalization_complex_from_original(self, setup_data, benchmark):
        """Benchmark the original noise_normalization_complex_from function"""
        noise_map = setup_data["noise_map"]

        ref = setup_data["ref"]

        def run():
            return original.noise_normalization_complex_from(noise_map)

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="noise_normalization_complex_from")
    def test_noise_normalization_complex_from_numba(self, setup_data, benchmark):
        """Benchmark the numba noise_normalization_complex_from function"""
        noise_map = setup_data["noise_map"]

        ref = setup_data["ref"]

        def run():
            return numba.noise_normalization_complex_from(noise_map)

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="noise_normalization_complex_from")
    def test_noise_normalization_complex_from_jax(self, setup_data, benchmark):
        """Benchmark the jax noise_normalization_complex_from function"""
        noise_map = setup_data["noise_map"]

        ref = setup_data["ref"]

        def run():
            return jax.noise_normalization_complex_from(noise_map).block_until_ready()

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)


# log_likelihood_function
class TestLogLikelihoodFunction:
    M = 1000
    N = 100
    K = 200
    P = 10

    @pytest.fixture
    def setup_data(self):
        """Fixture to set up test data"""
        M = self.M
        self.N
        K = self.K
        P = self.P

        dirty_image = gen_dirty_image(M)
        w_tilde = gen_w_tilde(M)
        noise_map = gen_noise_map(K)
        mapping_matrix = gen_mapping_matrix(M)
        neighbors, neighbors_sizes = gen_neighbors(M, P)

        ref = original.log_likelihood_function(
            dirty_image,
            w_tilde,
            noise_map,
            mapping_matrix,
            neighbors,
            neighbors_sizes,
        )
        return {
            "dirty_image": dirty_image,
            "w_tilde": w_tilde,
            "noise_map": noise_map,
            "mapping_matrix": mapping_matrix,
            "neighbors": neighbors,
            "neighbors_sizes": neighbors_sizes,
            "ref": ref,
        }

    @pytest.mark.benchmark(group="log_likelihood_function")
    def test_log_likelihood_function_original(self, setup_data, benchmark):
        """Benchmark the original log_likelihood_function function"""
        dirty_image = setup_data["dirty_image"]
        w_tilde = setup_data["w_tilde"]
        noise_map = setup_data["noise_map"]
        mapping_matrix = setup_data["mapping_matrix"]
        neighbors = setup_data["neighbors"]
        neighbors_sizes = setup_data["neighbors_sizes"]

        ref = setup_data["ref"]

        def run():
            return original.log_likelihood_function(
                dirty_image,
                w_tilde,
                noise_map,
                mapping_matrix,
                neighbors,
                neighbors_sizes,
            )

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="log_likelihood_function")
    def test_log_likelihood_function_numba(self, setup_data, benchmark):
        """Benchmark the numba log_likelihood_function function"""
        dirty_image = setup_data["dirty_image"]
        w_tilde = setup_data["w_tilde"]
        noise_map = setup_data["noise_map"]
        mapping_matrix = setup_data["mapping_matrix"]
        neighbors = setup_data["neighbors"]
        neighbors_sizes = setup_data["neighbors_sizes"]

        ref = setup_data["ref"]

        def run():
            return numba.log_likelihood_function(
                dirty_image,
                w_tilde,
                noise_map,
                mapping_matrix,
                neighbors,
                neighbors_sizes,
            )

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="log_likelihood_function")
    def test_log_likelihood_function_jax(self, setup_data, benchmark):
        """Benchmark the jax log_likelihood_function function"""
        dirty_image = setup_data["dirty_image"]
        w_tilde = setup_data["w_tilde"]
        noise_map = setup_data["noise_map"]
        mapping_matrix = setup_data["mapping_matrix"]
        neighbors = setup_data["neighbors"]
        neighbors_sizes = setup_data["neighbors_sizes"]

        ref = setup_data["ref"]

        def run():
            return jax.log_likelihood_function(
                dirty_image,
                w_tilde,
                noise_map,
                mapping_matrix,
                neighbors,
                neighbors_sizes,
            ).block_until_ready()

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)
