from __future__ import annotations

import numpy as np
import pytest

from autojax import original, numba, jax


class TestWTildeDataInterferometer:
    M = 1000
    N = 1000

    @pytest.fixture
    def setup_data(self):
        """Fixture to set up test data"""
        np.random.seed(20250101)  # For reproducibility
        M = self.M
        N = self.N

        visibilities_real = np.random.normal(size=N)
        noise_map_real = np.random.normal(size=N)
        uv_wavelengths = np.random.normal(size=(N, 2))
        grid_radians_slim = np.random.normal(size=(M, 2))
        native_index_for_slim_index = np.random.randint(0, M, size=(M, 2))

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
                native_index_for_slim_index
            ).block_until_ready()

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)


class TestWTildeCurvatureInterferometer:
    M = 200
    N = 100

    @pytest.fixture
    def setup_data(self):
        """Fixture to set up test data"""
        np.random.seed(20250101)  # For reproducibility
        M = self.M
        N = self.N

        noise_map_real = np.random.rand(N)
        uv_wavelengths = np.random.rand(N, 2)
        grid_radians_slim = np.random.rand(M, 2)

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
                grid_radians_slim
            ).block_until_ready()

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)


class TestDataVectorFrom:
    N: int = 1000

    @pytest.fixture
    def setup_data(self):
        """Fixture to set up test data"""
        np.random.seed(20250101)
        N = self.N

        mapping_matrix = np.random.rand(N, N)
        dirty_image = np.random.rand(N)

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
        np.random.seed(20250101)
        N = self.N

        w_tilde = np.random.rand(N, N)
        mapping_matrix = np.random.rand(N, N)

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

    @pytest.fixture
    def setup_data(self):
        """Fixture to set up test data"""
        np.random.seed(20250101)
        M = self.M
        N = self.N

        coefficient = np.random.rand()
        neighbors_sizes = np.random.randint(0, N + 1, M)
        neighbors = np.full((M, N), -1, dtype=np.int64)
        for i in range(M):
            neighbors[i, :neighbors_sizes[i]] = np.sort(np.random.choice(M, neighbors_sizes[i], replace=False))

        ref = original.constant_regularization_matrix_from(
            coefficient, neighbors, neighbors_sizes
        )

        return {
            "coefficient": coefficient,
            "neighbors": neighbors,
            "neighbors_sizes": neighbors_sizes,
            "ref": ref,
        }

    @pytest.mark.benchmark(group="constant_regularization_matrix_from")
    def test_constant_regularization_matrix_from_original(self, setup_data, benchmark):
        """Benchmark the original constant_regularization_matrix_from function"""
        coefficient = setup_data["coefficient"]
        neighbors = setup_data["neighbors"]
        neighbors_sizes = setup_data["neighbors_sizes"]

        ref = setup_data["ref"]

        def run():
            return original.constant_regularization_matrix_from(
                coefficient, neighbors, neighbors_sizes
            )

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="constant_regularization_matrix_from")
    def test_constant_regularization_matrix_from_numba(self, setup_data, benchmark):
        """Benchmark the numba constant_regularization_matrix_from function"""
        coefficient = setup_data["coefficient"]
        neighbors = setup_data["neighbors"]
        neighbors_sizes = setup_data["neighbors_sizes"]

        ref = setup_data["ref"]

        def run():
            return numba.constant_regularization_matrix_from(
                coefficient, neighbors, neighbors_sizes
            )

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)

    @pytest.mark.benchmark(group="constant_regularization_matrix_from")
    def test_constant_regularization_matrix_from_jax(self, setup_data, benchmark):
        """Benchmark the jax constant_regularization_matrix_from function"""
        coefficient = setup_data["coefficient"]
        neighbors = setup_data["neighbors"]
        neighbors_sizes = setup_data["neighbors_sizes"]

        ref = setup_data["ref"]

        def run():
            return jax.constant_regularization_matrix_from(
                coefficient, neighbors, neighbors_sizes
            ).block_until_ready()

        result = benchmark(run)
        np.testing.assert_allclose(result, ref)
