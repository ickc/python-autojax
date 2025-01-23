from __future__ import annotations

import numpy as np
import pytest

from autojax import original, numba, jax


class TestWTildeDataInterferometer:
    
    @pytest.fixture
    def setup_data(self):
        """Fixture to set up test data"""
        np.random.seed(20250101)  # For reproducibility

        N = 1000
        M = 1000
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

    def test_w_tilde_data_interferometer_from_numba(self, setup_data):

        """Test the w_tilde_data_interferometer_from function"""
        visibilities_real = setup_data["visibilities_real"]
        noise_map_real = setup_data["noise_map_real"]
        uv_wavelengths = setup_data["uv_wavelengths"]
        grid_radians_slim = setup_data["grid_radians_slim"]
        native_index_for_slim_index = setup_data["native_index_for_slim_index"]

        ref = setup_data["ref"]

        result = numba.w_tilde_data_interferometer_from(
            visibilities_real,
            noise_map_real,
            uv_wavelengths,
            grid_radians_slim,
            native_index_for_slim_index,
        )
        np.testing.assert_allclose(result, ref)

    def test_w_tilde_data_interferometer_from_jax(self, setup_data):
        """Test the w_tilde_data_interferometer_from function"""
        visibilities_real = setup_data["visibilities_real"]
        noise_map_real = setup_data["noise_map_real"]
        uv_wavelengths = setup_data["uv_wavelengths"]
        grid_radians_slim = setup_data["grid_radians_slim"]
        native_index_for_slim_index = setup_data["native_index_for_slim_index"]

        ref = setup_data["ref"]

        result = jax.w_tilde_data_interferometer_from(
            visibilities_real,
            noise_map_real,
            uv_wavelengths,
            grid_radians_slim,
            native_index_for_slim_index,
        )
        np.testing.assert_allclose(result, ref)

class TestWTildeCurvatureInterferometer:

    @pytest.fixture
    def setup_data(self):
        """Fixture to set up test data"""
        np.random.seed(20250101)  # For reproducibility

        N = 1000
        M = 1000
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

    def test_w_tilde_curvature_interferometer_from_numba(self, setup_data):
        """Test the w_tilde_curvature_interferometer_from function"""
        noise_map_real = setup_data["noise_map_real"]
        uv_wavelengths = setup_data["uv_wavelengths"]
        grid_radians_slim = setup_data["grid_radians_slim"]

        ref = setup_data["ref"]

        result = numba.w_tilde_curvature_interferometer_from(
            noise_map_real,
            uv_wavelengths,
            grid_radians_slim,
        )
        np.testing.assert_allclose(result, ref)

    def test_w_tilde_curvature_interferometer_from_jax(self, setup_data):
        """Test the w_tilde_curvature_interferometer_from function"""
        noise_map_real = setup_data["noise_map_real"]
        uv_wavelengths = setup_data["uv_wavelengths"]
        grid_radians_slim = setup_data["grid_radians_slim"]

        ref = setup_data["ref"]

        result = jax.w_tilde_curvature_interferometer_from(
            noise_map_real,
            uv_wavelengths,
            grid_radians_slim,
        )
        np.testing.assert_allclose(result, ref)
