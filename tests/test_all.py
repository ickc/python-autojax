from __future__ import annotations

import numpy as np
import pytest

from autojax.original import w_tilde_data_interferometer_from
from autojax.numba import w_tilde_data_interferometer_from as w_tilde_data_interferometer_from_numba
from autojax.jax import w_tilde_data_interferometer_from as w_tilde_data_interferometer_from_jax


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

        ref = w_tilde_data_interferometer_from(
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

        result = w_tilde_data_interferometer_from_numba(
            visibilities_real,
            noise_map_real,
            uv_wavelengths,
            grid_radians_slim,
            native_index_for_slim_index,
        )
        np.testing.assert_array_almost_equal(result, ref)

    def test_w_tilde_data_interferometer_from_jax(self, setup_data):
        """Test the w_tilde_data_interferometer_from function"""
        visibilities_real = setup_data["visibilities_real"]
        noise_map_real = setup_data["noise_map_real"]
        uv_wavelengths = setup_data["uv_wavelengths"]
        grid_radians_slim = setup_data["grid_radians_slim"]
        native_index_for_slim_index = setup_data["native_index_for_slim_index"]

        ref = setup_data["ref"]

        result = w_tilde_data_interferometer_from_jax(
            visibilities_real,
            noise_map_real,
            uv_wavelengths,
            grid_radians_slim,
            native_index_for_slim_index,
        )
        np.testing.assert_allclose(result, ref)
