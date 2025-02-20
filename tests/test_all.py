from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pytest
from jax import numpy as jnp

from autojax import jax, numba, original

tests_generated: list[str] = [
    # "mask_2d_centres_from",
    "mask_2d_circular_from",
    "w_tilde_data_interferometer_from",
    "w_tilde_curvature_interferometer_from",
    # "w_tilde_curvature_preload_interferometer_from",
    # "w_tilde_via_preload_from",
    "data_vector_from",
    "curvature_matrix_via_w_tilde_from",
    # "curvature_matrix_via_w_tilde_curvature_preload_interferometer_from",
    "constant_regularization_matrix_from",
    "reconstruction_positive_negative_from",
    "noise_normalization_complex_from",
    "log_likelihood_function",
    # "log_likelihood_function_via_preload_method",
]
tests_loaded: list[str] = tests_generated + [
    "w_tilde_via_preload_from",
]


tests_all: list[str] = [
    # "mask_2d_centres_from",
    "mask_2d_circular_from",
    "w_tilde_data_interferometer_from",
    "w_tilde_curvature_interferometer_from",
    "w_tilde_curvature_preload_interferometer_from",
    "w_tilde_via_preload_from",
    "data_vector_from",
    "curvature_matrix_via_w_tilde_from",
    "curvature_matrix_via_w_tilde_curvature_preload_interferometer_from",
    "constant_regularization_matrix_from",
    "reconstruction_positive_negative_from",
    "noise_normalization_complex_from",
    "log_likelihood_function",
    "log_likelihood_function_via_preload_method",
]


def deterministic_seed(string: str, *numbers: int) -> int:
    """Generate a deterministic seed from the class name."""
    hash_value = hashlib.md5(repr((string, numbers)).encode()).hexdigest()  # Get a hash from class name
    return int(hash_value, 16) % (2**32)  # Convert to an integer within a reasonable range


@dataclass
class Data:
    """Test data."""

    _pixel_scales: float = 0.2
    _centre: float = 0.0
    radius: float = 3.0
    coefficient: float = 1.0

    def dict(self) -> dict:
        return {
            "radius": self.radius,
            "coefficient": self.coefficient,
            "shape_native": self.shape_native,
            "pixel_scales": self.pixel_scales,
            "centre": self.centre,
            "mapping_matrix": self.mapping_matrix,
            "dirty_image": self.dirty_image,
            "visibilities_real": self.visibilities_real,
            "uv_wavelengths": self.uv_wavelengths,
            "grid_radians_2d": self.grid_radians_2d,
            "grid_radians_slim": self.grid_radians_slim,
            "native_index_for_slim_index": self.native_index_for_slim_index,
            "w_tilde": self.w_tilde,
            "neighbors_sizes": self.neighbors_sizes,
            "neighbors": self.neighbors,
            "data_vector": self.data_vector,
            "data": self.data,
            "noise_map": self.noise_map,
            "noise_map_real": self.noise_map_real,
            "curvature_reg_matrix": self.curvature_reg_matrix,
        }

    @property
    def M(self) -> int:
        return self.grid_radians_slim.shape[0]

    @property
    def N(self) -> int:
        raise NotImplementedError
    
    @property
    def N_PRIME(self) -> int:
        raise NotImplementedError

    @property
    def K(self) -> int:
        raise NotImplementedError

    @property
    def P(self) -> int:
        raise NotImplementedError

    @property
    def S(self) -> int:
        raise NotImplementedError

    @property
    def pix_pixels(self) -> int:
        return self.S

    @property
    def shape_native(self) -> tuple[int, int]:
        """Get the shape of the native grid."""
        return self.M, self.M

    @property
    def shape_masked_pixels_2d(self) -> tuple[int, int]:
        """Get the shape of the masked grid."""
        return self.N, self.N

    @property
    def pixel_scales(self) -> tuple[float, float]:
        """Get the pixel scales of the native grid."""
        return self._pixel_scales, self._pixel_scales

    @property
    def centre(self) -> tuple[float, float]:
        """Get the centre of the native grid."""
        return self._centre, self._centre

    @property
    def mapping_matrix(self) -> np.ndarray[tuple[int, int], np.float64]:
        raise NotImplementedError

    @property
    def dirty_image(self) -> np.ndarray[tuple[int], np.float64]:
        raise NotImplementedError

    @property
    def uv_wavelengths(self) -> np.ndarray[tuple[int, int], np.float64]:
        raise NotImplementedError

    @cached_property
    def real_space_mask(self) -> np.ndarray[tuple[int, int], np.bool]:
        return original.mask_2d_circular_from((self.N_PRIME, self.N_PRIME), self.pixel_scales, self.radius, self.centre)

    @cached_property
    def native_index_for_slim_index(self) -> np.ndarray[tuple[int, int], np.int64]:
        return np.ascontiguousarray(np.argwhere(~self.real_space_mask))

    @cached_property
    def grid_radians_2d(self) -> np.ndarray[tuple[int, int, int], np.float64]:
        N = self.N_PRIME
        arcsec = np.pi / 648000
        m = self._pixel_scales * arcsec
        c = 9.9 * arcsec  # hard-coded to match the dataset
        x = np.mgrid[:N, :N]
        res = np.empty((N, N, 2))
        res[:, :, 0] = -m * x[0] + c
        res[:, :, 1] = m * x[1] - c
        return res

    @cached_property
    def grid_radians_slim(self) -> np.ndarray[tuple[int, int], np.float64]:
        return self.grid_radians_2d[~self.real_space_mask]

    @property
    def pix_indexes_for_sub_slim_index(self) -> np.ndarray[tuple[int, int], np.int64]:
        raise NotImplementedError

    @property
    def pix_size_for_sub_slim_index(self) -> np.ndarray[tuple[int], np.int64]:
        raise NotImplementedError

    @property
    def pix_weights_for_sub_slim_index(self) -> np.ndarray[tuple[int, int], np.float64]:
        raise NotImplementedError

    @property
    def neighbors_sizes(self) -> np.ndarray[tuple[int], np.int64]:
        raise NotImplementedError

    @property
    def neighbors(self) -> np.ndarray[tuple[int], np.int64]:
        raise NotImplementedError

    @property
    def data(self) -> np.ndarray[tuple[int], np.complex128]:
        raise NotImplementedError

    @property
    def noise_map(self) -> np.ndarray[tuple[int], np.complex128]:
        raise NotImplementedError

    # calculated
    @cached_property
    def visibilities_real(self) -> np.ndarray[tuple[int], np.float64]:
        return np.ascontiguousarray(self.data.real)

    @cached_property
    def noise_map_real(self) -> np.ndarray[tuple[int], np.float64]:
        return np.ascontiguousarray(self.noise_map.real)

    @cached_property
    def data_vector(self) -> np.ndarray[tuple[int], np.float64]:
        return original.data_vector_from(self.mapping_matrix, self.dirty_image)

    @cached_property
    def w_tilde(self) -> np.ndarray[tuple[int, int], np.float64]:
        return original.w_tilde_curvature_interferometer_from(
            self.noise_map_real,
            self.uv_wavelengths,
            self.grid_radians_slim,
        )

    @cached_property
    def w_tilde_preload(self) -> np.ndarray[tuple[int, int], np.float64]:
        return original.w_tilde_curvature_preload_interferometer_from(
            self.noise_map_real,
            self.uv_wavelengths,
            self.shape_masked_pixels_2d,
            self.grid_radians_2d,
        )

    @cached_property
    def curvature_matrix(self) -> np.ndarray[tuple[int, int], np.float64]:
        return original.curvature_matrix_via_w_tilde_from(self.w_tilde, self.mapping_matrix)

    @cached_property
    def regularization_matrix(self) -> np.ndarray[tuple[int, int], np.float64]:
        return original.constant_regularization_matrix_from(
            self.coefficient,
            self.neighbors,
            self.neighbors_sizes,
        )

    @cached_property
    def curvature_reg_matrix(self) -> np.ndarray[tuple[int, int], np.float64]:
        return self.curvature_matrix + self.regularization_matrix


@dataclass
class DataLoaded(Data):
    """Load data from file."""

    path: Path = Path(__file__).parent / "data.npz"

    def __post_init__(self):
        self._data = np.load(self.path)

    def dict(self) -> dict:
        res = super().dict()
        return res | {
            "pix_pixels": self.pix_pixels,
            "shape_masked_pixels_2d": self.shape_masked_pixels_2d,
            "w_tilde_preload": self.w_tilde_preload,
            "pix_indexes_for_sub_slim_index": self.pix_indexes_for_sub_slim_index,
            "pix_size_for_sub_slim_index": self.pix_size_for_sub_slim_index,
            "pix_weights_for_sub_slim_index": self.pix_weights_for_sub_slim_index,
        }

    @property
    def M(self) -> int:
        return self.mapping_matrix.shape[0]

    @property
    def N(self) -> int:
        return 30  # hard-coded for this particular dataset
    
    @property
    def N_PRIME(self) -> int:
        return 100  # hard-coded for this particular dataset

    @property
    def K(self) -> int:
        return self.uv_wavelengths.shape[0]

    @property
    def P(self) -> int:
        return self.neighbors.shape[1]

    @property
    def S(self) -> int:
        return self.mapping_matrix.shape[1]

    @property
    def dirty_image(self):
        return self._data["dirty_image"]

    @property
    def data(self):
        return self._data["data"]

    @property
    def noise_map(self):
        return self._data["noise_map"]

    @property
    def uv_wavelengths(self):
        return self._data["uv_wavelengths"]

    @property
    def mapping_matrix(self):
        return self._data["mapping_matrix"]

    @property
    def neighbors(self):
        return self._data["neighbors"]

    @property
    def neighbors_sizes(self):
        return self._data["neighbors_sizes"]

    @property
    def pix_indexes_for_sub_slim_index(self):
        return self._data["pix_indexes_for_sub_slim_index"]

    @property
    def pix_size_for_sub_slim_index(self):
        return self._data["pix_size_for_sub_slim_index"]

    @property
    def pix_weights_for_sub_slim_index(self):
        return self._data["pix_weights_for_sub_slim_index"]


@dataclass
class DataGenerated(Data):
    """Generate data for testing."""

    N_: int = 30
    N_PRIME_: int = 100
    K_: int = 1024
    P_: int = 32
    S_: int = 256

    @property
    def N(self) -> int:
        return self.N_
    
    @property
    def N_PRIME(self) -> int:
        return self.N_PRIME_

    @property
    def K(self) -> int:
        return self.K_

    @property
    def P(self) -> int:
        return self.P_

    @property
    def S(self) -> int:
        return self.S_

    # random
    @cached_property
    def mapping_matrix(self) -> np.ndarray[tuple[int, int], np.float64]:
        """Generate a mapping matrix."""
        M = self.M
        S = self.S
        mapping_matrix = np.zeros((M, S))
        # make up some sparse mapping matrix, non-zero values are close to the scaled diagonal
        R = 0.01
        for i in range(M):
            for j in range(S):
                r = np.abs((i + 1) / M - (j + 1) / S)
                if r < R:
                    mapping_matrix[i, j] = R - r
        # normalize
        mapping_matrix /= mapping_matrix.sum(axis=1).reshape(-1, 1)
        return mapping_matrix

    @cached_property
    def dirty_image(self) -> np.ndarray[tuple[int], np.float64]:
        """Generate a random dirty image."""
        M = self.M
        rng = np.random.default_rng(deterministic_seed("dirty_image", M))
        return rng.random(M)

    @cached_property
    def uv_wavelengths(self) -> np.ndarray[tuple[int, int], np.float64]:
        """Generate random uv wavelengths."""
        K = self.K
        rng = np.random.default_rng(deterministic_seed("uv_wavelengths", K, 2))
        return rng.random((K, 2))

    # def pix_indexes_for_sub_slim_index
    # def pix_size_for_sub_slim_index
    # def pix_weights_for_sub_slim_index


    @cached_property
    def neighbors_sizes(self) -> np.ndarray[tuple[int], np.int64]:
        """Generate random neighbors_sizes."""
        S = self.S
        P = self.P
        rng = np.random.default_rng(deterministic_seed("neighbors", S, P))
        return rng.integers(0, P + 1, S)

    @cached_property
    def neighbors(self) -> np.ndarray[tuple[int], np.int64]:
        """Generate random neighbors."""
        S = self.S
        P = self.P
        neighbors_sizes = self.neighbors_sizes
        rng = np.random.default_rng(deterministic_seed("neighbors", S, P))
        neighbors = np.full((S, P), -1, dtype=np.int64)
        for i in range(S):
            neighbors[i, : neighbors_sizes[i]] = np.sort(
                rng.choice(
                    S,
                    neighbors_sizes[i],
                    replace=False,
                )
            )
        return neighbors

    @cached_property
    def data(self) -> np.ndarray[tuple[int], np.complex128]:
        """Generate random data map."""
        K = self.K
        rng = np.random.default_rng(deterministic_seed("data", K))
        return rng.random(2 * K).view(np.complex128)

    @cached_property
    def noise_map(self) -> np.ndarray[tuple[int], np.complex128]:
        """Generate random noise map."""
        K = self.K
        rng = np.random.default_rng(deterministic_seed("noise_map", K))
        return rng.random(2 * K).view(np.complex128)


@dataclass
class Reference:
    """Generate reference values for testing."""

    data: Data
    tests: tuple[str, ...]
    mod = original

    @cached_property
    def ref(self) -> dict:
        data_dict = self.data.dict()

        res = {}
        for test in self.tests:
            func = getattr(self.mod, test)
            sig = inspect.signature(func)
            args = [data_dict[key] for key in sig.parameters]
            res[test] = func(*args)
        return res


@pytest.fixture(
    params=(DataLoaded, DataGenerated),
    scope="module",
)
def data_bundle(request):
    Data = request.param
    data = Data()
    tests = tests_generated if Data is DataGenerated else tests_loaded
    ref = Reference(data, tests)
    return data, ref


class AutoTestMeta(type):
    """Metaclass to auto-generate pytest-benchmark tests"""

    def __new__(cls, name, bases, namespace):
        new_cls = super().__new__(cls, name, bases, namespace)

        def create_test(test: str):
            if new_cls.mode == "benchmark":

                @pytest.mark.benchmark
                def test_method(self, data_bundle, benchmark):
                    data, ref = data_bundle
                    if test not in ref.tests:
                        pytest.skip(f"Skip {test} from {type(data).__name__}")

                    benchmark.group = f"{test}_{type(data).__name__}"

                    data_dict = data.dict()
                    if new_cls.mod == jax:
                        data_dict = {k: jnp.array(v) if isinstance(v, np.ndarray) else v for k, v in data_dict.items()}
                    ref_dict = ref.ref
                    func = getattr(self.mod, test)
                    sig = inspect.signature(func)
                    args = [data_dict[key] for key in sig.parameters]
                    run = (lambda: func(*args).block_until_ready()) if new_cls.mod == jax else (lambda: func(*args))
                    res = benchmark(run)
                    np.testing.assert_allclose(res, ref_dict[test], rtol=2e-6)

            else:

                @pytest.mark.unittest
                def test_method(self, data_bundle):
                    data, ref = data_bundle
                    if test not in ref.tests:
                        pytest.skip(f"Skip {test} from {type(data).__name__}")

                    data_dict = data.dict()
                    ref_dict = ref.ref
                    func = getattr(self.mod, test)
                    sig = inspect.signature(func)
                    args = [data_dict[key] for key in sig.parameters]
                    run = (lambda: func(*args).block_until_ready()) if new_cls.mod == jax else (lambda: func(*args))
                    res = run()
                    np.testing.assert_allclose(res, ref_dict[test], rtol=2e-6)

            test_method.__name__ = f"test_{test}_{new_cls.mod.__name__.split(".")[-1]}"
            return test_method

        for test_name in tests_all:
            method = create_test(test_name)
            setattr(new_cls, method.__name__, method)

        return new_cls


# Example usage in a TestCase
class TestNumba(metaclass=AutoTestMeta):
    mod = numba
    mode = "unittest"


class TestJax(metaclass=AutoTestMeta):
    mod = jax
    mode = "unittest"


class TestBenchOriginal(metaclass=AutoTestMeta):
    mod = original
    mode = "benchmark"


class TestBenchNumba(metaclass=AutoTestMeta):
    mod = numba
    mode = "benchmark"


class TestBenchJax(metaclass=AutoTestMeta):
    mod = jax
    mode = "benchmark"


# special case


@pytest.fixture(scope="module")
def data_loaded():
    return DataLoaded()


class TestLogLikelihood:

    ref: float = -13401.986947103405

    @pytest.mark.benchmark(group="test_log_likelihood_data_loaded")
    def test_original(self, data_loaded, benchmark):
        data_dict = data_loaded.dict()
        func = original.log_likelihood_function
        sig = inspect.signature(func)
        args = [data_dict[key] for key in sig.parameters]

        def run():
            return func(*args)

        res = benchmark(run)
        np.testing.assert_allclose(res, self.ref)

    @pytest.mark.benchmark(group="test_log_likelihood_data_loaded")
    def test_numba(self, data_loaded, benchmark):
        data_dict = data_loaded.dict()
        func = numba.log_likelihood_function
        sig = inspect.signature(func)
        args = [data_dict[key] for key in sig.parameters]

        def run():
            return func(*args)

        res = benchmark(run)
        np.testing.assert_allclose(res, self.ref)

    @pytest.mark.benchmark(group="test_log_likelihood_data_loaded")
    def test_jax(self, data_loaded, benchmark):
        data_dict = data_loaded.dict()
        data_dict = {k: jnp.array(v) if isinstance(v, np.ndarray) else v for k, v in data_dict.items()}
        func = jax.log_likelihood_function
        sig = inspect.signature(func)
        args = [data_dict[key] for key in sig.parameters]

        def run():
            return func(*args).block_until_ready()

        res = benchmark(run)
        np.testing.assert_allclose(res, self.ref)

    @pytest.mark.benchmark(group="test_log_likelihood_data_loaded")
    def test_via_preload_method_original(self, data_loaded, benchmark):
        data_dict = data_loaded.dict()
        func = original.log_likelihood_function_via_preload_method
        sig = inspect.signature(func)
        args = [data_dict[key] for key in sig.parameters]

        def run():
            return func(*args)

        res = benchmark(run)
        np.testing.assert_allclose(res, self.ref)
