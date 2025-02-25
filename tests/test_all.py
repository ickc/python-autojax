from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pytest
from jax import numpy as jnp
from jax.experimental import sparse
from numba import jit

from autojax import jax, numba, original

tests_generated: list[str] = [
    # "mask_2d_centres_from",
    "mask_2d_circular_from",
    "w_tilde_data_interferometer_from",
    "w_tilde_curvature_interferometer_from",
    # "w_tilde_curvature_preload_interferometer_from",
    "w_tilde_via_preload_from",
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
    N_: int = 30
    coefficient: float = 1.0

    def dict(self) -> dict:
        return {
            "radius": self.radius,
            "coefficient": self.coefficient,
            "shape_native": self.shape_native,
            "shape_masked_pixels_2d": self.shape_masked_pixels_2d,
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
            "w_tilde_preload": self.w_tilde_preload,
            "curvature_preload": self.w_tilde_preload,
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
        return self.N_

    @property
    def radius(self) -> float:
        return ((self.N_ + 1) // 2) * self._pixel_scales

    @property
    def N_PRIME(self) -> int:
        return self.N_

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
        d = self._pixel_scales * arcsec
        g_000 = 9.9 * arcsec  # hard-coded to match the dataset
        g_001 = -g_000
        I, J = np.mgrid[:N, :N]
        g = np.empty((N, N, 2))
        g[:, :, 0] = -d * I + g_000
        g[:, :, 1] = d * J + g_001
        return g

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

    @cached_property
    def neighbors_grid(self):
        """Convert a neighbors array to a grid, primarily for visualization."""
        neighbors = self.neighbors
        S, P = neighbors.shape
        grid = np.zeros((S, S), dtype=bool)
        for i in range(S):
            for p in range(P):
                j = neighbors[i, p]
                if j != -1:
                    grid[i, j] = True
                    grid[j, i] = True
        return grid

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
            "pix_indexes_for_sub_slim_index": self.pix_indexes_for_sub_slim_index,
            "pix_size_for_sub_slim_index": self.pix_size_for_sub_slim_index,
            "pix_weights_for_sub_slim_index": self.pix_weights_for_sub_slim_index,
        }

    @property
    def M(self) -> int:
        return self.mapping_matrix.shape[0]

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

    K_: int = 1024
    P_: int = 32
    S_: int = 256

    @property
    def K(self) -> int:
        return self.K_

    @property
    def P(self) -> int:
        return self.P_

    @property
    def S(self) -> int:
        return self.S_

    @staticmethod
    @jit("float64[:, :](int64, int64)", nopython=True, nogil=True, parallel=False)
    def _gen_mapping_matrix(M: int, S: int) -> np.ndarray[tuple[int, int], np.float64]:
        """Generate a mapping matrix."""
        mapping_matrix = np.zeros((M, S))
        # make up some sparse mapping matrix, non-zero values are close to the scaled diagonal
        R = max(0.0018521191598264723, 2.0 * np.abs(1.0 / M - 1.0 / S))
        for i in range(M):
            for j in range(S):
                r = np.abs((i + 1) / M - (j + 1) / S)
                if r < R:
                    mapping_matrix[i, j] = R - r
        # normalize
        mapping_matrix /= mapping_matrix.sum(axis=1).reshape(-1, 1)
        return mapping_matrix

    @cached_property
    def mapping_matrix(self) -> np.ndarray[tuple[int, int], np.float64]:
        """Generate a mapping matrix."""
        return self._gen_mapping_matrix(self.M, self.S)

    # random
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
        neighbors = self.neighbors
        return (neighbors != -1).sum(axis=1)

    @cached_property
    def neighbors(self) -> np.ndarray[tuple[int, int], np.int64]:
        """Generate random neighbors."""
        S = self.S
        P = self.P
        rng = np.random.default_rng(deterministic_seed("neighbors", S, P))
        if S <= 0 or P <= 0:
            raise ValueError("S and P must be positive integers.")
        if (S * P) % 2 != 0:
            raise ValueError("S*P must be even to generate a valid neighbors array.")

        # Generate the list of stubs
        stubs = []
        for s in range(S):
            stubs.extend([s] * P)

        # Shuffle the stubs to randomize connections
        rng.shuffle(stubs)

        # Initialize neighbor lists for each node
        neighbors = [[] for _ in range(S)]

        # Pair the stubs and build the neighbor lists
        for i in range(0, len(stubs), 2):
            a = stubs[i]
            b = stubs[i + 1]
            neighbors[a].append(b)
            neighbors[b].append(a)

        # remove self-loop
        for i, neighbor in enumerate(neighbors):
            if i in neighbor:
                neighbor.remove(i)
                neighbor.append(-1)
        # Convert to a numpy array of integers
        return np.array(neighbors, dtype=int)

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

            test_method.__name__ = f"test_{test}_{new_cls.mod.__name__.split('.')[-1]}"
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


class TestWTilde:
    """Compute w_tilde via various methods.

    This adds on top of existing benchmarks to compare the performance of the preload method.

    The test names are a bit strange, but is designed to be filtered like this:

        pytest -m benchmark -k w_tilde_curvature_interferometer_from
    """

    @pytest.mark.benchmark
    def test_w_tilde_curvature_interferometer_from_preload_original(self, data_bundle, benchmark):
        data, ref = data_bundle
        data_dict = data.dict()

        test = "w_tilde_curvature_interferometer_from"
        benchmark.group = f"{test}_{type(data).__name__}"

        noise_map_real = data_dict["noise_map_real"]
        uv_wavelengths = data_dict["uv_wavelengths"]
        shape_masked_pixels_2d = data_dict["shape_masked_pixels_2d"]
        grid_radians_2d = data_dict["grid_radians_2d"]
        native_index_for_slim_index = data_dict["native_index_for_slim_index"]

        def run():
            w_preload = original.w_tilde_curvature_preload_interferometer_from(
                noise_map_real,
                uv_wavelengths,
                shape_masked_pixels_2d,
                grid_radians_2d,
            )
            return original.w_tilde_via_preload_from(w_preload, native_index_for_slim_index)

        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["w_tilde_curvature_interferometer_from"])

    @pytest.mark.benchmark
    def test_w_tilde_curvature_interferometer_from_compact_numba(self, data_bundle, benchmark):
        data, ref = data_bundle
        data_dict = data.dict()

        test = "w_tilde_curvature_interferometer_from"
        benchmark.group = f"{test}_{type(data).__name__}"

        noise_map_real = data_dict["noise_map_real"]
        uv_wavelengths = data_dict["uv_wavelengths"]
        grid_radians_2d = data_dict["grid_radians_2d"]
        native_index_for_slim_index = data_dict["native_index_for_slim_index"]
        pixel_scale = data._pixel_scales

        def run():
            w_compact = numba.w_tilde_curvature_compact_interferometer_from(
                noise_map_real,
                uv_wavelengths,
                pixel_scale,
                grid_radians_2d,
            )
            return numba.w_tilde_via_compact_from(w_compact, native_index_for_slim_index)

        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["w_tilde_curvature_interferometer_from"])

    @pytest.mark.benchmark
    def test_w_tilde_curvature_interferometer_from_compact_jax(self, data_bundle, benchmark):
        data, ref = data_bundle
        data_dict = data.dict()

        test = "w_tilde_curvature_interferometer_from"
        benchmark.group = f"{test}_{type(data).__name__}"

        noise_map_real = jnp.array(data_dict["noise_map_real"])
        uv_wavelengths = jnp.array(data_dict["uv_wavelengths"])
        grid_radians_2d = jnp.array(data_dict["grid_radians_2d"])
        native_index_for_slim_index = jnp.array(data_dict["native_index_for_slim_index"])
        pixel_scale = data._pixel_scales

        def run():
            w_compact = jax.w_tilde_curvature_compact_interferometer_from(
                noise_map_real,
                uv_wavelengths,
                pixel_scale,
                grid_radians_2d,
            )
            return jax.w_tilde_via_compact_from(w_compact, native_index_for_slim_index).block_until_ready()

        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["w_tilde_curvature_interferometer_from"])


class TestCurvatureMatrix:
    """Compute curvature matrix via various methods."""

    @pytest.mark.benchmark
    def test_curvature_matrix_original(self, data_bundle, benchmark):
        data, ref = data_bundle
        data_dict = data.dict()

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        noise_map_real = data_dict["noise_map_real"]
        uv_wavelengths = data_dict["uv_wavelengths"]
        grid_radians_slim = data_dict["grid_radians_slim"]
        mapping_matrix = data_dict["mapping_matrix"]

        def run():
            w_tilde = original.w_tilde_curvature_interferometer_from(
                noise_map_real,
                uv_wavelengths,
                grid_radians_slim,
            )
            curvature_matrix = original.curvature_matrix_via_w_tilde_from(
                w_tilde,
                mapping_matrix,
            )
            return curvature_matrix

        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"])

    @pytest.mark.benchmark
    def test_curvature_matrix_numba(self, data_bundle, benchmark):
        data, ref = data_bundle
        data_dict = data.dict()

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        noise_map_real = data_dict["noise_map_real"]
        uv_wavelengths = data_dict["uv_wavelengths"]
        grid_radians_slim = data_dict["grid_radians_slim"]
        mapping_matrix = data_dict["mapping_matrix"]

        def run():
            w_tilde = numba.w_tilde_curvature_interferometer_from(
                noise_map_real,
                uv_wavelengths,
                grid_radians_slim,
            )
            curvature_matrix = numba.curvature_matrix_via_w_tilde_from(
                w_tilde,
                mapping_matrix,
            )
            return curvature_matrix

        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"])

    @pytest.mark.benchmark
    def test_curvature_matrix_jax(self, data_bundle, benchmark):
        data, ref = data_bundle
        data_dict = data.dict()

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        noise_map_real = jnp.array(data_dict["noise_map_real"])
        uv_wavelengths = jnp.array(data_dict["uv_wavelengths"])
        grid_radians_slim = jnp.array(data_dict["grid_radians_slim"])
        mapping_matrix = jnp.array(data_dict["mapping_matrix"])

        def run():
            w_tilde = jax.w_tilde_curvature_interferometer_from(
                noise_map_real,
                uv_wavelengths,
                grid_radians_slim,
            )
            curvature_matrix = jax.curvature_matrix_via_w_tilde_from(
                w_tilde,
                mapping_matrix,
            )
            return curvature_matrix.block_until_ready()

        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"])

    @pytest.mark.benchmark
    def test_curvature_matrix_jax_BCOO(self, data_bundle, benchmark):
        data, ref = data_bundle
        data_dict = data.dict()

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        noise_map_real = jnp.array(data_dict["noise_map_real"])
        uv_wavelengths = jnp.array(data_dict["uv_wavelengths"])
        grid_radians_slim = jnp.array(data_dict["grid_radians_slim"])
        mapping_matrix = sparse.BCOO.fromdense(data_dict["mapping_matrix"])

        def run():
            w_tilde = jax.w_tilde_curvature_interferometer_from(
                noise_map_real,
                uv_wavelengths,
                grid_radians_slim,
            )
            curvature_matrix = jax.curvature_matrix_via_w_tilde_from(
                w_tilde,
                mapping_matrix,
            )
            return curvature_matrix.block_until_ready()

        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"])

    @pytest.mark.benchmark
    def test_curvature_matrix_preload_original(self, data_bundle, benchmark):
        data, ref = data_bundle
        data_dict = data.dict()
        if isinstance(data, DataGenerated):
            pytest.skip(f"Skip test_curvature_matrix_preload_original from {type(data).__name__}")

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        noise_map_real = data_dict["noise_map_real"]
        uv_wavelengths = data_dict["uv_wavelengths"]
        shape_masked_pixels_2d = data_dict["shape_masked_pixels_2d"]
        grid_radians_2d = data_dict["grid_radians_2d"]
        pix_indexes_for_sub_slim_index = data_dict["pix_indexes_for_sub_slim_index"]
        pix_size_for_sub_slim_index = data_dict["pix_size_for_sub_slim_index"]
        pix_weights_for_sub_slim_index = data_dict["pix_weights_for_sub_slim_index"]
        native_index_for_slim_index = data_dict["native_index_for_slim_index"]
        pix_pixels = data_dict["pix_pixels"]

        def run():
            curvature_preload = original.w_tilde_curvature_preload_interferometer_from(
                noise_map_real,
                uv_wavelengths,
                shape_masked_pixels_2d,
                grid_radians_2d,
            )
            curvature_matrix = original.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
                curvature_preload,
                pix_indexes_for_sub_slim_index,
                pix_size_for_sub_slim_index,
                pix_weights_for_sub_slim_index,
                native_index_for_slim_index,
                pix_pixels,
            )
            return curvature_matrix

        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"])

    @pytest.mark.benchmark
    def test_curvature_matrix_compact_jax(self, data_bundle, benchmark):
        data, ref = data_bundle
        data_dict = data.dict()

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        noise_map_real = jnp.array(data_dict["noise_map_real"])
        uv_wavelengths = jnp.array(data_dict["uv_wavelengths"])
        grid_radians_2d = jnp.array(data_dict["grid_radians_2d"])
        native_index_for_slim_index = jnp.array(data_dict["native_index_for_slim_index"])
        mapping_matrix = jnp.array(data_dict["mapping_matrix"])
        pixel_scale = data._pixel_scales

        def run():
            w_compact = jax.w_tilde_curvature_compact_interferometer_from(
                noise_map_real,
                uv_wavelengths,
                pixel_scale,
                grid_radians_2d,
            )
            curvature_matrix = jax.curvature_matrix_via_w_compact_from(
                w_compact,
                native_index_for_slim_index,
                mapping_matrix,
            )
            return curvature_matrix.block_until_ready()

        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"])

    @pytest.mark.benchmark
    def test_curvature_matrix_compact_jax_BCOO(self, data_bundle, benchmark):
        data, ref = data_bundle
        data_dict = data.dict()

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        noise_map_real = jnp.array(data_dict["noise_map_real"])
        uv_wavelengths = jnp.array(data_dict["uv_wavelengths"])
        grid_radians_2d = jnp.array(data_dict["grid_radians_2d"])
        native_index_for_slim_index = jnp.array(data_dict["native_index_for_slim_index"])
        mapping_matrix = sparse.BCOO.fromdense(data_dict["mapping_matrix"])
        pixel_scale = data._pixel_scales

        def run():
            w_compact = jax.w_tilde_curvature_compact_interferometer_from(
                noise_map_real,
                uv_wavelengths,
                pixel_scale,
                grid_radians_2d,
            )
            curvature_matrix = jax.curvature_matrix_via_w_compact_from(
                w_compact,
                native_index_for_slim_index,
                mapping_matrix,
            )
            return curvature_matrix.block_until_ready()

        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"])
