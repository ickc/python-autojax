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

RTOL: float = 2e-6

tests_generated: list[str] = [
    "constant_regularization_matrix_from",
    # "curvature_matrix_via_w_tilde_curvature_preload_interferometer_from",
    "curvature_matrix_via_w_tilde_from",
    "data_vector_from",
    # "log_likelihood_function_via_w_tilde_preload_from",
    "log_likelihood_function_via_w_tilde_from",
    "mapping_matrix_from",
    # "mask_2d_centres_from",
    "mask_2d_circular_from",
    "noise_normalization_complex_from",
    "reconstruction_positive_negative_from",
    "w_tilde_curvature_interferometer_from",
    # "w_tilde_curvature_preload_interferometer_from",
    "w_tilde_data_interferometer_from",
    "w_tilde_via_preload_from",
]
tests_loaded: list[str] = tests_generated + [
    "w_tilde_via_preload_from",
]


tests_all: list[str] = [
    "constant_regularization_matrix_from",
    "curvature_matrix_via_w_tilde_curvature_preload_interferometer_from",
    "curvature_matrix_via_w_tilde_from",
    "data_vector_from",
    "log_likelihood_function_via_w_tilde_preload_from",
    "log_likelihood_function_via_w_tilde_from",
    "mapping_matrix_from",
    # "mask_2d_centres_from",
    "mask_2d_circular_from",
    "noise_normalization_complex_from",
    "reconstruction_positive_negative_from",
    "w_tilde_curvature_interferometer_from",
    "w_tilde_curvature_preload_interferometer_from",
    "w_tilde_data_interferometer_from",
    "w_tilde_via_preload_from",
]


def deterministic_seed(string: str, *numbers: int) -> int:
    """Generate a deterministic seed from the class name."""
    hash_value = hashlib.md5(repr((string, numbers)).encode()).hexdigest()  # Get a hash from class name
    return int(hash_value, 16) % (2**32)  # Convert to an integer within a reasonable range


@dataclass
class Data:
    """Test data."""

    pixel_scale: float = 0.2
    _centre: float = 0.0
    N_: int = 30
    coefficient: float = 1.0
    B: int = 3  # Delaunay

    def dict(self) -> dict:
        return {
            "radius": self.radius,
            "coefficient": self.coefficient,
            "shape_native": self.shape_native,
            "shape_masked_pixels_2d": self.shape_masked_pixels_2d,
            "pixel_scales": self.pixel_scales,
            "pixels": self.S,
            "total_mask_pixels": self.M,
            "centre": self.centre,
            "mapping_matrix": self.mapping_matrix,
            "dirty_image": self.dirty_image,
            "visibilities_real": self.visibilities_real,
            "uv_wavelengths": self.uv_wavelengths,
            "grid_radians_2d": self.grid_radians_2d,
            "grid_radians_slim": self.grid_radians_slim,
            "native_index_for_slim_index": self.native_index_for_slim_index,
            "slim_index_for_sub_slim_index": self.slim_index_for_sub_slim_index,
            "sub_fraction": self.sub_fraction,
            "w_tilde": self.w_tilde,
            "w_tilde_preload": self.w_tilde_preload,
            "w_compact": self.w_compact,
            "curvature_preload": self.w_tilde_preload,
            "neighbors_sizes": self.neighbors_sizes,
            "neighbors": self.neighbors,
            "data_vector": self.data_vector,
            "data": self.data,
            "noise_map": self.noise_map,
            "noise_map_real": self.noise_map_real,
            "curvature_reg_matrix": self.curvature_reg_matrix,
            "pix_indexes_for_sub_slim_index": self.pix_indexes_for_sub_slim_index,
            "pix_size_for_sub_slim_index": self.pix_size_for_sub_slim_index,
            "pix_weights_for_sub_slim_index": self.pix_weights_for_sub_slim_index,
        }

    @property
    def M(self) -> int:
        return self.grid_radians_slim.shape[0]

    @property
    def N(self) -> int:
        return self.N_

    @property
    def radius(self) -> float:
        return ((self.N_ + 1) // 2) * self.pixel_scale

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
        return self.pixel_scale, self.pixel_scale

    @property
    def centre(self) -> tuple[float, float]:
        """Get the centre of the native grid."""
        return self._centre, self._centre

    @cached_property
    def mapping_matrix(self) -> np.ndarray[tuple[int, int], np.float64]:
        return original.mapping_matrix_from(
            self.pix_indexes_for_sub_slim_index,
            self.pix_size_for_sub_slim_index,
            self.pix_weights_for_sub_slim_index,
            self.S,
            self.M,
            self.slim_index_for_sub_slim_index,
            self.sub_fraction,
        )

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

    @property
    def slim_index_for_sub_slim_index(self) -> np.ndarray[tuple[int], np.int64]:
        return np.arange(self.M)

    @property
    def sub_fraction(self) -> np.ndarray[tuple[int], np.float64]:
        return np.ones(self.M)

    @cached_property
    def grid_radians_2d(self) -> np.ndarray[tuple[int, int, int], np.float64]:
        N = self.N_PRIME
        arcsec = np.pi / 648000
        d = self.pixel_scale * arcsec
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
        return np.full(self.M, self.B, dtype=np.int64)

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
    def w_compact(self) -> np.ndarray[tuple[int, int], np.float64]:
        return numba.w_compact_curvature_interferometer_from(
            self.N,
            self.noise_map_real,
            self.uv_wavelengths,
            self.pixel_scale,
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
        }

    @property
    def M(self) -> int:
        return self.pix_indexes_for_sub_slim_index.shape[0]

    @property
    def K(self) -> int:
        return self.uv_wavelengths.shape[0]

    @property
    def P(self) -> int:
        return self.neighbors.shape[1]

    @property
    def S(self) -> int:
        return self.neighbors_sizes.size

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
    def neighbors(self):
        return self._data["neighbors"]

    @property
    def neighbors_sizes(self):
        return self._data["neighbors_sizes"]

    @property
    def pix_indexes_for_sub_slim_index(self):
        return self._data["pix_indexes_for_sub_slim_index"]

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

    @cached_property
    def pix_indexes_for_sub_slim_index(self) -> np.ndarray[tuple[int, int], np.int64]:
        M = self.M
        S = self.S
        B = self.B
        res = np.empty((M, B), dtype=int)
        for m in range(M):
            # 0 <= s_low < S
            s_low = m * S // M
            s_high = s_low + B
            # ensure not out of bounds
            if s_high > S:
                s_high = S
                s_low = s_high - B
            res[m, :] = np.arange(s_low, s_high)
        return res

    @cached_property
    def pix_weights_for_sub_slim_index(self) -> np.ndarray[tuple[int, int], np.float64]:
        M = self.M
        S = self.S
        B = self.B
        rng = np.random.default_rng(deterministic_seed("pix_weights_for_sub_slim_index", M, S, B))
        res = 0.01 + rng.random((M, B))
        res /= res.sum(axis=1).reshape(-1, 1)
        return res

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
    data_dict_jax = {k: jnp.array(v) if isinstance(v, np.ndarray) else v for k, v in data.dict().items()}
    return data, ref, data_dict_jax


def get_run(func, data_dict, jax=False):
    sig = inspect.signature(func)
    args = [data_dict[key] for key in sig.parameters]

    def run():
        return func(*args)

    def run_jax():
        return func(*args).block_until_ready()

    return run_jax if jax else run


def get_run_composed_from(func1, func2, data_dict, jax=False):
    sig1 = inspect.signature(func1)
    args1 = [data_dict[key] for key in sig1.parameters]
    sig2 = inspect.signature(func2)
    args2 = [data_dict[key] for key in sig2.parameters][1:]

    def run():
        return func2(func1(*args1), *args2)

    def run_jax():
        return func2(func1(*args1), *args2).block_until_ready()

    return run_jax if jax else run


def get_run_composed_from_prepend(func1, func2, data_dict, jax=False):
    sig1 = inspect.signature(func1)
    args1 = [data_dict[key] for key in sig1.parameters]
    sig2 = inspect.signature(func2)
    args2 = [data_dict[key] for key in sig2.parameters][:-1]

    def run():
        return func2(*args2, func1(*args1))

    def run_jax():
        return func2(*args2, func1(*args1)).block_until_ready()

    return run_jax if jax else run


class AutoTestMeta(type):
    """Metaclass to auto-generate pytest-benchmark tests"""

    def __new__(cls, name, bases, namespace):
        new_cls = super().__new__(cls, name, bases, namespace)

        def create_test(test: str):
            if new_cls.mode == "benchmark":

                @pytest.mark.benchmark
                def test_method(self, data_bundle, benchmark):
                    data, ref, data_dict_jax = data_bundle
                    if test not in ref.tests:
                        pytest.skip(f"Skip {test} from {type(data).__name__}")

                    benchmark.group = f"{test}_{type(data).__name__}"

                    data_dict = data_dict_jax if new_cls.mod == jax else data.dict()
                    ref_dict = ref.ref
                    func = getattr(self.mod, test)
                    run = get_run(func, data_dict, new_cls.mod == jax)
                    res = benchmark(run)
                    np.testing.assert_allclose(res, ref_dict[test], rtol=RTOL)

            else:

                @pytest.mark.unittest
                def test_method(self, data_bundle):
                    data, ref, data_dict_jax = data_bundle
                    if test not in ref.tests:
                        pytest.skip(f"Skip {test} from {type(data).__name__}")

                    data_dict = data_dict_jax if new_cls.mod == jax else data.dict()
                    ref_dict = ref.ref
                    func = getattr(self.mod, test)
                    run = get_run(func, data_dict)
                    res = run()
                    np.testing.assert_allclose(res, ref_dict[test], rtol=RTOL)

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


class TestLogLikelihood:
    """Add the original preload method to the benchmark too.

    Example
    -------

        pytest -m benchmark -k test_log_likelihood_function
    """

    @pytest.mark.benchmark
    def test_log_likelihood_function_via_w_tilde_preload_from_original(self, data_bundle, benchmark):
        data, ref, _ = data_bundle
        benchmark.group = f"log_likelihood_function_via_w_tilde_from_{type(data).__name__}"

        data_dict = data.dict() | {"pix_pixels": data.S}
        ref_dict = ref.ref
        func = original.log_likelihood_function_via_w_tilde_preload_from
        run = get_run(func, data_dict)
        res = benchmark(run)
        np.testing.assert_allclose(res, ref_dict["log_likelihood_function_via_w_tilde_from"], rtol=RTOL)

    @pytest.mark.benchmark
    def test_log_likelihood_function_via_w_compact_from_numba(self, data_bundle, benchmark):
        data, ref, _ = data_bundle
        benchmark.group = f"log_likelihood_function_via_w_tilde_from_{type(data).__name__}"

        data_dict = data.dict() | {"pix_pixels": data.S}
        ref_dict = ref.ref
        func = numba.log_likelihood_function_via_w_compact_from
        run = get_run(func, data_dict)
        res = benchmark(run)
        np.testing.assert_allclose(res, ref_dict["log_likelihood_function_via_w_tilde_from"], rtol=RTOL)

    @pytest.mark.benchmark
    def test_log_likelihood_function_via_w_compact_from_jax(self, data_bundle, benchmark):
        data, ref, data_dict_jax = data_bundle
        benchmark.group = f"log_likelihood_function_via_w_tilde_from_{type(data).__name__}"

        data_dict = data.dict() | {"pix_pixels": data.S}
        ref_dict = ref.ref
        func = jax.log_likelihood_function_via_w_compact_from
        run = get_run(func, data_dict_jax, jax=True)
        res = benchmark(run)
        np.testing.assert_allclose(res, ref_dict["log_likelihood_function_via_w_tilde_from"], rtol=RTOL)


class TestWTilde:
    """Compute w_tilde via various methods.

    This adds on top of existing benchmarks to compare the performance of the preload method.

    The test names are a bit strange, but is designed to be filtered like this:

        pytest -m benchmark -k w_tilde_curvature_interferometer_from

    This compares

    1. direct computation of ``w_tilde``
    2. (prefixed by ``_compact``/``_preload``) compute ``w_tilde`` on in the preload/compact form
    3. (prefixed by ``_expanded``) compute ``w_tilde`` as above and then expand fully

    (1) and (3) should be compared if ``w_tilde`` is needed in the full form.
    (1) and (2) should be compared if ``w_tilde`` is needed regardless of the form.
    """

    @pytest.mark.benchmark
    def test_w_tilde_curvature_interferometer_original_preload(self, data_bundle, benchmark):
        data, _, _ = data_bundle
        data_dict = data.dict()

        test = "w_tilde_curvature_interferometer_from"
        benchmark.group = f"{test}_{type(data).__name__}"
        run = get_run(
            original.w_tilde_curvature_preload_interferometer_from,
            data_dict,
        )
        benchmark(run)

    @pytest.mark.benchmark
    def test_w_tilde_curvature_interferometer_from_numba_compact(self, data_bundle, benchmark):
        data, _, _ = data_bundle
        data_dict = data.dict() | {
            "grid_size": data.N,
            "pixel_scale": data.pixel_scale,
        }

        test = "w_tilde_curvature_interferometer_from"
        benchmark.group = f"{test}_{type(data).__name__}"
        run = get_run(
            numba.w_compact_curvature_interferometer_from,
            data_dict,
        )
        benchmark(run)

    @pytest.mark.benchmark
    def test_w_tilde_curvature_interferometer_from_jax_compact(self, data_bundle, benchmark):
        data, _, data_dict_jax = data_bundle
        data_dict = data_dict_jax | {
            "grid_size": data.N,
            "pixel_scale": data.pixel_scale,
        }

        test = "w_tilde_curvature_interferometer_from"
        benchmark.group = f"{test}_{type(data).__name__}"

        run = get_run(
            jax.w_compact_curvature_interferometer_from,
            data_dict,
            jax=True,
        )
        benchmark(run)

    @pytest.mark.benchmark
    def test_w_tilde_curvature_interferometer_from_original_preload_expanded(self, data_bundle, benchmark):
        data, ref, _ = data_bundle
        data_dict = data.dict()

        test = "w_tilde_curvature_interferometer_from"
        benchmark.group = f"{test}_{type(data).__name__}"
        run = get_run_composed_from(
            original.w_tilde_curvature_preload_interferometer_from,
            original.w_tilde_via_preload_from,
            data_dict,
        )
        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["w_tilde_curvature_interferometer_from"], rtol=RTOL)

    @pytest.mark.benchmark
    def test_w_tilde_curvature_interferometer_from_numba_compact_expanded(self, data_bundle, benchmark):
        data, ref, _ = data_bundle
        data_dict = data.dict() | {
            "grid_size": data.N,
            "pixel_scale": data.pixel_scale,
        }

        test = "w_tilde_curvature_interferometer_from"
        benchmark.group = f"{test}_{type(data).__name__}"
        run = get_run_composed_from(
            numba.w_compact_curvature_interferometer_from,
            numba.w_tilde_via_compact_from,
            data_dict,
        )
        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["w_tilde_curvature_interferometer_from"], rtol=RTOL)

    @pytest.mark.benchmark
    def test_w_tilde_curvature_interferometer_from_jax_compact_expanded(self, data_bundle, benchmark):
        data, ref, data_dict_jax = data_bundle
        data_dict = data_dict_jax | {
            "grid_size": data.N,
            "pixel_scale": data.pixel_scale,
        }

        test = "w_tilde_curvature_interferometer_from"
        benchmark.group = f"{test}_{type(data).__name__}"

        run = get_run_composed_from(
            jax.w_compact_curvature_interferometer_from,
            jax.w_tilde_via_compact_from,
            data_dict,
            jax=True,
        )
        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["w_tilde_curvature_interferometer_from"], rtol=RTOL)


class TestCurvatureMatrix:
    """Compute curvature matrix via various methods.

    The input w can be w_tilde, preload, or compact. w_tilde is allowed
        to consider that it can be expanded in memory outside the MCMC loop.
    The input mapping_matrix must be its sparse form, such as pix_weights_for_sub_slim_index, ...
        This is because the mapping matrix has to be generated on the fly anyway,
        so even the dense form must be generated from the sparse form at some
        point in the MCMC loop.
    """

    # original
    @pytest.mark.benchmark
    def test_curvature_matrix_original(self, data_bundle, benchmark):
        """From w_tilde, construct dense mapping matrix."""
        data, ref, _ = data_bundle
        data_dict = data.dict() | {"pix_pixels": data.S}

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        run = get_run_composed_from_prepend(
            original.mapping_matrix_from,
            original.curvature_matrix_via_w_tilde_from,
            data_dict,
        )
        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"], rtol=RTOL)

    @pytest.mark.benchmark
    def test_curvature_matrix_original_preload_direct(self, data_bundle, benchmark):
        """From w-preload, internal sparse mapping matrix."""
        data, ref, _ = data_bundle
        data_dict = data.dict() | {"pix_pixels": data.S}

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        run = get_run(
            original.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from,
            data_dict,
        )
        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"], rtol=RTOL)

    # numba
    @pytest.mark.benchmark
    def test_curvature_matrix_numba(self, data_bundle, benchmark):
        """From w_tilde, construct dense mapping matrix."""
        data, ref, _ = data_bundle
        data_dict = data.dict() | {
            "grid_size": data.N,
            "pixel_scale": data.pixel_scale,
        }

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        run = get_run_composed_from_prepend(
            numba.mapping_matrix_from,
            numba.curvature_matrix_via_w_tilde_from,
            data_dict,
        )
        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"], rtol=RTOL)

    @pytest.mark.benchmark
    def test_curvature_matrix_numba_compact_sparse_direct(self, data_bundle, benchmark):
        """From w_compact, internal sparse mapping matrix, direct 4-loop matmul."""
        data, ref, _ = data_bundle
        data_dict = data.dict() | {
            "grid_size": data.N,
            "pixel_scale": data.pixel_scale,
            "pixels": data.S,
        }

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        run = get_run(
            numba.curvature_matrix_via_w_compact_sparse_mapping_matrix_direct_from,
            data_dict,
        )
        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"], rtol=RTOL)

    @pytest.mark.benchmark
    def test_curvature_matrix_numba_compact_sparse(self, data_bundle, benchmark):
        """From w_compact, internal sparse mapping matrix, sparse matmul."""
        data, ref, _ = data_bundle
        data_dict = data.dict() | {
            "grid_size": data.N,
            "pixel_scale": data.pixel_scale,
            "pixels": data.S,
        }

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        run = get_run(
            numba.curvature_matrix_via_w_compact_sparse_mapping_matrix_from,
            data_dict,
        )
        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"], rtol=RTOL)

    # jax
    @pytest.mark.benchmark
    def test_curvature_matrix_jax(self, data_bundle, benchmark):
        """From w_tilde, construct dense mapping matrix."""
        data, ref, data_dict_jax = data_bundle
        data_dict = data_dict_jax

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        run = get_run_composed_from_prepend(
            jax.mapping_matrix_from,
            jax.curvature_matrix_via_w_tilde_from,
            data_dict,
            jax=True,
        )
        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"], rtol=RTOL)

    @pytest.mark.benchmark
    def test_curvature_matrix_jax_BCOO(self, data_bundle, benchmark):
        """From w_tilde, construct BCOO mapping matrix."""
        data, ref, data_dict_jax = data_bundle
        data_dict = data_dict_jax

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        run = get_run_composed_from_prepend(
            jax.mapping_matrix_from_BCOO,
            jax.curvature_matrix_via_w_tilde_from,
            data_dict,
            jax=True,
        )
        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"], rtol=RTOL)

    @pytest.mark.benchmark
    def test_curvature_matrix_jax_compact_sparse(self, data_bundle, benchmark):
        """From w_compact, internal sparse mapping matrix."""
        data, ref, data_dict_jax = data_bundle
        data_dict = data_dict_jax | {
            "pixels": data.S,
            "grid_size": data.N,
            "pixel_scale": data.pixel_scale,
        }

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        run = get_run(
            jax.curvature_matrix_via_w_compact_sparse_mapping_matrix_from,
            data_dict,
            jax=True,
        )
        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"], rtol=RTOL)

    @pytest.mark.benchmark
    def test_curvature_matrix_jax_compact_sparse_BCOO(self, data_bundle, benchmark):
        """From w_compact, left BCOO mapping matrix, right internal sparse mapping matrix."""
        data, ref, data_dict_jax = data_bundle
        data_dict = data_dict_jax | {
            "pixels": data.S,
            "grid_size": data.N,
            "pixel_scale": data.pixel_scale,
        }

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        run = get_run(
            jax.curvature_matrix_via_w_compact_sparse_mapping_matrix_from_BCOO,
            data_dict,
            jax=True,
        )
        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"], rtol=RTOL)
