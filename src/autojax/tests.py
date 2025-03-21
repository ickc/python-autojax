from __future__ import annotations

import hashlib
import inspect
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pytest
from jax import numpy as jnp
from numba import jit, prange

from . import jax, numba, original


def _get_env_int(var_name: str, default: int) -> int:
    return int(os.getenv(var_name, default))


def _get_env_bool(var_name: str, default: bool) -> bool:
    return os.getenv(var_name) is not None if default is False else bool(os.getenv(var_name))


AUTOJAX_GRID_SIZE: int = _get_env_int("AUTOJAX_GRID_SIZE", 30)
AUTOJAX_N_MAPPING_NEIGHBORS: int = _get_env_int("AUTOJAX_N_MAPPING_NEIGHBORS", 3)  # Delaunay
AUTOJAX_DATA_SIZE: int = _get_env_int("AUTOJAX_DATA_SIZE", 1024)
AUTOJAX_NEIGHBOR_SIZE: int = _get_env_int("AUTOJAX_NEIGHBOR_SIZE", 32)
AUTOJAX_SRC_IMG_SIZE: int = _get_env_int("AUTOJAX_SRC_IMG_SIZE", 256)

RTOL: float = 5e-6

tests_numba: set[str] = {
    "curvature_matrix_via_w_compact_sparse_mapping_matrix_from",
    "sparse_mapping_matrix_transpose_matmul",
    "w_compact_curvature_interferometer_from",
    "w_tilde_via_compact_from",
}

tests: list[str] = [
    "constant_regularization_matrix_from",
    "curvature_matrix_via_w_compact_sparse_mapping_matrix_from",
    "curvature_matrix_via_w_tilde_curvature_preload_interferometer_from",
    "curvature_matrix_via_w_tilde_from",
    "data_vector_from",
    "log_likelihood_function_via_w_compact_from",
    "log_likelihood_function_via_w_tilde_from",
    "mapping_matrix_from",
    # "mask_2d_centres_from",
    "mask_2d_circular_from",
    "noise_normalization_complex_from",
    "reconstruction_positive_negative_from",
    "sparse_mapping_matrix_transpose_matmul",
    "w_compact_curvature_interferometer_from",
    "w_tilde_curvature_interferometer_from",
    "w_tilde_curvature_preload_interferometer_from",
    "w_tilde_data_interferometer_from",
    "w_tilde_via_compact_from",
    "w_tilde_via_preload_from",
]


def deterministic_seed(string: str, *numbers: int) -> int:
    """Generate a deterministic seed from the class name."""
    hash_value = hashlib.md5(repr((string, numbers)).encode()).hexdigest()  # Get a hash from class name
    return int(hash_value, 16) % (2**32)  # Convert to an integer within a reasonable range


@jit(nopython=True, nogil=True, parallel=False)
def neighbors_grid(neighbors: np.ndarray[tuple[int, int], np.int64]) -> np.ndarray[tuple[int, int], np.bool_]:
    S, P = neighbors.shape
    grid = np.zeros((S, S), dtype=np.bool_)
    for i in range(S):
        for p in range(P):
            j = neighbors[i, p]
            if j != -1:
                grid[i, j] = True
                grid[j, i] = True
    return grid


@jit(nopython=True, nogil=True, parallel=False)
def gen_neighbors(S, P, rng) -> np.ndarray[tuple[int, int], np.int64]:
    """Generate random neighbors."""
    if S <= 0 or P <= 0:
        raise ValueError("S and P must be positive integers.")
    if (S * P) % 2 != 0:
        raise ValueError("S*P must be even to generate a valid neighbors array.")

    # Generate the list of stubs and shuffle the stubs to randomize connections
    # each s < S appears P times
    stubs = np.repeat(np.arange(S, dtype=np.int64), P)
    rng.shuffle(stubs)
    # (SP/2, 2)
    stubs = stubs.reshape(-1, 2)

    neighbors = np.empty((S, P), dtype=np.int64)
    counts = np.zeros(S, dtype=np.int64)

    for s1, s2 in stubs:
        neighbors[s1, counts[s1]] = s2
        counts[s1] += 1
        neighbors[s2, counts[s2]] = s1
        counts[s2] += 1
    # neighbors[s, ...] should have been accessed exactly P times
    # for s in range(S):
    #     assert counts[s] == P

    # S is out of bound and indicates sentinel value of not-a-neighbor
    # will be replaced by -1
    # Remove self-loops
    for i in range(S):
        for j in range(P):
            if neighbors[i, j] == i:
                neighbors[i, j] = S

    # neighbors can be duplicated
    for i in range(S):
        unique = np.unique(neighbors[i])
        n = unique.size
        if n < P:
            neighbors[i, :n] = unique[:]
            neighbors[i, n:] = S

    neighbors.sort()
    # replace sentinel back to -1
    for s in range(S):
        for p in range(P):
            if neighbors[s, p] == S:
                neighbors[s, p] = -1
    return neighbors


@jit(nopython=True, nogil=True, parallel=True)
def gen_pix_indexes_for_sub_slim_index(
    M: int,
    S: int,
    B: int,
) -> np.ndarray[tuple[int, int], np.int64]:
    res = np.empty((M, B), dtype=np.int64)
    for m in prange(M):
        # 0 <= s_low < S
        s_low = m * S // M
        s_high = s_low + B
        # ensure not out of bounds
        if s_high > S:
            s_high = S
            s_low = s_high - B
        res[m, :] = np.arange(s_low, s_high)
    return res


@dataclass
class Data:
    """Test data."""

    pixel_scale: float = 0.2
    _centre: float = 0.0
    N_: int = AUTOJAX_GRID_SIZE
    coefficient: float = 1.0
    B: int = AUTOJAX_N_MAPPING_NEIGHBORS

    def dict(self) -> dict:
        return {
            "centre": self.centre,
            "coefficient": self.coefficient,
            "curvature_preload": self.w_tilde_preload,
            "curvature_reg_matrix": self.curvature_reg_matrix,
            "data_vector": self.data_vector,
            "data": self.data,
            "dirty_image": self.dirty_image,
            "grid_radians_2d": self.grid_radians_2d,
            "grid_radians_slim": self.grid_radians_slim,
            "grid_size": self.N,
            "mapping_matrix": self.mapping_matrix,
            "matrix": self.mapping_matrix,  # for unit tests
            "native_index_for_slim_index": self.native_index_for_slim_index,
            "neighbors_sizes": self.neighbors_sizes,
            "neighbors": self.neighbors,
            "noise_map_real": self.noise_map_real,
            "noise_map": self.noise_map,
            "pix_indexes_for_sub_slim_index": self.pix_indexes_for_sub_slim_index,
            "pix_pixels": self.pix_pixels,
            "pix_size_for_sub_slim_index": self.pix_size_for_sub_slim_index,
            "pix_weights_for_sub_slim_index": self.pix_weights_for_sub_slim_index,
            "pixel_scale": self.pixel_scale,
            "pixel_scales": self.pixel_scales,
            "pixels": self.S,
            "radius": self.radius,
            "shape_masked_pixels_2d": self.shape_masked_pixels_2d,
            "shape_native": self.shape_native,
            "slim_index_for_sub_slim_index": self.slim_index_for_sub_slim_index,
            "sub_fraction": self.sub_fraction,
            "total_mask_pixels": self.M,
            "uv_wavelengths": self.uv_wavelengths,
            "visibilities_real": self.visibilities_real,
            "w_compact": self.w_compact,
            "w_tilde_preload": self.w_tilde_preload,
            "w_tilde": self.w_tilde,
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
        return neighbors_grid(self.neighbors)

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

    path: Path = Path(__file__).parent.parent.parent / "dataset" / "data.npz"

    def __post_init__(self):
        self._data = np.load(self.path)

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

    K_: int = AUTOJAX_DATA_SIZE
    P_: int = AUTOJAX_NEIGHBOR_SIZE
    S_: int = AUTOJAX_SRC_IMG_SIZE

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
        return gen_pix_indexes_for_sub_slim_index(self.M, self.S, self.B)

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
        return gen_neighbors(S, P, rng)

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

    @cached_property
    def ref(self) -> dict:
        data_dict = self.data.dict()

        res = {}
        for test in tests:
            mod = numba if test in tests_numba else original
            func = getattr(mod, test)
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
    ref = Reference(data)
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
        mod_name = new_cls.mod.__name__.split(".")[-1]

        def create_test(test: str):

            @pytest.mark.benchmark
            def test_method(self, data_bundle, benchmark):
                data, ref, data_dict_jax = data_bundle

                try:
                    func = getattr(self.mod, test)
                except AttributeError:
                    pytest.skip(f"{mod_name}.{test} not implemented")

                benchmark.group = f"{test}_{type(data).__name__}"

                data_dict = data_dict_jax if new_cls.mod == jax else data.dict()
                ref_dict = ref.ref
                run = get_run(func, data_dict, new_cls.mod == jax)
                res = benchmark(run)
                np.testing.assert_allclose(res, ref_dict[test], rtol=RTOL)

            test_method.__name__ = f"test_{test}_{mod_name}"
            return test_method

        for test_name in tests:
            method = create_test(test_name)
            setattr(new_cls, method.__name__, method)

        return new_cls


# Example usage in a TestCase
class TestOriginal(metaclass=AutoTestMeta):
    mod = original


class TestNumba(metaclass=AutoTestMeta):
    mod = numba


class TestJax(metaclass=AutoTestMeta):
    mod = jax


# special case


class TestWTilde:
    """Compute w_tilde via various methods.

    This adds on top of existing benchmarks to compare the performance of the preload method.

    The test names are a bit strange, but is designed to be filtered like this:

        pytest -k w_tilde_curvature_interferometer_from

    This compares

    1. direct computation of ``w_tilde``
    2. (prefixed by ``_compact``/``_preload``) compute ``w_tilde`` on in the preload/compact form
    3. (prefixed by ``_expanded``) compute ``w_tilde`` as above and then expand fully

    (1) and (3) should be compared if ``w_tilde`` is needed in the full form.
    (1) and (2) should be compared if ``w_tilde`` is needed regardless of the form.
    """

    @pytest.mark.benchmark
    def test_w_tilde_curvature_interferometer_from_original_preload(self, data_bundle, benchmark):
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
        data_dict = data.dict()

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
        data_dict = data_dict_jax

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
        data_dict = data.dict()

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
        data_dict = data_dict_jax

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
        data_dict = data.dict()

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
        data_dict = data.dict()

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
        data_dict = data.dict()

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
    def test_curvature_matrix_numba_sparse(self, data_bundle, benchmark):
        """From w_tilde, internal sparse mapping matrix."""
        data, ref, _ = data_bundle
        data_dict = data.dict()

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        run = get_run(
            numba.curvature_matrix_via_w_wilde_sparse_mapping_matrix_from,
            data_dict,
        )
        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"], rtol=RTOL)

    @pytest.mark.benchmark
    def test_curvature_matrix_numba_compact_sparse_direct(self, data_bundle, benchmark):
        """From w_compact, internal sparse mapping matrix, direct 4-loop matmul."""
        data, ref, _ = data_bundle
        data_dict = data.dict()

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
        data_dict = data.dict()

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
    def test_curvature_matrix_jax_sparse(self, data_bundle, benchmark):
        """From w_tilde, internal sparse mapping matrix."""
        data, ref, data_dict_jax = data_bundle
        data_dict = data_dict_jax

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        run = get_run(
            jax.curvature_matrix_via_w_wilde_sparse_mapping_matrix_from,
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
        data_dict = data_dict_jax

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
        data_dict = data_dict_jax

        test = "curvature_matrix"
        benchmark.group = f"{test}_{type(data).__name__}"

        run = get_run(
            jax.curvature_matrix_via_w_compact_sparse_mapping_matrix_from_BCOO,
            data_dict,
            jax=True,
        )
        res = benchmark(run)
        np.testing.assert_allclose(res, ref.ref["curvature_matrix_via_w_tilde_from"], rtol=RTOL)
