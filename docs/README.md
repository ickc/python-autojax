# DiRAC: revealing the nature of dark matter with the James Webb space telescope and JAX

![GitHub Actions](https://github.com/ickc/python-autojax/workflows/Unit%20tests/badge.svg)
[![Documentation Status](https://github.com/ickc/python-autojax/workflows/GitHub%20Pages/badge.svg)](https://ickc.github.io/python-autojax)

[![GitHub Releases](https://img.shields.io/github/tag/ickc/python-autojax.svg?label=github+release)](https://github.com/ickc/python-autojax/releases)
<!-- [![PyPI Package latest release](https://img.shields.io/pypi/v/autojax.svg)](https://pypi.org/project/autojax)
[![Supported versions](https://img.shields.io/pypi/pyversions/autojax.svg)](https://pypi.org/project/autojax)
[![Supported implementations](https://img.shields.io/pypi/implementation/autojax.svg)](https://pypi.org/project/autojax)
[![PyPI Wheel](https://img.shields.io/pypi/wheel/autojax.svg)](https://pypi.org/project/autojax)
[![Development Status](https://img.shields.io/pypi/status/autojax.svg)](https://pypi.python.org/pypi/autojax/)
[![Downloads](https://img.shields.io/pypi/dm/autojax.svg)](https://pypi.python.org/pypi/autojax/)
![License](https://img.shields.io/pypi/l/autojax.svg) -->

# Introduction

This is a small self-contained repo for the 3 months project "DiRAC: revealing the nature of dark matter with the James Webb space telescope and JAX". It is organized in 3 modules. `original` is copied from [Jammy2211/dirac_rse_interferometer](https://github.com/Jammy2211/dirac_rse_interferometer) (which itself is copied from various other repos such as AutoGalaxy and AutoLens), and then they are ported to JAX in the `jax` module. While the `original` is already implemented in Numba, a `numba` module is also provided here, mainly as a starting point to port from `original` to something more vectorized first. Often time the `jax` implementation is the same as the `numba` implementation here, or a close variant of it due to differences between Numba and JAX.

As part of the goal to port to JAX implementation is to speed up, benchmark is provided to compare the 3 implementations. See instructions below to see how to run it.

For a logbook style repo that contains every notes about the project and how to run the AutoGalaxy family of softwares, see [ickc/log-PyAutoLens](https://github.com/ickc/log-PyAutoLens). There's some scripts under `external/` here that can only be run using the environment documented there, as it uses some external (to us) dependencies recorded as git submodules over there.

# Installing the project

## pip

```sh
pip install -e .[tests]
```

## conda/mamba

```sh
conda env create -f environment.yml
conda activate autojax
# update using
conda env update --name autojax --file environment.yml --prune
```

## pixi

```sh
pixi install
# prepend everything you run by pixi run, such as
pixi run pytest
```

:::{Note}
From this point on, pixi is assumed and every command is prepended with `pixi run`. For other ways of loading environment, simply load your environment as usual and disregard this prefix.
:::

# Supported platforms

The codebase has been tested on macOS with CPU, x64 Linux with CPU, and Linux with GPU via CUDA.

By default, it is run on CPU on either platform if you simply run

```sh
pixi run pytest
```

On Linux, if you want to run on NVidia GPU with CUDA, you can use

```sh
pixi run --environment cuda pytest
```

Unfortunately, running on AMD GPU such as MI300X are not tested as it is not straigtforward to setup an environment and due to time constraint it becomes out of scope to this project. See [JAX documentation on supported platforms](https://docs.jax.dev/en/latest/installation.html#supported-platforms) for instructions on using docker to set up the environment.

# Running unit tests and benchmarks

```sh
pixi run pytest
```

This should runs the tests and also give you benchmark information comparing different implementations.

By default, it will run the benchmark as well. If you want to run unit tests only,

```sh
pixi run pytest --benchmark-disable
```

## Benchmark framework

`pytest-benchmark` is chosen to benchmark the performance between implementations.

There are 2 sets of data available, one is a repackaged data from upstream PyAutoGalaxy projects, called {py:class}`autojax.tests.DataLoaded`, another is mock data to facilitate the generation of input data with appropriate properties while can scale arbitrarily to given array dimensions, {py:class}`autojax.tests.DataGenerated`.

:::{Note}
To run test only on one kind of data, use pytest filter, e.g.

```sh
pixi run pytest -k DataGenerated
```
:::

All available functions are tested via meta-programming (i.e. all combinations of tests between different implementations are automatically generated). Two more specialized tests, {py:class}`autojax.tests.TestWTilde` and {py:class}`autojax.tests.TestCurvatureMatrix`, are given for more in-depth comparison on various different implementations to calculate `w_tilde` and `curvature_matrix`.
You can filter them like this,

```sh
pixi run pytest -k 'TestWTilde or TestCurvatureMatrix'
```

## Advanced benchmarking

# Products of this project

The `autojax` library is organized in 3 main modules, `autojax.original`, `autojax.numba`, and `autojax.jax`. All functions from `autojax.original` is originated from the original PyAutoGalaxy family of projects. Most of these functions (which are already implemented in Numba) are reimplemented in Numba, and then in JAX, with the exception of the `preload` functions, replaced by an equivalent but different `compact` functions.

A utility can be run to see what's included, run by

```bash
pixi run python -m autojax.util.mod_diff
```

Most of these function reimplementations are straight forward, except in the following cases.

## $\tilde{w}$

{py:class}`autojax.tests.TestWTilde` implement multiple different ways of calculating $\tilde{w}$ to compare the performance between implementations. You can run it by

```sh
pixi run pytest -k w_tilde_curvature_interferometer_from
```

Although the documentation of the test class has already explained, we will mention it again here.

It includes

```console
-------------------------------------------------- benchmark 'w_tilde_curvature_interferometer_from_DataLoaded': 9 tests --------------------------------------------------
Name (time in us)                                                                            Mean                StdDev                   OPS            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_w_tilde_curvature_interferometer_from_numba_compact[DataLoaded]                     579.2566 (1.0)         24.5549 (1.17)     1,726.3507 (1.0)         875           1
test_w_tilde_curvature_interferometer_from_jax_compact[DataLoaded]                       914.4974 (1.58)        20.8980 (1.0)      1,093.4968 (0.63)         33           1
test_w_tilde_curvature_interferometer_from_jax_compact_expanded[DataLoaded]            1,173.1748 (2.03)        28.6571 (1.37)       852.3879 (0.49)         41           1
test_w_tilde_curvature_interferometer_from_numba_compact_expanded[DataLoaded]          1,228.3708 (2.12)       123.1428 (5.89)       814.0864 (0.47)        681           1
test_w_tilde_curvature_interferometer_from_original_preload[DataLoaded]                7,654.0062 (13.21)      194.1332 (9.29)       130.6505 (0.08)        121           1
test_w_tilde_curvature_interferometer_from_original_preload_expanded[DataLoaded]       8,847.4153 (15.27)      330.9996 (15.84)      113.0274 (0.07)        120           1
test_w_tilde_curvature_interferometer_from_numba[DataLoaded]                          57,778.1492 (99.75)    2,981.0978 (142.65)      17.3076 (0.01)         19           1
test_w_tilde_curvature_interferometer_from_jax[DataLoaded]                            81,576.6430 (140.83)     468.4998 (22.42)       12.2584 (0.01)          7           1
test_w_tilde_curvature_interferometer_from_original[DataLoaded]                      586,633.8498 (>1000.0)  8,917.3414 (426.71)       1.7046 (0.00)          5           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

The `original`, `numba`, and `jax` variants are self-explanatory.

Without any additional suffix, they are the direct method that computes $\tilde{w}$ directly.

In the `original` implementation, there is a `preload` method that calculate only the unique values in the $\tilde{w}$ matrix, based on the differences on the grid lattice, as shown by the structure of the equation,

$$\tilde{W}_{ij} = \sum_{k=1}^N \frac{1}{n_k^2} \cos(2\pi[(g_{i1} - g_{j1})u_{k0} + (g_{i0} - g_{j0})u_{k1}]),$$

where it only depends on the $Δ_{ij} = \vec{g}_i - \vec{g}_j$ where $g$ is the lattice in radian.

The `compact` method, implemented in {py:class}`autojax.jax.w_compact_curvature_interferometer_from`, took a similar approach as `preload`,
with a few differences:

- It no longer calculates from $\vec{g}$ where the lattice constant is baked in. Instead, the `pixel_scale` is passed to it and it assumes the lattice is in integer steps, and hence can be calculated directly on-the-fly.
- It shifts the difference grid $Δ$ to the corner, and hence all the indices are positive, and hence there's no longer 4 different cases to handle, and reduces the overhead by negative indexing (and hence reduce cache misses of walking backward).
- It further reduces the amount of computation by roughly a factor of 2, as $\cos x = \cos -x$ and hence we can WLOG assume the row difference to be non-negative.

Look at {py:class}`autojax.jax.w_tilde_via_compact_from` to see how $\tilde{w}$ matrix is related to w-compact representation (which is not a matrix, as it cannot be represented as a linear tranformation between them).

The `expanded` variants also expand the w-compact to $\tilde{w}$ after calculating w-compact. I.e. it is the combined cost of calculating w-compact first,
and then expanded it to $\tilde{w}$ in memory.
This illustrates that even if the end goal is to calculate $\tilde{w}$ in memory,
it is still much faster to get w-compact as an intermediate step.

:::{Hint}
Always calculate w-compact first, regardless what you're going to do next in the next section.
:::

## $F$

{py:class}`autojax.tests.TestCurvatureMatrix` compute the curvature matrix $F$ via various methods.

As $F = T^T \tilde{w} T$, where $T$ is the mapping matrix,

The input $w$ can be $\tilde{w}$, or in `preload`, or `compact` representation. $\tilde{w}$ is also
considered here as it can be expanded in memory outside the MCMC loop.
The input $T$ is considered to be started in its sparse form (i.e. from `pix_weights_for_sub_slim_index`, `pix_indexes_for_sub_slim_index`).
This is because the mapping matrix has to be generated on the fly anyway,
so even the dense form must be generated from the sparse form at some
point in the MCMC loop.

Hereby we refer `pix_weights_for_sub_slim_index`, `pix_indexes_for_sub_slim_index`, etc. as the internal sparse mapping matrix representation.

:::{Note}
Throughout these implementations, the mapping matrix is assumed to not contain sub-pixels
and hence `pix_weights_for_sub_slim_index`, `pix_indexes_for_sub_slim_index` are equivalent to a representation of sparse matrix,
which is generally not true if there are sub-pixels.
:::

It can be run by

```sh
pixi run pytest -k TestCurvatureMatrix -vv
```

```console
------------------------------------------------- benchmark 'curvature_matrix_DataLoaded': 11 tests --------------------------------------------------
Name (time in us)                                                       Mean                StdDev                   OPS            Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------------------
test_curvature_matrix_numba_sparse[DataLoaded]                      948.8185 (1.0)        145.1067 (1.92)     1,053.9423 (1.0)         708           1
test_curvature_matrix_numba_compact_sparse[DataLoaded]            1,472.3449 (1.55)        75.4961 (1.0)        679.1887 (0.64)        625           1
test_curvature_matrix_jax_BCOO[DataLoaded]                        1,522.2917 (1.60)       399.0087 (5.29)       656.9043 (0.62)         18           1
test_curvature_matrix_jax_sparse[DataLoaded]                      1,803.8645 (1.90)       108.8925 (1.44)       554.3654 (0.53)         28           1
test_curvature_matrix_jax_compact_sparse_BCOO[DataLoaded]         1,932.4616 (2.04)       126.1022 (1.67)       517.4747 (0.49)         25           1
test_curvature_matrix_jax_compact_sparse[DataLoaded]              1,951.4066 (2.06)        93.3363 (1.24)       512.4509 (0.49)         21           1
test_curvature_matrix_original_preload_direct[DataLoaded]         2,684.7593 (2.83)       165.9653 (2.20)       372.4729 (0.35)        379           1
test_curvature_matrix_numba_compact_sparse_direct[DataLoaded]     2,859.9388 (3.01)       131.5643 (1.74)       349.6578 (0.33)        317           1
test_curvature_matrix_jax[DataLoaded]                             4,095.5833 (4.32)       156.3028 (2.07)       244.1655 (0.23)         14           1
test_curvature_matrix_original[DataLoaded]                        6,436.0768 (6.78)       462.8975 (6.13)       155.3742 (0.15)        128           1
test_curvature_matrix_numba[DataLoaded]                           8,752.8459 (9.22)     3,395.5921 (44.98)      114.2486 (0.11)        126           1
------------------------------------------------------------------------------------------------------------------------------------------------------
```

The `original`, `numba`, and `jax` variants are self-explanatory.
We will explains the other one by one below.

`test_curvature_matrix_original`, `test_curvature_matrix_numba`, `test_curvature_matrix_jax` assumes `w_tilde` is already in memory, and construct the dense mapping matrix $T$ via
internal sparse mapping matrix representation.
This represents the most direct mathematical definition, and is slowest.

`test_curvature_matrix_original_preload_direct` benchmark the {py:class}`autojax.original.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from` that calculates the curvature matrix $F$ from the preload $w$ representation
and the internal sparse mapping matrix representation via a direct 4-loop representing the 2 sum of the 2 matrix multiplications. Note that this is doing redundant calculations which consumes more FLOP than necessary comparing to performing the matrix multiplication in 2 passes (such as $T^T (\tilde{w} T)$). See the docstrings for cost analysis to understand why.

`test_curvature_matrix_numba_compact_sparse_direct` benchmark the {py:class}`autojax.numba.curvature_matrix_via_w_compact_sparse_mapping_matrix_direct_from` that does similarly (compute $F$ directly from w-compact and internal sparse mapping matrix representation via a direct 4-loop). Again, this is doing redundant FLOP, but has the smallest memory footprint. See the docstrings for cost analysis to understand why.


`test_curvature_matrix_numba_sparse` and `test_curvature_matrix_jax_sparse` assumes `w_tilde` is already in memory, and perform a custom matmul of $\tilde{w}$ with the internal sparse mapping matrix representation. These methods should be fastest, as obtaining the $\tilde{w}_ij$ elements are most direct and hence reduces IO overhead, at the expense of memory footprint. It is expected to be faster than multiplying with the dense mapping matrix $T$, as the dense matrix has to be constructed from the internal sparse representation in the MCMC-loop anyway, and the redundant multiplications with a lot of zeros is unnessary.

`test_curvature_matrix_numba_compact_sparse` and `test_curvature_matrix_jax_compact_sparse` are calculated from w-compact and the internal sparse mapping matrix representation. This first of all calculates $Ω = \tilde{w} T$ with a custom matmul, where the $\tilde{w}_{ij}$ elements are indexed from w-compact first and $T$ is constructed from its internal sparse representation. It then perform a $T^T Ω$ with another custom matmul of internal sparse $T$ with any general dense matrix. This is expected to be second best comparing to `test_curvature_matrix_numba_sparse` and `test_curvature_matrix_jax_sparse` in speed but has a much smaller memory footprint.

Finally, there are `test_curvature_matrix_jax_compact_sparse_BCOO` and `test_curvature_matrix_jax_BCOO` where the $T$ matrix is constructed to the `BCOO` sparse matrix reprentation in JAX directly from the internal sparse representation. This requires an additional conversion (from internal sparse to `BCOO` sparse). But it has an advantage of delegating the matmul to JAX which potentially is more optimized. However, the conversion utilizes an undocumented property of BCOO in JAX that out of bound indices are dropped.
`test_curvature_matrix_jax_BCOO` is fastest with $\tilde{w}$ fully expanded in memory to improve indexing performance while doing sparse matrix matmul to avoid redundant calculations.

`test_curvature_matrix_jax_sparse` comes in a close second, and is recommended as it doesn't rely on experimental JAX sparse matrix support nor an undocumented behavior mentioned above.

# Misc.

## Experiments

Under `experiments/`, there is a set of experiments, under the framework of JAX, testing various ways to perform reduced sum over vector(s) of length $K$ without expanding to higher dimensional N-D arrays that consumes memory proportional to $K$. Head over to `experiments/README.md` to see the conclusion there. Based on that study, `scan` is used to ensure none of the intermediate memory use scales as `K` (the number of visibilities).

## 🔪 The Sharp Bits 🔪

TODO: expand this section

- `static_argnums`
- multithreading on CPU with JAX
- The "internal sparse matrix representation" in PyAutoArray such as `neighbors`, `pix_indexes_for_sub_slim_index` and their implications with JAX.

### `static_argnums`

Beware that it can backfire as any changes requires a recompilation.

One pattern to use in this kind of situation is closure, where an example is in {py:class}`autojax.jax.w_compact_curvature_interferometer_from`. (This is a good closure example but not specifically related to removing `static_argnums`.) This kind of pattern is useful if you write a certain function that you know will not be used in other places (i.e. a private function), and you can instead define it locally inside a bigger public function, Here you can see that `δ_mn0` is created and used in the inner private function.

## Numba vs. JAX

### Should Numba be dropped completely?

Regarding the dependencies on Numba and JAX in general, I think removing Numba as a dependency is not necessary and may be harmful in some cases. In autojax, numba and JAX live happily alongside each other as long as the import are done correctly (e.g. no `import jax.numpy as np` to mixes `np` and `jnp`).

There are these kinds of situations

1. application hotspots that you optimize as much as you can
2. essential functions that you need, and is sufficient as long as it is not too slow. It can be either
    1. implemented in Numba
    1. implemented in pure Numpy

It sounds like (1) is definitely going to be ported from Numba to JAX.

The problem of completely dropping Numba from anywhere in the codebase is you need to deal with (2.1). I.e. you either port it to JAX (1), or write it in a pure Numpy way (2.2). Either way it can takes non-trivial amount of time, and might actually be slower (as shown in the benchmark here). Just to reiterate, even if you `numba.jit` a function that work perfectly in pure Numpy, you can get huge speed up by putting it inside `numba.jit` because of the Python object model leading to inefficiency in pure Numpy operation. (E.g. Consider `X = A @ B + C`, the intermediate Python object `A @ B` are created, which can be avoided when jitted. That by the way is why NumExpr exists for this specialty case.)

A random example is in {py:class}`autojax.tests.gen_neighbors`. Here I need to generate mock neighbors array with suitable property. It is only needed when I run test and benchmark, so it doesn't have to be super-fast (i.e. it is not in the (1) case), but it is not obvious how to do (2.2), and pure Python is too slow even for the purpose of testing. So numba-jit (2.1) in this case is no-brainer and save a lot of time in both developing and running tests.

While this particular example is a unit test and hence Numba can be made an optional dependency, it is highly likely that there exists some examples in the core functionality that you encounter this too.

### Performance between them on a single CPU core

Assuming comparing features where they overlap, and focusing only on a single CPU core,
my assertion would be that it is, on average impossible for a JAX implementation to be meaningfully faster than a Numba implementation.
It is because

1. An algorithm one can express in JAX can equally be expressed in Numba, but not the other way around.
2. And if (1) is true, then even if we only focus on the subset of algorithms that can equally be expressed in JAX and Numba, any performance difference will then be due to the compilers, and the hints the language can passes more to the compiler. This is in similar situation with comparing C/C++, Fortran, Julia, etc. While there are differences, they are O(1) in speed between each other. While the XLA compiler is interesting in the sense that it compiles specifically for a given shape of array, the compiler can have more hints (on top of knowing the exact CPU architecture at run time similar to Numba and Julia) to optimize the loops and chunks given the shape, fundamentally it cannot beats other compilers much further. For example, XLA on the CPU actually delegates to LLVM compiler eventually, similar to Numba and Julia.

Of the cases I see JAX beats Numba in `autojax`, usually it is either simple linear algebra or vectorized operations. That’s probably the case where the `jnp.array` and `jax.jit` optimized with the shape information (and/or memory alignment too).
In all other cases, the Numba implementation easily beats the JAX’s, basically because the Numba algorithm cannot be effectively expressed in JAX, so that the JAX version while looks like doing similar thing, but has additional costs such as creating some array in memory.

Note that we are talking about same algorithm implemented in different framework.
But one reason people can see huge speedup after porting it to JAX is that the JAX programming paradigm (functional, vectorized, static shape, etc.) forces you to write in a way that is efficient (i.e. it forces you to change the algorithm).
Another aspect JAX excels at is the kind of high level, automatic optimization it will do for you such as fusions of operations.
It can be done by hand in other languages including Numba but can be tedious. (In this aspect, JAX is actually changing your algorithm behind the scene.) But on the other hand, fusion is automagic which is difficult to foresee if it will happen. In one case in `autojax`, I anticipated it should happen but it actually doesn’t. But in Numba you can control that directly.

My bet is that in these cases, if you backport the algorithm change and implement it in Numba as well, it would results in huge speedup too.
That’s the case comparing `original` and `numba` in autojax, as `original` is also implemented in Numba. And in most cases the reimplementation makes it much faster than `original` in my benchmark.

To conclude, frameworks differences make something easy in one but difficult in another. JAX has its benefits in many regards. E.g. there can be optimization by JAX behind the scene that is too tiresome to perform in Numba. My point above is essentially, for the cases where a `numba` implementation in `autojax` is faster than the `jax` equivalent on a single core, for most of them you’ll never be able to flip it around by further optimization.

It is another matter for multiple CPU cores though... More on that later.

## Beyond porting

Some thoughts on the architecture of the parent libraries as they evolve with this porting effort, which can be subjective:

Avoid automagic such as runtime import check and dispatch your `jit` to `numba.jit` vs. `jax.jit`, or `numpy` vs. `jax.numpy`. This can leads to unexpected behavior at best. You could do sensible defaults, but the difference between that and automagic is, automagic tries to make decision for the users hiding the details, but sensible defaults are, making defaults that is sensible but also giving the user controls. E.g. `DEFAULT_JIT = numba|jax`, `ARRAY_CONVERSION = ALWAYS_NUMPY|ALWAYS_JAX|NEVER`. This also makes your code less stateful and hence easier to reason with.

Treating the “library code” that does the computation and the end user facing interface such as classes separately.
Here the “library code” means something like a lib implemented in whatever language, be it C/C++/Numba/JAX/Julia, and then you have your interface calling them eventually. It decouples your implementation details from the interface, with the disadvantage that there’s more rabbit holes to go through. With the example of `autojax`, you can do something like this illustrative pseudo code:

```py
import os

import autojax

DEFAULT_JIT = os.environ.get("DEFAULT_JIT", "jax")  # or some sort of config
LUT = {
    "jax": autojax.jax,
    "numba": autojax.numba,
}
FALLBACK_MOD = autojax.numba
...


class SomethingLikeThis:
    mode: str = DEFAULT_JIT
    ...

    @property
    def mod(self):
        try:
            return LUT[self.mode]
        except KeyError:
            ...

    def mask(self):
        mask_2d_circular_from = getattr(self.mod, "mask_2d_circular_from", None)
        # fall back to Numba if no JAX implementation is around
        if mask_2d_circular_from is None:
            mask_2d_circular_from = getattr(FALLBACK_MOD, "mask_2d_circular_from")
        # you can even do more metaprogramming to auto-retrieve these args, see autojax.tests for example
        shape_native = getattr(self, ...)
        return mask_2d_circular_from(shape_native, ...)
...

# then end-user can actually also have control:
something = SomethingLikeThis(mode="numba")
# or this can be passed around in various ways to the interface that the user control...
```

Here the library is also dealing with simple input types/classes, i.e. avoid the need of subclassing which might couples your implementation details (such as `np` vs. `jnp`) to your interface (such as the attributes that it presents.)

Porting the majority of the codebase from Numba to JAX is complicated, especially with these couplings.
But it is possible that with this restructuring, you could make the porting more manageable.
It could be manageable to the point that keeping 2 implementations around is easy, and in that case you can “have your cake and eat it too”.

One final remark is, you could flip everything around and refactor everything into `autojax` and then have your libraries calling `autojax` this way... But of course that’s independent to the architectural suggestion above.
