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

For a logbook style repo that contains every notes about the project and how to run the AutoGalaxy family of softwares, see [ickc/log-PyAutoLens](https://github.com/ickc/log-PyAutoLens).

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

# Running tests and benchmark

```sh
pytest
# or if you use pixi
pixi run pytest
```

This should runs the tests and also give you benchmark information comparing different implementations.
