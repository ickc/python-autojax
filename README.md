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
