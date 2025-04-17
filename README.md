# SuperFaB

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://hsgg.github.io/SphericalFourierBesselDecompositions.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://hsgg.github.io/SphericalFourierBesselDecompositions.jl/dev)
[![Build Status](https://github.com/hsgg/SphericalFourierBesselDecompositions.jl/workflows/CI/badge.svg)](https://github.com/hsgg/SphericalFourierBesselDecompositions.jl/actions)

*SuperFaB* is a code for cosmological spherical Fourier-Bessel (SFB) analysis.
The details of the code are presented in [2102.10079](https://arxiv.org/abs/2102.10079).

The name of this package is **SphericalFourierBesselDecompositions.jl**.

For installation instructions and a tutorial, see the
[Documentation](https://hsgg.github.io/SphericalFourierBesselDecompositions.jl/dev).


## pySuperFaB: Python wrapper for SphericalFourierBesselDecompositions.jl

This is a simple Python wrapper for [`SphericalFourierBesselDecompositions.jl`](https://github.com/hsgg/SphericalFourierBesselDecompositions.jl).


### Installation

```
pip install .
```

The only dependency (from a Python perspective) is `juliacall`, which is
available in conda as `pyjuliacall`.


### Usage

You can use this as any other Python package. Under the hood it uses JuliaCall,
which on first import will automatically download Julia if you don't already
have it. It will also then download the
`SphericalFourierBesselDecompositions.jl` Julia package.

Use like so:
```
from pysuperfab import SFB

kmin = 0.05
rmin = 0.0
rmax = 2300.0
amodes = SFB.AnlmModes(kmin, rmin, rmax)
```
That should start looking *really* familiar from the
`SphericalFourierBesselDecompositions.jl` documentation.


## How to make a release

1. Edit version field in `Project.toml`.
2. Edit version field in `pysuperfab/juliapkg.json`.
3. Edit version field in `pyproject.toml`.
4. Commit version updates.
5. Comment with `@JuliaRegistrator register()` on that commit. This will
   publicize the package in the Julia General registry.
5. Python package could be uploaded... somwhere
