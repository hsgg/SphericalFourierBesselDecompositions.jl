# SuperFaB

*SuperFaB* is a code for cosmological spherical Fourier-Bessel (SFB) analysis.
The details of the code are presented in [2102.10079](https://arxiv.org/abs/2102.10079).


## Installation

To install *SuperFaB*, start the [Julia](https://julialang.org/) REPL and type
```julia
]add SphericalFourierBesselDecompositions
```

The package makes use of some python packages (e.g. healpy) that are only
supported on MacOSX and Linux. The above command *should* work if healpy is
Already installed, but if problems occur, see
[PyCall](https://github.com/JuliaPy/PyCall.jl).


