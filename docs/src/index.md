```@meta
CurrentModule = SphericalFourierBesselDecompositions
```

# SuperFaB Documentation

The purpose of this [Julia](https://julialang.org/) module is to provide an
efficient implementation for spherical Fourier-Bessel decomposition of a scalar
field. The details of the algorithm are presented in
[2102.10079](https://arxiv.org/abs/2102.10079). SuperFaB is implemented in the
package
[SphericalFourierBesselDecompositions.jl](https://github.com/hsgg/SphericalFourierBesselDecompositions.jl).


## Contents

```@contents
```


## Installation

To install *SuperFaB*, start the [Julia](https://julialang.org/) REPL and type
```julia
]add SphericalFourierBesselDecompositions
```

```@meta
# If that doesn't work (there is a mandatory waiting period for the Julia
# General registry), install it directly
# ```julia
# ]add https://github.com/hsgg/SphericalFourierBesselDecompositions.jl.git
# ```
```

The package makes use of some python packages (i.e.
[healpy](https://github.com/healpy/healpy)) that are only supported on MacOSX
and Linux. The above command *should* work if healpy is already installed, but
if problems occur when first using the package, see
[PyCall](https://github.com/JuliaPy/PyCall.jl). Specifically, if you do
`ENV["PYTHON"]=""; ]build PyCall`, then Julia will download its own python and
use Conda to download `healpy` and dependencies once you load *SuperFaB*.
