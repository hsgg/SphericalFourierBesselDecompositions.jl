# Tutorial: Radial Basis Functions

In this tutorial we will show how to obtain the radial basis functions
$g_{n\ell}(r)$, for given $k_{\rm max}$, $r_{\rm min}$, and $r_{\rm max}$.

Further, we assume that
[SphericalFourierBesselDecompositions.jl](https://github.com/hsgg/SphericalFourierBesselDecompositions.jl)
is already installed, and that you have a basic understanding of what the
package is supposed to do.

First, load the package and create a shortcut
```julia
using SphericalFourierBesselDecompositions
const SFB = SphericalFourierBesselDecompositions
```
We will always assume that this shortcut has been created, as the package `SFB`
does not export any symbols itself. Make the shortcut `const` can be important
for performance.

To perform a SFB decomposition, we create a modes object `amodes`. That will construct
the `n,l,m`-modes and basis functions.
```julia
kmax = 0.05
rmin = 500.0
rmax = 1500.0
amodes = SFB.AnlmModes(kmax, rmin, rmax)
```
Here, `kmax` is the maximum k-mode to be calculated, `rmin` and `rmax` are the
radial boundaries.

To access the radial basis functions `gnl(r)`, use
```julia
r = 555.5
rr = range(500.0, 1000.0, length=100)
gnl_r = amodes.basisfunctions.gnl[n,l+1](r)
gnlr = @. amodes.basisfunctions.gnl[n,l+1](rr)
```
