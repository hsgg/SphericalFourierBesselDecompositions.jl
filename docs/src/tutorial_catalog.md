# Tutorial: Catalogue and Window

In this tutorial we assume that your basic starting point is that you have a
catalogue of galaxy positions $(r,\theta,\phi)$ and some way to calculate the
window function, for example each voxel being proportional to the number of
galaxies in a random catalogue.

Further, we assume that
[SphericalFourierBesselDecompositions.jl](https://github.com/hsgg/SphericalFourierBesselDecompositions.jl)
is already installed, and that you have a basic understanding of what the
package is supposed to do.

First, load the package and create a shortcut
```julia
using SphericalFourierBesselDecompositions
SFB = SphericalFourierBesselDecompositions
```
We will always assume that this shortcut has been created, as the package `SFB`
does not export any symbols itself.

To perform a SFB decomposition, we create a modes object `amodes` that contains
the modes and basis functions, and for the pseudo-SFB power spectrum we create
`cmodes`,
```julia
kmax = 0.05
rmin = 500.0
rmax = 1500.0
amodes = SFB.AnlmModes(kmax, rmin, rmax)
cmodes = SFB.ClnnModes(amodes, Δnmax=0)
```
Here, `kmax` is the maximum k-mode to be calculated, `rmin` and `rmax` are the
radial boundaries. `Δnmax` specifies how many off-diagonals `k ≠ k'` to
include.

The objects `amodes` and `cmodes` are used to access elements in the arrays
produced by the routines below. For example, performing the SFB transform on a
catalog, we use `anlm = SFB.cat2anlm(...)`, and the individual modes can be
accessed via `anlm[SFB.getidx(amodes, n, l, m)]`.

The window function is described by
```julia
nr = 50
wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)
win = SFB.SeparableArray(phi, mask, name1=:phi, name2=:mask)
win_rhat_ln = SFB.win_rhat_ln(win, wmodes, amodes)
Veff = SFB.integrate_window(win, wmodes)
```
where `nr` is the number of radial bins, `phi` is an array of length `nr`, and
`mask` is a HEALPix mask in ring order. In general, `win` can be a 2D-array,
where the first dimension is radial, and the second dimension is the HEALPix
mask at each radius. Using a `SeparableArray` uses Julia's dispatch mechanism
to call more efficient specialized algorithms when the radial and angular
window are separable. `SFB.win_rhat_ln()` performs the radial transform of the
window, `SFB.integrate_window()` is a convenient way to calculate the effective
volume `Veff`.

The SFB decomposition for a catalogue of galaxies is now performed with
```julia
anlm = SFB.cat2amln(rθϕ, amodes, nbar, win_rhat_ln)
CNobs = SFB.amln2clnn(anlm, cmodes)
```
where `rθϕ` is a `3 × Ngalaxies` array with the `r`, `θ`, and `ϕ` coordinates
of each galaxy in the survey, and `nbar = Ngalaxies / Veff` is the average
number density. The second line calculates the pseudo-SFB power spectrum.

Shot noise and pixel window are removed with
```julia
Nobs_th = SFB.win_lnn(win, wmodes, cmodes) ./ nbar
pixwin = SFB.pixwin(cmodes)
Cobs = @. (CNobs - Nobs_th) / pixwin ^ 2
```

Window deconvolution is performed with bandpower binning:
```julia
w̃mat, vmat = SFB.bandpower_binning_weights(cmodes; Δℓ=Δℓ, Δn=Δn)
bcmodes = SFB.ClnnBinnedModes(w̃mat, vmat, cmodes)
bcmix = SFB.power_win_mix(win, w̃mat, vmat, wmodes, bcmodes)
C = bcmix \ (w̃mat * Cobs)
```
The first line calculates binning matrices `w̃` and `v` for bin sizes `Δℓ ~
1/fsky` and `Δn = 1`, the second line describes modes similar to `cmodes` but
for bandpower binned modes. The coupling matrix is calculated in the third
line, and the last line does the binning and deconvolves the window function.

To compare with a theoretical prediction, we calculate the deconvolved binning
matrix `wmat`,
```julia
using LinearAlgebra
wmat = bcmix * SFB.power_win_mix(win, w̃mat, I, wmodes, bcmodes)
```

The modes of the pseudo-SFB power spectrum are given by
```julia
lkk = SFB.getlkk(bcmodes)
```
where for a given `i` the element `lkk[1,i]` is the ℓ-mode, `lkk[2,i]` is the
`n`-mode, `lkk[3,i]` is the `n'`-mode of the pseudo-SFB power spectrum element
`C[i]`.

An unoptimized way to calculate the covariance matrix is
```julia
VW = SFB.calc_covariance_exact_chain(C_th, nbar, win, wmodes, cmodes)
V = inv(bcmix) * w̃mat * VW * w̃mat' * inv(bcmix)'
```
