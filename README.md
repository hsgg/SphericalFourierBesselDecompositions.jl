# SuperFaB

*SuperFaB* is a code for cosmological spherical Fourier-Bessel (SFB) analysis.
The details of the code are presented in [2102.10079](https://arxiv.org/abs/2102.10079).


## Installation

To install *SuperFaB*, start the [Julia](https://julialang.org/) REPL and type
```julia
]add SphericalFourierBesselDecompositions
```
<!-- If that doesn't work (there is a mandatory waiting period for the Julia General registry), install it directly
```julia
]add https://github.com/hsgg/SphericalFourierBesselDecompositions.jl.git
``` -->

The package makes use of some python packages (e.g.
[healpy](https://github.com/healpy/healpy)) that are only supported on MacOSX
and Linux. The above command *should* work if healpy is already installed, but
if problems occur when first using the package, see
[PyCall](https://github.com/JuliaPy/PyCall.jl). Specifically, try
`ENV["PYTHON"]=""; ]build PyCall`.


## Basic Usage

Load the package and create a shortcut
```julia
julia> using SphericalFourierBesselDecompositions
julia> SFB = SphericalFourierBesselDecompositions
```

To perform a SFB decomposition, we create a modes object `amodes` that contains
the modes and basis functions, and for the pseudo-SFB power spectrum we create
`cmodes`,
```julia
julia> amodes = SFB.AnlmModes(kmax, rmin, rmax)
julia> cmodes = SFB.ClnnModes(amodes, Δnmax=0)
```
Here, `kmax` is the maximum k-mode to be calculated, `rmin` and `rmax` are the
radial boundaries. `Δnmax` specifies how many off-diagonals `k ≠ k'` to
include.

The window function is described by
```julia
julia> wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)
julia> win = SFB.SeparableArray(phi, mask, name1=:phi, name2=:mask)
julia> win_rhat_ln = SFB.win_rhat_ln(win, wmodes, amodes)
julia> Veff = SFB.integrate_window(win, wmodes)
```
where `nr` is the number of radial bins, `phi` is an array of length `nr`, and
`mask` is a HEALPix mask. In general, `win` can be a 2D-array, where the first
dimension is radial, and the second dimension is the HEALPix mask at each
radius. Using a `SeparableArray` uses Julia's dispatch mechanism to call more
efficient specialized algorithms when the radial and angular window are
separable. `SFB.win_rhat_ln()` performs the radial transform of the window,
`SFB.integrate_window()` is a convenient way to calculate the effective volume
`Veff`.

The SFB decomposition is now performed with
```julia
julia> anlm = SFB.cat2amln(rθϕ, amodes, nbar, win_rhat_ln)
julia> CNobs = SFB.amln2clnn(anlm, cmodes)
```
where `rθϕ` is a `3 × Ngalaxies` array with the `r`, `θ`, and `ϕ` coordinates
of each galaxy in the survey, and `nbar = Ngalaxies / Veff` is the average
number density. The second line calculates the pseudo-SFB power spectrum.

Shot noise and pixel window are calculated with
```julia
julia> Nobs_th = SFB.win_lnn(win, wmodes, cmodes) ./ nbar
julia> pixwin = SFB.pixwin(cmodes)
```

Window deconvolution is performed with bandpower binning:
```julia
julia> w̃mat, vmat = SFB.bandpower_binning_weights(cmodes; Δℓ=Δℓ, Δn=Δn)
julia> bcmodes = SFB.ClnnBinnedModes(w̃mat, vmat, cmodes)
julia> bcmix = SFB.power_win_mix(win, w̃mat, vmat, wmodes, bcmodes)
julia> Cobs = @. (CNobs - Nobs_th) / pixwin ^ 2
julia> C = bcmix \ (w̃mat * Cobs)
```
The first line calculates binning matrices `w̃` and `v` for bin sizes `Δℓ ~
1/fsky` and `Δn = 1`, the second line describes modes similar to `cmodes` but
for bandpower binned modes. The coupling matrix is calculated in the third
line, the shot noise and pixel window are corrected in the fourth line, and the
last line does the binning and deconvolves the window function.

To compare with a theoretical prediction, we calculate the deconvolved binning
matrix `wmat`,
```julia
julia> using LinearAlgebra
julia> wmat = bcmix * SFB.power_win_mix(win, w̃mat, I, wmodes, bcmodes)
```

The modes of the pseudo-SFB power spectrum are given by
```julia
julia> lkk = SFB.getlkk(bcmodes)
```
where for a given `i` the element `lkk[1,i]` is the ℓ-mode, `lkk[2,i]` is the
`n`-mode, `lkk[3,i]` is the `n'`-mode of the pseudo-SFB power spectrum element
`C[i]`.

An unoptimized way to calculate the covariance matrix is
```julia
julia> VW = SFB.calc_covariance_exact_chain(C_th, nbar, win, wmodes, cmodes)
julia> V = inv(bcmix) * w̃mat * VW * w̃mat' * inv(bcmix)'
```
