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
already installed, but if problems occur when first using the package, see
[PyCall](https://github.com/JuliaPy/PyCall.jl). Specifically, try
`ENV["PYTHON"]=""; ]build PyCall`.


## Basic Usage

Load the package:
```julia
julia> using SphericalFourierBesselDecompositions
julia> SFB = SphericalFourierBesselDecompositions
```
The second line is only for convenience, but will be assumed for the rest of
this document.

To perform a SFB decomposition, a few caches need to be created that contain
the basis functions
```julia
julia> amodes = SFB.AnlmModes(kmax, rmin, rmax)
julia> cmodes = SFB.ClnnModes(amodes, Δnmax=0)
```
Here, `kmax` is the maximum k-mode to be calculated, `rmin` and `rmax` are the
radial boundaries. `amodes` is used for the SFB decomposition, `cmodes` is used
to calculate the SFB power spectrum, and `Δnmax` specifies how many
off-diagonals `k ≠ k'` to include.

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
radius. Using a `SeparableArray` makes dispatches to more efficient specialized
algorithms when the radial and angular window are separable.
`SFB.win_rhat_ln()` performs the radial transform of the window,
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
line, and the last line could also be written `C = inv(bcmix) * w̃mat * Cobs`.

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
`lkk[1,:]` are the ℓ-modes, `lkk[2,:]` are the `n`-modes, `lkk[3,:]` the
`n'`-modes for the pseudo-SFB power spectrum `C[:]`.

An unoptimized way to calculate the covariance matrix is
```julia
julia> VW = SFB.calc_covariance_exact_chain(C_th, nbar, win, wmodes, cmodes)
julia> V = inv(bcmix) * w̃mat * VW * w̃mat' * inv(bcmix)'
```
