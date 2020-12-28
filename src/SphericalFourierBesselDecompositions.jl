#!/usr/bin/env julia


module SphericalFourierBesselDecompositions

include("lib/HealPy.jl")
include("lib/SciPy.jl")
include("lib/Splines.jl")
include("gnl.jl")  # SphericalBesselGnls
include("modes.jl")  # AnlmModes, ClnnModes, ClnnBinnedModes
include("SeparableArrays.jl")
include("windows.jl")  # window function related
include("theory.jl")  # mostly for testing the package, may be split at some point
include("NDIterators.jl")
include("wigner_chains.jl")  # Wigner symbol related
include("window_chains.jl")  # window function related
include("covariance.jl")  # mostly theory, may be split at some point
include("cat2anlm.jl")

using Statistics
#using Roots
using .HealPy
using .SciPy
using .Splines
using .GNL
using .Modes
using .Windows
using .SeparableArrays
using .Theory
using .Covariance
using .WignerChains
using .WindowChains
using .NDIterators
using .Cat2Anlm

#using QuadGK
#using FastGaussQuadrature
using Distributed
#using Base.Threads

## for testing only
#using MeasureAngularPowerSpectra
#using Profile
#using PyPlot


################# spherical Bessel function jl ##########################3

function sphericalharmonicsy(l, m, θ, ϕ)
    # scipy has ℓ,m and θ,ϕ reversed
    return scipy.special.sph_harm(m, l, ϕ, θ)
end


###################### utility functions #########################

# gen_withcache()
function gen_withcache(nnn, fname, gen_fn)
    dims = length(nnn)
    @assert dims == 2
    nnn_cache = fill(Int64(0), dims)
    arr = fill(Float64(NaN), nnn_cache...)
    if isfile(fname)  # check what's in the cache
        read!(fname, nnn_cache)
    end
    if all(nnn_cache .>= nnn)  # cache is sufficient
        open(fname, "r") do f
            read!(f, nnn_cache)
            arr = read!(f, Array{Float64}(undef, nnn_cache...))
            arr = collect(arr[1:nnn[1],1:nnn[2]])  # can this be generalized to N-dims
        end
    else  # cache is not sufficient
        arr = gen_fn(nnn)
        @assert all(size(arr) .== nnn)
        open(fname, "w") do f
            nnn_cache = [nnn[i] for i=1:dims]  # 'write()' cannot do tuples
            write(f, Int64.(nnn_cache))
            write(f, Float64.(arr))
        end
    end
    return arr
end



####################### a_nlm -> c_lnn #####################################

function alm2cl(alm1, alm2, lmax, lmin=0)
    # Note: our alm are in a different order than Healpix. In our array the
    # (l,m) pairs occur in the order
    # [(0,0),(1,0),(1,1),(2,0),(2,1),(2,2),(3,0),...].
    cl = fill(NaN, lmax+1 - lmin)
    b = 1
    for l=lmin:lmax
        c = real(alm1[b] * alm2[b])
        for m=1:l
            c += 2 * real(alm1[b+m] * conj(alm2[b+m]))
        end
        cl[l+1-lmin] = c / (2 * l + 1)
        b += l + 1
    end
    return cl
end



function amln2clnn(anlm, cmodes::ClnnModes)
    clnn = fill(NaN, getlnnsize(cmodes))
    for n̄=1:cmodes.amodes.nmax, Δn=0:cmodes.Δnmax
        n1 = n̄ + Δn
        n2 = n̄
        lmax = minimum(cmodes.amodes.lmax_n[[n1,n2]])
        lmsize = getlmsize(lmax)
        n1_idxs = getnlmsize(cmodes.amodes, n1 - 1) .+ (1:lmsize)
        n2_idxs = getnlmsize(cmodes.amodes, n2 - 1) .+ (1:lmsize)
        lnn_idxs = getidx.(cmodes, 0:lmax, n1, n2)
        clnn[lnn_idxs] .= alm2cl(anlm[n1_idxs], anlm[n2_idxs], lmax)
    end
    return clnn
end


function pixwin(cmodes)
    hppwin = hp.pixwin(cmodes.amodes.nside, lmax=cmodes.amodes.lmax, pol=false)
    lnnsize = getlnnsize(cmodes)
    sfbpwin = fill(NaN, lnnsize)
    for i=1:lnnsize
        l, = getlnn(cmodes, i)
        sfbpwin[i] = hppwin[l+1]
    end
    return sfbpwin
end


############################ convenience functions for plotting #####################

function get_ck(knl, clkk, l, dn; allmodes=false)
    #@show l, dn, lmin
    lmax, nmax, nmax = size(clkk) .+ (-1,0,0)
    if nmax - dn < 0
        return Float64[], Float64[]
    end
    nmin = (!allmodes && l == 0) ? 2 : 1
    k = fill(NaN, nmax-dn-nmin+1)
    ck = fill(NaN, nmax-dn-nmin+1)
    for n1=nmin:nmax-dn
        n2 = n1 + dn
        k[n1-nmin+1] = (knl[n1,l+1] + knl[n2,l+1]) / 2
        ck[n1-nmin+1] = (clkk[l+1,n1,n2] + clkk[l+1,n2,n1]) / 2
    end
    return k, ck
end


function get_ck_err(knl, clkk, l, dn; allmodes=false)
    k0, cl0 = get_ck(knl, clkk, l, 0; allmodes=allmodes)
    k, cl = get_ck(knl, clkk, l, dn; allmodes=allmodes)

    cl0spline = Spline1D(k0, cl0)
    Δcl² = @. (cl0spline(k)^2 + cl^2) / (2*l + 1)

    return @. √Δcl²
end


function predict_clnn(knl, pk, rmax, lmin=0)
    kernel(k, k_n, l) = begin
        #@show typeof(k) typeof(k_n) typeof(l)
        k^2 * sphericalbesselj(l-1, k * rmax) / (k_n^2 - k^2)
    end
    nmax, lmax = size(knl) .+ (0,lmin-1)
    clnn = fill(NaN, lmax+1-lmin, nmax, nmax)
    for l=lmin:lmin, n1=1:nmax, n2=max(n1-1,1):min(n1+1,nmax)
        n2 = n1
        k_n1 = knl[n1,l+1-lmin]
        k_n2 = knl[n2,l+1-lmin]
        integrand(k) = pk(k) * kernel(k, k_n1, l) * kernel(k, k_n2, l)
        I,E = quadgk(integrand, 0.0, 10 * maximum(knl))  # TODO: infinity?
        clnn[l+1-lmin,n1,n2] = I
    end
    return (4 * rmax / π) * clnn
end

############################ Window function stuff #####################

function gen_mask(nside, fsky)
    npix = hp.nside2npix(nside)
    mask = fill(0.0, npix)
    θ, ϕ = hp.pix2ang(nside, 0:npix-1)
    @show extrema(θ) extrema(ϕ)
    θₘₐₓ = acos(1 - 2*fsky)
    @show θₘₐₓ
    for i=1:length(θ)
        if θ[i] <= θₘₐₓ
            mask[i] = 1.0
        end
    end
    return mask
end


function make_window(wmodes::ConfigurationSpaceModes, features...)
    rmin = wmodes.rmin
    rmax = wmodes.rmax
    r = wmodes.r
    Δr = wmodes.Δr
    nr = wmodes.nr
    nside = wmodes.nside
    @show rmin rmax nr Δr nside

    features = features..., :separable

    phi = fill(1.0, nr)
    mask = fill(1.0, wmodes.npix)

    if :radial in features
        r0 = rmax * 0.55
        for i=1:nr
            phi[i] = exp(- (r[i] / r0)^2)
        end
        features = filter(i -> i != :radial, features)
    end

    if :step_rmin in features
        r0 = rmin + 100.0
        for i=1:nr
            phi[i] = (r[i] < r0) ? 0.0 : phi[i]
        end
        features = filter(i -> i != :step_rmin, features)
    end

    if :step_rmax in features
        r0 = rmax - 100.0
        for i=1:nr
            phi[i] = (r[i] > r0) ? 0.0 : phi[i]
        end
        features = filter(i -> i != :step_rmax, features)
    end


    if :ang_75 in features
        mask = gen_mask(nside, 0.75)
        features = filter(i -> i != :ang_75, features)
    end


    if :ang_half in features
        mask = gen_mask(nside, 1/2)
        features = filter(i -> i != :ang_half, features)
    end


    if :ang_quarter in features
        mask = gen_mask(nside, 1/4)
        features = filter(i -> i != :ang_quarter, features)
    end

    if :ang_eighth in features
        mask = gen_mask(nside, 1/8)
        features = filter(i -> i != :ang_eighth, features)
    end

    win = phi * mask'
    maxwin = maximum(win)

    if :separable in features
        phi ./= maxwin
        win = @SeparableArray phi mask
        features = filter(i -> i != :separable, features)
    else
        win ./= maxwin
    end

    if length(features) != 0
        @error "Did not recognize all features" features
        @assert false
    end

    return win
end


############################ ToDO #################################
# - Remove non-vector lnn representation: may still be needed for plotting.
# - covariance matrix of C: how to plot?
#   * marginalized error bars on Cℓ(n,n')
#   * covariance and correlation matrices

# Done:
# - Use ArbFloats to calculate knl and gnl, move splines inside SphericalBesselGnl
# - window function: calculate from W_nlm. (complicated, needs ∫gnl gnl gnl dr)
# - angular window function:
#   - MASTER approach: I suspect this works great for an indicator window, not
#     so much if we want to allow for LOS-dependent depth.
#   - Samushia angular limits: This is ideal, but potentially a lot more work.
#   - Allow binnings to be a float
# - Samushia in radial direction: rmin and rmax
# - kmin, kmax determine nmin(ℓ), nmax(ℓ), ℓmin, ℓmax
# - (ℓ,m) are already combined into a single dimension. Now include 'n' in that
#   dimension as well. This has the advantage that I can more easily make n(ℓ)
#   non-uniform, and I can more easily represent the mixing matrix as a matrix.
# - Check normalization of transformed window" do we include δ^D(k - k')?
# - Code problems:
#   * Where does (4π)^2/2 come from? Ans: use correct jl normalization
#   * Error that goes as ∝-√ℓ/k^2 Ans: use correct jl normalization
# - Use different boundary condition: potential should be best: continous and
#   smooth at boundary.
# - normalization: Why do I need to divide by nbar? Use ΔΩpix.
# - Why is the constant c different from the monopole? Is this just noise? Use ΔΩpix.
# - Correct pixwin
# - validate with MC


end


# vim: set sw=4 et sts=4 :
