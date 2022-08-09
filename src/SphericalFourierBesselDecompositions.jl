# Copyright (c) 2020 California Institute of Technology (“Caltech”). U.S.
# Government sponsorship acknowledged.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   • Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   • Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   • Neither the name of Caltech nor its operating division, the Jet
#   Propulsion Laboratory, nor the names of its contributors may be used to
#   endorse or promote products derived from this software without specific
#   prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


module SphericalFourierBesselDecompositions

include("HealPy.jl")
include("healpix_helpers.jl")
include("Splines.jl")
include("gnl.jl")  # SphericalBesselGnls
include("modes.jl")  # AnlmModes, ClnnModes, ClnnBinnedModes
include("SeparableArrays.jl")
include("MyBroadcast.jl")
include("LMcalcStructs.jl")
include("windows.jl")  # window function related
include("NDIterators.jl")
include("cat2anlm.jl")
include("wigner_chains.jl")  # Wigner symbol related
include("window_chains.jl")  # window function related
include("theory.jl")  # mostly for testing the package, may be split at some point
include("covariance.jl")  # mostly theory, may be split at some point

using Statistics
using FastTransforms
using WignerD
#using Roots
using Healpix
using Scanf
using .MyBroadcast
using .HealPy
using .HealpixHelpers
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
using PyCall

## for testing only
#using MeasureAngularPowerSpectra
#using Profile
#using PyPlot


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
    nmax = cmodes.amodes.nmax
    for n̄=1:nmax, Δn=0:cmodes.Δnmax
        n1 = n̄
        n2 = n̄ + Δn
        if n1 > nmax || n2 > nmax
            continue
        end
        lmax = minimum(cmodes.amodes.lmax_n[[n1,n2]])
        lmsize = getlmsize(lmax)
        n1_idxs = getnlmsize(cmodes.amodes, n1 - 1) .+ (1:lmsize)
        n2_idxs = getnlmsize(cmodes.amodes, n2 - 1) .+ (1:lmsize)
        lnn_idxs = getidx.(cmodes, 0:lmax, n1, n2)
        clnn[lnn_idxs] .= alm2cl(anlm[n1_idxs], anlm[n2_idxs], lmax)
    end
    return clnn
end


@doc raw"""
    pixwin(cmodes)

    Return the angular pixel window for application to a Clnn object.
"""
function pixwin(cmodes)
    hppwin = Healpix.pixwin(cmodes.amodes.nside, pol=false)
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
    npix = nside2npix(nside)
    mask = fill(0.0, npix)
    θ, ϕ = pix2angRing(nside, 1:npix)
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

function gen_win_insep_cossin(mask, rmin, rmax, r, l, m)
    nr = length(r)
    win = fill(0.0, nr, length(mask))
    nside = npix2nside(length(mask))
    pix = (1:length(mask))[mask .!= 0]
    θ, ϕ = pix2angRing(nside, pix)
    @show extrema(ϕ)
    θmin, θmax = extrema(θ)
    dr = rmax - rmin
    dθ = θmax - θmin
    rmid = (rmax + rmin) / 2
    #@show l,m dr dθ
    r0fn(θ, ϕ) = begin
        rmid - (dr/2) * cos(θ * l*π/dθ) * cos(m*ϕ)
    end
    phifn(r,θ,ϕ) = exp(- ((r - r0fn(θ,ϕ)) / dr)^2)
    for i=1:length(pix)
        #win[:,pix[i]] .= r0fn(θ[i], ϕ[i])
        win[:,pix[i]] .= phifn.(r, θ[i], ϕ[i])
    end
    win ./= maximum(win)
    return win
end

function rotate_euler!(alm::Alm, α, β, γ)
    for l=0:alm.lmax
        @show l
        Dlmm = wignerD(l, α, β, γ)
        m = 0:l
        ii = almIndex.(Ref(alm), l, m)
        ii_negm = 1:l
        ii_posm = l+1:2*l+1  # also includes m = 0
        alm_posm = alm[ii]
        alm_negm = (@. (-1)^m * conj(alm_posm))[end:-1:2]
        C = Dlmm[ii_posm,ii_negm]
        D = Dlmm[ii_posm,ii_posm]
        alm.alm[ii] .= C * alm_negm + D * alm_posm
    end
    return alm
end

function rotate_euler(mask::HealpixMap, α, β, γ)  # only one not modifying its input
    nside = npix2nside(length(mask))
    alm = map2alm(mask)
    rotate_euler!(alm, α, β, γ)
    mask_rot = alm2map(alm, nside)
    return mask_rot
end

function rotate_euler!(win::SeparableArray, α, β, γ)
    hpmask = HealpixMap{eltype(win),Healpix.RingOrder}(win.mask)
    win.mask .= rotate_euler(hpmask, α, β, γ)
    win.mask ./= maximum(win.mask)
    return win
end

function rotate_euler!(win, α, β, γ)
    for i=1:nr
        hpmask = HealpixMap{eltype(win),Healpix.RingOrder}(win[i,:])
        win[i,:] .= rotate_euler(hpmask, α, β, γ)
    end
    return win
end

function make_window(wmodes::ConfigurationSpaceModes, features...)
    rmin = wmodes.rmin
    rmax = wmodes.rmax
    r = wmodes.r
    Δr = wmodes.Δr
    nr = wmodes.nr
    nside = wmodes.nside
    @show rmin rmax nr Δr nside

    #features = features..., :separable

    phi = fill(1.0, nr)
    mask = fill(1.0, wmodes.npix)
    win = fill(1.0, nr, length(mask))
    features = filter(i -> i != :fullsky, features)

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

    if :ang_sixteenth in features
        mask = gen_mask(nside, 1/16)
        features = filter(i -> i != :ang_sixteenth, features)
    end

    if :rotate in features
        if hp != PyNULL()
            rot = hp.Rotator(coord=["E", "G"])
            mask = rot.rotate_map_pixel(mask)
        else
            a = -0.0004052885
            b = 1.05048844473
            c = 1.68221794936
            mask = HealpixMap{Float64,Healpix.RingOrder}(mask)
            mask = rotate_euler(mask, a, b, c)
        end
        features = filter(i -> i != :rotate, features)
    end

    if :flip in features
        maskflipped = deepcopy(mask)
        maskflipped .= 0
        reso = Resolution(nside)
        for i=1:length(mask)
            θ, ϕ = pix2angRing(reso, i)
            θ = π - θ
            p = ang2pixRing(reso, θ, ϕ)
            maskflipped[p] = mask[i]
        end
        features = filter(i -> i != :flip, features)
    end


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

    win = phi * mask'
    maxwin = maximum(win)

    if :separable in features
        phi ./= maxwin
        win = @SeparableArray phi mask
        features = filter(i -> i != :separable, features)
    else
        win ./= maxwin
    end

    # Todo: The code structure below here makes more sense. To use, it
    # successively incorporate features from above. However, the above assumes
    # the order it is in, so start from the bottom!

    for feat in features
        sfeat = string(feat)
        println("Processing feature $feat...")

        if occursin("radial_cossin_l", sfeat)
            numform, l, m = @scanf(sfeat, "radial_cossin_l%f_m%f", Float64, Float64)
            @assert numform == 2
            l /= 10
            m /= 10
            @show numform,l,m
            win = gen_win_insep_cossin(mask, rmin, rmax, r, l, m)
        end

        if occursin("rotate_", sfeat)
            numform, α, β, γ = @scanf(sfeat, "rotate_%f_%f_%f", Float64, Float64, Float64)
            @assert numform == 3
            α *= π/180
            β *= π/180
            γ *= π/180
            @show α,β,γ
            rotate_euler!(win, α, β, γ)
        end

        if feat == :binary_mask
            if typeof(win) <: SeparableArray
                for i=1:length(win.mask)
                    win.mask[i] = (win.mask[i] > 0.5) ? 1 : 0
                end
            else
                for i=1:nr
                    for j=1:size(win,2)
                        win[i,j] = (win[i,j] > 0.5) ? avg : 0
                    end
                end
            end
        end

        features = filter(i -> i != feat, features)
    end

    @assert maximum(win) == 1

    return win
end


@doc raw"""
    calc_fsky(win, wmodes)

This functions returns a measure of the sky fraction covered by the survey with
window `win`. The exact implementation is considered an implementation detail
and can change in the future.
"""
function calc_fsky(win, wmodes)
    meanmask = mean(win, dims=1)[:]
    fsky = sum(meanmask .> 0) ./ length(meanmask)
end


@doc raw"""
    xyz2rtp(xyz)

Convert the Cartesian positions in `xyz` to spherical coordinates `rθϕ`. The
first dimension is of length 3, the second is the number of galaxies. Assumes a
flat geometry.
"""
function xyz2rtp(xyz)
    Ngalaxies = size(xyz,2)
    rθϕ = Array{eltype(xyz)}(undef, 3, Ngalaxies)
    for i=1:Ngalaxies
        x = xyz[1,i]
        y = xyz[2,i]
        z = xyz[3,i]
        r = √(x^2 + y^2 + z^2)
        θ = acos(z/r)
        ϕ = atan(y/r, x/r)
        rθϕ[1,i] = r
        rθϕ[2,i] = θ
        rθϕ[3,i] = ϕ
    end
    return rθϕ
end


function sphericalharmonicsy(l, m, θ, ϕ)
    rY1 = sphevaluate(θ, ϕ, l, abs(m))
    rY2 = sphevaluate(θ, ϕ, l, -abs(m))
    if m < 0
        return (rY1 - im*rY2) / √2
    elseif m == 0
        return rY1 + 0*im
    else
        return (-1)^m * (rY1 + im*rY2) / √2
    end
end


function realsphericalharmonicsy(l, m, θ, ϕ)
    #Ylm = sphericalharmonicsy(l, m, θ, ϕ)
    #if m < 0
    #    return √2 * (-1)^m * imag(Ylm)
    #elseif m == 0
    #    return real(Ylm)
    #else
    #    return √2 * (-1)^m * real(Ylm)
    #end
    return sphevaluate.(θ, ϕ, l, m)
end


function get_full_basisfuncreal_nlm(amodes, wmodes, n, l, m)
    gnl = amodes.basisfunctions.gnl[n,l+1].(wmodes.r)
    θ, ϕ = pix2angRing(amodes.nside, 1:nside2npix(amodes.nside))
    ylm = realsphericalharmonicsy.(l, m, θ, ϕ)
    return @SeparableArray gnl ylm
end


end


# vim: set sw=4 et sts=4 :
