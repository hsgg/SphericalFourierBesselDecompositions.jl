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


module Cat2Anlm

export cat2amln, cat2nlm, winweights2galweights
export field2anlm, anlm2field


using SharedArrays
using Healpix
using ProgressMeter
using ..HealpixHelpers
using ..SeparableArrays
using ..MyMathFunctions
using ..Modes

using ..Windows

#using Statistics


function sortout(rθϕ, nside, weight)
    # Sort by r. This helps reduce recomputations in the spline for gnl(r).
    p = sortperm(rθϕ[1,:])
    rθϕ = rθϕ[:,p]

    ## Sort by healpix pixel, r stays sorted within a pixel.
    ## Note: The idea here is that sorting by pixels avoids an awkward access
    ## pattern when creating the healpix map for a given ℓ. However, we don't do
    ## this, as it also thins out the galaxies and that leads to temporary
    ## variables being recomputed much more frequently in the spline for gnl(r).
    #θ = collect(rθϕ[2,:])
    #ϕ = collect(rθϕ[3,:])
    #pix = ang2pixRing(nside, θ, ϕ)
    #p = sortperm(pix)
    #rθϕ = rθϕ[:,p]

    r = collect(rθϕ[1,:])
    θ = collect(rθϕ[2,:])
    ϕ = collect(rθϕ[3,:])
    weight = collect(weight[p])
    return r, θ, ϕ, weight
end


########################### transform jl ##########################################

function transform_gnl(npix, pix, r, sphbesg_nl, weight)
    Ngal = length(pix)
    map = fill(0.0, npix)
    for i=1:Ngal
        #@show pix[i],r[i]
        map[pix[i]] += weight[i] * sphbesg_nl(r[i])
    end
    return map
end


# pmu = pixel management unit
function make_pmu_pmupix(pix)
    pmu = unique(sort(pix))
    pmupix = Array{Int64,1}(indexin(pix, pmu))
    return pmu, pmupix
end


function transform_gnl_spmap!(map, npix, pmu, pmupix, r, sphbesg_nl, weight)
    Ngal = length(pmupix)
    spmap = fill(0.0, length(pmu))
    for i=1:Ngal
        spmap[pmupix[i]] += weight[i] * sphbesg_nl(r[i])
    end
    @. map[pmu] = spmap
    return map
end


function transform_jnl_binned_quadratic(npix, pix, k, l, r, rmax)
    # Note: r must be sorted
    Ngal = length(pix)
    map = fill(0.0, npix)
    # Note: want Δr to be much less than distance between zeros, or k*Δr ≪ π
    nr = 25 * ceil(Int, 2 * k * rmax)
    Δr = rmax / nr
    #@show Δr,1/k,nr
    idx = 1
    ksphbes_prev = sphericalbesselj(l, 0.0)
    for i=1:nr
        rmid = rmax * (i-0.5) / nr
        rhi = rmax * i / nr
        ksphbes_mid = sphericalbesselj(l, k * rmid)
        ksphbes_next = sphericalbesselj(l, k * rhi)
        a = ((ksphbes_prev - ksphbes_mid) + (ksphbes_next - ksphbes_mid)) / Δr^2
        b = (ksphbes_next - ksphbes_prev) / Δr
        c = ksphbes_mid
        while idx <= Ngal && r[idx] <= rhi
            x = r[idx] - rmid
            map[pix[idx]+1] += (a * x + b) * x + c
            idx += 1
        end
        ksphbes_prev = ksphbes_next
    end
    return map
end


function gen_jl_cache(lmax, qmax)
    # Note: We need Δq=kₘₐₓ⋅Δr to be much less than the distance between zeros,
    # or Δq ≪ π. We choose Δq = π / (2π * 4), or about 25 per zero. Then, the
    # number of bins we need is nᵣ = qmax / Δq + 1 = 8 * qmax + 1. Nah, let's
    # do much more until we get convergence for (n,ℓ)=(1,0) mode.
    nr = ceil(Int, 50 * qmax) + 1
    q = range(0.0, qmax, length=nr)
    jl = [sphericalbesselj.(l, q) for l=0:lmax]
    return q, jl
end


function transform_jnl_binned_quadratic_cached(npix, pix, k, l, r, rmax, jl_q, jl)
    # Note: r must be sorted
    Ngal = length(pix)
    map = fill(0.0, npix)
    mynr = findfirst(jl_q .>= k*rmax)
    idx = 1
    rhi = 0.0
    #test_kjl = Float64[]
    #@show nr,mynr,qmax,k,Δr
    for i=2:mynr-1
        rlo = jl_q[i-1] / k
        rmid = jl_q[i] / k
        rhi = jl_q[i+1] / k
        #Ngal_Δr = length(r[@. rlo < r <= rhi])
        #@show i,l,rlo,rmid,rhi,idx,Ngal_Δr
        ksphbes0 = jl[i-1]
        ksphbes1 = jl[i]
        ksphbes2 = jl[i+1]
        t0 = rlo - rmid
        t2 = rhi - rmid
        ksphbes_deriv0 = (ksphbes0 - ksphbes1) / t0
        ksphbes_deriv2 = (ksphbes2 - ksphbes1) / t2
        a = (ksphbes_deriv2 - ksphbes_deriv0) / (t2 - t0)
        b = (t2 * ksphbes_deriv0 - t0 * ksphbes_deriv2) / (t2 - t0)
        c = ksphbes1
        while idx <= Ngal && r[idx] <= rhi && r[idx] <= rmax
            t = r[idx] - rmid
            map[pix[idx]+1] += (a * t + b) * t + c
            #push!(test_kjl, (a * t + b) * t + c)
            idx += 1
        end
    end
    #@assert rhi >= rmax  # numerical round-off
    @assert rhi + eps(rhi) >= rmax
    #figure()
    #plot(r, k * sphericalbesselj.(l, k * r), label="Exact")
    #scatter(jl_q./k, k .* jl, label="Knots")
    #plot(r, test_kjl, label="Approx")
    #plot(r, test_kjl ./ (k * sphericalbesselj.(l, k*r)) .- 1, label="Error")
    #xlabel(L"r")
    #ylabel(L"k\,j_\ell(kr)")
    #legend()
    return map
end


function get_masked_pixels(win_rhat_ln)
    masked_pixels = Int[]
    npix = size(win_rhat_ln,1)
    for i=1:npix
        if all(win_rhat_ln[i,:,:] .== 0)
            push!(masked_pixels, i)
        end
    end
    return masked_pixels
end

function set_masked_pixels!(map, masked_pixels)
    for p in masked_pixels
        map[p] = UNSEEN
    end
end


####################### weight(r) -> weight(gal) ###############################
@doc raw"""
    winweights2galweights(weights, wmodes, rθϕ)

Returns an array with the weight for each galaxy. `weights` is a 2D-array where
the first index goes over `r`, the second over healpix pixel `p`.
"""
function winweights2galweights(weights, wmodes, rθϕ)
    rmin = wmodes.rmin
    rmax = wmodes.rmax
    rmid = wmodes.r
    rbounds = get_rbounds(wmodes)
    Δr = wmodes.Δr
    nside = wmodes.nside
    reso = Resolution(nside)
    Ngal = size(rθϕ,2)
    w = fill(1.0, Ngal)
    for i=1:Ngal
        r, θ, ϕ = rθϕ[:,i]
        pix = ang2pixRing(reso, θ, ϕ)
        bin = (r == rmax) ? length(rmid) : searchsortedlast(rbounds, r)
        if !(1 <= bin <= length(rmid))
            @error "r out of bounds" i r,θ,ϕ bin,pix rmin,rmax length(rmid),extrema(rmid) length(rbounds),extrema(rbounds) size(weights) size(w)
        end
        w[i] = weights[bin,pix]
    end
    return w
end


####################### catalogue -> a_nlm #####################################
@doc raw"""
    cat2amln(rθϕ, amodes, nbar, win_rhat_ln, weights)

Computes the spherical Fourier-Bessel decomposition coefficients from a
catalogue of sources. The number density is measured from the survey as $\bar n
= N_\mathrm{gals} / V_\mathrm{eff}$.

`weights` is an array containing a weight for each galaxy.

# Example
```julia-repl
julia> using SphericalFourierBesselDecompositions
julia> cat2amln(rθϕ, ...)
```
"""
function cat2amln(rθϕ, amodes, nbar, win_rhat_ln, weight=ones(eltype(rθϕ), size(rθϕ,2)))
    T = promote_type(eltype(rθϕ), eltype(win_rhat_ln))
    r, θ, ϕ, weight = sortout(rθϕ, amodes.nside, weight)
    @show nbar length(r)
    sphbesg = amodes.basisfunctions
    knl = amodes.knl
    pix = ang2pixRing.(Ref(Resolution(amodes.nside)), θ, ϕ)
    pmu, pmupix = make_pmu_pmupix(pix)
    #@time masked_pixels = get_masked_pixels(win_rhat_ln)  # doesn't seem to help
    @show typeof(pmu) typeof(pmupix)
    npix = nside2npix(amodes.nside)
    ΔΩpix = 4π / npix
    lmax = amodes.lmax

    map = HealpixMap{T,Healpix.RingOrder}(fill(T(0), npix))
    alm = mymap2alm(map; lmax)
    almref = Ref(alm)
    anlm = fill(NaN+im*NaN, getnlmsize(amodes))

    idxs = fill(0, length(alm))

    for n=1:amodes.nmax
        @show n,amodes.nmax,0:amodes.lmax_n[n]
        #@time @sync @distributed for l=0:amodes.lmax_n[n]
        @time for l=0:amodes.lmax_n[n]
            #@show n,l
            #map0 = transform_gnl(npix, pix, r, sphbesg.gnl[n,l+1], weight)
            map .= 0
            transform_gnl_spmap!(map, npix, pmu, pmupix, r, sphbesg.gnl[n,l+1], weight)
            #@time map2 = transform_jnl_binned_quadratic(npix, pix, knl[n,l+1], l, r, rmax)

            map .*= 1 / (nbar * ΔΩpix)

            c = @view win_rhat_ln[:,l+1,n]
            @. map = map - c

            #set_masked_pixels!(map, masked_pixels)

            # HealPy:
            #alm = hp.map2alm(map, lmax=l, use_weights=true)
            #idx = hp.Alm.getidx.(l, l, 0:l) .+ 1  # python is 0-indexed

            # Healpix.jl:
            mymap2alm!(map, alm)
            idx = @view idxs[1:l+1]
            @. idx = almIndex(almref, l, 0:l)

            baseidx = getidx(amodes, n, l, 0)
            @views @. anlm[baseidx:(baseidx+l)] = alm.alm[idx]
        end
    end
    @assert all(isfinite.(anlm))
    return anlm
end

const cat2nlm = cat2amln


@doc raw"""
    field2anlm(f_xyz, wmodes::ConfigurationSpaceModes, amodes)

Calculate the SFB coefficients for the real field `f_xyz`, where the real
field is in the format of an `nr × npix` matrix.

For `wmodes` see [`ConfigurationSpaceModes`](@ref). For `amodes` see
[`AnlmModes`](@ref).
"""
function field2anlm(f_xyz, wmodes::ConfigurationSpaceModes, amodes)
    rθϕ = fill(0.0, 3, 0)
    nbar = 1.0
    f_rhatln = win_rhat_ln(f_xyz, wmodes, amodes)
    f_nlm = -cat2amln(rθϕ, amodes, nbar, f_rhatln, [])
    return f_nlm
end


@doc raw"""
    anlm2field(f_nlm, wmodes::ConfigurationSpaceModes, amodes)

Transform the SFB coefficients `f_nlm` into configuration space. The resulting
matrix will be of the form `nr × npix`, where `nr` is the number of radial
bins given by `wmodes`, and `npix` is the number of HEALPixels.

For `wmodes` see [`ConfigurationSpaceModes`](@ref). For `amodes` see
[`AnlmModes`](@ref).
"""
function anlm2field(f_nlm, wmodes::ConfigurationSpaceModes, amodes)
    T = real(eltype(f_nlm))
    f_xyz = similar(f_nlm, T, (wmodes.nr, wmodes.npix))

    nlmodes = ClnnModes(amodes; Δnmax=0)
    gnl = fill(T(0), getlnnsize(nlmodes), wmodes.nr)
    for nl=1:getlnnsize(nlmodes)
        l, n, _ = getlnn(nlmodes, nl)
        gnl[nl,:] = amodes.basisfunctions.(n, l, wmodes.r)
    end

    @show size(f_xyz) length(f_xyz)

    # Approach: n-first, r-last
    @time @showprogress for ir=1:wmodes.nr

        alm = Alm(amodes.lmax, amodes.lmax)

        for (i, (l,m)) in enumerate(each_ell_m(alm))

            alm_i = complex(T)(0)

            for n=1:amodes.nmax_l[l+1]
                nl = getidx(nlmodes, l, n)
                nlm = getidx(amodes, n, l, m)
                alm_i += gnl[nl,ir] * f_nlm[nlm]
            end

            alm.alm[i] = alm_i
        end

        hpmap = alm2map(alm, amodes.nside)

        f_xyz[ir,:] .= hpmap
    end

    return f_xyz
end


end


# vim: set sw=4 et sts=4 :
