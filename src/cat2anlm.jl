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

export cat2amln, winweights2galweights
export field2anlm


using SharedArrays
using Healpix
using ..HealpixHelpers
using ..SeparableArrays
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


function transform_gnl_spmap(npix, pmu, pmupix, r, sphbesg_nl, weight)
    Ngal = length(pmupix)
    spmap = fill(0.0, length(pmu))
    for i=1:Ngal
        spmap[pmupix[i]] += weight[i] * sphbesg_nl(r[i])
    end
    map = fill(0.0, npix)
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
    anlm = fill(NaN+im*NaN, getnlmsize(amodes))
    #anlm = SharedArray{Complex{Float64}}(getnlmsize(amodes))
    #anlm .= NaN+NaN*im
    for n=1:amodes.nmax
        @show n,amodes.nmax,0:amodes.lmax_n[n]
        #@time @sync @distributed for l=0:amodes.lmax_n[n]
        @time for l=0:amodes.lmax_n[n]
            #@show n,l
            #map0 = transform_gnl(npix, pix, r, sphbesg.gnl[n,l+1], weight)
            map1 = transform_gnl_spmap(npix, pmu, pmupix, r, sphbesg.gnl[n,l+1], weight)
            #@time map2 = transform_jnl_binned_quadratic(npix, pix, knl[n,l+1], l, r, rmax)
            #map3 = transform_jnl_binned_quadratic_cached(npix, pix, knl[n,l+1], l, r, rmax, jl_q, jl[l+1])
            map = map1
            #@show map
            #@show extrema(map1 ./ map0 .- 1)
            #@show extrema(map2 ./ map0 .- 1)
            #@show extrema(map3 ./ map0 .- 1)
            #readline()
            #close("all")
            #@show mean(map ./ map1),std(map ./ map1)
            #@show mean(map ./ map2),std(map ./ map2)
            #@show mean(map ./ map3),std(map ./ map3)

            map .*= 1 / (nbar * ΔΩpix)
            mapT = convert(Vector{T}, map)

            c = @view win_rhat_ln[:,l+1,n]
            #@show size(map) size(c)
            #@show n,l mean(map) mean(c)
            #@show mean(map[c .!= 0]) mean(c[c .!= 0])
            #@show mean(map[c .== 0]) mean(c[c .== 0])
            @. mapT = mapT - c
            #@show n,l,mean(map),median(map)
            #@show map

            # TODO: check manual implementation for a single ℓ, check libsharp,
            #       how does healpix do it?
            # TODO: start with small nside for small ℓ, dangerous for pixel window

            #set_masked_pixels!(map, masked_pixels)

            # HealPy:
            #alm = hp.map2alm(map, lmax=l, use_weights=true)
            #idx = hp.Alm.getidx.(l, l, 0:l) .+ 1  # python is 0-indexed

            # Healpix.jl:
            maphp = HealpixMap{T,Healpix.RingOrder}(mapT)
            alm = mymap2alm(maphp, lmax=l)
            idx = almIndex(alm, l, 0:l)

            baseidx = getidx(amodes, n, l, 0)
            @views @. anlm[baseidx:(baseidx+l)] = alm.alm[idx]
        end
    end
    @assert all(isfinite.(anlm))
    return anlm
end


# field2anlm(): Convenience function to transform a given field (not catalog)
# into SFB space. It can probably be optimized quite a bit.
# Unfortunately, the inverse is not straightforward, because of the layout of
# `f_nlm` as a vector and `f` as a matrix.
function field2anlm(f, wmodes::ConfigurationSpaceModes, amodes)
    rθϕ = fill(0.0, 3, 0)
    nbar = 1.0
    f_rhatln = win_rhat_ln(f, wmodes, amodes)
    f_nlm = -cat2amln(rθϕ, amodes, nbar, f_rhatln, [])
    return f_nlm
end



end


# vim: set sw=4 et sts=4 :
