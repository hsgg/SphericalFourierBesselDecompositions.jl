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

export cat2amln


using SharedArrays
using Healpix
#using ..HealPy
using ..SeparableArrays
using ..Modes


function sortout(rθϕ, nside)
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
    return r, θ, ϕ
end


########################### transform jl ##########################################

function transform_gnl(npix, pix, r, sphbesg_nl)
    Ngal = length(pix)
    map = fill(0.0, npix)
    for i=1:Ngal
        #@show pix[i],r[i]
        map[pix[i]] += sphbesg_nl(r[i])
    end
    return map
end


# pmu = pixel management unit
function make_pmu_pmupix(pix)
    pmu = unique(sort(pix))
    pmupix = Array{Int64,1}(indexin(pix, pmu))
    return pmu, pmupix
end


function transform_gnl_spmap(npix, pmu, pmupix, r, sphbesg_nl)
    Ngal = length(pmupix)
    spmap = fill(0.0, length(pmu))
    for i=1:Ngal
        spmap[pmupix[i]] += sphbesg_nl(r[i])
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


####################### catalogue -> a_nlm #####################################
@doc raw"""
    cat2amln(rθϕ, amodes, nbar, win_rhat_ln)

Computes the spherical Fourier-Bessel decomposition coefficients from a
catalogue of sources. The number density is measured from the survey as $\bar n
= N_\mathrm{gals} / V_\mathrm{eff}$.

# Example
```julia-repl
julia> using SphericalFourierBesselDecompositions
julia> cat2amln(rθϕ, ...)
```
"""
function cat2amln(rθϕ, amodes, nbar, win_rhat_ln)
    r, θ, ϕ = sortout(rθϕ, amodes.nside)
    @show nbar length(r)
    sphbesg = amodes.basisfunctions
    knl = amodes.knl
    pix = ang2pixRing.(Ref(Resolution(amodes.nside)), θ, ϕ) # .+ 1  # python is 0-indexed
    pmu, pmupix = make_pmu_pmupix(pix)
    @show typeof(pmu) typeof(pmupix)
    npix = nside2npix(amodes.nside)
    ΔΩpix = 4π / npix
    #anlm = fill(NaN+im*NaN, getnlmsize(amodes))
    anlm = SharedArray{Complex{Float64}}(getnlmsize(amodes))
    anlm .= NaN+NaN*im
    for n=1:amodes.nmax
        @show n,amodes.nmax,0:amodes.lmax_n[n]
        #@time @sync @distributed for l=0:amodes.lmax_n[n]
        @time for l=0:amodes.lmax_n[n]
            #@show n,l
            #map0 = transform_gnl(npix, pix, r, sphbesg.gnl[n,l+1])
            map1 = transform_gnl_spmap(npix, pmu, pmupix, r, sphbesg.gnl[n,l+1])
            #@time map2 = transform_jnl_binned_quadratic(npix, pix, knl[n,l+1], l, r, rmax)
            #map3 = transform_jnl_binned_quadratic_cached(npix, pix, knl[n,l+1], l, r, rmax, jl_q, jl[l+1])
            map = map1
            #@show map
            #@show extrema(map1 ./ map0 .- 1)
            #@show extrema(map2 ./ map0 .- 1)
            #@show extrema(map3 ./ map0 .- 1)
            #hp.mollview(map0, title="direct")
            #hp.mollview(map1, title="direct, splined")
            #hp.mollview(map2, title="quadratic")
            #hp.mollview(map3, title="quadratic cached")
            #hp.mollview(map0 .- map, title="map1 - map")
            #hp.mollview(map1 .- map, title="map1 - map")
            #hp.mollview(map2 .- map, title="map2 - map")
            #hp.mollview(map3 .- map, title="map3 - map")
            #readline()
            #close("all")
            #@show mean(map ./ map1),std(map ./ map1)
            #@show mean(map ./ map2),std(map ./ map2)
            #@show mean(map ./ map3),std(map ./ map3)

            map .*= 1 / (nbar * ΔΩpix)

            c = win_rhat_ln[:,l+1,n]
            #@show size(map) size(c)
            #@show map c mean(map) median(map)
            @. map = map - c
            #@show n,l,mean(map),median(map)
            #@show map

            # TODO: check manual implementation for a single ℓ, check libsharp,
            #       how does healpix do it?
            # TODO: start with small nside for small ℓ, dangerous for pixel window

            # HealPy:
            #alm = hp.map2alm(map, lmax=l, use_weights=true)
            #idx = hp.Alm.getidx.(l, l, 0:l) .+ 1  # python is 0-indexed

            # Healpix.jl:
            maphp = HealpixMap{Float64,Healpix.RingOrder}(map)
            alm = map2alm(maphp, lmax=l)  # TODO: readFullWeights, applyFullWeights!
            idx = almIndex(alm, l, 0:l)

            baseidx = getidx(amodes, n, l, 0)
            @. anlm[baseidx:(baseidx+l)] = alm.alm[idx]
        end
    end
    @assert all(isfinite.(anlm))
    return anlm
end



end


# vim: set sw=4 et sts=4 :
