# This module includes some convenience functions that help in the transition
# from healpy to healpix.jl.
module HealpixHelpers

export mymap2alm

using Healpix
#using ..HealPy


######## Base interfaces

Base.ndims(map::HealpixMap) = 1

Base.ndims(alm::Alm) = ndims(alm.alm)
Base.size(alm::Alm) = size(alm.alm)
Base.size(alm::Alm, d) = size(alm.alm, d)
Base.length(alm::Alm) = length(alm.alm)
Base.iterate(alm::Alm) = iterate(alm.alm)
Base.iterate(alm::Alm, i) = iterate(alm.alm, i)
Base.getindex(alm::Alm, i) = alm.alm[i]


######## Healpix.jl piracy

function Healpix.pix2angRing(nside::Integer, pix::AbstractArray)
    reso = Resolution(nside)
    θ = fill(NaN, length(pix))
    ϕ = fill(NaN, length(pix))
    for i=1:length(pix)
        θ[i], ϕ[i] = pix2angRing(reso, pix[i])
    end
    return θ, ϕ
end


function Healpix.udgrade(map::Vector, new_nside::Integer)
    hpmap = HealpixMap{Float64,Healpix.RingOrder}(map)
    newmap = udgrade(hpmap, new_nside)
    return newmap  # will usually want to keep it as a HealpixMap
end


######## convenience functions

function mymap2alm_healpixjl(map::HealpixMap; lmax=3*map.resolution.nside-1)
    #if map.resolution.nside >= 32
    #    # Note: This makes the field2anlm() test fail:
    #    map2 = deepcopy(map)
    #    applyFullWeights!(map2)  # only nside>=32 is supported
    #    return map2alm(map2, lmax=lmax, niter=0)
    #end
    return map2alm(map, lmax=lmax)
end

#function mymap2alm_healpy(map::HealpixMap; lmax=3*map.resolution.nside-1)
#    alm = hp.map2alm(Vector(map), lmax=lmax, use_weights=true)
#    return Alm(lmax, lmax, Vector(alm))
#end

mymap2alm = mymap2alm_healpixjl
#mymap2alm = mymap2alm_healpy


end


# vim: set sw=4 et sts=4 :
