# This module includes some convenience functions that help in the transition
# from healpy to healpix.jl.
module HealpixHelpers

export mymap2alm

using Healpix


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


function mymap2alm(map::HealpixMap; lmax=3*npix2nside(length(map))-1)
    map2 = deepcopy(map)
    #applyFullWeights!(map2)  # only nside>=32 is supported
    #return map2alm(map2, lmax=lmax, niter=0)
    return map2alm(map2, lmax=lmax)
end


end


# vim: set sw=4 et sts=4 :
