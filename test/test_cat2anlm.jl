#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
SFB = SphericalFourierBesselDecompositions

using Test
using LinearAlgebra
using Profile

win_features = [(), (:separable,)]

@testset "Cat2Anlm f=$features" for features in win_features
    rmin = 500.0
    rmax = 1000.0
    kmax = 0.05
    nbar = 1e-4
    nr = 100
    amodes = SFB.AnlmModes(kmax, rmin, rmax)
    wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)
    win = SFB.make_window(wmodes, features...)
    win_rhat_ln = SFB.win_rhat_ln(win, wmodes, amodes)

    Ngalaxies = round(Int, (2*rmax)^3 * nbar)
    xyz = 2 * rmax * rand(3, Ngalaxies) .- rmax
    rθϕ = Array{Float32}(undef, 3, Ngalaxies)
    for i=1:Ngalaxies
        x⃗ = xyz[:,i]
        r = norm(x⃗)
        n̂ = x⃗ / r
        θ = acos(n̂[3])
        ϕ = atan(n̂[2], n̂[1])
        rθϕ[1,i] = r
        rθϕ[2,i] = θ
        rθϕ[3,i] = ϕ
    end
    rθϕ = SFB.apply_window(rθϕ, win, wmodes)
    @assert all(@. rmin <= rθϕ[1,:] <= rmax)

    anlm = SFB.cat2amln(rθϕ, amodes, nbar, win_rhat_ln)  #compile
    Profile.clear()
    @time @profile SFB.cat2amln(rθϕ, amodes, nbar, win_rhat_ln)
    Profile.print()
end



# vim: set sw=4 et sts=4 :
