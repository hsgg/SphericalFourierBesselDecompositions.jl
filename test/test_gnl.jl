#!/usr/bin/env julia

using SphericalFourierBesselDecompositions
const SFB = SphericalFourierBesselDecompositions

using Test

@testset "GNL" begin
    rmin = 0.0
    rmax = 2000.0

    kmax1 = 0.0614
    amodes = SFB.AnlmModes(kmax1, rmin, rmax, cache=false)
    cmodes = SFB.ClnnModes(amodes, Δnmax=0)
    knl1 = amodes.knl[isfinite.(amodes.knl)]
    lkk1 = SFB.getlkk(cmodes)
    @test all(@. knl1 < kmax1)
    @test all(@. lkk1[2,:] < kmax1)
    @test all(@. lkk1[3,:] < kmax1)

    kmax2 = 0.1
    amodes = SFB.AnlmModes(kmax2, rmin, rmax, cache=false)
    cmodes = SFB.ClnnModes(amodes, Δnmax=0)
    knl2 = amodes.knl[isfinite.(amodes.knl)]
    lkk2 = SFB.getlkk(cmodes)
    @test all(@. knl2 < kmax2)
    @test all(@. lkk2[2,:] < kmax2)
    @test all(@. lkk2[3,:] < kmax2)

    s = @. knl2 < kmax1
    @show length(knl1) length(knl2[s])


    SFB.GNL.calc_knl_potential(0.05, 500.0, 1000.0; nmax=10, lmax=10)
end






# vim: set sw=4 et sts=4 :
