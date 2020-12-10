#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
SFB = SphericalFourierBesselDecompositions

using Test


@testset "Covariance Matrices" begin
    rmin = 500.0
    rmax = 1000.0
    amodes = SFB.AnlmModes(3, 5, rmin, rmax)
    cmodes = SFB.ClnnModes(amodes)
    wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)
    win = SFB.make_window(wmodes, :radial, :ang_quarter)

    CNlnn
end


# vim: set sw=4 et sts=4 :
