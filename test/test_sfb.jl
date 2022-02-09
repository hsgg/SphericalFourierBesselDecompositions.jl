using SphericalFourierBesselDecompositions
const SFB = SphericalFourierBesselDecompositions

using Test


@testset "top-level" begin
    rmin = 500.0
    rmax = 1000.0
    kmax = 0.02
    amodes = SFB.AnlmModes(kmax, rmin, rmax)
    cmodes = SFB.ClnnModes(amodes, Δnmax=0)
    pixwin = SFB.pixwin(cmodes)
end


# vim: set sw=4 et sts=4 :
