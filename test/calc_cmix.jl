using Revise
using SphericalFourierBesselDecompositions
const SFB = SphericalFourierBesselDecompositions
using Test
using BenchmarkTools
using Profile, FlameGraphs, ProfileView

#@testset "calc_cmix()" begin

    kmax = 0.05
    rmin = 1500.0
    rmax = 2500.0
    nr = 720
    nside = 128

    println("Calculate amodes...")
    amodes = @time SFB.AnlmModes(kmax, rmin, rmax; nside)
    println("Calculate cmodes...")
    cmodes = @time SFB.ClnnModes(amodes)
    println("Calculate wmodes...")
    wmodes = @time SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)

    println("Make a window...")
    win1 = @time SFB.make_window(wmodes, :radial, :ang_quarter)

    lnnsize = SFB.getlnnsize(cmodes)

    cmix = SFB.power_win_mix(win1, win1, wmodes, cmodes, lnn_min=lnnsize-1)

    @profview SFB.power_win_mix(win1, win1, wmodes, cmodes, lnn_min=lnnsize-10)

#end
