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


    cmix = SFB.power_win_mix(win1, win1, wmodes, cmodes)


    println("Calculate r √Δr gnl...")
    r, Δr = SFB.window_r(wmodes)
    rsdrgnlr = @time r .* .√Δr .* SFB.Windows.precompute_gnlr(amodes, wmodes)

    println("Calculate W_lm(r)...")
    LMAX = 2 * amodes.lmax
    W1r_lm, L1M1cache = SFB.Windows.optimize_Wr_lm_layout(SFB.Windows.calc_Wr_lm(win1, LMAX, amodes.nside), LMAX)

    println("calc_cmixii_old...")
    lnnsize = SFB.getlnnsize(cmodes)
    @show lnnsize
    i = lnnsize
    i′ = lnnsize
    gg1 = Array{Float64,1}(undef, nr)
    gg2 = Array{Float64,1}(undef, nr)
    @time SFB.Windows.calc_cmixii_old(i, i′, cmodes, rsdrgnlr, W1r_lm, W1r_lm, L1M1cache, false, false, gg1, gg2)
    @time SFB.Windows.calc_cmixii_old(i, i′, cmodes, rsdrgnlr, W1r_lm, W1r_lm, L1M1cache, false, false, gg1, gg2)
    @btime SFB.Windows.calc_cmixii_old($i, $i′, $cmodes, $rsdrgnlr, $W1r_lm, $W1r_lm, $L1M1cache, false, false, $gg1, $gg2)

    #println("calc_cmix()...")
    SFB.Windows.calc_cmix(cmodes, rsdrgnlr, W1r_lm, W1r_lm, L1M1cache, false, false)
    @profview SFB.Windows.calc_cmix(cmodes, rsdrgnlr, W1r_lm, W1r_lm, L1M1cache, false, false)

#end
