#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
SFB = SphericalFourierBesselDecompositions

using Test
using DelimitedFiles

using Profile


@testset "Covariance Matrices" begin
    rmin = 500.0
    rmax = 1000.0
    nbar = 3e-4
    amodes = SFB.AnlmModes(1, 2, rmin, rmax)
    cmodes = SFB.ClnnModes(amodes)
    wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)

    win = SFB.make_window(wmodes, :radial, :ang_quarter)
    @time wmix = SFB.calc_wmix(win, wmodes, amodes)  # too big!
    @time wmix′ = SFB.calc_wmix(win, wmodes, amodes, neg_m=true)  # too big!

    I_LM_ln_ln, LMcache = SFB.WindowChains.calc_I_LM_nl_nl(win, wmodes, amodes)
    wmix1 = SFB.window_wmix(1, 0, 0, 1, 0, 0, I_LM_ln_ln, LMcache)
    @show wmix1 wmix[1]
    @test wmix1 ≈ wmix[1]  rtol=2e-3


    w̃mat, vmat = SFB.bandpower_binning_weights(cmodes; Δℓ=1, ΔΔn=1, Δn̄=1)
    bcmodes = SFB.ClnnBinnedModes(w̃mat, vmat, cmodes)
    bcmix = SFB.power_win_mix(win, w̃mat, vmat, wmodes, bcmodes)

    NW_th = SFB.win_lnn(win, wmodes, cmodes) ./ nbar
    N_th = bcmix \ NW_th

    pkdata, header = readdlm((@__DIR__)*"/data/pk_m.dat", header=true)
    pk = SFB.Spline1D(Float64.(pkdata[:,1]), Float64.(pkdata[:,2]),
                      extrapolation=SFB.Splines.powerlaw)
    C_th = SFB.gen_Clnn_theory(pk, cmodes)
    CN_th = C_th .+ N_th

    @time VW_direct = SFB.calc_covariance_exact_direct(CN_th, wmix, wmix′, cmodes)
    @time VW_chain = SFB.calc_covariance_exact_chain(CN_th, win, wmodes, cmodes)

    Profile.clear()
    @profile SFB.calc_covariance_exact_chain(CN_th, win, wmodes, cmodes)
    Profile.print()

    @show size(VW_direct) size(VW_chain)
    @test size(VW_direct) == size(VW_chain)

    @show VW_chain[1:2,1:2]
    @show VW_direct[1:2,1:2]
    @show VW_chain[1:2,1:2] ./ VW_direct[1:2,1:2]
    @test VW_chain ≈ VW_direct  rtol=5e-2
end


# vim: set sw=4 et sts=4 :
