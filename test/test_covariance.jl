#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
SFB = SphericalFourierBesselDecompositions

using Test
using DelimitedFiles
using LinearAlgebra

using Profile


@testset "Covariance Matrices" begin
    @testset "Full window" begin
        rmin = 500.0
        rmax = 1000.0
        nbar = 3e-4
        amodes = SFB.AnlmModes(5, 5, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=1)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)
        win = SFB.make_window(wmodes, :radial, :ang_quarter)

        # calc shot noise
        w̃mat, vmat = SFB.bandpower_binning_weights(cmodes; Δℓ=1, Δn=1)
        bcmodes = SFB.ClnnBinnedModes(w̃mat, vmat, cmodes)
        bcmix = SFB.power_win_mix(win, w̃mat, vmat, wmodes, bcmodes)
        NW_th = SFB.win_lnn(win, wmodes, cmodes) ./ nbar
        N_th = bcmix \ NW_th
        N_th = 0

        # calc theoretical power
        pkdata, header = readdlm((@__DIR__)*"/data/pk_m.dat", header=true)
        pk = SFB.Spline1D(Float64.(pkdata[:,1]), Float64.(pkdata[:,2]),
                          extrapolation=SFB.Splines.powerlaw)
        C_th = SFB.gen_Clnn_theory(pk, cmodes)
        CN_th = C_th .+ N_th

        ## calc covariance, direct
        #@time wmix = SFB.calc_wmix(win, wmodes, amodes)  # too big!
        #@time wmix′ = SFB.calc_wmix(win, wmodes, amodes, neg_m=true)  # too big!
        #@time VW_direct = SFB.calc_covariance_exact_direct(CN_th, wmix, wmix′, cmodes)

        # calc covariance, chain
        @time VW_chain = SFB.calc_covariance_exact_chain(CN_th, win, wmodes, cmodes)
        Profile.clear()
        @time @profile SFB.calc_covariance_exact_chain(CN_th, win, wmodes, cmodes)
        Profile.print()

        #@show size(VW_direct) size(VW_chain)
        #@test size(VW_direct) == size(VW_chain)

        #@test issymmetric(VW_direct)
        @test issymmetric(VW_chain)
        #@test all(isfinite.(VW_direct))
        @test all(isfinite.(VW_chain))

        #@show VW_chain
        #@show VW_direct
        #@show VW_chain ./ VW_direct
        #@test VW_chain ≈ VW_direct
    end
end


# vim: set sw=4 et sts=4 :
