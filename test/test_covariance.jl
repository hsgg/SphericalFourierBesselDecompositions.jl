#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
const SFB = SphericalFourierBesselDecompositions

using Test
using DelimitedFiles
using LinearAlgebra

#using Profile

win_features = [(), (:separable,)]

@testset "Covariance Matrix f=$features" for features in win_features
    @testset "Full window" begin
        rmin = 500.0
        rmax = 1000.0
        nbar = 3e-4
        amodes = SFB.AnlmModes(2, 2, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=0)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)
        win = SFB.make_window(wmodes, :radial, :ang_quarter, :rotate, features...)

        # calc shot noise
        w̃mat, vmat = SFB.bandpower_binning_weights(cmodes; Δℓ=1, Δn1=1, Δn2=1)
        bcmodes = SFB.ClnnBinnedModes(w̃mat, vmat, cmodes)
        bcmix = SFB.power_win_mix(win, w̃mat, vmat, wmodes, bcmodes)
        NW_th = SFB.win_lnn(win, wmodes, cmodes) ./ nbar
        N_th = bcmix \ NW_th

        # calc theoretical power
        pkdata, header = readdlm((@__DIR__)*"/data/pk_m.dat", header=true)
        pk = SFB.Spline1D(Float64.(pkdata[:,1]), Float64.(pkdata[:,2]),
                          extrapolation=SFB.Splines.powerlaw)
        C_th = SFB.gen_Clnn_theory(pk, cmodes)
        CN_th = C_th .+ N_th

        # calc covariance, chain
        @time VW_A1 = SFB.Covariance.calc_covariance_exact_A1(CN_th, win, wmodes, cmodes)
        @time VW_chain = SFB.calc_covariance_exact_chain(C_th, nbar, win, wmodes, cmodes)

        #Profile.clear()
        #@time @profile SFB.calc_covariance_exact_chain(CN_th, win, wmodes, cmodes)
        #Profile.print()

        @show size(VW_A1) size(VW_chain)
        @test size(VW_A1) == size(VW_chain)

        @test issymmetric(VW_A1)
        @test issymmetric(VW_chain)
        @test all(isfinite.(VW_A1))
        @test all(isfinite.(VW_chain))

        @show VW_chain
        @show VW_A1
        @show VW_chain ./ VW_A1
        @test_broken VW_chain ≈ VW_A1  # We don't really expect these to agree.
    end
end


# vim: set sw=4 et sts=4 :
