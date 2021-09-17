#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
SFB = SphericalFourierBesselDecompositions

using Test
using LinearAlgebra

@testset "Mixing matrices" begin
    @testset "No window" begin
        # Note: inaccuracies in this test are dominated by inaccuracies in healpix.
        rmin = 500.0
        rmax = 1000.0
        amodes = SFB.AnlmModes(3, 5, rmin, rmax)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)
        win = SFB.make_window(wmodes, :separable)

        # wmix
        wmix = SFB.calc_wmix(win, wmodes, amodes)
        diff = wmix - I
        for i′=1:size(diff,2), i=1:size(diff,1)
            @test isapprox(diff[i,i′], 0, atol=1e-2)
        end
        @test isapprox(wmix, I, atol=1e-2)

        # wmix_negm
        wmix_negm = SFB.calc_wmix(win, wmodes, amodes, neg_m=true)
        for i′=1:size(diff,2), i=1:size(diff,1)
            n, l, m = SFB.getnlm(amodes, i)
            n′, l′, m′ = SFB.getnlm(amodes, i′)
            correct = (n==n′ && l==l′ && m==-m′) ? 1 : 0
            @test isapprox(wmix_negm[i,i′], correct, atol=1e-2)
        end

        # M
        cmodes = SFB.ClnnModes(amodes, Δnmax=1)
        mmix1 = SFB.power_win_mix(wmix, wmix_negm, cmodes)
        @test isapprox(mmix1, I, atol=1e-2)
        M = SFB.power_win_mix(win, wmodes, cmodes)
        @test M ≈ mmix1
        @test isapprox(M, I, atol=1e-2)

        # CWlnn
        pk(k) = 1e4 * (k/1e-2)^(-3.1)
        Clnn = SFB.gen_Clnn_theory(pk, cmodes)
        CnlmNLM = SFB.Clnn2CnlmNLM(Clnn, cmodes)
        CWlnn1 = SFB.sum_m_lmeqLM(wmix * CnlmNLM * wmix', cmodes)
        CWlnn2 = M * Clnn
        @test CWlnn1 ≈ CWlnn2

        # Nshot
        nbar = 3e-4
        Nshotobs = SFB.win_lnn(win, wmodes, cmodes) ./ nbar
        Nshotobs2 = SFB.sum_m_lmeqLM(wmix, cmodes) ./ nbar
        @test isapprox(Nshotobs, Nshotobs2, rtol=1e-2)
    end


    # Test more complex windows
    win_descriptions = [
                        (:ang_75,),
                        (:ang_75, :radial),
                        (:ang_half, :radial),
                        (:ang_quarter, :radial),
                        (:separable, :ang_75,),
                        (:separable, :ang_75, :radial),
                        (:separable, :ang_half, :radial),
                        (:separable, :ang_quarter, :radial),
                      ]
    @testset "Window $(win_features...)" for win_features in win_descriptions
        rmin = 500.0
        rmax = 1000.0
        amodes = SFB.AnlmModes(2, 5, rmin, rmax)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)
        win = SFB.make_window(wmodes, win_features...)

        # wmix
        wmix = SFB.calc_wmix(win, wmodes, amodes)
        wmix_negm = SFB.calc_wmix(win, wmodes, amodes, neg_m=true)

        # M
        cmodes = SFB.ClnnModes(amodes, Δnmax=1)
        mmix1 = SFB.power_win_mix(wmix, wmix_negm, cmodes)
        M = SFB.power_win_mix(win, wmodes, cmodes)
        @test M ≈ mmix1

        # CWlnn
        pk(k) = 1e4 * (k/1e-2)^(-3.1)
        Clnn = SFB.gen_Clnn_theory(pk, cmodes)
        CnlmNLM = SFB.Clnn2CnlmNLM(Clnn, cmodes)
        CWlnn1 = SFB.sum_m_lmeqLM(wmix * CnlmNLM * wmix', cmodes)
        CWlnn2 = M * Clnn
        @test CWlnn1 ≈ CWlnn2

        # Nshot
        nbar = 3e-4
        Nshotobs = SFB.win_lnn(win, wmodes, cmodes) ./ nbar
        Nshotobs2 = SFB.sum_m_lmeqLM(wmix, cmodes) ./ nbar
        @test isapprox(Nshotobs, Nshotobs2, rtol=1e-2)

        # Bandpower binning basics
        fsky = sum(win[1,:]) / size(win,2)
        Δℓ = round(Int, 1 / fsky)
        w̃, v = SFB.bandpower_binning_weights(cmodes; Δℓ=Δℓ, Δn=1)
        N = w̃ * M * v
        w = inv(N) * w̃ * M
        ṽ = M * v * inv(N)
        @test w * v ≈ I  # these really just test linear algebra
        @test w̃ * ṽ ≈ I  # these really just test linear algebra

        # Bandpower binning optimizations
        bcmodes = SFB.ClnnBinnedModes(w̃, v, cmodes)
        Nmix = SFB.power_win_mix(win, w̃, v, wmodes, bcmodes)
        @test Nmix ≈ N
        w̃M = SFB.power_win_mix(win, w̃, I, wmodes, bcmodes)
        Mv = SFB.power_win_mix(win, I, v, wmodes, bcmodes)
        @test w̃M ≈ w̃ * M
        @test inv(Nmix) * w̃M ≈ w
        @test Mv * inv(Nmix) ≈ ṽ
    end
end


# vim: set sw=4 et sts=4 :
