#!/usr/bin/env julia

using SphericalFourierBesselDecompositions
const SFB = SphericalFourierBesselDecompositions
using Test

@testset "Theory" begin

    @testset "Conversions nlm_NLM <--> lnN" begin
        rmin = 500.0
        rmax = 1000.0
        amodes = SFB.AnlmModes(3, 5, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=1)

        # realistic
        pk(k) = 1e4 * (k/1e-2)^(-3.1)
        Clnn = SFB.gen_Clnn_theory(pk, cmodes)
        CnlmNLM = SFB.Clnn2CnlmNLM(Clnn, cmodes)
        Clnn1 = SFB.sum_m_lmeqLM(CnlmNLM, cmodes)
        CnlmNLM1 = SFB.Clnn2CnlmNLM(Clnn1, cmodes)
        Clnn2 = SFB.sum_m_lmeqLM(CnlmNLM1, cmodes)
        @test Clnn ≈ Clnn1
        @test Clnn ≈ Clnn2

        # random pk
        for i=1:10
            Clnn = rand(SFB.getlnnsize(cmodes))
            CnlmNLM = SFB.Clnn2CnlmNLM(Clnn, cmodes)
            Clnn1 = SFB.sum_m_lmeqLM(CnlmNLM, cmodes)
            CnlmNLM1 = SFB.Clnn2CnlmNLM(Clnn1, cmodes)
            Clnn2 = SFB.sum_m_lmeqLM(CnlmNLM1, cmodes)
            @test Clnn ≈ Clnn1
            @test Clnn ≈ Clnn2
        end

        # random pk
        for i=1:10
            CnlmNLM = rand(Complex{Float64}, SFB.getnlmsize(amodes), SFB.getnlmsize(amodes))
            Clnn1 = SFB.sum_m_lmeqLM(CnlmNLM, cmodes)
            CnlmNLM1 = SFB.Clnn2CnlmNLM(Clnn1, cmodes)
            Clnn2 = SFB.sum_m_lmeqLM(CnlmNLM1, cmodes)
            CnlmNLM2 = SFB.Clnn2CnlmNLM(Clnn2, cmodes)
            # Note: cannot compare with CnlmNLM, because information is lost in first conversion.
            @test Clnn1 ≈ Clnn2
            @test CnlmNLM1 ≈ CnlmNLM2
        end
    end


    @testset "Local average effect" begin
        function testaccess()
            w = rand(ComplexF64, 100, 100)
            w′ = rand(ComplexF64, 100, 100)
            nl = 12
            NL = 21
            m = 4
            M = 3

            @time SFB.Theory.get_anlm_r(w, nl, m)
            @time SFB.Theory.get_anlm_r(w, nl, m)
            @time SFB.Theory.get_anlm_r(w, nl, m)

            @time SFB.get_wmix(w, w′, nl, m, NL, M)
            @time SFB.get_wmix(w, w′, nl, m, NL, M)
            @time SFB.get_wmix(w, w′, nl, m, NL, M)
        end
        testaccess()


        rmin = 500.0
        rmax = 1000.0
        nr = 250
        nbar = 3e-4
        amodes = SFB.AnlmModes(0.02, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=Inf)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)
        amodes_red = SFB.AnlmModes(0.015, rmin, rmax)

        win = SFB.make_window(wmodes, :fullsky)
        weights = 1
        wW = weights * win
        cmix_wW = SFB.power_win_mix(wW, wmodes, cmodes)
        fskyz = ones(wmodes.nr)
        Veff = SFB.integrate_window(win, wmodes)

        wWmix = SFB.calc_wmix(wW, wmodes, amodes_red)
        wWmix_negm = SFB.calc_wmix(wW, wmodes, amodes_red; neg_m=true)
        wWtildemix = SFB.calc_wmix(wW ./ fskyz, wmodes, amodes_red)
        wWtildemix_negm = SFB.calc_wmix(wW ./ fskyz, wmodes, amodes_red; neg_m=true)

        N1 = SFB.win_lnn(wW, wmodes, cmodes) ./ nbar
        NW = SFB.win_lnn(win, wmodes, cmodes) ./ nbar
        Wnlm = SFB.field2anlm(win, wmodes, amodes_red)
        wWnlm = SFB.field2anlm(wW, wmodes, amodes_red)

        println("Calculate N1, N23, N4:")
        @time Nobs, N1, N23, N4 = SFB.calc_NobsA(N1, NW, cmix_wW, nbar, Veff, cmodes)
        @time Nobs, N1, N23, N4 = SFB.calc_NobsA_z(N1, nbar, cmodes, amodes_red, wWmix, wWmix_negm, wWtildemix, wWtildemix_negm)

        println("Calculate T23:")
        @time T23 = SFB.Theory.calc_T23(wWmix, wWmix_negm, wWnlm, Wnlm, amodes_red, cmodes, Veff)
        @time T23 = SFB.Theory.calc_T23_z(cmix_wW, cmodes, amodes_red, wWmix, wWmix_negm, wWtildemix, wWtildemix_negm)

    end

end


# vim: set sw=4 et sts=4 :
