#!/usr/bin/env julia

using SphericalFourierBesselDecompositions
const SFB = SphericalFourierBesselDecompositions


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
    #rmin = 500.0
    #rmax = 1000.0
    #amodes = SFB.AnlmModes(3, 5, rmin, rmax)
    #cmodes = SFB.ClnnModes(amodes, Δnmax=1)

    function testagain()
        w = rand(ComplexF64, 100, 100)
        w′ = rand(ComplexF64, 100, 100)
        nl = 12
        NL = 21
        m = 4
        M = 3

        @time SFB.Theory.get_anlm_r(w, nl, m)
        @time SFB.Theory.get_anlm_r(w, nl, m)
        @time SFB.Theory.get_anlm_r(w, nl, m)

        @time SFB.Theory.get_anlmNLM_r(w, w′, nl, m, NL, M)
        @time SFB.Theory.get_anlmNLM_r(w, w′, nl, m, NL, M)
        @time SFB.Theory.get_anlmNLM_r(w, w′, nl, m, NL, M)
        @time SFB.Theory.get_anlmNLM_r(1, 1, 1, m, 1, M)
        @time SFB.Theory.get_anlmNLM_r(1, 1, 1, m, 1, M)
        @time SFB.Theory.get_anlmNLM_r(1, 1, 1, m, 1, M)
    end
    testagain()
end


# vim: set sw=4 et sts=4 :
