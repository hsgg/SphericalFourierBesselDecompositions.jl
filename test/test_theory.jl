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

        @time SFB.Theory.get_anlmNLM_r(w, w′, nl, m, NL, M)
        @time SFB.Theory.get_anlmNLM_r(w, w′, nl, m, NL, M)
        @time SFB.Theory.get_anlmNLM_r(w, w′, nl, m, NL, M)
    end
    testaccess()


    rmin = 500.0
    rmax = 1000.0
    nr = 250
    amodes = SFB.AnlmModes(0.02, rmin, rmax)
    cmodes = SFB.ClnnModes(amodes, Δnmax=Inf)
    wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)

    win = SFB.make_window(wmodes, :fullsky)
    cmix = SFB.power_win_mix(win, wmodes, cmodes)
    wmix = SFB.calc_wmix(win, wmodes, amodes)
    wmix_negm = SFB.calc_wmix(win, wmodes, amodes; neg_m=true)
    fskyinvlnn = SFB.win_lnn(win, wmodes, cmodes)

    println("Calculate T23:")
    @time T23 = SFB.Theory.calc_T23_z(cmix, cmodes, amodes, wmix, wmix_negm, wmix, wmix_negm, fskyinvlnn)
    @time T23 = SFB.Theory.calc_T23_z(cmix, cmodes, amodes, wmix, wmix_negm, wmix, wmix_negm, fskyinvlnn)
    @time T23 = SFB.Theory.calc_T23_z(cmix, cmodes, amodes, wmix, wmix_negm, wmix, wmix_negm, fskyinvlnn)

    lnnsize = SFB.getlnnsize(cmodes)
    for i=1:lnnsize, j=1:lnnsize
        lμ, nμ, nν = SFB.getlnn(cmodes, i)
        lσ, nσ, nα = SFB.getlnn(cmodes, j)

        # calculate correct T23 for full-sky, constant unity weighting,
        # constant unity radial selection
        T23_correct = 0
        isnonzero = (lμ == lσ == 0 && ((nα == nν && nσ == nμ) || (nα == nμ && nσ == nν)))
        is2 = (lμ == lσ == 0 && ((nα == nν && nσ == nμ) && (nα == nμ && nσ == nν)))
        if isnonzero
            if !is2
                T23_correct = 1
            else
                T23_correct = 2
            end
        end

        # compare
        if abs(T23[i,j] - T23_correct) > 1e-4
            @error "T23 incorrect" i,j lμ,nμ,nν lσ,nσ,nα T23[i,j] T23_correct
        end
        @test abs(T23[i,j] - T23_correct) < 1e-4
    end
end


# vim: set sw=4 et sts=4 :
