#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
SFB = SphericalFourierBesselDecompositions

using Test


@testset "Window Chains" begin
    false && @testset "k = 1" begin
        k = 1
        rmin = 500.0
        rmax = 1000.0
        amodes = SFB.AnlmModes(3, 5, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=1)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)
        win = SFB.make_window(wmodes, :radial, :ang_quarter)

        # single modes
        I_LM_ln_ln, LMcache = SFB.window_chains.calc_I_LM_nl_nl(win, wmodes, amodes)
        ell = [1]
        n1 = [1]
        n2 = [1]
        I_LM_l_l = SFB.window_chains.NeqLView(I_LM_ln_ln, ell, n1, n2)
        wk = SFB.window_chain(ell, I_LM_l_l, LMcache)
        @show wk
        @test wk ≈ 0.08538664002556066 * (2*ell[1]+1) rtol=1e-6


        # all modes
        wk_lnni = SFB.window_chain(k, win, wmodes, cmodes)
        @show wk_lnni
        @test all(isreal.(wk_lnni))
        wk_lnni = real.(wk_lnni)

        wlnn = SFB.win_lnn(win, wmodes, cmodes)
        @show wlnn
        @show size(wk_lnni) size(wlnn)
        for i=1:SFB.getlnnsize(cmodes)
            l, n1, n2 = SFB.getlnn(cmodes, i)
            wk = wk_lnni[i] / (2*l+1)
            @show l,n1,n2,wk_lnni[i],wk,wlnn[i]
            @test wk ≈ wlnn[i] rtol=1e-6
        end
    end


    @testset "k = 2" begin
        k = 2
        rmin = 500.0
        rmax = 1000.0
        amodes = SFB.AnlmModes(2, 2, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=1)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)
        win = SFB.make_window(wmodes, :radial, :ang_quarter)

        # comparison
        @time M = SFB.power_win_mix(win, wmodes, cmodes)
        @time M = SFB.power_win_mix(win, wmodes, cmodes)

        # single mode
        I_LM_ln_ln, LMcache = SFB.window_chains.calc_I_LM_nl_nl(win, wmodes, amodes)
        ell = [1, 1]
        n  = [1, 1]
        n′ = [1, 1]
        I_LM_l_l = SFB.window_chains.NeqLView(I_LM_ln_ln, ell, n, n′)
        @show ell,n,n′
        wk = SFB.window_chain(ell, I_LM_l_l, LMcache)
        lnnA = SFB.getidx(cmodes,ell[2],n′[2],n[2])
        lnnB = SFB.getidx(cmodes,ell[1],n[1],n′[1])
        Mab = M[lnnA, lnnB]
        @show wk Mab*(2*ell[2]+1)
        @test wk ≈ Mab * (2*ell[2]+1)

        # another mode
        I_LM_ln_ln, LMcache = SFB.window_chains.calc_I_LM_nl_nl(win, wmodes, amodes)
        ell = [2, 0]
        n  = [1, 1]
        n′ = [2, 1]
        I_LM_l_l = SFB.window_chains.NeqLView(I_LM_ln_ln, ell, n, n′)
        @show ell,n,n′
        wk = SFB.window_chain(ell, I_LM_l_l, LMcache)
        lnnA = SFB.getidx(cmodes,ell[2],n′[2],n[2])
        lnnB = SFB.getidx(cmodes,ell[1],n[1],n′[1])
        Mab = M[lnnA, lnnB]
        @show wk Mab*(2*ell[2]+1)
        @test wk ≈ Mab * (2*ell[2]+1)

        # yet another mode
        I_LM_ln_ln, LMcache = SFB.window_chains.calc_I_LM_nl_nl(win, wmodes, amodes)
        ell = [0, 0]
        n  = [1, 1]
        n′ = [1, 2]
        I_LM_l_l = SFB.window_chains.NeqLView(I_LM_ln_ln, ell, n, n′)
        @show ell,n,n′
        wk = SFB.window_chain(ell, I_LM_l_l, LMcache)
        lnnA = SFB.getidx(cmodes,ell[2],n′[2],n[2])
        lnnB = SFB.getidx(cmodes,ell[1],n[1],n′[1])
        Mab = M[lnnA, lnnB]
        @show wk Mab*(2*ell[2]+1)
        @test wk ≈ Mab * (2*ell[2]+1)

        # guess what? another mode
        I_LM_ln_ln, LMcache = SFB.window_chains.calc_I_LM_nl_nl(win, wmodes, amodes)
        ell = [0, 0]
        n  = [1, 1]
        n′ = [2, 2]
        I_LM_l_l = SFB.window_chains.NeqLView(I_LM_ln_ln, ell, n, n′)
        @show ell,n,n′
        wk = SFB.window_chain(ell, I_LM_l_l, LMcache)
        lnnA = SFB.getidx(cmodes,ell[2],n′[2],n[2])
        lnnB = SFB.getidx(cmodes,ell[1],n[1],n′[1])
        Mab = M[lnnA, lnnB]
        @show wk Mab*(2*ell[2]+1)
        @show wk/(Mab*(2*ell[2]+1))
        @test wk ≈ Mab * (2*ell[2]+1)


        ## all modes
        #@time wk_lnni = SFB.window_chain(k, win, wmodes, cmodes)
        #@time wk_lnni = SFB.window_chain(k, win, wmodes, cmodes)
        #@test all(isreal.(wk_lnni))
        #wk_lnni = real.(wk_lnni)

        #@show size(wk_lnni) size(M)
        #@test size(wk_lnni) == size(M)
        #for i2=1:SFB.getlnnsize(cmodes), i1=1:SFB.getlnnsize(cmodes)
        #    l1, n1, n1′ = SFB.getlnn(cmodes, i1)
        #    l2, n2, n2′ = SFB.getlnn(cmodes, i2)
        #    i2′ = SFB.getidx(cmodes, l2, n2′, n2)
        #    wk = wk_lnni[i1,i2′] / (2*l2+1)
        #    @show l1,n1,n1′,l2,n2,n2′,wk,M[i2,i1]
        #    @test wk ≈ M[i2,i1]
        #end
    end
end


# vim: set sw=4 et sts=4 :
