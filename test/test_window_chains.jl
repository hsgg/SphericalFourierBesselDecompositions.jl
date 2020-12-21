#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
SFB = SphericalFourierBesselDecompositions

using Test


@testset "Window Chains" begin
    # Note: a lot of the differences here come down to the choice of nside.
    @testset "k = 1" begin
        k = 1
        rmin = 500.0
        rmax = 1000.0
        amodes = SFB.AnlmModes(3, 5, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=1)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)
        win = SFB.make_window(wmodes, :radial, :ang_quarter)

        wlnn = SFB.win_lnn(win, wmodes, cmodes)
        @show wlnn

        cache1 = SFB.WindowChains.WindowChainsCacheWignerChain(win, wmodes, amodes)
        cache2 = SFB.WindowChains.WindowChainsCacheFullWmix(win, wmodes, amodes)
        cache3 = SFB.WindowChains.WindowChainsCacheFullWmixOntheflyWmix(win, wmodes, amodes)
        cache4 = SFB.WindowChains.WindowChainsCacheSeparableWmix(win, wmodes, amodes)
        cache5 = SFB.WindowChains.WindowChainsCacheSeparableWmixOntheflyWlmlm(win, wmodes, amodes)
        cache = SFB.WindowChainsCache(win, wmodes, amodes)

        # single modes
        ell = [1]
        n1 = [1]
        n2 = [1]
        wk0 = (2*ell[1] + 1) * wlnn[SFB.getidx(cmodes, ell[1], n1[1], n2[1])]
        wk1 = SFB.window_chain(ell, n1, n2, cache1)
        wk2 = SFB.window_chain(ell, n1, n2, cache2)
        wk3 = SFB.window_chain(ell, n1, n2, cache3)
        wk4 = SFB.window_chain(ell, n1, n2, cache4)
        wk5 = SFB.window_chain(ell, n1, n2, cache5)
        wk = SFB.window_chain(ell, n1, n2, cache)
        @show wk wk0 wk1 wk2 wk3 wk4 wk5
        @test wk ≈ wk0
        @test wk ≈ wk1
        @test wk ≈ wk2
        @test wk ≈ wk3
        @test wk ≈ wk4
        @test wk ≈ wk5


        # all modes
        wk_lnni = SFB.window_chain(k, win, wmodes, cmodes)
        @show wk_lnni
        @test all(isreal.(wk_lnni))
        wk_lnni = real.(wk_lnni)

        @show size(wk_lnni) size(wlnn)
        for i=1:SFB.getlnnsize(cmodes)
            l, n1, n2 = SFB.getlnn(cmodes, i)
            wk = wk_lnni[i] / (2*l+1)
            @show l,n1,n2,wk_lnni[i],wk,wlnn[i]
            @test wk ≈ wlnn[i] rtol=1e-3
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
        @time M′ = SFB.power_win_mix(win, wmodes, cmodes, interchange_NN′=true)

        cache1 = SFB.WindowChains.WindowChainsCacheWignerChain(win, wmodes, amodes)
        cache2 = SFB.WindowChains.WindowChainsCacheFullWmix(win, wmodes, amodes)
        cache3 = SFB.WindowChains.WindowChainsCacheFullWmixOntheflyWmix(win, wmodes, amodes)
        cache4 = SFB.WindowChains.WindowChainsCacheSeparableWmix(win, wmodes, amodes)
        cache5 = SFB.WindowChains.WindowChainsCacheSeparableWmixOntheflyWlmlm(win, wmodes, amodes)
        cache = SFB.WindowChainsCache(win, wmodes, amodes)

        # single modes
        test_single(ell, n1, n2) = begin
            @show ell,n1,n2
            lnnA = SFB.getidx(cmodes,ell[2],n2[2],n1[2])
            lnnB = SFB.getidx(cmodes,ell[1],n1[1],n2[1])
            wk0 = (2*ell[2] + 1) * M′[lnnA, lnnB]
            wk1 = SFB.window_chain(ell, n1, n2, cache1)
            wk2 = SFB.window_chain(ell, n1, n2, cache2)
            wk3 = SFB.window_chain(ell, n1, n2, cache3)
            wk4 = SFB.window_chain(ell, n1, n2, cache4)
            wk5 = SFB.window_chain(ell, n1, n2, cache5)
            wk = SFB.window_chain(ell, n1, n2, cache)
            @show wk wk0 wk1 wk2 wk3 wk4 wk5
            @test wk ≈ wk0
            @test wk ≈ wk1
            @test wk ≈ wk2
            @test wk ≈ wk3
            @test wk ≈ wk4
            @test wk ≈ wk5
        end
        test_single([1,1], [1,1], [1,1])
        test_single([2,0], [1,1], [2,1])
        test_single([0,0], [1,1], [1,2])
        test_single([0,0], [1,1], [2,2])

        # all modes
        @time wk_lnni = SFB.window_chain(k, win, wmodes, cmodes)
        @time wk_lnni = SFB.window_chain(k, win, wmodes, cmodes)
        @test all(isreal.(wk_lnni))

        @show size(wk_lnni) size(M)
        @test size(wk_lnni) == size(M)
        for i2=1:SFB.getlnnsize(cmodes), i1=1:SFB.getlnnsize(cmodes)
            l1, n1, n1′ = SFB.getlnn(cmodes, i1)
            l2, n2, n2′ = SFB.getlnn(cmodes, i2)
            i2′ = SFB.getidx(cmodes, l2, n2′, n2)
            wk = wk_lnni[i1,i2′] / (2*l2+1)
            @show l1,n1,n1′,l2,n2,n2′,wk,M′[i2,i1]
            @test wk ≈ M′[i2,i1]
        end
    end
end


# vim: set sw=4 et sts=4 :
