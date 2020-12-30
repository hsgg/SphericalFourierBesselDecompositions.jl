#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
SFB = SphericalFourierBesselDecompositions

using Test


@testset "Window Chains" begin
    # Note: a lot of the differences here come down to the choice of nside and
    # details of how healpy is used.


    @testset "Wmix" begin
        rmin = 500.0
        rmax = 1000.0
        amodes = SFB.AnlmModes(3, 5, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=1)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)
        win = SFB.make_window(wmodes, :radial, :ang_quarter, :rotate)

        cache2 = SFB.WindowChains.WindowChainsCacheFullWmix(win, wmodes, amodes)
        cache3 = SFB.WindowChains.WindowChainsCacheFullWmixOntheflyWmix(win, wmodes, amodes)
        cache4 = SFB.WindowChains.WindowChainsCacheSeparableWmix(win, wmodes, amodes)
        cache5 = SFB.WindowChains.WindowChainsCacheSeparableWmixOntheflyWlmlm(win, wmodes, amodes)

        ## test cache2 consistency
        #@show real.(cache2.wmix') real.(cache2.wmix)
        #@show imag.(cache2.wmix') imag.(cache2.wmix)
        @test cache2.wmix' ≈ cache2.wmix
        #@show real.(cache2.wmix_negm') real.(cache2.wmix_negm)
        #@show imag.(cache2.wmix_negm') imag.(cache2.wmix_negm)
        nlmsize = SFB.getnlmsize(amodes)
        for i=1:nlmsize, i′=1:nlmsize
            n1, l1, m1 = SFB.getnlm(amodes, i)
            n2, l2, m2 = SFB.getnlm(amodes, i′)
            @test cache2.wmix_negm[i′,i] ≈ (-1)^(m1+m2) * cache2.wmix_negm[i,i′]
        end

        get_win(nlm1, nlm2; verbose=false) = begin
            n1, l1, m1 = nlm1
            n2, l2, m2 = nlm2

            nl1 = SFB.getidx(cache2.amodes, n1, l1, 0)
            nl2 = SFB.getidx(cache2.amodes, n2, l2, 0)
            w2 = SFB.WindowChains.get_wmix(cache2.wmix, cache2.wmix_negm,
                                           nl1, m1, nl2, m2)
            verbose && @show cache2.wmix[nl1+abs(m1),nl2+abs(m2)]
            verbose && @show cache2.wmix'[nl1+abs(m1),nl2+abs(m2)]
            verbose && @show conj(cache2.wmix[nl2+abs(m2),nl1+abs(m1)])
            verbose && @show cache2.wmix_negm[nl1+abs(m1),nl2+abs(m2)]
            verbose && @show cache2.wmix_negm'[nl1+abs(m1),nl2+abs(m2)]
            verbose && @show conj(cache2.wmix_negm[nl2+abs(m2),nl1+abs(m1)])

            wlmlm4 = SFB.WindowChains.get_wlmlm(cache4, l1, m1, l2, m2)
            inlnl4 = cache4.Ilnln[l1+1,n1,l2+1,n2]
            w4 = inlnl4 * wlmlm4

            wlmlm5 = SFB.WindowChains.window_wmix(l1, m1, l2, m2,
                                                  cache5.Wlm, cache5.LMcache)
            inlnl5 = cache5.Ilnln[l1+1,n1,l2+1,n2]
            w5 = inlnl5 * wlmlm5
            return w2, w4, w5
        end

        test_mode(nlm1, nlm2; verbose=true) = begin
            w2a, w4a, w5a = get_win(nlm1, nlm2; verbose=verbose)
            verbose && @show nlm1,nlm2
            verbose && @show w2a w4a w5a
            @test w2a ≈ w4a ≈ w5a

            w2b, w4b, w5b = get_win(nlm2, nlm1; verbose=verbose)
            verbose && @show nlm2,nlm1
            verbose && @show w2b w4b w5b
            @test w2b ≈ w4b ≈ w5b

            nlm1[3] = -nlm1[3]
            nlm2[3] = -nlm2[3]
            w2c, w4c, w5c = get_win(nlm1, nlm2; verbose=verbose)
            verbose && @show nlm1,nlm2,nlm1[3]+nlm2[3]
            verbose && @show w2c w4c w5c
            @test w2c ≈ w4c ≈ w5c

            @test w2b ≈ conj(w2a)
            @test w2c ≈ (-1)^(nlm1[3] + nlm2[3]) * conj(w2a)
            @test w4b ≈ conj(w4a)
            @test w4c ≈ (-1)^(nlm1[3] + nlm2[3]) * conj(w4a)
            @test w5b ≈ conj(w5a)
            @test w5c ≈ (-1)^(nlm1[3] + nlm2[3]) * conj(w5a)
        end

        test_mode([1, 1, -1], [1, 0, 0])
        test_mode([1, 1, 1], [1, 1, 1])
        test_mode([1, 1, -1], [1, 1, 1])

        nlmsize = SFB.getnlmsize(amodes)
        for j=1:nlmsize, i=1:nlmsize
            n1, l1, m1 = SFB.getnlm(amodes, i)
            n2, l2, m2 = SFB.getnlm(amodes, j)
            test_mode([n1, l1, m1], [n2, l2, m2]; verbose=false)
            test_mode([n1, l1, -m1], [n2, l2, m2]; verbose=false)
            test_mode([n1, l1, m1], [n2, l2, -m2]; verbose=false)
            test_mode([n1, l1, -m1], [n2, l2, -m2]; verbose=false)
        end
    end


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
        test_single([0,1], [1,1], [1,1])

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


    @testset "k = 4" begin
        rmin = 500.0
        rmax = 1000.0
        amodes = SFB.AnlmModes(1, 5, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=0)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)
        win = SFB.make_window(wmodes, :radial, :ang_quarter, :rotate)

        cache1 = SFB.WindowChains.WindowChainsCacheWignerChain(win, wmodes, amodes)
        cache2 = SFB.WindowChains.WindowChainsCacheFullWmix(win, wmodes, amodes)
        cache3 = SFB.WindowChains.WindowChainsCacheFullWmixOntheflyWmix(win, wmodes, amodes)
        cache4 = SFB.WindowChains.WindowChainsCacheSeparableWmix(win, wmodes, amodes)
        cache5 = SFB.WindowChains.WindowChainsCacheSeparableWmixOntheflyWlmlm(win, wmodes, amodes)
        cache = SFB.WindowChainsCache(win, wmodes, amodes)

        ell = [2, 3, 4, 5]
        #ell = [0, 0, 0, 1]
        n1 = [1, 1, 1, 1]
        n2 = [1, 1, 1, 1]
        #ell = [0, 1]
        #n1 = [1, 1]
        #n2 = [1, 1]

        println("Compiling...")
        @time wk1 = SFB.window_chain(ell, n1, n2, cache1)
        @time wk2 = SFB.window_chain(ell, n1, n2, cache2)
        @time wk3 = SFB.window_chain(ell, n1, n2, cache3)
        @time wk4 = SFB.window_chain(ell, n1, n2, cache4)
        @time wk5 = SFB.window_chain(ell, n1, n2, cache5)
        @time wk = SFB.window_chain(ell, n1, n2, cache)
        @show wk1
        @show wk2
        @show wk3
        @show wk4
        @show wk5
        @show wk

        @test_broken wk1 ≈ wk
        @test wk2 ≈ wk
        @test wk3 ≈ wk
        @test wk4 ≈ wk
        @test wk5 ≈ wk

        #cc = SFB.NDIterator([0,0,0,0], [1,1,1,1])
        #while SFB.advance(cc)
        #    wk = SFB.window_chain(ell, n1, n2, cache2, cc)
        #    @show cc,wk
        #end
        #@test isfinite(wk)
        return

        # benchmark
        lnnsize = SFB.getlnnsize(cmodes)
        @show lnnsize lnnsize^2
        #@time for i=1:lnnsize; SFB.window_chain(ell, n1, n2, cache1); end
        @time for i=1:lnnsize; SFB.window_chain(ell, n1, n2, cache2); end
        @time for i=1:lnnsize; SFB.window_chain(ell, n1, n2, cache3); end
        @time for i=1:lnnsize; SFB.window_chain(ell, n1, n2, cache4); end
        @time for i=1:lnnsize; SFB.window_chain(ell, n1, n2, cache5); end
        @time for i=1:lnnsize; SFB.window_chain(ell, n1, n2, cache); end
    end
end


# vim: set sw=4 et sts=4 :
