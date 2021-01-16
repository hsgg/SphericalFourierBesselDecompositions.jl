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
        win = SFB.make_window(wmodes, :radial, :ang_quarter, :rotate)

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
        @test_skip wk ≈ wk1
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
            @test wk ≈ wlnn[i]
        end
    end


    @testset "k = 2" begin
        k = 2
        rmin = 500.0
        rmax = 1000.0
        amodes = SFB.AnlmModes(2, 2, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=1)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)
        win = SFB.make_window(wmodes, :radial, :ang_quarter, :rotate)

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
            @test_skip wk ≈ wk1
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


    @testset "Wk method agreement" begin
        rmin = 500.0
        rmax = 1000.0
        amodes = SFB.AnlmModes(8, 4, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=0)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)
        win = SFB.make_window(wmodes, :radial, :ang_quarter, :rotate)

        cache1 = SFB.WindowChains.WindowChainsCacheWignerChain(win, wmodes, amodes)
        cache2 = SFB.WindowChains.WindowChainsCacheFullWmix(win, wmodes, amodes)
        cache3 = SFB.WindowChains.WindowChainsCacheFullWmixOntheflyWmix(win, wmodes, amodes)
        cache4 = SFB.WindowChains.WindowChainsCacheSeparableWmix(win, wmodes, amodes)
        cache5 = SFB.WindowChains.WindowChainsCacheSeparableWmixOntheflyWlmlm(win, wmodes, amodes)
        cache = SFB.WindowChainsCache(win, wmodes, amodes)

        test_individual(ell, n1, n2) = begin
            @time wk1 = SFB.window_chain(ell, n1, n2, cache1)
            @time wk2 = SFB.window_chain(ell, n1, n2, cache2)
            @time wk3 = SFB.window_chain(ell, n1, n2, cache3)
            @time wk4 = SFB.window_chain(ell, n1, n2, cache4)
            @time wk5 = SFB.window_chain(ell, n1, n2, cache5)
            @time wk = SFB.window_chain(ell, n1, n2, cache)
            @show ell n1 n2
            @show wk1 wk2 wk3 wk4 wk5 wk
            @test_skip wk1 ≈ wk
            @test wk2 ≈ wk
            @test wk3 ≈ wk
            @test wk4 ≈ wk
            @test wk5 ≈ wk
            return wk
        end

        println("Testing individual modes...")
        for l1=0:amodes.lmax
            test_individual([l1], [1], [1])
            for l2=0:amodes.lmax
                test_individual([l1, l2], [1, 1], [1, 1])
                for l3=0:amodes.lmax
                    test_individual([l1, l2, l3], [1, 1, 1], [1, 1, 1])
                end
            end
        end
        #test_individual([0, 0, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1])
        #test_individual([1, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1])
        #test_individual([2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1])
        #test_individual([3, 3, 3, 3], [1, 1, 2, 2], [2, 2, 1, 1])

        # check n ↔ n′ symmetry doesn't exist
        ell = [2, 1, 0]
        n1 = [4, 2, 2]
        n2 = [2, 3, 3]
        test_individual(ell, n1, n2)
        test_individual(ell, n2, n1)

        # check antisymmetry doesn't exist
        ell = [2, 1, 0, 1]
        n1 = [4, 2, 2, 2]
        n2 = [2, 3, 3, 2]
        test_individual(ell, n1, n2)
        test_individual(ell, n2, n1)
        ell[1:2] .= ell[2:-1:1]
        n1[1:2] .= n1[2:-1:1]
        n2[1:2] .= n2[2:-1:1]
        test_individual(ell, n1, n2)

        # check anticyclic symmetry
        #ell = [2, 1, 0, 1]
        #n1 = [2, 3, 3, 2]
        #n2 = [4, 2, 2, 2]
        for i=1:10
            k = 4
            p1 = k:-1:1
            ell = rand(0:amodes.lmax, k)
            n1 = rand(1:amodes.nmax, k)
            n2 = rand(1:amodes.nmax, k)
            wk = test_individual(ell, n1, n2)
            wkp1 = test_individual(ell[p1], n2[p1], n1[p1])
            @show wk wkp1
            @test wk ≈ wkp1
        end

        return

        # These checks can fail:
        # check k=3 n,N ↔ n′,N′ symmetry, Nah, not a symmetry
        for i=1:10
            k = 3
            ell = rand(0:amodes.lmax, k)
            n1 = rand(1:amodes.nmax, k)
            n2 = rand(1:amodes.nmax, k)
            wk = test_individual(ell, n1, n2)
            n1[2:3] .= n1[3:-1:2]
            n2[2:3] .= n2[3:-1:2]
            wkp1 = test_individual(ell, n2, n1)
            @show wk wkp1
            @test wk ≈ wkp1
        end
    end


    @testset "Symmetricizing" begin
        rmin = 500.0
        rmax = 1000.0
        amodes = SFB.AnlmModes(8, 4, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=0)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)
        win = SFB.make_window(wmodes, :radial, :ang_quarter, :rotate)
        cache = SFB.WindowChainsCache(win, wmodes, amodes)

        ell = [1, 2, 3, 4]
        n1 = [1, 2, 3, 4]
        n2 = [5, 6, 7, 8]
        symmetries = [1=>0, 2=>1, 3=>2]
        wkfull = SFB.window_chain(ell, n1, n2, cache, symmetries)
        @show wkfull
        @test wkfull ≈ -1.2707026010569427e-8  # only useful for detecting changes
    end


    @testset "Wk single pixel mask" begin
        rmin = 500.0
        rmax = 1000.0
        nside = 16
        amodes = SFB.AnlmModes(8, 4, rmin, rmax; nside=nside)
        cmodes = SFB.ClnnModes(amodes, Δnmax=0)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)
        mask = fill(0.0, SFB.hp.nside2npix(amodes.nside))
        mask[805+1] = 1.0
        phi = fill(1.0, wmodes.nr)
        win = SFB.SeparableArray(phi, mask, name1=:phi, name2=:mask)

        cache1 = SFB.WindowChains.WindowChainsCacheWignerChain(win, wmodes, amodes)
        cache2 = SFB.WindowChains.WindowChainsCacheFullWmix(win, wmodes, amodes)
        cache3 = SFB.WindowChains.WindowChainsCacheFullWmixOntheflyWmix(win, wmodes, amodes)
        cache4 = SFB.WindowChains.WindowChainsCacheSeparableWmix(win, wmodes, amodes)
        cache5 = SFB.WindowChains.WindowChainsCacheSeparableWmixOntheflyWlmlm(win, wmodes, amodes)
        cache = SFB.WindowChainsCache(win, wmodes, amodes)

        test_individual(ell, n1, n2) = begin
            @time wk1 = SFB.window_chain(ell, n1, n2, cache1)
            @time wk2 = SFB.window_chain(ell, n1, n2, cache2)
            @time wk3 = SFB.window_chain(ell, n1, n2, cache3)
            @time wk4 = SFB.window_chain(ell, n1, n2, cache4)
            @time wk5 = SFB.window_chain(ell, n1, n2, cache5)
            @time wk = SFB.window_chain(ell, n1, n2, cache)
            @show ell n1 n2
            @show wk1 wk2 wk3 wk4 wk5 wk
            @test_skip wk1 ≈ wk
            @test wk2 ≈ wk
            @test wk3 ≈ wk
            @test wk4 ≈ wk
            @test wk5 ≈ wk
            return wk
        end

        println("Testing individual modes...")
        for l1=0:amodes.lmax
            test_individual([l1], [1], [1])
            for l2=0:amodes.lmax
                test_individual([l1, l2], [1, 1], [1, 1])
                for l3=0:amodes.lmax
                    test_individual([l1, l2, l3], [1, 1, 1], [1, 1, 1])
                end
            end
        end
    end


    @testset "Other tests" begin
        rmin = 0.0
        rmax = 2000.0
        amodes = SFB.AnlmModes(0.03, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=0)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)
        @show amodes.lmax_n

        phi = fill(1.0, wmodes.nr)
        mask = fill(1.0, SFB.hp.nside2npix(amodes.nside))
        win = SFB.SeparableArray(phi, mask, name1=:phi, name2=:mask)

        wlm, LMcache = SFB.WindowChains.calc_Wlm(mask, amodes.lmax, amodes.nside)
        lrange = 48:48
        for l1=lrange, l2=lrange, m1=-l1:l1, m2=-l2:l2
            w = SFB.WindowChains.window_wmix(l1, m1, l2, m2, wlm, LMcache)
            if l1 == l2 && m1 == m2
                @test w ≈ 1
                if !(w ≈ 1)
                    @error "Wlmlm unexpected result" l1,l2 m1,m2 w
                end
            else
                @test w ≈ 0  atol=1e-10
                if !(abs(w) <= 1e-10)
                    @error "Wlmlm unexpected result" l1,l2 m1,m2 w
                end
            end
        end

        #return

        # Note: the following tests can only be done reasonably fast with
        # WignerFamilies instead of WignerSymbols in the window chain.

        cache = SFB.WindowChainsCache(win, wmodes, amodes)
        @show typeof(cache)
        for l=0:amodes.lmax
            nmax = amodes.nmax_l[l+1]
            @test cache.Ilnln[l+1,1:nmax,l+1,1:nmax] ≈ I  atol=1e-4
            Ik = SFB.WindowChains.calc_kprod(cache.Ilnln, [l,l] .+ 1, [1,1], [1,1])
            @test Ik ≈ 1  atol=1e-4
        end

        ell = [48, 48]
        n1 = [1, 1]
        n2 = [1, 1]
        wk = SFB.window_chain(ell, n1, n2, cache)
        @show wk wk/(2*ell[1]+1)
        @test wk/(2*ell[1]+1) ≈ 1

        for m1=-ell[1]:ell[1], m2=-ell[2]:ell[2]
            w = SFB.WindowChains.get_wlmlm(cache, ell[end], m2, ell[1], m1)
            w2 = SFB.WindowChains.window_wmix(ell[end], m2, ell[1], m1, wlm, LMcache)
            if m1 != m2 || ell[1] != ell[2]
                @test w ≈ 0  atol=1e-10
                @test w2 ≈ 0  atol=1e-10
            else
                @test w ≈ 1
                @test w2 ≈ 1
                if !(w ≈ 1)
                    @error "Wlmlm unexpected result" ell m1,m2 w w2
                end
            end
        end


        ell = [48, 0]
        n1 = [1, 1]
        n2 = [2, 2]
        wk = SFB.window_chain(ell, n1, n2, cache)
        @show wk
    end
end


# vim: set sw=4 et sts=4 :
