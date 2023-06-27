#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
const SFB = SphericalFourierBesselDecompositions

using Test
using LinearAlgebra


@testset "Modes" begin
    @testset "AnlmModes" begin
        rmin = 100.0
        rmax = 10000.0
        kmax = 0.01
        #rmin = 500.0
        #rmax = 3000.0
        #kmax = 0.2

        @time modes = SFB.AnlmModes(kmax, rmin, rmax, cache=false)
        @show modes.kmax modes.nmax modes.lmax modes.nside
        @test 2*modes.nside >= modes.lmax
        @test maximum(modes.lmax_n) == modes.lmax
        @test maximum(modes.nmax_l) == modes.nmax
        @test length(modes.lmax_n) == modes.nmax
        @test length(modes.nmax_l) == modes.lmax + 1
        @test size(modes.knl,1) == modes.nmax  # knl is larger than needed
        @test size(modes.knl,2) == modes.lmax + 1  # knl is larger than needed
        @show SFB.getnlmsize(modes)
        @time for n=1:modes.nmax, l=0:modes.lmax_n[n], m=0:l
            idx = SFB.getidx(modes, n, l, m)
            n2, l2, m2 = SFB.getnlm(modes, idx)
            @test n == n2
            @test l == l2
            @test m == m2
        end
    end

    @testset "ClnnModes" begin
        rmin = 500.0
        rmax = 2000.0
        kmax = 0.02
        Δnmax = 4
        @time anlmmodes = SFB.AnlmModes(kmax, rmin, rmax)
        #@time anlmmodes = SFB.AnlmModes(2, 0, rmin, rmax)
        @time clnnmodes = SFB.ClnnModes(anlmmodes; Δnmax=Δnmax)
        @show anlmmodes.nmax anlmmodes.lmax anlmmodes.nside
        @show SFB.getlnnsize(clnnmodes)
        lmax = anlmmodes.lmax
        idxmax = 0
        @time for l=0:lmax, nA=1:anlmmodes.nmax_l[l+1], nB=nA:anlmmodes.nmax_l[l+1]
            if abs(nB - nA) > Δnmax
                @test !SFB.isvalidlnn(clnnmodes, l, nA, nB)
                continue
            end
            @test SFB.isvalidlnn(clnnmodes, l, nA, nB)
            idx = SFB.getidx(clnnmodes, l, nA, nB)

            idxmax = max(idx, idxmax)

            l2, nA2, nB2 = SFB.getlnn(clnnmodes, idx)
            @test l == l2
            @test nA == nA2
            @test nB == nB2
        end
        lnnsize = SFB.getlnnsize(clnnmodes)
        @test idxmax == lnnsize  # tests that idx goes from 1:lnnsize

        # test getidx, getlkk
        amodes = SFB.AnlmModes(0.05, 0.0, 1000.0, nside=8)
        cmodes = SFB.ClnnModes(amodes, Δnmax=0)
        nmax = cmodes.amodes.nmax
        @show nmax cmodes.amodes.nmax_l cmodes.Δnmax
        k1_old = 0.0
        l = 4
        for n1=1:nmax
            n2 = n1

            v = SFB.isvalidlnn(cmodes, l, n1, n2)
            if n1 >= 15 && l >= 4
                @test !v
                continue
            end

            i = SFB.getidx(cmodes, l, n1, n2)

            l′, n1′, n2′ = SFB.getlnn(cmodes, i)

            i′ = SFB.getidx(cmodes, l′, n1′, n2′)

            L, k1, k2 = SFB.getlkk(cmodes, i)

            @show (l,n1,n2),i,v,(l′,n1′,n2′),i′,(L,k1,k2)

            @test l == l′
            @test n1 == n1′
            @test n2 == n2′

            @test L == l
            @test k1 == k2
            @test k1 > k1_old
            k1_old = k1
        end
    end

    @testset "Bandpower binning with select" begin
        rmin = 0.0
        rmax = 1000.0
        kmax = 0.03
        amodes = SFB.AnlmModes(kmax, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=0)
        lkk = SFB.getlkk(cmodes)
        select = @. lkk[2,:] < 0.02

        lnnsize = SFB.getlnnsize(cmodes)
        lnnselect = sum(select)

        @time w̃0, v0 = SFB.bandpower_binning_weights(cmodes; Δℓ=1, Δn1=1, Δn2=1)
        @test size(w̃0) == (lnnsize,lnnsize)
        @test size(v0) == (lnnsize,lnnsize)

        @time w̃1, v1 = SFB.bandpower_binning_weights(cmodes; Δℓ=1, Δn1=1, Δn2=1, select)
        @test size(w̃1) == (lnnselect,lnnselect)
        @test size(v1) == (lnnselect,lnnselect)

        @time w̃2, v2 = SFB.bandpower_binning_weights(cmodes; Δℓ=2, Δn1=3, Δn2=3)
        @test size(w̃2,1) < size(w̃1,1)
        @test size(w̃2,2) == lnnsize
        @test size(v2,1) == lnnsize
        @test size(v2,2) < size(v1,2)

        @time w̃3, v3 = SFB.bandpower_binning_weights(cmodes; Δℓ=2, Δn1=3, Δn2=3, select)
        @test size(w̃3,1) < size(w̃2,1)
        @test size(w̃3,2) == lnnselect
        @test size(v3,1) == lnnselect
        @test size(v3,2) < size(v2,2)
    end

    @testset "ClnnBinnedModes" begin
        rmin = 1370.0
        rmax = 3540.0
        kmax = 0.04
        amodes = SFB.AnlmModes(kmax, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=0)
        @time w̃, v = SFB.bandpower_binning_weights(cmodes, Δℓ=18)
        @time bcmodes = SFB.ClnnBinnedModes(w̃, v, cmodes)
        @show unique(sort(bcmodes.LKK[1,:]))
        incomplete_bins = SFB.get_incomplete_bins(w̃)
        @show incomplete_bins
        @show bcmodes.LKK[2,incomplete_bins]
        @show maximum(kmax .- bcmodes.LKK[2,incomplete_bins])
        @test all(bcmodes.LKK[2,incomplete_bins] .>= kmax - 0.002884)


        rmin = 500.0
        rmax = 2000.0
        nmax = 10
        lmax = 11
        amodes = SFB.AnlmModes(nmax, lmax, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=0)

        @time w̃, v = SFB.bandpower_binning_weights(cmodes, Δℓ=1)
        @time bcmodes = SFB.ClnnBinnedModes(w̃, v, cmodes)
        @show bcmodes.LKK[1,:]
        @show unique(bcmodes.LKK[1,:])
        @test all(unique(bcmodes.LKK[1,:]) .≈ 0:lmax)

        @time w̃, v = SFB.bandpower_binning_weights(cmodes, Δℓ=2)
        @time bcmodes = SFB.ClnnBinnedModes(w̃, v, cmodes)
        @test all(unique(bcmodes.LKK[1,:]) .≈ 0.5:2:lmax)

        @time w̃, v = SFB.bandpower_binning_weights(cmodes, Δℓ=3)
        @time bcmodes = SFB.ClnnBinnedModes(w̃, v, cmodes)
        @show bcmodes.LKK[1,:]
        @show unique(bcmodes.LKK[1,:])
        @show unique(SFB.getlkk(bcmodes)[1,:])
        @test all(unique(bcmodes.LKK[1,:]) .≈ 1:3:lmax)


        rmin = 500.0
        rmax = 2000.0
        nmax = 10
        lmax = 10
        amodes = SFB.AnlmModes(nmax, lmax, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=0)

        w̃, v = SFB.bandpower_binning_weights(cmodes, Δℓ=1)
        bcmodes = SFB.ClnnBinnedModes(w̃, v, cmodes)
        @test all(unique(bcmodes.LKK[1,:]) .≈ 0:lmax)
        @test w̃ == I
        @test v == I

        @show bcmodes.LKK[1,:]
        @show bcmodes.LKK[2,:]
        @show bcmodes.LKK[3,:]
        @test bcmodes.LKK[2,:] == bcmodes.LKK[3,:]
        @show unique(bcmodes.LKK[1,:])
        @show unique(bcmodes.LKK[2,:])
        @show unique(bcmodes.LKK[3,:])

        i = 91
        @show "====",i
        l, k, k′ = SFB.getlkk(bcmodes, i)
        idx = SFB.getidxapprox(bcmodes, l, k, k′)
        L, K, K′ = SFB.getlkk(bcmodes, idx)
        @show (l,k,k′),i (l,K,K′),idx
        @test i == idx

        i = 12
        @show "====",i
        l, k, k′ = SFB.getlkk(bcmodes, i)
        idx = SFB.getidxapprox(bcmodes, l, k, k′)
        L, K, K′ = SFB.getlkk(bcmodes, idx)
        @show (l,k,k′),i (l,K,K′),idx
        @test i == idx

        for i=1:SFB.getlnnsize(bcmodes)
            l, k, k′ = SFB.getlkk(bcmodes, i)
            idx = SFB.getidxapprox(bcmodes, l, k, k′)
            @test i == idx
        end
    end
end


# vim: set sw=4 et sts=4 :
