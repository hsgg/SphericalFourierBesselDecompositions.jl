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
        @time anlmmodes = SFB.AnlmModes(kmax, rmin, rmax)
        #@time anlmmodes = SFB.AnlmModes(2, 0, rmin, rmax)
        @time clnnmodes = SFB.ClnnModes(anlmmodes, Δnmax=4)
        @show anlmmodes.nmax anlmmodes.lmax anlmmodes.nside
        @show clnnmodes.Δnmax clnnmodes.Δnmax_l
        @show SFB.getlnnsize(clnnmodes)
        @test maximum(clnnmodes.Δnmax_l) == clnnmodes.Δnmax
        idx2 = 1
        @time for l=0:clnnmodes.amodes.lmax, Δn=0:clnnmodes.Δnmax_l[l+1], n̄=1:clnnmodes.amodes.nmax_l[l+1]-Δn
            idx = SFB.getidx(clnnmodes, l, n̄+Δn, n̄)
            idx3 = SFB.getidx(clnnmodes, l, n̄, n̄+Δn)
            @test idx == idx3
            l2, n2, n2′ = SFB.getlnn(clnnmodes, idx)
            Δn2 = n2′ - n2
            n̄2 = n2
            #@show idx, l,Δn,n̄, l2,Δn2,n̄2
            @test l2 <= clnnmodes.amodes.lmax
            @test Δn2 <= clnnmodes.Δnmax_l[l+1]
            @test n̄2 <= clnnmodes.amodes.nmax_l[l+1] - Δn
            @test l == l2
            @test Δn == Δn2
            @test n̄ == n̄2
            @test idx == idx2
            idx2 += 1
        end

        # test getidx, getlkk
        amodes = SFB.AnlmModes(0.05, 0.0, 1000.0, nside=8)
        cmodes = SFB.ClnnModes(amodes, Δnmax=0)
        nmax = cmodes.amodes.nmax
        @show nmax cmodes.amodes.nmax_l cmodes.Δnmax_l
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
