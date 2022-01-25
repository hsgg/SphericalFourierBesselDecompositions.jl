#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
SFB = SphericalFourierBesselDecompositions

using Test
using LinearAlgebra
using Statistics
using Healpix

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


    # Inhomogeneous masks
    @testset "Inhomogeneous Window sep & insep" begin
        rmin = 500.0
        rmax = 1000.0
        amodes = SFB.AnlmModes(2, 5, rmin, rmax)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)

        win1 = SFB.make_window(wmodes, :ang_75, :radial, :rotate, :separable)
        @. win1.mask = rand()
        sidx = (1:length(win1.mask))[win1.mask .> 0]
        sidx = 1:(length(sidx) ÷ 2)
        @. win1.mask[sidx] = 0.5 * win1.mask[sidx]

        win2 = win1[:,:]
        @show sum(win1.mask) ./ length(win1.mask)
        @show extrema(win1.mask)
        @show typeof(win1) typeof(win2)
        @show size(win1) size(win2)

        # wmix
        wmix1 = SFB.calc_wmix(win1, wmodes, amodes)
        wmix2 = SFB.calc_wmix(win2, wmodes, amodes)
        wmix1_negm = SFB.calc_wmix(win1, wmodes, amodes, neg_m=true)
        wmix2_negm = SFB.calc_wmix(win2, wmodes, amodes, neg_m=true)
        @show typeof(wmix1) typeof(wmix2)
        @test wmix1 ≈ wmix2  rtol=1e-10
        @test wmix1_negm ≈ wmix2_negm  rtol=1e-10

        # win_rhat_ln
        win_rhat_ln1 = SFB.win_rhat_ln(win1, wmodes, amodes)[:,:,:]
        win_rhat_ln2 = SFB.win_rhat_ln(win2, wmodes, amodes)
        @test win_rhat_ln1 ≈ win_rhat_ln2  rtol=1e-10
        for k=1:size(win_rhat_ln1,3), j=1:size(win_rhat_ln1,2), i=1:size(win_rhat_ln1,1)
            @test win_rhat_ln1[i,j,k] ≈ win_rhat_ln2[i,j,k] rtol=1e-10
        end

        # M
        cmodes = SFB.ClnnModes(amodes, Δnmax=1)
        M = SFB.power_win_mix(wmix1, wmix1_negm, cmodes)
        M1 = SFB.power_win_mix(win1, wmodes, cmodes)
        M2 = SFB.power_win_mix(win2, wmodes, cmodes)
        @show extrema(M .- M1)
        @show extrema(M .- M2)
        @test M ≈ M1  rtol=1e-10
        @test M ≈ M2  rtol=1e-10

        # CWlnn
        pk(k) = 1e4 * (k/1e-2)^(-3.1)
        Clnn = SFB.gen_Clnn_theory(pk, cmodes)
        CnlmNLM = SFB.Clnn2CnlmNLM(Clnn, cmodes)
        CWlnn1 = SFB.sum_m_lmeqLM(wmix1 * CnlmNLM * wmix1', cmodes)
        CWlnn2 = M * Clnn
        @show extrema((CWlnn1 .- CWlnn2) ./ CWlnn1)
        @test CWlnn1 ≈ CWlnn2 rtol=1e-2

        # Nshot
        nbar = 3e-4
        Nshotobs1 = SFB.win_lnn(win1, wmodes, cmodes) ./ nbar
        Nshotobs2 = SFB.win_lnn(win2, wmodes, cmodes) ./ nbar
        @test Nshotobs1 ≈ Nshotobs2  rtol=1e-10

        # Bandpower binning
        fsky = sum(win1[1,:]) / size(win1,2)
        Δℓ = round(Int, 1 / fsky)
        w̃, v = SFB.bandpower_binning_weights(cmodes; Δℓ=Δℓ, Δn=1)
        bcmodes = SFB.ClnnBinnedModes(w̃, v, cmodes)
        Nmix1 = SFB.power_win_mix(win1, w̃, v, wmodes, bcmodes)
        Nmix2 = SFB.power_win_mix(win2, w̃, v, wmodes, bcmodes)
        @show extrema(Nmix1 .- Nmix2)
        @test Nmix1 ≈ Nmix2  rtol=1e-10
        w̃M1 = SFB.power_win_mix(win1, w̃, I, wmodes, bcmodes)
        Mv1 = SFB.power_win_mix(win1, I, v, wmodes, bcmodes)
        w̃M2 = SFB.power_win_mix(win2, w̃, I, wmodes, bcmodes)
        Mv2 = SFB.power_win_mix(win2, I, v, wmodes, bcmodes)
        @show extrema(w̃M1 .- w̃M2)
        @test w̃M1 ≈ w̃M2  rtol=1e-10
        @test inv(Nmix1) * w̃M1 ≈ inv(Nmix2) * w̃M2  rtol=1e-10
        @test Mv1 * inv(Nmix1) ≈ Mv2 * inv(Nmix2)  rtol=1e-10
    end


    # Test win_rhat_ln() with basisfunctions
    @testset "win(rhat,l,n) with basis" begin
        rmin = 500.0
        rmax = 1000.0
        nmax = 5
        lmax = 6
        nr = 250
        nside = 64
        amodes = SFB.AnlmModes(nmax, lmax, rmin, rmax, nside=nside)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)

        # test win_rhat_ln() with basis functions
        win = fill(1.0, nr, nside2npix(nside))
        p = 1
        for n=1:nmax, l=0:lmax  # assert nmax*(lmax+1) ≤ npix
            r, Δr = SFB.window_r(wmodes)
            win[:,p] .= amodes.basisfunctions.gnl[n,l+1].(r)
            p += 1
        end
        win_rhat_ln = SFB.win_rhat_ln(win, wmodes, amodes)
        for n=1:nmax, l=0:lmax  # assert nmax*(lmax+1) ≤ npix
            p = 1
            for n′=1:nmax, l′=0:lmax
                if l == l′ && n == n′
                    @test win_rhat_ln[p,l+1,n] ≈ 1  atol=1e-4
                elseif l == l′ && n != n′
                    @test win_rhat_ln[p,l+1,n] ≈ 0  atol=1e-4
                #else # l != l′
                end
                p += 1
            end
        end
    end


    # Test win_rhat_ln() with FKP weights
    @testset "weights a la FKP" begin
        rmin = 500.0
        rmax = 1000.0
        kmax =  0.02
        nr = 250
        nside = 64
        amodes = SFB.AnlmModes(kmax, rmin, rmax, nside=nside)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)

        win = SFB.make_window(wmodes, :ang_sixteenth, :radial_cossin_l00_m00)

        nbar_pk = 3e-4 * 1e4
        weights1 = fill(0.25, size(win))
        weights2 = @. 1 / (1 + win * nbar_pk)

        # win_rhat_ln
        win_rhat_ln0 = SFB.win_rhat_ln(win, wmodes, amodes)
        win_rhat_ln1 = SFB.win_rhat_ln(win .* weights1, wmodes, amodes)
        win_rhat_ln2 = SFB.win_rhat_ln(win .* weights2, wmodes, amodes)

        #w0_nlmNLM = SFB.calc_wmix(4*weights1, wmodes, amodes)
        #w1_nlmNLM = SFB.calc_wmix(weights1, wmodes, amodes)
        w2_nlmNLM = SFB.calc_wmix(weights2, wmodes, amodes)

        #@test maximum(abs.(w0_nlmNLM - I)) ≤ 1e-4
        #@test maximum(abs.(w1_nlmNLM - I/4)) ≤ 1e-5
        #@show size(w0_nlmNLM)
        @show SFB.getklm.(amodes, [1,SFB.getnlmsize(amodes)])

        rθϕ = fill(0.0f0, 3, 0)
        nbar = 3e-4
        anlm0 = SFB.cat2amln(rθϕ, amodes, nbar, win_rhat_ln0, [])
        anlm1 = SFB.cat2amln(rθϕ, amodes, nbar, win_rhat_ln1, [])
        anlm2 = SFB.cat2amln(rθϕ, amodes, nbar, win_rhat_ln2, [])

        #@test maximum(abs, w1_nlmNLM * anlm0 - anlm1) ≤ 1e-4 * maximum(abs, anlm1)
        @test maximum(abs, w2_nlmNLM * anlm0 - anlm2) ≤ 1e-4 * maximum(abs, anlm2)
        @show (w2_nlmNLM * anlm0 - anlm2)[1:1]
        @show length(anlm0)


        #n = 2
        #l = 3
        #c0 = win_rhat_ln0[:,l+1,n]
        #c1 = win_rhat_ln1[:,l+1,n]
        #c2 = win_rhat_ln2[:,l+1,n]
        #@show n,l
        #@show mean(c0) mean(c0[c0 .!= 0]) mean(c0[c0 .== 0])
        #@show mean(c1) mean(c1[c1 .!= 0]) mean(c1[c1 .== 0])
        #@show mean(c2) mean(c2[c2 .!= 0]) mean(c2[c2 .== 0])
        #@test mean(c0[c0 .!= 0]) ≈ -3666.014102794932
        #@test mean(c1[c1 .!= 0]) ≈ -3666.014102794932 * 0.25
        #@test mean(c2[c2 .!= 0]) ≈ -481.9511514849361  # unexpected
        #@test mean(c0[c0 .== 0]) == 0
        #@test mean(c1[c1 .== 0]) == 0
        #@test mean(c2[c2 .== 0]) == 0
        #@test mean(c2[c2 .!= 0]) ≈ -1220    rtol=1e-2  # expected

        #@show win[[1,10,nr],[1,5000]]
        #@show weights1[[1,10,nr],[1,5000]]
        #@show weights2[[1,10,nr],[1,5000]]

        ## wmix
        #wmix1 = SFB.calc_wmix(win1, wmodes, amodes)
        #wmix1_negm = SFB.calc_wmix(win1, wmodes, amodes, neg_m=true)
        #@show typeof(wmix1)

        ## M
        #cmodes = SFB.ClnnModes(amodes, Δnmax=1)
        #M = SFB.power_win_mix(wmix1, wmix1_negm, cmodes)
        #M1 = SFB.power_win_mix(win1, wmodes, cmodes)
        #@show extrema(M .- M1)
        #@test M ≈ M1  rtol=1e-10

        ## CWlnn
        #pk(k) = 1e4 * (k/1e-2)^(-3.1)
        #Clnn = SFB.gen_Clnn_theory(pk, cmodes)
        #CnlmNLM = SFB.Clnn2CnlmNLM(Clnn, cmodes)
        #CWlnn1 = SFB.sum_m_lmeqLM(wmix1 * CnlmNLM * wmix1', cmodes)

        ## Nshot
        #nbar = 3e-4
        #Nshotobs1 = SFB.win_lnn(win1, wmodes, cmodes) ./ nbar

        ## Bandpower binning
        #fsky = sum(win1[1,:]) / size(win1,2)
        #Δℓ = round(Int, 1 / fsky)
        #w̃, v = SFB.bandpower_binning_weights(cmodes; Δℓ=Δℓ, Δn=1)
        #bcmodes = SFB.ClnnBinnedModes(w̃, v, cmodes)
        #Nmix1 = SFB.power_win_mix(win1, w̃, v, wmodes, bcmodes)
        #w̃M1 = SFB.power_win_mix(win1, w̃, I, wmodes, bcmodes)
        #Mv1 = SFB.power_win_mix(win1, I, v, wmodes, bcmodes)
    end


    # complex windows
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
        @test M ≈ mmix1  rtol=1e-10

        # CWlnn
        pk(k) = 1e4 * (k/1e-2)^(-3.1)
        Clnn = SFB.gen_Clnn_theory(pk, cmodes)
        CnlmNLM = SFB.Clnn2CnlmNLM(Clnn, cmodes)
        CWlnn1 = SFB.sum_m_lmeqLM(wmix * CnlmNLM * wmix', cmodes)
        CWlnn2 = M * Clnn
        @test CWlnn1 ≈ CWlnn2  rtol=1e-10

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
        @test w̃M ≈ w̃ * M  rtol=1e-10
        @test inv(Nmix) * w̃M ≈ w  rtol=1e-10
        @test Mv * inv(Nmix) ≈ ṽ  rtol=1e-10
    end
end


# vim: set sw=4 et sts=4 :
