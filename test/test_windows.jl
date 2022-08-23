#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
const SFB = SphericalFourierBesselDecompositions

using Test
using LinearAlgebra
using Statistics
using Healpix



@testset "Mixing matrices" begin

    run_tests = true


    run_tests && @testset "Single-pixel masks" begin
        rmin = 900.0
        rmax = 1000.0
        nside = 64
        amodes = SFB.AnlmModes(3, 5, rmin, rmax, nside=nside)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 100, amodes.nside)
        win = SFB.make_window(wmodes, :separable)
        win.phi .= 0
        win.mask .= 0

        rr, Δr = SFB.window_r(wmodes)
        Ωₚ = 4π / nside2npix(nside)

        for idx_m=1:41:100, idx_r=1:41:100
            win.phi[idx_r] = 1
            win.mask[idx_m] = 1
            @show idx_r,idx_m

            nlmsize = SFB.getnlmsize(amodes)

            w1 = SFB.calc_wmix(win, wmodes, amodes)
            w3 = SFB.calc_wmix(win, wmodes, amodes, neg_m=true)
            wmix1 = fill(NaN*im, nlmsize, nlmsize)
            wmix3 = fill(NaN*im, nlmsize, nlmsize)
            wmix5 = fill(NaN*im, nlmsize, nlmsize)
            wmix7 = fill(NaN*im, nlmsize, nlmsize)

            wmix0 = fill(NaN*im, nlmsize, nlmsize)
            wmix2 = fill(NaN*im, nlmsize, nlmsize)
            wmix4 = fill(NaN*im, nlmsize, nlmsize)
            wmix6 = fill(NaN*im, nlmsize, nlmsize)
            for nlm=1:nlmsize, NLM=1:nlmsize
                n, l, m = SFB.getnlm(amodes, nlm)
                N, L, M = SFB.getnlm(amodes, NLM)
                r = rr[idx_r]
                θ, ϕ = pix2angRing(Resolution(nside), idx_m)
                ylm = SFB.sphericalharmonicsy(l, m, θ, ϕ)
                ylnm = SFB.sphericalharmonicsy(l, -m, θ, ϕ)
                yLM = SFB.sphericalharmonicsy(L, M, θ, ϕ)
                yLnM = SFB.sphericalharmonicsy(L, -M, θ, ϕ)
                gnl = amodes.basisfunctions
                wmix0[nlm,NLM] = Δr * r^2 * gnl(n, l, r) * gnl(N, L, r) * Ωₚ * conj(ylm) * yLM
                wmix2[nlm,NLM] = Δr * r^2 * gnl(n, l, r) * gnl(N, L, r) * Ωₚ * conj(ylnm) * yLM
                wmix4[nlm,NLM] = Δr * r^2 * gnl(n, l, r) * gnl(N, L, r) * Ωₚ * conj(ylm) * yLnM
                wmix6[nlm,NLM] = Δr * r^2 * gnl(n, l, r) * gnl(N, L, r) * Ωₚ * conj(ylnm) * yLnM

                nl = SFB.getidx(amodes, n, l, 0)
                NL = SFB.getidx(amodes, N, L, 0)
                wmix1[nlm,NLM] = SFB.Windows.get_wmix(w1, w3, nl, m, NL, M)
                wmix3[nlm,NLM] = SFB.Windows.get_wmix(w1, w3, nl, -m, NL, M)
                wmix5[nlm,NLM] = SFB.Windows.get_wmix(w1, w3, nl, m, NL, -M)
                wmix7[nlm,NLM] = SFB.Windows.get_wmix(w1, w3, nl, -m, NL, -M)

                #@show (n,l,m),(N,L,M)
                #@show w1[nlm,NLM]
                #@show w3[nlm,NLM]
                #@show wmix1[nlm,NLM]
                #@show wmix3[nlm,NLM]
                #@show wmix5[nlm,NLM]
                #@show wmix7[nlm,NLM]

                #w0 = round(wmix0, sigdigits=4)
                #w1 = round(wmix[nlm,NLM], sigdigits=4)
                #@show (n,l,m),(N,L,M),w0,w1
                #@test wmix[nlm,NLM] ≈ wmix0  rtol=1e-3 atol=1e-10
            end
            @test w1 ≈ wmix0  rtol=1e-3
            @test w3 ≈ wmix2  rtol=1e-3
            @test wmix1 == w1
            @test wmix3 == w3
            @test wmix1 ≈ wmix0  rtol=1e-3
            @test wmix3 ≈ wmix2  rtol=1e-3
            @test wmix5 ≈ wmix4  rtol=1e-3
            @test wmix7 ≈ wmix6  rtol=1e-3
            #@assert false

            win.phi[idx_r] = 0
            win.mask[idx_m] = 0
        end
    end


    run_tests && @testset "No window" begin
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
        @test wmix ≈ I  atol=1e-2

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


    run_tests && @testset "win_lnn()" begin
    #@testset "win_lnn()" begin
        rmin = 500.0
        rmax = 1000.0
        kmax = 0.019  # ≈ 50,000 nlm-modes
        #kmax = 0.025  # ≈ 12.5 seconds in serial
        #kmax = 0.030  # ≈ 30 seconds in parallel, 42 sec serial
        @time amodes = SFB.AnlmModes(kmax, rmin, rmax, cache=false)
        @time wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 2032, amodes.nside)
        @time win = SFB.make_window(wmodes, :radial, :ang_sixteenth, :rotate)

        @time cmodes = SFB.ClnnModes(amodes, Δnmax=100)
        wlnn = SFB.win_lnn(win, wmodes, cmodes)
        wlnn = SFB.win_lnn(win, wmodes, cmodes)

        wmix = SFB.calc_wmix(win, wmodes, amodes)
        wmix = SFB.calc_wmix(win, wmodes, amodes)
        @show wmix[123,121]
        # only useful for detecting changes:
        #@test wmix[123,121] ≈ 0.0031756395970370306 + 0.02813773208852665im  # healpy :rotate E->G
        @test wmix[123,121] ≈ -0.025014220949702827 - 1.01407936505596e-5im
    end


    # Inhomogeneous masks
    run_tests && @testset "Inhomogeneous Window sep & insep" begin
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
    run_tests && @testset "win(rhat,l,n) with basis" begin
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
    run_tests && @testset "Window $(win_features...)" for win_features in win_descriptions
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
