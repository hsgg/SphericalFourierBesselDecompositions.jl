#!/usr/bin/env julia


using Revise
using SphericalFourierBesselDecompositions
const SFB = SphericalFourierBesselDecompositions

using Test
using LinearAlgebra
using Statistics
using Healpix



@testset "Mixing matrices" begin

    run_tests = true
    #run_tests = false


    run_tests && @testset "calc_Wr_lm()" begin
        nside = 64
        nr = 240
        lmax = 256
        rmin = 500.0
        rmax = 1000.0
        @show 4 * nside - 1
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, nside)
        win = SFB.make_window(wmodes, :ang_quarter, :radial, :separable)

        Wr_lm = SFB.Windows.calc_Wr_lm(win, lmax, nside)

        Wr_00 = √(4 * π) * mean(win, dims=2)[:]

        @show extrema(Vector{Float64}(Wr_lm[:,1]))
        @show extrema(Wr_00)

        @test Wr_00 ≈ Wr_lm[:,1]  atol=1e-3
    end


    run_tests && @testset "Single-pixel masks" begin
        rmin = 900.0
        rmax = 1000.0
        nside = 64
        nr = 100
        npix = nside2npix(nside)
        amodes = SFB.AnlmModes(3, 5, rmin, rmax, nside=nside)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)
        win = SFB.make_window(wmodes, :separable)
        win.phi .= 0
        win.mask .= 0

        rr, Δr = SFB.window_r(wmodes)
        Ωₚ = 4π / nside2npix(nside)

        # select a few points
        idxs_r = rand(1:nr, 2)
        idxs_m = rand(1:npix, 2)

        for idx_m in idxs_m, idx_r in idxs_r
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


    run_tests && @testset "Wmix orthogonality" begin
    #@testset "Wmix orthogonality" begin
        rmin = 500.0
        rmax = 1000.0
        amodes = SFB.AnlmModes(3, 5, rmin, rmax)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, 1000, amodes.nside)
        win = ones(wmodes.nr, wmodes.npix)

        wmix = SFB.calc_wmix(win, wmodes, amodes)
        wmix_negm = SFB.calc_wmix(win, wmodes, amodes; neg_m=true)

        nlmsize = SFB.getnlmsize(amodes)
        for i=1:nlmsize, j=1:nlmsize
            n, l, m = SFB.getnlm(amodes, i)
            N, L, M = SFB.getnlm(amodes, j)
            nl = SFB.getidx(amodes, n, l, 0)
            NL = SFB.getidx(amodes, N, L, 0)
            w = SFB.get_wmix(wmix, wmix_negm, nl, m, NL, M)

            if i != j
                @test (n != N) || (l != L) || (m != M)
                @test w ≈ 0  atol=1e-5

                w = SFB.get_wmix(wmix, wmix_negm, nl, m, NL, -M)
                @test w ≈ 0  atol=1e-5

                w = SFB.get_wmix(wmix, wmix_negm, nl, -m, NL, M)
                @test w ≈ 0  atol=1e-5

                w = SFB.get_wmix(wmix, wmix_negm, nl, -m, NL, -M)
                @test w ≈ 0  atol=1e-5
            else
                @test (n == N) && (l == L) && (m == M)
                @test w ≈ 1  atol=1e-5

                if m != 0
                    w = SFB.get_wmix(wmix, wmix_negm, nl, m, NL, -M)
                    @test w ≈ 0  atol=1e-5

                    w = SFB.get_wmix(wmix, wmix_negm, nl, -m, NL, M)
                    ret = @test w ≈ 0  atol=1e-5
                    if ret isa Test.Fail
                        @error ret i,j n,l,m N,L,M nl,-m NL,M w
                    end

                    w = SFB.get_wmix(wmix, wmix_negm, nl, -m, NL, -M)
                    @test w ≈ 1  atol=1e-5
                end
            end
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
        nr = 250
        @time amodes = SFB.AnlmModes(kmax, rmin, rmax, cache=false)
        @time wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)
        @time cmodes = SFB.ClnnModes(amodes, Δnmax=Inf)
        @time win = SFB.make_window(wmodes, :radial, :ang_sixteenth, :separable, :rotate, :dense)

        wlnn = SFB.win_lnn(win, wmodes, cmodes)
        wmix = SFB.calc_wmix(win, wmodes, amodes)

        @show wmix[123,121]
        # only useful for detecting changes:
        #@test wmix[123,121] ≈ 0.0031756395970370306 + 0.02813773208852665im  # healpy :rotate E->G
        #@test wmix[123,121] ≈ -0.025014220949702827 - 1.01407936505596e-5im  # nr=2032
        @test wmix[123,121] ≈ -0.025087015337107783 - 1.0170304578086492e-5im  # nr=250
        wlnn2 = SFB.sum_m_lmeqLM(wmix, cmodes)

        @test wlnn ≈ wlnn2  rtol=1e-3
    end


    run_tests && @testset "Wlnn, Wmix, Cmix: l=L=0" begin
        rmin = 0.0
        rmax = 1000.0
        nr = 2032

        kmax = 0.01
        #kmax = 0.019
        #kmax = 0.025  # ≈ 12.5 seconds in serial
        #kmax = 0.030  # ≈ 30 seconds in parallel, 42 sec serial
        @time amodes = SFB.AnlmModes(kmax, rmin, rmax, cache=false)
        #@time amodes = SFB.AnlmModes(2, 0, rmin, rmax, cache=false)
        @show SFB.getnlmsize(amodes)

        @time wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)
        @time cmodes = SFB.ClnnModes(amodes, Δnmax=Inf)
        @time win = SFB.make_window(wmodes, :radial_expmrr0, :fullsky)
        nmax = amodes.nmax

        wlnn1 = SFB.win_lnn(win, wmodes, cmodes)
        wmix = SFB.calc_wmix(win, wmodes, amodes)
        cmix = SFB.power_win_mix(win, wmodes, cmodes)

        w0nn0 = SFB.W0nn_expmrr0(wmodes, cmodes)
        w0nn1 = SFB.get_0nn(wlnn1, cmodes)
        w0nn2 = fill(0.0, size(w0nn1))
        for i=1:nmax, j=1:nmax
            nlm = SFB.getidx(amodes, i, 0, 0)
            NLM = SFB.getidx(amodes, j, 0, 0)
            w0nn2[i,j] = wmix[nlm,NLM]
        end

        @show norm(w0nn0-w0nn1)/norm(w0nn0)
        @show norm(w0nn0-w0nn2)/norm(w0nn0)
        @test w0nn0 ≈ w0nn1  rtol=1e-6
        @test w0nn0 ≈ w0nn2  rtol=1e-6
        @test w0nn1 ≈ w0nn2  rtol=1e-6

        # compare with cmix
        cmix0 = SFB.set_T1_ell0_expmrr0!(deepcopy(cmix), wmodes, cmodes)
        #@show cmix cmix0
        @show norm(cmix0-cmix)/norm(cmix0)
        @test cmix ≈ cmix0  rtol=1.1e-6

        # bonus: higher ell
        wlnn2 = SFB.sum_m_lmeqLM(wmix, cmodes)
        @test wlnn1 ≈ wlnn2  rtol=1e-6
    end


    run_tests && @testset "win_lnn() rmin=0" begin
    #@testset "win_lnn() rmin=0" begin
        rmin = 0.0
        rmax = 1000.0
        kmax = 0.02
        nr = 1000
        @time amodes = SFB.AnlmModes(kmax, rmin, rmax, cache=false)
        @time cmodes = SFB.ClnnModes(amodes, Δnmax=100)
        @time wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)
        @time win = SFB.make_window(wmodes, :radial_expmrr0, :separable)
        #@time cmix = SFB.power_win_mix(win, wmodes, cmodes)

        wlnn = SFB.win_lnn(win, wmodes, cmodes)
        wmix = SFB.calc_wmix(win, wmodes, amodes)
        wlnn2 = SFB.sum_m_lmeqLM(wmix, cmodes)
        @test wlnn ≈ wlnn2  atol=1e-6

        fskyz2fskyinvlnn(fskyz) = begin
            num_angpix = nside2npix(wmodes.nside)
            fskyinv = SFB.SeparableArray(1 ./ fskyz, ones(num_angpix), name1=:phi, name2=:mask)
            fskyinvlnn = SFB.win_lnn(fskyinv, wmodes, cmodes)
            return fskyinvlnn
        end


        nmax = amodes.nmax
        lmax = amodes.lmax


        # uniform radial selection
        fskyinvlnn0 = fskyz2fskyinvlnn(ones(nr))
        for l=0:lmax, n1=1:nmax, n2=1:nmax
            if SFB.isvalidlnn(cmodes, l, n1, n2)
                idx = SFB.getidx(cmodes, l, n1, n2)
                if n1 == n2
                    @test fskyinvlnn0[idx] ≈ 1  rtol=1e-4
                else
                    @test fskyinvlnn0[idx] ≈ 0  atol=1e-4
                end
            end
        end


        # phi(r) = ℯ^{-r/r0}
        fskyz = win.phi
        fskyinvlnn1 = SFB.get_0nn(fskyz2fskyinvlnn(fskyz), cmodes)
        fskyinvlnn2 = SFB.fskyinv0nn_expmrr0(wmodes, cmodes)

        ret = @test fskyinvlnn1 ≈ fskyinvlnn2  rtol=1e-5

        if ret isa Test.Fail
            @error ret norm(fskyinvlnn1 - fskyinvlnn2) norm(fskyinvlnn1) norm(fskyinvlnn1 - fskyinvlnn2) / norm(fskyinvlnn1)
        end
    end


    # Inhomogeneous masks
    run_tests && @testset "Inhomogeneous Window sep & insep" begin
        rmin = 500.0
        rmax = 1000.0
        nr = 100
        amodes = SFB.AnlmModes(2, 5, rmin, rmax)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)

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
                        (:fullsky,),
                        (:radial_expmrr0,),
                        (:ang_75,),
                        (:ang_75, :radial),
                        #(:ang_half, :radial),
                        (:ang_quarter, :radial),
                        (:separable, :fullsky,),
                        (:separable, :radial_expmrr0,),
                        (:separable, :ang_75,),
                        (:separable, :ang_75, :radial),
                        #(:separable, :ang_half, :radial),
                        (:separable, :ang_quarter, :radial),
                      ]
    run_tests && @testset "Window $(win_features...)" for win_features in win_descriptions
        println()
        @show win_features
        rmin = 500.0
        rmax = 1000.0
        nr = 200
        amodes = SFB.AnlmModes(2, 0, rmin, rmax)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)
        win = SFB.make_window(wmodes, win_features...)

        # wmix
        wmix = SFB.calc_wmix(win, wmodes, amodes)
        wmix_negm = SFB.calc_wmix(win, wmodes, amodes, neg_m=true)


        # M
        cmodes = SFB.ClnnModes(amodes, Δnmax=1)
        M0 = SFB.power_win_mix(wmix, wmix_negm, cmodes)
        M1 = SFB.power_win_mix(win, wmodes, cmodes)

        off_diagonal_index = SFB.getidx(cmodes, 0, 1, 2)
        @show M0[1,off_diagonal_index], M0[off_diagonal_index,1]
        @show M1[1,off_diagonal_index], M1[off_diagonal_index,1]

        if :fullsky in win_features
            # norm(I) = 1
            @show norm(M0-I)
            @show norm(M1-I)
            @test M0 ≈ I  rtol=1e-5
            @test M1 ≈ I  rtol=1e-5
        end

        if rmin == 0 && :radial_expmrr0 in win_features
            M2 = deepcopy(M0)
            M3 = deepcopy(M1)
            SFB.set_T1_ell0_expmrr0!(M2, wmodes, cmodes)
            SFB.set_T1_ell0_expmrr0!(M3, wmodes, cmodes)
            @show M2[1,off_diagonal_index], M2[off_diagonal_index,1]
            @show M3[1,off_diagonal_index], M3[off_diagonal_index,1]
            @test M2[1,1] ≈ M0[1,1]
            @test M3[1,1] ≈ M1[1,1]
            @test M2 ≈ M0  rtol=1e-6
            @test M3 ≈ M1  rtol=1e-6
        end

        @show M0[1:3,1:3] M1[1:3,1:3]
        @test M1 ≈ M0  rtol=1e-6
        #continue


        # CWlnn
        pk(k) = 1e4 * (k/1e-2)^(-3.1)
        Clnn = SFB.gen_Clnn_theory(pk, cmodes)
        Clnn[off_diagonal_index] = 1e5
        CnlmNLM = SFB.Clnn2CnlmNLM(Clnn, cmodes)
        CWlnn0 = M0 * Clnn
        CWlnn1 = M1 * Clnn
        CWlnn2 = SFB.sum_m_lmeqLM(wmix * CnlmNLM * wmix', cmodes)
        @show Clnn CWlnn0 CWlnn1 CWlnn2
        @test CWlnn2 ≈ CWlnn1  rtol=1e-6
        @test CWlnn1 ≈ CWlnn0  rtol=1e-6
        @test CWlnn2 ≈ CWlnn0  rtol=1e-6
        #continue

        # Nshot
        nbar = 3e-4
        Nshotobs = SFB.win_lnn(win, wmodes, cmodes) ./ nbar
        Nshotobs2 = SFB.sum_m_lmeqLM(wmix, cmodes) ./ nbar
        @test isapprox(Nshotobs, Nshotobs2, rtol=1e-2)

        # Bandpower binning basics
        fsky = sum(win[1,:]) / size(win,2)
        Δℓ = round(Int, 1 / fsky)
        w̃, v = SFB.bandpower_binning_weights(cmodes; Δℓ=Δℓ, Δn=1)
        N = w̃ * M1 * v
        w = inv(N) * w̃ * M1
        ṽ = M1 * v * inv(N)
        @test w * v ≈ I  # these really just test linear algebra
        @test w̃ * ṽ ≈ I  # these really just test linear algebra

        # Bandpower binning optimizations
        bcmodes = SFB.ClnnBinnedModes(w̃, v, cmodes)
        Nmix = SFB.power_win_mix(win, w̃, v, wmodes, bcmodes)
        @test Nmix ≈ N
        w̃M = SFB.power_win_mix(win, w̃, I, wmodes, bcmodes)
        Mv = SFB.power_win_mix(win, I, v, wmodes, bcmodes)
        @test w̃M ≈ w̃ * M1  rtol=1e-6
        @test inv(Nmix) * w̃M ≈ w  rtol=1e-10
        @test Mv * inv(Nmix) ≈ ṽ  rtol=1e-10
    end
end


# vim: set sw=4 et sts=4 :
