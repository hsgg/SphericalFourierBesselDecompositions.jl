#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
const SFB = SphericalFourierBesselDecompositions

using Test
using LinearAlgebra
#using Profile
using Healpix

#using PyPlot

@testset "Cat2Anlm" begin

    run_tests = true

    run_tests && @testset "basis function transform" begin
        rmin = 0.0
        rmax = 1000.0
        kmax = 0.02
        nbar = 3e-4
        nr = 200
        nside = 32
        fkp_nbar_pk = 3e-4 * 1e4
        amodes = SFB.AnlmModes(kmax, rmin, rmax, nside=nside)
        cmodes = SFB.ClnnModes(amodes, Δnmax=0)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)
        rθϕ = zeros(3, 0)

        #idxmax = min(10, SFB.getnlmsize(amodes))
        idxmax = SFB.getnlmsize(amodes)
        idxworst = 0
        wurstkaese = 0.0
        for idx in rand(1:idxmax, 10)  # no need to be exhaustive every time
            n, l, m = SFB.getnlm(amodes, idx)
            win = SFB.get_full_basisfuncreal_nlm(amodes, wmodes, n, l, m)[:,:]
            wrhatln = SFB.win_rhat_ln(win, wmodes, amodes)
            anlm = -SFB.cat2amln(rθϕ, amodes, nbar, wrhatln, [])
            @show idx n,l,m anlm[idx]  # anlm[1:idx-1] anlm[idx+1:end]
            @test eltype(wrhatln) <: Real
            expectation = (m == 0) ? 1 : (-1)^m / √2
            @test anlm[idx] ≈ expectation  atol=1e-3
            @test maximum(abs, anlm[1:idx-1]) ≤ 2e-3
            @test maximum(abs, anlm[idx+1:end]) ≤ 2e-3
            anlm[idx] -= expectation
            worst = maximum(abs, anlm)
            idxworst = (worst > wurstkaese) ? idx : idxworst
            wurstkaese = (worst > wurstkaese) ? worst : wurstkaese
        end
        n, l, m = SFB.getnlm(amodes, idxworst)
        @show idxworst n,l,m wurstkaese
    end


    run_tests && @testset "Cat2Anlm with weights" begin
        rmin = 0.0
        rmax = 1000.0
        kmax = 0.01
        nbar = 3e-4
        nr = 200
        fkp_nbar_pk = 3e-4 * 1e4
        amodes = SFB.AnlmModes(kmax, rmin, rmax)
        cmodes = SFB.ClnnModes(amodes, Δnmax=0)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)
        win = SFB.make_window(wmodes, :radial_cossin_l00_m00)

        T = Float64
        Ngalaxies = round(Int, (2*rmax)^3 * nbar)
        xyz = 2 * T(rmax) * rand(T, 3, Ngalaxies) .- T(rmax)
        rθϕ = SFB.xyz2rtp(xyz)
        rθϕ = SFB.apply_window(rθϕ, win, wmodes)
        @assert all(@. rmin <= rθϕ[1,:] <= rmax)

        weights_1_r = ones(size(win))
        #weights_const_r = fill(0.25, size(win))
        weights_fkp_r = @. 1 / (1 + win * fkp_nbar_pk)
        #@show size(weights_1_r) size(weights_const_r) size(weights_fkp_r)

        wrhatln_1 = SFB.win_rhat_ln(win .* weights_1_r, wmodes, amodes)
        #wrhatln_const = SFB.win_rhat_ln(win .* weights_const_r, wmodes, amodes)
        wrhatln_fkp = SFB.win_rhat_ln(win .* weights_fkp_r, wmodes, amodes)
        #@show size(wrhatln_1) size(wrhatln_const) size(wrhatln_fkp)
        wrhatln_0 = zeros(size(wrhatln_1))

        #weights_1 = SFB.winweights2galweights(weights_1_r, wmodes, rθϕ)
        #weights_const = SFB.winweights2galweights(weights_const_r, wmodes, rθϕ)
        weights_fkp = SFB.winweights2galweights(weights_fkp_r, wmodes, rθϕ)
        #@show size(weights_1) size(weights_const) size(weights_fkp)

        #@test all(weights_1 .== 1)
        #@test all(weights_const .== 0.25)

        Ngal = size(rθϕ,2)
        Veff = SFB.integrate_window(win, wmodes)
        nbar_s = Ngal / Veff
        ##nbar_s = nbar
        #anlm_1 = SFB.cat2amln(rθϕ, amodes, nbar_s, wrhatln_1, weights_1)
        #anlm_const = SFB.cat2amln(rθϕ, amodes, nbar_s, wrhatln_const, weights_const)
        #rθϕ = fill(0.0, 3, 0)
        anlm_fkp = SFB.cat2amln(rθϕ, amodes, nbar_s, wrhatln_fkp, weights_fkp)

        #@test all(anlm_const .== 0.25 * anlm_1)

        #NW = SFB.win_lnn(win, wmodes, cmodes) ./ nbar_s
        #NwW_1 = SFB.win_lnn(win .* weights_1_r .^ 2, wmodes, cmodes) ./ nbar_s
        #NwW_const = SFB.win_lnn(win .* weights_const_r .^ 2, wmodes, cmodes) ./ nbar_s
        #NwW_fkp = SFB.win_lnn(win .* weights_fkp_r .^ 2, wmodes, cmodes) ./ nbar_s
        #cmix_1 = SFB.power_win_mix(win .* weights_1_r, wmodes, cmodes)
        #cmix_const = SFB.power_win_mix(win .* weights_const_r, wmodes, cmodes)
        #cmix_fkp = SFB.power_win_mix(win .* weights_fkp_r, wmodes, cmodes)
        #Nobs_1 = SFB.calc_NobsA(NwW_1, NW, cmix_1, nbar_s, Veff, cmodes)
        #Nobs_const = SFB.calc_NobsA(NwW_const, NW, cmix_const, nbar_s, Veff, cmodes)
        #Nobs_fkp = SFB.calc_NobsA(NwW_fkp, NW, cmix_fkp, nbar_s, Veff, cmodes)

        # round for readability
        #Nobs_1 = Float16.(Nobs_1)
        #Nobs_const = Float16.(Nobs_const)
        #Nobs_fkp = Float16.(Nobs_fkp)
        #anlm_1 = Complex{Float16}.(anlm_1)
        #anlm_const = Complex{Float16}.(anlm_const)
        anlm_fkp = Complex{Float16}.(anlm_fkp)

        #nyi(l,n,n′) = SFB.getidx(cmodes,l,n,n′)
        #idx = [nyi(0,1,1), nyi(1,1,1), nyi(2,1,1), nyi(3,1,1), nyi(4,1,1)]
        #@show .√Nobs_1[idx]
        #@show .√Nobs_const[idx]
        #@show .√Nobs_fkp[idx]

        myi(n,l,m) = SFB.getidx(amodes,n,l,m)
        idx = [myi(1,0,0), myi(1,1,0), myi(1,1,1), myi(1,2,0), myi(1,2,1), myi(1,2,2)]
        #@show abs.(anlm_1[idx])
        #@show abs.(anlm_const[idx])
        #@show abs.(anlm_fkp[idx])
        @show anlm_fkp[idx]
    end


    run_tests && @testset "field2anlm()" begin
        rmin = 500.0
        rmax = 1000.0
        nmax = 2
        lmax = 3
        nside = 256
        nr = 100
        amodes = SFB.AnlmModes(nmax, lmax, rmin, rmax, nside=nside)
        wmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)
        f = SFB.make_window(wmodes)
        f .= 0

        rr, Δr = SFB.window_r(wmodes)
        gnl = amodes.basisfunctions
        Ωₚ = 4π / nside2npix(nside)

        Δj = 3 * nside^2 + 7
        Δi = 41
        for j=1:Δj:nside2npix(nside), i=1:Δi:nr
            f[i,j] = 1

            r = rr[i]
            θ, ϕ = pix2angRing(Resolution(nside), j)
            @show i,j,r,θ,ϕ

            anlm1 = round.(SFB.field2anlm(f, wmodes, amodes), sigdigits=6)

            nlmsize = SFB.getnlmsize(amodes)
            anlm0 = fill(NaN*im, nlmsize)
            for nlm=1:nlmsize
                n, l, m = SFB.getnlm(amodes, nlm)
                ylm = SFB.sphericalharmonicsy(l, m, θ, ϕ)
                anlm0[nlm] = round.(Δr * r^2 * gnl(n,l,r) * conj(ylm) * Ωₚ, sigdigits=6)
                danlm = anlm0[nlm] - anlm1[nlm]
                #@show n,l,m,anlm0[nlm],anlm1[nlm]
                #@assert anlm0[nlm] ≈ anlm1[nlm]  rtol=1e-5 atol=1e-15
            end
            #@show anlm0 anlm1
            @test anlm0 ≈ anlm1  rtol=1e-5
            @assert isapprox(anlm0, anlm1, rtol=1e-5)

            f[i,j] = 0
        end
    end
end


# vim: set sw=4 et sts=4 :
