#!/usr/bin/env julia

using SphericalFourierBesselDecompositions
const SFB = SphericalFourierBesselDecompositions
using SpecialFunctions
using FastGaussQuadrature

using Test


function test_orthonormality(amodes)
    rmin = amodes.rmin
    rmax = amodes.rmax
    #for l=0:amodes.lmax
    for l=[0:5; 50:50:amodes.lmax; amodes.lmax]
        nmaxl = amodes.nmax_l[l+1]
        n1set = [1:10; 2 .^ (4:20); nmaxl]
        n1set = unique(n1set[n1set .<= nmaxl])
        #@show l,nmaxl,n1set
        @time for n1=n1set
            n2set = n1 .+ [0:9; 2 .^ (4:20); nmaxl]
            n2set = unique(n2set[n2set .<= nmaxl])
            for n2=n2set
                # Note: by increasing 'nr' we can get much better agreement,
                # but it will also add significant time to the test.
                nr = 16 * (n1 + n2)
                nodes, weights = gausslegendre(nr)
                r = @. (rmin + rmax) / 2 + (rmax - rmin) / 2 * nodes
                Δr = @. weights * (rmax - rmin) / 2

                gnl1 = amodes.basisfunctions.(n1, l, r)
                gnl2 = amodes.basisfunctions.(n2, l, r)
                δᴷ = sum(@. Δr * r^2 * gnl1 * gnl2)

                #@show l,n1,n2,δᴷ
                if n1 == n2
                    @test δᴷ ≈ 1  atol=1e-6
                else
                    @test δᴷ ≈ 0  atol=1e-6
                end
            end
        end
    end
end


@testset "GNL" begin
    @testset "$boundary" for boundary in [SFB.GNL.potential, SFB.GNL.velocity, SFB.GNL.sphericalbessel_kF]
        @testset "rmin=$rmin" for rmin in [0.0, 500.0]#, 2000.0]
            @testset "kmax=$kmax" for kmax in [0.01, 0.1]
                println("=====")
                @show boundary rmin kmax

                #rmax = 2000.0
                rmax = 2467.10

                do_plots = false

                if do_plots
                    # plot eqn for determining knl
                    figure()
                    k = 1e-5:1e-4:1.2kmax
                    knl = SFB.GNL.calc_knl(maximum(k), rmin, rmax; boundary)
                    l = 0:2
                    z = SFB.GNL.get_knl_zero_func(boundary).(k, l', rmin, rmax)
                    plot(k, z)
                    xlabel("k")
                    ylabel("\$k_{n\\ell}\$ zeros  function")
                    hlines(0, extrema(k)..., color="0.75")
                    vlines(knl[:,1], extrema(z)..., color="0.75")
                    vlines(knl[:,2], extrema(z)..., color="0.75", ls="--")
                    vlines(knl[:,3], extrema(z)..., color="0.75", ls=":")

                    @show knl[1:3,1]
                    @show knl[1:3,2]
                    @show knl[1:3,3]
                end

                do_plots && if rmin == 0
                    # plot jl'
                    figure()
                    j0 = @. sphericalbesselj(0, k*rmax)
                    j1 = @. sphericalbesselj(1, k*rmax)
                    j2 = @. sphericalbesselj(2, k*rmax)
                    j0_p = @. -j1
                    j1_p = @. j0 - (1+1) / (k*rmax) * j1
                    j2_p = @. j1 - (2+1) / (k*rmax) * j2
                    plot(k, j0_p)
                    plot(k, j1_p)
                    plot(k, j2_p)
                    xlabel("k")
                    ylabel("\$j'_\\ell(k*r_{\\rm max}\$)")
                    hlines(0, extrema(k)..., color="0.75")
                    vlines(knl[:,1], extrema(j0_p)..., color="0.75")
                    vlines(knl[:,2], extrema(j1_p)..., color="0.75", ls="--")
                    vlines(knl[:,3], extrema(j2_p)..., color="0.75", ls=":")

                    # plot jl
                    figure()
                    nr = 2000
                    Δr = (rmax - rmin) / nr
                    r = range(rmin + Δr/2, rmax - Δr/2, length=nr)
                    j0 = @. sphericalbesselj(0, knl[1:3,1]' * r)
                    j1 = @. sphericalbesselj(1, knl[1:3,2]' * r)
                    j2 = @. sphericalbesselj(2, knl[1:3,3]' * r)
                    norm0 = sum((@. r^2 * Δr * j0^2), dims=1)
                    norm1 = sum((@. r^2 * Δr * j1^2), dims=1)
                    norm2 = sum((@. r^2 * Δr * j2^2), dims=1)
                    @. j0 /= √norm0
                    @. j1 /= √norm1
                    @. j2 /= √norm2
                    @show norm0 norm1 norm2
                    plot(r, j0)
                    gca().set_prop_cycle(nothing)
                    plot(r, j1, "--")
                    gca().set_prop_cycle(nothing)
                    plot(r, j2, ":")
                    plot(NaN, NaN, "k", label="\$\\ell=0\$")
                    plot(NaN, NaN, "k--", label="\$\\ell=1\$")
                    plot(NaN, NaN, "k:", label="\$\\ell=2\$")
                    xlabel("r")
                    ylabel("\$j_\\ell(k_{n\\ell}*r)\$")
                    hlines(0, extrema(r)..., color="0.75")
                    legend()

                    # test for orthonormality
                    for l=0:2
                        if l == 0
                            jl = j0
                        elseif l == 1
                            jl = j1
                        elseif l == 2
                            jl = j2
                        end
                        for n1=1:3, n2=1:3
                            δᴷ = sum(@. Δr * r^2 * jl[:,n1] * jl[:,n2])
                            @show l,n1,n2,δᴷ
                            if n1 == n2
                                @test δᴷ ≈ 1
                            else
                                @test δᴷ ≈ 0  atol=1e-7
                            end
                        end
                    end
                end


                do_cache = (kmax < 0.02)
                @show do_cache
                @time amodes = SFB.AnlmModes(kmax, rmin, rmax, cache=do_cache; boundary)
                @show amodes.lmax amodes.nmax amodes.lmax_n amodes.nmax_l
                @test amodes.lmax == maximum(amodes.lmax_n)
                @test amodes.nmax == maximum(amodes.nmax_l)

                @time cmodes = SFB.ClnnModes(amodes, Δkmax=0)
                knl = amodes.knl[isfinite.(amodes.knl)]
                lkk = SFB.getlkk(cmodes)
                @test all(@. knl < kmax)
                @test all(@. lkk[2,:] < kmax)
                @test all(@. lkk[3,:] < kmax)

                s = @. knl < kmax
                @show length(knl) length(knl[s])

                if boundary == SFB.GNL.sphericalbessel_kF
                    knl = amodes.knl
                    lmax = size(knl,2) - 1
                    for l in rand(0:lmax,5), n in rand(1:size(knl,1), 5)  # no need to test every combination
                        @show l,n
                        nr = @. 3 + 12 * ceil(Int, 2 * knl[n,l+1] * (rmax - rmin))
                        Δr = (rmax - rmin) / nr
                        myr = rmin:Δr:rmax+Δr/2
                        jl = √(2/π) * knl[n,l+1] * SFB.GNL.sphericalbesselj.(l, knl[n,l+1]*myr)
                        gnl = amodes.basisfunctions.(n, l, myr)
                        @test jl ≈ gnl
                    end
                else
                    println("Testing orthonormality ($boundary,rmin=$rmin,kmax=$kmax)...")
                    @time test_orthonormality(amodes)
                end
            end
        end
    end
end






# vim: set sw=4 et sts=4 :
