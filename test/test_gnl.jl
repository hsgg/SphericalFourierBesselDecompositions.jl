#!/usr/bin/env julia

using SphericalFourierBesselDecompositions
const SFB = SphericalFourierBesselDecompositions

using Test

@testset "GNL" begin
    @testset "$boundary" for boundary in [SFB.GNL.potential, SFB.GNL.velocity]
        @testset "rmin=$rmin" for rmin in [0.0, 500.0]
            @testset "kmax=$kmax" for kmax in [0.0614]#, 0.1]
                rmax = 2000.0

                #figure()
                #k = 1e-5:1e-4:1e-2
                #knl = SFB.GNL.calc_knl(maximum(k), rmin, rmax; boundary)
                #l = 0:2
                #z = SFB.GNL.get_knl_zero_func(boundary).(k, l', rmin, rmax)
                #plot(k, z)
                #hlines(0, extrema(k)..., color="0.75")
                #vlines(knl[:,1], extrema(z)..., color="0.75")
                #vlines(knl[:,2], extrema(z)..., color="0.75", ls="--")
                #vlines(knl[:,3], extrema(z)..., color="0.75", ls=":")

                amodes = SFB.AnlmModes(kmax, rmin, rmax, cache=false; boundary)
                @show amodes.lmax amodes.nmax amodes.lmax_n amodes.nmax_l
                @test amodes.lmax == maximum(amodes.lmax_n)
                @test amodes.nmax == maximum(amodes.nmax_l)

                cmodes = SFB.ClnnModes(amodes, Δnmax=0)
                knl = amodes.knl[isfinite.(amodes.knl)]
                lkk = SFB.getlkk(cmodes)
                @test cmodes.Δnmax == maximum(cmodes.Δnmax_l)
                @test cmodes.Δnmax == maximum(cmodes.Δnmax_n)
                @test all(@. knl < kmax)
                @test all(@. lkk[2,:] < kmax)
                @test all(@. lkk[3,:] < kmax)

                #figure()
                #r = rmin:1.0:rmax
                #plot(r, amodes.basisfunctions.(1, 0, r))

                s = @. knl < kmax
                @show length(knl) length(knl[s])
            end
        end
    end
end






# vim: set sw=4 et sts=4 :
