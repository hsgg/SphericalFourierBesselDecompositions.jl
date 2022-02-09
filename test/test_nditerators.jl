#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
const SFB = SphericalFourierBesselDecompositions

using Test


@testset "NDIterators" begin
    # test iteration
    N = 4
    Lmin = fill(0, N)
    Lmax = rand(1:9, N)
    ndi = SFB.NDIterator(Lmin, Lmax)
    for L1=0:Lmax[1], L2=0:Lmax[2], L3=0:Lmax[3], L4=0:Lmax[4]
        @test SFB.advance(ndi) == true
        @test ndi[1] == L1
        @test ndi[2] == L2
        @test ndi[3] == L3
        @test ndi[4] == L4
    end
    @test SFB.advance(ndi) == false

    # max < min should give no iteration
    N = 4
    Lmin = [2, 3, 4, 5]
    Lmax = [2, 3, 3, 5]
    L = SFB.NDIterator(Lmin, Lmax)
    @test SFB.advance(L) == false

    # test construction
    N = 4
    Lmin = fill(0, N)
    Lmax = rand(1:9, N)
    L1 = SFB.NDIterator(Lmin, Lmax)
    M1 = SFB.NDIterator(-L1, L1)
    M2 = SFB.NDIterator(0, L1)
    M3 = SFB.NDIterator(L1, 10)
    M4 = SFB.NDIterator(0, 10)
    M5 = SFB.NDIterator(0, 10; N=5)
    @test length(L1) == N
    @test length(M1) == N
    @test length(M2) == N
    @test length(M3) == N
    @test length(M4) == 1
    @test length(M5) == 5

    # test broadcast
    L = SFB.NDIterator(0, 10, N=6)
    SFB.advance(L)  # make L valid
    arr = @. L + 1
    @test arr == fill(1, 6)

    # test array conversion
    @test convert(Array, L) == fill(0,6)
    #Array(L)  # doesn't work

    # firstindex, lastindex
    @test L[begin] == 0
    @test L[end] == 0
end



# vim: set sw=4 et sts=4 :
