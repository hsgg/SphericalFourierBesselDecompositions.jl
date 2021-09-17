#!/usr/bin/env julia


using Test

using SphericalFourierBesselDecompositions
SFB = SphericalFourierBesselDecompositions
using .SFB.SeparableArrays

## This might be quicker instead of loading the full SFB module:
#include("../src/SeparableArrays.jl")
#using .SeparableArrays


@testset "SeparableArrays" begin
    @testset "SeparableArrays matrix functionality" begin
        phi = rand(5)
        mask = rand(3)
        A = phi * mask'  # our reference
        S = SeparableArray(phi, mask; name1=:phi, name2=:mask)
        for i in eachindex(S)
            @test S[i] == A[i]
        end
        for i=1:length(S)
            @test S[i] == A[i]
        end
        for i=1:size(S,1), j=1:size(S,2)
            @test S[i,j] == A[i,j]
        end
        @test all(S .== A)
        @test S[5,3,1] == A[5,3,1]
        @test_throws BoundsError S[5,3,2]
        @test_throws BoundsError S[6,1]
        @test_throws BoundsError S[2,4]
        @test_throws BoundsError S[0,1]
        @test_throws BoundsError S[1,0]
        @test_throws BoundsError S[6,0]
        @test ndims(S) == ndims(A)
        @test S.arr1 == phi
        @test S.arr2 == mask
        @test S.phi == phi
        @test S.mask == mask
        @test propertynames(S) == (:phi, :mask)
        S2 = @SeparableArray phi mask
        @test S2 == S
        @test propertynames(S2) == (:phi, :mask)
        @test S2.phi == S.phi
        @test S2.mask == S.mask

        # squaring
        S3 = S .^ 2
        S4 = S .* S
        #S5 = [S[i,j]*S[i,j] for i=1:size(S,1),j=1:size(S,2)]
        S5 = exponentiate(S, 2)
        @show S3 S4 S5
        @show (S4 .- S3) ./ eps.(2*S3)
        @show (S5 .- S3) ./ eps.(2*S3)
        @test all(abs.(S4 .- S3) .<= eps.(2*S4))
        @test all(abs.(S5 .- S3) .<= eps.(2*S5))
        @test all(abs.(S5 .- S4) .<= eps.(2*S5))
        #@test typeof(S3) <: SeparableArray
        #@test typeof(S4) <: SeparableArray
        @test typeof(S5) <: SeparableArray
    end


    @testset "SeparableArrays ndims>2 functionality" begin
        phi = rand(2,3)
        mask = rand(4,5)
        A = [phi[i,j]*mask[k,l]
             for i=1:size(phi,1), j=1:size(phi,2), k=1:size(mask,1), l=1:size(mask,2)]
        S = SeparableArray(phi, mask; name1=:phi, name2=:mask)

        pass = true
        for i in eachindex(S)
            S[i] == A[i] || (pass = false)
            #@show i S[i] A[i]
            #@assert pass
        end
        @test pass

        pass = true
        for i=1:length(S)
            S[i] == A[i] || (pass = false)
        end
        @test pass

        pass = true
        for i=1:size(S,1), j=1:size(S,2), k=1:size(S,3), l=1:size(S,4)
            S[i,j,k,l] == A[i,j,k,l] || (pass = false)
        end
        @test pass

        @test all(S .== A)

        @test_throws BoundsError S[5,3,2,3]
        @test_throws BoundsError S[1,0,1,1]
        @test_throws BoundsError S[6,0,1,1]
        @test_throws BoundsError S[0,1,1,1]
        @test_throws BoundsError S[0,6,1,1]
        @test_throws BoundsError S[1,1,1,0]
        @test_throws BoundsError S[1,1,6,0]
        @test_throws BoundsError S[1,1,0,1]
        @test_throws BoundsError S[1,1,0,6]
        @test ndims(S) == ndims(A)
        @test S.arr1 == phi
        @test S.arr2 == mask
        @test S.phi == phi
        @test S.mask == mask
        @test propertynames(S) == (:phi, :mask)
        S2 = @SeparableArray phi mask
        @test S2 == S
        @test propertynames(S2) == (:phi, :mask)
        @test S2.phi == S.phi
        @test S2.mask == S.mask
    end


    @testset "SeparableArrays (1,1) timing and allocations" begin
        phi = rand(1000)
        mask = rand(3400)
        # our reference
        A = phi * mask'
        S = @SeparableArray phi mask

        S[1,1]  # compile
        @time S[1,1]

        S[:,110]
        S[110,:]
        @time S[:,110]
        @time S[110,:]
        @time phi[110]*mask
        @time phi[110]*mask
    end

    @testset "SeparableArrays (2,2) timing and allocations" begin
        phi = rand(100,120)
        mask = rand(340,110)
        # our reference
        A = [phi[i] * mask[j,k] for i=1:length(phi), j=1:size(mask,1), k=1:size(mask,2)]
        S = @SeparableArray phi mask

        S[1,1,1,1]  # compile
        @time S[1,1,1,1]

        S[1,:,2,1]
        S[1,2,2,:]
        @time S[1,:,2,1]
        @time S[1,2,2,:]
    end
end




# vim: set sw=4 et sts=4 :
