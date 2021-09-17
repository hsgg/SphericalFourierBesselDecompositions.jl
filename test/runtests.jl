#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
ENV["JULIA_DEBUG"] = SphericalFourierBesselDecompositions
SFB = SphericalFourierBesselDecompositions

using Test

using Random
randseed = rand(UInt64)
@show randseed
Random.seed!(randseed)

using LinearAlgebra

long_tests = true


## runtests:
@testset "SphericalFourierBesselDecompositions" begin
    #include("test_separablearrays.jl")
    #include("test_gnl.jl")
    #include("test_cat2anlm.jl")
    include("test_windows.jl")
    include("test_theory.jl")  # should probably rename from "theory" to "lnn<->nlm conversion"
    include("test_modes.jl")
    include("test_wigner_chains.jl")
    include("test_nditerators.jl")
    include("test_window_chains.jl")
    include("test_covariance.jl")
end


# vim: set sw=4 et sts=4 :
