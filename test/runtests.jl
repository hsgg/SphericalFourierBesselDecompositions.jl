#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
ENV["JULIA_DEBUG"] = SphericalFourierBesselDecompositions
const SFB = SphericalFourierBesselDecompositions

using Test

using Random
randseed = rand(UInt64)
@show randseed
Random.seed!(randseed)
#Random.seed!(0xdffe22467edd2cf0)  # needs more lenient SeparableArrays test

using LinearAlgebra



## runtests:
@testset "SphericalFourierBesselDecompositions" begin
    include("test_toplevel.jl")
    include("test_separablearrays.jl")
    include("test_gnl.jl")
    include("test_cat2anlm.jl")
    include("test_windows.jl")
    include("test_theory.jl")  # should probably rename from "theory" to "lnn<->nlm conversion"
    include("test_modes.jl")
    include("test_wigner_chains.jl")
    include("test_nditerators.jl")
    include("test_window_chains.jl")
    include("test_covariance.jl")
    include("test_sfb.jl")

    ## Only intended for interactive testing:
    # include("calc_cmix.jl")
end


# vim: set sw=4 et sts=4 :
