#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
ENV["JULIA_DEBUG"] = SphericalFourierBesselDecompositions

using Random
randseed = rand(UInt64)
@show randseed
Random.seed!(randseed)


## runtests:
include("test_separablearrays.jl")
include("test_windows.jl")
include("test_theory.jl")
include("test_modes.jl")
include("test_window_chains.jl")


# vim: set sw=4 et sts=4 :
