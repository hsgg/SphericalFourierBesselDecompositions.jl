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
include("test_theory.jl")  # should probably rename this from "theory" to "lnn<->nlm conversion"
include("test_modes.jl")
include("test_wigner_chains.jl")
include("test_nditerators.jl")
include("test_window_chains.jl")
#include("test_covariance.jl")


# vim: set sw=4 et sts=4 :
