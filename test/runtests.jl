#!/usr/bin/env julia


using Test

using SphericalFourierBesselDecompositions
#SFB = SphericalFourierBesselDecompositions
#using .SFB.SeparableArrays

using Random

ENV["JULIA_DEBUG"] = SphericalFourierBesselDecompositions

randseed = rand(UInt64)
@show randseed
Random.seed!(randseed)


## runtests:
include("test_separablearrays.jl")
include("test_windows.jl")
include("test_theory.jl")
include("test_modes.jl")


# vim: set sw=4 et sts=4 :
