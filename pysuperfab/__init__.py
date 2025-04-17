import juliacall

jl = juliacall.newmodule("pysuperfab")

jl.seval("import SphericalFourierBesselDecompositions as SFB")

SFB = jl.SFB
