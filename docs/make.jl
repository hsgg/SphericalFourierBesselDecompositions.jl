#push!(LOAD_PATH, "../src/")
using Pkg
Pkg.activate("..")

using Documenter, SphericalFourierBesselDecompositions

#makedocs(sitename="SphericalFourierBesselDecompositions.jl Documentation")

makedocs(sitename="SphericalFourierBesselDecompositions.jl Documentation",
	 format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true"))
