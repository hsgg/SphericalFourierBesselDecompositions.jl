#push!(LOAD_PATH, "../src/")

using Pkg
Pkg.activate(".")

using Documenter, SphericalFourierBesselDecompositions


makedocs(sitename="SphericalFourierBesselDecompositions.jl Documentation",
	 format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true"))

#println(Base.ARGS)
#if "deploy" in Base.ARGS
deploydocs(repo="github.com/hsgg/SphericalFourierBesselDecompositions.jl.git")
#end


# vim: set sw=4 et sts=4 :
