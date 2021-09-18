using SphericalFourierBesselDecompositions
using Documenter

DocMeta.setdocmeta!(SphericalFourierBesselDecompositions, :DocTestSetup, :(using SphericalFourierBesselDecompositions); recursive=true)

magickopts = `-density 300 -background none -define icon:auto-resize=256,128,96,64,48,32,16`
cmd = `convert $(@__DIR__)/src/assets/favicon.svg $magickopts $(@__DIR__)/src/assets/favicon.ico`
@show cmd
run(cmd)

makedocs(;
    modules=[SphericalFourierBesselDecompositions],
    authors="Henry Gebhardt <henry.s.gebhardt@jpl.nasa.gov> and contributors",
    repo="https://github.com/hsgg/SphericalFourierBesselDecompositions.jl/blob/{commit}{path}#{line}",
    sitename="SphericalFourierBesselDecompositions.jl",
    #sitename="[SphericalFourierBesselDecompositions.jl](https://github.com/hsgg/SphericalFourierBesselDecompositions.jl)",
    #sitename="<a href=\"https://github.com/hsgg/SphericalFourierBesselDecompositions.jl\">SphericalFourierBesselDecompositions.jl</a>",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://hsgg.github.io/SphericalFourierBesselDecompositions.jl",
	assets = ["assets/favicon.ico"],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial: Catalogue and Window" => "tutorial_catalog.md",
        "Reference" => "reference.md",
        "Index" => "myindex.md",
    ],
)

deploydocs(;
    repo="github.com/hsgg/SphericalFourierBesselDecompositions.jl",
)
