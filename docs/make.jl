using SphericalFourierBesselDecompositions
using Documenter

DocMeta.setdocmeta!(SphericalFourierBesselDecompositions, :DocTestSetup, :(using SphericalFourierBesselDecompositions); recursive=true)

magickcmd = "magick"
if isnothing(Sys.which(magickcmd))
    magickcmd = "convert"
end
magickopts = `-density 300 -background none -define icon:auto-resize=256,128,96,64,48,32,16`
cmd = `$magickcmd $(@__DIR__)/src/assets/favicon.svg $magickopts $(@__DIR__)/src/assets/favicon.ico`
@show cmd
run(cmd)

makedocs(;
    modules=[SphericalFourierBesselDecompositions],
    authors="Henry Gebhardt <henry.s.gebhardt@jpl.nasa.gov> and contributors",
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
        "Tutorial: Radial basis functions" => "tutorial_gnl.md",
        "Reference" => "reference.md",
        "Index" => "myindex.md",
    ],
)

deploydocs(;
    repo="github.com/hsgg/SphericalFourierBesselDecompositions.jl",
)
