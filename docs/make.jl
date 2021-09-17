using SphericalFourierBesselDecompositions
using Documenter

DocMeta.setdocmeta!(SphericalFourierBesselDecompositions, :DocTestSetup, :(using SphericalFourierBesselDecompositions); recursive=true)

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
        assets=String[],
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
