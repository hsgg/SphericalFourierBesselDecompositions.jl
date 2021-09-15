# Reference

The functions in here are exported into the main module. We will often assume
that the user has defined the shortcut to the module:
```julia
SFB = SphericalFourierBesselDecompositions
```
Then, all functions can be called via `SFB.funcname()`. For example, the
`SphericalBesselGnl()` constructor in the `SFB.GNL` module is called via
`SFB.SphericalBesselGnl()`.

```@contents
Pages = ["reference.md"]
```

## SFB

```@autodocs
Modules = [SphericalFourierBesselDecompositions]
```

## SFB.GNL

```@autodocs
Modules = [SphericalFourierBesselDecompositions.GNL]
```

## SFB.Modes

```@autodocs
Modules = [SphericalFourierBesselDecompositions.Modes]
```

## SFB.SeparableArrays

```@autodocs
Modules = [SphericalFourierBesselDecompositions.SeparableArrays]
```

## SFB.Cat2Anlm

```@autodocs
Modules = [SphericalFourierBesselDecompositions.Cat2Anlm]
```

## SFB.WindowChains

```@autodocs
Modules = [SphericalFourierBesselDecompositions.WindowChains]
```
