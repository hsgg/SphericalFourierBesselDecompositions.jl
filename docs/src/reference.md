# Reference

The functions in here are exported into the main module. We will often assume
that the user has defined the shortcut to the module:
```julia
const SFB = SphericalFourierBesselDecompositions
```
Then, all functions can be called via `SFB.funcname()`. For example, the
`SFB.GNL.SphericalBesselGnl()` constructor in the `SFB.GNL` module is called
via `SFB.SphericalBesselGnl()`. Making `SFB` a `const` can be important for
performance if individual functions from the `SFB` module are called within a
tight loop.

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

## SFB.Windows

```@autodocs
Modules = [SphericalFourierBesselDecompositions.Windows]
```

## SFB.WindowChains

```@autodocs
Modules = [SphericalFourierBesselDecompositions.WindowChains]
```
