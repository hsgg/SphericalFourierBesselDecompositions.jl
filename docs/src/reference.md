# Reference

The functions in here are exported into the main module. We will often assume
that the user has defined the shortcut to the module:
```julia
import SphericalFourierBesselDecompositions as SFB
```
Then, all functions can be called via `SFB.funcname()`. For example, the
`SFB.SphericalBesselGnl()` constructor in the `SFB.GNLs` module is called
via `SFB.SphericalBesselGnl()`.

```@contents
Pages = ["reference.md"]
```

## SFB

```@autodocs
Modules = [SphericalFourierBesselDecompositions]
```

## SFB.GNLs

```@autodocs
Modules = [SphericalFourierBesselDecompositions.GNLs]
```

### SFB.GNL.CryoGNLs

```@autodocs
Modules = [SphericalFourierBesselDecompositions.GNLs.CryoGNLs]
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

## SFB.Theory

```@autodocs
Modules = [SphericalFourierBesselDecompositions.Theory]
```

## SFB.MyBroadcast

```@autodocs
Modules = [SphericalFourierBesselDecompositions.MyBroadcast]
```

## SFB.MyBroadcast.MeshedArrays

```@autodocs
Modules = [SphericalFourierBesselDecompositions.MyBroadcast.MeshedArrays]
```
