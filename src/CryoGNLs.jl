# This module contains the functions needed for generating the radial basis
# functions numerically using the 󰓎CRYOFUNK󰓎  method, https://arxiv.org/abs/2109.13352 .



############# core CryoGNL implementation:

@doc raw"""
CryoGNLs

To generate the radial basis functions, first define the radial binning, e.g.,
```julia
rmin = 1000.0
rmax = 4000.0
nbins = 1000
Δr = (rmax - rmin) / nbins
rmid = range(rmin + Δr/2, rmax - Δr/2, length=nbins)
```
Then, to generate the radial basis functions at `ℓ=3`,
```julia
ell = 3
k, g, ginv = CryoGNLs.get_radial_cryofunks(ell, rmid, Δr)
```
The functions `gnl(r)` are now saved in `g`, so that `g[i,n]` is `gnl(rmid[i])`
and the corresponding `k`-mode is `k[n]`.
"""
module CryoGNLs

export get_radial_cryofunks

using LinearAlgebra
using KrylovKit: eigsolve


function radial_greens_function(L, r1, r2)
    if r1 <= r2
        return r2 / (2*L+1) * (r1/r2)^L
    else
        return r2 / (2*L+1) * (r2/r1)^(L+1)
    end
end


function get_radial_greens_matrix(L, ri, Δr)
    Δro2Lp1 = Δr / (2 * L + 1)
    G = fill(NaN, length(ri), length(ri))
    Threads.@threads for j=1:length(ri)
        for i=j:length(ri)
            # G[i,j] = G[j,i] = Δr * (ri[i] / ri[j]) * radial_greens_function(L, ri[i], ri[j])
            G[i,j] = G[j,i] = Δro2Lp1 * ri[j] * (ri[j] / ri[i])^L
        end
    end
    return Hermitian(G)
end


function get_radial_cryofunks(L, ri, Δr)
    B = diagm(@. √Δr * ri)
    G = get_radial_greens_matrix(L, ri, Δr)
    e = eigen(G)
    k = @. 1 / √e.values
    p = sortperm(k)
    k = k[p]
    Z = e.vectors[:,p]
    g = inv(B) * Z
    ginv = Z' * B
    return k, g, ginv
end


# This is an optimized version when the number of eigenvectors needed is small.
function get_radial_cryofunks(L, ri, Δr, nmax, G)
    B = diagm(@. √Δr * ri)
    # G = get_radial_greens_matrix(L, ri, Δr)

    x0 = ones(size(G,1))
    krylovdim = min(length(ri), ceil(Int, nmax*1.05 + 2))
    eval, evec, conv = eigsolve(G, x0, nmax; krylovdim)
    if length(eval) != conv.converged
        @error "Only $(conv.converged) of $(length(eval)) eigenvectors successfully calculated." eval evec conv L ri Δr nmax
        error("Failed to converge all requested eigenvectors.")
    end

    k = @. 1 / √eval
    p = sortperm(k)
    k = k[p]

    Z = fill(NaN, length(ri), length(p))
    for i in axes(Z,2)
        Z[:,i] .= evec[p[i]]
    end

    g = B \ Z
    ginv = Z' * B
    return k, g, ginv
end


end # module CryoGNLs


################ CryoGNL interface ###############

using .CryoGNLs


struct CryoGNL{Tcache} <: AbstractGNL
    nmax::Int
    lmax::Int
    rmin::Float64
    rmax::Float64
    knl::Array{Float64,2}
    boundary::BoundaryConditions
    gnl::Tcache
end


function CryoGNL(nmax, lmax, rmin, rmax; nrbins=10*nmax, kwargs...)
    println("CryoGNL() unused args: $kwargs")

    Δr = (rmax - rmin) / nrbins
    rmid = range(rmin + Δr/2, rmax - Δr/2, length=nrbins)

    nmax = min(nmax, nrbins)

    knl = fill(NaN, nmax, lmax + 1)
    gnl = fill(Spline1D(T=Float64), size(knl))

    @time for l in 0:lmax
        k, g, _ = CryoGNLs.get_radial_cryofunks(l, rmid, Δr)
        knl[:,l+1] .= k[1:nmax]

        for n=1:nmax
            gnl[n,l+1] = Spline1D(rmid, g[:,n])
        end
    end

    println("Created CryoGNL()")
    T = Float64
    return CryoGNL(nmax, lmax, T(rmin), T(rmax), knl, cryognl, gnl)
end


function CryoGNL(kmax, rmin, rmax; nrbins=Inf, kwargs...)
    println("CryoGNL() unused args: $kwargs")

    if nrbins == Inf
        drmax = 1 / (12 * kmax)  # guesstimate
        nrbins = ceil(Int, (rmax - rmin) / drmax)
    end

    Δr = (rmax - rmin) / nrbins
    rmid = range(rmin + Δr/2, rmax - Δr/2, length=nrbins)

    # estimate nmax
    k, _, _ = CryoGNLs.get_radial_cryofunks(0, rmid, Δr)
    nmax = sum(k .<= kmax)
    my_nmax = nmax

    # estimate lmax
    lmax = ceil(Int64, kmax * rmax)


    knl = fill(NaN, nmax, lmax + 1)
    gnl = fill(Spline1D(T=Float64), size(knl))
    G = CryoGNLs.get_radial_greens_matrix(0, rmid, Δr)
    Ratio = @. rmid / rmid'

    @time for l in 0:lmax
        @show lmax,l,nmax,my_nmax

        k, g, _ = @time CryoGNLs.get_radial_cryofunks(l, rmid, Δr, my_nmax, G)
        @. G.data = G.data * Ratio * (2 * l + 1) / (2 * l + 3)
        # G = CryoGNLs.Hermitian(G.data)

        my_nmax = findlast(k .<= kmax)
        # @show nrbins,nmax,my_nmax

        if isnothing(my_nmax)
            lmax = l - 1
            break
        end

        @. knl[1:my_nmax,l+1] = k[1:my_nmax]

        for n=1:my_nmax
            gnl[n,l+1] = Spline1D(rmid, g[:,n])
        end
    end

    println("Created CryoGNL()")
    T = Float64
    return CryoGNL(nmax, lmax, T(rmin), T(rmax), knl, cryognl, gnl)
end




# vim: set sw=4 et sts=4 :
