# Copyright (c) 2020 California Institute of Technology (“Caltech”). U.S.
# Government sponsorship acknowledged.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   • Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   • Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   • Neither the name of Caltech nor its operating division, the Jet
#   Propulsion Laboratory, nor the names of its contributors may be used to
#   endorse or promote products derived from this software without specific
#   prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Purpose: compute knl and gnl(r) for potential boundary condition

# Notes: Currently we use arbitrary precision numbers to calculate the odd
# cases where Float64 is insufficient. There are several caveats to be aware
# of.
#
# First, we save the coefficients cnl and dnl in double precision. In most
# circumstances that will be sufficient. However, there may be edge cases where
# that needs to be changed. A possible solution is to also use arbitrary
# precision to store cnl and dnl when necessary.
#
# Another solution may be to find good implementations of logbesselj() and
# logbessely() to avoid arbitrary precision.
#
# Second, knl are calculate confidently, and Float64 is sufficient to store
# them. We may use that to do entirely different approaches to calculating
# gnl(r): anything from brute-force arbitrary precision to logbessel
# implementations to an integral representation, series representation, or what
# not.
#
# Third, and perhaps more important is the question of how to test the accuracy
# of the calculated basis function gnl(r). Given confidence in knl, we would
# need to test that the gnl(r) adhere to Bessel's differential equation and
# satisfy the boundary conditions at rmin and rmax.


module GNL


export SphericalBesselGnl


using SpecialFunctions
using ArbNumerics  # Nemo conflicts currently
using Roots
using ..Splines

using Distributed
using SharedArrays

# We don't really need high precision. We need large range with floatmax() >>
# 1e300, and Arb with 1024 bits precision does that for us.
const HighprecisionFloat = ArbFloat{1024}


############## calculate zeros ##################3

function doublesecant_method(fn, a, b; maxevals=1000, xrtol=eps(1.0))
    fa = fn(a)
    fb = fn(b)
    #@show typeof(a) typeof(fa)
    #@show fa fb
    @assert sign(fa) * sign(fb) <= 0
    neval = 0
    while a != b && neval < maxevals
        m = (fb - fa) / (b - a)
        c = b - fb / m
        #if c==b || c==a
        if abs(c - b) <= xrtol * c || abs(c - a) <= xrtol * c
            return c
        end
        fc = fn(c)
        d = b - (fb + fc) / m
        #@show a,c,d,b
        #@show a-c,a-d,b-c,b-d
        #@show typeof(a) typeof(b)
        #@show typeof(c) typeof(b-d)
        #@assert a <= c <= b
        #@show fa,fc,fb
        if sign(fa) * sign(fc) <= 0
            b = c
            fb = fc
        elseif sign(fb) * sign(fc) <= 0
            a = c
            fa = fc
        else
            a = c
            fa = fc
            b = c
            fb = fc
        end

        # also try to move the other side of the interval
        if a <= d <= b
            fd = fn(d)
            if sign(fa) * sign(fd) <= 0
                b = d
                fb = fd
            elseif sign(fb) * sign(fd) <= 0
                a = d
                fa = fd
            else
                a = d
                fa = fd
                b = d
                fb = fd
            end
        end

        neval += 1
    end
    @error "maxevals reached" maxevals
    return typeof(a)(NaN)
end


function calc_next_fn_zero(func, x, δ; maxevals=1000, xrtol=1e-10)
    #@show "==="
    f = func(x)
    #@show x,f,typeof(f)
    xnew = x + δ
    fnew = func(xnew)
    while sign(fnew) == sign(f)
        #@show xnew,fnew
        x = xnew
        f = fnew
        xnew += δ
        fnew = func(xnew)
    end
    #@show x,f
    #@show xnew,fnew
    #@show xnew-2δ,func(xnew-2δ)
    #@show xnew-2δ,func(xnew-2δ)*fnew
    interval = (x, xnew)
    #@show interval func.(interval)
    #@time xnew = find_zero(func, xnew, maxevals=maxevals, xrtol=xrtol)
    #@show xnew,func(xnew)
    #xnew = find_zero(func, interval, maxevals=maxevals, xrtol=xrtol)
    #@show xnew,func(xnew)
    #@show x,xnew
    #xatol = xnew*1e-10
    #xnew = find_zero(func, interval, Roots.Brent(), maxevals=maxevals, xrtol=xrtol, xatol=xatol)
    #@show xnew,func(xnew)
    #xnew = find_zero(func, interval, Roots.FalsePosition(), maxevals=maxevals, xrtol=xrtol)
    #@show xnew,func(xnew)
    #xnew = Roots.secant_method(func, interval, maxevals=maxevals, xrtol=xrtol)
    #@show xnew,func(xnew)
    xnew = doublesecant_method(func, x, xnew, maxevals=maxevals, xrtol=xrtol)
    #@show xnew,func(xnew)
    @assert isfinite(xnew)
    return xnew
end


@doc raw"""
    calc_first_n_zeros(func, nmax; δ=π/20, xmin=0.0)

Calculate the first `nmax` zeros of the function `func`. Assumes that zeros are
spaced more than `δ` apart. First zero could be `xmin`.
"""
function calc_first_n_zeros(func, nmax; δ=π/20, xmin=0.0)
    xn = fill(NaN, nmax)
    for n=1:nmax
        xn[n] = try
            calc_next_fn_zero(func, xmin, δ)
        catch e
            isa(e, SpecialFunctions.AmosException) || rethrow(e)
            T = HighprecisionFloat
            Float64(calc_next_fn_zero(x->T(func(x)), T(xmin), T(δ)))
        end
        xmin = xn[n] + δ
    end
    return xn
end


@doc raw"""
    calc_zeros(func, xmin, xmax; δ=π/20)

Calculate all zeros of the function `func` between `xmin` and `xmax`. Assumes
that zeros are spaced more than `δ` apart.
"""
function calc_zeros(func, xmin, xmax; δ=π/20)
    xn = Float64[]
    while xmin <= xmax
        x = try
            calc_next_fn_zero(func, xmin, δ)
        catch e
            #println(e)
            isa(e, SpecialFunctions.AmosException) || rethrow(e)
            T = HighprecisionFloat
            Float64(calc_next_fn_zero(x->T(func(x)), T(xmin), T(δ)))
        end
        if xmin <= x <= xmax
            push!(xn, x)
        end
        xmin = x + δ
    end
    return xn
end


############## calculate knl for potential boundary conditions ##################

function knl_zero_function_potential(k, l, rmin, rmax)
    #@show k,l,rmin,rmax
    scale = k^3 / (1 + abs(k)^3)
    jlmax = besselj(l-1//2, k*rmax)
    (k*rmin == 0) && return jlmax
    ylmax = bessely(l-1//2, k*rmax) * scale
    jlmin = besselj(l+3//2, k*rmin)
    ylmin = bessely(l+3//2, k*rmin) * scale
    return jlmax * ylmin - jlmin * ylmax
end


@doc raw"""
    calc_knl_potential(nmax, lmax, rmin, rmax)
    calc_knl_potential(kmax, rmin, rmax; nmax=typemax(Int64), lmax=typemax(Int64))

Calculate the `knl` for potential boundary conditions.
"""
function calc_knl_potential(nmax, lmax, rmin, rmax)
    knl = fill(NaN, nmax, lmax+1)
    for l=0:lmax
        δ = π/rmax/4
        xmin = (l + 3//2) / rmax

        func0(k) = knl_zero_function_potential(k, l, rmin, rmax)
        knl[:,l+1] = calc_first_n_zeros(func0, nmax, δ=δ, xmin=xmin)
    end
    @assert all(knl .> 0)
    return knl
end


function calc_knl_potential(kmax, rmin, rmax; nmax=typemax(Int64), lmax=typemax(Int64))
    nmax_calc = ceil(Int64, kmax * rmax / π) + 1
    lmax_calc = ceil(Int64, kmax * rmax)
    kmax_lim = (lmax > lmax_calc && nmax > nmax_calc)
    nmax = min(nmax, nmax_calc)
    lmax = min(lmax, lmax_calc)
    knl = fill(NaN, nmax, lmax+1)
    for l=0:lmax
        δ = π/rmax/4
        kmin = (l + 3//2) / rmax

        func0(k) = knl_zero_function_potential(k, l, rmin, rmax)
        kn = calc_zeros(func0, kmin, kmax, δ=δ)

        if nmax < length(kn)
            @error "nmax too small" nmax length(kn) l δ kmin rmin rmax
            throw("nmax too small")
            @assert nmax >= length(kn)
        end
        kmax_lim && (l==lmax) && @assert length(kn) == 0
        for i=1:length(kn)
            knl[i,l+1] = kn[i]
        end
    end
    @assert all(knl[@. isfinite(knl)] .> 0)
    return knl
end


############## calculate gnl(r) for potential boundary conditions ##################


function sphericalbesselj(nu, x::T) where {T<:ArbNumber}
    besselj_nuhalf_x = besselj(nu + 1//2, x)
    if abs(x) ≤ sqrt(eps(real(zero(besselj_nuhalf_x))))
        nu == 0 ? one(besselj_nuhalf_x) : zero(besselj_nuhalf_x)
    else
        √((float(T))(π)/2x) * besselj_nuhalf_x
    end
end
sphericalbessely(nu, x::T) where {T<:ArbNumber} = √((float(T))(π)/2x) * bessely(nu + 1//2, x)

sphericalbesselj(nu, x) = SpecialFunctions.sphericalbesselj(nu, x)
sphericalbessely(nu, x) = SpecialFunctions.sphericalbessely(nu, x)



function calc_sphbes_gnl(q, l, c::Tyl, d::Tyl) where {Tyl}
    jl = sphericalbesselj(l, Tyl(q))
    (d == 0) && return c * jl
    yl = sphericalbessely(l, Tyl(q))
    return c * jl + d * yl
end


bes_yl(l, x) = begin
    (x == 0) && return -typeof(x)(Inf)
    isnan(x) && return typeof(x)(NaN)
    return bessely(l,x)
end


function calc_cnl_dnl(knl, n, l, rmin, rmax)
    #dc1 = - besselj(l-1+1//2, knl*rmax) / bessely(l-1+1//2, knl*rmax)
    #@show dc1
    dc2 = - besselj(l+1+1//2, knl*rmin) / bes_yl(l+1+1//2, knl*rmin)
    #@show dc2
    dc = dc2 #@. √(dc1 * dc2)
    #@show dc
    gnl_rmin = calc_sphbes_gnl(knl*rmin, l, one(dc), dc)
    gnl_rmax = calc_sphbes_gnl(knl*rmax, l, one(dc), dc)
    #@show gnl_rmin gnl_rmax
    num_one = (rmax^3 * gnl_rmax^2 - rmin^3 * gnl_rmin^2) / 2
    #@show num_one
    @assert num_one >= 0
    # Note: the sign doesn't really matter
    cnl = (-1)^(n + (1-floor(Int, 1/(l+1))) * (1-floor(Int,1/n))) / √num_one
    dnl = dc * cnl
    return cnl, dnl
end


function calc_cnl_dnl(knl, rmin, rmax, Tyl=Float64)
    nmax, lmax = size(knl) .- (0,1)
    cnl = fill(Tyl(NaN), size(knl))
    dnl = fill(Tyl(NaN), size(knl))
    n_float64 = 0
    n_arbfloat = 0
    for n=1:nmax, l=0:lmax
        !isfinite(knl[n,l+1]) && continue
        c, d = try
            n_float64 += 1
            c1, d1 = calc_cnl_dnl(knl[n,l+1], n, l, rmin, rmax)
            Tyl(c1), Tyl(d1)
        catch e
            isa(e, SpecialFunctions.AmosException) || rethrow(e)
            T = HighprecisionFloat
            n_arbfloat += 1
            c1, d1 = calc_cnl_dnl(T(knl[n,l+1]), n, l, T(rmin), T(rmax))
            Tyl(c1), Tyl(d1)
        end
        cnl[n,l+1] = c
        dnl[n,l+1] = d
    end
    @show n_float64,n_arbfloat
    return cnl, dnl
end


function gen_gnl_cache(knl, rmin, rmax, sphbesg)
    nmax, lmax = size(knl) .- (0,1)
    #T = typeof(Spline1D(0.1:0.1:1.0, Float64.(sphbesg.(1, 0, 0.1:0.1:1.0))))
    gnl = fill(Spline1D(), size(knl))
    #gnl = SharedArray{T}(size(knl)...)
    for n=1:nmax
        for l=0:lmax
            k = knl[n,l+1]
            isfinite(k) || continue
            nr = 12 * ceil(Int, 2 * k * (rmax - rmin))
            Δr = (rmax - rmin) / nr
            myr = rmin:Δr:rmax+Δr/2
            myg = sphbesg.(n,l,myr)
            gnl[n,l+1] = Spline1D(myr, Float64.(myg))
        end
    end
    return gnl
end


##################### SphericalBesselGnl ###############################33

struct SphericalBesselGnl{Tcache,Tyl}
    nmax::Int
    lmax::Int
    rmin::Float64
    rmax::Float64
    knl::Array{Float64,2}
    cnl::Array{Tyl,2}
    dnl::Array{Tyl,2}
    gnl::Tcache
end


function SphericalBesselGnl(nmax, lmax, rmin, rmax, knl, cnl, dnl)
    sphbesg = SphericalBesselGnl(nmax, lmax, rmin, rmax, knl, cnl, dnl, nothing)
    #return sphbesg  # Don't commit
    gnl = gen_gnl_cache(knl, rmin, rmax, sphbesg)
    return SphericalBesselGnl(nmax, lmax, rmin, rmax, knl, cnl, dnl, gnl)
end


@doc raw"""
    SphericalBesselGnl(nmax, lmax, rmin, rmax)
    SphericalBesselGnl(kmax, rmin, rmax; nmax=typemax(Int64), lmax=typemax(Int64))

Generate `gnl(n,l,r)`. Returns a struct that can be called for calculating
`gnl`. Note that the last argument is `r`, *not* `kr`.
"""
function SphericalBesselGnl(nmax, lmax, rmin, rmax; cache=true)
    (rmin < rmax) || @error "rmin >= rmax" rmin rmax
    knl = calc_knl_potential(nmax, lmax, rmin, rmax)
    cnl, dnl = calc_cnl_dnl(knl, rmin, rmax)
    if !cache
        return SphericalBesselGnl(nmax, lmax, rmin, rmax, knl, cnl, dnl, nothing)
    end
    return SphericalBesselGnl(nmax, lmax, rmin, rmax, knl, cnl, dnl)
end

function SphericalBesselGnl(kmax, rmin, rmax; cache=true, nmax=typemax(Int64), lmax=typemax(Int64))
    (rmin < rmax) || @error "rmin >= rmax" rmin rmax
    knl = calc_knl_potential(kmax, rmin, rmax; nmax, lmax)
    cnl, dnl = calc_cnl_dnl(knl, rmin, rmax)
    nmax, lmax = size(knl) .- (0,1)
    if !cache
        return SphericalBesselGnl(nmax, lmax, rmin, rmax, knl, cnl, dnl, nothing)
    end
    return SphericalBesselGnl(nmax, lmax, rmin, rmax, knl, cnl, dnl)
end


# evaluate gnl(r)

function evaluate(sphbesg::SphericalBesselGnl{Nothing,Tyl}, n, l, r) where{Tyl}
    k = sphbesg.knl[n,l+1]
    c = sphbesg.cnl[n,l+1]
    d = sphbesg.dnl[n,l+1]
    return calc_sphbes_gnl(k*r, l, c, d)
end

function evaluate(sphbesg::SphericalBesselGnl{Array{Tspl,2},Tyl}, n, l, r) where {Tspl,Tyl}
    return sphbesg.gnl[n,l+1](r)
end

(sphbesg::SphericalBesselGnl)(n, l, r) = evaluate(sphbesg, n, l, r)


end


# vim: set sw=4 et sts=4 :
