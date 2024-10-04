# Copyright (c) 2024 California Institute of Technology (“Caltech”). U.S.
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


# Purpose: Define the interface for generating gnl(r). The frontend, `GNL()`,
# to create an AbstractGNL is in the top-level
# SphericalFourierBesselDecompositions.jl file.



@doc raw"""
    GNLs

Module to define radial basis functions. The specific basis functions are
defined in an enum, `BoundaryConditions`.
"""
module GNLs


############# GNL interface #######################################

abstract type AbstractGNL end
using ..Splines


# This will allow to call GNL.potential:
@doc raw"""
    BoundaryConditions

This enum defines the radial basis functions.
"""
@enum BoundaryConditions potential velocity sphericalbessel_kF cryognl #density
abstract type GNL end  # workaround to be able to do GNL.potential, etc.
Base.getproperty(t::Type{GNL}, b::Symbol) = begin
    if b in Symbol.(instances(BoundaryConditions))
        return eval(b)
    else
        return getfield(t, b)  # fallback, needed for printing
    end
end



export AbstractGNL
export GNL
export SphericalBesselGnl  # compat, equal to GNL



@doc raw"""
    GNL(nmax, lmax, rmin, rmax)
    GNL(kmax, rmin, rmax; nmax=typemax(Int64), lmax=typemax(Int64))

Generate `gnl(n,l,r)`. Returns a struct that can be called for calculating
`gnl`. Note that the last argument is `r`, *not* `kr`.
"""
function GNL(nmax, lmax, rmin, rmax; boundary=potential, kwargs...)
    (rmin < rmax) || @error "rmin >= rmax" rmin rmax

    if boundary == cryognl
        return CryoGNL(nmax, lmax, rmin, rmax; kwargs...)
    end
    return SphericalBesselGNL(nmax, lmax, rmin, rmax; boundary, kwargs...)
end

function GNL(kmax, rmin, rmax; boundary=potential, kwargs...)
    (rmin < rmax) || @error "rmin >= rmax" rmin rmax

    if boundary == cryognl
        return CryoGNL(kmax, rmin, rmax; kwargs...)
    end
    return SphericalBesselGNL(kmax, rmin, rmax; boundary, kwargs...)
end



# evaluate gnl(r)

(sphbesg::AbstractGNL)(n, l, r) = evaluate(sphbesg, n, l, r)

evaluate(sphbesg::AbstractGNL, n, l, r) = sphbesg.gnl[n,l+1](r)

# derivative gnl'(r)
Splines.derivative(sphbesg::AbstractGNL, n, l, r) = derivative(sphbesg.gnl[n,l+1], r)



# compatibility:
const SphericalBesselGnl = GNL


# Specific implementations:
include("SphericalBesselGNLs.jl")
include("CryoGNLs.jl")


end


# vim: set sw=4 et sts=4 :
