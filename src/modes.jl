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


# Purpose: This module deals with the way objects are stored in memory, as well
# as the aspects of binning into bandpowers. This is accessed by exporting
# three structures: AnlmModes, ClnnModes, and ClnnBinnedModes.

# TODO:
#
# - When binning, we should only consider Δℓ and Δn, where the resulting bin
# contains the modes within a volume Δℓ⋅Δn². That is, we need to consider what
# the actual binning strategy in the n-n′ direction shall be. Probably, we want
# to do something simple such as hafl the modes in n-n′=1 go to the diagonal,
# the other half to the first off-diagonal in binned space. However, this is
# lower priority, because we have rₘᵢₙ boundary conditions.


module Modes

using ..GNL
using Statistics
using LinearAlgebra
using SparseArrays

export AnlmModes, ClnnModes, ClnnBinnedModes
export estimate_nside, estimate_nr
export getnlmsize, getlmsize, getnlm, getidx, getklm, isvalidnlm
export getlnnsize, getlnn, isvalidlnn, isvalidlnn_symmetric
export getlkk, getidxapprox
export bandpower_binning_weights, bandpower_eigen_weights
export get_incomplete_bins, find_most_hi_k_convolved_mode, find_klarge, get_hi_k_affection



########################## AnlmModes: define ordering of modes in data vector ########

## Note: The maximum is lmax=3*nside-1, which is OK for bandpower limited
## functions. In general, it may be better to use lmax=2*nside. Furthermore,
## our algorithms frequently need to reference quantities up to L=2*lmax.
#estimate_nside(lmax) = 2^max(2, ceil(Int, log2((lmax + 1) / 3)))
estimate_nside(lmax) = 2^max(2, ceil(Int, log2((2*lmax + 1) / 2)))
#estimate_nside(lmax) = 4 * 2^max(2, ceil(Int, log2((lmax + 1) / 3)))
#estimate_nside(lmax) = 32 * 2^max(2, ceil(Int, log2((lmax + 1) / 3)))


@doc raw"""
    AnlmModes(kmax, rmin, rmax; cache=true, nside=nothing, boundary=SFB.GNL.potential)
    AnlmModes(nmax, lmax, rmin, rmax; cache=true, nside=nothing, boundary=SFB.GNL.potential)

This is where we define which modes are included. As our criterium, we set a
maximum qmax = kmax * rmax, and we include all modes below that.

The modes are arranged in the following order. The fastest loop is through 'm',
then 'l', finally 'n', from small number to larger number. We restrict 'm' to
m >= 0, and we assume a real field.

Example:
```julia-repl
julia> kmax = 0.05
julia> rmin = 500.0
julia> rmax = 1000.0
julia> modes = AnlmModes(kmax, rmin, rmax)
```
"""
struct AnlmModes{T}
    rmin::Float64
    rmax::Float64
    kmax::Float64
    basisfunctions::T

    # calculated, public:
    nmax::Int64
    lmax::Int64
    nmax_l::Array{Int64,1}
    lmax_n::Array{Int64,1}
    nside::Int64
    knl::Array{Float64,2}
end

Base.broadcastable(s::AnlmModes) = s


function AnlmModes(kmax::Real, rmin::Real, rmax::Real; cache=true, nside=nothing, nmax=typemax(Int64), lmax=typemax(Int64), boundary=GNL.potential)
    sphbesg = SphericalBesselGnl(kmax, rmin, rmax; nmax, lmax, cache, boundary)
    knl = sphbesg.knl
    nmax, lmax = size(knl) .- (0,1)
    modes = @. knl <= kmax
    mylmax(n) = begin
        last = findlast(modes[n,:]) - 1
        isa(last, Integer) || return -1
        return last
    end
    mynmax(l) = begin
        last = findlast(modes[:,l+1])
        isa(last, Integer) || return -1
        return last
    end
    nmax = findlast(modes[:,1])
    lmax_n = [mylmax(n) for n=1:nmax]
    lmax = findlast(modes[1,:]) - 1
    nmax_l = [mynmax(l) for l=0:lmax]
    if !all(nmax_l .> 0)
        lbad = findfirst(nmax_l .<= 0) - 1
        @error "nmax_l out of range" kmax rmin rmax cache lbad nmax_l[lbad+1] knl[:,lbad+1]
    end
    @assert all(lmax_n .>= 0)
    #@show nmax lmax lmax_n nmax_l
    if nside == nothing
        nside = estimate_nside(lmax)
    end
    kmax = maximum(x->isnan(x) ? -Inf : x, knl)
    knl = collect(knl[1:nmax,1:lmax+1])
    return AnlmModes(rmin, rmax, kmax, sphbesg, nmax, lmax, nmax_l, lmax_n, nside, knl)
end


function AnlmModes(nmax::Int, lmax::Int, rmin::Real, rmax::Real; cache=true, nside=nothing, boundary=GNL.potential)
    lmax_n = [lmax for n=1:nmax]
    nmax_l = [nmax for l=0:lmax]
    sphbesg = SphericalBesselGnl(nmax, lmax, rmin, rmax; cache, boundary)
    if nside == nothing
        nside = estimate_nside(lmax)
    end
    knl = sphbesg.knl
    kmax = maximum(x->isnan(x) ? -Inf : x, knl)
    return AnlmModes(rmin, rmax, kmax, sphbesg, nmax, lmax, nmax_l, lmax_n, nside, knl)
end


function getlmsize(lmax)
    return lmax * (lmax + 1) ÷ 2 + lmax + 1
end


function getnlmsize(modes::AnlmModes, nmax=modes.nmax)
    s = 0
    for n=1:nmax
        s += getlmsize(modes.lmax_n[n])
    end
    return s
end


function isvalidnlm(amodes::AnlmModes, n, l, m)
    if n > amodes.nmax
        return false
    end
    if l > amodes.lmax_n[n]
        return false
    end
    return abs(m) <= l
end


function getnlm(modes::AnlmModes, idx)
    n = 1
    nmodes = getlmsize(modes.lmax_n[n])
    while idx > nmodes
        idx -= nmodes
        n += 1
        nmodes = getlmsize(modes.lmax_n[n])
    end
    l = 0
    nmodes = l + 1
    while idx > nmodes
        idx -= nmodes
        l += 1
        nmodes = l + 1
    end
    m = idx - 1
    return n, l, m
end


function getidx(modes::AnlmModes, n, l, m)
    @assert n >= 1
    @assert l >= 0
    @assert m >= 0
    idx = 1
    idx += getnlmsize(modes, n-1)
    idx += getlmsize(l-1)
    idx += m
    return idx
end


function getklm(modes::AnlmModes, n, l, m)
    k = modes.knl[n,l+1]
    return k, l, m
end

function getklm(modes::AnlmModes, idx)
    n, l, m = getnlm(modes, idx)
    k, l, m = getklm(modes, n, l, m)
    return k, l, m
end


########################## ClnnModes: define ordering of modes in data vector ########

@doc raw"""
    ClnnModes(::AnlmModes)
    ClnnModes(::AnlmModes, ::AnlmModes)

This is where we define which modes are included in the power spectrum, given a
`AnlmModes` struct.

The modes are arranged in the following order. The fastest loop is through 'n̄',
then 'Δn', finally 'ℓ', from small number to larger number. We make the
convention that `n̄` is the smaller of the k-modes, and `Δn >= 0`.

More useful is the labeling by `ℓ, n₁, n₂`. In that case we make the convention
that `n₁ = n̄` and `n₂ = n̄ + Δn`.

If two `AnlmModes` are given, then we assume a cross-correlation is desired.
The `S` parameter specifies whether the resulting modes are symmetric (S=true)
on interchange of k1 and k2, or not (S=false). This is useful for auto- and
cross-correlation, respectively.
"""
struct ClnnModes{S,Ta,Tb}
    amodesA::AnlmModes{Ta}
    amodesB::AnlmModes{Tb}
    Δkmax::Float64
    Δnmax::Int
    # cache:
    lnn::Matrix{Int}
end

Base.broadcastable(x::ClnnModes) = x

# backwards compatibility:
Base.getproperty(cmodes::ClnnModes{true}, property::Symbol) = begin
    if property == :amodes
        return getfield(cmodes, :amodesA)
    end
    return getfield(cmodes, property)
end


# Here we will sort first by l, then Δn, then n
function sort_lnn(lnn)
    lnnsize = size(lnn,2)
    idxs_sorted = sort(1:lnnsize;
         lt = (i,j) -> begin
             li, ni1, ni2 = @view lnn[:,i]
             lj, nj1, nj2 = @view lnn[:,j]
             Δni = ni2 - ni1
             Δnj = nj2 - nj1
             if li != lj
                 return li < lj
             elseif Δni != Δnj
                 return Δni < Δnj
             else
                 return ni1 < nj1
             end
         end)
    return collect(lnn[:,idxs_sorted])
end


function calc_lnn(amodesA, amodesB; Δkmax=Inf, Δnmax=typemax(Int), symmetric_kk=false)
    lmax = min(amodesA.lmax, amodesB.lmax)
    ell = 0:lmax
    knlA = amodesA.knl[:,ell.+1]
    knlB = amodesB.knl[:,ell.+1]

    Δkmax_out = -Inf
    Δnmax_out = 0

    lnn = Int64[]
    for l=0:lmax
        nAmax = amodesA.nmax_l[l+1]
        nBmax = amodesB.nmax_l[l+1]
        for nA = 1:nAmax, nB = (symmetric_kk ? nA : 1):nBmax
            kA = knlA[nA,l+1]
            kB = knlB[nB,l+1]
            Δk = kB - kA
            if abs(Δk) <= Δkmax  &&  abs(nB - nA) <= Δnmax
                append!(lnn, (l, nA, nB))
                Δkmax_out = max(abs(Δk), Δkmax_out)
                Δnmax_out = max(abs(nB - nA), Δnmax_out)
            end
        end
    end
    lnn = reshape(lnn, 3, :)

    lnn = sort_lnn(lnn)

    return collect(lnn), Δkmax_out, Δnmax_out
end


function ClnnModes(amodes::AnlmModes{Ta}; Δkmax=Inf, Δnmax=typemax(Int)) where {Ta}
    lnn, Δkmax, Δnmax = calc_lnn(amodes, amodes; Δkmax, Δnmax, symmetric_kk=true)
    return ClnnModes{true,Ta,Ta}(amodes, amodes, Δkmax, Δnmax, lnn)
end


function ClnnModes(amodesA::AnlmModes{Ta}, amodesB::AnlmModes{Tb}; Δkmax=Inf, Δnmax=typemax(Int)) where {Ta,Tb}
    lnn, Δkmax, Δnmax = calc_lnn(amodesA, amodesB; Δkmax, Δnmax, symmetric_kk=false)
    return ClnnModes{false,Ta,Tb}(amodesA, amodesB, Δkmax, Δnmax, lnn)
end


getlnnsize(modes::ClnnModes) = size(modes.lnn,2)


getlnn(cmodes::ClnnModes, idx) = begin
    l = cmodes.lnn[1,idx]
    n1 = cmodes.lnn[2,idx]
    n2 = cmodes.lnn[3,idx]
    return l, n1, n2
end

getlnn(cmodes::ClnnModes) = cmodes.lnn


function getidx(cmodes::ClnnModes{S}, l, n1, n2) where {S}
    if S
        n1, n2 = minmax(n1, n2)
    end
    lnnsize = getlnnsize(cmodes)
    idx = findfirst(1:lnnsize) do i
        all(cmodes.lnn[:,i] .== (l, n1, n2))
    end
    if isnothing(idx)
        lA = min(l, cmodes.amodesA.lmax)
        lB = min(l, cmodes.amodesB.lmax)
        n1A = min(n1, cmodes.amodesA.nmax_l[lA+1])
        n2B = min(n2, cmodes.amodesB.nmax_l[lB+1])
        Δk = cmodes.amodesB.knl[n2B,lB+1] - cmodes.amodesA.knl[n1A,lA+1]
        @error "Cannot find index" l,n1,n2 lnnsize idx S cmodes.Δkmax cmodes.amodesA.lmax cmodes.amodesB.lmax cmodes.amodesA.nmax cmodes.amodesB.nmax isvalidlnn(cmodes, l, n1, n2) cmodes.amodesA.nmax_l[lA+1] cmodes.amodesB.nmax_l[lB+1] cmodes.amodesA.knl[n1A,lA+1] cmodes.amodesB.knl[n2B,lB+1] Δk
    end
    return idx
end


function isvalidlnn(cmodes::ClnnModes{S}, l, n1, n2, ::Val{VERBOSE}=Val(false)) where {S,VERBOSE}
    if VERBOSE
        @show S
    end
    (S && n2 < n1) && return false

    abs(n2 - n1) <= cmodes.Δnmax || return false

    if VERBOSE
        @show size(cmodes.amodesA.knl)
        @show size(cmodes.amodesB.knl)
    end
    size(cmodes.amodesA.knl,2) >= l+1  ||  return false
    size(cmodes.amodesB.knl,2) >= l+1  ||  return false

    size(cmodes.amodesA.knl,1) >= n1  ||  return false
    size(cmodes.amodesB.knl,1) >= n2  ||  return false

    kA = cmodes.amodesA.knl[n1,l+1]
    kB = cmodes.amodesB.knl[n2,l+1]
    if VERBOSE
        @show kA,kB
    end
    isnan(kA) && return false
    isnan(kB) && return false

    Δk = kB - kA
    if VERBOSE
        @show Δk
    end
    abs(Δk) <= cmodes.Δkmax  ||  return false

    return true
end


function isvalidlnn_symmetric(cmodes::ClnnModes, l, n1, n2, args...)
    n1, n2 = minmax(n1, n2)
    return isvalidlnn(cmodes, l, n1, n2, args...)
end


function check_isvalidclnn_symmetric(cmodes::ClnnModes, l, n1, n2)
    if !isvalidlnn_symmetric(cmodes, l, n1, n2)
        lmax = cmodes.amodes.lmax
        nmax = cmodes.amodes.nmax
        nmax_l = cmodes.amodes.nmax_l
        nmax_li = nmax_l[max(1,min(l+1,length(nmax_l)))]
        Δnmax = cmodes.Δnmax
        Δnmax_l = cmodes.Δnmax_l
        Δnmax_li = Δnmax_l[max(1,min(l+1,length(Δnmax_l)))]
        Δnmax_n = cmodes.Δnmax_n
        Δnmax_n1i = Δnmax_n[max(1,min(n1,length(Δnmax_n)))]
        Δnmax_n2i = Δnmax_n[max(1,min(n2,length(Δnmax_n)))]
        @error "lnn not valid" l,n1,n2 lmax nmax,nmax_li Δnmax,Δnmax_li,Δnmax_n1i,Δnmax_n2i
        error("lnn not valid")
    end
end


@doc raw"""
    getlkk(::ClnnModes, [i])
    getlkk(::ClnnBinnedModes, [i])

Get the physical modes ℓ, k, and k' corresponding to the index `i`. If `i` is
left out, an array `lkk` of all modes is returned so that `lkk[1,:]` are all
the ℓ-values, `lkk[2,:]` all the k-values, and `lkk[3,:]` are all the
k'-values.
"""
function getlkk(cmodes::ClnnModes, i)
    l, n1, n2 = getlnn(cmodes, i)
    k1 = cmodes.amodesA.knl[n1,l+1]
    k2 = cmodes.amodesB.knl[n2,l+1]
    return l, k1, k2
end


function getlkk(cmodes::ClnnModes)
    l, k1, k2 = getlkk(cmodes, 1)
    lkk = fill(eltype(promote(l, k1, k2))(0), 3, getlnnsize(cmodes))
    for i=1:size(lkk,2)
        l, k1, k2 = getlkk(cmodes, i)
        lkk[1,i] = l
        lkk[2,i] = k1
        lkk[3,i] = k2
    end
    return lkk
end


function getlkk(cmodes::ClnnModes, l, n1, n2)
    k1 = cmodes.amodesA.knl[n1,l+1]
    k2 = cmodes.amodesB.knl[n2,l+1]
    return l, k1, k2
end


function estimate_nr(cmodes::ClnnModes; quick=false)
    # Note this is a pretty shitty algorithm. Speeding it up should be fairly easy.
    lnnsize = getlnnsize(cmodes)
    if quick
        @show 8*cmodes.amodes.nmax  2.5*cmodes.amodes.lmax
        return 8 * cmodes.amodes.nmax
    end
    Nsampmax = 1
    for i=1:lnnsize, i′=1:lnnsize
        l, n, n′ = getlnn(cmodes, i)
        L, N, N′ = getlnn(cmodes, i′)
        Nsamp1 = 8 * (n + N) + l + L
        Nsamp2 = 8 * (n′ + N′) + l + L
        Nsampmax = max(Nsampmax, Nsamp1, Nsamp2)
    end
    return Nsampmax
end


############################### ClnnBinnedModes ###########################
# The ClnnBinnedModes struct is basically the same as the ClnnModes struct.
# However, it adds to it functionality that allows the modes to be different
# from the indices. That is, if the actual mode is ℓ and its index l, then we
# don't require ℓ=l.

struct ClnnBinnedModes{T,S,Ta,Tb}
    cmodes::ClnnModes{S,Ta,Tb}
    LKK::Array{T,2}  # l=LKK[1,:], k1=LKK[2,:], k2=LKK[3,:]
end

Base.broadcastable(s::ClnnBinnedModes) = s


function ClnnBinnedModes(w̃, v, cmodes::ClnnModes{S}) where {S}
    (w̃ != I) && @assert all(sum(w̃, dims=2) .≈ 1)  # ensure w̃ is normalized
    LKK = getlkk(cmodes) * w̃'
    if S
        # ensure k1 <= k2
        for i=1:size(LKK,2)
            LKK[2:3,i] .= extrema(LKK[2:3,i])
        end
    end
    return ClnnBinnedModes(cmodes, LKK)
end


getlnnsize(bcmodes::ClnnBinnedModes) = size(bcmodes.LKK,2)

getlkk(bcmodes::ClnnBinnedModes, i) = begin
    #@show i,bcmodes.LKK[2,i]
    bcmodes.LKK[1,i], bcmodes.LKK[2,i], bcmodes.LKK[3,i]
end

getlkk(bcmodes::ClnnBinnedModes) = bcmodes.LKK


# _getcblnn_helper(): get surrounding interval range
_getcblnn_helper(arr, i) = begin
    lo = arr[max(1, i - 1)]
    mi = arr[i]
    hi = arr[min(i + 1, length(arr))]
    if mi < lo && mi < hi
        hi = min(lo, hi)
        lo = mi - (hi - mi) / 2
    elseif mi > lo && mi > hi
        lo = max(lo, hi)
        hi = mi + (mi - lo) / 2
    end
    @assert lo <= mi <= hi
    lo = (lo + mi) / 2
    hi = (hi + mi) / 2
    return lo, hi
end

getidxapprox(bcmodes::ClnnBinnedModes{T,S}, ℓ, k1, k2) where{T,S} = begin
    if S
        k1, k2 = minmax(k1, k2)
    end
    i = findfirst(i -> begin
                      ℓ_lo, ℓ_hi = _getcblnn_helper(bcmodes.LKK[1,:], i)
                      (ℓ_lo <= ℓ <= ℓ_hi) || return false
                      #@show i,ℓ_lo,ℓ_hi

                      k1_lo, k1_hi = _getcblnn_helper(bcmodes.LKK[2,:], i)
                      #@show i,k1_lo,k1_hi
                      (k1_lo <= k1 <= k1_hi) || return false

                      k2_lo, k2_hi = _getcblnn_helper(bcmodes.LKK[3,:], i)
                      (k2_lo <= k2 <= k2_hi) || return false
                      #@show i,k2_lo,k2_hi
                      return true
                  end,
                  1:getlnnsize(bcmodes))
    return i
end

# Same as getidxapprox(), except that the agreement needs to be ~machine precision.
getidx(bcmodes::ClnnBinnedModes{T,true}, ℓ, k1in, k2in) where {T} = begin
    k1, k2 = minmax(k1in, k2in)
    return getidx(bcmodes.LKK, ℓ, k1, k2)
end

getidx(bcmodes::ClnnBinnedModes, ℓ, k1in, k2in) = begin
    lnnsize = size(LKK,2)
    i = findfirst(i -> begin
                      L = LKK[1,i]
                      (L ≈ ℓ) || return false
                      k1 = LKK[2,i]
                      (k1 ≈ k1in) || return false
                      k2 = LKK[3,i]
                      (k2 ≈ k2in) || return false
                      return true
                  end,
                  1:lnnsize)
    return i
end



########## calculate binning and debinning matrices

function getidx!(iLNN, imax, iL, iN1, iN2)
    idx = findfirst(1:imax) do i
        all(iLNN[:,i] .== (iL, iN1, iN2))
    end
    if isnothing(idx)
        imax += 1
        iLNN[:,imax] .= iL, iN1, iN2
        idx = imax
    end
    imax = max(idx, imax)
    return idx, imax
end

function bandpower_binning_weights(cmodes::ClnnModes; Δℓ=1, Δn1=1, Δn2=1, select=:all)
    lmax = min(cmodes.amodesA.lmax, cmodes.amodesB.lmax)
    nAmax = cmodes.amodesA.nmax
    nBmax = cmodes.amodesB.nmax
    lnnsize = getlnnsize(cmodes)

    if select == :all
        select = fill(true, lnnsize)
    end

    LNNsizemax = ((lnnsize ÷ Δℓ + 1) ÷ Δn1 + 1) ÷ Δn2 + 1 + lmax + nAmax + nBmax
    w̃ = fill(0.0, LNNsizemax, lnnsize)

    iLNN = fill(0, 3, LNNsizemax)
    LNNsize = 0
    for i=1:lnnsize
        select[i] || continue
        l, n1, n2 = getlnn(cmodes, i)

        # lowest bin shall always contain `Δℓ * Δn1 * Δn2` modes
        iL = (l ÷ Δℓ) + 1
        iN1 = ((n1 - 1) ÷ Δn1) + 1
        iN2 = ((n2 - 1) ÷ Δn2) + 1

        I, LNNsize = getidx!(iLNN, LNNsize, iL, iN1, iN2)

        w̃[I,i] += 1
    end

    w̃ = collect(w̃[1:LNNsize,select])
    @show LNNsize,lnnsize
    @show size(w̃)

    w̃ = w̃ ./ sum(w̃, dims=2)  # normalize

    v = pinv(w̃)

    @assert all(sum(w̃, dims=2) .≈ 1)

    w̃, v = sparse(w̃), sparse(v)
    return w̃, v
end


function bandpower_eigen_weights(cmix; ϱ=1.5)
    @show cmix
    @show issymmetric(cmix)
    # we assume only positive eigenvalues are allowed
    f = eigen(cmix)
    λ = f.values
    @show λ
    @show f.vectors
    n = findlast(@. λ[end] / λ > ϱ)
    n = isnothing(n) ? 1 : n+1
    R = f.vectors[:,n:end]'
    @show λ[n:end]
    @show isposdef(cmix) det(cmix)
    @show isposdef(f.vectors) det(f.vectors)
    @show size(R)
    for i=1:size(f.vectors,2), j=1:size(f.vectors,2)
        @show i,j,dot(f.vectors[:,i], f.vectors[:,j])
    end
    Rinv = inv(f.vectors)[n:end,:]'
    @show R R' Rinv
    @show R*R' R'*R
    @show R*Rinv Rinv*R
    D = inv(diagm(sum(R, dims=2)[:]))
    @show diag(D)
    @show maximum(diag(D)) / minimum(diag(D))
    @show maximum(abs.(diag(D))) / minimum(abs.(diag(D)))
    w = D * R
    Θ = Rinv * inv(D)
    @show w Θ
    @show sum(w, dims=2)
    @show logabsdet(Θ*w)
    @show w*Θ (w*Θ)^10 (w*Θ)^100 (w*Θ)^1000
    @show Θ*w (Θ*w)^10 (Θ*w)^100 (Θ*w)^1000
    @show sum(Θ*w, dims=1)
    @show sum(Θ*w, dims=2)
    @assert R * Rinv ≈ I
    return w, Θ
end


function get_incomplete_bins(w̃; nmodes_per_bin=nothing)
    if isnothing(nmodes_per_bin)
        # Guess normal number of modes per bin
        nbins = Int[]  # histogram of number of modes per bin
        for i=1:size(w̃, 1)
            nmodes = sum(@. w̃[i,:] != 0)
            if nmodes > length(nbins)
                append!(nbins, fill(0, nmodes - length(nbins)))
            end
            nbins[nmodes] += 1
        end
        nmodes_per_bin = length(nbins)
        @show nbins
    end
    @show nmodes_per_bin

    incomplete_bins = Int[]
    for i=1:size(w̃, 1)
        nmodes = sum(@. w̃[i,:] != 0)
        if nmodes != nmodes_per_bin
            push!(incomplete_bins, i)
        end
    end
    return incomplete_bins
end


# find_most_hi_k_convolved_mode(): Here find the mode that is most affected by
# mode mixing from hi-k modes. This is useful to find the maximum k that is
# needed to get unbiased results from the deconvolution step. `bcmixinv` is the
# inverse of the mixing matrix, `lkk` describes the modes, `kmax` is the
# maximum mode we are interested in, and `ktrans` is the current best estimate
# for the largest needed k.
function find_most_hi_k_convolved_mode(bcmixinv, lkk, kmax, ktrans)
    s_hi = @. lkk[2,:] > ktrans
    s_lo = @. lkk[2,:] <= ktrans
    idxmax = 1
    ratiomax = 0.0
    for idx in 1:size(lkk, 2)
        (lkk[2,idx] <= kmax) || continue
        hi = sum(abs.(bcmixinv[idx,s_hi]))
        lo = sum(abs.(bcmixinv[idx,s_lo]))
        ratio = hi / (lo + hi)
        if ratio >= ratiomax
            idxmax = idx
            ratiomax = ratio
        end
    end
    return idxmax, ratiomax
end


function find_klarge(bcmixinv, lkk, kmax; threshold=0.01)
    zerofn(klarge) = begin
        idx, hifraction = find_most_hi_k_convolved_mode(bcmixinv, lkk, kmax, klarge)
        return hifraction - threshold
    end

    karr = sort(lkk[2,:])

    idx = findfirst(k -> zerofn(k) < 0, karr)

    klarge = nextfloat(karr[idx])

    return klarge
end


function get_hi_k_affection(bcmixinv, lkk, klarge)
    s_hi = @. lkk[2,:] > klarge
    s_lo = @. lkk[2,:] <= klarge
    lnnsize = length(lkk[2,:])
    hifracts = fill(NaN, lnnsize)
    for idx in 1:lnnsize
        hi = sum(abs.(bcmixinv[idx,s_hi]))
        lo = sum(abs.(bcmixinv[idx,s_lo]))
        hifracts[idx] = hi / (lo + hi)
    end
    return hifracts
end



end


# vim: set sw=4 et sts=4 :
