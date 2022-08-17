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

import Base.length, Base.iterate


########################## AnlmModes: define ordering of modes in data vector ########

## Note: The maximum is lmax=3*nside-1, which is OK for bandpower limited
## functions. In general, it may be better to use lmax=2*nside. Furthermore,
## our algorithms frequently need to reference quantities up to L=2*lmax.
#estimate_nside(lmax) = 2^max(2, ceil(Int, log2((lmax + 1) / 3)))
estimate_nside(lmax) = 2^max(2, ceil(Int, log2((2*lmax + 1) / 2)))
#estimate_nside(lmax) = 4 * 2^max(2, ceil(Int, log2((lmax + 1) / 3)))
#estimate_nside(lmax) = 32 * 2^max(2, ceil(Int, log2((lmax + 1) / 3)))


@doc raw"""
    AnlmModes(kmax, rmin, rmax; cache=true, nside=nothing)
    AnlmModes(nmax, lmax, rmin, rmax; cache=true, nside=nothing)

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

# for the @__dot syntax:
length(s::AnlmModes) = 1
iterate(s::AnlmModes) = s, nothing
iterate(s::AnlmModes, x) = nothing


function AnlmModes(kmax::Real, rmin::Real, rmax::Real; cache=true, nside=nothing, nmax=typemax(Int64), lmax=typemax(Int64))
    sphbesg = SphericalBesselGnl(kmax, rmin, rmax, cache=cache; nmax, lmax)
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


function AnlmModes(nmax::Int, lmax::Int, rmin::Real, rmax::Real; cache=true, nside=nothing)
    lmax_n = [lmax for n=1:nmax]
    nmax_l = [nmax for l=0:lmax]
    sphbesg = SphericalBesselGnl(nmax, lmax, rmin, rmax, cache=cache)
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

This is where we define which modes are included in the power spectrum, given a
`AnlmModes` struct.

The modes are arranged in the following order. The fastest loop is through 'n̄',
then 'Δn', finally 'ℓ', from small number to larger number. We make the
convention that `n̄` is the smaller of the k-modes, and `Δn >= 0`.

More useful is the labeling by `ℓ, n₁, n₂`. In that case we make the convention
that `n₁ = n̄` and `n₂ = n̄ + Δn`.
"""
struct ClnnModes
    amodes::AnlmModes
    knl::Array{Float64,2}
    Δnmax::Int64
    Δnmax_l::Array{Int64,1}
    Δnmax_n::Array{Int64,1}
    symmetric::Bool
end

# for the @__dot syntax:
length(s::ClnnModes) = 1
iterate(s::ClnnModes) = s, nothing
iterate(s::ClnnModes, x) = nothing


function ClnnModes(amodes::AnlmModes; Δnmax=typemax(Int64), symmetric=true)
    @assert symmetric  # cross-correlations are not yet implemented
    Δnmax = min(Δnmax, amodes.nmax-1)
    knl = amodes.knl
    Δnmax_l = fill(0, amodes.lmax+1)
    Δnmax_n = fill(0, amodes.nmax)
    for l=0:amodes.lmax
        Δnmax_l[l+1] = min(Δnmax, amodes.nmax_l[l+1]-1)
        for Δn=0:Δnmax_l[l+1]
            for n=1:amodes.nmax_l[l+1]-Δn
                Δnmax_n[n] = max(Δnmax_n[n], Δn)
            end
        end
    end
    return ClnnModes(amodes, knl, Δnmax, Δnmax_l, Δnmax_n, symmetric)
end


function getnsize(modes::ClnnModes, l, Δn, n̄max=modes.amodes.nmax_l[l+1]-Δn)
    symfac = (modes.symmetric || Δn==0) ? 1 : 2
    return symfac * n̄max
end


function getnΔnsize(modes::ClnnModes, l, Δnmax=modes.Δnmax_l[l+1])
    n̄center = modes.amodes.nmax_l[l+1]
    n̄offdiag = n̄center * (n̄center - 1) ÷ 2 - (n̄center - Δnmax) * (n̄center - Δnmax - 1) ÷ 2
    symfac = (modes.symmetric) ? 1 : 2
    s2 = n̄center + symfac * n̄offdiag
    if Δnmax < 0
        s2 = 0
    end
    return s2
end


function getlnnsize(modes::ClnnModes, lmax=modes.amodes.lmax)
    s = 0
    for l=0:lmax
        s += getnΔnsize(modes, l)
    end
    return s
end


function getlnn(cmodes::ClnnModes, idx)
    l = 0
    nmodes = getnΔnsize(cmodes, l)
    while idx > nmodes
        idx -= nmodes
        l += 1
        nmodes = getnΔnsize(cmodes, l)
    end
    Δn = 0
    nmodes = getnsize(cmodes, l, Δn)
    while idx > nmodes
        idx -= nmodes
        Δn += 1
        nmodes = getnsize(cmodes, l, Δn)
    end
    n̄ = idx
    n1 = n̄
    n2 = n1 + Δn
    return l, n1, n2
end


function isvalidlnn(cmodes::ClnnModes, l, n1, n2)
    @assert cmodes.symmetric
    Δn = n2 - n1
    n̄ = n1
    return (0 <= l <= cmodes.amodes.lmax &&
            0 <= Δn <= cmodes.Δnmax_l[l+1] &&
            1 <= n̄ <= cmodes.amodes.nmax_l[l+1] - Δn)
end


function isvalidlnn_symmetric(cmodes::ClnnModes, l, n1, n2)
    n1, n2 = minmax(n1, n2)
    return isvalidlnn(cmodes, l, n1, n2)
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


function getidx(cmodes::ClnnModes, l, n1, n2)
    check_isvalidclnn_symmetric(cmodes, l, n1, n2)
    Δn = abs(n1 - n2)
    n̄ = min(n1, n2)
    idx = 0
    idx += getlnnsize(cmodes, l-1)
    idx += getnΔnsize(cmodes, l, Δn-1)
    idx += n̄
    return idx
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
    k1 = cmodes.knl[n1,l+1]
    k2 = cmodes.knl[n2,l+1]
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


function estimate_nr(cmodes::ClnnModes)
    # Note this is a pretty shitty algorithm. Speeding it up should be fairly easy.
    lnnsize = getlnnsize(cmodes)
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

struct ClnnBinnedModes{T}
    cmodes::ClnnModes
    LKK::Array{T,2}  # l=LKK[1,:], k1=LKK[2,:], k2=LKK[3,:]
end

# for the @__dot syntax:
length(s::ClnnBinnedModes) = 1
iterate(s::ClnnBinnedModes) = s, nothing
iterate(s::ClnnBinnedModes, x) = nothing


function ClnnBinnedModes(w̃, v, cmodes::ClnnModes)
    (w̃ != I) && @assert all(sum(w̃, dims=2) .≈ 1)  # ensure w̃ is normalized
    LKK = getlkk(cmodes) * w̃'
    if cmodes.symmetric
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

getidxapprox(bcmodes::ClnnBinnedModes, ℓ, k1in, k2in) = begin
    @assert bcmodes.cmodes.symmetric
    k1 = min(k1in, k2in)
    k2 = max(k1in, k2in)
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
getidx(bcmodes::ClnnBinnedModes, ℓ, k1in, k2in) = begin
    @assert bcmodes.cmodes.symmetric
    return getidx(bcmodes.LKK, ℓ, k1in, k2in)
end
getidx(LKK::AbstractArray{T,2}, ℓ, k1in, k2in) where {T<:Real} = begin
    lnnsize = size(LKK,2)
    k1 = min(k1in, k2in)
    k2 = max(k1in, k2in)
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

function bandpower_binning_weights(cmodes::ClnnModes; Δℓ=1, Δn=1)
    # Note: Should really change this to bin in only Δℓ and Δn. Binning in ΔΔn
    # and Δn̄ is untested.
    ΔΔn, Δn̄ = Δn, Δn
    @show Δℓ, ΔΔn, Δn̄
    lmax = cmodes.amodes.lmax
    Δnmax_l = cmodes.Δnmax_l
    n̄max_l = cmodes.amodes.nmax_l
    nmax = cmodes.amodes.nmax

    LNNsizemax = 0
    for l=0:Δℓ:lmax
        lrange = l:min(l+Δℓ-1,lmax)
        for Δn=0:ΔΔn:maximum(Δnmax_l[lrange.+1]), n̄=1:Δn̄:maximum(n̄max_l[lrange.+1])
            LNNsizemax += 1
        end
    end
    lnnsize = getlnnsize(cmodes)
    @show LNNsizemax lnnsize

    w̃ = fill(0.0, LNNsizemax, lnnsize)
    i = 1
    for llo=0:Δℓ:lmax
        lrange = llo:min(llo+Δℓ-1,lmax)
        @show lrange
        for Δnlo=0:ΔΔn:maximum(Δnmax_l[lrange.+1]), n̄lo=1:Δn̄:maximum(n̄max_l[lrange.+1])
            Δnrange = [Δnlo:min(Δnlo+ΔΔn-1,Δnmax_l[l+1]) for l in lrange]
            n̄range = [n̄lo:min(n̄lo+Δn̄-1,n̄max_l[l+1]) for l in lrange]
            #@show Δnrange n̄range
            num = 0
            for il=1:length(lrange), Δn in Δnrange[il], n̄ in n̄range[il]
                #@error "Mode ignored" i lrange Δnrange n̄range
                l = lrange[il]
                n1 = n̄
                n2 = n̄ + Δn
                isvalidlnn(cmodes, l, n1, n2) || continue
                j = getidx(cmodes, l, n1, n2)
                #@show i,j, l,Δn,n̄
                w̃[i,j] += 1
                num += 1
            end
            if num != 0
                i += 1
            end
        end
    end
    LNNsize = i - 1
    w̃ = w̃[1:LNNsize,:]
    @show LNNsize

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
