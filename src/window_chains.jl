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


@doc raw"""
    WindowChains

This module exposes several ways to calculate a chain of window functions. All
require a cache to be created that speeds up subsequent calls. The type of
cache determines which method is used.

To create the cache, simply call one of
```
julia> cache = SFB.WindowChains.WindowChainsCacheWignerChain(win, wmodes, amodes)
julia> cache = SFB.WindowChains.WindowChainsCacheFullWmix(win, wmodes, amodes)
julia> cache = SFB.WindowChains.WindowChainsCacheFullWmixOntheflyWmix(win, wmodes, amodes)
julia> cache = SFB.WindowChains.WindowChainsCacheSeparableWmix(win, wmodes, amodes)
julia> cache = SFB.WindowChains.WindowChainsCacheSeparableWmixOntheflyWlmlm(win, wmodes, amodes)
julia> cache = SFB.WindowChainsCache(win, wmodes, amodes)
```
The last one will automatically select the typically fastest algorithm.

Then to calculate an element of a chain of windows, use `window_chain(ell, n1,
n2, cache)`. For example, for a chain of four windows,
```
julia> ell = [0, 1, 2, 3]
julia> n1 = [1, 2, 3, 4]
julia> n2 = [5, 6, 7, 8]
julia> Wk = SFB.window_chain(ell, n1, n2, cache)
```
"""
module WindowChains

export WindowChainsCache
export window_chain
export calc_wmix_all

using ..Windows
using WignerSymbols
using WignerFamilies
using Healpix
using ..HealpixHelpers
using ..NDIterators
using ..SeparableArrays
using ..WignerChains
using ..Modes
using ..LMcalcStructs
using ProgressMeter
using ..MyBroadcast




######### WindowChainsCache ##################################################

abstract type WindowChainsCache{T<:Real} end

struct WindowChainsCacheWignerChain{T,LMC} <: WindowChainsCache{T}
    I_LM_ln_ln::Array{Complex{T},5}
    LMcache::LMC
end

struct WindowChainsCacheFullWmix{T} <: WindowChainsCache{T}
    wmix::Array{Complex{T},2}
    wmix_negm::Array{Complex{T},2}
    amodes::AnlmModes
end

struct WindowChainsCacheFullWmixOntheflyWmix{T,LMC} <: WindowChainsCache{T}
    I_LM_ln_ln::Array{Complex{T},5}
    LMcache::LMC
end

struct WindowChainsCacheSeparableWmix{T,LMC} <: WindowChainsCache{T}
    Wlmlm::Array{Complex{T},2}
    Wlmlm_negm::Array{Complex{T},2}
    LMcache::LMC
    Ilnln::Array{T,4}
    amodes::AnlmModes
end

struct WindowChainsCacheSeparableWmixOntheflyWlmlm{T,LMC} <: WindowChainsCache{T}
    Wlm::Array{Complex{T},1}
    LMcache::LMC
    Ilnln::Array{T,4}
    amodes::AnlmModes
end


# Using wigner chain
function WindowChainsCacheWignerChain(win, wmodes, amodes)
    @warn "Using poorly tested Window Chain implementation. Will give incorrect results."
    lmax = amodes.lmax
    nmax = amodes.nmax
    nmax_l = amodes.nmax_l
    check_nsamp(amodes, wmodes)

    # Wr_lm
    LMAX = 2 * amodes.lmax
    Wr_lm = calc_Wr_lm(win, LMAX, amodes.nside)
    alm = Alm(LMAX, LMAX)
    LMcache = LMcalcStruct(LMAX)
    lmsize = numberOfAlms(LMAX)
    #@show LMcache
    #@show typeof(LMcache)

    # gnlr
    r, Δr = window_r(wmodes)
    gnl = amodes.basisfunctions
    gnlr = fill(NaN, length(r), size(gnl.knl)...)
    for l=0:amodes.lmax, n=1:amodes.nmax_l[l+1]
        @. gnlr[:,n,l+1] = gnl(n,l,r)
    end

    # I_LM_ln_ln
    I_LM_ln_ln = fill(im*NaN, lmsize, lmax+1, nmax, lmax+1, nmax)
    for l2=0:lmax, n2=1:nmax_l[l2+1], l1=0:lmax, n1=1:nmax_l[l1+1], iLM=1:lmsize
        #@show l2, n2, l1, n1, iLM, lmsize
        !isnan(I_LM_ln_ln[iLM,l1+1,n1,l2+1,n2]) && continue
        I = Δr * sum(@. r^2 * gnlr[:,n1,l1+1] * gnlr[:,n2,l2+1] * Wr_lm[:,iLM])
        I_LM_ln_ln[iLM,l1+1,n1,l2+1,n2] = I
        I_LM_ln_ln[iLM,l2+1,n2,l1+1,n1] = I
    end

    return WindowChainsCacheWignerChain(I_LM_ln_ln, LMcache)
end


# full wmix
function WindowChainsCacheFullWmix(win, wmodes, amodes)
    wmix = calc_wmix(win, wmodes, amodes)
    wmix′ = calc_wmix(win, wmodes, amodes, neg_m=true)
    return WindowChainsCacheFullWmix(wmix, wmix′, amodes)
end


# full wmix with on-the-fly wmix
function WindowChainsCacheFullWmixOntheflyWmix(win, wmodes, amodes)
    cache = WindowChainsCacheWignerChain(win, wmodes, amodes)
    return WindowChainsCacheFullWmixOntheflyWmix(cache.I_LM_ln_ln, cache.LMcache)
end


# separable wmix
function WindowChainsCacheSeparableWmix(win, wmodes, amodes)
    Wlmlms = calc_Wlmlm(win.mask, amodes.lmax, amodes.nside)
    Ilnln = calc_Ilnln(win.phi, wmodes, amodes)
    return WindowChainsCacheSeparableWmix(Wlmlms..., Ilnln, amodes)
end


# separable wmix with on-the-fly Wlmlm
function WindowChainsCacheSeparableWmixOntheflyWlmlm(win, wmodes, amodes)
    Wlm, LMcache = calc_Wlm(win.mask, amodes.lmax, amodes.nside)
    Ilnln = calc_Ilnln(win.phi, wmodes, amodes)
    return WindowChainsCacheSeparableWmixOntheflyWlmlm(Wlm, LMcache, Ilnln, amodes)
end


# choose best
WindowChainsCache(win, wmodes, amodes) = WindowChainsCacheFullWmix(win, wmodes, amodes)
WindowChainsCache(win::SeparableArray, wmodes, amodes) = WindowChainsCacheSeparableWmix(win, wmodes, amodes)


############ window_chains() #################################################

# wigner chains
function window_chain(ell, n1, n2, cache::WindowChainsCacheWignerChain{T}) where {T}
    I_LM_l_l = NeqLView(cache.I_LM_ln_ln, ell, n2, n1)
    return window_chain_wigner_chain(ell, I_LM_l_l, cache.LMcache)
end


# full wmix
function window_chain(ell, n1, n2, cache::WindowChainsCacheFullWmix{T}) where {T}
    Wk = T(0)
    m = NDIterator(-ell, ell)
    while advance(m)
        nl2 = getidx(cache.amodes, n2[end], ell[end], 0)
        nl1 = getidx(cache.amodes, n1[1], ell[1], 0)
        w = Windows.get_wmix(cache.wmix, cache.wmix_negm, nl2, m[end], nl1, m[1])
        for i=2:length(ell)
            nl2 = getidx(cache.amodes, n2[i-1], ell[i-1], 0)
            nl1 = getidx(cache.amodes, n1[i], ell[i], 0)
            w *= Windows.get_wmix(cache.wmix, cache.wmix_negm, nl2, m[i-1], nl1, m[i])
        end
        Wk += real(w)
        #@show m,w
    end
    return Wk
end


# full wmix, on-the-fly
function window_chain(ell, n1, n2, cache::WindowChainsCacheFullWmixOntheflyWmix{T}) where {T}
    I_LM_l_l = NeqLView(cache.I_LM_ln_ln, ell, n2, n1)
    return window_chain_onthefly_wmix(ell, I_LM_l_l, cache.LMcache)
end


# separable wmix
function window_chain(ell, n1, n2, cache::WindowChainsCacheSeparableWmix{T}) where {T}
    Wk = T(0)
    m = NDIterator(-ell, ell)
    while advance(m)
        w = get_wlmlm(cache, ell[end], m[end], ell[1], m[1])
        for i=2:length(ell)
            w *= get_wlmlm(cache, ell[i-1], m[i-1], ell[i], m[i])
        end
        #@show m,w
        Wk += real(w)
    end
    Ik = calc_kprod(cache.Ilnln, ell .+ 1, n1, n2)
    return Ik * Wk
end


# separable wmix with on-the-fly wmix
function window_chain(ell, n1, n2, cache::WindowChainsCacheSeparableWmixOntheflyWlmlm{T}) where {T}
    Wk = window_chain_onthefly_wmix(ell, cache.Wlm, cache.LMcache)
    Ik = calc_kprod(cache.Ilnln, ell .+ 1, n1, n2)
    return Ik * Wk
end


############ specialty window_chain() ############

# calculate full window chain matrix
function window_chain(k, win, wmodes::ConfigurationSpaceModes, cmodes::ClnnModes; cache=nothing)
    if isnothing(cache)
        cache = WindowChainsCache(win, wmodes, cmodes.amodes)
    end
    lnnsize = getlnnsize(cmodes)
    Wk = fill(NaN, fill(lnnsize,k)...)
    ell = Array{Int}(undef, k)
    nn1 = Array{Int}(undef, k)
    nn2 = Array{Int}(undef, k)
    lnn = NDIterator(1, lnnsize; N=k)
    while advance(lnn)
        for i=1:k
            l, n1, n2 = getlnn(cmodes, lnn[i])
            ell[i] = l
            nn1[i] = n1
            nn2[i] = n2
        end
        Wk[lnn...] = window_chain(ell, nn1, nn2, cache)
    end
    return Wk
end


# symmetry for some of the n1,n2
@doc raw"""
    window_chain(ell, n1, n2, cache, symmetries)

This version adds up several window chains taking into account the
`symmetries`. `symmetries` is an array of pairs of numbers specifying the
symmetries to consider. Each pair specifies the `ℓnn′` index and the type of
symmetry. For example, when `k≥3`, then `symmetries = [1=>0, 2=>1, 3=>2]`
would specify that no symmetries are taken into account for the first `lnn′`
combination, the second symmetry will flip `n` and `n′` and add the result only
when `n ≠ n′`, and the third will add the result regardless whether the `n`
equal or not.
"""
function window_chain(ell, n1, n2, cache, symmetries)
    if isempty(symmetries)
        #@show ell n1 n2
        return window_chain(ell, n1, n2, cache)
    end

    Wk = window_chain(ell, n1, n2, cache, symmetries[2:end])

    i, sym = symmetries[1]

    if sym >= 1
        if n1[i] == n2[i]
            if sym == 2
                Wk *= 2
            end
        else
            n1[i], n2[i] = n2[i], n1[i]
            Wk += window_chain(ell, n1, n2, cache, symmetries[2:end])
            n1[i], n2[i] = n2[i], n1[i]
        end
    end

    return Wk
end


############# helpers for WindowChainsCache ##################################

function calc_Wr_lm(win, LMAX, Wnside)
    nr = size(win,1)
    Wr_lm = fill(NaN*im, nr, getlmsize(LMAX))
    for i=1:nr
        W = udgrade(win[i,:], Wnside)
        Wr_lm[i,:] .= mymap2alm(W, lmax=LMAX).alm
    end
    return Wr_lm
end

function calc_Wr_lm(win::SeparableArray, LMAX, Wnside)  # specialize
    mask = udgrade(win.mask, Wnside)
    wlm = mymap2alm(mask, lmax=LMAX).alm
    return SeparableArray(win.phi, wlm, name1=:phi, name2=:wlm)
end


function calc_Wlm(mask, lmax, nside)
    LMAX = 2 * lmax
    mask = udgrade(mask, nside)
    Wlm = mymap2alm(mask, lmax=LMAX)
    LMcache = LMcalcStruct(LMAX)
    return Wlm.alm, LMcache
end


function calc_Wlmlm(mask, lmax, nside)
    wlm, LMcache = calc_Wlm(mask, lmax, nside)
    LMAX = 2 * lmax
    lmsize = numberOfAlms(LMAX)
    wlmlm = fill(NaN*im, lmsize, lmsize)
    wlmlm_negm = fill(NaN*im, lmsize, lmsize)
    #@showprogress 1 "Wlmlm: " for l=0:lmax, m=0:l, L=0:lmax, M=0:L
    @showprogress 1 "Wlmlm: " for M=0:lmax, m=0:lmax, L=M:lmax, l=m:lmax
        i = LMcache[l+1,abs(m)+1]
        j = LMcache[L+1,abs(M)+1]
        wlmlm[i,j] = window_wmix(l, m, L, M, wlm, LMcache)
        wlmlm_negm[i,j] = window_wmix(l, -m, L, M, wlm, LMcache)
    end
    return wlmlm, wlmlm_negm, LMcache
end


function get_wlmlm(cache::WindowChainsCacheSeparableWmix, l::Int, m::Int, L::Int, M::Int)
    i = cache.LMcache[l+1,abs(m)+1]
    j = cache.LMcache[L+1,abs(M)+1]
    if m >= 0
        if M >= 0
            return cache.Wlmlm[i,j]
        else
            return (-1)^(m+M) * conj(cache.Wlmlm_negm[i,j])
        end
    else # m < 0
        w = if M >= 0
            return cache.Wlmlm_negm[i,j]
        else
            return (-1)^(m+M) * conj(cache.Wlmlm[i,j])
        end
    end
end


function calc_Ilnln(phi, wmodes, amodes)
    check_nsamp(amodes, wmodes)

    r, Δr = window_r(wmodes)
    gnl = amodes.basisfunctions
    gnlr = fill(NaN, length(r), size(gnl.knl)...)
    for l=0:amodes.lmax, n=1:amodes.nmax_l[l+1]
        @. gnlr[:,n,l+1] = gnl(n,l,r)
    end


    # Ilnln
    lmax = amodes.lmax
    nmax = amodes.nmax
    nmax_l = amodes.nmax_l
    Ilnln = fill(NaN, lmax+1, nmax, lmax+1, nmax)
    @showprogress 1 "Ilnln: " for l2=0:lmax, n2=1:nmax_l[l2+1], l1=0:lmax, n1=1:nmax_l[l1+1]
        #@show l2, n2, l1, n1
        !isnan(Ilnln[l1+1,n1,l2+1,n2]) && continue
        I = Δr * sum(@. r^2 * gnlr[:,n1,l1+1] * gnlr[:,n2,l2+1] * phi)
        Ilnln[l1+1,n1,l2+1,n2] = I
        Ilnln[l2+1,n2,l1+1,n1] = I
    end

    return Ilnln
end


########### helpers for window_chain() #######################################

# This struct encodes how to access I_LM_lᵢnᵢ_lⱼnⱼ such that we can access it as I[LM,i,j].
struct NeqLView{T}
    I_LM_ln_ln::Array{T,5}
    ell::Array{Int,1}
    n1::Array{Int,1}
    n2::Array{Int,1}
end
NeqLView(I, ell, n1, n2) = NeqLView(I, convert(Array{Int}, ell), convert(Array{Int}, n1), convert(Array{Int}, n2))

Base.getindex(v::NeqLView, LM::Int, i::Int, j::Int) = v.I_LM_ln_ln[LM, v.ell[i]+1, v.n1[i], v.ell[j]+1, v.n2[j]]
Base.getindex(v::NeqLView, ::Colon, i::Int, j::Int) = v.I_LM_ln_ln[:, v.ell[i]+1, v.n1[i], v.ell[j]+1, v.n2[j]]


function window_chain_wigner_chain(ell, I_LM_l_l, LMcache)
    # This implementation is actually quite slow. Instead, use 'window_chain()' below.
    T = Float64
    k = length(ell)

    # only iterate over triangles
    Lmin = fill(0, k)
    Lmax = fill(0, k)
    Lmin[1] = abs(ell[1] - ell[end])
    Lmax[1] = ell[1] + ell[end]
    for i=2:k
        Lmin[i] = abs(ell[i] - ell[i-1])
        Lmax[i] = ell[i] + ell[i-1]
    end
    #@show Lmin Lmax

    wk = T(0)
    L = NDIterator(Lmin, Lmax)
    while advance(L)
        twoLplus1_w3j000 = √(2*L[1] + 1) * wigner3j(T, ell[end], ell[1], L[1], 0, 0, 0)
        for i=2:k
            twoLplus1_w3j000 *= √(2*L[i] + 1) * wigner3j(T, ell[i-1], ell[i], L[i], 0, 0, 0)
        end

        # sum over M[i]
        M = NDIterator(-L, L)
        while advance(M)
            w3jk = wigner3j_chain(ell, L, M)
	    #@show L,M,w3jk
            (w3jk == 0) && continue

            LM = LMcache[L[1]+1,abs(M[1])+1]
            I = I_LM_l_l[LM, k, 1]
            (M[1] < 0) && (I = conj(I))
            #@show L[1],M[1],I
            Iprod = I
            for i=2:k
                LM = LMcache[L[i]+1,abs(M[i])+1]
                I = I_LM_l_l[LM, i-1, i]
                (M[i] < 0) && (I = conj(I))
                Iprod *= I
                #@show L[i],M[i],I
            end

            dwk = twoLplus1_w3j000 * real(Iprod) * w3jk  # imag must sum out
            wk += dwk
	    #@show L,M,dwk,Iprod
        end
    end
    wk *= (4*π)^(-k/2) * prod(@. 2 * ell + 1)
    return wk
end


function calc_kprod(arr, ell, n1, n2)
    Ik = arr[ell[end],n2[end],ell[1],n1[1]]
    for i=2:length(ell)
        Ik *= arr[ell[i-1],n2[i-1],ell[i],n1[i]]
    end
    return Ik
end



########### helpers for on-the-fly wmix #######################################

# Faster implementation using a direct summation.
function window_chain_onthefly_wmix(ell, I_LM_l_l::NeqLView, LMcache)
    # Note: can probably speed this up a bit by calculating the all Wlmlm for
    # given ell before summing over the m.
    T = Float64
    k = length(ell)
    wk = T(0)
    m = NDIterator(-ell, ell)
    while advance(m)
        w = window_wmix(ell[end], m[end], ell[1], m[1], I_LM_l_l[:,k,1], LMcache)
        for i=2:k
            w *= window_wmix(ell[i-1], m[i-1], ell[i], m[i], I_LM_l_l[:,i-1,i], LMcache)
        end
        wk += real(w)
        #@show m,w
    end
    return wk
end


# Faster implementation using a direct summation.
function window_chain_onthefly_wmix(ell, Wlm, LMcache)
    # Note: can probably speed this up a bit by calculating the all Wlmlm for
    # given ell before summing over the m.
    T = Float64
    k = length(ell)
    wk = T(0)
    m = NDIterator(-ell, ell)
    while advance(m)
        w = window_wmix(ell[end], m[end], ell[1], m[1], Wlm, LMcache)
        for i=2:k
            w *= window_wmix(ell[i-1], m[i-1], ell[i], m[i], Wlm, LMcache)
        end
        wk += real(w)
    end
    return wk
end


# calculate window function given lm,LM.
function window_wmix_wignerfamilies(l, m, L, M, Wlm, LMcache)
    T = Float64
    wigs000 = wigner3j_f(l, L, 0, 0)
    wigs = wigner3j_f(l, L, -m, M)
    #@show eachindex(wigs000) eachindex(wigs)
    #@show length(wigs000) length(wigs)
    #@assert length(wigs000) >= length(wigs)
    #@show wigs000 wigs
    M1 = m - M
    w = T(0)*im
    for L1 in eachindex(wigs)
        L1M1 = LMcache[L1+1,abs(M1)+1]
        Iterm = Wlm[L1M1]
        if M1 < 0
            Iterm = (-1)^M1 * conj(Iterm)
        end
        #@show M1,Iterm
        w += √T(2*L1 + 1) * wigs000[L1] * wigs[L1] * Iterm
    end
    #@show l,L,-m,M,w
    return (-1)^m * √((2*l + 1)*(2*L + 1) / (4*T(π))) * w
end
function window_wmix_wignersymbols(l, m, L, M, Wlm, LMcache)
    T = Float64
    M1 = m - M
    w = T(0)*im
    for L1 in max(abs(L-l),abs(M1)):(L+l)
        L1M1 = LMcache[L1+1,abs(M1)+1]
        Iterm = Wlm[L1M1]
        if M1 < 0
            Iterm = (-1)^M1 * conj(Iterm)
        end
        wig000 = Windows.wigner3j000(l, L, L1)
        w3j = wigner3j(T, l, L, L1, -m, M)
        w += √T(2*L1 + 1) * wig000 * w3j * Iterm
    end
    return (-1)^m * √((2*l + 1)*(2*L + 1) / (4*T(π))) * w
end
window_wmix = window_wmix_wignerfamilies
#window_wmix = window_wmix_wignersymbols



#################################################################
# This section really belongs into windows.jl, but that would create a circular
# dependency. We put it here, so that we can use the specialization to
# separable arrays that was developed here.

function calc_wmix_all(win, wmodes::ConfigurationSpaceModes, amodes::AnlmModes)
    wmix = calc_wmix(win, wmodes, amodes)
    wmix′ = calc_wmix(win, wmodes, amodes, neg_m=true)
    return wmix, wmix′
end

# specialize to SeparableArray
function calc_wmix_all(win::SeparableArray, wmodes::ConfigurationSpaceModes, amodes::AnlmModes)
    cache = WindowChainsCacheSeparableWmix(win, wmodes, amodes)
    wmix, wmix_negm = wkcache2wmix_v2(cache)
    return wmix, wmix_negm
end


function wkcache2wmix(cache::WindowChainsCacheSeparableWmix)
    amodes = cache.amodes
    nlmsize = getnlmsize(amodes)
    wmix = fill(NaN*im, nlmsize, nlmsize)
    wmix_negm = fill(NaN*im, nlmsize, nlmsize)
    @showprogress 1 "wmix:  " for j=1:nlmsize, i=1:nlmsize
        n, l, m = getnlm(amodes, i)
        N, L, M = getnlm(amodes, j)
        wlmlm = get_wlmlm(cache, l, m, L, M)
        wlmlm_negm = get_wlmlm(cache, l, -m, L, M)
        Ilnln = cache.Ilnln[l+1,n,L+1,N]
        wmix[i,j] = wlmlm * Ilnln
        wmix_negm[i,j] = wlmlm_negm * Ilnln
    end
    return wmix, wmix_negm
end


function wkcache2wmix_v2(cache::WindowChainsCacheSeparableWmix)
    amodes = cache.amodes
    nlmsize = getnlmsize(amodes)
    wmix = fill(NaN*im, nlmsize, nlmsize)
    wmix_negm = fill(NaN*im, nlmsize, nlmsize)

    p = Progress(nlmsize^2, desc="wmix sep: ", dt=Windows.progressmeter_update_interval, showspeed=true)
    wmix = mybroadcast(1:nlmsize, (1:nlmsize)') do ii,jj
        out = Array{ComplexF64}(undef, length(ii))
        for idx in 1:length(out)
            i = ii[idx]
            j = jj[idx]
            n, l, m = getnlm(amodes, i)
            N, L, M = getnlm(amodes, j)
            wlmlm = get_wlmlm(cache, l, m, L, M)
            Ilnln = cache.Ilnln[l+1,n,L+1,N]
            out[idx] = wlmlm * Ilnln
        end
        next!(p, step=length(out), showvalues=[(:batchsize, length(out))])
        return out
    end

    p = Progress(nlmsize^2, desc="wmix_negm sep: ", dt=Windows.progressmeter_update_interval, showspeed=true)
    wmix_negm = mybroadcast(1:nlmsize, (1:nlmsize)') do ii,jj
        out = Array{ComplexF64}(undef, length(ii))
        for idx in 1:length(out)
            i = ii[idx]
            j = jj[idx]
            n, l, m = getnlm(amodes, i)
            N, L, M = getnlm(amodes, j)
            wlmlm_negm = get_wlmlm(cache, l, -m, L, M)
            Ilnln = cache.Ilnln[l+1,n,L+1,N]
            out[idx] = wlmlm_negm * Ilnln
        end
        next!(p, step=length(out), showvalues=[(:batchsize, length(out))])
        return out
    end

    return wmix, wmix_negm
end


end


# vim: set sw=4 et sts=4 :
