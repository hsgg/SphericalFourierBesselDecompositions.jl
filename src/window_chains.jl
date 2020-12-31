#!/usr/bin/env julia

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

using ..Windows
using WignerSymbols
using WignerFamilies
using ..HealPy
using ..NDIterators
using ..SeparableArrays
using ..WignerChains
using ..Modes


######### WindowChainsCache ##################################################

abstract type WindowChainsCache{T<:Real} end

struct WindowChainsCacheWignerChain{T} <: WindowChainsCache{T}
    I_LM_ln_ln::Array{Complex{T},5}
    LMcache::Array{Array{Int,1},1}
end

struct WindowChainsCacheFullWmix{T} <: WindowChainsCache{T}
    wmix::Array{Complex{T},2}
    wmix_negm::Array{Complex{T},2}
    amodes::AnlmModes
end

struct WindowChainsCacheFullWmixOntheflyWmix{T} <: WindowChainsCache{T}
    I_LM_ln_ln::Array{Complex{T},5}
    LMcache::Array{Array{Int,1},1}
end

struct WindowChainsCacheSeparableWmix{T} <: WindowChainsCache{T}
    Wlmlm::Array{Complex{T},2}
    Wlmlm_negm::Array{Complex{T},2}
    LMcache::Array{Array{Int,1},1}
    Ilnln::Array{T,4}
    amodes::AnlmModes
end

struct WindowChainsCacheSeparableWmixOntheflyWlmlm{T} <: WindowChainsCache{T}
    Wlm::Array{Complex{T},1}
    LMcache::Array{Array{Int,1},1}
    Ilnln::Array{T,4}
    amodes::AnlmModes
end


# Using wigner chain
function WindowChainsCacheWignerChain(win, wmodes, amodes)
    lmax = amodes.lmax
    nmax = amodes.nmax
    nmax_l = amodes.nmax_l
    check_nsamp(amodes, wmodes)

    # Wr_lm
    LMAX = 2 * amodes.lmax
    Wr_lm = calc_Wr_lm(win, LMAX, amodes.nside)
    LMcache = [hp.Alm.getidx.(LMAX, L, 0:L) .+ 1 for L=0:LMAX]
    lmsize = hp.Alm.getsize(LMAX)
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
    wmix′ = collect(wmix′')  # put negative m into the second index
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
        w = get_wmix(cache.wmix, cache.wmix_negm, nl2, m[end], nl1, m[1])
        for i=2:length(ell)
            nl2 = getidx(cache.amodes, n2[i-1], ell[i-1], 0)
            nl1 = getidx(cache.amodes, n1[i], ell[i], 0)
            w *= get_wmix(cache.wmix, cache.wmix_negm, nl2, m[i-1], nl1, m[i])
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
        W = hp.ud_grade(win[i,:], Wnside)
        Wr_lm[i,:] .= hp.map2alm(W, lmax=LMAX)
    end
    return Wr_lm
end

function calc_Wr_lm(win::SeparableArray, LMAX, Wnside)  # specialize
    mask = hp.ud_grade(win.mask, Wnside)
    wlm = hp.map2alm(mask, lmax=LMAX)
    return SeparableArray(win.phi, wlm, name1=:phi, name2=:wlm)
end


function calc_Wlm(mask, lmax, nside)
    LMAX = 2 * lmax
    mask = hp.ud_grade(mask, nside)
    Wlm = hp.map2alm(mask, lmax=LMAX)
    LMcache = [hp.Alm.getidx.(LMAX, L, 0:L) .+ 1 for L=0:LMAX]
    return Wlm, LMcache
end


function calc_Wlmlm(mask, lmax, nside)
    wlm, LMcache = calc_Wlm(mask, lmax, nside)
    LMAX = 2 * lmax
    lmsize = hp.Alm.getsize(LMAX)
    wlmlm = fill(NaN*im, lmsize, lmsize)
    wlmlm_negm = fill(NaN*im, lmsize, lmsize)
    for l=0:lmax, m=0:l, L=0:lmax, M=0:L
        i = LMcache[l+1][abs(m)+1]
        j = LMcache[L+1][abs(M)+1]
        wlmlm[i,j] = window_wmix(l, m, L, M, wlm, LMcache)
        wlmlm_negm[i,j] = window_wmix(l, m, L, -M, wlm, LMcache)
    end
    return wlmlm, wlmlm_negm, LMcache
end


function get_wlmlm(cache::WindowChainsCacheSeparableWmix, l::Int, m::Int, L::Int, M::Int)
    i = cache.LMcache[l+1][abs(m)+1]
    j = cache.LMcache[L+1][abs(M)+1]
    if m >= 0
        if M >= 0
            return cache.Wlmlm[i,j]
        else
            return cache.Wlmlm_negm[i,j]
        end
    else # m < 0
        w = if M >= 0
            cache.Wlmlm_negm[i,j]
        else
            cache.Wlmlm[i,j]
        end
        return conj(isodd(m+M) ? -w : w)
    end
end


function calc_Ilnln(phi, wmodes, amodes)
    check_nsamp(amodes, wmodes)

    # gnlr
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
    for l2=0:lmax, n2=1:nmax_l[l2+1], l1=0:lmax, n1=1:nmax_l[l1+1]
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

            LM = LMcache[L[1]+1][abs(M[1])+1]
            I = I_LM_l_l[LM, k, 1]
            (M[1] < 0) && (I = conj(I))
            #@show L[1],M[1],I
            Iprod = I
            for i=2:k
                LM = LMcache[L[i]+1][abs(M[i])+1]
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
function window_wmix(l, m, L, M, Wlm, LMcache)
    T = Float64
    wigs000 = wigner3j_f(l, L, 0, 0)
    #@show l,L,-m,M
    wigs = wigner3j_f(l, L, -m, M)
    #@show eachindex(wigs000) eachindex(wigs)
    #@show length(wigs000) length(wigs)
    @assert length(wigs000) >= length(wigs)
    M1 = m - M
    w = T(0)*im
    for L1 in eachindex(wigs)
        L1M1 = LMcache[L1+1][abs(M1)+1]
        Iterm = Wlm[L1M1]
        if M1 < 0
            Iterm = (-1)^M1 * conj(Iterm)
        end
        w += √T(2*L1 + 1) * wigs000[L1] * wigs[L1] * Iterm
    end
    return (-1)^m * √((2*l + 1)*(2*L + 1) / (4*T(π))) * w
end


function get_wmix(w, w′, nl, m, NL, M)
    (m >= 0 && M >= 0) && return w[nl+m, NL+M]
    (m >= 0 && M < 0)  && return w′[nl+m, NL-M]
    (m < 0  && M >= 0) && return (-1)^(m+M) * conj(w′[nl-m, NL+M])
    return (-1)^(m-M) * conj(w[nl-m, NL-M])
end




end


# vim: set sw=4 et sts=4 :
