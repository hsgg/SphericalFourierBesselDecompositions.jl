#!/usr/bin/env julia


module WindowChains

export window_chain

using ..Windows
using WignerSymbols
using ..HealPy
using ..NDIterators
using ..SeparableArrays
using ..WignerChains
using ..Modes


# This struct encodes how to access I_LM_lᵢnᵢ_lⱼnⱼ such that we can access it as I[LM,i,j].
struct NeqLView{T}
    I_LM_ln_ln::Array{T,5}
    ell::Array{Int,1}
    n1::Array{Int,1}
    n2::Array{Int,1}
end
NeqLView(I, ell, n1, n2) = NeqLView(I, convert(Array{Int}, ell), convert(Array{Int}, n1), convert(Array{Int}, n2))

Base.getindex(v::NeqLView, LM::Int, i::Int, j::Int) = begin
    v.I_LM_ln_ln[LM, v.ell[i]+1, v.n1[i], v.ell[j]+1, v.n2[j]]
end



function window_chain(k, win, wmodes::ConfigurationSpaceModes, cmodes::ClnnModes)
    amodes = cmodes.amodes
    I_LM_ln_ln, LMcache = calc_I_LM_nl_nl(win, wmodes, amodes)
    lmax = amodes.lmax
    nmax = amodes.nmax
    lnnsize = getlnnsize(cmodes)
    Wk_lnni = fill(NaN, fill(lnnsize,k)...)
    lnn = NDIterator(1, lnnsize; N=k)
    while advance(lnn)
        ll = Array{Int}(undef, k)
        nn1 = Array{Int}(undef, k)
        nn2 = Array{Int}(undef, k)
        for i=1:k
            l, n1, n2 = getlnn(cmodes, lnn[i])
            ll[i] = l
            nn1[i] = n1
            nn2[i] = n2
        end
        Wk_lnni[lnn...] = window_chain(ll, nn1, nn2, I_LM_ln_ln, LMcache)
    end
    return Wk_lnni
end
function window_chain(ell, n1, n2, I_LM_ln_ln, LMcache)
    I_LM_l_l = NeqLView(I_LM_ln_ln, ell, n2, n1)
    return window_chain(ell, I_LM_l_l, LMcache)
end
function window_chain(ell, I_LM_l_l, LMcache)
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


########### calc_I_LM_nl_nl #############3

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


# calculate radial mixers
function calc_radial_mixing(lmax, nmax_l, gnlr, phi, r, Δr)
    nmax = maximum(nmax_l)
    gnlgNLϕ = fill(NaN, nmax, lmax+1, nmax, lmax+1)
    ggϕint = fill(NaN, length(phi))
    for L=0:lmax, N=1:nmax_l[L+1]
        for l=0:lmax, n=1:nmax_l[l+1]
            !isnan(gnlgNLϕ[n,l+1,N,L+1]) && continue
            # sanity check
            check_radial_integral_convergence(n, l, N, L, r, Δr, amodes)
            @. ggϕint = r^2 * gnlr[:,n,l+1] * gnlr[:,N,L+1] * phi
            gg = Δr * sum(ggϕint)
            gnlgNLϕ[n,l+1,N,L+1] = gg
            gnlgNLϕ[N,L+1,n,l+1] = gg
        end
    end
    return gnlgNLϕ
end


function check_radial_integral_convergence(n1, l1, n2, l2, r, Δr, amodes)
    Nsamp = 8 * (n1 + n2) + l1 + l2
    if Nsamp > length(r)
        @error "Radial integral may not converge" Nsamp length(r) n1 l1 n2 l2 amodes.knl[n,l+1] amodes.knl[N,L+1] amodes.rmin amodes.rmax extrema(r) Δr
        throw(ErrorException("Nsamp > nr"))
    end
end


function calc_I_LM_nl_nl(win, wmodes, amodes)
    lmax = amodes.lmax
    nmax = amodes.nmax
    nmax_l = amodes.nmax_l

    # Wr_lm
    LMAX = 2 * amodes.lmax
    Wr_lm = calc_Wr_lm(win, LMAX, wmodes.nside)
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
        check_radial_integral_convergence(n1, l1, n2, l2, r, Δr, amodes)
        I = Δr * sum(@. r^2 * gnlr[:,n1,l1+1] * gnlr[:,n2,l2+1] * Wr_lm[:,iLM])
        I_LM_ln_ln[iLM,l1+1,n1,l2+1,n2] = I
        I_LM_ln_ln[iLM,l2+1,n2,l1+1,n1] = I
    end

    return I_LM_ln_ln, LMcache
end



end


# vim: set sw=4 et sts=4 :
