#!/usr/bin/env julia


module window_chains

export window_chain

using ..Windows
using WignerSymbols
using ..HealPy
using ..NDIterators
using ..SeparableArrays
using ..WignerChains


function I_nlnl2ll(I_LM_ln_ln, ll, nn)
    I_LM_l_l = fill(im*NaN, size(I_LM_ln_ln,1), length(ll), length(ll))
    for j=1:length(ll), i=1:length(ll)
        I_LM_l_l[:,i,j] .= I_LM_ln_ln[:,i,nn[i],j,nn[j]]
    end
    return I_LM_l_l
end

function window_chain(k, win, wmodes, cmodes)
    I_LM_ln_ln = calc_I_LM_nl_nl(win, wmodes, cmodes)
    lmax = cmodes.amodes.lmax
    nmax = cmodes.amodes.nmax
    Wk_ll_nn = fill(im*NaN, fill(lmax+1,k)..., fill(nmax,k)...)
    ll = NDIterator(0, lmax; N=k)
    while advance(ll)
        nmax_l = @. cmodes.nmax_l[ll+1]
        nn = NDIterator(1, nmax_l)
        while nn
            Wk_ll_nn[ll..., nn...] = window_chain(ll, nn, I_LM_ln_ln)
        end
    end
    return Wk_ll_nn
end
function window_chain(ell, nn, I_LM_ln_ln)
    I_LM_l_l = I_nlnl2ll(I_LM_ln_ln, ell, nn)
    return window_chain(ell, I_LM_l_l)
end
function window_chain(ell, I_LM_l_l)
    T = Complex{Float64}
    #@assert length(ell) == length(nn)
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
    @show Lmin Lmax

    wk = T(0)
    L = NDIterator(Lmin, Lmax)
    while advance(L)
        twoLplus1_w3j000 = (2*L[1] + 1) * wigner3j(T, ell[end], ell[1], L[1], 0, 0, 0)
        for i=2:k
            twoLplus1_w3j000 *= (2*L[i] + 1) * wigner3j(T, ell[i-1], ell[i], L[i], 0, 0, 0)
        end

        # sum over M[i]
        M = NDIterator(-L, L)
        while advance(M)
            w3jk = wigner3j_chain(ell, L, M)
            (w3jk == 0) && continue

            Iprod = get_I(I_LM_l_l, L[1], M[1], ell[end], ell[1])
            for i=2:k
                Iprod *= get_I(I_LM_l_l, L[i], M[i], ell[i-1], ell[i])
            end

            wk += twoLplus1_w3j000 * Iprod * w3jk
        end
    end
    wk *= (4*π)^(-k/2) * prod(@. 2 * ell + 1)
    return wk
end


########### calc_I_LM_nl_nl #############3

function calc_Wr_lm(win, LMAX, Wnside)
    nr = size(win,1)
    Wr_lm = fill(NaN*im, nr, getlmsize(LMAX))
    @time for i=1:nr
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
            check_radial_integral_convergence(n, l, N, L, r, Δr, cmodes)
            @. ggϕint = r^2 * gnlr[:,n,l+1] * gnlr[:,N,L+1] * phi
            gg = Δr * sum(ggϕint)
            gnlgNLϕ[n,l+1,N,L+1] = gg
            gnlgNLϕ[N,L+1,n,l+1] = gg
        end
    end
    return gnlgNLϕ
end


function check_radial_integral_convergence(n1, l1, n2, l2, r, Δr, cmodes)
    Nsamp = 8 * (n1 + n2) + l1 + l2
    if Nsamp > length(r)
        @error "Radial integral may not converge" Nsamp length(r) n1 l1 n2 l2 cmodes.amodes.knl[n,l+1] cmodes.amodes.knl[N,L+1] cmodes.amodes.rmin cmodes.amodes.rmax extrema(r) Δr
        throw(ErrorException("Nsamp > nr"))
    end
end


function calc_I_LM_nl_nl(win, wmodes, cmodes)
    #cmodes = bcmodes.cmodes
    amodes = cmodes.amodes

    # Wr_lm
    LMAX = 2 * amodes.lmax
    Wr_lm = calc_Wr_lm(win, LMAX, Wnside)
    LMcache = [hp.Alm.getidx.(LMAX, L, 0:L) .+ 1 for L=0:LMAX]

    # gnlr
    r, Δr = window_r(wmodes)
    gnl = amodes.basisfunctions
    gnlr = fill(NaN, length(r), size(gnl.knl)...)
    @time for l=0:amodes.lmax, n=1:amodes.nmax_l[l+1]
        @. gnlr[:,n,l+1] = gnl(n,l,r)
    end

    # I_LM_ln_ln
    I_LM_ln_ln = fill(im*NaN, length(LMcache), lmax+1, nmax, lmax+1, nmax)
    for l2=0:lmax, n2=1:nmax_l[l2], l1=l2:lmax, n1=n2:nmax_l[l1], iLM=1:length(LMcache)
        LM = LMcache[iLM]
        !isnan(I_LM_ln_ln[LM,l1+1,n1,l2+1,n2]) && continue
        check_radial_integral_convergence(n1, l1, n2, l2, r, Δr, cmodes)
        I = Δr * sum(@. r^2 * gnlr[:,n1,l1+1] * gnlr[:,n2,l2+1] * Wr_lm[:,LM])
        I_LM_ln_ln[LM,l1+1,n1,l2+1,n2] = I
        I_LM_ln_ln[LM,l2+1,n2,l1+1,n1] = I
    end

    return I_LM_ln_ln
end



end


# vim: set sw=4 et sts=4 :
