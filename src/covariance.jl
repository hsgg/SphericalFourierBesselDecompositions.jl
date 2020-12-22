#!/usr/bin/env julia


module Covariance

export calc_covariance_volumescaling,
       calc_covariance_efstathiou,
       calc_covariance_exact_chain

using ..Modes
using ..Windows
using ..SeparableArrays
using ..Theory
using ..WindowChains


function calc_covariance_volumescaling(CBlnn, bcmodes, Veff, Vsfb, Δℓ, Δn)
    lnnsize = getlnnsize(bcmodes)
    covar_volumescaling = fill(0.0, lnnsize, lnnsize)
    @time for j=1:lnnsize, i=1:lnnsize
        L, K, K′ = getlkk(bcmodes, j)
        l, k, k′ = getlkk(bcmodes, i)
        C1 = 0.0
        C2 = 0.0
        C3 = 0.0
        C4 = 0.0
        # Need to ensure that what we calculate exists in 'CBlnn'
        idx = getidxapprox(bcmodes, l, k, K)
        if typeof(idx) <: Integer
            C1 = CBlnn[idx]
        end
        idx = getidxapprox(bcmodes, L, k′, K′)
        if typeof(idx) <: Integer
            C2 = CBlnn[idx]
        end
        idx = getidxapprox(bcmodes, l, k, K′)
        if typeof(idx) <: Integer
            C3 = CBlnn[idx]
        end
        idx = getidxapprox(bcmodes, L, k′, K)
        if typeof(idx) <: Integer
            C4 = CBlnn[idx]
        end
        if i == j
            @show i,j,(l,k,k′),(L,K,K′),C1,C2,C3,C4
        end
        if l == L
            Nmodes = (Veff / Vsfb) * (2*L+1) * Δℓ * Δn^2
            covar_volumescaling[i,j] = (C1*C2 + C3*C4) / Nmodes
        end
    end
    return covar_volumescaling
end


function calc_covariance_efstathiou(CBlnn, win, wmodes, w̃mat, vmat, bcmodes)
    # calculate covariance matrix
    win² = exponentiate(win, 2)
    @time bcmix⁽²⁾ᵃ = power_win_mix(win², w̃mat, vmat, wmodes, bcmodes, div2Lp1=true)
    @time bcmix⁽²⁾ᵇ = power_win_mix(win², w̃mat, vmat, wmodes, bcmodes, div2Lp1=true, interchange_NN′=true)
    #plot_cmix(bcmix⁽²⁾ᵃ, "\$M^{(2)a}\$", id*"_bcmix2a")
    #plot_cmix(bcmix⁽²⁾ᵇ, "\$M^{(2)b}\$", id*"_bcmix2b")

    lnnsize = getlnnsize(bcmodes)
    CBlnN_CBlnN_bcmix⁽²⁾ = fill(0.0, lnnsize, lnnsize)
    @time for j=1:lnnsize, i=1:lnnsize
        L, K, K′ = getlkk(bcmodes, j)
        l, k, k′ = getlkk(bcmodes, i)
        C1 = 0.0
        C2 = 0.0
        C3 = 0.0
        C4 = 0.0
        # Need to ensure that what we calculate exists in 'CBlnn'
        idx = getidxapprox(bcmodes, l, k, K)
        if typeof(idx) <: Integer
            C1 = CBlnn[idx]
        end
        idx = getidxapprox(bcmodes, L, k′, K′)
        if typeof(idx) <: Integer
            C2 = CBlnn[idx]
        end
        idx = getidxapprox(bcmodes, l, k, K′)
        if typeof(idx) <: Integer
            C3 = CBlnn[idx]
        end
        idx = getidxapprox(bcmodes, L, k′, K)
        if typeof(idx) <: Integer
            C4 = CBlnn[idx]
        end
        if i == j
            @show i,j,(l,k,k′),(L,K,K′),C1,C2,C3,C4
        end
        CBlnN_CBlnN_bcmix⁽²⁾[i,j] = C1*C2*bcmix⁽²⁾ᵃ[i,j] + C3*C4*bcmix⁽²⁾ᵇ[i,j]
    end
    return CBlnN_CBlnN_bcmix⁽²⁾
end


function calc_covariance_exact_chain(CNlnn, win, wmodes, cmodes; Δℓ=1, Δn=1)
    T = Float64
    amodes = cmodes.amodes
    wccache = WindowChainsCache(win, wmodes, cmodes.amodes)
    lnnsize = getlnnsize(cmodes)
    A1 = fill(T(0), lnnsize, lnnsize)
    for j=1:lnnsize, i=j:lnnsize
        l, n, n′ = getlnn(cmodes, i)
        L, N, N′ = getlnn(cmodes, j)
        #L + l > 2 && continue
        if abs(L-l) > Δℓ || abs(n-N) > Δn || abs(n′-N′) > Δn
            continue
        end
        @show i,(l,n,n′),j,(L,N,N′)
        ell = [0, L, 0, l]
        enn = [0, N, 0, n′]
        enn′ = [0, N′, 0, n]
        A = T(0)
        for k=1:lnnsize
            l1, n1, n2 = getlnn(cmodes, k)
            ell[1] = l1
            enn[1] = n1
            enn′[1] = n2
            W4 = fill(NaN, lnnsize)
            for m=1:lnnsize
                l3, n4, n3 = getlnn(cmodes, m)
                ell[3] = l3
                enn[3] = n4
                enn′[3] = n3
                W4[m] = window_chain(ell, enn, enn′, wccache)
                enn[2], enn′[2] = enn′[2], enn[2]
                W4[m] += window_chain(ell, enn, enn′, wccache)
            end
            A += CNlnn[k] * (CNlnn' * W4)
        end
        A /= (2*l+1) * (2*L+1)
        A1[i,j] = A1[j,i] = A
    end
    return A1
end



end


# vim: set sw=4 et sts=4 :
