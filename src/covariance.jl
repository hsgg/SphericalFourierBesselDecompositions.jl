#!/usr/bin/env julia


module Covariance

export calc_covariance_exact, calc_covariance_volumescaling, calc_covariance_efstathiou, calc_covariance_exact2

using ..Modes
using ..Windows
using ..SeparableArrays
using ..Theory
using ..WindowChains


function get_wmix(w, w′, nl, m, NL, M)
    (m >= 0 && M >= 0) && return w[nl+m, NL+M]
    (m >= 0 && M < 0)  && return w′[nl+m, NL-M]
    (m < 0  && M >= 0) && return (-1)^(m+M) * conj(w′[nl-m, NL+M])
    return (-1)^(m-M) * conj(w[nl-m, NL-M])
end


# used to be calc_covariance_exact()
function calc_covariance_exact_direct(CNlnn, wmix, wmix′, cmodes)
    amodes = cmodes.amodes
    lnnsize = getlnnsize(cmodes)
    VWlnnLNN = fill(0.0, lnnsize, lnnsize)
    for j=1:lnnsize, i=j:lnnsize
        l, n, n′ = getlnn(cmodes, i)
        L, N, N′ = getlnn(cmodes, j)
        @show (l,n,n′),(L,N,N′)
        if abs(L-l) > 1 || abs(n-N) > 1 || abs(n′-N′) > 1
            continue
        end
        V = 0.0
        for i334=1:lnnsize, i112=1:lnnsize
            l1, n1, n2 = getlnn(cmodes, i112)
            l3, n3, n4 = getlnn(cmodes, i334)
            nl = getidx(amodes, n, l, 0)
            n1l1 = getidx(amodes, n1, l1, 0)
            n2l1 = getidx(amodes, n2, l1, 0)
            NL = getidx(amodes, N, L, 0)
            N′L = getidx(amodes, N′, L, 0)
            n4l3 = getidx(amodes, n4, l3, 0)
            n3l3 = getidx(amodes, n3, l3, 0)
            n′l = getidx(amodes, n′, l, 0)
            wwww = 0.0
            for m1=-l1:l1, M=-L:L, m3=-l3:l3, m=-l:l
                wwww += real(get_wmix(wmix, wmix′, nl, m, n1l1, m1)
                             * (conj(get_wmix(wmix, wmix′, n2l1, m1, NL, M))
                                * conj(get_wmix(wmix, wmix′, N′L, M, n4l3, m3))
                                + conj(get_wmix(wmix, wmix′, n2l1, m1, N′L, M))
                                * conj(get_wmix(wmix, wmix′, NL, M, n4l3, m3)))
                             * get_wmix(wmix, wmix′, n3l3, m3, nl, m))
            end
            V += CNlnn[i112] * CNlnn[i334] * wwww
        end
        VWlnnLNN[i,j] = V / (2*l+1) / (2*L+1)
        VWlnnLNN[j,i] = V / (2*l+1) / (2*L+1)
    end
    return VWlnnLNN
end


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


function calc_covariance_exact2(CNlnn, wmix, cmodes, Veff)
    amodes = cmodes.amodes
    CNAnlmNLM = calc_CNAnlmNLM(CNlnn, wmix, cmodes, Veff)
    #CNAnlmNLM_negm = calc_CNAnlmNLM(CNlnn, wmix, cmodes, Veff; neg_m=true)
    CNAnlmNLM_negm = CNAnlmNLM
    CNWAnlmNLM = wmix * CNAnlmNLM * wmix'
    lnnsize = getlnnsize(cmodes)
    VWAlnnLNN = fill(NaN, lnnsize, lnnsize)
    for j=1:lnnsize, i=j:lnnsize
        l, n, n′ = getlnn(cmodes, i)
        L, N, N′ = getlnn(cmodes, j)
        @show (l,n,n′),(L,N,N′)
        #if abs(L-l) > 1 || abs(n-N) > 1 || abs(n′-N′) > 1
        #    continue
        #end
        V = 0.0
        nl = getidx(amodes, n, l, 0)
        n′l = getidx(amodes, n′, l, 0)
        NL = getidx(amodes, N, L, 0)
        N′L = getidx(amodes, N′, L, 0)
        for m=0:l, M=0:L
            # Note: this part needs improvement for sign(m)≠sign(M).
            V += real(CNWAnlmNLM[nl+m,NL+M] * conj(CNWAnlmNLM[n′l+m,N′L+M])
                      + CNWAnlmNLM[nl+m,N′L+M] * conj(CNWAnlmNLM[n′l+m,NL+M]))
            if m > 0
                V += (-1)^m * real(CNWAnlmNLM[nl+m,NL+M] * CNWAnlmNLM[n′l+m,N′L+M]
                          + CNWAnlmNLM[nl+m,N′L+M] * CNWAnlmNLM[n′l+m,NL+M])
            end
            if M > 0
                V += (-1)^M * real(CNWAnlmNLM[nl+m,NL+M] * CNWAnlmNLM[n′l+m,N′L+M]
                          + CNWAnlmNLM[nl+m,N′L+M] * CNWAnlmNLM[n′l+m,NL+M])
            end
            if m > 0 && M > 0
                V += (-1)^(m+M) * real(CNWAnlmNLM[nl+m,NL+M] * conj(CNWAnlmNLM[n′l+m,N′L+M])
                          + CNWAnlmNLM[nl+m,N′L+M] * conj(CNWAnlmNLM[n′l+m,NL+M]))
            end
        end
        VWAlnnLNN[i,j] = V / (2*l+1) / (2*L+1)
        VWAlnnLNN[j,i] = VWAlnnLNN[i,j]
    end
    return VWAlnnLNN
end


function calc_covariance_exact_chain(CNlnn, win, wmodes, cmodes)
    amodes = cmodes.amodes
    I_LM_ln_ln, LMcache = WindowChains.calc_I_LM_nl_nl(win, wmodes, amodes)
    lnnsize = getlnnsize(cmodes)
    A1 = fill(0.0, lnnsize, lnnsize)
    for j=1:lnnsize, i=j:lnnsize
        l, n, n′ = getlnn(cmodes, i)
        L, N, N′ = getlnn(cmodes, j)
        ell = [0, L, 0, l]
        enn = [0, N, 0, n′]
        enn′ = [0, N′, 0, n]
        A = 0.0
        for k=1:lnnsize
            l1, n1, n2 = getlm(cmodes, k)
            ell[1] = l1
            enn[1] = n1
            enn′[1] = n2
            W4 = fill(NaN, lnnsize)
            for m=1:lnnsize
                l3, n4, n3 = getlm(cmodes, m)
                ell[3] = l3
                enn[3] = n4
                enn′[3] = n3
                W4[m] = window_chain(ell, enn, enn′, I_LM_ln_ln, LMcache)
            end
            A += CNlnn[k] * (CNlnn' * W4)
        end
        A /= (2*l+1) * (2*L+1)
        A1[i,j] = A[j,i] = A
    end
    return A1
end


end


# vim: set sw=4 et sts=4 :
