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


module Covariance

export calc_covariance_modecounting,
       calc_covariance_efstathiou,
       calc_covariance_exact_chain

using ..Modes
using ..Windows
using ..SeparableArrays
using ..Theory
using ..WindowChains


function calc_covariance_modecounting(CBlnn, bcmodes, fvol, Δℓ, Δn)
    lnnsize = getlnnsize(bcmodes)
    V_mc = fill(0.0, lnnsize, lnnsize)
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
        #(i == j) && @show i,j,(l,k,k′),(L,K,K′),C1,C2,C3,C4
        if l == L
            Nmodes = fvol * (2*L+1) * Δℓ * Δn
            V_mc[i,j] = (C1*C2 + C3*C4) / Nmodes
        end
    end
    return V_mc
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


################ exact calculation ################

function calc_covariance_exact_chain(Clnn, nbar, win, wmodes, cmodes; Δℓ=1, Δn=1)
    if any(@. Clnn != 0) && nbar != 0
        @time VW = calc_covariance_exact_A1(Clnn, win, wmodes, cmodes, Δℓ=Δℓ, Δn=Δn)
        @time VW .+= calc_covariance_exact_A2(Clnn, nbar, win, wmodes, cmodes, Δℓ=Δℓ, Δn=Δn)
        @time VW .+= calc_covariance_exact_A3(nbar, win, wmodes, cmodes, Δℓ=Δℓ, Δn=Δn)
    else
        @time VW = calc_covariance_exact_A3(nbar, win, wmodes, cmodes, Δℓ=Δℓ, Δn=Δn)
    end
    return VW
end


function calc_covariance_exact_A1(Clnn, win, wmodes, cmodes; Δℓ=1, Δn=1)
    T = Float64
    amodes = cmodes.amodes
    wccache = WindowChainsCache(win, wmodes, cmodes.amodes)
    lnnsize = getlnnsize(cmodes)
    symmetries = [1=>1, 2=>2, 3=>1]
    A1 = fill(T(0), lnnsize, lnnsize)
    for j=1:lnnsize, i=j:lnnsize
        l, n, n′ = getlnn(cmodes, i)
        L, N, N′ = getlnn(cmodes, j)
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
                W4[m] = window_chain(ell, enn, enn′, wccache, symmetries)
            end
            A += Clnn[k] * (Clnn' * W4)
        end
        A /= (2*l+1) * (2*L+1)
        A1[i,j] = A1[j,i] = A
    end
    return A1
end


function calc_covariance_exact_A2(Clnn, nbar, win, wmodes, cmodes; Δℓ=1, Δn=1)
    T = Float64
    amodes = cmodes.amodes
    wccache = WindowChainsCache(win, wmodes, cmodes.amodes)
    lnnsize = getlnnsize(cmodes)
    symmetries = [1=>1, 2=>2, 3=>2]
    A2 = fill(T(0), lnnsize, lnnsize)
    for j=1:lnnsize, i=j:lnnsize
        l, n, n′ = getlnn(cmodes, i)
        L, N, N′ = getlnn(cmodes, j)
        if abs(L-l) > Δℓ || abs(n-N) > Δn || abs(n′-N′) > Δn
            continue
        end
        @show i,(l,n,n′),j,(L,N,N′)
        ell = [0, L, l]
        enn = [0, N, n′]
        enn′ = [0, N′, n]
        W3 = fill(NaN, lnnsize)
        for k=1:lnnsize
            l1, n1, n2 = getlnn(cmodes, k)
            ell[1] = l1
            enn[1] = n1
            enn′[1] = n2
            W3[k] = window_chain(ell, enn, enn′, wccache, symmetries)
        end
        A = (Clnn' * W3) / ((2*l+1) * (2*L+1) * nbar)
        A2[i,j] = A2[j,i] = A
    end
    return A2
end


function calc_covariance_exact_A3(nbar, win, wmodes, cmodes; Δℓ=1, Δn=1)
    T = Float64
    amodes = cmodes.amodes
    wccache = WindowChainsCache(win, wmodes, cmodes.amodes)
    lnnsize = getlnnsize(cmodes)
    A3 = fill(T(0), lnnsize, lnnsize)
    for j=1:lnnsize, i=j:lnnsize
        l, n, n′ = getlnn(cmodes, i)
        L, N, N′ = getlnn(cmodes, j)
        if abs(L-l) > Δℓ || abs(n-N) > Δn || abs(n′-N′) > Δn
            continue
        end
        @show i,(l,n,n′),j,(L,N,N′)
        ell = [L, l]
        enn = [N, n′]
        enn′ = [N′, n]
        W2 = window_chain(ell, enn, enn′, wccache)
        if N == N′
            W2 *= 2
        else
            enn[1], enn′[1] = N′, N
            W2 += window_chain(ell, enn, enn′, wccache)
        end
        A = W2 / ((2*l+1) * (2*L+1) * nbar^2)
        A3[i,j] = A3[j,i] = A
    end
    return A3
end



end


# vim: set sw=4 et sts=4 :
