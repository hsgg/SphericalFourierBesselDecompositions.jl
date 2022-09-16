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


module Theory

export gen_Clnn_theory
export Clnn2CnlmNLM, sum_m_lmeqLM
export add_local_average_effect_nowindow
export calc_NobsA, calc_NobsA_z
export calc_CobsA, calc_CobsA_z


using ..Modes
using ..Windows
using ..WindowChains
using ..Cat2Anlm
using ..Utils
using QuadGK
using LinearAlgebra
using SparseArrays
using ProgressMeter

using ..MyBroadcast


function gen_Clnn_theory(pk, cmodes)
    # Currently this only supports the simplest model. Be ready to support more.
    Clnn = fill(0.0, getlnnsize(cmodes))
    for i=1:length(Clnn)
        l, k1, k2 = getlkk(cmodes, i)
        if k1 == k2
            Clnn[i] = pk(k1)
        end
    end
    return Clnn
end


######################### Clnn <--> CnlmNLM #######################

function Clnn2CnlmNLM(Clnn, cmodes)
    amodes = cmodes.amodes
    nlmsize = getnlmsize(amodes)
    CnlmNLM = fill(0.0, nlmsize, nlmsize)
    for i′=1:nlmsize, i=1:nlmsize
        n, l, m = getnlm(amodes, i)
        n′, l′, m′ = getnlm(amodes, i′)
        if l′==l && m′==m
            if isvalidlnn_symmetric(cmodes, l, n, n′)
                idx = getidx(cmodes, l, n, n′)
                CnlmNLM[i,i′] = Clnn[idx]
            end
        end
    end
    return CnlmNLM
end


function sum_m_lmeqLM(A_nlm_NLM, modes::AnlmModes)
    clnn = fill(NaN, modes.lmax+1, modes.nmax, modes.nmax)
    for n1=1:modes.nmax, n2=1:modes.nmax
        b1 = getnlmsize(modes, n1 - 1) + 1
        b2 = getnlmsize(modes, n2 - 1) + 1
        lmax = minimum(modes.lmax_n[[n1,n2]])
        for l=0:lmax
            L = l
            b = b1 + getlmsize(l-1)
            B = b2 + getlmsize(L-1)
            cl = real(A_nlm_NLM[b,B])
            for m=1:l
                M = m
                cl += 2 * real(A_nlm_NLM[b+m,B+M])
            end
            clnn[l+1,n1,n2] = cl / (2*l + 1)
        end
    end
    return clnn
end

sum_m_lmeqLM(anlm, cmodes::ClnnModes) = clnn2clΔnn̄(sum_m_lmeqLM(anlm, cmodes.amodes), cmodes)


function clnn2clΔnn̄(clnn, cmodes::ClnnModes)
    lnnsize = getlnnsize(cmodes)
    clΔnn̄ = fill(NaN, lnnsize)
    for i=1:lnnsize
        l, n1, n2 = getlnn(cmodes, i)
        clΔnn̄[i] = (clnn[l+1,n1,n2] + clnn[l+1,n2,n1]) / 2
    end
    return clΔnn̄
end


function clΔnn̄2clnn(clΔnn̄, cmodes::ClnnModes)
    lmax = cmodes.amodes.lmax
    nmax = cmodes.amodes.nmax
    lnnsize = getlnnsize(cmodes)
    clnn = fill(NaN, lmax + 1, nmax, nmax)
    for i=1:lnnsize
        l, n1, n2 = getlnn(cmodes, i)
        clnn[l+1,n1,n2] = clΔnn̄[i]
        clnn[l+1,n2,n1] = clΔnn̄[i]
    end
    return clnn
end


#clnn2clΔnn̄(clnn, bcmodes::Modes.ClnnBinnedModesOld) = clnn2clΔnn̄(clnn, bcmodes.fake_cmodes)
#clΔnn̄2clnn(clΔnn̄, bcmodes::Modes.ClnnBinnedModesOld) = clΔnn̄2clnn(clΔnn̄, bcmodes.fake_cmodes)



###################### Local average effect #########################################

function add_local_average_effect_nowindow(CNobs, cmodes, Veff)
    # assumes there is no window
    nmax = cmodes.amodes.nmax_l[1]
    dn00 = √(4*π) .* [quadgk(r->r^2 * cmodes.amodes.basisfunctions(n,0,r),
                             cmodes.amodes.rmin, cmodes.amodes.rmax)[1]
                      for n=1:nmax]
    C0nn = [CNobs[getidx(cmodes, 0, n, n)] for n=1:nmax]
    C011 = C0nn[1]
    A = sum(@. dn00^2 / Veff * C0nn / C011)
    @show A
    CNlocalobs = fill(NaN, size(CNobs))
    for i=1:length(CNlocalobs)
        l, n, n′ = getlnn(cmodes, i)
        Bnn′ = 0.0
        if l == 0
            Bnn′ = C0nn[n] + Clnn[n′] - A*C011 - 6*C0nn[n]*C0nn[n′]/Veff
        end
        Clnn′ = CNobs[getidx(cmodes, 0, n, n′)]
        Clnn′ = (1 + 3 * A * C011 / Veff) * Clnn′ - dn00[n] * dn00[n′] / Veff * Bnn′
        CNlocalobs[idx] = Clnn′
    end
    return CNlocalobs
end


############# shot noise

function calc_dn00(cmodes)
    nmax = cmodes.amodes.nmax_l[1]
    dn00 = √(4*π) .* [quadgk(r->r^2 * cmodes.amodes.basisfunctions(n,0,r),
			     cmodes.amodes.rmin, cmodes.amodes.rmax)[1]
		      for n=1:nmax]
    return dn00
end

function calc_dn00obs(dn00, Wlnn, cmodes)
    nmax = length(dn00)
    dn00obs = fill(0.0, nmax)
    for n=1:nmax
	for N=1:nmax
	    if isvalidlnn(cmodes, 0, n, N)
		idx = getidx(cmodes, 0, n, N)
		dn00obs[n] += Wlnn[idx] * dn00[N]
	    end
	end
    end
    return dn00obs
end

function calc_DWlnn(cmix, cmodes, dn00)
    lnnsize = size(cmix,2)
    δdd = fill(0.0, lnnsize)
    for i=1:lnnsize
	l, n, n′ = getlnn(cmodes, i)
	if l == 0
	    δdd[i] = dn00[n] * dn00[n′]
	end
    end
    DWlnn = cmix * δdd
    return DWlnn
end

@doc raw"""
    calc_NobsA(NwW_th, NW_th, cmix_wW, nbar, Veff, cmodes)

Calculate the observed shot noise including the local average effect for a
constant nbar.
"""
function calc_NobsA(NwW_th, NW_th, cmix_wW, nbar, Veff, cmodes)
    dn00 = calc_dn00(cmodes)
    dn00obs = calc_dn00obs(dn00, nbar .* NW_th, cmodes)

    trNWD = dn00'dn00obs / (Veff * nbar)
    DwWlnn = calc_DWlnn(cmix_wW, cmodes, dn00 / √Veff)

    N1 = NwW_th
    N23 = 2/nbar * DwWlnn
    N4 = trNWD * DwWlnn
    Nobs = N1 - N23 + N4

    return Nobs, N1, N23, N4
end


@doc raw"""
    calc_NobsA_z(NwW_th, nbar, cmodes, amodes_red, wWmix, wWmix_negm, wWtildemix, wWtildemix_negm)

Calculate the observed shot noise including the local average effect for
measured nbar(z).
"""
function calc_NobsA_z(NwW_th, nbar, cmodes, amodes_red, wWmix, wWmix_negm, wWtildemix, wWtildemix_negm)

    # N1
    N1 = NwW_th

    # N23
    lnnsize = getlnnsize(cmodes)
    nmax = amodes_red.nmax
    N23 = fill(0.0, lnnsize)
    for i=1:lnnsize

        lμ, nμ, nν = getlnn(cmodes, i)

        if !isvalidnlm(amodes_red, nμ, lμ, 0) || !isvalidnlm(amodes_red, nν, lμ, 0)
            continue
        end

        nμlμ = getidx(amodes_red, nμ, lμ, 0)
        nνlμ = getidx(amodes_red, nν, lμ, 0)

        for nλ=1:nmax
            nλl0 = getidx(amodes_red, nλ, 0, 0)
            for mμ=-lμ:lμ
                wWtilde2 = get_wmix(wWtildemix, wWtildemix_negm, nμlμ, mμ, nλl0, 0)
                wW2 = get_wmix(wWmix, wWmix_negm, nλl0, 0, nνlμ, mμ)

                wWtilde3 = get_wmix(wWtildemix, wWtildemix_negm, nνlμ, mμ, nλl0, 0)
                wW3 = get_wmix(wWmix, wWmix_negm, nλl0, 0, nμlμ, mμ)

                N23[i] += real(wWtilde2 * wW2 + wWtilde3 * wW3)
            end
        end

        N23[i] /= (2 * lμ + 1)
    end
    N23 ./= nbar


    # N4
    N4 = N23 / 2  # blissfull simplicity


    NobsA = N1 - N23 + N4

    return NobsA, N1, N23, N4
end

############# clustering terms

function get_anlm_r(anlm, nl, m)
    if m >= 0
        return anlm[nl+m]
    end
    return (-1)^m * conj(anlm[nl-m])
end


function calc_T23(wW_nlm_NLM, wW_nlm_NLM_negm, wW_nlm, W_nlm, amodes_red::AnlmModes, cmodes::ClnnModes, Veff)
    T = Float64
    wWs_nlm = conj(wW_nlm)
    lnnsize = getlnnsize(cmodes)
    #TlnnLNN = fill(T(0), lnnsize, lnnsize)
    TlnnLNN = spzeros(T, lnnsize, lnnsize)
    println("Calculate T matrix...")
    @time for j=1:lnnsize, i=1:lnnsize
        l_μ, n_μ, n_ν = getlnn(cmodes, i)
        l_ρ, n_ρ, n_ω = getlnn(cmodes, j)
        if !isvalidnlm(amodes_red, n_μ, l_μ, 0) || !isvalidnlm(amodes_red, n_ν, l_μ, 0) ||
            !isvalidnlm(amodes_red, n_ρ, l_ρ, 0) || !isvalidnlm(amodes_red, n_ω, l_ρ, 0)
            continue
        end
        nl_νμ = getidx(amodes_red, n_ν, l_μ, 0)
        nl_μμ = getidx(amodes_red, n_μ, l_μ, 0)
        nl_ρρ = getidx(amodes_red, n_ρ, l_ρ, 0)
        nl_ωρ = getidx(amodes_red, n_ω, l_ρ, 0)
        #doshow = (i ∈ [3] && j ∈ [1])
        for m_μ=-l_μ:l_μ
            Tμ = Complex{T}(0)
            Tν = Complex{T}(0)
            for m_ρ=-l_ρ:l_ρ
                wW_μ_ρ = get_wmix(wW_nlm_NLM, wW_nlm_NLM_negm, nl_μμ, m_μ, nl_ρρ, m_ρ)
                wW_ν_ρ = get_wmix(wW_nlm_NLM, wW_nlm_NLM_negm, nl_νμ, m_μ, nl_ρρ, m_ρ)
                W_ω = get_anlm_r(W_nlm, nl_ωρ, m_ρ)
                Tμ += wW_μ_ρ * W_ω
                Tν += wW_ν_ρ * W_ω
                #if doshow
                #    @show m_μ,m_ρ,wW_μ_ρ,W_ω,wW_μ_ρ*W_ω
                #end
            end
            wWs_ν = get_anlm_r(wWs_nlm, nl_νμ, m_μ)
            wWs_μ = get_anlm_r(wWs_nlm, nl_μμ, m_μ)
            #if doshow
            #    @show m_μ,wWs_ν,Tμ,wWs_μ*Tν
            #    @show m_μ,real(wWs_ν * Tμ + conj(wWs_μ * Tν)),wWs_ν * Tμ,conj(wWs_μ * Tν)
            #end
            TlnnLNN[i,j] += real(wWs_ν * Tμ + conj(wWs_μ * Tν))
        end
        #if doshow
        #    @show (l_μ,n_μ,n_ν),(l_ρ,n_ρ,n_ω),TlnnLNN[i,j]
        #end
        TlnnLNN[i,j] /= Veff * (2*l_μ + 1)
    end
    return TlnnLNN
end


@doc raw"""
    calc_CobsA_term4(C_th, cmix_W, cmix_wW, Veff, cmodes)

Calculate the observed power spectrum including the local average effect for a
constant nbar.

The `method` is by default an approximate formula.
"""
function calc_CobsA_term4(C_th, cmix_W, cmix_wW, Veff, cmodes)
    CwW_th = cmix_wW * C_th
    dn00 = calc_dn00(cmodes)
    DWlnn = calc_DWlnn(cmix_W, cmodes, dn00 / √Veff)
    DwWlnn = calc_DWlnn(cmix_wW, cmodes, dn00 / √Veff)

    lnnsize = getlnnsize(cmodes)
    DWlnn2lp1 = fill(0.0, lnnsize)
    for i=1:lnnsize
        l, n, n′ = getlnn(cmodes, i)
        DWlnn2lp1[i] = (2 * l + 1) * DWlnn[i]
    end

    trCWD = C_th'DWlnn2lp1

    return trCWD * DwWlnn

    #CwWA = CwW_th - trCWD * DwWlnn
    #return CwWA
end


function calc_T23_z(cmix_wW, cmodes, amodes_red, wWmix, wWmix_negm, W̃mix, W̃mix_negm)
    lnnsize = getlnnsize(cmodes)
    nmax0 = amodes_red.nmax
    p = Progress(lnnsize^2, 1.0, "T23: ")
    T23mat = mybroadcast(1:lnnsize, (1:lnnsize)') do ii,jj
        out = Array{Float64}(undef, length(ii))
        for k=1:length(ii)
            i = ii[k]
            j = jj[k]
            lμ, nμ, nν = getlnn(cmodes, i)
            lρ, nρ, nκ = getlnn(cmodes, j)

            if (# Yes, both terms 2 and 3 need all four to be valid modes.
                !isvalidnlm(amodes_red, nκ, lρ, 0) ||
                !isvalidnlm(amodes_red, nρ, lρ, 0) ||
                !isvalidnlm(amodes_red, nμ, lμ, 0) ||
                !isvalidnlm(amodes_red, nν, lμ, 0)
               )
                out[k] = 0.0
                continue
            end
            nκlρ = getidx(amodes_red, nκ, lρ, 0)
            nρlρ = getidx(amodes_red, nρ, lρ, 0)
            nμlμ = getidx(amodes_red, nμ, lμ, 0)
            nνlμ = getidx(amodes_red, nν, lμ, 0)

            T23 = 0.0
            for nλ=1:nmax0
                nλl0 = getidx(amodes_red, nλ, 0, 0)

                for mμ=-lμ:lμ
                    wW_λ00_ν = get_wmix(wWmix, wWmix_negm, nλl0, 0, nνlμ, mμ)
                    wW_λ00_μ = get_wmix(wWmix, wWmix_negm, nλl0, 0, nμlμ, mμ)

                    for mρ=-lρ:lρ
                        W̃ = get_wmix(W̃mix, W̃mix_negm, nκlρ, mρ, nλl0, 0)
                        wW_μ_ρ = get_wmix(wWmix, wWmix_negm, nμlμ, mμ, nρlρ, mρ)
                        wW_ν_ρ = get_wmix(wWmix, wWmix_negm, nνlμ, mμ, nρlρ, mρ)

                        T23 += real(W̃ * (wW_μ_ρ * wW_λ00_ν + wW_ν_ρ * wW_λ00_μ))

                        if nκ != nρ
                            # We assume that whatever we are multiplying is
                            # symmetric in nρ and nκ, and that the redundant
                            # values are not stored. Hence, we need to
                            # explicitly account for those terms.
                            W̃ = get_wmix(W̃mix, W̃mix_negm, nρlρ, mρ, nλl0, 0)
                            wW_μ_κ = get_wmix(wWmix, wWmix_negm, nμlμ, mμ, nκlρ, mρ)
                            wW_κ_ν = get_wmix(wWmix, wWmix_negm, nκlρ, mρ, nνlμ, mμ)

                            T23 += real(W̃ * (wW_μ_κ * wW_λ00_ν + wW_κ_ν * wW_λ00_μ))
                        end
                    end
                end
            end
            out[k] = T23 / (2*lμ + 1)
        end
        next!(p, step=length(ii), showvalues=[(:batchsize, length(ii))])
        return out
    end

    return T23mat
end


function calc_C4lnn_z(C_th, cmix_W, cmix_wW, cmodes, fskyinvlnn)

    CW = cmix_W * C_th
    CW0nn = get_0nn(CW, cmodes)

    finv_n00_n00 = get_0nn(fskyinvlnn, cmodes)

    fCWf = finv_n00_n00 * CW0nn * finv_n00_n00

    fCWf_lnn, _ = get_lnn_from_0nn(fCWf, cmodes)

    C4lnn = cmix_wW * fCWf_lnn

    return C4lnn
end


end


# vim: set sw=4 et sts=4 :
