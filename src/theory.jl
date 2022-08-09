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
export calc_NobsA, calc_CobsA, calc_CNobsA
export calc_NobsA_z

# deprecated:
export add_local_average_effect, calc_CNAnlmNLM

using ..Modes
using ..Windows
using ..WindowChains
using ..Cat2Anlm
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

    NwWA = NwW_th - (2/nbar - trNWD) * DwWlnn
    return NwWA
end


@doc raw"""
    calc_NobsA_z(NwW_th, NW_th, cmix_wW, nbar, Veff, cmodes)

Calculate the observed shot noise including the local average effect for
measured nbar(z).
"""
function calc_NobsA_z(NwW_th, NW_th, cmix_wW, nbar, Veff, cmodes, amodes_red, wWmix, wWmix_negm, fskyinvlnn)
    #dn00 = calc_dn00(cmodes)
    #dn00obs = calc_dn00obs(dn00, nbar .* NW_th, cmodes)
    #trNWD = dn00'dn00obs / (Veff * nbar)
    #DwWlnn = calc_DWlnn(cmix_wW, cmodes, dn00 / √Veff)

    nmax0 = cmodes.amodes.nmax_l[1]

    # N4
    N234arr = fill(0.0, getlnnsize(cmodes))
    for i=1:length(N234arr)
        l, n1, n2 = getlnn(cmodes, i)

        if !isvalidnlm(amodes_red, n1, l, 0) || !isvalidnlm(amodes_red, n2, l, 0)
            continue
        end

        nmaxl = amodes_red.nmax_l[l+1]
        nl_μμ = getidx(amodes_red, n1, l, 0)
        nl_νμ = getidx(amodes_red, n2, l, 0)

        N234 = 0.0
        for nrho=1:nmaxl, nlambda=1:nmaxl
            fNf = 0.0

            # add N4 term:
            if isvalidlnn_symmetric(cmodes, 0, nrho, nlambda)
                for nϵ=1:nmax0, nα=1:nmax0
                    if isvalidlnn_symmetric(cmodes, 0, nrho, nϵ) && isvalidlnn_symmetric(cmodes, 0, nϵ, nα) && isvalidlnn_symmetric(cmodes, 0, nα, nlambda)
                        j1 = getidx(cmodes, 0, nrho, nϵ)
                        j2 = getidx(cmodes, 0, nϵ, nα)
                        j3 = getidx(cmodes, 0, nα, nlambda)
                        fNf += fskyinvlnn[j1] * NW_th[j2] * fskyinvlnn[j3]
                        #@show fNf
                    end
                end
            end

            # add N2 term:
            if isvalidlnn_symmetric(cmodes, 0, nrho, nlambda)
                j2 = getidx(cmodes, 0, nrho, nlambda)
                fNf -= fskyinvlnn[j2] / nbar
            end
            #@show fNf

            # add N3 term:
            if isvalidlnn_symmetric(cmodes, 0, nlambda, nrho)
                j3 = getidx(cmodes, 0, nlambda, nrho)
                fNf -= fskyinvlnn[j3] / nbar
            end
            #@show fNf

            #@show (l,n1,n2),nrho,nlambda,fNf

            if fNf == 0
                continue
            end

            nl_ρ = getidx(amodes_red, nrho, 0, 0)
            nl_λ = getidx(amodes_red, nlambda, 0, 0)
            for m=-l:l
                wW_rhomu = get_anlmNLM_r(wWmix, wWmix_negm, nl_μμ, m, nl_ρ, 0)
                wW_lamnu = get_anlmNLM_r(wWmix, wWmix_negm, nl_νμ, m, nl_λ, 0)
                N234 += real(wW_rhomu * fNf * wW_lamnu)
            end
        end
        N234arr[i] = 1 / (2*l + 1) * N234
    end


    @show NwW_th[1:5]
    @show NW_th[1:5]
    @show N234arr[1:5]

    NwWA = NwW_th + N234arr
    return NwWA
end


function get_anlm_r(anlm, nl, m)
    if m >= 0
        return anlm[nl+m]
    end
    return (-1)^m * conj(anlm[nl-m])
end

const get_anlmNLM_r = Windows.get_wmix


function calc_terms23_transform(wW_nlm_NLM, wW_nlm_NLM_negm, wW_nlm, W_nlm, amodes_red::AnlmModes, cmodes::ClnnModes, Veff)
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
                wW_μ_ρ = get_anlmNLM_r(wW_nlm_NLM, wW_nlm_NLM_negm, nl_μμ, m_μ, nl_ρρ, m_ρ)
                wW_ν_ρ = get_anlmNLM_r(wW_nlm_NLM, wW_nlm_NLM_negm, nl_νμ, m_μ, nl_ρρ, m_ρ)
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


function calc_T23_z(cmix_wW, cmodes, amodes_red, wWmix, wWmix_negm, Wmix, Wmix_negm, fskyinvlnn)
    lnnsize = getlnnsize(cmodes)
    #T23mat = fill(0.0, lnnsize, lnnsize)
    nmax0 = cmodes.amodes.nmax_l[1]
    #p = Progress(lnnsize^2, 0.2, "T23: ")
    #Threads.@threads for i=1:lnnsize
    #for j=1:lnnsize
    T23mat = mybroadcast2d(1:lnnsize, (1:lnnsize)') do ii,jj
        out = Array{Float64}(undef, length(ii))
        for k=1:length(ii)
            i = ii[k]
            j = jj[k]
            lμ, nμ, nν = getlnn(cmodes, i)
            lσ, nσ, nα = getlnn(cmodes, j)

            if !isvalidnlm(amodes_red, nσ, lσ, 0) ||
                !isvalidnlm(amodes_red, nα, lσ, 0) ||
                !isvalidnlm(amodes_red, nμ, lμ, 0) ||
                !isvalidnlm(amodes_red, nν, lμ, 0)
                # Yes, both terms 2 and 3 need all four to be valid modes.
                out[k] = 0.0
                continue
            end
            nσlσ = getidx(amodes_red, nσ, lσ, 0)
            nαlσ = getidx(amodes_red, nα, lσ, 0)
            nμlμ = getidx(amodes_red, nμ, lμ, 0)
            nνlμ = getidx(amodes_red, nν, lμ, 0)

            T23 = 0.0
            for nβ=1:nmax0, nλ=1:nmax0
                if !isvalidlnn_symmetric(cmodes, 0, nβ, nλ)
                    continue
                end
                f0nβnλ = fskyinvlnn[getidx(cmodes, 0, nβ, nλ)]

                nβl0 = getidx(amodes_red, nβ, 0, 0)
                nλl0 = getidx(amodes_red, nλ, 0, 0)

                for mμ=-lμ:lμ
                    wW_nλ00_nνlμmμ = get_anlmNLM_r(wWmix, wWmix_negm, nλl0, 0, nνlμ, mμ)
                    wW_nλ00_nμlμmμ = get_anlmNLM_r(wWmix, wWmix_negm, nλl0, 0, nμlμ, mμ)
                    for mσ=-lσ:lσ
                        W_nαlσmσ_nβ00 = get_anlmNLM_r(Wmix, Wmix_negm, nαlσ, mσ, nβl0, 0)
                        wW_μ_σ = get_anlmNLM_r(wWmix, wWmix_negm, nμlμ, mμ, nσlσ, mσ)
                        wW_ν_σ = get_anlmNLM_r(wWmix, wWmix_negm, nνlμ, mμ, nσlσ, mσ)

                        T23 += real(f0nβnλ * W_nαlσmσ_nβ00 * (wW_nλ00_nνlμmμ * wW_μ_σ + wW_nλ00_nμlμmμ * wW_ν_σ))
                    end
                end
            end
            #T23mat[i,j] = T23 / (2*lμ + 1)
            out[k] = T23 / (2*lμ + 1)
            if nσ != nα
                # We assume that whatever we are multiplying is symmetric in nσ and
                # nα, and that the redundant values are not stored. Hence, we need
                # to explicitly account for those symmetric terms.
                out[k] *= 2
            end
        end
        #next!(p, step=length(ii))
        return out
    end

    return T23mat
end


function calc_C4lnn_z(C_th, cmix_W, cmix_wW, cmodes, fskyinvlnn)
    nmax0 = cmodes.amodes.nmax[1]

    CW = cmix_W * C_th
    CW0nn = [isvalidlnn_symmetric(cmodes, 0, nϵ, nα) ? CW[getidx(cmodes, 0, nϵ, nα)] : 0.0
             for nϵ=1:nmax0, nα=1:nmax0]

    finv_n00_n00 = [isvalidlnn_symmetric(cmodes, 0, nϵ, nα) ? fskyinvlnn[getidx(cmodes, 0, nϵ, nα)] : 0.0
                    for nϵ=1:nmax0, nα=1:nmax0]

    fCWf = finv_n00_n00 * CW0nn * finv_n00_n00


    lnnsize = getlnnsize(cmodes)
    C4lnn = fill(0.0, lnnsize)
    for i=1:lnnsize
        lμ, nμ, nν = getlnn(cmodes, i)
        C4 = 0.0
        for nρ=1:nmax0, nλ=nρ:nmax0  # cmix already doubled-includes symmetric terms
            if isvalidlnn(cmodes, 0, nρ, nλ)  # cmix already doubled-includes symmetric terms
                j = getidx(cmodes, 0, nρ, nλ)
                C4 += cmix_wW[i,j] * fCWf[nρ,nλ]
            end
        end
        C4lnn[i] = C4
    end

    return C4lnn
end



@doc raw"""
    calc_CNobsA(Clnn, Nobs_th, cmix, nbar, Veff, cmodes, wk_cache=nothing; kwargs...)

Calculate the observed power spectrum including shot noise with the local
average effect for a constant nbar.

`kwargs` are passed to [calc_CobsA](@ref).
"""
function calc_CNobsA(Clnn, Nobs_th, cmix, nbar, Veff, cmodes, wk_cache=nothing; kwargs...)
    error("outdated function")
    NobsA = calc_NobsA(Nobs_th, cmix, nbar, Veff, cmodes)
    CobsA = calc_CNobsA(Clnn, Nobs_th, cmix, nbar, Veff, cmodes, wk_cache=nothing; kwargs...)
    return CobsA .+ NobsA
end


#################### obsolete local average effect functions ###################

function add_local_average_effect(CNlnn, cmix, Wlnn, cmodes, Veff)
    base.depwarn("'add_local_average_effect()' is deprecated. Use 'calc_NobsA()' and 'calc_CobsA()' instead.", :add_local_average_effect)
    # Note 1: We only implement the δᴷₗ₀ terms, as the others are negligible
    # when Veff is large.
    #
    # Note 2: We assume there is no bandwidth binning.

    nmax = cmodes.amodes.nmax_l[1]

    # dn00
    dn00 = √(4*π) .* [quadgk(r->r^2 * cmodes.amodes.basisfunctions(n,0,r),
                             cmodes.amodes.rmin, cmodes.amodes.rmax)[1]
                      for n=1:nmax]

    # dn00obs
    dn00obs = fill(0.0, size(dn00))
    for n=1:nmax
        for N=1:nmax
            if isvalidlnn(cmodes, 0, n, N)
                idx = getidx(cmodes, 0, n, N)
                dn00obs[n] += Wlnn[idx] * dn00[N]
            end
        end
    end

    # sum_dn00obs_C0nn
    sum_dn00obs_C0nn = fill(0.0, nmax)
    for n=1:nmax
        for n′′=1:nmax
            if isvalidlnn(cmodes, 0, n, n′′)
                i′′ = getidx(cmodes, 0, n, n′′)
                sum_dn00obs_C0nn[n] += dn00obs[n′′] * CNlnn[i′′]
            end
        end
    end

    # Dlnnobs
    Dlnnobs = fill(0.0, size(CNlnn))
    for i=1:length(Dlnnobs)
        l1, n1, n2 = getlnn(cmodes, i)
        for n1′=1:nmax, n2′=1:nmax
            if isvalidlnn(cmodes, 0, n1′, n2′)
                i′ = getidx(cmodes, 0, n1′, n2′)
                Dlnnobs[i] += (2*l1 + 1) * cmix[i,i′] * dn00[n1′] * dn00[n2′]
            end
        end
    end

    # sum_Dobs_CN
    sum_Dobs_CN = Dlnnobs' * CNlnn

    # debug
    C011 = CNlnn[getidx(cmodes, 0, 1, 1)]
    A = sum_Dobs_CN / Veff / C011
    #@show dn00 dn00obs Dlnnobs sum_dn00obs_C0nn sum_Dobs_CN Veff C011 A
    @show "first" Dlnnobs[1:10] Dlnnobs[11:20] sum_Dobs_CN Veff C011 A
    @show sum_dn00obs_C0nn[1]

    # CNAlnn
    CNAlnn = deepcopy(CNlnn)
    for n=1:nmax, n′=1:nmax
        if isvalidlnn(cmodes, 0, n, n′)
            i = getidx(cmodes, 0, n, n′)
            CNAlnn[i] -= dn00[n′] / Veff * sum_dn00obs_C0nn[n]
            @show n′,dn00[n′],n,sum_dn00obs_C0nn[n]
            CNAlnn[i] -= dn00[n] / Veff * sum_dn00obs_C0nn[n′]
            CNAlnn[i] += dn00[n] * dn00[n′] / Veff^2 * sum_Dobs_CN
        end
    end
    return CNAlnn
end


function calc_CNAnlmNLM(CNlnn::AbstractVector, wmix, cmodes, Veff)
    base.depwarn("'calc_CNAnlmNLM()' is deprecated. Use 'calc_NobsA()' and 'calc_CobsA()' instead.", :calc_CNAnlmNLM)
    # Note 1: We only implement the δᴷₗ₀ terms, as the others are negligible
    # when Veff is large.
    #
    # Note 2: We assume there is no bandwidth binning.

    amodes = cmodes.amodes
    nmax = cmodes.amodes.nmax_l[1]

    # dn00
    dn00 = √(4*π) .* [quadgk(r->r^2 * cmodes.amodes.basisfunctions(n,0,r),
                             cmodes.amodes.rmin, cmodes.amodes.rmax)[1]
                      for n=1:nmax]

    # dnlm
    dnlm = fill(0.0, getnlmsize(amodes))
    for i=1:length(dnlm)
        n, l, m = getnlm(amodes, i)
        if l == 0 && m == 0
            dnlm[i] = dn00[n]
        end
    end

    dnlmobs = wmix * dnlm

    # sum_dnlmobs_Clnn
    sum_dnlmobs_Clnn = fill(0.0im, getnlmsize(amodes))
    for i=1:length(sum_dnlmobs_Clnn)
        n, l, m = getnlm(amodes, i)
        nmaxl = cmodes.amodes.nmax_l[l+1]
        for n′′=1:nmaxl
            if isvalidlnn(cmodes, l, n, n′′)
                i′′ = getidx(amodes, n′′, l, m)
                idx′′ = getidx(cmodes, l, n, n′′)
                sum_dnlmobs_Clnn[i] += dnlmobs[i′′] * CNlnn[idx′′]
            end
        end
    end

    # Dlnnobs
    Dlnnobs = fill(0.0, getlnnsize(cmodes))
    for i=1:length(Dlnnobs)
        l1, n1, n2 = getlnn(cmodes, i)
        i1base = getidx(amodes, n1, l1, 0)
        i2base = getidx(amodes, n2, l1, 0)
        D = real(conj(dnlmobs[i1base]) * dnlmobs[i2base])
        for m=1:l1
            D += 2 * real(conj(dnlmobs[i1base+m]) * dnlmobs[i2base+m])
        end
        Dlnnobs[i] = D
    end

    # sum_Dobs_CN
    sum_Dobs_CN = Dlnnobs' * CNlnn

    # debug
    C011 = CNlnn[getidx(cmodes, 0, 1, 1)]
    A = sum_Dobs_CN / Veff / C011
    #@show dnlm dnlmobs Dlnnobs sum_dnlmobs_Clnn sum_Dobs_CN Veff C011 A
    @show "second" Dlnnobs[1:10] Dlnnobs[11:20] sum_Dobs_CN Veff C011 A
    @show sum_dnlmobs_Clnn[getidx.(amodes, 1, 0, 0)]

    # CWnlmNLM
    CWnlmNLM = fill(im*NaN, getnlmsize(amodes), getnlmsize(amodes))
    for i′=1:size(CWnlmNLM,2), i=1:size(CWnlmNLM,1)
        n, l, m = getnlm(amodes, i)
        n′, l′, m′ = getnlm(amodes, i′)
        CW = 0.0im
        if l==l′ && m==m′
            if isvalidlnn(cmodes, l, n, n′)
                idx = getidx(cmodes, l, n, n′)
                CW += CNlnn[idx]
            end
        end
        if l′==0 && m′==0
            CW -= dn00[n′] / Veff * sum_dnlmobs_Clnn[i]
            if l==l′ && m==m′
                @show n′,l′,m′,dn00[n′],n,l,m,sum_dnlmobs_Clnn[i]
            end
        end
        if l==0 && m==0
            CW -= dn00[n] / Veff * conj(sum_dnlmobs_Clnn[i′])
        end
        if l′==0 && m′==0 && l==0 && m==0
            CW += dn00[n] * dn00[n′] / Veff^2 * sum_Dobs_CN
        end
        CWnlmNLM[i,i′] = CW
    end
    return CWnlmNLM
end



end


# vim: set sw=4 et sts=4 :
