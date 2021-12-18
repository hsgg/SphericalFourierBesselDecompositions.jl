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

# deprecated:
export add_local_average_effect, calc_CNAnlmNLM

using ..Modes
using QuadGK


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
            if isvalidlnn(cmodes, l, n, n′)
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
    # currently assumes there is no window
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
    DWlnn = calc_DWlnn(cmix_wW, cmodes, dn00 / √Veff)

    NwWA = NwW_th - (2/nbar - trNWD) * DWlnn
    return NwWA
end


function calc_Dlnnobs(cmix, dn00, cmodes)
    lnnsize = getlnnsize(cmodes)
    nmax = cmodes.amodes.nmax
    Dlnnobs = fill(0.0, lnnsize)
    for i=1:lnnsize
	l1, n1, n2 = getlnn(cmodes, i)
	for n1′=1:nmax, n2′=1:nmax
	    if isvalidlnn(cmodes, 0, n1′, n2′)
		i′ = getidx(cmodes, 0, n1′, n2′)
		Dlnnobs[i] += (2*l1 + 1) * cmix[i,i′] * dn00[n1′] * dn00[n2′]
	    end
	end
    end
    return Dlnnobs
end


function calc_W3l1n1n2(L, N, N′, dn00dn00V, cmodes, wk_cache)
    lnnsize = getlnnsize(cmodes)
    nmax = cmodes.amodes.nmax
    ell = [L, -1, 0]
    na = [N, 0, 0]
    nb = [N′, 0, 0]
    W3nn = fill(NaN, nmax, nmax)
    W3l1n1n2 = fill(NaN, lnnsize)
    for i1=1:lnnsize
	l1, n1, n2 = getlnn(cmodes, i1)
	ell[2] = l1
	na[2] = n1
	nb[2] = n2
	for n=1:nmax
	    na[3] = n
	    for n′=1:nmax
		nb[3] = n′
		W3nn[n,n′] = window_chain(ell, na, nb, wk_cache)
		if na[1] != nb[1]
		    na[1], nb[1] = nb[1], na[1]
		    W3nn[n,n′] += window_chain(ell, na, nb, wk_cache)
		else
		    W3nn[n,n′] *= 2
		end
	    end
	end
	W3l1n1n2[i1] = tr(dn00dn00V'W3nn)
    end
    return W3l1n1n2
end


@doc raw"""
    calc_CobsA(Clnn, Nobs_th, cmix, nbar, Veff, cmodes, wk_cache=nothing;
        method=:fast, Lmax=1)

Calculate the observed power spectrum including the local average effect for a
constant nbar.

The `method` is by default an approximate formula.
"""
function calc_CobsA(Clnn, Nobs_th, cmix, nbar, Veff, cmodes, wk_cache=nothing;
        method=:fast, Lmax=1)
    dn00 = calc_dn00(cmodes)
    dn00obs = calc_dn00obs(dn00, nbar .* Nobs_th, cmodes)
    DWlnn = calc_DWlnn(cmix, cmodes, dn00 / √Veff)
    D̃lnnobs = calc_Dlnnobs(cmix, dn00 / √Veff, cmodes)
    lnnsize = getlnnsize(cmodes)

    CW3 = fill(0.0, lnnsize)
    if method != :fast
	LNNidx = Int[]
	if method == :exact_sparse
	    # only calculate ℓ < Δℓ to save time.
	    Lmax = min(cmodes.amodes.lmax, Lmax)
	    for l=0:Lmax, n=1:2, n′=n:2
		if isvalidlnn(cmodes, l, n, n′)
		    push!(LNNidx, getidx(cmodes, l, n, n′))
		end
	    end
	else
	    LNNidx = collect(1:lnnsize)
	end
	dn00dn00V = dn00 * dn00' / Veff
	for LNN=LNNidx
	    L, N, N′ = getlnn(cmodes, LNN)
	    # expensive:
	    W3l1n1n2 = calc_W3l1n1n2(L, N, N′, dn00dn00V, cmodes, wk_cache)
	    CW3[LNN] = Clnn'W3l1n1n2 / (2*L+1)
	end
    else
	# approximation
	CW3 .= 2 * D̃lnnobs'Clnn * DWlnn
    end

    ClnnobsA = (cmix * Clnn
                + D̃lnnobs'Clnn * DWlnn  # only lnn=011 really matters
		- CW3)
    return ClnnobsA
end


@doc raw"""
    calc_CNobsA(Clnn, Nobs_th, cmix, nbar, Veff, cmodes, wk_cache=nothing; kwargs...)

Calculate the observed power spectrum including shot noise with the local
average effect for a constant nbar.

`kwargs` are passed to [calc_CobsA](@ref).
"""
function calc_CNobsA(Clnn, Nobs_th, cmix, nbar, Veff, cmodes, wk_cache=nothing; kwargs...)
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
