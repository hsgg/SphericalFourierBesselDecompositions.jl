# Purpose: Here we provide functions to calculate various transforms of the
# window function, as needed for shot noise, for the SFB transform itself, and
# for mode decoupling.
#
# We define the abstract type 'WindowFunction' so that we can specialize on the
# following subtypes. All of them follow the same philosophy that the window
# function gives the probability of a type of galaxy being in the survey in a
# voxel, and the 'WindowFunction' object acts like a 2D array with the first
# dimension being indexing the radius, the second dimension a healpix on the
# sky.
#
#   * 'WindowFunction3D': This is a window function that specifies the window
#   in every voxel of the survey. The voxels are given at each radius by a
#   healpix map.
#
#   * 'WindowFunction3DSeparable': This window function type assumes that the
#   radial selection and angular mask are separable, i.e., W(r⃗) = M(r̂)⋅ϕ(r).
#
#
# Todo:
#
# - power_win_mix() is rather slow on large problems. Generally, there are
# several ways to improve the speed.
#
#   * First, improve single-core performance.
#
#   * Second, specialize on window functions that have separable radial and
#   angular selection functions, and then take a perturbative or similar
#   approach to calculate corrections.
#
#   * Third, parallelize.
#
#   * Fourth, exploit symmetry of (2ℓ+1)*M.


module Windows

export ConfigurationSpaceModes
export window_r, apply_window, apodize_window
export win_rhat_ln, integrate_window, calc_wmix, power_win_mix, win_lnn
export check_nsamp

using ..Modes
using ..HealPy
using ..SeparableArrays
using LinearAlgebra
using SpecialFunctions
using WignerSymbols
using SparseArrays
#using WignerFamilies  # needs some fixes, but will be much faster than WignerSymbols

using Distributed
using SharedArrays
#using Base.Threads  # Threads.@threads macro, export JULIA_NUM_THREADS=8

#using QuadGK  # for testing

#using Profile


#SparseArrays.rowvals(mat) = 1:size(mat,1)  # we also want to use it for full vectors


struct ConfigurationSpaceModes{Tarr}
    rmin::Real
    rmax::Real
    Δr::Real
    r::Tarr
    nr::Integer
    npix::Integer
    nside::Integer
end

# for the @__dot syntax:
Base.length(w::ConfigurationSpaceModes) = 1
Base.iterate(w::ConfigurationSpaceModes) = s, nothing
Base.iterate(w::ConfigurationSpaceModes, x) = nothing


function ConfigurationSpaceModes(rmin, rmax, nr, nside)
    Δr = (rmax - rmin) / nr
    r = range(rmin+Δr/2, rmax-Δr/2, length=nr)  # midpoints
    npix = hp.nside2npix(nside)
    return ConfigurationSpaceModes(rmin, rmax, Δr, r, nr, npix, nside)
end


window_r(wmodes::ConfigurationSpaceModes) = wmodes.r, wmodes.Δr


# only a basic implementation, with lots of edge cases poorly handled
function apodize_window(win, wmodes::ConfigurationSpaceModes, smooth=50.0)
    winapod = deepcopy(win)
    r, Δr = window_r(wmodes)
    nr = length(r)
    npix = size(win,2)
    nside = hp.npix2nside(npix)
    Δi = ceil(Int, smooth / Δr)
    if iseven(Δi)
        Δi += 1
    end
    weights = @. exp(-((1:Δi) - Δi/2)^2 / (2 * 10^2))
    weights .*= Δi / sum(weights)
    @show weights
    for j=1:npix, i=Δi:nr-Δi
        mmin = i - Δi ÷ 2
        mmax = i + Δi ÷ 2
        w = win[mmin:mmax,j] .* weights
        winapod[i,j] = mean(w)
    end
    return winapod
end


function apply_window(rθϕ, win, rmin, rmax, win_r, win_Δr)
    nside = hp.npix2nside(size(win,2))
    r_out = Float32[]
    θ_out = Float32[]
    ϕ_out = Float32[]
    r = rθϕ[1,:]
    θ = rθϕ[2,:]
    ϕ = rθϕ[3,:]
    idx_ang = hp.ang2pix(nside, θ, ϕ) .+ 1
    Wmax = maximum(win)
    @show extrema(r)
    for i=1:length(r)
        !(rmin <= r[i] <= rmax) && continue
        idx_r = ceil(Int, (r[i] - rmin) / win_Δr)
        (idx_r == 0) && (idx_r += 1)
        (idx_r == size(win,1)+1) && (idx_r -= 1)
        if rand() <= win[idx_r,idx_ang[i]] / Wmax
            # include in survey
            push!(r_out, r[i])
            push!(θ_out, θ[i])
            push!(ϕ_out, ϕ[i])
        end
    end
    return [r_out θ_out ϕ_out]'
end

apply_window(rθϕ, win, wmodes::ConfigurationSpaceModes) = begin
    apply_window(rθϕ, win, wmodes.rmin, wmodes.rmax, wmodes.r, wmodes.Δr)
end


function integrate_window(win, wmodes::ConfigurationSpaceModes)
    nr = size(win, 1)
    npix = size(win,2)
    r = wmodes.r
    Δr = wmodes.Δr
    ΔΩpix = 4*π / npix
    radial = [sum(win[i,:]) for i=1:size(win,1)]
    Veff = Δr * ΔΩpix * sum(@. radial * r^2)
    return Veff
end


function win_rhat_ln(win, wmodes::ConfigurationSpaceModes, amodes::AnlmModes)
    gnl = amodes.basisfunctions
    nr = size(win, 1)
    r, Δr = window_r(wmodes)
    W_rhat_ln = fill(NaN, size(win,2), amodes.lmax+1, amodes.nmax)
    check_nsamp_1gnl(amodes, wmodes)
    for n=1:amodes.nmax, l=0:amodes.lmax_n[n]
        l==0 && @show n,l
        int_nowin = @. r^2 * gnl(n, l, r)
        W_rhat_ln[:,l+1,n] .= win' * int_nowin
    end
    @. W_rhat_ln *= Δr
    #@assert all(isfinite.(W_rhat_ln))  # not all needs to be finite if we limit by kmax
    return W_rhat_ln
end

# specialize on separable windows
function win_rhat_ln(win::SeparableArray, wmodes::ConfigurationSpaceModes, amodes::AnlmModes)
    gnl = amodes.basisfunctions
    nr = size(win, 1)
    r, Δr = window_r(wmodes)
    W_ln = fill(NaN, amodes.lmax+1, amodes.nmax)
    check_nsamp_1gnl(amodes, wmodes)
    for n=1:amodes.nmax, l=0:amodes.lmax_n[n]
        l==0 && @show n,l,amodes.nmax
        W_ln[l+1,n] = Δr * sum(@. r^2 * gnl(n, l, r) * win.phi)
    end
    return SeparableArray(win.mask, W_ln, name1=:mask, name2=:w_ln)
end


# This should be very performant
function calc_wmix(win, wmodes::ConfigurationSpaceModes, amodes::AnlmModes; neg_m=false)
    nlmsize = getnlmsize(amodes)
    wmix = fill(NaN*im, nlmsize, nlmsize)
    @debug "wmix" length(wmix), size(wmix)

    r, Δr = window_r(wmodes)
    nr = length(r)

    LMAX = 2 * amodes.lmax
    Wr_lm = calc_Wr_lm(win, LMAX, amodes.nside)
    @debug "Wr_lm" LMAX amodes.nside size(Wr_lm) Wr_lm[:,1]

    LMLM = fill(0, LMAX+1, LMAX+1)
    for L=0:LMAX, M=0:L
        LMLM[L+1,M+1] = hp.Alm.getidx(LMAX, L, M) + 1
    end
    @debug "LMLM" size(LMLM)

    check_nsamp(amodes, wmodes)

    gnl = amodes.basisfunctions
    @time for n′=1:amodes.nmax, n=1:amodes.nmax, l′=0:amodes.lmax_n[n′], l=0:amodes.lmax_n[n]
        ibase = getidx(amodes, n, l, 0)
        i′base = getidx(amodes, n′, l′, 0)
        ibase==1 && @show ibase,i′base, n,n′, l,l′, nlmsize

        gg1 = @. r^2 * gnl(n,l,r) * gnl(n′,l′,r)
        ## debug
        #gg1_quadgk,E = quadgk(r->r^2 * gnl(n,l,r) * gnl(n′,l′,r), amodes.rmin, amodes.rmax)
        #@debug "gg1" sum(gg1)*Δr gg1_quadgk,E
        gg1sum = sum(gg1)

        for m=0:l, m′=0:l′
            if neg_m
                m = -m
            end
            i = ibase + abs(m)
            i′ = i′base + m′
            M = m - m′

            L1 = max(abs(l-l′),abs(M)):(l+l′)
            w3j1 = wigner3j.(Float64, l, l′, L1, -m, m′)
            #w3j_f = wigner3j_f(Float64, l, l′, -m, m′)
            #w3j2 = w3j_f.symbols
            #L2 = eachindex(w3j_f)
            #@show l,l′ m,m′ M L1 L2
            #@show w3j1 w3j2
            #@assert all(@. abs(w3j1 - w3j2) < eps(1.0))
            #@assert all(L1 .== L2)
            L = L1
            w3j = w3j1
            #@show w3j

            w3j000 = @. wigner3j000(L, l, l′)
            #@show w3j000
            #@show size(w3j) size(L) size(w3j000)
            #@show eachindex(w3j) eachindex(L) eachindex(w3j000)
            gaunt = @. √((2*L+1) * (2*l+1) * (2*l′+1) / (4*π)) * w3j000 * w3j
            #@debug "Gaunt" l,m l′,m′ L M gaunt[1] 1/√(4*π) √(4*π)*gaunt[1]

            w_ang = 0.0im
            for j=1:length(L)
                #@assert -L[j] <= M <= L[j]
                #LM = hp.Alm.getidx(LMAX, L[j], abs(M)) + 1
                LM = LMLM[L[j]+1,abs(M)+1]
                w_ang += gaunt[j] * (gg1' * Wr_lm[:,LM])
                #@debug "Wr_lm" L[j],M Wr_lm[1,LM] Wr_lm[1,LM]/√(4*π)
            end
            if M < 0
                w_ang = conj(w_ang)
            end
            #@debug "wmix" i,i′ l,m l′,m′ w_ang gg1sum
            wmix[i,i′] = (-1)^m * w_ang * Δr
            #@show wmix[i,i′]
            @assert isfinite(wmix[i,i′])
        end
    end
    @assert all(isfinite.(wmix))
    return wmix
end


# This should be very performant
function win_lnn(win, wmodes::ConfigurationSpaceModes, cmodes::ClnnModes)
    lnnsize = getlnnsize(cmodes)
    Wlnn = fill(NaN, lnnsize)
    @show length(Wlnn), size(Wlnn)

    nr = size(win, 1)
    r, Δr = window_r(wmodes)

    # Note: the maximum ℓ we need is 0. However, healpy changes precision, and
    # for comparison we use the same lmax as elsewhere.
    Wr_00 = Array{Float64,1}(calc_Wr_lm(win, 2*cmodes.amodes.lmax, cmodes.amodes.nside)[:,1])
    @show size(Wr_00) typeof(Wr_00)
    @assert all(isfinite.(Wr_00))

    check_nsamp(cmodes.amodes, wmodes)

    gnl = cmodes.amodes.basisfunctions
    @time for i=1:lnnsize
        l, n, n′ = getlnn(cmodes, i)
        #@show i, l,n,n′, lnnsize

        gg = @. r^2 * gnl(n,l,r) * gnl(n′,l,r) * Wr_00

        Wlnn[i] = Δr * sum(gg) / √(4π)
        if !isfinite(Wlnn[i])
            @error "not finite" i l,n,n′ extrema(gg) Δr sum(gg) Wlnn[i]
            break
        end
    end
    @assert all(isfinite.(Wlnn))
    return Wlnn
end


function wigner3j000(l, l′, L)
    (abs(l-l′) <= L <= l+l′) || return 0.0
    J = l + l′ + L
    iseven(J) || return 0.0
    wig3j = (-1)^(J÷2) * exp(0.5*loggamma(1+J-2l) + 0.5*loggamma(1+J-2l′)
                             + 0.5*loggamma(1+J-2L) - 0.5*loggamma(1+J+1)
                             + loggamma(1+J/2)
                             - loggamma(1+J/2-l) - loggamma(1+J/2-l′)
                             - loggamma(1+J/2-L))
    return wig3j
end


function power_win_mix(wmix, wmix_negm, cmodes)
    amodes = cmodes.amodes
    nlmsize = getnlmsize(amodes)
    lnnsize = getlnnsize(cmodes)
    mmix = fill(NaN, lnnsize, lnnsize)
    for i′=1:lnnsize, i=1:lnnsize
        l, n, n′ = getlnn(cmodes, i)
        L, N, N′ = getlnn(cmodes, i′)
        mix = 0.0
        for m=0:l, M=0:L
            j = getidx(amodes, n, l, m)
            j′ = getidx(amodes, n′, l, m)
            J = getidx(amodes, N, L, M)
            J′ = getidx(amodes, N′, L, M)
            mix += real(wmix[j,J] * conj(wmix[j′,J′]))
            if m>0
                mix += real(conj(wmix_negm[j,J]) * wmix_negm[j′,J′])
            end
            if M>0
                mix += real(wmix_negm[j,J] * conj(wmix_negm[j′,J′]))
            end
            if m>0 && M>0
                mix += real(conj(wmix[j,J]) * wmix[j′,J′])
            end
        end
        mmix[i,i′] = mix / (2*l + 1)
    end
    return mmix
end


function calc_Wr_lm(win, LMAX, Wnside)
    nr = size(win,1)
    Wr_lm = fill(NaN*im, nr, getlmsize(LMAX))
    @time for i=1:nr
        W = hp.ud_grade(win[i,:], Wnside)
        Wr_lm[i,:] .= hp.map2alm(W, lmax=LMAX)
    end
    return Wr_lm
end

# specialize
function calc_Wr_lm(win::SeparableArray, LMAX, Wnside)
    mask = hp.ud_grade(win.mask, Wnside)
    wlm = hp.map2alm(mask, lmax=LMAX)
    return SeparableArray(win.phi, wlm, name1=:phi, name2=:wlm)
end


function optimize_Wr_lm_layout(Wr_lm)
    return Wr_lm  # Don't change the semantics! If speed is a problem, we will have to revisit this.
    #LMmax = size(Wr_lm,2)
    #w = Array{eltype(Wr_lm),1}[]
    #for lm=1:LMmax
    #    wr = Wr_lm[:,lm]
    #    push!(w, wr)
    #end
    #return w
end

optimize_Wr_lm_layout(Wr_lm::SeparableArray) = Wr_lm  # noop for SeparableArray



function cmix_kernel(gg1, gg2, wr)
    real(dot(gg1, wr) * conj(dot(gg2, wr)))
end


function test_cmix_kernel()
    gg1 = rand(100)
    gg2 = rand(100)
    wr = rand(Complex{Float64}, 100)
    l = 5
    L = 10
    LMAX = l + L
    L1M1cache = [hp.Alm.getidx.(LMAX, L, 0:L) .+ 1 for L=0:LMAX]
    Wnside = estimate_nside(LMAX)
    win = rand(100, hp.nside2npix(Wnside))
    @show typeof(win)
    Wr_lm = optimize_Wr_lm_layout(calc_Wr_lm(win, LMAX, Wnside))
    @show typeof(Wr_lm)

    # compile
    cmix_kernel(gg1, gg2, wr)
    calc_cmix_ang(l, L, L1M1cache, gg1, gg2, Wr_lm)

    s = 0.0
    @time cmix_kernel(gg1, gg2, wr)
    @time cmix_kernel(gg1, gg2, Wr_lm[1])

    @show typeof(l) typeof(L) typeof(L1M1cache) typeof(gg1) typeof(gg2) typeof(Wr_lm)

    @time calc_cmix_ang(l, L, L1M1cache, gg1, gg2, Wr_lm)
    @time m_ang = calc_cmix_ang(l, L, L1M1cache, gg1, gg2, Wr_lm)
    @show typeof(m_ang)
end


function calc_cmix_ang(l, L, L1M1cache, gg1, gg2, Wr_lm)
    #@debug "calc_cmix_ang" l L size(gg1) size(gg2) size(Wr_lm) size(L1M1cache) size(Wr_lm[1])
    m_ang = 0.0
    for L1=abs(l-L):2:(l+L)
        L1M1 = L1M1cache[L1+1][1]
        s = cmix_kernel(gg1, gg2, Wr_lm[:,L1M1])
        for M1=1:L1
            #@debug "" L1 M1
            L1M1 = L1M1cache[L1+1][M1+1]
            s += 2 * cmix_kernel(gg1, gg2, Wr_lm[:,L1M1])
        end
        m_ang += s * wigner3j000(l, L, L1)^2
    end
    return m_ang
end


function calc_cmixii(i, i′, cmodes, r, Δr, gnlr, Wr_lm, L1M1cache,
                     div2Lp1, interchange_NN′)
    l, n, n′ = getlnn(cmodes, i)
    L, N, N′ = getlnn(cmodes, i′)
    if interchange_NN′
        tmp = N′
        N′ = N
        N = tmp
    end

    showthis = ((l==L==0 && n==N′==1 && n′==N==2) || (l==L==0 && n==N′==2 && n′==N==1))
    showthis && @show "huzzah",i,i′, n,n′, N,N′, l,L
    #@show i,i′, (l,n,n′), (L,N,N′)

    gg1 = @. r^2 * gnlr[:,n,l+1] * gnlr[:,N,L+1]
    gg2 = @. r^2 * gnlr[:,n′,l+1] * gnlr[:,N′,L+1]

    mix = calc_cmix_ang(l, L, L1M1cache, gg1, gg2, Wr_lm)

    mix = 1 / (4*π) * mix * Δr^2
    if !div2Lp1
        mix *= (2*L+1)
    end

    return mix
end


# specialize separable
function calc_cmixii(i, i′, cmodes, r, Δr, gnlgNLϕ, ang_mix::AbstractMatrix,
                     div2Lp1, interchange_NN′)
    l, n, n′ = getlnn(cmodes, i)
    L, N, N′ = getlnn(cmodes, i′)
    if interchange_NN′
        tmp = N′
        N′ = N
        N = tmp
    end
    #@show i,i′, n,n′, N,N′, l,l′

    #@show typeof(r) typeof(phi)
    gg1 = gnlgNLϕ[n,l+1,N,L+1]
    gg2 = gnlgNLϕ[n′,l+1,N′,L+1]

    m_ang = ang_mix[l+1,L+1]

    mix = m_ang * gg1 * gg2
    if !div2Lp1
        mix *= (2*L+1)
    end

    return mix
end


# calculate power spectrum mode-coupling matrix
function power_win_mix(win, wmodes::ConfigurationSpaceModes, cmodes::ClnnModes;
                       div2Lp1=false, interchange_NN′=false)
    amodes = cmodes.amodes
    lnnsize = getlnnsize(cmodes)
    #mix = fill(NaN, lnnsize, lnnsize)
    mix = SharedArray{Float64}(lnnsize, lnnsize)
    @show length(mix), size(mix)
    @show lnnsize^2, lnnsize

    r, Δr = window_r(wmodes)

    LMAX = 2 * amodes.lmax
    Wr_lm = optimize_Wr_lm_layout(calc_Wr_lm(win, LMAX, amodes.nside))

    # the gnl precomputation only saves about 10% time
    gnl = amodes.basisfunctions
    gnlr = fill(NaN, length(r), size(gnl.knl)...)
    @time for l=0:amodes.lmax, n=1:amodes.nmax_l[l+1]
        @. gnlr[:,n,l+1] = gnl(n,l,r)
    end
    check_nsamp(amodes, wmodes)

    L1M1cache = [hp.Alm.getidx.(LMAX, L, 0:L) .+ 1 for L=0:LMAX]


    ## crashes at end:
    #@time @threads for i′=1:lnnsize
    #    @show i′, lnnsize
    #    @time for i=1:lnnsize
    #        mix[i,i′] = calc_cmixii(i, i′, cmodes, r, Δr, gnlr, Wr_lm, L1M1cache)
    #        #@show i,i′, mix[i,i′]
    #    end
    #end

    ## too slow:
    #@time mix = pmap(idx -> calc_cmixii(idx[1], idx[2], cmodes, r, Δr, gnlr, Wr_lm, L1M1cache),
    #                 CartesianIndices((1:lnnsize, 1:lnnsize)))

    #@show "full"
    #@time @sync @distributed for i′=1:lnnsize
    #    @time for i=1:lnnsize
    #        mix[i,i′] = calc_cmixii(i, i′, cmodes, r, Δr, gnlr, Wr_lm, L1M1cache)
    #        #@show i,i′, mix[i,i′]
    #    end
    #end
    #mix1 = deepcopy(mix)

    #@show "symmetric"
    #@time @sync @distributed for i′=1:lnnsize
    #    L, = getlnn(cmodes, i′)
    #    @time for i=i′:lnnsize
    #        l, = getlnn(cmodes, i)
    #        mix[i,i′] = calc_cmixii(i, i′, cmodes, r, Δr, gnlr, Wr_lm, L1M1cache)
    #        mix[i′,i] = (2*l+1) / (2*L+1) * mix[i,i′]
    #        #@show i,i′, mix[i,i′]
    #    end
    #end
    #@assert mix == mix1

    #@show "symmetric pmap"
    @time pmap(i′ -> begin
                   L, = getlnn(cmodes, i′)
                   @show i′,L,lnnsize
                   @time for i=i′:lnnsize
                       l, = getlnn(cmodes, i)
                       mix[i,i′] = calc_cmixii(i, i′, cmodes, r, Δr, gnlr,
                                               Wr_lm, L1M1cache, div2Lp1,
                                               interchange_NN′)
                       mix[i′,i] = (2*l+1) / (2*L+1) * mix[i,i′]
                       @show i,i′, mix[i,i′]
                   end
                   return i′  # return something that doesn't take much memory
               end,
               1:lnnsize)
    #@assert mix == mix1
    @assert all(isfinite.(mix))
    return mix
end


nzind(vec::AbstractVector) = 1:length(vec)
nzind(vec::SparseVector) = vec.nzind

Base.getindex(::UniformScaling{T}, ::Colon, m::Integer) where {T} = sparsevec([m], [T(1)])
Base.getindex(::UniformScaling{T}, m::Integer, ::Colon) where {T} = sparsevec([m], [T(1)])


# binned cmix
function power_win_mix(w̃mat, vmat, r, Δr, gnlr, Wr_lm, L1M1cache, bcmodes;
                       div2Lp1=false, interchange_NN′=false)
    cmodes = bcmodes.cmodes
    lnnsize = getlnnsize(cmodes)
    LNNsize1 = (typeof(w̃mat) <: UniformScaling) ? lnnsize : size(w̃mat,1)
    LNNsize2 = (typeof(vmat) <: UniformScaling) ? lnnsize : size(vmat,2)
    mix = fill(NaN, LNNsize1, LNNsize2)
    #mix = SharedArray{Float64}(LNNsize, LNNsize)
    @show length(mix), size(mix)

    #@time @sync @distributed for m=1:LNNsize
    @time for m=1:LNNsize2
        @show m, LNNsize2
        vmat_m = vmat[:,m]
        vnzrange = nzind(vmat_m)
        #@show typeof(vmat_m) typeof(vnzrange) size(vmat) vnzrange length(vnzrange) vmat_m[vnzrange]
        @time for n=1:LNNsize1
            #@show m,n,LNNsize
            w̃mat_n = w̃mat[n,:]
            w̃nzrange = nzind(w̃mat_n)
            c = 0.0
            for i in w̃nzrange, i′ in vnzrange
                v = vmat_m[i′]
                v==0 && continue
                w̃ = w̃mat_n[i]
                w̃==0 && continue
                c += w̃ * v * calc_cmixii(i, i′, cmodes, r, Δr, gnlr, Wr_lm,
                                         L1M1cache, div2Lp1, interchange_NN′)
            end
            mix[n,m] = c
        end
    end
    return mix
end


function calc_angular_mixing_matrix(lmax, wlm)
    LMAX = 2 * lmax
    Wℓ = hp.alm2cl(wlm, lmax=LMAX)
    ang_mix = fill(NaN, lmax+1, lmax+1)
    for L=0:lmax, l=0:lmax
        s = 0.0
        for L1=abs(L-l):2:(L+l)
            wig = wigner3j000(l, L, L1)
            s += wig^2 * (2*L1+1) * Wℓ[L1+1]
        end
        ang_mix[l+1,L+1] = 1 / (4π) * s  # 2*L+1 will be included later, if not symmetric
    end
    return ang_mix
end


check_nsamp_1gnl(amodes, wmodes::ConfigurationSpaceModes) = check_nsamp_1gnl(amodes, wmodes.nr)
function check_nsamp_1gnl(amodes, nr)
    num_imprecise = 0
    max_Nsamp = 0
    lmax = amodes.lmax
    nmax_l = amodes.nmax_l
    for L=0:lmax, N=1:nmax_l[L+1]
        Nsamp = 8 * N  # + L
        max_Nsamp = max(max_Nsamp, Nsamp)
        if Nsamp > nr
            num_imprecise += 1
        end
    end
    if num_imprecise > 0
        @warn "Radial integral over one gnl(r) unlikely to converge" num_imprecise max_Nsamp nr amodes.rmin amodes.rmax amodes.lmax amodes.nmax amodes.nmax_l
        #throw(ErrorException("Nsamp > nr"))
    end
end


check_nsamp(amodes, wmodes::ConfigurationSpaceModes) = check_nsamp(amodes, wmodes.nr)
function check_nsamp(amodes, nr)
    num_imprecise = 0
    max_Nsamp = 0
    lmax = amodes.lmax
    nmax_l = amodes.nmax_l
    for L=0:lmax, N=1:nmax_l[L+1]
        for l=0:lmax, n=1:nmax_l[l+1]
            Nsamp = 8 * (n + N)  # + l + L
            max_Nsamp = max(max_Nsamp, Nsamp)
            if Nsamp > nr
                num_imprecise += 1
            end
        end
    end
    if num_imprecise > 0
        @warn "Radial integrals unlikely to converge" num_imprecise max_Nsamp nr amodes.rmin amodes.rmax amodes.lmax amodes.nmax amodes.nmax_l
        #throw(ErrorException("Nsamp > nr"))
    end
end


# calculate radial mixers
function calc_radial_mixing(lmax, nmax_l, gnlr, phi, r, Δr)
    nmax = maximum(nmax_l)
    gnlgNLϕ = fill(NaN, nmax, lmax+1, nmax, lmax+1)
    ggϕint = fill(NaN, length(phi))
    for L=0:lmax, N=1:nmax_l[L+1]
        for l=0:lmax, n=1:nmax_l[l+1]
            !isnan(gnlgNLϕ[n,l+1,N,L+1]) && continue
            @. ggϕint = r^2 * gnlr[:,n,l+1] * gnlr[:,N,L+1] * phi
            gg = Δr * sum(ggϕint)
            gnlgNLϕ[n,l+1,N,L+1] = gg
            gnlgNLϕ[N,L+1,n,l+1] = gg
        end
    end
    return gnlgNLϕ
end


# specialized for separable window
function power_win_mix(w̃mat, vmat, r, Δr, gnlr, Wr_lm::SeparableArray, L1M1cache, bcmodes;
                       div2Lp1=false, interchange_NN′)
    cmodes = bcmodes.cmodes
    lnnsize = getlnnsize(cmodes)
    LNNsize1 = (typeof(w̃mat) <: UniformScaling) ? lnnsize : size(w̃mat,1)
    LNNsize2 = (typeof(vmat) <: UniformScaling) ? lnnsize : size(vmat,2)
    mix = fill(NaN, LNNsize1, LNNsize2)
    @show length(mix), size(mix)

    println("Calculate angular and radial mixing:")
    check_nsamp(cmodes.amodes, length(gnlr[:,1,1]))
    lmax = bcmodes.cmodes.amodes.lmax
    nmax_l = bcmodes.cmodes.amodes.nmax_l
    @time ang_mix = calc_angular_mixing_matrix(lmax, Wr_lm.wlm)
    @time gnlgNLϕ = calc_radial_mixing(lmax, nmax_l, gnlr, Wr_lm.phi, r, Δr)

    println("Calculate binned mixing matrix:")
    #@time @sync @distributed for m=1:LNNsize
    @time for m=1:LNNsize2
        #@show m, LNNsize
        vmat_m = vmat[:,m]
        vnzrange = nzind(vmat_m)
        #@show typeof(vmat_m) typeof(vnzrange) size(vmat) vnzrange length(vnzrange) vmat_m[vnzrange]
        for n=1:LNNsize1
            #@show m,n,LNNsize
            w̃mat_n = w̃mat[n,:]
            w̃nzrange = nzind(w̃mat_n)
            c = 0.0
            for i in w̃nzrange, i′ in vnzrange
                v = vmat_m[i′]
                v==0 && continue
                w̃ = w̃mat_n[i]
                w̃==0 && continue
                c += w̃ * v * calc_cmixii(i, i′, cmodes, r, Δr, gnlgNLϕ,
                                         ang_mix, div2Lp1, interchange_NN′)
            end
            mix[n,m] = c
        end
    end
    return mix
end


# calculate binned power spectrum mode-coupling matrix
function power_win_mix(win, w̃mat, vmat, wmodes::ConfigurationSpaceModes, bcmodes::ClnnBinnedModes; div2Lp1=false, interchange_NN′=false)
    cmodes = bcmodes.cmodes
    amodes = cmodes.amodes
    lnnsize = getlnnsize(cmodes)
    LNNsize = getlnnsize(bcmodes)
    @show LNNsize^2, LNNsize, lnnsize

    r, Δr = window_r(wmodes)

    println("Calculate Wr_lm:")
    LMAX = 2 * amodes.lmax
    @time Wr_lm = optimize_Wr_lm_layout(calc_Wr_lm(win, LMAX, amodes.nside))

    println("Calculate gnlr:")
    gnl = amodes.basisfunctions
    gnlr = fill(NaN, length(r), size(gnl.knl)...)
    @time for l=0:amodes.lmax, n=1:amodes.nmax_l[l+1]
        @. gnlr[:,n,l+1] = gnl(n,l,r)
    end
    check_nsamp(amodes, wmodes)

    L1M1cache = [hp.Alm.getidx.(LMAX, L, 0:L) .+ 1 for L=0:LMAX]

    mix = power_win_mix(w̃mat, vmat, r, Δr, gnlr, Wr_lm, L1M1cache, bcmodes; div2Lp1=div2Lp1, interchange_NN′=interchange_NN′)

    @assert all(isfinite.(mix))
    return mix
end



end


# vim: set sw=4 et sts=4 :
