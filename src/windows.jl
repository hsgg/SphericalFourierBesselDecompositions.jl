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
export window_r, get_rbounds, apply_window, apodize_window
export win_rhat_ln, integrate_window, calc_wmix, power_win_mix, win_lnn
export get_wmix
export check_nsamp, calc_intr_gg_fn
export calc_fvol

using ..Modes
using ..SeparableArrays
using ..Splines
using ..MyBroadcast
using ..HealpixHelpers
using ..LMcalcStructs
using ..GNL
using Healpix
using LinearAlgebra
using SpecialFunctions
using WignerSymbols
using WignerFamilies
using SparseArrays
using Random
using LoopVectorization
#using FastGaussQuadrature

using Distributed
using SharedArrays
using ProgressMeter
using Base.Threads
using ThreadsX

const progressmeter_update_interval = haskey(ENV, "PROGRESSMETER_UPDATE_INTERVAL") ? parse(Int, ENV["PROGRESSMETER_UPDATE_INTERVAL"]) : 1

#using QuadGK  # for testing

#using Profile
#using PyPlot


#SparseArrays.rowvals(mat) = 1:size(mat,1)  # we also want to use it for full vectors


struct ConfigurationSpaceModes{Tarr,T<:Real}
    rmin::T
    rmax::T
    Δr::T
    r::Tarr
    nr::Integer
    npix::Integer
    nside::Integer
end

# for the @__dot syntax:
Base.length(w::ConfigurationSpaceModes) = 1
Base.iterate(w::ConfigurationSpaceModes) = w, nothing
Base.iterate(w::ConfigurationSpaceModes, x) = nothing


@doc raw"""
    ConfigurationSpaceModes(rmin, rmax, nr, nside)

A struct to describe and define the voxelization scheme.
"""
function ConfigurationSpaceModes(rmin, rmax, nr, nside)
    Δr = (rmax - rmin) / nr
    r = range(rmin+Δr/2, rmax-Δr/2, length=nr)  # midpoints
    npix = nside2npix(nside)
    return ConfigurationSpaceModes(rmin, rmax, Δr, r, nr, npix, nside)
end


@doc raw"""
    window_r(wmodes::ConfigurationSpaceModes)

Get the $r$-values of the radial bins and corresponding widths $\Delta r$, e.g.,
```julia
r, Δr = SFB.window_r(wmodes)
```
"""
window_r(wmodes::ConfigurationSpaceModes) = wmodes.r, wmodes.Δr


function get_rbounds(wmodes::ConfigurationSpaceModes)
    rmin = wmodes.rmin
    rmax = wmodes.rmax
    nr = wmodes.nr
    rbounds = range(rmin, rmax, length=nr+1)
    return rbounds
end


# only a basic implementation, with lots of edge cases poorly handled
function apodize_window(win, wmodes::ConfigurationSpaceModes, smooth=50.0)
    winapod = deepcopy(win)
    r, Δr = window_r(wmodes)
    nr = length(r)
    npix = size(win,2)
    nside = npix2nside(npix)
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

@doc raw"""
    apply_window(rθϕ, win, wmodes::ConfigurationSpaceModes; rng=Random.GLOBAL_RNG)
    apply_window(rθϕ::AbstractArray{T}, win, rmin, rmax, win_r, win_Δr; rng=Random.GLOBAL_RNG) where {T<:Real}

The function `apply_window()` takes a sample of points in `rθϕ` and filters out
points with probability specified by `1-win/maximum(win)`. Thus, all points are
retained where `win == maximum(win)`, and points are filtered out with
proportional probability so that none are kept where `win <= 0`.
"""
function apply_window(rθϕ::AbstractArray{T}, win, rmin, rmax, win_r, win_Δr; rng=Random.GLOBAL_RNG) where {T<:Real}
    Ngals = size(rθϕ, 2)
    npix = size(win, 2)
    nside = npix2nside(npix)
    reso = Resolution(nside)
    ooWmax = 1 / maximum(win)
    nr = length(win_r)
    #insample = Array{Bool}(undef, Ngals)
    #insample = BitArray(undef, Ngals)
    insample = ThreadsX.map(1:Ngals) do i
        r = rθϕ[1,i]
        if !(rmin <= r <= rmax)
            return false
        end
        θ = rθϕ[2,i]
        ϕ = rθϕ[3,i]

        idx_r = ceil(Int, (r - rmin) / win_Δr)
        if idx_r == 0  # if r == rmin
            idx_r = 1
        elseif idx_r > nr
            return false
        end

        idx_ang = ang2pixRing(reso, θ, ϕ)
        if !(1 <= idx_ang <= npix)
            @error "healpixel outside healpix map" reso nside θ ϕ idx_ang npix
        end

        if rand(rng) <= win[idx_r,idx_ang] * ooWmax
            return true
        end
        return false
    end
    return collect(rθϕ[:,insample])
end

apply_window(rθϕ, win, wmodes::ConfigurationSpaceModes; rng=Random.GLOBAL_RNG) = begin
    apply_window(rθϕ, win, wmodes.rmin, wmodes.rmax, wmodes.r, wmodes.Δr; rng)
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


function calc_fvol(win, wmodes::ConfigurationSpaceModes; Wthreshold=0.1)
    nr = size(win, 1)
    npix = size(win,2)
    r = wmodes.r
    Δr = wmodes.Δr
    ΔΩpix = 4*π / npix
    radial = [sum(w->w>Wthreshold, win[i,:]) for i=1:size(win,1)]
    V = Δr * ΔΩpix * sum(@. radial * r^2)
    Vsfb = (4*π/3) * (wmodes.rmax^3 - wmodes.rmin^3)
    return V / Vsfb
end


function win_rhat_ln(win, wmodes::ConfigurationSpaceModes, amodes::AnlmModes)
    gnl = amodes.basisfunctions
    r, Δr = window_r(wmodes)
    W_rhat_ln = fill(eltype(win)(NaN), size(win,2), amodes.lmax+1, amodes.nmax)
    check_nsamp_1gnl(amodes, wmodes)
    for n=1:amodes.nmax, l=0:amodes.lmax_n[n]
        l==0 && @show n,l
        int_nowin = @. r^2 * gnl(n, l, r)
        W_rhat_ln[:,l+1,n] .= win'int_nowin
    end
    @. W_rhat_ln *= Δr
    #@assert all(isfinite.(W_rhat_ln))  # not all needs to be finite if we limit by kmax
    return W_rhat_ln
end

# specialize on separable windows
function win_rhat_ln(win::SeparableArray, wmodes::ConfigurationSpaceModes, amodes::AnlmModes)
    gnl = amodes.basisfunctions
    r, Δr = window_r(wmodes)
    W_ln = fill(eltype(win)(NaN), amodes.lmax+1, amodes.nmax)
    check_nsamp_1gnl(amodes, wmodes)
    for n=1:amodes.nmax, l=0:amodes.lmax_n[n]
        l==0 && @show n,l,amodes.nmax
        W_ln[l+1,n] = Δr * sum(@. r^2 * gnl(n, l, r) * win.phi)
    end
    return SeparableArray(win.mask, W_ln, name1=:mask, name2=:w_ln)
end


function calc_wmix_ii(l, m, l′, m′, gg1, Wr_lm, LMLM; buffer1=zeros(0), buffer2=zeros(0))
    M = m - m′

    gaunt, L = calc_gaunts_L(l, l′, -m, m′; buffer1, buffer2)  # 4 allocations

    aM = abs(M)

    w_ang = 0.0im
    @views for j=1:length(L)
        LM = LMLM[L[j]+1,aM+1]
        w_ang += gaunt[j] * (gg1'Wr_lm[:,LM])  # this takes time to compute
        #@debug "Wr_lm" L[j],M Wr_lm[1,LM] Wr_lm[1,LM]/√(4*π)
    end

    if M < 0
        w_ang = (-1)^M * conj(w_ang)
    end
    #if l==1 && m==-1 && l′==0
    #    @debug "wmix" l,m l′,m′ w_ang gg1Wrlm
    #end
    return (-1)^m * w_ang
end


# This should be very performant
# Also checkout calc_wmix_all() in window_chains.jl.
function calc_wmix(win, wmodes::ConfigurationSpaceModes, amodes::AnlmModes; neg_m=false)
    T = ComplexF64

    nlmodes = ClnnModes(amodes, Δnmax=0)  # to make it easy to iterate over l,n
    nlsize = getlnnsize(nlmodes)

    nlmsize = getnlmsize(amodes)
    lmax = amodes.lmax
    wmix = fill(NaN*im, nlmsize, nlmsize)
    @show length(wmix), size(wmix), nlsize

    println("Calculate Wr_lm:")
    LMAX = 2 * amodes.lmax
    #Wr_lm, LMLM = optimize_Wr_lm_layout(calc_Wr_lm(win, LMAX, amodes.nside), LMAX)
    Wr_lm, LMLM = calc_Wr_lm(win, LMAX, amodes.nside), LMcalcStruct(LMAX)
    #@debug "Wr_lm" LMAX amodes.nside size(Wr_lm) Wr_lm[:,1]

    println("Calculate gnlr:")
    @time gnlr = precompute_gnlr(amodes, wmodes)
    r, Δr = window_r(wmodes)
    nr = Int64(wmodes.nr)
    @time @. gnlr *= r * √Δr  # part of the integral measure


    println("Starting wmix calculation:")
    p = Progress(nlsize^2, progressmeter_update_interval, "wmix full: ")
    @time mybroadcast(1:nlsize, (1:nlsize)') do nlarr, n′l′arr
        gg1 = Vector{real(T)}(undef, nr)
        wtmp = Vector{T}(undef, (lmax+1)^2)
        buffer1 = Vector{real(T)}(undef, 0)
        buffer2 = Vector{real(T)}(undef, 0)

        for i=1:length(nlarr)
            l, n, _ = getlnn(nlmodes, nlarr[i])
            l′, n′, _ = getlnn(nlmodes, n′l′arr[i])

            ibase = getidx(amodes, n, l, 0)
            i′base = getidx(amodes, n′, l′, 0)

            @views @. gg1 = gnlr[:,n,l+1] * gnlr[:,n′,l′+1]

            ll = 1:((l+1)*(l′+1))
            w = @views reshape(wtmp[ll], l+1, l′+1)

            for m=0:l, m′=0:l′
                am = m
                if neg_m
                    m = -m
                end
                w[am+1,m′+1] = calc_wmix_ii(l, m, l′, m′, gg1, Wr_lm, LMLM; buffer1, buffer2)
            end

            mm = ibase .+ (0:l)
            mm′ = i′base .+ (0:l′)
            @views @. wmix[mm,mm′] = w
        end
        next!(p, step=length(nlarr), showvalues=[(:batchsize, length(nlarr))])
        return zero(real(T))  # must return something broadcastable for mybroadcast()
    end
    #@assert all(isfinite, wmix)
    return wmix
end


function get_wmix(w, w′, nl, m, NL, M)
    if m >= 0
        if M >= 0
            return w[nl+m, NL+M]
        end
        return (-1)^(m+M) * conj(w′[nl+m, NL-M])
    end
    if M >= 0
        return w′[nl-m, NL+M]
    end
    return (-1)^(m-M) * conj(w[nl-m, NL-M])
end


# This should be very performant
function win_lnn(win, wmodes::ConfigurationSpaceModes, cmodes::ClnnModes)
    println("Calculate Wr_00:")
    # Note: the maximum ℓ we need here is 0. However, healpy changes precision,
    # and for comparison we use the same lmax as elsewhere.
    @time Wr_00 = Vector{Float64}(calc_Wr_lm(win, 2*cmodes.amodes.lmax, cmodes.amodes.nside)[:,1])
    @assert all(isfinite, Wr_00)

    r, Δr = window_r(wmodes)
    Wlnn = calc_intr_gg_fn(Spline1D(r, Wr_00 / √(4π)), wmodes, cmodes; derivative=0)
    return Wlnn
end


function calc_intr_gg_fn(func, wmodes::ConfigurationSpaceModes, cmodes::ClnnModes; derivative=0)
    check_nsamp(cmodes.amodes, wmodes)

    println("Calculate gnlr:")
    @time gnlr, nodes, weights = precompute_gnlr_nodes_weights(cmodes.amodes, wmodes; derivative)
    fn = func.(nodes)
    @time @. gnlr *= nodes * √weights * √fn  # add part of the integral measure and Wr_00

    println("Calculate lnn integral:")
    lnnsize = getlnnsize(cmodes)
    @time Wlnn = mybroadcast(1:lnnsize) do ii
        out = Vector{Float64}(undef, length(ii))
        for i=1:length(ii)
            l, n, n′ = getlnn(cmodes, ii[i])
            #@show i, l,n,n′, lnnsize

            @views sgg = gnlr[:,n,l+1]' * gnlr[:,n′,l+1]

            out[i] = sgg
        end
        return out
    end
    @assert all(isfinite, Wlnn)
    return Wlnn
end


function wigner3j000(l, l′, L)::Float64
    (abs(l-l′) <= L <= l+l′) || return 0.0
    J = l + l′ + L
    iseven(J) || return 0.0
    wig3j = (-1)^(J÷2) * exp(0.5*loggamma(1+J-2l) + 0.5*loggamma(1+J-2l′)
                             + 0.5*loggamma(1+J-2L) - 0.5*loggamma(1+J+1)
                             + loggamma(1+J÷2)
                             - loggamma(1+J÷2-l) - loggamma(1+J÷2-l′)
                             - loggamma(1+J÷2-L))
    return wig3j
end


function calc_w3j_f(l, l′, m, m′, buffer::AbstractVector{T}) where {T<:Real}
    w = WignerF(T, l, l′, m, m′)

    buflen = length(w.nₘᵢₙ:w.nₘₐₓ)
    if length(buffer) < buflen
        resize!(buffer, buflen)
    end
    bufferview = @view buffer[1:buflen]

    w3j_f = WignerSymbolVector(bufferview, w.nₘᵢₙ:w.nₘₐₓ)

    wigner3j_f!(w, w3j_f)  # allocation happening here. why?

    return w3j_f
end


# calc_gaunts_L(): Will modify both buffers, and use one of them for the returned array.
function calc_gaunts_L(l, l′, m, m′; buffer1=zeros(0), buffer2=zeros(0))
    T = Float64
    w3j = calc_w3j_f(l, l′, m, m′, buffer1)
    w3j000 = calc_w3j_f(l, l′, 0, 0, buffer2)
    #w3j000 = @. wigner3j000(l, l′, L)

    L = eachindex(w3j)
    gaunt = w3j.symbols
    @views @. gaunt *= √((2*L+1) * (2*l+1) * (2*l′+1) / (4*T(π))) * w3j000[L]
    #gaunt = @. √((2*L+1) * (2*l+1) * (2*l′+1) / (4*π)) * w3j000 * w3j

    return gaunt, L
end


function power_win_mix_ii(lnn, LNN, wmix, wmix_negm, amodes)
    l, n, n′ = lnn
    L, N, N′ = LNN
    mix = 0.0
    for m=0:l, M=0:L
        j = getidx(amodes, n, l, m)
        j′ = getidx(amodes, n′, l, m)
        J = getidx(amodes, N, L, M)
        J′ = getidx(amodes, N′, L, M)

        j0 = getidx(amodes, n, l, 0)
        j′0 = getidx(amodes, n′, l, 0)
        J0 = getidx(amodes, N, L, 0)
        J′0 = getidx(amodes, N′, L, 0)

        mix += real(wmix[j,J] * conj(wmix[j′,J′]))

        if m>0
            w1 = get_wmix(wmix, wmix_negm, j0, -m, J0, M)
            w2 = get_wmix(wmix, wmix_negm, j′0, -m, J′0, M)
            mix += real(w1 * conj(w2))

            #mix += real(wmix_negm[j,J] * conj(wmix_negm[j′,J′]))
        end

        if M>0
            w1 = get_wmix(wmix, wmix_negm, j0, m, J0, -M)
            w2 = get_wmix(wmix, wmix_negm, j′0, m, J′0, -M)
            mix += real(w1 * conj(w2))

            #mix += real(conj(wmix_negm[J,j]) * wmix_negm[J′,j′])
        end

        if m>0 && M>0
            w1 = get_wmix(wmix, wmix_negm, j0, -m, J0, -M)
            w2 = get_wmix(wmix, wmix_negm, j′0, -m, J′0, -M)
            mix += real(w1 * conj(w2))

            #mix += real(conj(wmix[j,J]) * wmix[j′,J′])
        end
    end
    return mix / (2*l + 1)
end


function power_win_mix(wmix, wmix_negm, cmodes)
    amodes = cmodes.amodes
    lnnsize = getlnnsize(cmodes)
    mmix = fill(NaN, lnnsize, lnnsize)
    for i′=1:lnnsize, i=1:lnnsize
        l, n, n′ = getlnn(cmodes, i)
        L, N, N′ = getlnn(cmodes, i′)
        mmix[i,i′] = power_win_mix_ii((l,n,n′), (L,N,N′), wmix, wmix_negm, amodes)
        if N != N′
            mmix[i,i′] += power_win_mix_ii((l,n,n′), (L,N′,N), wmix, wmix_negm, amodes)
        end
    end
    return mmix
end


function calc_Wr_lm(win, LMAX, Wnside)
    nr = size(win,1)
    Wr_lm = fill(NaN*im, nr, getlmsize(LMAX))
    @time for i=1:nr
        #@show i,nr
        W = udgrade(win[i,:], Wnside)
        Wr_lm[i,:] .= mymap2alm(W, lmax=LMAX).alm
    end
    return Wr_lm
end

# specialize
function calc_Wr_lm(win::SeparableArray, LMAX, Wnside)
    mask = udgrade(win.mask, Wnside)
    wlm = mymap2alm(mask, lmax=LMAX)
    @assert ndims(win.phi) == ndims(wlm) == 1
    return SeparableArray(win.phi, wlm, name1=:phi, name2=:wlm)
end


function precompute_gnlr(amodes, wmodes)
    r, Δr = window_r(wmodes)
    gnl = amodes.basisfunctions
    gnlr = fill(NaN, length(r), size(gnl.knl)...)
    Threads.@threads for l=0:amodes.lmax
        for n=1:amodes.nmax_l[l+1]
            @views @. gnlr[:,n,l+1] = gnl(n,l,r)
        end
    end
    check_nsamp(amodes, wmodes)
    return gnlr
end


function precompute_gnlr_nodes_weights(amodes, wmodes; derivative=0)
    nr = wmodes.nr
    rmin = wmodes.rmin
    rmax = wmodes.rmax

    # Trapezoidal nodes and weights
    nodes, weights = window_r(wmodes)

    ## Gauss-Legendre nodes and weights
    #nodes, weights = gausslegendre(nr)
    #@. weights *= (rmax - rmin) / 2
    #@. nodes = (rmin + rmax) / 2 + (rmax - rmin) / 2 * nodes

    gnl = (derivative == 1) ? (n,l,x)->GNL.Splines.derivative(amodes.basisfunctions,n,l,x) : amodes.basisfunctions
    gnlr = fill(NaN, nr, size(amodes.basisfunctions.knl)...)
    #Threads.@threads
    for l=0:amodes.lmax
        for n=1:amodes.nmax_l[l+1]
            @views @. gnlr[:,n,l+1] = gnl(n,l,nodes)
        end
    end
    check_nsamp(amodes, wmodes)
    return gnlr, nodes, weights
end



function optimize_Wr_lm_layout(Wr_lm, LMAX)
    L1M1cache_in = LMcalcStruct(LMAX)
    #return Wr_lm, L1M1cache_in

    L1M1cache_out = LMcalcStructMfast()
    Wr_lm_out = Array{eltype(Wr_lm)}(undef, size(Wr_lm))

    for l=0:LMAX
        for m=0:l
            lm_in = L1M1cache_in[l+1,m+1]
            lm_out = L1M1cache_out[l+1,m+1]
            Wr_lm_out[:,lm_out] .= Wr_lm[:,lm_in]
        end
    end

    return Wr_lm_out, L1M1cache_out
end

function optimize_Wr_lm_layout(Wr_lm::SeparableArray, LMAX)
    L1M1cache = LMcalcStruct(LMAX)
    return Wr_lm, L1M1cache  # noop for SeparableArray
end


function cmix_kernel(gg1, gg2, w1r, w2r)
    real(dot(gg1, w1r) * conj(dot(gg2, w2r)))
end


function test_cmix_kernel()
    nr = 100
    gg1 = rand(nr)
    gg2 = rand(nr)
    w1r = rand(Complex{Float64}, nr)
    w2r = rand(Complex{Float64}, nr)
    l = 20
    L = 20
    LMAX = l + L
    Wnside = estimate_nside(LMAX)
    win = rand(nr, nside2npix(Wnside))
    @show typeof(win)
    Wr_lm, L1M1cache = optimize_Wr_lm_layout(calc_Wr_lm(win, LMAX, Wnside), LMAX)
    @show typeof(Wr_lm)

    # compile
    cmix_kernel(gg1, gg2, w1r, w2r)
    calc_cmix_ang(l, L, L1M1cache, gg1, gg2, Wr_lm, Wr_lm)

    s = 0.0
    @time cmix_kernel(gg1, gg2, wr, wr)
    @time cmix_kernel(gg1, gg2, Wr_lm[:,1], Wr_lm[:,1])

    @show typeof(l) typeof(L) typeof(L1M1cache) typeof(gg1) typeof(gg2) typeof(Wr_lm)

    @time calc_cmix_ang(l, L, L1M1cache, gg1, gg2, Wr_lm, Wr_lm)
    @time m_ang = calc_cmix_ang(l, L, L1M1cache, gg1, gg2, Wr_lm, Wr_lm)
    @show typeof(m_ang)
end


function calc_cmix_ang(l, L, L1M1cache, gg1, gg2, W1r_lm, W2r_lm)
    #@debug "calc_cmix_ang" l L size(gg1) size(gg2) size(Wr_lm) size(L1M1cache) size(Wr_lm[1])
    m_ang = 0.0
    @views for L1=abs(l-L):2:(l+L)
        L1M1 = L1M1cache[L1+1,1]
        #@show L1,L1M1
        s = cmix_kernel(gg1, gg2, W1r_lm[:,L1M1], W2r_lm[:,L1M1])
        for M1=1:L1
            #@debug "" L1 M1
            L1M1 = L1M1cache[L1+1,M1+1]
            s += 2 * cmix_kernel(gg1, gg2, W1r_lm[:,L1M1], W2r_lm[:,L1M1])
        end
        m_ang += s * wigner3j000(l, L, L1)^2
    end
    return m_ang
end


function calc_cmixii(i, L, N, N′, rsdrgnlr, cmodes::ClnnModes, W1r_lm, W2r_lm, L1M1cache, div2Lp1, gg1, gg2)
    l, n, n′ = getlnn(cmodes, i)
    #L, N, N′ = getlnn(cmodes, i′)

    #showthis = ((l==L==0 && n==N′==1 && n′==N==2) || (l==L==0 && n==N′==2 && n′==N==1))
    #showthis && @show "huzzah",i,i′, n,n′, N,N′, l,L
    #@show i,i′, (l,n,n′), (L,N,N′)

    @views @. gg1 = rsdrgnlr[:,n,l+1] * rsdrgnlr[:,N,L+1]
    @views @. gg2 = rsdrgnlr[:,n′,l+1] * rsdrgnlr[:,N′,L+1]

    mix = calc_cmix_ang(l, L, L1M1cache, gg1, gg2, W1r_lm, W2r_lm)

    mix = mix / (4*π)
    if !div2Lp1
        mix *= (2*L+1)
    end

    return mix
end


# for backward compatiblity
function calc_cmixii_old(i, i′, cmodes::ClnnModes, rsdrgnlr, W1r_lm, W2r_lm, L1M1cache, div2Lp1, interchange_NN′, gg1, gg2)
    L, N, N′ = getlnn(cmodes, i′)
    if interchange_NN′
        N, N′ = N′, N
    end

    mix = calc_cmixii(i, L, N, N′, rsdrgnlr, cmodes, W1r_lm, W2r_lm, L1M1cache, div2Lp1, gg1, gg2)

    if !interchange_NN′ && N != N′
        mix += calc_cmixii(i, L, N′, N, rsdrgnlr, cmodes, W1r_lm, W2r_lm, L1M1cache, div2Lp1, gg1, gg2)
    end

    return mix
end


# specialize separable window
function calc_cmixii_separable(i, i′, cmodes, gnlgNLϕ1, gnlgNLϕ2, ang_mix::AbstractMatrix,
                     div2Lp1, interchange_NN′)
    l, n, n′ = getlnn(cmodes, i)
    L, N, N′ = getlnn(cmodes, i′)
    if interchange_NN′
        N, N′ = N′, N
    end
    #@show i,i′, n,n′, N,N′, l,l′

    #@show typeof(r) typeof(phi)
    gg1 = gnlgNLϕ1[n,l+1,N,L+1]
    gg2 = gnlgNLϕ2[n′,l+1,N′,L+1]

    m_ang = ang_mix[l+1,L+1]

    mix = m_ang * gg1 * gg2

    if !interchange_NN′ && N != N′
        gg1 = gnlgNLϕ1[n,l+1,N′,L+1]
        gg2 = gnlgNLϕ2[n′,l+1,N,L+1]
        mix += m_ang * gg1 * gg2
    end

    if !div2Lp1
        mix *= (2*L+1)
    end

    return mix
end


function calc_cmix(cmodes, rsdrgnlr, W1r_lm, W2r_lm, L1M1cache, div2Lp1, interchange_NN′)
    println("cmix full:")
    lnnsize = getlnnsize(cmodes)

    p = Progress(lnnsize^2, progressmeter_update_interval, "cmix full: ")

    #@time for i′=1:lnnsize
    #@time Threads.@threads for i′=1:lnnsize
    #@time @tturbo for i′=1:lnnsize
    mix = @time mybroadcast(1:lnnsize, (1:lnnsize)') do ii,ii′
        gg1 = Array{Float64}(undef, size(rsdrgnlr,1))
        gg2 = Array{Float64}(undef, size(rsdrgnlr,1))
        mout = Array{Float64}(undef, length(ii′))

        for idx=1:length(ii′)
            i = ii[idx]
            i′ = ii′[idx]

            L, N, N′ = getlnn(cmodes, i′)
            if interchange_NN′
                N, N′ = N′, N
            end

            mixii′ = calc_cmixii(i, L, N, N′, rsdrgnlr, cmodes,
                                        W1r_lm, W2r_lm, L1M1cache, div2Lp1, gg1, gg2)
            #l, n, n′ = getlnn(cmodes, i)
            #@show i,i′,(l,n,n′),(L,N,N′),mixii′

            if (!interchange_NN′) && (N != N′)
                # Since we only save the symmetric part where N′ >= N
                mixii′ += calc_cmixii(i, L, N′, N, rsdrgnlr, cmodes,
                                             W1r_lm, W2r_lm, L1M1cache, div2Lp1, gg1, gg2)
                #@show mixii′
            end

            mout[idx] = mixii′
        end
        next!(p, step=length(ii′), showvalues=[(:batchsize, length(ii′)), (:counter, p.counter)])
        return mout
    end

    return mix
end


# Should probably remove these, because they are unique more by chance than by design
power_win_mix(win, wmodes::ConfigurationSpaceModes, cmodes::ClnnModes; kwargs...) = power_win_mix(win, win, wmodes, cmodes; kwargs...)
power_win_mix(win, w̃, v, wmodes::ConfigurationSpaceModes, bcmodes::ClnnBinnedModes; kwargs...) = power_win_mix(win, win, w̃, v, wmodes, bcmodes; kwargs...)


@doc raw"""
    power_win_mix(win1, win2, wmodes, cmodes; div2Lp1=false, interchange_NN′=false)
    power_win_mix(win1, win2, w̃mat, vmat, wmodes, bcmodes; div2Lp1=false, interchange_NN′=false)
    power_win_mix(wmix, wmix_negm, cmodes)

This function is used to calculate the coupling matrix $\mathcal{M}_{\ell nn'}^{LNN'}$
-- the first version without any binning, the second version also takes the
binning matrices `w̃mat` and `vmat` to calculate the coupling matrix of the
binned modes $\mathcal{N}_{LNN'}^{lnn'}$. These assume the symmetry between $N$
and $N'$.

The last version is probably not useful except for testing. It takes a fully
calculated window mixing matrix to calculate the coupling matrix brute-force.

If `div2Lp1=true` then the whole matrix is divided by $2L+1$.

If `interchange_NN′=true` then calculate the same, but with $N$ and $N'$
interchanged, which might be useful for the covariance matrix.

Either version of `power_win_mix()` will specialize to a separable window
function if `win` is a `SeparableArray`.

The basic usage is to multiply the power spectrum Clnn by this matrix, and the
assuption is that there is symmetry in the exchange of `k_n` and `k_n′`. (Note
that this assumed symmetry, however, destroyes the symmetry in the coupling
matrix.)
"""
function power_win_mix(win1, win2, wmodes::ConfigurationSpaceModes, cmodes::ClnnModes;
                       div2Lp1=false, interchange_NN′=false)
    amodes = cmodes.amodes
    lnnsize = getlnnsize(cmodes)
    mix = fill(NaN, lnnsize, lnnsize)
    #mix = SharedArray{Float64}(lnnsize, lnnsize)
    @show length(mix), size(mix)
    @show lnnsize^2, lnnsize

    r, Δr = window_r(wmodes)

    LMAX = 2 * amodes.lmax
    W1r_lm, L1M1cache = optimize_Wr_lm_layout(calc_Wr_lm(win1, LMAX, amodes.nside), LMAX)
    W2r_lm, L1M1cache = W1r_lm, L1M1cache
    if !(win2 === win1)
        W2r_lm, L1M1cache = optimize_Wr_lm_layout(calc_Wr_lm(win2, LMAX, amodes.nside), LMAX)
    end

    println("Calculate r*√dr*gnlr:")
    @time rsdrgnlr = r .* .√Δr .* precompute_gnlr(amodes, wmodes)

    mix = calc_cmix(cmodes, rsdrgnlr, W1r_lm, W2r_lm, L1M1cache, div2Lp1, interchange_NN′)

    @assert all(isfinite.(mix))
    return mix
end


# specialize to Separable window
function power_win_mix(win1::SeparableArray, win2::SeparableArray, wmodes::ConfigurationSpaceModes,
        cmodes::ClnnModes; kwargs...)
    # Rather than writing a new specialized method, re-use what we already have:
    bcmodes = ClnnBinnedModes(I, I, cmodes)
    return power_win_mix(win1, win2, I, I, wmodes, bcmodes; kwargs...)
end


nzind(vec::AbstractVector) = 1:length(vec)
nzind(vec::SparseVector) = vec.nzind

Base.getindex(::UniformScaling{T}, ::Colon, m::Integer) where {T} = sparsevec([m], [T(1)])
Base.getindex(::UniformScaling{T}, m::Integer, ::Colon) where {T} = sparsevec([m], [T(1)])


# binned cmix
function _power_win_mix(w̃mat, vmat, rsdrgnlr, W1r_lm, W2r_lm, L1M1cache, bcmodes;
                       div2Lp1=false, interchange_NN′=false)
    cmodes = bcmodes.cmodes
    lnnsize = getlnnsize(cmodes)
    LNNsize1 = (typeof(w̃mat) <: UniformScaling) ? lnnsize : size(w̃mat,1)
    LNNsize2 = (typeof(vmat) <: UniformScaling) ? lnnsize : size(vmat,2)

    # Use pmap() to allow distributed parallel computing:
    n_idxs = SeparableArray(LNNsize1:-1:1, ones(Int, LNNsize2))
    m_idxs = SeparableArray(ones(Int, LNNsize1), LNNsize2:-1:1)
    batchsize = (LNNsize1 * LNNsize2) ÷ (nworkers()^2) + 1
    @show LNNsize1, LNNsize2, batchsize
    gg1 = Array{Float64}(undef, size(rsdrgnlr,1))
    gg2 = Array{Float64}(undef, size(rsdrgnlr,1))
    mix = @showprogress progressmeter_update_interval "cmix: " pmap((n,m) -> begin
            w̃mat_n = w̃mat[n,:]
            vmat_m = vmat[:,m]
            w̃nzrange = nzind(w̃mat_n)
            vnzrange = nzind(vmat_m)
            c = 0.0
            for i in w̃nzrange, i′ in vnzrange
                v = vmat_m[i′]
                v==0 && continue
                w̃ = w̃mat_n[i]
                w̃==0 && continue
                c += w̃ * v * calc_cmixii_old(i, i′, cmodes, rsdrgnlr, W1r_lm, W2r_lm,
                                         L1M1cache, div2Lp1, interchange_NN′, gg1, gg2)
            end
            return c
        end,
        n_idxs,
        m_idxs,
        batch_size=batchsize,
    )

    return collect(mix[end:-1:1,end:-1:1])
end



function calc_angular_mixing_matrix(lmax, w1lm, w2lm)
    Wℓ = alm2cl(w1lm, w2lm)
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
    max_nr_needed = 0
    lmax = amodes.lmax
    nmax_l = amodes.nmax_l
    for L=0:lmax, N=1:nmax_l[L+1]
        nr_needed = 8 * N
        max_nr_needed = max(max_nr_needed, nr_needed)
        if nr_needed > nr
            num_imprecise += 1
        end
    end
    if num_imprecise > 0
        @warn "Radial integral over one gnl(r) unlikely to converge" num_imprecise max_nr_needed nr amodes.rmin amodes.rmax amodes.lmax amodes.nmax amodes.nmax_l
        #throw(ErrorException("nr_needed > nr"))
    end
end


check_nsamp(amodes, wmodes::ConfigurationSpaceModes) = check_nsamp(amodes, wmodes.nr)
function check_nsamp(amodes, nr)
    num_imprecise = 0
    max_nr_needed = 0
    lmax = amodes.lmax
    nmax_l = amodes.nmax_l
    for L=0:lmax, N=1:nmax_l[L+1]
        for l=0:lmax, n=1:nmax_l[l+1]
            nr_needed = 8 * (n + N)
            max_nr_needed = max(max_nr_needed, nr_needed)
            if nr_needed > nr
                num_imprecise += 1
            end
        end
    end
    if num_imprecise > 0
        @warn "Radial integrals unlikely to converge" num_imprecise max_nr_needed nr amodes.rmin amodes.rmax amodes.lmax amodes.nmax amodes.nmax_l
        #throw(ErrorException("nr_needed > nr"))
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
function _power_win_mix(w̃mat, vmat, rsdrgnlr, W1r_lm::SeparableArray, W2r_lm::SeparableArray, L1M1cache, bcmodes;
                       div2Lp1=false, interchange_NN′=false)
    cmodes = bcmodes.cmodes
    lnnsize = getlnnsize(cmodes)
    LNNsize1 = (typeof(w̃mat) <: UniformScaling) ? lnnsize : size(w̃mat,1)
    LNNsize2 = (typeof(vmat) <: UniformScaling) ? lnnsize : size(vmat,2)
    mix = fill(NaN, LNNsize1, LNNsize2)
    @show length(mix), size(mix)

    println("Calculate angular and radial mixing:")
    check_nsamp(cmodes.amodes, length(rsdrgnlr[:,1,1]))
    lmax = bcmodes.cmodes.amodes.lmax
    nmax_l = bcmodes.cmodes.amodes.nmax_l
    @time ang_mix = calc_angular_mixing_matrix(lmax, W1r_lm.wlm, W2r_lm.wlm)
    @time gnlgNLϕ1 = calc_radial_mixing(lmax, nmax_l, rsdrgnlr, W1r_lm.phi, 1, 1)
    @time gnlgNLϕ2 = calc_radial_mixing(lmax, nmax_l, rsdrgnlr, W2r_lm.phi, 1, 1)

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
                c += w̃ * v * calc_cmixii_separable(i, i′, cmodes, gnlgNLϕ1, gnlgNLϕ2,
                                         ang_mix, div2Lp1, interchange_NN′)
            end
            mix[n,m] = c
        end
    end
    return mix
end


# calculate binned power spectrum mode-coupling matrix
function power_win_mix(win1, win2, w̃mat, vmat, wmodes::ConfigurationSpaceModes, bcmodes::ClnnBinnedModes; kwargs...)
    cmodes = bcmodes.cmodes
    amodes = cmodes.amodes
    lnnsize = getlnnsize(cmodes)
    LNNsize = getlnnsize(bcmodes)
    @show LNNsize^2, LNNsize, lnnsize

    r, Δr = window_r(wmodes)

    println("Calculate Wr_lm:")
    LMAX = 2 * amodes.lmax
    @time W1r_lm, L1M1cache = optimize_Wr_lm_layout(calc_Wr_lm(win1, LMAX, amodes.nside), LMAX)
    @time W2r_lm, L1M1cache = optimize_Wr_lm_layout(calc_Wr_lm(win1, LMAX, amodes.nside), LMAX)

    println("Calculate r*√dr*gnlr:")
    @time rsdrgnlr = r .* .√Δr .* precompute_gnlr(amodes, wmodes)

    mix = _power_win_mix(w̃mat, vmat, rsdrgnlr, W1r_lm, W2r_lm, L1M1cache, bcmodes; kwargs...)

    @assert all(isfinite.(mix))
    return mix
end



end


# vim: set sw=4 et sts=4 :
