@doc raw"""
    MyBroadcast

This module defines functions `mybroadcast` and `mybroadcast2d`.
"""
module MyBroadcast

export mybroadcast, mybroadcast2d

using Base.Threads


include("MeshedArrays.jl")
using .MeshedArrays


function calc_i_per_thread(time, i_per_thread_old; batch_avgtime=1.1, batch_maxadjust=2.0)
    adjust = batch_avgtime / time  # if we have accurate measurement of time
    adjust = min(batch_maxadjust, adjust)  # limit upward adjustment
    adjust = max(1/batch_maxadjust, adjust)  # limit downward adjustment

    if adjust < 1
        i_per_thread_new = floor(Int, adjust * i_per_thread_old)
    else
        i_per_thread_new = ceil(Int, adjust * i_per_thread_old)
    end
    i_per_thread_new = max(1, i_per_thread_new)  # must be at least 1

    return i_per_thread_new
end


function mybroadcast!(out, fn, arr)
    ntasks = length(arr)

    ifirst = 1
    i_per_thread = Atomic{Int}(1)
    last_ifirst = Atomic{Int}(0)  # doesn't need to be atomic
    lk = Threads.Condition()

    @sync while ifirst <= ntasks
        iset = ifirst:min(ntasks, ifirst + i_per_thread[] - 1)
        #@show ifirst,i_per_thread[]

        @spawn begin
            time = @elapsed begin
                idxs = eachindex(out, arr)[iset]
                out[idxs] .= fn(arr[idxs])
            end

            i_per_thread_new = calc_i_per_thread(time, length(iset))
            lock(lk) do
                if last_ifirst[] < iset[1]
                    i_per_thread[] = i_per_thread_new
                    last_ifirst[] = iset[1]
                end
            end
        end

        ifirst = iset[end] + 1
        yield()  # let some threads finish so that i_per_thread[] gets updated asap
    end

    return out
end


function mybroadcast(fn, arr)
    Treturn = Base.return_types(fn, (eltype(arr),))[1]
    out = similar(arr, Treturn)
    mybroadcast!(out, fn, arr)
    return out
end


function calc_outsize(x, y)
    outsize = fill(1, max(ndims(x), ndims(y)))
    outsize[1:ndims(x)] .= size(x)
    for d=1:ndims(y)
        if outsize[d] == 1
            outsize[d] = size(y, d)
        elseif size(y, d) != 1 && outsize[d] != size(y, d)
            error("size(x) = $(size(x)) and size(y) = $(size(y)) cannot be broadcast")
        end
    end
    return (outsize...,)
end


function mybroadcast2d!(out, fn, x, y)
    ntasks = prod(calc_outsize(x, y))
    @assert size(out) == calc_outsize(x, y)

    ifirst = 1
    i_per_thread = Atomic{Int}(1)
    last_ifirst = Atomic{Int}(0)  # doesn't need to be atomic
    lk = Threads.Condition()

    all_indices = eachindex(out, x, y)

    @sync while ifirst <= ntasks
        ilast = min(ntasks, ifirst + i_per_thread[] - 1)
        iset = ifirst:ilast  # no need to interpolate local variables

        Threads.@spawn begin
            t0 = time_ns()

            idxs = all_indices[iset]
            xs = x[idxs]
            ys = y[idxs]
            outs = fn(xs, ys)
            out[idxs] .= outs

            time = (time_ns() - t0) / 1e9

            i_per_thread_new = calc_i_per_thread(time, length(iset))
            lock(lk) do
                if last_ifirst[] < iset[1]
                    i_per_thread[] = i_per_thread_new
                    last_ifirst[] = iset[1]
                end
            end
        end

        ifirst = ilast + 1
        yield()  # let some threads finish so that i_per_thread[] gets updated asap
    end

    return out
end


function mybroadcast2d(fn, x, y)
    Treturn = eltype(Base.return_types(fn, (eltype(x), eltype(y)))[1])

    outsize = calc_outsize(x, y)
    #@show outsize, size(x), size(y)
    if size(x) != outsize
        x = MeshedArray(outsize, x)
    end
    if size(y) != outsize
        y = MeshedArray(outsize, y)
    end
    #@show outsize, size(x), size(y)

    out = Array{Treturn}(undef, outsize...)

    mybroadcast2d!(out, fn, x, y)

    return out
end


end


# vim: set sw=4 et sts=4 :
