@doc raw"""
    MyBroadcast

This module defines the function `mybroadcast`. It behave similarly to a
threaded broadcast, except that it tries to batch iterations such that each
batch takes about 1.0 seconds to perform.

The idea is to automatically adjust the number of iterations per batch so that
overhead per iteration is low and batch size is small so that the threads keep
getting scheduled.

For example, imagine that the execution time per iteration increases. With a
static scheduler, this would mean that the first threads finish long before the
last thread. This avoids that by adjusting the number of iterations so that
each batch should take approximately 1.0 seconds.
"""
module MyBroadcast

export mybroadcast

using Base.Threads


include("MeshedArrays.jl")
using .MeshedArrays


function calc_i_per_thread(time, i_per_thread_old; batch_avgtime=1.0, batch_maxadjust=2.0)
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


function calc_outsize(x...)
    outsize = fill(1, maximum(ndims.(x)))
    outsize[1:ndims(x[1])] .= size(x[1])
    for i=2:length(x)
        for d=1:ndims(x[i])
            if outsize[d] == 1
                outsize[d] = size(x[i], d)
            elseif size(x[i], d) != 1 && outsize[d] != size(x[i], d)
                error("cannot find common broadcast dimensions size.(x) = $(size.(x))")
            end
        end
    end
    return (outsize...,)
end


function mybroadcast!(out, fn, x...)
    ntasks = prod(calc_outsize(x...))
    @assert size(out) == calc_outsize(x...)

    ifirst = 1
    i_per_thread = Atomic{Int}(1)
    last_ifirst = 0
    lk = Threads.Condition()

    num_free_threads = Atomic{Int}(Threads.nthreads())
    @assert num_free_threads[] > 0

    all_indices = eachindex(out, x...)

    @sync while ifirst <= ntasks
        while num_free_threads[] < 1
            # Don't spawn the next batch until a thread is free. This has
            # several implications. First, Ctrl-C actually works (seems like
            # threadid=1 is the one catching the signal, and no tasks are
            # waiting on the other threads so they actually finish instead of
            # continuing in the background). Second, printing and ProgressMeter
            # actually work. Why? Not sure. Maybe because printing uses locks
            # and yields()? Maybe tasks need to be cleaned up?
            yield()
        end
        num_free_threads[] -= 1

        ilast = min(ntasks, ifirst + i_per_thread[] - 1)
        iset = ifirst:ilast  # no need to interpolate local variables

        @async Threads.@spawn begin
            time = @elapsed begin
                idxs = all_indices[iset]
                xs = (x[i][idxs] for i=1:length(x))
                outs = fn(xs...)
                out[idxs] .= outs
            end

            i_per_thread_new = calc_i_per_thread(time, length(iset))
            lock(lk) do
                if last_ifirst < iset[1]
                    i_per_thread[] = i_per_thread_new
                    last_ifirst = iset[1]
                end
            end
            num_free_threads[] += 1
        end

        ifirst = ilast + 1
        yield()  # let some threads finish so that i_per_thread[] gets updated asap
    end

    return out
end


function mybroadcast(fn, x...)
    Treturn = eltype(Base.return_types(fn, (eltype.(x)...,))[1])

    outsize = calc_outsize(x...)
    #@show outsize, size.(x)
    xs = [y for y in x]
    for i=1:length(xs)
        if size(xs[i]) != outsize
            xs[i] = MeshedArray(outsize, xs[i])
        end
    end
    #@show outsize, size.(xs)

    out = Array{Treturn}(undef, outsize...)

    mybroadcast!(out, fn, xs...)

    return out
end


end


# vim: set sw=4 et sts=4 :
