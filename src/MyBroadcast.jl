@doc raw"""
    MyBroadcast

This module defines the function `mybroadcast`. It behave similarly to a
threaded broadcast, except that it tries to batch iterations such that each
batch takes about 0.5 seconds to perform.

The idea is to automatically adjust the number of iterations per batch so that
overhead per iteration is low and batch size is small so that the threads keep
getting scheduled.

For example, imagine that the execution time per iteration increases. With a
static scheduler, this would mean that the first threads finish long before the
last thread. This avoids that by adjusting the number of iterations so that
each batch should take approximately 0.5 seconds.

So why batch iterations? Imagine you need to allocate a buffer for each
iteration, and this buffer can be shared for sequentially run iterations.
Allocating a separate buffer would add a lot of overhead, so that traditional
`map()` can take longer than the serial implementation. Batching avoids that
pitfall.
"""
module MyBroadcast

export mybroadcast

using Base.Threads


include("MeshedArrays.jl")
using .MeshedArrays


function calc_i_per_thread(time, i_per_thread_old; batch_avgtime=0.5, batch_maxadjust=2.0)
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


function get_new_batch!(next_ifirst_channel, ntasks, batchsize)
    ifirst = take!(next_ifirst_channel)
    ilast = min(ntasks, ifirst + batchsize - 1)
    put!(next_ifirst_channel, ilast + 1)
    iset = ifirst:ilast
    return iset
end


function mybroadcast!(out, fn, x...; num_threads=Threads.nthreads())
    ntasks = prod(calc_outsize(x...))
    @assert size(out) == calc_outsize(x...)

    errorchannel = Channel{Any}(num_threads)

    next_ifirst_channel = Channel{Int}(1)  # this channel is used to synchronize all the threads
    put!(next_ifirst_channel, 1)  # first task starts at iteration 1, threads update this as they take work

    all_indices = eachindex(out, x...)

    # worker threads process the data
    @threads for _ in 1:num_threads
        try
            batchsize = 1

            # worker threads feed themselves
            iset = get_new_batch!(next_ifirst_channel, ntasks, batchsize)

            while length(iset) > 0

                time = @elapsed begin
                    idxs = all_indices[iset]

                    xs = (x[i][idxs] for i=1:length(x))
                    outs = fn(xs...)

                    out[idxs] .= outs
                end

                batchsize = calc_i_per_thread(time, length(iset))

                iset = get_new_batch!(next_ifirst_channel, ntasks, batchsize)
            end
        catch e
            if e isa InvalidStateException
                @info "Exiting thread $(Threads.threadid()) due to closed channel"
            else
                # we caused the exception
                close(next_ifirst_channel)  # notify other threads
                bt = catch_backtrace()
                @warn "Exception in thread $(Threads.threadid()):\n  $e"
                put!(errorchannel, (Threads.threadid(), e, bt))
            end
        end
    end


    num_failed_tasks = 0
    while isready(errorchannel)
        num_failed_tasks += 1
        tid, e, stack = take!(errorchannel)
        println(stdout)
        @error "Exception in thread $tid of $num_threads:\n  $e"
        showerror(stdout, e, stack)
        println(stdout)
    end
    if num_failed_tasks > 0
        println(stdout)
        @error "Exceptions in threads" num_failed_tasks num_threads
        error("Exceptions in threads")
    end

    return out
end


function mybroadcast(fn, x...; kwargs...)
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

    mybroadcast!(out, fn, xs...; kwargs...)

    return out
end


end


# vim: set sw=4 et sts=4 :
