# This module introduces an N-dimensional iterator for use in for-loops with
# arbitrary number of variables. Each variable will be incremented by one.
#
# Currently, we implement an iterator interface for the iteration over the
# elements of the iterator, not to advance the iterator.
#
# TODO: Implement iterator interface defined by Julia documentation.
# TODO: Implement AbstracArray interface defined by Julia documentation.

module NDIterators

export NDIterator
export advance


struct NDIterator{T} # <: AbstractArray{T,1}
    idx::Array{T,1}
    idxmin::Array{T,1}
    idxmax::Array{T,1}
    i::Array{Int,0}
end


function NDIterator(idxmin::AbstractArray, idxmax::AbstractArray)
    @assert length(idxmin) == length(idxmax)
    idx = deepcopy(idxmin)
    idx[end] -= 1
    ndi = NDIterator(idx, idxmin, idxmax, fill(length(idx)))
    for i=1:length(idx)
        if idxmax[i] < idxmin[i]
            ndi.i[] = 0
        end
    end
    return ndi
end
NDIterator(idxmin::Integer, idxmax::Integer; N=1) = NDIterator(fill(idxmin,N), fill(idxmax,N))
NDIterator(idxmin::Integer, idxmax) = NDIterator(fill(idxmin, length(idxmax)), convert(Array, idxmax))
NDIterator(idxmin, idxmax::Integer) = NDIterator(convert(Array, idxmin), fill(idxmax, length(idxmin)))
NDIterator(idxmin, idxmax) = NDIterator(convert(Array, idxmin), convert(Array, idxmax))


Base.getindex(ndi::NDIterator, i::Int) = ndi.idx[i]
Base.:-(ndi::NDIterator) = - ndi.idx
Base.convert(::Type{T}, ndi::NDIterator) where {T<:AbstractArray} = T(ndi.idx)

Base.firstindex(ndi::NDIterator) = 1
Base.lastindex(ndi::NDIterator) = length(ndi.idx)

Base.show(io::IO, ndi::NDIterator) = show(io, ndi.idx)

# for @__dot__ syntax
Base.length(ndi::NDIterator) = length(ndi.idx)
Base.iterate(ndi::NDIterator, i=1) = i > length(ndi) ? nothing : (ndi.idx[i], i+1)
Base.eltype(::NDIterator{T}) where {T} = T


function advance(ndi::NDIterator)
    i = ndi.i[]
    N = length(ndi.idx)
    while true
        if i <= 0
            ndi.i[] = i
            return false
        end
        ndi.idx[i] += 1
        if ndi.idx[i] > ndi.idxmax[i]
            ndi.idx[i] = ndi.idxmin[i] - 1
            i -= 1
        elseif i < N
            i += 1
        else
            break
        end
    end
    ndi.i[] = i
    return true
end


end


# vim: set sw=4 et sts=4 :
