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


# This module introduces an N-dimensional iterator for use in for-loops with
# arbitrary number of variables. Each variable will be incremented by one.
#
# Currently, we implement an iterator interface for the iteration over the
# elements of the iterator, not to advance the iterator.
#
# TODO: Implement iterator interface defined by Julia documentation. The object
# that the iterator returns is the array of indices.


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
