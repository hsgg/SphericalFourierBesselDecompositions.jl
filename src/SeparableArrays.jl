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


# Purpose: This provides a type of array that is representable as a
# multiplication by two arrays. For example, the array `S`
#
#   S[1,2] = d1[1] * d2[2]
#
# is representable as the multiplication of elements from arrays `d1` and `d2`.
# Then, instead of saving the entire 2D matrix `S`, we can simply save the
# vectors `d1` and `d2`, saving space, and saving computation time as we can
# specialize some algorithms.
#
# It is not possible to write to a SeparableArray. This is because the meaning
# of an operation such as `S[3,4] = 5.0` is undefined: do you want to change
# the first array or the second array?


module SeparableArrays


export SeparableArray, @SeparableArray
export exponentiate, elementwise_mult


struct SeparableArray{T,N,A1,A2} <: AbstractArray{T,N}
    arr1::A1
    arr2::A2
    name1::Symbol
    name2::Symbol
end


function SeparableArray(arr1, arr2; name1=:arr1, name2=:arr2)
    T = promote_type(eltype(arr1), eltype(arr2))
    N = ndims(arr1) + ndims(arr2)
    A1 = typeof(arr1)
    A2 = typeof(arr2)
    return SeparableArray{T,N,A1,A2}(arr1, arr2, name1, name2)
end


macro SeparableArray(arr1, arr2)
    sym1 = "$arr1"
    sym2 = "$arr2"
    return :( SeparableArray($(esc(arr1)), $(esc(arr2)); name1=Symbol($sym1), name2=Symbol($sym2)) )
end


Base.size(A::SeparableArray) = size(A.arr1)..., size(A.arr2)...


function Base.getindex(A::SeparableArray, i::Int)
    N1 = length(A.arr1)
    m = (i-1) % N1 + 1
    n = (i-1) ÷ N1 + 1
    return A.arr1[m] * A.arr2[n]
end


function calc_index(I, arr, off, A)
    idx = I[1+off]
    sz = size(arr, 1)
    L = 1
    1 <= idx <= sz || throw(BoundsError(A, I))
    for i=2:ndims(arr)
        L *= sz
        sz = size(arr, i)
        1 <= I[i+off] <= sz || throw(BoundsError(A, I))
        idx += (I[i+off] - 1) * L
    end
    return arr[idx]
end


function Base.getindex(A::SeparableArray{T,N,A1,A2}, I::Vararg{Int,N}) where {T,N,A1,A2}
    arr1 = getfield(A, :arr1)
    D1 = ndims(arr1)
    V1 = calc_index(I, arr1, 0, A)
    arr2 = getfield(A, :arr2)
    V2 = calc_index(I, arr2, D1, A)
    return V1 * V2
end


function Base.getproperty(s::SeparableArray, f::Symbol)
    #@show "getproperty",f
    if f == getfield(s, :name1)
        return getfield(s, :arr1)
    elseif f == getfield(s, :name2)
        return getfield(s, :arr2)
    else
        return getfield(s, f)
    end
end


function Base.propertynames(s::SeparableArray, private::Bool=false)
    tuple = (s.name1, s.name2)
    if private
        tuple = tuple..., fieldnames(typeof(s))...
    end
    return Tuple(unique(tuple))
end


# Optimizations
@doc raw"""
    exponentiate(s::SeparableArray, exp::Number)

This function is used for elementwise exponentiation of the array 's'. It could
be made more elegant by extending the broadcast syntax. PRs welcome.
"""
function exponentiate(s::SeparableArray, exponent::Number)
    a1 = s.arr1 .^ exponent
    a2 = s.arr2 .^ exponent
    return SeparableArray(a1, a2, name1=s.name1, name2=s.name2)
end
exponentiate(s, e) = s .^ e
#Base.broadcasted(::typeof(^), s::SeparableArray, e::Number) = exponentiate(s, e)


function elementwise_mult(a1::SeparableArray, a2::SeparableArray)
    arr1 = a1.arr1 .* a2.arr1
    arr2 = a1.arr2 .* a2.arr2
    return SeparableArray(arr1, arr2; name1=a1.name1, name2=a1.name2)
end
function elementwise_mult(a1::Union{Number,AbstractVector}, a2::SeparableArray)
    arr1 = a1 .* a2.arr1
    arr2 = a2.arr2
    return SeparableArray(arr1, arr2; name1=a2.name1, name2=a2.name2)
end
elementwise_mult(a1::SeparableArray, a2::Union{Number,AbstractVector}) = elementwise_mult(a2, a1)
elementwise_mult(a1, a2) = a1 .* a2  # fallback



end


# vim: set sw=4 et sts=4 :
