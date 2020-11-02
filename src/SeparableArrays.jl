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
export exponentiate


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


function Base.propertynames(s::SeparableArray, private=false)
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
function exponentiate(s::SeparableArray, exp::Number)
    a1 = s.arr1 .^ exp
    a2 = s.arr2 .^ exp
    return SeparableArray(a1, a2, name1=s.name1, name2=s.name2)
end



end


# vim: set sw=4 et sts=4 :
