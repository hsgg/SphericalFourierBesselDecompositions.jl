@doc raw"""
    MeshedArrays

A 'MeshedArray' struct is a kind of view into a smaller array 'x'. It is useful
for treating a 1D array as a 2D array ignoring one of the dimensions. A
'MeshedArray' is an 'AbstractArray'.
"""
module MeshedArrays

export MeshedArray

#using StaticArrays


struct MeshedArray{T,N,Tarr,Tsz} <: AbstractArray{T,N}
    totsize::Tsz
    x::Tarr
end

MeshedArray(sz, x) = begin
    for n=1:ndims(x)
        if size(x,n) != 1
            if size(x,n) != sz[n]
                error("dimension $n in 'x' $(size(x)) must match 'sz' $sz or be 1")
            end
        end
    end
    return MeshedArray{eltype(x), length(sz), typeof(x), typeof(sz)}(sz, x)
end


Base.eltype(a::MeshedArray{T}) where {T} = T


Base.ndims(a::MeshedArray{T,N}) where {T, N <: Integer} = N

Base.size(a::MeshedArray) = a.totsize

Base.length(a::MeshedArray) = prod(size(a))

Base.IndexStyle(::Type{<:MeshedArray}) = IndexLinear()

Base.getindex(a::MeshedArray, i::Int) = begin
    iout = 0
    szx = 1
    i0 = i - 1  # zero-based index
    for n=1:ndims(a.x)-1
        if size(a.x, n) != 1
            i0, d0 = divrem(i0, size(a.x, n))
            iout += szx * d0
            szx *= size(a.x, n)
        else
            i0 = div(i0, size(a, n))
        end
    end
    if size(a.x, ndims(a.x)) != 1
        i0, d0 = divrem(i0, size(a, ndims(a.x)))
        iout += szx * d0
    end
    return a.x[iout+1]
end

Base.getindex(a::MeshedArray, ii::AbstractArray{Int}) = begin
    return getindex.(Ref(a), ii)
end

function _dd0_to_iout(dd0, x)
    szx = 1
    iout = 0
    for n=1:ndims(x)
        if size(x, n) != 1
            iout = szx * dd0[n]
            szx *= size(x, n)
        end
    end
    return 1 + iout
end

function _dd0_plus_one!(dd0, a)
    dd0[1] += 1
    for n=1:length(dd0)-1
        if dd0[n] >= size(a, n)
            dd0[n] = 0
            dd0[n+1] += 1
        end
    end
    n = length(dd0)
    if dd0[n] >= size(a, n)
        dd0[n] = 0
    end
end

Base.getindex(a::MeshedArray{T}, ii::UnitRange{Int}) where {T} = begin
#Base.getindex(a::MeshedArray{T}, ii::StepRange) where {T} = begin
    out = Array{T}(undef, length(ii))
    if length(ii) < 1
        return out
    end

    dd0 = Array{Int}(undef, ndims(a.x))
    #dd0 = MVector{ndims(a.x),Int}(undef)

    i0 = ii[1] - 1  # zero-based index
    for n=1:ndims(a.x)
        i0, dd0[n] = divrem(i0, size(a, n))
    end
    iout = _dd0_to_iout(dd0, a.x)
    #@show 1,dd0,iout
    out[1] = a.x[iout]


    for k=2:length(ii)
        _dd0_plus_one!(dd0, a)

        iout = _dd0_to_iout(dd0, a.x)

        #@show k,dd0,iout

        val = a.x[iout]
        out[k] = val
    end

    return out
end

Base.getindex(a::MeshedArray{T,N}, I::Vararg{Int,N}) where {T,N} = begin
    iout = 0
    szx = 1
    for n=1:ndims(a.x)
        #@assert I[n] <= size(a, n)
        if size(a.x, n) != 1
            d = I[n]
            iout += szx * (d - 1)
        end
        szx *= size(a.x, n)
    end
    #for n=ndims(a.x)+1:ndims(a)
    #    @assert I[n] <= size(a, n)
    #end
    return a.x[iout+1]
end



end


# vim: set sw=4 et sts=4 :
