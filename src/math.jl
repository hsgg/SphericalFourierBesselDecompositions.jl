module MyMathFunctions


export sphericalharmonicsy, realsphericalharmonicsy


using FastTransforms: sphevaluate


function sphericalharmonicsy(l, m, θ, ϕ)
    T = ComplexF64
    rY1 = sphevaluate(θ, ϕ, l, abs(m))
    rY2 = sphevaluate(θ, ϕ, l, -abs(m))
    if m < 0
        return T((rY1 - im*rY2) / √2)
    elseif m == 0
        return T(rY1)
    else
        return T((-1)^m * (rY1 + im*rY2) / √2)
    end
end


function realsphericalharmonicsy(l, m, θ, ϕ)
    #Ylm = sphericalharmonicsy(l, m, θ, ϕ)
    #if m < 0
    #    return √2 * (-1)^m * imag(Ylm)
    #elseif m == 0
    #    return real(Ylm)
    #else
    #    return √2 * (-1)^m * real(Ylm)
    #end
    return sphevaluate.(θ, ϕ, l, m)
end


end
