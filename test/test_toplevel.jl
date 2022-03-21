
function sphhary(l::Int, m::Int, θ::T, ϕ::T) where {T<:AbstractFloat}
    if l == m == 0
        return 1 / √(4*T(π)) + im*0
    elseif l == 1
        if m == -1
            return √(3/(8*T(π))) * sin(θ) * ℯ^(-im*ϕ)
        elseif m == 0
            return √(3/(4*T(π))) * cos(θ) + im*0
        else
            return -√(3/(8*T(π))) * sin(θ) * ℯ^(im*ϕ)
        end
    end
    @error "sphhary() cannot evaluate" l m θ ϕ
    @assert false
    return T(NaN)
end


@testset "SphericalFourierBesselDecompositions toplevel" begin
    @testset "sphericalharmonicsy()" begin
        θ, ϕ = rand(2)
        @show θ, ϕ

        # ℓ = 0
        y00 = SFB.sphericalharmonicsy(0,0,θ,ϕ)
        y00_d = sphhary(0,0,θ,ϕ)
        @test y00 ≈ 1/√(4π)  rtol=eps(abs(y00))
        @test y00_d ≈ 1/√(4π)  rtol=eps(abs(y00))

        # ℓ = 1
        y1m1 = SFB.sphericalharmonicsy(1,-1,θ,ϕ)
        y1m1_d = sphhary(1,-1,θ,ϕ)
        y1m1_c = √(3/(8π)) * sin(θ) * ℯ^(-im*ϕ)
        @test y1m1 ≈ y1m1_c
        @test y1m1_d ≈ y1m1_c

        y10 = SFB.sphericalharmonicsy(1,0,θ,ϕ)
        y10_d = sphhary(1,0,θ,ϕ)
        y10_c = √(3/(4π)) * cos(θ) + 0*im
        @test y10 ≈ y10_c
        @test y10_d ≈ y10_c

        y1p1 = SFB.sphericalharmonicsy(1,1,θ,ϕ)
        y1p1_d = sphhary(1,1,θ,ϕ)
        y1p1_c = -√(3/(8π)) * sin(θ) * ℯ^(im*ϕ)
        @test y1p1 ≈ y1p1_c
        @test y1p1_d ≈ y1p1_c

    end
end


# vim: set sw=4 et sts=4 :
