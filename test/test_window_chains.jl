#!/usr/bin/env julia


using SphericalFourierBesselDecompositions
SFB = SphericalFourierBesselDecompositions

using Test

@testset "Window Chains" begin
    @testset "W^3j_k" begin
        @testset "k = 1" begin
            @testset "ell = $ell" for ell=0:10
                w3j_1_slow = SFB.window_chains.calc_w3j_k([ell], [0], [0])
                w3j_1_fast = SFB.window_chains.calc_w3j_1([ell], [0], [0])
                @test w3j_1_slow ≈ (-1)^ell * √(2*ell+1) rtol=eps(10.0)
                @test w3j_1_fast ≈ (-1)^ell * √(2*ell+1) rtol=eps(10.0)

                for L=1:10, M=-L:L
                    w3j_1_slow = SFB.window_chains.calc_w3j_k([ell], [L], [M])
                    w3j_1_fast = SFB.window_chains.calc_w3j_1([ell], [L], [M])
                    @test w3j_1_slow ≈ 0 atol=eps(1.0)
                    @test w3j_1_fast ≈ 0 atol=eps(1.0)
                end
            end
            k = 1
            @show k
        end


        @testset "k = 2" begin
            lmax = 5
            SFB.window_chains.calc_w3j_k([0,0], [0,0], [0,0]) # compile
            SFB.window_chains.calc_w3j_2([0,0], [0,0], [0,0]) # compile
            SFB.window_chains.calc_w3j_2_simple([0,0], [0,0], [0,0]) # compile
            tslow = 0.0
            tfast = 0.0
            tsimple = 0.0
            @time for l1=0:lmax, l2=0:lmax, L1=0:lmax, L2=0:lmax, M1=-L1:L1, M2=-L2:L2
                ell = [l1, l2]
                L = [L1, L2]
                M = [M1, M2]
                #@show ell,L,M
                tslow += @elapsed w3j_2_slow = SFB.window_chains.calc_w3j_k(ell, L, M)
                tfast += @elapsed w3j_2_fast = SFB.window_chains.calc_w3j_2(ell, L, M)
                tsimple += @elapsed w3j_2_simple = SFB.window_chains.calc_w3j_2_simple(ell, L, M)
                @test w3j_2_fast ≈ w3j_2_simple atol=eps(10.0)
                @test w3j_2_slow ≈ w3j_2_simple atol=eps(10.0)
            end
            k = 2
            @show k tsimple tslow tfast
        end


        @testset "k = 3" begin
            lmax = 3
            SFB.window_chains.calc_w3j_k([0,1,0], [0,0,0], [0,0,0]) # compile
            SFB.window_chains.calc_w3j_3([0,1,0], [0,0,0], [0,0,0]) # compile
            tslow = 0.0
            tfast = 0.0
            @time for l1=0:lmax, l2=0:lmax, l3=0:lmax, L1=0:lmax, L2=0:lmax, L3=0:lmax
                for M1=-L1:L1, M2=-L2:L2, M3=-L3:L3
                    ell = [l1, l2, l3]
                    L = [L1, L2, L3]
                    M = [M1, M2, M3]
                    #@show ell,L,M
                    tslow += @elapsed w3j_3_slow = SFB.window_chains.calc_w3j_k(ell, L, M)
                    tfast += @elapsed w3j_3_fast = SFB.window_chains.calc_w3j_3(ell, L, M)
                    @test w3j_3_slow ≈ w3j_3_fast atol=eps(10.0)
                end
            end
            k = 3
            @show k tslow tfast
        end


        @testset "k = 4" begin
            lmax = 2
            SFB.window_chains.calc_w3j_k([0,1,0,0], [0,0,0,0], [0,0,0,0]) # compile
            SFB.window_chains.calc_w3j_4([0,1,0,0], [0,0,0,0], [0,0,0,0]) # compile
            tslow = 0.0
            tfast = 0.0
            @time for l1=0:lmax, l2=0:lmax, l3=0:lmax, l4=0:lmax
                for L1=0:lmax, L2=0:lmax, L3=0:lmax, L4=0:lmax
                    for M1=-L1:L1, M2=-L2:L2, M3=-L3:L3, M4=-L4:L4
                        ell = [l1, l2, l3, l4]
                        L = [L1, L2, L3, L4]
                        M = [M1, M2, M3, M4]
                        #@show ell,L,M
                        tslow += @elapsed w3j_4_slow = SFB.window_chains.calc_w3j_k(ell, L, M)
                        tfast += @elapsed w3j_4_fast = SFB.window_chains.calc_w3j_4(ell, L, M)
                        @test w3j_4_slow ≈ w3j_4_fast atol=eps(10.0)
                    end
                end
            end
            k = 4
            @show k tslow tfast
        end


        @testset "k = 5" begin
            lmax = 1
            SFB.window_chains.calc_w3j_k([0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0]) # compile
            SFB.window_chains.calc_w3j_5([0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0]) # compile
            tslow = 0.0
            tfast = 0.0
            @time for l1=0:lmax, l2=0:lmax, l3=0:lmax, l4=0:lmax, l5=0:lmax
                for L1=0:lmax, L2=0:lmax, L3=0:lmax, L4=0:lmax, L5=0:lmax
                    for M1=-L1:L1, M2=-L2:L2, M3=-L3:L3, M4=-L4:L4, M5=-L5:L5
                        ell = [l1, l2, l3, l4, l5]
                        L = [L1, L2, L3, L4, L5]
                        M = [M1, M2, M3, M4, M5]
                        #@show ell,L,M
                        tslow += @elapsed w3j_5_slow = SFB.window_chains.calc_w3j_k(ell, L, M)
                        tfast += @elapsed w3j_5_fast = SFB.window_chains.calc_w3j_5(ell, L, M)
                        @test w3j_5_slow ≈ w3j_5_fast atol=eps(10.0)
                    end
                end
            end
            k = 5
            @show k tslow tfast
        end
    end


    false && @testset "W_k" begin
        ell = [1, 1, 1, 1]
        I_LM_nl1_n12 = []
        SFB.window_chains.window_chain(ell, I_LM_nl1_n12)
    end
end


# vim: set sw=4 et sts=4 :
