#!/usr/bin/env julia


@testset "Window Chains" begin
    ell = [1, 1, 1, 1]
    I_LM_nl1_n12 = []
    SFB.window_chains.window_chain(ell, I_LM_nl1_n12)
end


# vim: set sw=4 et sts=4 :
