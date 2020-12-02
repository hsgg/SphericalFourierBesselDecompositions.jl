#!/usr/bin/env julia


module window_chains

using ..Windows
using WignerSymbols


function window_chain(nn, ll)
    @assert length(ll) == length(nn)
    k = length(nn)
    #Inlnl_lm = ...
    wk = 0.0
    for i=1:k
    end
    wk *= prod(@. 2 * ll + 1)
    return wk
end


################## W^3j_k ####################################################

# w3j_k(ell, L, M): This is the frontend to calculate the W^3j_k. It would be
# phantastic to dispatch on the length of the arrays `ell`, `L`, and `M`.
function w3j_k(ell, L, M)
    @assert length(ell) == length(L) == length(M)
    k = length(ell)
    if k == 1
        return calc_w3j_1(ell, L, M)
    elseif k == 2
        return calc_w3j_2(ell, L, M)
    elseif k == 3
        return calc_w3j_3(ell, L, M)
    #elseif k == 4
    #    return calc_w3j_4(ell, L, M)  # not that fast
    #elseif k == 5
    #    return calc_w3j_5(ell, L, M)  # not that fast
    else
        return calc_w3j_k(ell, L, M)
    end
end


# This is the stupid implementation
function calc_w3j_k(ell, L, M)
    T = Float64
    k = length(ell)

    # check for trivial zeros
    (abs(ell[end] - ell[1]) <= L[1] <= ell[end] + ell[1]) || return T(0)
    for i=2:k
        (abs(ell[i-1] - ell[i]) <= L[i] <= ell[i-1] + ell[i]) || return T(0)
    end

    # calculate
    w3jk = T(0)
    mm = fill(0, k)
    skip_m1 = false
    for m1=-ell[1]:ell[1]
        # set m's
        mm[1] = m1
        for i=2:k
            mm[i] = mm[i-1] - M[i]
            if abs(mm[i]) > ell[i]
                skip_m1 = true
                break
            end
        end
        if skip_m1
            skip_m1 = false
            continue
        end
        if -mm[end] + mm[1] + M[1] != 0
            continue
        end

        #@show m1,ell,mm,L,M

        # calculate term
        w = wigner3j(T, ell[end], ell[1], L[1], -mm[end], mm[1])
        #@show w
        for i=2:k
            w *= wigner3j(T, ell[i-1], ell[i], L[i], -mm[i-1], mm[i])
            #@show w
        end
        w *= (-1)^sum(mm)
        #@show w

        # sum it up
        w3jk += w
    end
    return w3jk
end


# k = 1
function calc_w3j_1(ell, L, M)
    T = Float64
    if L[1] != 0 || M[1] != 0
        return T(0)
    end
    return (-1)^ell[1] * √(2*ell[1] + 1)
end


# k = 2
function calc_w3j_2(ell, L, M)
    T = Float64
    if L[1] != L[2] || M[1] != - M[2]
        return T(0)
    end
    if !(abs(L[1] - ell[1]) <= ell[2] <= L[1] + ell[1])
        return T(0)
    end
    return (-1)^(M[1]) / (2*L[1] + 1)
end

# k = 2, simple
function calc_w3j_2_simple(ell, L, M)
    T = Float64
    l1 = ell[1]
    l2 = ell[2]
    L1 = L[1]
    L2 = L[2]
    M1 = M[1]
    M2 = M[2]
    s4 = T(0)
    for m1=-l1:l1, m2=-l2:l2
        p = (-1)^(m1 + m2)
        w1 = wigner3j(T, l2, l1, L1, -m2, m1, M1)
        w2 = wigner3j(T, l1, l2, L2, -m1, m2, M2)
        s4 += p * w1 * w2
    end
    return s4
end


# k = 3
function calc_w3j_3(ell, L, M)
    T = Float64
    p = (-1)^(sum(ell) + sum(L))
    w3jm = wigner3j(T, L[1], L[2], L[3], -M[1], -M[2], -M[3])
    w6j = wigner6j(T, L[1], L[2], L[3], ell[2], ell[3], ell[1])
    return p * w3jm * w6j
end


# k = 4
function calc_w3j_4(ell, L, M)
    T = Float64
    M5 = M[1] + M[4]
    M5′ = - (M[2] + M[3])
    (M5 == M5′) || return T(0)
    L5min = max(abs(L[1] - L[4]), abs(L[2] - L[3]), abs(M5))
    L5max = min(L[1] + L[4], L[2] + L[3])
    w = T(0)
    for L5=L5min:L5max
        #@show L[1], L5, L[4], M[1], -M5, M[4]
        w3a = wigner3j(T, L[1], L5, L[4], M[1], -M5, M[4])
        w3b = wigner3j(T, L[2], L5, L[3], M[2], M5, M[3])
        w6a = wigner6j(T, L[1], L5, L[4], ell[3], ell[4], ell[1])
        w6b = wigner6j(T, L[2], L5, L[3], ell[3], ell[2], ell[1])
        w += (-1)^(L5-M5) * (2*L5 + 1) * w3a * w3b * w6a * w6b
    end
    return (-1)^(L[2] + L[3] + ell[2] + ell[4]) * w
end


# k = 5
function calc_w3j_5(ell, L, M)
    T = Float64
    M6 = - (M[1] + M[2])
    M7 = - (M[3] + M[4])
    M6′ = M[5] - M7
    (M6 == M6′) || return T(0)
    L6min = max(abs(L[1] - L[2]), abs(M6))
    L6max = L[1] + L[2]
    w = T(0)
    for L6=L6min:L6max
        L7min = max(abs(L[3] - L[4]), abs(L[5] - L6), abs(M7))
        L7max = min(L[3] + L[4], L[5] + L6)
        for L7=L7min:L7max
            p = (-1)^(L6-M6+L7-M7) * (2*L6+1) * (2*L7+1)
            w3a = wigner3j(T, L[1], L[2], L6, M[1], M[2], M6)
            w3b = wigner3j(T, L6, L[5], L7, -M6, M[5], -M7)
            w3c = wigner3j(T, L7, L[3], L[4], M7, M[3], M[4])
            w6a = wigner6j(T, L[1], L[2], L6, ell[2], ell[5], ell[1])
            w6b = wigner6j(T, L6, L[5], L7, ell[4], ell[2], ell[5])
            w6c = wigner6j(T, L7, L[3], L[4], ell[3], ell[4], ell[2])
            w += p * w3a * w3b * w3c * w6a * w6b * w6c
        end
    end
    return (-1)^(ell[1] + ell[2] + ell[3] + L[5]) * w
end



end


# vim: set sw=4 et sts=4 :
