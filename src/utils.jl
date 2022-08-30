module utils

export fskyinv0nn_expmrr0, W_n00_n00_expmrr0, get_0nn


using ..Modes
using LinearAlgebra


# analytic solutions
function fskyinv0nn_expmrr0(wmodes, cmodes; r0sign=1)
    rmax = wmodes.rmax
    r0 = r0sign * rmax / 2
    nmax = cmodes.amodes.nmax
    rfirst = wmodes.r[1]
    normalization = exp(- rfirst / r0)  # first evaluation is in the bin center, and that sets the normalization of phi(r)
    expRr0 = exp(rmax / r0)

    fskyinv0nn = fill(0.0, nmax, nmax)
    for n1=1:nmax, n2=1:nmax
        if isvalidlnn_symmetric(cmodes, 0, n1, n2)
            l, k1, k2 = getlkk(cmodes, 0, n1, n2)
            m1_nn = (-1)^(n1 + n2)
            bracket1 = m1_nn + expRr0
            bracket2 = m1_nn - expRr0
            prefac1 = r0 / (1 + (k1+k2)^2 * r0^2)
            prefac2 = r0 / (1 + (k1-k2)^2 * r0^2)
            result = normalization * (prefac1 * bracket1 - prefac2 * bracket2) / rmax

            fskyinv0nn[n1,n2] = result
        end
    end
    @assert issymmetric(fskyinv0nn)
    return fskyinv0nn
end

W_n00_n00_expmrr0(wmodes, cmodes) = fskyinvlnn_expmrr0(wmodes, cmodes; r0sign=-1)


function get_0nn(Clnn, cmodes)
    nmax = cmodes.amodes.nmax
    matrix = fill(0.0, nmax, nmax)
    for n1=1:nmax, n2=1:nmax
        if isvalidlnn_symmetric(cmodes, 0, n1, n2)
            matrix[n1,n2] = Clnn[getidx(cmodes, 0, n1, n2)]
        end
    end
    @assert issymmetric(matrix)
    return matrix
end


end


# vim: set sw=4 et sts=4 :
