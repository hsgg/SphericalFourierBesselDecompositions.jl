module utils

export fskyinv0nn_expmrr0, W0nn_expmrr0, get_0nn,
    calc_T23_expmrr0, calc_T4_expmrr0, set_T1_ell0_expmrr0!


using ..Modes
using LinearAlgebra


# analytic solutions
function fskyinv0nn_expmrr0(wmodes, cmodes; r0sign=1)
    rmax = wmodes.rmax
    r0 = r0sign * rmax / 2 / 3
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

W0nn_expmrr0(wmodes, cmodes) = fskyinv0nn_expmrr0(wmodes, cmodes; r0sign=-1)


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


function calc_T23_expmrr0(wmodes, cmodes)
    f0nn = fskyinv0nn_expmrr0(wmodes, cmodes)
    W0nn = W0nn_expmrr0(wmodes, cmodes)

    parens = W0nn * f0nn * W0nn

    lnnsize = getlnnsize(cmodes)
    T23 = fill(0.0, lnnsize, lnnsize)

    for i=1:lnnsize, j=1:lnnsize

        l, n1, n2 = getlnn(cmodes, i)
        L, N1, N2 = getlnn(cmodes, j)

        if l != 0 || L != 0
            continue
        end

        T23[i,j] = parens[N2,n1] * W0nn[n2,N1] + parens[N2,n2] * W0nn[n1,N1]

        if N1 != N2
            T23[i,j] *= 2
        end
    end

    return T23
end


function calc_T4_expmrr0(wmodes, cmodes)
    f0nn = fskyinv0nn_expmrr0(wmodes, cmodes)
    W0nn = W0nn_expmrr0(wmodes, cmodes)

    parens = W0nn * f0nn * W0nn

    lnnsize = getlnnsize(cmodes)
    T4 = fill(0.0, lnnsize, lnnsize)

    for i=1:lnnsize, j=1:lnnsize

        l, n1, n2 = getlnn(cmodes, i)
        L, N1, N2 = getlnn(cmodes, j)

        if l != 0 || L != 0
            continue
        end

        T4[i,j] = parens[n1,N1] * parens[N2,n2]

        if N1 != N2
            T4[i,j] *= 2
        end
    end

    return T4
end


function set_T1_ell0_expmrr0!(cmix, wmodes, cmodes)
    W0nn = W0nn_expmrr0(wmodes, cmodes)

    nmax = cmodes.amodes.nmax

    for n1=1:nmax, n2=1:nmax, N1=1:nmax, N2=1:nmax

        if !(isvalidlnn_symmetric(cmodes, 0, n1, n2) && isvalidlnn_symmetric(cmodes, 0, N1, N2))
            continue
        end

        i = getidx(cmodes, 0, n1, n2)
        j = getidx(cmodes, 0, N1, N2)

        cmix[i,j] = W0nn[n1,N1] * W0nn[N2,n2]

        if N1 != N2
            cmix[i,j] *= 2
        end
    end

    return cmix
end


end


# vim: set sw=4 et sts=4 :
