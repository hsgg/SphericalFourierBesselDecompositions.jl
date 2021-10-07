#!/usr/bin/env julia


module HealPy

export hp

using PyCall

const hp = PyNULL()

function __init__()
    if Sys.iswindows()
        @warn "HealPy is not supported on Windows."
    else
        copy!(hp, pyimport_conda("healpy", "healpy", "conda-forge"))
    end
end


end


# vim: set sw=4 et sts=4 :
