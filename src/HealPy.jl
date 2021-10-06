#!/usr/bin/env julia


module HealPy

export hp

using PyCall

const hp = PyNULL()

function __init__()
    try
        copy!(hp, pyimport_conda("healpy", "healpy", "conda-forge"))
    catch
        @warn "Could not load healpy."
    end
end


end


# vim: set sw=4 et sts=4 :
