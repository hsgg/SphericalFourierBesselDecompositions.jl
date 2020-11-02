#!/usr/bin/env julia


module HealPy

export hp

using PyCall

const hp = PyNULL()

function __init__()
    copy!(hp, pyimport_conda("healpy", "healpy", "conda-forge"))
end


end


# vim: set sw=4 et sts=4 :
