#!/usr/bin/env julia


module SciPy


export scipy

using PyCall

const scipy = PyNULL()

function __init__()
    copy!(scipy, pyimport_conda("scipy", "scipy"))
end


end


# vim: set sw=4 et sts=4 :
