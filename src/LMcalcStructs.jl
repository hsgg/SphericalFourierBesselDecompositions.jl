module LMcalcStructs

export LMcalcStruct, LMcalcStructMfast


###################### Healpix-style: (l,m) = [(0,0), (1,0), (2,0), ..., (1,1), (2,1), ...]
struct LMcalcStruct
    lmax::Int
    tval::Int
end
LMcalcStruct(lmax) = LMcalcStruct(lmax, 2 * lmax + 1)

Base.getindex(lm::LMcalcStruct, lp1::Int, mp1::Int) = begin
    m = mp1 - 1
    tval = lm.tval
    #return lp1 + m * lmax - ((m - 1) * m) ÷ 2
    return lp1 + ((m * (tval - m)) >> 1)
end


###################### m-fast-style: (l,m) = [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), ...]
struct LMcalcStructMfast end

Base.getindex(::LMcalcStructMfast, lp1::Int, mp1::Int) = begin
    return mp1 + ((lp1 - 1) * lp1) ÷ 2
end




end
