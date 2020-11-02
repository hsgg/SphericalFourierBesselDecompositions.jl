# This is a module to calculate a Spline.

# TODO:
# - Derivatives and integrals are not correct in extrapolation region.


module Splines

export Spline1D, Spline1Dtrans, Spline1Dloglog
export evaluate
export derivative
export integrate

#using TimerOutputs


@enum ExtrapolationType zero=0 boundary=1 linear=2 powerlaw=3

struct Spline1D{k,T}
    x::Array{T,1}
    y::Array{T,1}
    ypp::Array{T,1}
    ilast::Array{Int,0}  # for caching
    last_abcdh::Array{T,1}  # for caching
    betainv::Array{T,1} # workspace
    extrapolation::ExtrapolationType
end


# To make fused loops work
if VERSION < v"0.7.0"
    import Base.size, Base.getindex
    size(s::Spline1D) = (1,)
    getindex(s::Spline1D, i) = s
else
    import Base.length, Base.iterate
    length(s::Spline1D) = 1
    iterate(s::Spline1D) = s, nothing
    iterate(s::Spline1D, x) = nothing
end


################# All splines
function Spline1D(x, y; k=3, extrapolation=boundary)
    # checks
    if k < 0 || k == 2 || k > 3
        error("k=$k is not supported. :(")
    end
    if k == 0 && length(x) != length(y) + 1
        error("k=0 requires length(x) == length(y) + 1 (no OnCell() support)")
    elseif k != 0 && length(x) != length(y)
        error("Length of 'x' ($(length(x))) and 'y' ($(length(y))) must match.")
    end
    length(x) >= k+1 || error("Need at least k+1 points (length(x)=$(length(x))).")

    # do it, man
    T = promote_type(eltype(x), eltype(y))
    xin = convert(Array{T,1}, deepcopy(x))
    yin = convert(Array{T,1}, deepcopy(y))
    spline = Spline1D{k,T}(xin, yin, [], fill(0), [], [], extrapolation)
    #@show spline.extrapolation
    return Spline1D(xin, yin, spline)
end


Spline1D(; k=3, T=Float64, extrapolation=boundary) = Spline1D{k,T}([], [], [], fill(0), [], [], extrapolation)


function evaluate_past_boundary(spl::Spline1D{k,T}, x::Number) where {k,T}
    if spl.extrapolation == zero
        return T(0)
    elseif spl.extrapolation == boundary
        idx = (x <= spl.x[1]) ? 1 : length(spl.y)
        return spl.y[idx]
    elseif spl.extrapolation == linear
        idx = (x < spl.x[1]) ? 1 : length(spl.x)
        x0 = spl.x[idx]
        y0 = spl.y[idx]
        yp0 = derivative(spl, x0)
        return yp0 * (x - x0) + y0
    elseif spl.extrapolation == powerlaw
        idx = (x < spl.x[1]) ? 1 : length(spl.x)
        x0 = spl.x[idx]
        y0 = spl.y[idx]
        yp0 = derivative(spl, x0)
        p = x0 * yp0 / y0
        #@show "powerlaw",x,x0,yp0,p
        return y0 * (x / x0)^p
    else
        error("Unkown ExtrapolationType '$(spl.extrapolation)'")
    end
end


#################### 0th-order spline ##################
function Spline1D(x, y, spl::Spline1D{0})
	spl.x .= x
	spl.y .= y
        spl.ilast[] = 0  # must initialize with an invalid index so that last_abcdh isn't mistakenly taken as valid
        return spl
end


function evaluate(spline::Spline1D{0}, x::Number)
	M = length(spline.y)
	i = findlargestsmaller(spline.x, x, spline)
        (i < 1 || i > M) && return evaluate_past_boundary(spline, x)
	@inbounds return spline.y[i]
end


function derivative(spline::Spline1D{0,T}, x::Number) where {T}
        xx = spline.x
        yy = spline.y
	N = length(xx)
	i = findlargestsmaller(xx, x, spline)
	if i <= 1 || i >= N
            return T(0)
	end
        if x == xx[i]
            if yy[i-1] == yy[i]
                return T(0)
            end
            return (yy[i-1] < yy[i]) ? T(Inf) : T(-Inf)
        elseif x == xx[i+1]
            if yy[i] == yy[i+1]
                return T(0)
            end
            return (yy[i] < yy[i+1]) ? T(Inf) : T(-Inf)
        end
        return T(0)
end


#################### 1st-order spline ##################
function Spline1D(x, y, spl::Spline1D{1})
    spl.x .= x
    spl.y .= y
    spl.ilast[] = 0
    return spl
end


function evaluate(spline::Spline1D{1}, x::Number)
        xx = spline.x
        yy = spline.y
	M = length(yy)
	i = findlargestsmaller(xx, x, spline)
        (i < 1 || i > M) && return evaluate_past_boundary(spline, x)
        @inbounds return yy[i] + (x - xx[i]) / (xx[i+1] - xx[i]) * (yy[i+1] - yy[i])
end


function derivative(spline::Spline1D{1,T}, x::Number) where {T}
        xx = spline.x
        yy = spline.y
	N = length(xx)
	i = findlargestsmaller(xx, x, spline)
	if i < 1 || i >= N
            return T(0)
	end
        @inbounds return (yy[i+1] - yy[i]) / (xx[i+1] - xx[i])
end


#################### 3rd-order spline ##################
Spline1D(x, y, spl::Spline1D{3}) = Spline1Dk3(x, y, spl)


function Ai(x, y, i)
	6(y[i+1] - y[i]) / (x[i+1] - x[i]) - 6(y[i] - y[i-1]) / (x[i] - x[i-1])
end


function yderiv_k2(x, y, i, j, k)
	((y[k]-y[i])*(x[j]-x[i])^2 - (y[j]-y[i])*(x[k]-x[i])^2) / (
		(x[k]-x[i]) * (x[j]-x[k]) * (x[j] - x[i]))
end


function yderiv_k3(x, y, i, j, k, l)
	# Note: I don't trust this mathematics, yet
	x1 = x[i]
	x2 = x[j]
	x3 = x[k]
	x4 = x[l]
	y1 = y[i]
	y2 = y[j]
	y3 = y[k]
	y4 = y[l]
	hinv = 1 / (x4 - x1)
	t2 = (x2 - x1) * hinv
	t3 = (x3 - x1) * hinv
	yp1 = hinv * t3 / (t2*t3*(1-t2^2) - t2^2*(1-t2*(1+t3))) * (
		y2 - y1 - (y4-y1)*t2^3
		- t2^2*(1-t2) / (t3^2*(1-t3)) * (y3 - y1 - (y4-y1)*t3^3))
	return yp1
end


# for k = 3
function Spline1Dk3(x, y, spline::Spline1D{3})
	N = length(x)

	resize!(spline.x, N)
	resize!(spline.y, N)
	resize!(spline.ypp, N)
        resize!(spline.last_abcdh, 5)
	if length(spline.betainv) < N-1
		resize!(spline.betainv, N-1)
	end
        ypp = spline.ypp
        betainv = spline.betainv

	u = ypp  # alias for clarity

	##ypp[1] = 0.0
	#ypp[1] = Ai(x, y, 2) / (3 * (x[3] - x[1]))
	#A2adj = - (x[2] - x[1]) * ypp[1]
	#a22adj = 0.0

	##ypp[N] = 0.0
	#ypp[N] = Ai(x, y, N-1) / (3  * (x[N] - x[N-2]))
	#ANadj = - (x[N] - x[N-1]) * ypp[N]
	#aNNadj = 0.0


	yp1 = yderiv_k2(x, y, 1, 2, 3)
	#yp1 = yderiv_k3(x, y, 1, 2, 3, 4)
	A2adj = 3 * yp1 - 3 * (y[2] - y[1]) / (x[2] - x[1])
	a22adj = - (x[2] - x[1]) / 2

	ypN = yderiv_k2(x, y, N, N-1, N-2)
	#ypN = yderiv_k3(x, y, N, N-1, N-2, N-3)
	ANadj = 3 * (y[N] - y[N-1]) / (x[N] - x[N-1]) - 3 * ypN
	aNNadj = - (x[N] - x[N-1]) / 2


	# forward substitution
	betainv[2] = 1 / (2 * (x[3] - x[1]) + a22adj)
	u[2] = Ai(x, y, 2) + A2adj
	for i=3:N-2
		xixim1betainvim1 = (x[i] - x[i-1]) * betainv[i-1]
		betainv[i] = 1 / (2(x[i+1] - x[i-1]) - (x[i] - x[i-1]) * xixim1betainvim1)
		u[i] = Ai(x, y, i) - xixim1betainvim1 * u[i-1]
	end
	i = N-1
	xixim1betainvim1 = (x[i] - x[i-1]) * betainv[i-1]
	betainv[i] = 1 / (2(x[i+1] - x[i-1]) + aNNadj - (x[i] - x[i-1]) * xixim1betainvim1)
	u[i] = Ai(x, y, i) + ANadj - xixim1betainvim1 * u[i-1]

	# backward substitution
	ypp[N-1] = betainv[N-1] * u[N-1]
	for i=N-2:-1:2
		ypp[i] = betainv[i] * (u[i] - (x[i+1] - x[i]) * ypp[i+1])
	end

	ypp[1] = (a22adj * ypp[2] - A2adj) / (x[2] - x[1])
	ypp[N] = (aNNadj * ypp[N-1] - ANadj) / (x[N] - x[N-1])

        spline.x .= x
	spline.y .= y
	spline.ypp .= ypp
        spline.ilast[] = 0
	spline.betainv .= betainv
	return spline
end



function findlargestsmaller_bisect(a, x::Number, nmin, nmax)::Int64
	nmid = div(nmin + nmax, 2)
	while nmid != nmin
		@inbounds if x >= a[nmid]
			nmin = nmid
		else
			nmax = nmid
		end
		nmid = div(nmin + nmax, 2)
	end
	return nmin
end

#global ihist = zeros(Int, 2001)

function findlargestsmaller(a, x, ilast::Int)
	N = length(a)

	@inbounds if x < a[1]
		return 0
	elseif x > a[end]
		return N + 1
	end

        nmin = (1 <= ilast <= N) ? ilast : 1
	@inbounds while x < a[nmin]
		nmin = div(nmin, 2)
	end

        # handle most common cases
        @inbounds if nmin < N && x < a[nmin + 1]  # i == ilast == nmin
            #global ihist[0 + 1000 + 100] += 1
            return nmin
        elseif nmin < N - 1 && x < a[nmin + 2]  # i == ilast + 1
            #global ihist[1 + 1000 + 100] += 1
            return nmin + 1
        end

        nmax = nmin  # nmin <= N
        @inbounds while x > a[nmax]  # x > a[N] is already excluded
		nmax = div(nmax + N, 2) + 1
	end

        i = findlargestsmaller_bisect(a, x, nmin, nmax)
        #global ihist[i - ilast + 1000] += 1
        return i
end


function findlargestsmaller(a, x, spline)
    i = findlargestsmaller(a, x, spline.ilast[])
    #spline.ilast[] = i
    return i
end


function get_abcdh(spline, i, x)
	@inbounds x0 = spline.x[i]
	@inbounds x1 = spline.x[i+1]
	h = x1 - x0
	t = (x - x0) / h
	@inbounds ypp0 = spline.ypp[i]
	@inbounds ypp1 = spline.ypp[i+1]
	h2 = h^2
	b = h2 * ypp0 / 2
	a = h2 * (ypp1 - ypp0) / 6

	@inbounds d = spline.y[i]
	@inbounds y1 = spline.y[i+1]
	@inbounds c = y1 - a - b - d
        spline.ilast[] = i
        spline.last_abcdh .= (a,b,c,d,h)
	return a, b, c, d, t, h
end


function eval_last(spline::Spline1D{3,T}, i, x)::T where {T}
    @inbounds a, b, c, d, h = spline.last_abcdh
    @inbounds x0 = spline.x[i]
    t = (x - x0) / h
    return @evalpoly(t, d, c, b, a)
end

function eval_i(spline::Spline1D{3,T}, i, x)::T where {T}
	a, b, c, d, t, h = get_abcdh(spline, i, T(x))
	return @evalpoly(t, d, c, b, a)
end


function evaluate(spline::Spline1D{3,T}, x::Number)::T where {T}
        ilast = spline.ilast[]
        i = findlargestsmaller(spline.x, T(x), spline)
        (i < 1 || i >= length(spline.x)) && return evaluate_past_boundary(spline, T(x))
        (i == ilast) && return eval_last(spline, i, x)
        return eval_i(spline, i, x)
end


function derivative(spline::Spline1D{3,T}, x::Number) where {T}
	N = length(spline.x)
	i = findlargestsmaller(spline.x, x, spline)
	if i < 1
            return T(0)
	elseif i > N
            return T(0)
	end
	a, b, c, d, t, h = get_abcdh(spline, i, x)
	@evalpoly(t, c, 2b, 3a) / h
end

# integrate():
#   Integrate the spline 'spline' over the interval [xlo, xhi].
function integrate(spline::Spline1D{3,T}, xlo::Number, xhi::Number) where {T}
	a = findlargestsmaller(spline.x, xlo, spline)
	b = findlargestsmaller(spline.x, xhi, spline)
	N = length(spline.x)

        I = T(0)
        if spline.extrapolation == zero && a > N-1
            return T(0)
        elseif spline.extrapolation == boundary
            if a < 1
                    I += (spline.x[1] - xlo) * spline.y[1]
                    xlo = spline.x[1]
            elseif a > N-1
                    return (xhi - xlo) * spline.y[N]
            end
            if b > N-1
                    I += (xhi - spline.x[N]) * spline.y[N]
                    xhi = spline.x[N]
            end
        end
        (a < 1) && (a = 1)
        (b > N-1) && (b = N-1)

	x0 = spline.x[a]
	y0 = spline.y[a]
	ypp0 = spline.ypp[a]
	x1 = spline.x[a+1]
	y1 = spline.y[a+1]
	ypp1 = spline.ypp[a+1]
	h = x1 - x0
	h2 = h^2
	bi = h2 * ypp0 / 2
	ai = h2 * (ypp1 - ypp0) / 6
	ci = y1 - y0 - ai - bi
	t = (xlo - x0) / h
	I += - h * t * @evalpoly(t, y0, ci/2, bi/3, ai/4)
        #dI = h * (ai/4 + bi/3 + ci/2 + y0)
        I1 = T(0)
        I2 = T(0)
        dI1 = h * (y0 + y1)
        dI2 = h^3 * (ypp0 + ypp1)
	@inbounds for i=a+2:b+1
                #I += dI
                #I += dI1/2 - dI2/24
                I1 += dI1
                I2 += dI2

		# prepare the next:
		x0 = x1
		y0 = y1
		ypp0 = ypp1
		x1 = spline.x[i]
		y1 = spline.y[i]
		ypp1 = spline.ypp[i]

		h = x1 - x0
		#h2 = h^2
		#bi = h2 * ypp0 / 2
		#ai = h2 * (ypp1 - ypp0) / 6
		#ci = y1 - y0 - ai - bi
                #dI = h * (ai/4 + bi/3 + ci/2 + y0)

                #dI = h * ((y0 + y1) * (1/2) - h2 * (ypp0 + ypp1) * (1/24))
                dI1 = h * (y0 + y1)
                dI2 = h^3 * (ypp0 + ypp1)
	end
        I += I1/2 - I2/24
        h = x1 - x0
        h2 = h^2
        bi = h2 * ypp0 / 2
        ai = h2 * (ypp1 - ypp0) / 6
        ci = y1 - y0 - ai - bi
	t = (xhi - x0) / h
	I += h * t * @evalpoly(t, y0, ci/2, bi/3, ai/4)

	return I
end



(s::Spline1D)(x) = evaluate(s, x)


######################## Spline1Dloglog
# This is a specialization of Spline1Dtrans. Reason: be faster.
struct Spline1Dloglog{Spline1DType}
	spline::Spline1DType
end


size(s::Spline1Dloglog) = (1,)
getindex(s::Spline1Dloglog, i) = s


# empty initialization
Spline1Dloglog(; k=3, T=Float64, extrapolation=boundary) = Spline1Dloglog(Spline1D(k=k, T=T, extrapolation=extrapolation))


# Spline1D(): Same as the other constructors, except that the inputs are
# transformed by the 'x_in_trans()' and 'y_in_trans()' functions.
# 'y_out_trans()' will typically be the inverse of 'y_in_trans()', as it is
# used to undo the transformation.
function Spline1Dloglog(x, y; k=3, extrapolation=boundary)
	xt = log.(x)
	yt = log.(y)
	spl = Spline1D(xt, yt; k=k, extrapolation=extrapolation)
        Spline1Dloglog(spl)
end

function Spline1D(x, y, spl::Spline1Dloglog)
    resize!(spl.spline.x, Int64(length(x)))
    resize!(spl.spline.y, Int64(length(y)))
    @. spl.spline.x = log(x)
    @. spl.spline.y = log(y)
    Spline1D(spl.spline.x, spl.spline.y, spl.spline)
    return spl
end

evaluate(s::Spline1Dloglog, x) = begin
    lnx = log(x)                  # ~20% of time
    #@show s.spline.extrapolation
    lnv = evaluate(s.spline, lnx) # ~30% of time
    v = exp(lnv)                  # ~50% of time
    return v
    #exp(evaluate(s.spline, log(x)))
end

(s::Spline1Dloglog)(x) = evaluate(s, x)

function derivative(s::Spline1Dloglog, x)
	xt = log(x)
	sval = evaluate(s.spline, xt)
	dyout = exp(sval)
	ds = derivative(s.spline, xt)
	dxt = inv(x)
	return dyout * ds * dxt
end


######################## Spline1Dtrans
struct Spline1Dtrans{Spline1DType}
	spline::Spline1DType
	x_in_trans::Function
	y_in_trans::Function
	y_out_trans::Function
	x_in_trans_deriv::Function
	y_out_trans_deriv::Function
end


size(s::Spline1Dtrans) = (1,)
getindex(s::Spline1Dtrans, i) = s


# Spline1D(): Same as the other constructors, except that the inputs are
# transformed by the 'x_in_trans()' and 'y_in_trans()' functions.
# 'y_out_trans()' will typically be the inverse of 'y_in_trans()', as it is
# used to undo the transformation.
function Spline1Dtrans(x, y, x_in_trans::Function, y_in_trans::Function, y_out_trans::Function,
                       x_in_trans_deriv::Function, y_out_trans_deriv::Function;
                       k=3, extrapolation=boundary)
	xt = x_in_trans.(x)
	yt = y_in_trans.(y)
	spl = Spline1D(xt, yt; k=k, extrapolation=extrapolation)
	Spline1Dtrans(spl, x_in_trans, y_in_trans, y_out_trans, x_in_trans_deriv, y_out_trans_deriv)
end

function find_inverse(fn)
	if fn == identity
		return identity
	elseif fn == log
		return exp
	elseif fn == exp
		return log
	else
		error("Inverse of function '$fn' not implemented here.")
		return exp
	end
end

function find_derivative(fn)
	if fn == identity
		return one
	elseif fn == log
		return inv
	elseif fn == exp
		return exp
	else
		error("Derivative of function '$fn' not implemented here.")
		return exp
	end
end

function Spline1Dtrans(x, y, x_in_trans::Function, y_in_trans::Function;
                       k=3, extrapolation=boundary)
	x_in_trans_deriv = find_derivative(x_in_trans)
	y_out_trans = find_inverse(y_in_trans)
	y_out_trans_deriv = find_derivative(y_out_trans)
	Spline1Dtrans(x, y, x_in_trans, y_in_trans, y_out_trans,
		      x_in_trans_deriv, y_out_trans_deriv; k=k, extrapolation=extrapolation)
end

function Spline1D(x, y, spl::Spline1Dtrans)
    resize!(spl.spline.x, length(x))
    resize!(spl.spline.y, length(y))
    @. spl.spline.x = spl.x_in_trans(x)
    @. spl.spline.y = spl.y_in_trans(y)
    Spline1D(spl.spline.x, spl.spline.y, spl.spline)
    return spl
end

evaluate(s::Spline1Dtrans, x) = s.y_out_trans(evaluate(s.spline, s.x_in_trans(x)))

(s::Spline1Dtrans)(x) = evaluate(s, x)

function derivative(s::Spline1Dtrans, x)
	xt = s.x_in_trans(x)
	sval = evaluate(s.spline, xt)
	dyout = s.y_out_trans(sval)
	ds = derivative(s.spline, xt)
	dxt = s.x_in_trans_deriv(x)
	return dyout * ds * dxt
end


end

# vim: set sw=4 et sts=4 :
